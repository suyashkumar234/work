"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision import transforms
import numpy as np

from models.grid_proto_fewshot import FewShotSeg

from dataloaders.dev_customized_medv1 import med_fewshot_val
from dataloaders.ManualAnnoDatasetv1 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv1 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric

from config_ssl_upload import ex

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid

from models.agun_model import AGUNet

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(pretrained_path = None, cfg=_config['model'])
    model.load_state_dict(torch.load(_config['reload_model_path'])['model'],strict = False)

    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS': # for CT we need to know statistics of 
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transform_param_limits=myaug.augs[_config['which_aug']],
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
            dataset_config = _config['DATASET_CONFIG']
            )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)


    te_dataset, te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split = _config['eval_fold'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func
    )

    ### dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set validation nodes ######')
    _scan_ids = te_parent.scan_ids
    # import sys
    mar_val_metric_nodes = {}
    # sys.exit()
    # mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'] + 1)

    _log.info('###### Starting validation ######')
    model.eval()

    with torch.no_grad():
        save_pred_buffer_ = {} # indexed by class
        for sup_idx in range(len(_scan_ids)):
            _log.info(f'###### Set validation nodes for support id {_scan_ids[sup_idx]} ######')
            mar_val_metric_nodes[_scan_ids[sup_idx]] = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'] )
            mar_val_metric_nodes[_scan_ids[sup_idx]].reset()

            save_pred_buffer = {}

            for curr_lb in test_labels:
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(curr_class = curr_lb, 
                                                        class_idx = [curr_lb], 
                                                        scan_idx = [sup_idx], 
                                                        npart=_config['task']['npart'])

                # way(1 for now) x part x shot x 3 x H x W] #
                support_images = [[shot.cuda() for shot in way]
                                    for way in support_batched['support_images']] # way x part x [shot x C x H x W]
                # print(len(support_images))
                # for way in support_images:
                #     print(len(way))
                #     for shot in way:
                #         print(shot.shape)
                suffix = 'mask'
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]

                # print('n_shots: ',len(support_images[0]))
                curr_scan_count = -1 # counting for current scan
                _lb_buffer = {} # indexed by scan

                last_qpart = 0 # used as indicator for adding result to buffer

                for sample_batched in testloader:

                    _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                    if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                        continue
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"][0]
                        outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                        _pred = np.zeros( outsize )
                        _pred.fill(np.nan)
                        _labels = np.zeros(outsize)
                        _qimgs = np.zeros((3,256,256,outsize[-1])) #outsize)
                        supimgs = np.zeros((3,256,256,outsize[-1])) #outsize)
                        supfgs = np.zeros((256,256,outsize[-1])) #outsize)
                        _means = np.zeros((1,1,1,outsize[-1]))
                        _stds = np.zeros((1,1,1,outsize[-1]))

                    q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                    query_images = [sample_batched['image'].cuda()]
                    query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                    # [way, [part, [shot x C x H x W]]] ->
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                    # plt.imshow(query_images[0].cpu().numpy()[0].transpose((1,2,0)))
                    # plt.show()
                    # print(len(sup_img_part),len(sup_img_part[0]),len(sup_img_part[0][0]),sup_img_part[0][0].shape,sup_img_part[0][0][0].shape)

                    query_pred, _, _, _ = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                    # print(query_pred.cpu().numpy().shape,query_labels.cpu().numpy().shape)
                    # print(query_pred.min(), query_pred.max()) #/3)**0.5)
                    query_pred = query_pred.argmax(dim = 1).float().cpu().numpy() 
                    # query_pred = query_pred / query_pred.max() #
                    # query_pred = query_pred.astype(np.float32)
                    # print(query_pred.shape)
                    # print(np.sum(query_pred))
                    # print(query_labels[0].shape)
                    _pred[..., ii] = query_pred[0].copy()
                    _labels[..., ii] = query_labels[0].detach().cpu().clone().numpy()
                    # print(query_images[0].shape)
                    _qimgs[..., ii] = query_images[0].detach().cpu().clone().numpy()
                    # print(_qimgs.shape)
                    _means[..., ii] = sample_batched["mean"][0]
                    _stds[..., ii] = sample_batched["std"][0]

                    # print(sup_fgm_part[0][0][0].cpu().numpy().shape)

                    supimgs[..., ii] = sup_img_part[0][0][0].cpu().numpy()
                    supfgs[..., ii] = sup_fgm_part[0][0][0].cpu().numpy()

                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        mar_val_metric_nodes[_scan_ids[sup_idx]].record(_pred[..., ii], _labels[..., ii], labels=[curr_lb], n_scan=curr_scan_count) 
                    else:
                        pass

                    ii += 1
                    # now check data format
                    if sample_batched["is_end"]:
                        if _config['dataset'] != 'C0':
                            # print(_qimgs.shape)
                            _lb_buffer[_scan_id] = [_pred.transpose(2,0,1),
                                                    _labels.transpose(2,0,1),
                                                    _qimgs.transpose(3,1,2,0), 
                                                    _means.transpose(3,1,2,0), 
                                                    _stds.transpose(3,1,2,0), 
                                                    supimgs.transpose(3,1,2,0),
                                                    supfgs.transpose(2,0,1)] # H, W, Z -> to Z H W
                        else:
                            lb_buffer[_scan_id] = [_pred, _labels, _qimgs, _means, _stds, sup_img_part, sup_fgm_part]

                save_pred_buffer[str(curr_lb)] = _lb_buffer

            save_pred_buffer_[_scan_ids[sup_idx]] = save_pred_buffer

        ### save results

        for _sup_id, saved_pred in save_pred_buffer_.items():
            for curr_lb, _preds in saved_pred.items():
                c = 0
                for _scan_id, item in _preds.items():
                    gif_frames = []
                    _pred, _label, _qimg, _mean, _std, _supimg, _supfg = item
                    print(_pred.shape, _qimg.shape, _label.shape, _mean.shape, _std.shape, _supimg.shape, _supfg.shape)
                    # print(_pred)
                    _pred = 255*(_pred - _pred.min(axis = (1,2), keepdims = True))/ (1.e-6 + _pred.max(axis = (1,2), keepdims = True) - _pred.min(axis = (1,2), keepdims = True))
                    _pred = _pred.astype(np.uint8)
                    # print(_label)
                    _label = 255*(_label).astype(np.uint8)
                    _supfg = (255*_supfg).astype(np.uint8)
                    
                    # _pred *= float(curr_lb)
                    # itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
                    # fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                    for i in range(_pred.shape[0]):
                        fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}_{i}.png')
                        # sitk.WriteImage(itk_pred, fid) #, True)
                        qim = _qimg[i]*_std[i] + _mean[i]
                        qim = 255*(qim - qim.min())/(qim.max() - qim.min() + 1e-6)
                        qim = qim.astype(np.uint8)
                        qim = Image.fromarray(qim, 'RGB').convert("RGBA") #_qimg[i])

                        im1 = np.stack([_pred[i]*0.8,np.zeros(_pred[i].shape),np.zeros(_pred[i].shape)], axis = 2)
                        im1 = im1.astype(np.uint8)
                        im1 = Image.fromarray(im1,'RGB').convert("RGBA")

                        im2 = np.stack([np.zeros(_label[i].shape),_label[i]*0.8,np.zeros(_label[i].shape)], axis = 2)
                        im2 = im2.astype(np.uint8)
                        im2 = Image.fromarray(im2,'RGB').convert("RGBA")
                        
                        qim = Image.blend(qim, im1, alpha = 0.5)
                        qim = Image.blend(qim, im2, alpha = 0.3)
                        # imageio.imwrite(fid, np.array(qim)) #,format='png')

                        supimg = _supimg[i]*_std[i] + _mean[i]
                        supimg = (supimg - supimg.min())/(supimg.max() - supimg.min())*255
                        supimg = supimg.astype(np.uint8)
                        # print(supimg.shape)
                        supimg = Image.fromarray(supimg, 'RGB').convert("RGBA")

                        supfg = _supfg[i]
                        supfg = np.stack([np.zeros(supfg.shape),supfg,np.zeros(supfg.shape)], axis = 2)
                        supfg = supfg.astype(np.uint8)
                        # print(supfg.shape)
                        supfg = Image.fromarray(supfg, 'RGB').convert("RGBA")

                        # print(supimg.shape, supfg.shape)

                        supimg = Image.blend(supimg, supfg, alpha = 0.4)

                        new_image = Image.new('RGB',(2*256, 256), (250,250,250))
                        new_image.paste(supimg,(0,0))
                        new_image.paste(qim,(256,0))

                        gif_frames.append(new_image)
                        new_image.save(fid)
                        
                    fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}.gif')
                    gif_frames[0].save(fid, format="GIF", append_images=gif_frames, save_all=True, duration=500, loop=0)
                    _log.info(f'###### {fid} has been saved #####')

        del save_pred_buffer

    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

    # compute dice scores by scan
    for i in range(len(_scan_ids)):
        print(f"========== Metrics for Support Scan {_scan_ids[i]} ============")
        m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_nodes[_scan_ids[i]].get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)

        m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_nodes[_scan_ids[i]].get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)

        mar_val_metric_nodes[_scan_ids[i]].reset() # reset this calculation node

        # write validation result to log file
        _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
        _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
        _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

        _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
        _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
        _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

        _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
        _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
        _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

        _log.info(f'mar_val batches classDice: {m_classDice}')
        _log.info(f'mar_val batches meanDice: {m_meanDice}')

        _log.info(f'mar_val batches classPrec: {m_classPrec}')
        _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

        _log.info(f'mar_val batches classRec: {m_classRec}')
        _log.info(f'mar_val batches meanRec: {m_meanRec}')

        print(f"============ Completed Support Scan {_scan_ids[i]} ============")

    _log.info(f'End of validation')
    return 1


