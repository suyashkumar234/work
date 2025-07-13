"""
Training the model - M1 Mac Compatible Version
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO # contains information about the dataset
import dataloaders.augutils as myaug # contains data augmentation functions

from util.utils import set_seed, t2n, to01, compose_wt_simple, get_tversky_loss
from util.metric import Metric
from util.device_utils import setup_device_and_threads, to_device  # New import

from config_ssl_upload import ex
import tqdm
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model" # sets an environment variable that tells PyTorch where to store and look for pre-trained models

@ex.automain # decorator
def main(_run, _config, _log): # code according to sacred xperimental framework setup _run: Sacred's run object that tracks the experiment
# _config: Contains all configuration parameters
# _log: Sacred's logger object
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)# exist_ok=True: Won't crash if directories already exist
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed']) # setting up random seed 
    
    # Modified device setup for M1 Mac compatibility
    device = setup_device_and_threads(_config) 
    
    # Only enable cudnn if using CUDA
    print(device.type)
    if device.type == 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True

    _log.info('###### Create model ######') # printing on terminal

    model = FewShotSeg(pretrained_path=None, cfg=_config['model']) # creating an instance of the model

    # Move model to appropriate device
    model = model.to(device)
    model.train()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    ###================== Transforms for data augmentation =============================== ###
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]}) # applying data augmentation to the training data
    ### ================================================================================== ###
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]] # labels excluded in training- all labels minus the labels in the current fold
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')
     # Create training dataset
    
    
    # dataset, tr_parent = med_fewshot(
    #     dataset_name=baseset_name,
    #     base_dir=_config['path'][data_name]['data_dir'],
    #     idx_split=_config['eval_fold'],
    #     mode='train',
    #     scan_per_load=_config['scan_per_load'],
    #     transforms=tr_transforms,
    #     act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'],
    #     n_ways=1,
    #     n_shots=1,
    #     nsup=_config['task']['n_shots'],
    #     fix_parent_len=_config["max_iters_per_load"] if _config['fix_length'] else None,
    #     max_iters_per_load=_config["max_iters_per_load"],
    #     min_fg=str(_config["min_fg_data"]),
    #     n_queries=1,
    #     exclude_list=_config["exclude_cls_list"],
    #     dataset_config=_config['DATASET_CONFIG'],
    #     client_eval=True if _config.get('client_eval', False) else False
    # )
    # dataset.norm_func = tr_parent.norm_func

    # # Create training dataloader
    # trainloader = DataLoader(
    #     dataset,
    #     batch_size=_config['batch_size'],
    #     shuffle=True,
    #     num_workers=_config['num_workers'],
    #     pin_memory=True,
    #     drop_last=True
    # )
    # when doing ssl we use the superpixel dataset,Raw Data → Dataset → DataLoader → Model → Loss → Optimizer → Backprop
    # tr_parent = SuperpixelDataset( # base dataset
    #     which_dataset = baseset_name,
    #     base_dir=_config['path'][data_name]['data_dir'],
    #     idx_split = _config['eval_fold'], # current fold- idx_split 
    #     mode='train',
    #     min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
    #     transforms =tr_transforms, # data augmentation
    #     transform_param_limits = myaug.augs[_config['which_aug']],#tr_transforms # data augmentation parameters- passing the parameter limits to the data augmentation function
    #     nsup = _config['task']['n_shots'], # number of support images
    #     scan_per_load = _config['scan_per_load'], # number of images to load at a time
    #     exclude_list = _config["exclude_cls_list"], # labels excluded in training
    #     superpix_scale = _config["superpix_scale"], # size of superpixels
    #     fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None, # if the dataset is C0, CHAOST2, or SABS, then we fix the length of the dataset to the number of images in the dataset     
    #     dataset_config = _config['DATASET_CONFIG']  # dataset configuration
    # )
    dataset, tr_parent = med_fewshot(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        mode='train',
        scan_per_load=_config['scan_per_load'],
        transforms=tr_transforms,
        act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'],
        n_ways=1,
        n_shots=1,
        nsup=_config['task']['n_shots'],
        fix_parent_len=_config["max_iters_per_load"] if _config['fix_length'] else None,
        max_iters_per_load=_config["max_iters_per_load"],
        min_fg=str(_config["min_fg_data"]),
        n_queries=1,
        exclude_list=_config["exclude_cls_list"],
        dataset_config=_config['DATASET_CONFIG'],
        client_eval=True if _config.get('client_eval', False) else False
    )
    dataset.norm_func = tr_parent.norm_func
    print("Dataset length:", len(dataset))   

# iteration: One batch of data processed (forward + backward pass).
# Load: A group of iterations, using a subset of the dataset loaded into memory at once.

    # print("About to print tr_parent", flush=True)
    # print(tr_parent, flush=True) # <dataloaders.GenericSuperDatasetv2.SuperpixelDataset object at 0x15a06ff20>
    # print("Printed tr_parent", flush=True)


    ### dataloaders: DataLoader is a class in PyTorch that wraps an iterable around the dataset to enable easy access to batches.
    trainloader = DataLoader(
        dataset, # tr_parent is the dataset object, it is a collection of data and labels, it is used to load the data and labels into the model
        batch_size=_config['batch_size'],
        shuffle=True, # Whether to randomize selection after each max_iter_per_load
        # shuffle=False, sequential 
        num_workers=_config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,  # Only pin memory for CUDA
        drop_last=True # ?
    )
    print("Number of batches per epoch:", len(trainloader))
    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim']) # setting up the optimizer, they are used to unpack (spread out) the contents of a list/tuple (*) or a dictionary (**) when calling a function or creating a new dictionary.

    elif _config['optim_type'] == 'adam':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma']) # learning rate scheduler

    # Modified weight composition for device compatibility
    my_weight = compose_wt_simple(_config["use_wce"], data_name) # weight composition for the loss function
    # compose_wt_simple: This is a function (likely defined elsewhere in your code) that creates a tensor of weights for each class in your segmentation task.
    # _config["use_wce"]: A config flag (probably a boolean) that determines whether to use Weighted Cross Entropy (WCE).
    # data_name: The name of the dataset, which may affect how the weights are set (e.g., different datasets may have different class imbalances).
    #nn.CrossEntropyLoss: The standard loss function for multi-class classification/segmentation in PyTorch.
    # ignore_index=_config['ignore_label']:
    # Tells the loss function to ignore pixels/labels with this value (e.g., 255), which are usually background or unlabeled regions.
    # weight=my_weight:
    # Passes the class weights tensor (from above) to the loss function.
    # If my_weight is not None, the loss for each class will be scaled by its corresponding weight.
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight) # loss function

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] #deciding the number of epochs 

    log_loss = {'loss': 0, 'align_loss': 0} 

    _log.info('###### Training ######')
    stime = time.time()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader): # trainloader is the dataloader defined in torch.utils.data , sample_batched is the batch of data
            # Prepare input
            i_iter += 1
            print(f"Processing iteration {i_iter}", flush=True)  # Debug print
            # Modified to use device-agnostic approach
            # print("About to print sample_batched", flush=True)
            # print(sample_batched, flush=True) 
            # print("Printed sample_batched", flush=True)
            support_images = [[shot.float().to(device) for shot in way] # 'support_images': [[tensor([...])]]  # Shape: [1 way][1 shot][C, H, W]
                              for way in sample_batched['support_images']] # support_images is a list of lists, each list contains the support images for a way
            support_fg_mask = [[shot[f'fg_mask'].float().to(device) for shot in way] # contains class_ids, support_images, support_fg_mask, support_bg_mask, query_images, query_labels, support_para, query_parameter, mean, std   
                               for way in sample_batched['support_mask']] # support_fg_mask is a list of lists, each list contains the foreground mask for a way
            support_bg_mask = [[shot[f'bg_mask'].float().to(device) for shot in way]
                               for way in sample_batched['support_mask']] # support_bg_mask is a list of lists, each list contains the background mask for a way
            
            query_images = [query_image.float().to(device)
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().to(device) for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            
            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            ########################################################################
            query_pred, align_loss, debug_vis, assign_mats = model(support_images,
                                                                    support_fg_mask, 
                                                                    support_bg_mask, 
                                                                    query_images, 
                                                                    isval = False, val_wsize = None) # passing the data to the model
            ########################################################################

            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(query_pred.argmax(dim = 1, keepdim = True), query_labels[None, ...], 0.3, 0.7 ,1.0)
            loss = query_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss # query_loss is the loss for the query images
            log_loss['align_loss'] += align_loss # align_loss is the loss for the alignment of the support images and the query images

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                fig, ax = plt.subplots(1,2)
                si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min() + 1e-6)
                ax[0].imshow(si)
                sm = support_fg_mask[0][0][0].cpu().numpy()
                ax[0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[1].imshow(qi)
                qm = query_labels[0].cpu().numpy()
                ax[1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0]
                ax[1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, time: {(nt-stime)/60} mins')
                print(i_iter)

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model':model.state_dict(),'opt':optimizer.state_dict(),'sch':scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up
