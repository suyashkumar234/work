#self supervised training
"""
Training the model - M1 Mac Compatible Version
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
# Fix for headless server environments (OpenCV/matplotlib)
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# os.environ['MPLBACKEND'] = 'Agg'
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np


from models.grid_proto_fewshot import FewShotSeg
#from models.contrastive import SupervisedPixelWiseContrastiveLoss
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.dataset_utils import DATASET_INFO # contains information about the dataset
import dataloaders.augutils as myaug # contains data augmentation functions

from util.utils import set_seed, t2n, to01, compose_wt_simple#, get_tversky_loss
from util.metric import Metric
#from util.device_utils import setup_device_and_threads, to_device  # New import
#from util.organ_visualization import create_organ_visualizer  # Updated import for organ visualization
#from util.visualization import visualize_tsne_with_labels
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
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"), exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources', ignore_errors=True)
    set_seed(_config['seed']) # setting up random seed 
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    
    # Modified device setup for M1 Mac compatibility
    #device = setup_device_and_threads(_config) 
    

    _log.info('###### Create model ######') # printing on terminal

    model = FewShotSeg(pretrained_path=None, cfg=_config['model']) # creating an instance of the model

    # Move model to appropriate device
    model = model.cuda()
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

    # For superpixel learning, we don't need to filter act_labels
    # since we work with binary superpixel masks
    excluded = _config["exclude_cls_list"]
    print(f"Excluded classes for superpixel learning: {excluded}")

    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms =tr_transforms, #
        transform_param_limits = myaug.augs[_config['which_aug']],#tr_transforms, #
        nsup = _config['task']['n_shots'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
        dataset_config = _config['DATASET_CONFIG']
    )
    
    # Set normalization function
    dataset_norm_func = tr_parent.norm_func

    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        #pin_memory=True if device.type == 'cuda' else False,  # Only pin memory for CUDA
        drop_last=True
    )
    
    # dataset, tr_parent = med_fewshot(
    #     dataset_name=baseset_name,
    #     base_dir=_config['path'][data_name]['data_dir'],
    #     idx_split=_config['eval_fold'],
    #     mode='train',
    #     scan_per_load=_config['scan_per_load'],
    #     transforms=tr_transforms,
    #     act_labels=act_labels,
    #     n_ways=1,
    #     n_shots=1,
    #     nsup=_config['task']['n_shots'],
    #     fix_parent_len=_config["max_iters_per_load"] if _config['fix_length'] else None,
    #     max_iters_per_load=_config["max_iters_per_load"],
    #     min_fg=str(_config["min_fg_data"]),
    #     n_queries=1,
    #     exclude_list=excluded,
    #     dataset_config=_config['DATASET_CONFIG'],
    #     client_eval=True if _config.get('client_eval', False) else False
    # )
    # #print(act_labels, excluded)
    # dataset.norm_func = tr_parent.norm_func
    # print("Dataset length:", len(dataset))   

    # ### dataloaders: DataLoader is a class in PyTorch that wraps an iterable around the dataset to enable easy access to batches.
    # trainloader = DataLoader(
    #     dataset, # tr_parent is the dataset object, it is a collection of data and labels, it is used to load the data and labels into the model
    #     batch_size=_config['batch_size'],
    #     shuffle=True, # Whether to randomize selection after each max_iter_per_load
    #     # shuffle=False, sequential 
    #     num_workers=_config['num_workers'],
    #     pin_memory=True if device.type == 'cuda' else False,  # Only pin memory for CUDA
    #     drop_last=True # ?
    # )

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

    log_loss = {'loss': 0, 'align_loss': 0, 'contrastive_loss': 0} 

    # # --- Buffers for accumulating features for superpixel learning ---
    # buffer_teacher_features = []
    # buffer_student_features = []
    # buffer_teacher_masks = []
    # buffer_student_masks = []
    # buffer_organ_classes = []
    # For superpixel learning, we work with class_id = 1 (all superpixels)
    # ---------------------------------------------------------------

    _log.info('###### Training ######')
    stime = time.time()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader): # trainloader is the dataloader defined in torch.utils.data , sample_batched is the batch of data
            # Prepare input
            i_iter += 1
            
            # Handle SuperpixelDataset format
            support_images = [[shot.float().cuda() for shot in way] 
                              for way in sample_batched['support_images']] 
            support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way] 
                               for way in sample_batched['support_mask']] 
            support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']] 

            query_images = [query_image.float().cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            # update the student encoder with the teacher encoder using momentum
            model.update_student_encoder(model.student_encoder)
            
            # Handle mean/std from SuperpixelDataset
            mean = sample_batched["mean"] if isinstance(sample_batched["mean"], list) else [sample_batched["mean"]]
            std = sample_batched["std"] if isinstance(sample_batched["std"], list) else [sample_batched["std"]]
            ########################################################################
            query_pred, align_loss, debug_vis, assign_mats, contrastive_loss = model(support_images,
                                                                           support_fg_mask,
                                                                           support_bg_mask,
                                                                           query_images,
                                                                           isval = False, val_wsize = None)
            ########################################################################

            query_loss = criterion(query_pred, query_labels) #+ get_tversky_loss(query_pred.argmax(dim = 1, keepdim = True), query_labels[None, ...], 0.3, 0.7 ,1.0)
            query_weight=1.0
            align_weight=1.0
            contrastive_weight=1.0
            # print(f'Query_weight-{query_weight}')
            # print(f'Align_weight-{align_weight}')
            # print(f'Contrastive_weight-{contrastive_weight}')
            loss = query_loss*query_weight + align_loss* align_weight + contrastive_loss*contrastive_weight
            loss.backward()
            optimizer.step()

            # # After optimizer step
            # teacher_params_after = [p.clone().detach() for p in model.teacher_encoder.parameters()]
            # student_params_after = [p.clone().detach() for p in model.student_encoder.parameters()]

            # # Compare
            # teacher_changed = any([(before != after).any() for before, after in zip(teacher_params_before, teacher_params_after)])
            # student_changed = any([(before != after).any() for before, after in zip(student_params_before, student_params_after)])

            # print(f"Teacher params changed: {teacher_changed}")
            # print(f"Student params changed: {student_changed}")

            scheduler.step()

            # Log loss
            # query_loss = query_loss.detach().data.cpu().numpy()
            # align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            # contrastive_loss = contrastive_loss.detach().data.cpu().numpy() if contrastive_loss != 0 else 0
            query_loss_val = query_loss.detach().data.cpu().numpy()
            align_loss_val = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            contrastive_loss_val = contrastive_loss.detach().data.cpu().numpy() if contrastive_loss != 0 else 0

            # Log weighted values
            _run.log_scalar('loss', query_loss_val * query_weight)
            _run.log_scalar('align_loss', align_loss_val * align_weight)
            _run.log_scalar('contrastive_loss', contrastive_loss_val * contrastive_weight)

            # Accumulate weighted losses
            log_loss['loss'] += query_loss_val * query_weight
            log_loss['align_loss'] += align_loss_val * align_weight
            log_loss['contrastive_loss'] += contrastive_loss_val * contrastive_weight

            # --- Accumulate features for superpixel learning ---
            # For superpixel learning, we use class_id = 1 (superpixel class)
            
            # For superpixel learning, accumulate all features
            # if organ_classes_flat.numel() > 0:
            #     buffer_teacher_features.append(supp_fts_teacher_flat)
            #     buffer_student_features.append(supp_fts_student_flat)
            #     buffer_teacher_masks.append((res_fg_msk_teacher_flat > 0.5).float())
            #     buffer_student_masks.append((res_fg_msk_student_flat > 0.5).float())
            #     buffer_organ_classes.append(organ_classes_flat)
            # ---------------------------------------------------------------

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0
                log_loss['contrastive_loss'] = 0

                fig, ax = plt.subplots(1,2)
                # Handle superpixel visualization - fix tensor shapes
                try:
                    # Support image: [0][0] gives us the first support image
                    si_tensor = support_images[0][0].cpu()  # Should be [C, H, W] or [B, C, H, W]
                    
                    # Handle potential batch dimension
                    if si_tensor.dim() == 4:  # [B, C, H, W]
                        si_tensor = si_tensor[0]  # Take first batch -> [C, H, W]
                    
                    if si_tensor.dim() == 3 and si_tensor.shape[0] == 3:  # [C, H, W] with 3 channels
                        si = (si_tensor * std[0] + mean[0]).numpy().transpose((1,2,0))  # [H, W, C]
                    elif si_tensor.dim() == 3:  # [C, H, W] but not 3 channels
                        si = si_tensor.numpy()
                        if si.shape[0] == 1:  # Single channel [1, H, W]
                            si = si.squeeze(0)  # [H, W]
                        else:  # Multiple channels, transpose
                            si = si.transpose((1,2,0))  # [H, W, C]
                    else:  # [H, W]
                        si = si_tensor.numpy()
                    
                    si = (si - si.min())/(si.max() - si.min() + 1e-6)
                    if len(si.shape) == 2:  # Grayscale
                        ax[0].imshow(si, cmap='gray')
                    else:
                        ax[0].imshow(si)
                        
                    # Support mask
                    sm = support_fg_mask[0][0].cpu().numpy()
                    if len(sm.shape) == 4:  # [B, C, H, W]
                        sm = sm[0, 0]  # Take first batch and channel
                    elif len(sm.shape) == 3:  # [C, H, W] or [B, H, W]
                        sm = sm.squeeze()
                    ax[0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                    # Query image
                    qi_tensor = query_images[0].cpu()  # Should be [C, H, W] or [B, C, H, W]
                    
                    # Handle potential batch dimension
                    if qi_tensor.dim() == 4:  # [B, C, H, W]
                        qi_tensor = qi_tensor[0]  # Take first batch -> [C, H, W]
                    
                    if qi_tensor.dim() == 3 and qi_tensor.shape[0] == 3:
                        qi = (qi_tensor * std[0] + mean[0]).numpy().transpose((1,2,0))
                    elif qi_tensor.dim() == 3:
                        qi = qi_tensor.numpy()
                        if qi.shape[0] == 1:  # Single channel
                            qi = qi.squeeze(0)  # [H, W]
                        else:  # Multiple channels
                            qi = qi.transpose((1,2,0))  # [H, W, C]
                    else:  # [H, W]
                        qi = qi_tensor.numpy()
                    
                    qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                    if len(qi.shape) == 2:
                        ax[1].imshow(qi, cmap='gray')
                    else:
                        ax[1].imshow(qi)
                        
                    # Query labels and predictions
                    qm = query_labels[0].cpu().numpy()
                    ax[1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                    qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0]
                    ax[1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)
                    
                except Exception as e:
                    print(f"Visualization error: {e}")
                    # Create simple placeholder plots
                    ax[0].text(0.5, 0.5, 'Support\nVisualization\nError', ha='center', va='center')
                    ax[1].text(0.5, 0.5, 'Query\nVisualization\nError', ha='center', va='center')

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, contrastive_loss: {contrastive_loss}, time: {(nt-stime)/60} mins')

                # # --- Organ visualization every print_interval (250 steps) ---
                # try:
                #     if buffer_teacher_features:
                #         # Concatenate all accumulated features
                #         all_teacher_features = torch.cat(buffer_teacher_features, dim=0)
                #         all_student_features = torch.cat(buffer_student_features, dim=0)
                #         all_teacher_masks = torch.cat(buffer_teacher_masks, dim=0)
                #         all_student_masks = torch.cat(buffer_student_masks, dim=0)
                #         all_organ_classes = torch.cat(buffer_organ_classes, dim=0)
                #         # Debug: print unique superpixel classes in buffer  
                #         print("Superpixel classes in buffer:", torch.unique(all_organ_classes))
                #         # Create organ visualizer for superpixel analysis
                #         viz_dir = f'{_run.observers[0].dir}/superpixel_visualizations'
                #         visualizer = create_organ_visualizer(dataset_name=baseset_name, save_dir=viz_dir, superpixel_mode=True)
                #         with torch.no_grad():
                #             viz_files = visualizer.visualize_contrastive_analysis(
                #                 teacher_features=all_teacher_features,
                #                 student_features=all_student_features,
                #                 teacher_masks=all_teacher_masks,
                #                 student_masks=all_student_masks,
                #                 organ_classes=all_organ_classes,
                #                 title=f"Self-Supervised Superpixel Contrastive Analysis - Step {i_iter+1}"
                #             )
                #         print(f'Superpixel contrastive analysis saved for iteration {i_iter+1}')
                #         print(f'   t-SNE: {viz_files["tsne"]}')
                #         print(f'   Similarity Matrix: {viz_files["similarity"]}')
                #         if viz_files.get("feature_maps"):
                #             print(f'   Feature Maps: {viz_files["feature_maps"]}')
                #         # Clear buffers after visualization
                #         buffer_teacher_features.clear()
                #         buffer_student_features.clear()
                #         buffer_teacher_masks.clear()
                #         buffer_student_masks.clear()
                #         buffer_organ_classes.clear()
                # except Exception as e:
                #     print(f'Superpixel visualization failed: {e}')
                #     import traceback
                #     traceback.print_exc()

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
            

#Supervised Training
# """
# Training the model - M1 Mac Compatible Version
# Extended from original implementation of PANet by Wang et al.
# """
# import os
# import shutil
# import torch
# import torch.nn as nn
# import torch.optim
# from torch.utils.data import DataLoader 
# from torch.optim.lr_scheduler import MultiStepLR
# import torch.backends.cudnn as cudnn
# import numpy as np
# import subprocess



# from models.grid_proto_fewshot import FewShotSeg
# #from models.contrastive import SupervisedPixelWiseContrastiveLoss
# from dataloaders.dev_customized_med import med_fewshot
# #from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
# from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
# from dataloaders.dataset_utils import DATASET_INFO # contains information about the dataset
# import dataloaders.augutils as myaug # contains data augmentation functions

# from util.utils import set_seed, t2n, to01, compose_wt_simple#, get_tversky_loss
# from util.metric import Metric
# #from util.device_utils import setup_device_and_threads, to_device  # New import
# #from util.organ_visualization import create_organ_visualizer  # Updated import for organ visualization
# #from util.visualization import visualize_tsne_with_labels
# from config_ssl_upload import ex
# import tqdm
# import time

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # config pre-trained model caching path
# os.environ['TORCH_HOME'] = "./pretrained_model" # sets an environment variable that tells PyTorch where to store and look for pre-trained models

# @ex.automain # decorator
# def main(_run, _config, _log): # code according to sacred xperimental framework setup _run: Sacred's run object that tracks the experiment
# # _config: Contains all configuration parameters
# # _log: Sacred's logger object
#     if _run.observers:
#         os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
#         os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
#         for source_file, _ in _run.experiment_info['sources']:
#             os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
#                         exist_ok=True)# exist_ok=True: Won't crash if directories already exist
#             _run.observers[0].save_file(source_file, f'source/{source_file}')
#         shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

#     set_seed(_config['seed']) # setting up random seed
#     cudnn.enabled = True
#     cudnn.benchmark = True
#     torch.cuda.set_device(device=_config['gpu_id'])
#     torch.set_num_threads(1)
   

#     _log.info('###### Create model ######')

#     model = FewShotSeg(pretrained_path=None, cfg=_config['model'])

#     model = model.cuda()
#     model.train()
    

#     # Move model to appropriate device
#     model = model.cuda()
#     model.train()

#     _log.info('###### Load data ######')
#     ### Training set
#     data_name = _config['dataset']
#     if data_name == 'SABS_Superpix':
#         baseset_name = 'SABS'
#     elif data_name == 'C0_Superpix':
#         raise NotImplementedError
#         baseset_name = 'C0'
#     elif data_name == 'CHAOST2_Superpix':
#         baseset_name = 'CHAOST2'
#     else:
#         raise ValueError(f'Dataset: {data_name} not found')

#     ###================== Transforms for data augmentation =============================== ###
#     tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]}) # applying data augmentation to the training data
#     ### ================================================================================== ###
#     assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

#     test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]] # labels excluded in training- all labels minus the labels in the current fold
#     _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
#     _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')
#      # Create training dataset

#     # Filter act_labels to exclude excluded classes
#     all_labels = list(DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'])
#     #print(all_labels)
#     excluded = _config["exclude_cls_list"]
#     act_labels = [lbl for lbl in all_labels if lbl not in excluded]
#     print(act_labels)

    
#     dataset, tr_parent = med_fewshot(
#          dataset_name=baseset_name,
#          base_dir=_config['path'][data_name]['data_dir'],
#          idx_split=_config['eval_fold'],
#          mode='train',
#          scan_per_load=_config['scan_per_load'],
#          transforms=tr_transforms,
#          act_labels=act_labels,
#          n_ways=1,
#          n_shots=1,
#          nsup=_config['task']['n_shots'],
#          fix_parent_len=_config["max_iters_per_load"] if _config['fix_length'] else None,
#          max_iters_per_load=_config["max_iters_per_load"],
#          min_fg=str(_config["min_fg_data"]),
#           n_queries=1,
#          exclude_list=excluded,
#          dataset_config=_config['DATASET_CONFIG'],
#          client_eval=True if _config.get('client_eval', False) else False
#      )
#      #print(act_labels, excluded)
#     dataset.norm_func = tr_parent.norm_func
#     print("Dataset length:", len(dataset))   

#      ### dataloaders: DataLoader is a class in PyTorch that wraps an iterable around the dataset to enable easy access to batches.
#     trainloader = DataLoader(
#          dataset, # tr_parent is the dataset object, it is a collection of data and labels, it is used to load the data and labels into the model
#          batch_size=_config['batch_size'],
#          shuffle=True, # Whether to randomize selection after each max_iter_per_load
#          # shuffle=False, sequential 
#          num_workers=_config['num_workers'],
#          #pin_memory=True if device.type == 'cuda' else False,  # Only pin memory for CUDA
#          drop_last=True # ?
#      )

#     _log.info('###### Set optimizer ######')
#     if _config['optim_type'] == 'sgd':
#         print("Using SGD optimizer")
#         optimizer = torch.optim.SGD(model.parameters(), **_config['optim']) # setting up the optimizer, they are used to unpack (spread out) the contents of a list/tuple (*) or a dictionary (**) when calling a function or creating a new dictionary.

#     elif _config['optim_type'] == 'adam':
#         print("Using ADAM optimizer")
#         optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
#     else:
#         raise NotImplementedError

#     scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma']) # learning rate scheduler

#     # Modified weight composition for device compatibility
#     my_weight = compose_wt_simple(_config["use_wce"], data_name) # weight composition for the loss function
#     # compose_wt_simple: This is a function (likely defined elsewhere in your code) that creates a tensor of weights for each class in your segmentation task.
#     # _config["use_wce"]: A config flag (probably a boolean) that determines whether to use Weighted Cross Entropy (WCE).
#     # data_name: The name of the dataset, which may affect how the weights are set (e.g., different datasets may have different class imbalances).
#     #nn.CrossEntropyLoss: The standard loss function for multi-class classification/segmentation in PyTorch.
#     # ignore_index=_config['ignore_label']:
#     # Tells the loss function to ignore pixels/labels with this value (e.g., 255), which are usually background or unlabeled regions.
#     # weight=my_weight:
#     # Passes the class weights tensor (from above) to the loss function.
#     # If my_weight is not None, the loss for each class will be scaled by its corresponding weight.
#     criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight) # loss function

#     i_iter = 0 # total number of iteration
#     n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] #deciding the number of epochs 

#     log_loss = {'loss': 0, 'align_loss': 0, 'contrastive_loss': 0} 

#     # # --- Buffers for accumulating features for liver (6) and spleen (1) ---
#     # buffer_teacher_features = []
#     # buffer_student_features = []
#     # buffer_teacher_masks = []
#     # buffer_student_masks = []
#     # buffer_organ_classes = []
#     # Generalize: all SABS organs except kidneys (2, 3)
#     #from dataloaders.dataset_utils import DATASET_INFO
#     sabs_label_indices = list(range(1, len(DATASET_INFO[baseset_name]['REAL_LABEL_NAME'])))
#     organs_of_interest = [idx for idx in sabs_label_indices if idx not in (2, 3)] 
#     # ---------------------------------------------------------------

#     _log.info('###### Training ######')
#     stime = time.time()
#     for sub_epoch in range(n_sub_epoches):
#         _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
#         for _, sample_batched in enumerate(trainloader): # trainloader is the dataloader defined in torch.utils.data , sample_batched is the batch of data
#             # Prepare input
#             i_iter += 1
#             #print(i_iter)
#             # Modified to use device-agnostic approach
#             # print("About to print sample_batched", flush=True)
#             # print(sample_batched, flush=True) 
#             # print("Printed sample_batched", flush=True)
#             support_images = [[shot.float().cuda() for shot in way] # 'support_images': [[tensor([...])]]  # Shape: [1 way][1 shot][C, H, W]
#                               for way in sample_batched['support_images']] # support_images is a list of lists, each list contains the support images for a way
#             support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] # contains class_ids, support_images, support_fg_mask, support_bg_mask, query_images, query_labels, support_para, query_parameter, mean, std   
#                                for way in sample_batched['support_mask']] # support_fg_mask is a list of lists, each list contains the foreground mask for a way
#             support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
#                                for way in sample_batched['support_mask']] # support_bg_mask is a list of lists, each list contains the background mask for a way

#             query_images = [query_image.float().cuda()
#                             for query_image in sample_batched['query_images']]
#             query_labels = torch.cat(
#                 [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
#             class_ids=  sample_batched['class_ids']
#             optimizer.zero_grad()
#             # update the student encoder with the teacher encoder using momentum
#             model.update_student_encoder(model.student_encoder)
            
#             mean = [m for m in sample_batched["mean"]]
#             std = [s for s in sample_batched["std"]]
#             ########################################################################
#             query_pred, align_loss, debug_vis, assign_mats, contrastive_loss, supp_fts_teacher, supp_fts_student, res_fg_msk_teacher, res_fg_msk_student = model(support_images,
#                                                                     support_fg_mask, 
#                                                                     support_bg_mask, 
#                                                                     query_images,class_ids, 
#                                                                     isval = False, val_wsize = None) # passing the data to the model
#             ########################################################################

#             query_loss = criterion(query_pred, query_labels) #+ get_tversky_loss(query_pred.argmax(dim = 1, keepdim = True), query_labels[None, ...], 0.3, 0.7 ,1.0)
#             loss = query_loss + align_loss + contrastive_loss
#             loss.backward()
#             optimizer.step()

#             # # After optimizer step
#             # teacher_params_after = [p.clone().detach() for p in model.teacher_encoder.parameters()]
#             # student_params_after = [p.clone().detach() for p in model.student_encoder.parameters()]

#             # # Compare
#             # teacher_changed = any([(before != after).any() for before, after in zip(teacher_params_before, teacher_params_after)])
#             # student_changed = any([(before != after).any() for before, after in zip(student_params_before, student_params_after)])

#             # print(f"Teacher params changed: {teacher_changed}")
#             # print(f"Student params changed: {student_changed}")

#             scheduler.step()

#             # Log loss
#             query_loss = query_loss.detach().data.cpu().numpy()
#             align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
#             contrastive_loss = contrastive_loss.detach().data.cpu().numpy() if contrastive_loss != 0 else 0

#             _run.log_scalar('loss', query_loss)
#             _run.log_scalar('align_loss', align_loss)
#             _run.log_scalar('contrastive_loss', contrastive_loss)
           

#             log_loss['loss'] += query_loss # query_loss is the loss for the query images
#             log_loss['align_loss'] += align_loss # align_loss is the loss for the alignment of the support images and the query images
#             log_loss['contrastive_loss'] += contrastive_loss # contrastive_loss is the loss for the contrastive loss

#             # # --- Accumulate features for liver and spleen only ---
#             # # Flatten features and masks as in the visualization code
#             # supp_fts_teacher_flat = supp_fts_teacher.view(-1, *supp_fts_teacher.shape[-3:])
#             # supp_fts_student_flat = supp_fts_student.view(-1, *supp_fts_student.shape[-3:])
#             # res_fg_msk_teacher_flat = res_fg_msk_teacher.view(-1, *res_fg_msk_teacher.shape[-2:])
#             # res_fg_msk_student_flat = res_fg_msk_student.view(-1, *res_fg_msk_student.shape[-2:])
#             # class_ids = sample_batched.get('class_ids', [1])
#             # n_ways = len(support_images)
#             # n_shots = len(support_images[0])
#             # sup_bsize = len(support_images[0][0])
#             # organ_classes = torch.zeros(n_ways, n_shots, sup_bsize).cuda()
#             # for way in range(n_ways):
#             #     organ_classes[way, :, :] = class_ids[way]
#             # organ_classes_flat = organ_classes.view(-1)
#             # # Only keep indices for liver and spleen
#             # mask_interest = torch.isin(organ_classes_flat.long(), torch.tensor(organs_of_interest, dtype=torch.long).cuda())
#             # if mask_interest.any():
#             #     buffer_teacher_features.append(supp_fts_teacher_flat[mask_interest])
#             #     buffer_student_features.append(supp_fts_student_flat[mask_interest])
#             #     buffer_teacher_masks.append((res_fg_msk_teacher_flat[mask_interest] > 0.5).float())
#             #     buffer_student_masks.append((res_fg_msk_student_flat[mask_interest] > 0.5).float())
#             #     buffer_organ_classes.append(organ_classes_flat[mask_interest])
#             # ---------------------------------------------------------------

#             # print loss and take snapshots
#             if (i_iter + 1) % _config['print_interval'] == 0:

#                 nt = time.time()

#                 loss = log_loss['loss'] / _config['print_interval']
#                 align_loss = log_loss['align_loss'] / _config['print_interval']

#                 log_loss['loss'] = 0
#                 log_loss['align_loss'] = 0
#                 log_loss['contrastive_loss'] = 0

#                 fig, ax = plt.subplots(1,2)
#                 si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
#                 si = (si - si.min())/(si.max() - si.min() + 1e-6)
#                 ax[0].imshow(si)
#                 sm = support_fg_mask[0][0][0].cpu().numpy()
#                 ax[0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

#                 qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
#                 qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
#                 ax[1].imshow(qi)
#                 qm = query_labels[0].cpu().numpy()
#                 ax[1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
#                 qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0]
#                 ax[1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)

#                 plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
#                 plt.close(fig)
              

#                 print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, contrastive_loss: {contrastive_loss}, time: {(nt-stime)/60} mins')


#                 # # --- Organ visualization every print_interval (250 steps) ---
#                 # try:
#                 #     if buffer_teacher_features:
#                 #         # Concatenate all accumulated features
#                 #         all_teacher_features = torch.cat(buffer_teacher_features, dim=0)
#                 #         all_student_features = torch.cat(buffer_student_features, dim=0)
#                 #         all_teacher_masks = torch.cat(buffer_teacher_masks, dim=0)
#                 #         all_student_masks = torch.cat(buffer_student_masks, dim=0)
#                 #         all_organ_classes = torch.cat(buffer_organ_classes, dim=0)
#                 #         # Debug: print unique organs in buffer
#                 #         #print("Organs in buffer:", torch.unique(all_organ_classes))
#                 #         # Create organ visualizer
#                 #         viz_dir = f'{_run.observers[0].dir}/organ_visualizations'
#                 #         visualizer = create_organ_visualizer(dataset_name=baseset_name, save_dir=viz_dir)
#                 #         with torch.no_grad():
#                 #             viz_files = visualizer.visualize_contrastive_analysis(
#                 #                 teacher_features=all_teacher_features,
#                 #                 student_features=all_student_features,
#                 #                 teacher_masks=all_teacher_masks,
#                 #                 student_masks=all_student_masks,
#                 #                 organ_classes=all_organ_classes,
#                 #                 title=f"Contrastive Analysis (All Organs Except Kidneys) - Step {i_iter+1}"
#                 #             )
#                 #         # print(f'Organ contrastive analysis (all except kidneys) saved for iteration {i_iter+1}')
#                 #         # print(f'   t-SNE: {viz_files["tsne"]}')
#                 #         # print(f'   Similarity Matrix: {viz_files["similarity"]}')
#                 #         if viz_files["feature_maps"]:
#                 #             #print(f'   Feature Maps: {viz_files["feature_maps"]}')
#                 #             print('')
#                 #         # Clear buffers after visualization
#                 #         buffer_teacher_features.clear()
#                 #         buffer_student_features.clear()
#                 #         buffer_teacher_masks.clear()
#                 #         buffer_student_masks.clear()
#                 #         buffer_organ_classes.clear()
#                 # except Exception as e:
#                 #     print(f'Organ visualization failed: {e}')
#                 #     import traceback
#                 #     traceback.print_exc()

#             if (i_iter + 1) % _config['save_snapshot_every'] == 0:
#                 _log.info('###### Taking snapshot ######')
#                 checkpoint_path = os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth')
#                 torch.save({'model':model.state_dict(),'opt':optimizer.state_dict(),'sch':scheduler.state_dict()},
#                             checkpoint_path)
             

            

#             if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
#                 if (i_iter + 1) % _config['max_iters_per_load'] == 0:
#                     _log.info('###### Reloading dataset ######')
#                     trainloader.dataset.reload_buffer()
#                     print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

#             if (i_iter - 2) > _config['n_steps']:
#                 return 1 # finish up
            
