"""
Experiment configuration file
Extended from config file from original PANet Repository
"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from platform import node
from datetime import datetime

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('mySSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train' # for now only allows 'train' 
    num_workers = 4 # 0 for debugging. 

    dataset = 'CHAOST2_Superpix' # i.e. abdominal MRI
    use_coco_init = True # initialize backbone with MS_COCO initialization. Anyway coco does not contain medical images

    ### Training
    n_steps = 100100
    batch_size = 1
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 500
    save_snapshot_every = 25000
    max_iters_per_load = 1000 # epoch size, interval for reloading the dataset
    scan_per_load = -1 # numbers of 3d scans per load for saving memory. If -1, load the entire dataset to the memory
    which_aug = 'sabs_aug' # standard data augmentation with intensity and geometric transforms
    input_size = (256, 256)
    min_fg_data='100' # when training with manual annotations, indicating number of foreground pixels in a single class single slice. This empirically stablizes the training process
    label_sets = 0 # which group of labels taking as training (the rest are for testing)
    exclude_cls_list = [2, 3] # testing classes to be excluded in training. Set to [] if testing under setting 1
    usealign = True # see vanilla PANet
    use_wce = True
    viz = 1

    ### Validation
    z_margin = 0 
    eval_fold = 0 # which fold for 5 fold cross validation
    support_idx=[4] # indicating which scan is used as support in testing. 
    val_wsize=2 # L_H, L_W in testing
    n_sup_part = 3 # number of chuncks in testing

    # Network
    modelname = 'resnet50' # resnet 101 backbone from torchvision fcn-deeplab
    clsname = 'grid_proto' # 
    resume = False
    reload_model_path = './exps/myexperiments_MIDDLE_0/mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold2_CHAOST2_Superpix_sets_0_1shot/4/snapshots/20000.pth' #'./runs/mySSL__CHAOST2_Superpix_sets_0_1shot/314_best_ref/snapshots/100000.pth' # path for reloading a trained model (overrides ms-coco initialization)
    proto_grid_size = 8 # L_H, L_W = (32, 32) / 8 = (4, 4)  in training
    feature_hw = [32, 32] # feature map size, should couple this with backbone in future

    # SSL
    superpix_scale = 'MIDDLE' #MIDDLE/ LARGE

    tversky_params = {'tversky_alpha' : 0.3,
                    'tversky_beta' : 0.7,
                    'tversky_gamma' : 1.0}

    lambda_loss = {'loss1':0.0, 'loss2':1.0, 'loss3':0.0, 'loss4':0.0, 'loss5': 0.0}

    accum_iter = 1


    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size' : proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
        'npart': n_sup_part 
    }

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3, 
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = ''

    exp_str = '_'.join(
        [exp_prefix]
        + [dataset,]
        + [f'sets_{label_sets}_{task["n_shots"]}shot'])

    path = {
        'log_dir': './runs',
        'SABS':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/SABS/Abdomen/RawData/Training/sabs_CT_normalized"
            },
        'C0':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data"
            },
        'CHAOST2':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/CHAOS/CHAOS_Train_Sets/Train_Sets/chaos_MR_T2_normalized/"
            },
        'SABS_Superpix':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/SABS/Abdomen/RawData/Training/sabs_CT_normalized"},
        'C0_Superpix':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data"},
        'CHAOST2_Superpix':{'data_dir': "E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/CHAOS/CHAOS_Train_Sets/Train_Sets/chaos_MR_T2_normalized/"},
        }

    DATASET_CONFIG = {'SABS':{'img_bname': f'E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/SABS/Cervix/RawData/Training/sabs_CT_normalized/image_*.nii.gz',
                        'out_dir': 'E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/SABS/Cervix/RawData/Training/sabs_CT_normalized',
                        'fg_thresh': 1e-4,
                        },
                      'CHAOST2':{
                       'img_bname': f'E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/CHAOS/CHAOS_Train_Sets/Train_Sets/chaos_MR_T2_normalized/image_*.nii.gz',
                          'out_dir': 'E:/Siladittya_JRF/cvpr2024/agun-sona-master/data/CHAOS/CHAOS_Train_Sets/Train_Sets/chaos_MR_T2_normalized',
                          'fg_thresh': 1e-4 + 50,
                        },
                     }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config