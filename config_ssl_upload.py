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

ex = Experiment('mySSL', save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

# Add these modifications to your config_ssl_upload.py:

@ex.config
def cfg():
    """Default configurations - M1 Mac optimized"""
    seed = 1234
    gpu_id = 0  # Will be ignored on M1 Mac, kept for compatibility
    mode = 'train'
    dataset = 'Sabs_Superpix'
    use_coco_init = True
    # Optimized for M1 Mac
    num_workers = 8 # M1 has good CPU cores, but don't oversubscribe

    ### Training - adjusted for M1 Mac memory constraints
    n_steps = 100100  # Reduced from 100100 for faster testing
    batch_size = 1   # Keep at 1 for memory efficiency
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 5000  # More frequent updates for shorter runs
    save_snapshot_every = 20000  # More frequent saves
    max_iters_per_load = 1000  # Reduced for M1 Mac
    scan_per_load = -1 # Load entire dataset if memory allows
    which_aug = 'sabs_aug'
    input_size = (256, 256)  # Keep reasonable size for M1
    min_fg_data='100'
    label_sets = 0
    exclude_cls_list = [2, 3]
    usealign = True
    use_wce = True
    viz = 1
    fix_length= False
    client_eval=True

    ### Validation
    z_margin = 0 
    eval_fold = 0
    support_idx=[4]
    val_wsize=2
    n_sup_part = 3

  
    # Network
    modelname = 'dlfcn_res101'  # This should work fine on M1
    clsname = 'grid_proto'
    resume = False
    reload_model_path = './exps/your_model_path.pth'  # Update this path
    proto_grid_size = 8
    feature_hw = [32, 32]

    # SSL
    superpix_scale = 'MIDDLE'
    
    # SSL Attention Configuration
    use_ssl_attention = False  # Enable/disable SSL attention module (self + cross attention)
    use_mask_attention = False  # Enable/disable mask-aware attention (foreground-focused)
    ssl_attention_heads = 8  # Number of attention heads (reduced for compatibility)
    ssl_attention_layers = 2  # Number of attention layers
    ssl_attention_dropout = 0.1  # Attention dropout rate
    
    # NOTE: Only one attention type should be True at a time:
    # - Both False: Simple model without attention
    # - use_ssl_attention=True: SSL self + cross attention
    # - use_mask_attention=True: Mask-aware attention

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
        'reload_model_path': reload_model_path,
        'use_ssl_attention': use_ssl_attention,
        'use_mask_attention': use_mask_attention,
        'ssl_attention_heads': ssl_attention_heads,
        'ssl_attention_layers': ssl_attention_layers,
        'ssl_attention_dropout': ssl_attention_dropout,
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

    exp_prefix = ''  # Identify M1 runs

    exp_str = '_'.join([
        exp_prefix,
        f'mask_attn_sets_{label_sets}',
        f'{task["n_shots"]}shot',
        f'fold_{eval_fold}'  # Add fold information
                ])

    path = {
        'log_dir': './exps',
        'SABS':{'data_dir': "/scratch/suyash.kumar.mec22.itbhu/cowpro/data/SABS/sabs_CT_normalized/"  # UPDATE THIS
            },
        'CHAOST2':{'data_dir': "/scratch/suyash.kumar.mec22.itbhu/cowpro/data/CHAOST2/chaos_MR_T2_normalized"  # UPDATE THIS
            },
        'SABS_Superpix':{'data_dir': "/home/suyash.kumar.mec22.itbhu/cowpro/data/SABS/sabs_CT_normalized/"},  # UPDATE THIS
        'CHAOST2_Superpix':{'data_dir': "/scratch/suyash.kumar.mec22.itbhu/cowpro/data/CHAOST2/chaos_MR_T2_normalized"},  # UPDATE THIS
        }

    # Update dataset config paths too
    DATASET_CONFIG = {'SABS':{'img_bname':  f'/scratch/suyash.kumar.mec22.itbhu/cowpro/data/SABS/sabs_CT_normalized/image_*.nii.gz',  # UPDATE
                        'out_dir': '/scratch/suyash.kumar.mec22.itbhu/cowpro/data/SABS/sabs_CT_normalized/',  # UPDATE
                        'fg_thresh': 1e-4,
                        },
                      'CHAOST2':{
                       'img_bname': f'/scratch/suyash.kumar.mec22.itbhu/cowpro/data/CHAOST2/chaos_MR_T2_normalized/image_*.nii.gz',  # UPDATE
                          'out_dir': '/scratch/suyash.kumar.mec22.itbhu/cowpro/data/CHAOST2/chaos_MR_T2_normalized',  # UPDATE
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