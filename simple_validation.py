#!/usr/bin/env python3
"""
Simplified validation script for your trained model
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import your model and dataloaders
from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from util.metric import Metric

def run_validation():
    """Run validation on your trained model"""
    
    # Configuration
    config = {
        'dataset': 'SABS_Superpix',
        'eval_fold': 0,
        'exclude_cls_list': [2, 3],  # Kidneys excluded from training
        'val_wsize': 2,
        'z_margin': 0,
        'task': {'n_shots': 1, 'npart': 3},
        'reload_model_path': './exps/myexp_MIDDLE_0/mySSL_train_SABS_Superpix_lbgroup0_scale_MIDDLE_vfold0_SABS_Superpix_sets_0_1shot/267/snapshots/30.pth',
        'path': {
            'SABS': {'data_dir': "/Users/suyash/Desktop/cowpro/data/SABS/sabs_CT_normalized"}
        }
    }
    
    print("=== Starting Validation ===")
    print(f"Loading model from: {config['reload_model_path']}")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_config = {
        'align': True,
        'use_coco_init': True,
        'which_model': 'dlfcn_res101',
        'cls_name': 'grid_proto',
        'proto_grid_size': 8,
        'feature_hw': [32, 32],
    }
    
    model = FewShotSeg(pretrained_path=None, cfg=model_config)
    
    # Load trained weights
    checkpoint = torch.load(config['reload_model_path'], map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Setup dataset
    baseset_name = 'SABS'
    max_label = 13
    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][0]
    print(f"Testing on labels: {sorted(test_labels)}")
    
    # For CT, we need to get the file paths for normalization statistics
    # But since this is validation, we'll use a simple approach
    def simple_normalize(x_in):
        # Use dataset statistics that you saw during training
        ct_mean = 0.17245544170525606 * 255  # Convert back from normalized
        ct_std = 0.24654958303807592 * 255
        return (x_in - ct_mean) / ct_std, ct_mean, ct_std
    
    norm_func = simple_normalize
    
    # Create test dataset
    te_dataset, te_parent = med_fewshot_val(
        dataset_name=baseset_name,
        base_dir=config['path']['SABS']['data_dir'],
        idx_split=config['eval_fold'],
        scan_per_load=-1,
        act_labels=test_labels,
        npart=config['task']['npart'],
        nsup=config['task']['n_shots'],
        extern_normalize_func=norm_func
    )
    
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"Test dataset loaded: {len(te_dataset)} samples")
    
    # Setup metrics
    scan_ids = te_parent.scan_ids
    metric_nodes = {}
    
    print("=== Running Validation ===")
    
    with torch.no_grad():
        for sup_idx in range(len(scan_ids)):
            scan_id = scan_ids[sup_idx]
            print(f"\\n--- Processing support scan {scan_id} ---")
            
            # Initialize metrics for this support scan
            metric_nodes[scan_id] = Metric(max_label=max_label, n_scans=len(te_dataset.dataset.pid_curr_load) - config['task']['n_shots'])
            metric_nodes[scan_id].reset()
            
            # Process each test label
            for curr_lb in sorted(test_labels):
                print(f"Testing label {curr_lb}...")
                
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(
                    curr_class=curr_lb,
                    class_idx=[curr_lb],
                    scan_idx=[sup_idx],
                    npart=config['task']['npart']
                )
                
                # Prepare support data
                support_images = [[shot.to(device) for shot in way] 
                                 for way in support_batched['support_images']]
                support_fg_mask = [[shot['fg_mask'].float().to(device) for shot in way]
                                  for way in support_batched['support_mask']]
                support_bg_mask = [[shot['bg_mask'].float().to(device) for shot in way]
                                  for way in support_batched['support_mask']]
                
                curr_scan_count = -1
                
                # Process query samples
                for sample_batched in testloader:
                    scan_id_query = sample_batched["scan_id"][0]
                    
                    # Skip support scans
                    if scan_id_query in te_parent.potential_support_sid:
                        continue
                        
                    if sample_batched["is_start"]:
                        curr_scan_count += 1
                        
                    # Get query data
                    q_part = sample_batched["part_assign"]
                    query_images = [sample_batched['image'].to(device)]
                    query_labels = sample_batched['label'].to(device)
                    
                    # Prepare support for this query part
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                    
                    # Run inference
                    model_output = model(sup_img_part, sup_fgm_part, sup_bgm_part, query_images,
                                       class_ids=[curr_lb], isval=True, val_wsize=config['val_wsize'])
                    
                    # Handle different return formats
                    if isinstance(model_output, tuple):
                        query_pred = model_output[0]
                    else:
                        query_pred = model_output
                    
                    # Convert to predictions
                    query_pred = query_pred.argmax(dim=1).float().cpu().numpy()
                    query_labels = query_labels.detach().cpu().numpy()
                    
                    # Record metrics (only within z_margin)
                    z_id = sample_batched["z_id"].item()
                    z_min = sample_batched["z_min"].item()
                    z_max = sample_batched["z_max"].item()
                    
                    if (z_id - z_max <= config['z_margin']) and (z_id - z_min >= -config['z_margin']):
                        metric_nodes[scan_id].record(query_pred[0], query_labels[0], 
                                                   labels=[curr_lb], n_scan=curr_scan_count)
    
    # Compute and display results
    print("\\n=== Validation Results ===")
    
    all_dice_scores = []
    all_precision_scores = []
    all_recall_scores = []
    
    for i, scan_id in enumerate(scan_ids):
        print(f"\\n--- Results for Support Scan {scan_id} ---")
        
        # Get metrics
        m_classDice, _, m_meanDice, _, m_rawDice = metric_nodes[scan_id].get_mDice(
            labels=sorted(test_labels), n_scan=None, give_raw=True)
        
        m_classPrec, _, m_meanPrec, _, m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = metric_nodes[scan_id].get_mPrecRecall(
            labels=sorted(test_labels), n_scan=None, give_raw=True)
        
        print(f"Class Dice scores: {m_classDice}")
        print(f"Mean Dice score: {m_meanDice}")
        print(f"Class Precision: {m_classPrec}")  
        print(f"Mean Precision: {m_meanPrec}")
        print(f"Class Recall: {m_classRec}")
        print(f"Mean Recall: {m_meanRec}")
        
        all_dice_scores.append(m_meanDice)
        all_precision_scores.append(m_meanPrec)
        all_recall_scores.append(m_meanRec)
        
        metric_nodes[scan_id].reset()
    
    # Overall statistics
    print(f"\\n=== Overall Results Across All Support Scans ===")
    print(f"Average Dice Score: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"Average Precision: {np.mean(all_precision_scores):.4f} ± {np.std(all_precision_scores):.4f}")
    print(f"Average Recall: {np.mean(all_recall_scores):.4f} ± {np.std(all_recall_scores):.4f}")
    
    print("\\n=== Validation Complete ===")
    return {
        'dice_scores': all_dice_scores,
        'precision_scores': all_precision_scores,
        'recall_scores': all_recall_scores,
        'mean_dice': np.mean(all_dice_scores),
        'mean_precision': np.mean(all_precision_scores),
        'mean_recall': np.mean(all_recall_scores)
    }

if __name__ == "__main__":
    try:
        results = run_validation()
        print(f"\\nValidation completed successfully!")
        print(f"Final Mean Dice Score: {results['mean_dice']:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()