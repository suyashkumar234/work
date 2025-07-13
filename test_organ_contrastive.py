#!/usr/bin/env python3
"""
Test script for organ-aware contrastive loss
This demonstrates how the new loss function clusters organs together
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models.contrastive_loss import OrganAwareContrastiveLoss

def create_synthetic_organ_data():
    """Create synthetic data with multiple organs"""
    # Create a 32x32 feature map with 256 channels
    H, W, C = 32, 32, 256
    
    # Create teacher features
    feat_teacher = torch.randn(C, H, W)
    
    # Create student features (similar but with some noise)
    feat_student = feat_teacher + 0.1 * torch.randn_like(feat_teacher)
    
    # Create mask with multiple organs
    mask = torch.zeros(H, W)
    
    # Organ 1: Top-left region
    mask[5:15, 5:15] = 1
    
    # Organ 2: Top-right region  
    mask[5:15, 17:27] = 1
    
    # Organ 3: Bottom region
    mask[17:27, 10:22] = 1
    
    return feat_teacher, feat_student, mask

def visualize_organ_clustering(feat_teacher, feat_student, mask, organ_sim, save_path):
    """Visualize organ clustering results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original mask
    axes[0, 0].imshow(mask.numpy(), cmap='gray')
    axes[0, 0].set_title('Original Mask')
    axes[0, 0].axis('off')
    
    # Teacher features (first channel)
    axes[0, 1].imshow(feat_teacher[0].numpy(), cmap='viridis')
    axes[0, 1].set_title('Teacher Features (Ch0)')
    axes[0, 1].axis('off')
    
    # Student features (first channel)
    axes[0, 2].imshow(feat_student[0].numpy(), cmap='viridis')
    axes[0, 2].set_title('Student Features (Ch0)')
    axes[0, 2].axis('off')
    
    # Organ similarity matrix
    im = axes[1, 0].imshow(organ_sim.numpy(), cmap='viridis')
    axes[1, 0].set_title('Organ Similarity Matrix')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Feature difference
    diff = feat_teacher[0] - feat_student[0]
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im = axes[1, 1].imshow(diff.numpy(), cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    axes[1, 1].set_title('Teacher-Student Difference')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Organ clusters visualization
    organ_clusters = organ_sim > 0.7
    axes[1, 2].imshow(organ_clusters.numpy(), cmap='tab10')
    axes[1, 2].set_title('Organ Clusters')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def test_organ_contrastive_loss():
    """Test the organ-aware contrastive loss"""
    print("Testing Organ-Aware Contrastive Loss...")
    
    # Create synthetic data
    feat_teacher, feat_student, mask = create_synthetic_organ_data()
    
    # Initialize the loss function
    loss_fn = OrganAwareContrastiveLoss(
        temperature=0.1,
        spatial_weight=0.3,
        feature_weight=0.7,
        spatial_threshold=5.0,
        min_organ_size=10
    )
    
    # Prepare inputs (add batch dimensions)
    feat_teacher_batch = feat_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    feat_student_batch = feat_student.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    mask_batch = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    print(f"Input shapes:")
    print(f"feat_teacher_batch: {feat_teacher_batch.shape}")
    print(f"feat_student_batch: {feat_student_batch.shape}")
    print(f"mask_batch: {mask_batch.shape}")
    
    # Compute loss
    loss = loss_fn(feat_teacher_batch, feat_student_batch, mask_batch)
    
    print(f"Organ-aware contrastive loss: {loss.item():.4f}")
    
    # Let's also compute the old pixel-wise loss for comparison
    from models.contrastive_loss import PixelWiseContrastiveLoss
    old_loss_fn = PixelWiseContrastiveLoss(temperature=0.1)
    old_loss = old_loss_fn(feat_teacher_batch, feat_student_batch, mask_batch)
    
    print(f"Pixel-wise contrastive loss: {old_loss.item():.4f}")
    
    # Create visualization to show organ clustering
    # Extract the organ similarity computation from the loss function
    fg_indices = torch.where(mask == 1)
    feat_teacher_fg = feat_teacher[:, fg_indices[0], fg_indices[1]]
    feat_student_fg = feat_student[:, fg_indices[0], fg_indices[1]]
    
    # Normalize features
    feat_teacher_fg = F.normalize(feat_teacher_fg, dim=0)
    feat_student_fg = F.normalize(feat_student_fg, dim=0)
    
    # Create spatial coordinates
    y_coords = fg_indices[0].float().unsqueeze(0)
    x_coords = fg_indices[1].float().unsqueeze(0)
    coords = torch.cat([y_coords, x_coords], dim=0)
    
    # Compute similarities
    spatial_dist = torch.cdist(coords.t(), coords.t())
    spatial_sim = torch.exp(-spatial_dist / 5.0)
    feature_sim_teacher = torch.mm(feat_teacher_fg.t(), feat_teacher_fg)
    feature_sim_student = torch.mm(feat_student_fg.t(), feat_student_fg)
    feature_sim = (feature_sim_teacher + feature_sim_student) / 2
    
    # Combined similarity
    organ_sim = 0.3 * spatial_sim + 0.7 * feature_sim
    
    # Visualize
    visualize_organ_clustering(feat_teacher, feat_student, mask, organ_sim, 'organ_clustering_test.png')
    
    print("\nKey differences:")
    print("1. Organ-aware loss clusters similar organs together")
    print("2. Uses spatial proximity + feature similarity")
    print("3. Positive pairs: pixels from same organ across encoders")
    print("4. Negative pairs: pixels from different organs")
    print("5. More meaningful for medical image segmentation")

if __name__ == "__main__":
    test_organ_contrastive_loss() 