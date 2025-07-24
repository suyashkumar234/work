#!/usr/bin/env python3
"""
Organ separation visualization utilities for supervised contrastive learning
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import DATASET_INFO for label mapping
from dataloaders.dataset_utils import DATASET_INFO

class OrganContrastiveVisualizer:
    """
    Visualization utility for organ-aware contrastive learning analysis.
    Supports t-SNE visualization, similarity matrices, and feature map analysis.
    """
    
    def __init__(self, dataset_name='SABS', save_dir='visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            dataset_name: Name of the dataset (SABS, CHAOST2, etc.)
            save_dir: Directory to save visualizations
        """
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Dataset-specific organ mappings
        self.organ_mappings = self._get_organ_mappings()
        
        # Color schemes for different organs
        self.colors = {
            'spleen': '#FF6B6B',      # Red
            'kidney': '#4ECDC4',      # Teal  
            'liver': '#45B7D1',       # Blue
            'pancreas': '#96CEB4',    # Green
            'background': '#D3D3D3',  # Light gray
            'unknown': '#FFD93D'      # Yellow
        }
        
        # Marker styles for training vs testing organs
        self.markers = {
            'training': 'o',  # Circle
            'testing': 's'    # Square
        }
        
    def _get_organ_mappings(self):
        """Get organ class mappings for different datasets using DATASET_INFO."""
        if self.dataset_name in DATASET_INFO:
            real_label_names = DATASET_INFO[self.dataset_name]['REAL_LABEL_NAME']
            # Map index to label name, lowercased for color matching
            return {idx: name.lower() for idx, name in enumerate(real_label_names)}
        else:
            # Default mapping
            return {
                0: 'background',
                1: 'organ_1',
                2: 'organ_2', 
                3: 'organ_3',
                4: 'organ_4',
                5: 'organ_5'
            }
    
    def visualize_tsne_features(self, teacher_features, student_features, 
                               teacher_masks, student_masks, organ_classes,
                               title="t-SNE Feature Visualization", 
                               max_points=200, perplexity=30):
        """
        Visualize teacher and student features using t-SNE.
        
        Args:
            teacher_features: (B, C, H, W) or (N, C) Teacher feature maps
            student_features: (B, C, H, W) or (N, C) Student feature maps  
            teacher_masks: (B, H, W) or (N,) Teacher binary masks (0/1)
            student_masks: (B, H, W) or (N,) Student binary masks (0/1)
            organ_classes: (B,) or (B,1,1) or (B, H, W) Organ class IDs
            title: Plot title
            max_points: Maximum number of points to visualize
            perplexity: t-SNE perplexity parameter
        """
        # Handle both flattened and unflattened input formats
        if teacher_features.dim() == 4:  # (B, C, H, W)
            # Normalize features
            teacher_features = F.normalize(teacher_features, dim=1)
            student_features = F.normalize(student_features, dim=1)
            
            # Flatten spatial dimensions
            B, C, H, W = teacher_features.shape
            teacher_features_flat = teacher_features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            student_features_flat = student_features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            teacher_masks_flat = teacher_masks.reshape(-1)  # (B*H*W,)
            student_masks_flat = student_masks.reshape(-1)  # (B*H*W,)

            # Ensure organ_classes is broadcasted to (B, H, W) before flattening
            if organ_classes.dim() == 1:
                # (B,) -> (B, H, W)
                organ_classes_expanded = organ_classes.view(B, 1, 1).expand(B, H, W)
            elif organ_classes.dim() == 3 and organ_classes.shape[1] == 1 and organ_classes.shape[2] == 1:
                # (B,1,1) -> (B, H, W)
                organ_classes_expanded = organ_classes.expand(B, H, W)
            else:
                # Already (B, H, W)
                organ_classes_expanded = organ_classes
            organ_classes_flat = organ_classes_expanded.reshape(-1)
            
            # Extract foreground features only
            teacher_fg = teacher_features_flat[teacher_masks_flat == 1]  # (N_fg, C)
            student_fg = student_features_flat[student_masks_flat == 1]  # (N_fg, C)
            
            # Get corresponding organ classes for foreground pixels
            teacher_organs = organ_classes_flat[teacher_masks_flat == 1]
            student_organs = organ_classes_flat[student_masks_flat == 1]
        else:  # (N, C) - already flattened
            # Normalize features
            teacher_features = F.normalize(teacher_features, dim=1)
            student_features = F.normalize(student_features, dim=1)
            
            # Extract foreground features only
            teacher_fg = teacher_features[teacher_masks == 1]  # (N_fg, C)
            student_fg = student_features[student_masks == 1]  # (N_fg, C)
            
            # Get corresponding organ classes for foreground pixels
            teacher_organs = organ_classes[teacher_masks == 1]
            student_organs = organ_classes[student_masks == 1]
        
        # Combine features for t-SNE
        all_features = torch.cat([teacher_fg, student_fg], dim=0)
        all_organs = torch.cat([teacher_organs, student_organs], dim=0)
        
        # Sample points if too many
        if len(all_features) > max_points:
            indices = torch.randperm(len(all_features))[:max_points]
            all_features = all_features[indices]
            all_organs = all_organs[indices]
        
        # Convert to numpy
        features_np = all_features.detach().cpu().numpy()
        organs_np = all_organs.detach().cpu().numpy()

        # --- Fix: Check sample count and adjust perplexity ---
        if len(features_np) < 2:
            print("Not enough samples for t-SNE visualization.")
            return None
        adj_perplexity = min(perplexity, max(1, len(features_np) // 3))
        if len(features_np) < 5:
            print("Too few samples for meaningful t-SNE, skipping.")
            return None
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=adj_perplexity, random_state=42, n_jobs=-1)
        features_2d = tsne.fit_transform(features_np)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # --- Use a color map for all organs, fallback if not in self.colors ---
        
        color_map = plt.get_cmap('tab20')
        unique_organs = np.unique(organs_np)
        organ_color_dict = {}
        for i, organ_id in enumerate(unique_organs):
            organ_name = self.organ_mappings.get(organ_id, f'Organ_{organ_id}')
            # Use predefined color if available, else assign from color map
            color = self.colors.get(organ_name, color_map(i % 20))
            organ_color_dict[organ_id] = color
            mask = organs_np == organ_id
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], label=organ_name, alpha=0.7, s=20)

        ax1.set_title(f"{title} - Organ Classes")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Teacher vs Student
        n_teacher = len(teacher_fg)
        if len(all_features) > n_teacher:
            n_teacher = min(n_teacher, max_points // 2)
            n_student = len(all_features) - n_teacher
            
            # Teacher points
            ax2.scatter(features_2d[:n_teacher, 0], features_2d[:n_teacher, 1], 
                       c='blue', label='Teacher', alpha=0.7, s=20, marker='o')
            
            # Student points  
            ax2.scatter(features_2d[n_teacher:, 0], features_2d[n_teacher:, 1], 
                       c='red', label='Student', alpha=0.7, s=20, marker='s')
        
        ax2.set_title(f"{title} - Teacher vs Student")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tsne_features_{self.dataset_name}_{timestamp}.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename
    
    # def visualize_similarity_matrix(self, teacher_features, student_features,
    #                                teacher_masks, student_masks, organ_classes,
    #                                title="Feature Similarity Matrix"):
    #     """
    #     Visualize similarity matrix between teacher and student features.
        
    #     Args:
    #         teacher_features: (B, C, H, W) or (N, C) Teacher feature maps
    #         student_features: (B, C, H, W) or (N, C) Student feature maps
    #         teacher_masks: (B, H, W) or (N,) Teacher binary masks (0/1)
    #         student_masks: (B, H, W) or (N,) Student binary masks (0/1)
    #         organ_classes: (B,) or (N,) Organ class IDs
    #         title: Plot title
    #     """
    #     # Handle both flattened and unflattened input formats
    #     if teacher_features.dim() == 4:  # (B, C, H, W)
    #         # Normalize features
    #         teacher_features = F.normalize(teacher_features, dim=1)
    #         student_features = F.normalize(student_features, dim=1)
            
    #         # Flatten spatial dimensions
    #         B, C, H, W = teacher_features.shape
    #         teacher_features_flat = teacher_features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    #         student_features_flat = student_features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    #         teacher_masks_flat = teacher_masks.reshape(-1)  # (B*H*W,)
    #         student_masks_flat = student_masks.reshape(-1)  # (B*H*W,)

    #         # Ensure organ_classes is broadcasted to (B, H, W) before flattening
    #         if organ_classes.dim() == 1:
    #             organ_classes_expanded = organ_classes.view(B, 1, 1).expand(B, H, W)
    #         elif organ_classes.dim() == 3 and organ_classes.shape[1] == 1 and organ_classes.shape[2] == 1:
    #             organ_classes_expanded = organ_classes.expand(B, H, W)
    #         else:
    #             organ_classes_expanded = organ_classes
    #         organ_classes_flat = organ_classes_expanded.reshape(-1)
            
    #         # Extract foreground features
    #         teacher_fg = teacher_features_flat[teacher_masks_flat == 1]  # (N_fg, C)
    #         student_fg = student_features_flat[student_masks_flat == 1]  # (N_fg, C)
            
    #         # Get organ classes for foreground pixels
    #         teacher_organs = organ_classes_flat[teacher_masks_flat == 1]
    #         student_organs = organ_classes_flat[student_masks_flat == 1]
    #     else:  # (N, C) - already flattened
    #         # Normalize features
    #         teacher_features = F.normalize(teacher_features, dim=1)
    #         student_features = F.normalize(student_features, dim=1)
            
    #         # Extract foreground features
    #         teacher_fg = teacher_features[teacher_masks == 1]  # (N_fg, C)
    #         student_fg = student_features[student_masks == 1]  # (N_fg, C)
            
    #         # Get organ classes for foreground pixels
    #         teacher_organs = organ_classes[teacher_masks == 1]
    #         student_organs = organ_classes[student_masks == 1]
        
    #     # Sample points if too many (for visualization)
    #     max_points = 1000
    #     if len(teacher_fg) > max_points:
    #         indices = torch.randperm(len(teacher_fg))[:max_points]
    #         teacher_fg = teacher_fg[indices]
    #         teacher_organs = teacher_organs[indices]
        
    #     if len(student_fg) > max_points:
    #         indices = torch.randperm(len(student_fg))[:max_points]
    #         student_fg = student_fg[indices]
    #         student_organs = student_organs[indices]
        
    #     # Compute similarity matrix
    #     similarity_matrix = torch.mm(teacher_fg, student_fg.t())  # (N_teacher, N_student)
        
    #     # Convert to numpy
    #     sim_matrix_np = similarity_matrix.detach().cpu().numpy()
    #     teacher_organs_np = teacher_organs.detach().cpu().numpy()
    #     student_organs_np = student_organs.detach().cpu().numpy()
        
    #     # Create visualization
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
    #     # Plot 1: Raw similarity matrix
    #     im1 = ax1.imshow(sim_matrix_np, cmap='viridis', aspect='auto')
    #     ax1.set_title(f"{title} - Raw Similarity")
    #     ax1.set_xlabel("Student Features")
    #     ax1.set_ylabel("Teacher Features")
    #     plt.colorbar(im1, ax=ax1)
        
    #     # Plot 2: Organ-aware similarity (average per organ pair)
    #     unique_teacher_organs = np.unique(teacher_organs_np)
    #     unique_student_organs = np.unique(student_organs_np)
        
    #     organ_sim_matrix = np.zeros((len(unique_teacher_organs), len(unique_student_organs)))
        
    #     for i, t_organ in enumerate(unique_teacher_organs):
    #         for j, s_organ in enumerate(unique_student_organs):
    #             t_mask = teacher_organs_np == t_organ
    #             s_mask = student_organs_np == s_organ
                
    #             if t_mask.sum() > 0 and s_mask.sum() > 0:
    #                 organ_sim = sim_matrix_np[t_mask][:, s_mask].mean()
    #                 organ_sim_matrix[i, j] = organ_sim
        
    #     # Create organ labels
    #     teacher_labels = [self.organ_mappings.get(org, f'Organ_{org}') for org in unique_teacher_organs]
    #     student_labels = [self.organ_mappings.get(org, f'Organ_{org}') for org in unique_student_organs]
        
    #     im2 = ax2.imshow(organ_sim_matrix, cmap='viridis', aspect='auto')
    #     ax2.set_title(f"{title} - Organ-Aware Similarity")
    #     ax2.set_xlabel("Student Organs")
    #     ax2.set_ylabel("Teacher Organs")
    #     ax2.set_xticks(range(len(student_labels)))
    #     ax2.set_xticklabels(student_labels, rotation=45)
    #     ax2.set_yticks(range(len(teacher_labels)))
    #     ax2.set_yticklabels(teacher_labels)
    #     plt.colorbar(im2, ax=ax2)
        
    #     plt.tight_layout()
        
    #     # Save plot
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"similarity_matrix_{self.dataset_name}_{timestamp}.png"
    #     plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     return filename
    
    # def visualize_feature_maps(self, teacher_features, student_features,
    #                           teacher_masks, student_masks, organ_classes,
    #                           title="Feature Map Visualization", num_channels=8):
    #     """
    #     Visualize feature maps from teacher and student encoders.
        
    #     Args:
    #         teacher_features: (B, C, H, W) or (N, C) Teacher feature maps
    #         student_features: (B, C, H, W) or (N, C) Student feature maps
    #         teacher_masks: (B, H, W) or (N,) Teacher binary masks (0/1)
    #         student_masks: (B, H, W) or (N,) Student binary masks (0/1)
    #         organ_classes: (B,) or (N,) Organ class IDs
    #         title: Plot title
    #         num_channels: Number of feature channels to visualize
    #     """
    #     # Handle both flattened and unflattened input formats
    #     if teacher_features.dim() == 4:  # (B, C, H, W)
    #         # Select a sample with foreground pixels
    #         fg_indices = torch.where(teacher_masks.sum(dim=(1, 2)) > 0)[0]
    #         if len(fg_indices) == 0:
    #             print("No foreground pixels found for visualization")
    #             return None
            
    #         sample_idx = fg_indices[0]
    #         organ_class = organ_classes[sample_idx].item()
    #         organ_name = self.organ_mappings.get(organ_class, f'Organ_{organ_class}')
            
    #         # Get feature maps for the sample
    #         teacher_feat = teacher_features[sample_idx]  # (C, H, W)
    #         student_feat = student_features[sample_idx]  # (C, H, W)
    #         teacher_mask = teacher_masks[sample_idx]  # (H, W)
    #         student_mask = student_masks[sample_idx]  # (H, W)
    #     else:  # (N, C) - already flattened
    #         # For flattened input, we need to reshape back to spatial dimensions
    #         # This is more complex, so we'll skip feature map visualization for flattened input
    #         print("Feature map visualization not supported for flattened input")
    #         return None
        
    #     # Select channels to visualize
    #     num_channels = min(num_channels, teacher_feat.shape[0])
    #     channels = torch.linspace(0, teacher_feat.shape[0]-1, num_channels).long()
        
    #     # Create visualization
    #     fig, axes = plt.subplots(4, num_channels, figsize=(4*num_channels, 16))
        
    #     for i, channel in enumerate(channels):
    #         # Teacher feature map
    #         teacher_channel = teacher_feat[channel].detach().cpu().numpy()
    #         axes[0, i].imshow(teacher_channel, cmap='viridis')
    #         axes[0, i].set_title(f'Teacher Ch{channel}')
    #         axes[0, i].axis('off')
            
    #         # Student feature map
    #         student_channel = student_feat[channel].detach().cpu().numpy()
    #         axes[1, i].imshow(student_channel, cmap='viridis')
    #         axes[1, i].set_title(f'Student Ch{channel}')
    #         axes[1, i].axis('off')
            
    #         # Teacher mask
    #         teacher_mask_np = teacher_mask.detach().cpu().numpy()
    #         axes[2, i].imshow(teacher_mask_np, cmap='gray')
    #         axes[2, i].set_title('Teacher Mask')
    #         axes[2, i].axis('off')
            
    #         # Student mask
    #         student_mask_np = student_mask.detach().cpu().numpy()
    #         axes[3, i].imshow(student_mask_np, cmap='gray')
    #         axes[3, i].set_title('Student Mask')
    #         axes[3, i].axis('off')
        
    #     plt.suptitle(f"{title} - {organ_name} (Class {organ_class})")
    #     plt.tight_layout()
        
    #     # Save plot
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"feature_maps_{self.dataset_name}_{organ_name}_{timestamp}.png"
    #     plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     return filename
    
    # def visualize_contrastive_analysis(self, teacher_features, student_features,
    #                                   teacher_masks, student_masks, organ_classes,
    #                                   title="Contrastive Learning Analysis"):
    #     """
    #     Comprehensive visualization of contrastive learning analysis.
        
    #     Args:
    #         teacher_features: (B, C, H, W) Teacher feature maps
    #         student_features: (B, C, H, W) Student feature maps
    #         teacher_masks: (B, H, W) Teacher binary masks (0/1)
    #         student_masks: (B, H, W) Student binary masks (0/1)
    #         organ_classes: (B,) Organ class IDs
    #         title: Plot title
    #     """
    #     #print(f"Generating contrastive learning analysis for {self.dataset_name} dataset...")
        
    #     # Generate all visualizations
    #     tsne_file = self.visualize_tsne_features(
    #         teacher_features, student_features, teacher_masks, student_masks, organ_classes,
    #         title=f"{title} - t-SNE Features"
    #     )
        
    #     sim_file = self.visualize_similarity_matrix(
    #         teacher_features, student_features, teacher_masks, student_masks, organ_classes,
    #         title=f"{title} - Similarity Matrix"
    #     )
        
    #     feat_file = self.visualize_feature_maps(
    #         teacher_features, student_features, teacher_masks, student_masks, organ_classes,
    #         title=f"{title} - Feature Maps"
    #     )
        
    #     # print(f"Visualizations saved:")
    #     # print(f"  - t-SNE: {tsne_file}")
    #     # print(f"  - Similarity Matrix: {sim_file}")
    #     # if feat_file:
    #     #     print(f"  - Feature Maps: {feat_file}")
        
    #     return {
    #         'tsne': tsne_file,
    #         'similarity': sim_file,
    #         'feature_maps': feat_file
    #     }

    # def visualize_multi_class_clusters(self, feature_tuples, title="t-SNE Multi-Organ Clusters", max_points=5000, perplexity=30):
    #     """
    #     Visualize clustering of multiple organs in feature space using t-SNE.
    #     Args:
    #         feature_tuples: List of tuples (features, masks, organ_class_id), one per organ.
    #             features: (B, C, H, W) or (N, C)
    #             masks: (B, H, W) or (N,)
    #             organ_class_id: int or str (label for this organ)
    #         title: Plot title
    #         max_points: Maximum number of points to visualize
    #         perplexity: t-SNE perplexity parameter
    #     """
    #     all_features = []
    #     all_organs = []
    #     for features, masks, organ_class_id in feature_tuples:
    #         if features.dim() == 4:
    #             B, C, H, W = features.shape
    #             features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    #             masks_flat = masks.reshape(-1)
    #         else:
    #             features_flat = features
    #             masks_flat = masks
    #         fg_features = features_flat[masks_flat == 1]
    #         fg_labels = torch.full((fg_features.shape[0],), organ_class_id, dtype=torch.long, device=features.device)
    #         all_features.append(fg_features)
    #         all_organs.append(fg_labels)
    #     if not all_features:
    #         print("No features to visualize.")
    #         return None
    #     all_features = torch.cat(all_features, dim=0)
    #     all_organs = torch.cat(all_organs, dim=0)
    #     # Downsample if too many points
    #     if len(all_features) > max_points:
    #         indices = torch.randperm(len(all_features))[:max_points]
    #         all_features = all_features[indices]
    #         all_organs = all_organs[indices]
    #     # t-SNE
    #     features_np = all_features.detach().cpu().numpy()
    #     organs_np = all_organs.detach().cpu().numpy()
    #     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    #     features_2d = tsne.fit_transform(features_np)
    #     # Plot
    #     plt.figure(figsize=(12, 8))
    #     for organ_id in np.unique(organs_np):
    #         organ_name = self.organ_mappings.get(organ_id, f'Organ_{organ_id}')
    #         color = self.colors.get(organ_name, self.colors['unknown'])
    #         mask = organs_np == organ_id
    #         plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=color, label=organ_name, alpha=0.7, s=20)
    #     plt.title(title)
    #     plt.xlabel("t-SNE 1")
    #     plt.ylabel("t-SNE 2")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"tsne_multi_organ_{self.dataset_name}_{timestamp}.png"
    #     plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
    #     plt.show()
    #     return filename

def create_organ_visualizer(dataset_name='SABS', save_dir='visualizations'):
    """
    Factory function to create an organ visualizer.
    
    Args:
        dataset_name: Name of the dataset
        save_dir: Directory to save visualizations
    
    Returns:
        OrganContrastiveVisualizer instance
    """
    return OrganContrastiveVisualizer(dataset_name=dataset_name, save_dir=save_dir) 