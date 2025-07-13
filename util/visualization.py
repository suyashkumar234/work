import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def visualize_tsne_features(feat_teacher, feat_student, mask_teacher, mask_student, save_path, n_samples=1000, perplexity=30):
    """
    t-SNE visualization of teacher-student feature embeddings with organ-level clustering using two masks
    Args:
        feat_teacher: (B, C, H, W) teacher features
        feat_student: (B, C, H, W) student features  
        mask_teacher: (B, 1, H, W) teacher binary mask, 1=foreground, 0=background
        mask_student: (B, 1, H, W) student binary mask, 1=foreground, 0=background
        save_path: path to save the visualization
        n_samples: number of pixels to sample for t-SNE (t-SNE is slow)
        perplexity: t-SNE perplexity parameter
    """
    # Flatten features
    B, C, H, W = feat_teacher.shape
    feat_teacher_flat = feat_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    feat_student_flat = feat_student.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    
    # Flatten masks
    mask_teacher_flat = mask_teacher.view(-1)
    mask_student_flat = mask_student.view(-1)
    
    # Intersection of foregrounds (where both masks == 1)
    fg_indices = torch.where((mask_teacher_flat == 1) & (mask_student_flat == 1))[0]
    if len(fg_indices) < 10:  # Not enough foreground pixels
        print("Not enough intersected foreground pixels for organ clustering")
        return
    
    # Sample foreground pixels for organ clustering
    n_fg_samples = min(n_samples // 2, len(fg_indices))
    fg_sample_indices = fg_indices[torch.randperm(len(fg_indices))[:n_fg_samples]]
    
    # Get foreground features
    feat_teacher_fg = feat_teacher_flat[fg_sample_indices]
    feat_student_fg = feat_student_flat[fg_sample_indices]
    
    # Normalize features
    feat_teacher_fg = F.normalize(feat_teacher_fg, dim=1)
    feat_student_fg = F.normalize(feat_student_fg, dim=1)
    
    # Create spatial coordinates for organ clustering
    y_coords = (fg_sample_indices // W).float()
    x_coords = (fg_sample_indices % W).float()
    coords = torch.stack([y_coords, x_coords], dim=1)
    
    # Compute spatial distances and feature similarities
    spatial_dist = torch.cdist(coords, coords)
    spatial_sim = torch.exp(-spatial_dist / 5.0)
    feature_sim_teacher = torch.mm(feat_teacher_fg, feat_teacher_fg.t())
    feature_sim_student = torch.mm(feat_student_fg, feat_student_fg.t())
    feature_sim = (feature_sim_teacher + feature_sim_student) / 2
    
    # Combined similarity for organ identification
    organ_sim = 0.3 * spatial_sim + 0.7 * feature_sim
    
    # Identify organ clusters using similarity threshold
    organ_clusters = organ_sim > 0.7
    organ_labels = torch.zeros(n_fg_samples, dtype=torch.long)
    
    # Assign organ cluster IDs
    current_cluster = 0
    for i in range(n_fg_samples):
        if organ_labels[i] == 0:  # Not assigned yet
            current_cluster += 1
            same_organ = organ_clusters[i]
            organ_labels[same_organ] = current_cluster
    
    # Sample background pixels (where either mask is 0)
    bg_indices = torch.where((mask_teacher_flat == 0) | (mask_student_flat == 0))[0]
    n_bg_samples = min(n_samples - n_fg_samples, len(bg_indices))
    bg_sample_indices = bg_indices[torch.randperm(len(bg_indices))[:n_bg_samples]]
    
    # Combine foreground and background samples
    all_indices = torch.cat([fg_sample_indices, bg_sample_indices])
    feat_teacher_sampled = feat_teacher_flat[all_indices].detach().cpu()
    feat_student_sampled = feat_student_flat[all_indices].detach().cpu()
    
    # Prepare data for t-SNE
    features = torch.cat([feat_teacher_sampled, feat_student_sampled], dim=0)  # (2*N, C)
    
    # Create labels: organ clusters + background
    labels = []
    # Teacher features
    for i in range(n_fg_samples):
        labels.append(organ_labels[i])  # Organ cluster ID
    for i in range(n_bg_samples):
        labels.append(0)  # Background
    # Student features  
    for i in range(n_fg_samples):
        labels.append(organ_labels[i])  # Same organ cluster ID
    for i in range(n_bg_samples):
        labels.append(0)  # Background
    
    # Convert to numpy
    features_np = features.numpy()
    labels_np = np.array(labels)
    
    # Apply PCA first to reduce dimensionality (t-SNE works better with lower dim)
    if features_np.shape[1] > 50:
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_np)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        features_pca = features_np
    
    # Apply t-SNE
    print(f"Running t-SNE on {features_pca.shape[0]} samples with {features_pca.shape[1]} features...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features_pca)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Get unique organ labels
    unique_labels = np.unique(labels_np)
    n_organs = len(unique_labels) - 1  # Exclude background (label 0)
    
    # Define colors for organs (excluding background)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_organs, 10)))
    
    # Plot background first (label 0)
    bg_mask = labels_np == 0
    plt.scatter(features_2d[bg_mask, 0], features_2d[bg_mask, 1], 
               c='gray', label='Background', alpha=0.5, s=20, marker='.')
    
    # Plot each organ cluster
    for i, label in enumerate(unique_labels[1:], 1):  # Skip background (label 0)
        mask_i = labels_np == label
        plt.scatter(features_2d[mask_i, 0], features_2d[mask_i, 1], 
                   c=colors[i-1], label=f'Organ {i}', alpha=0.7, 
                   s=30, edgecolors='white', linewidth=0.5)
    
    plt.title(f'Organ-Level t-SNE Visualization\n(n_samples={n_samples}, n_organs={n_organs}, perplexity={perplexity})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Total features: {features_np.shape[1]}\n"
    stats_text += f"Background pixels: {(labels_np == 0).sum()}\n"
    for i, label in enumerate(unique_labels[1:], 1):
        stats_text += f"Organ {i}: {(labels_np == label).sum()}\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to: {save_path}")
    
    return features_2d, labels_np

def visualize_tsne_with_labels(feat, pseudo_mask, save_path, n_samples=1000, perplexity=30):
    # feat: (B, C, H, W)
    # pseudo_mask: (B, H, W) or (B, 1, H, W), integer labels
    B, C, H, W = feat.shape
    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
    pseudo_mask_flat = pseudo_mask.view(-1)
    # Only use pixels with a region label (pseudo_mask > 0)
    region_indices = torch.where(pseudo_mask_flat > 0)[0]
    n_samples = min(n_samples, len(region_indices))
    sample_indices = region_indices[torch.randperm(len(region_indices))[:n_samples]]
    features = feat_flat[sample_indices].detach().cpu().numpy()
    labels = pseudo_mask_flat[sample_indices].detach().cpu().numpy()
    # t-SNE
    from sklearn.manifold import TSNE
    features_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(features)
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Region {label}', s=10)
    plt.legend()
    plt.title('t-SNE of Features Colored by Pseudo-Mask Regions')
    plt.savefig(save_path)
    plt.close()

# def visualize_feature_similarity(feat_teacher, feat_student, mask, save_path):
#     """
#     Visualize similarity matrix between teacher and student features
#     """
#     # Flatten features
#     B, C, H, W = feat_teacher.shape
#     feat_teacher_flat = feat_teacher.permute(0, 2, 3, 1).reshape(-1, C)
#     feat_student_flat = feat_student.permute(0, 2, 3, 1).reshape(-1, C)
    
#     # Handle different mask shapes
#     if mask.dim() == 4:  # (B, 1, H, W)
#         mask_flat = mask.view(-1)
#     elif mask.dim() == 3:  # (B, H, W)
#         mask_flat = mask.view(-1)
#     elif mask.dim() == 2:  # (H, W)
#         mask_flat = mask.view(-1)
#     else:
#         raise ValueError(f"Unexpected mask shape: {mask.shape}")
    
#     # Normalize features
#     feat_teacher_norm = F.normalize(feat_teacher_flat, dim=1)
#     feat_student_norm = F.normalize(feat_student_flat, dim=1)
    
#     # Compute similarity matrix
#     similarity = torch.mm(feat_teacher_norm, feat_student_norm.t())
    
#     # Plot
#     plt.figure(figsize=(10, 8))
#     im = plt.imshow(similarity.detach().cpu().numpy(), cmap='viridis', aspect='auto')
#     plt.colorbar(im, label='Cosine Similarity')
#     plt.title('Teacher-Student Feature Similarity Matrix', fontsize=14, fontweight='bold')
#     plt.xlabel('Student Features', fontsize=12)
#     plt.ylabel('Teacher Features', fontsize=12)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Similarity matrix saved to: {save_path}")

# def visualize_feature_maps(feat_teacher, feat_student, mask, save_path, num_channels=8):
#     """
#     Visualize feature maps from teacher and student encoders
#     """
#     # Select channels to visualize
#     num_channels = min(num_channels, feat_teacher.shape[1])
    
#     fig, axes = plt.subplots(4, num_channels, figsize=(20, 8))
    
#     # Teacher features
#     for i in range(num_channels):
#         feat = feat_teacher[0, i].detach().cpu().numpy()
#         im = axes[0, i].imshow(feat, cmap='viridis')
#         axes[0, i].set_title(f'Teacher Ch{i}', fontsize=10)
#         axes[0, i].axis('off')
    
#     # Student features
#     for i in range(num_channels):
#         feat = feat_student[0, i].detach().cpu().numpy()
#         im = axes[1, i].imshow(feat, cmap='viridis')
#         axes[1, i].set_title(f'Student Ch{i}', fontsize=10)
#         axes[1, i].axis('off')
    
#     # Difference (Teacher - Student)
#     for i in range(num_channels):
#         diff = feat_teacher[0, i].detach().cpu().numpy() - feat_student[0, i].detach().cpu().numpy()
#         max_abs = max(abs(diff.min()), abs(diff.max()))
#         im = axes[2, i].imshow(diff, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
#         axes[2, i].set_title(f'Diff Ch{i}', fontsize=10)
#         axes[2, i].axis('off')
    
#     # Mask overlay
#     mask_vis = mask[0, 0].detach().cpu().numpy()
#     axes[3, 0].imshow(mask_vis, cmap='gray')
#     axes[3, 0].set_title('Mask', fontsize=10)
#     axes[3, 0].axis('off')
    
#     # Hide unused subplots
#     for i in range(1, num_channels):
#         axes[3, i].axis('off')
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Feature maps saved to: {save_path}") 