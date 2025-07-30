import torch
import torch.nn as nn
import torch.nn.functional as F

    class SupervisedPixelWiseContrastiveLoss(nn.Module):
      """
      Self-Supervised pixel-wise InfoNCE contrastive loss for
   online-target learning.
      Uses binary masks (0/1) WITHOUT organ class information
   to create positive pairs 
      (same image regions) and negative pairs (foreground vs 
  background).
      
      Args:
          temperature: Temperature parameter for contrastive 
  learning
          num_negatives: Number of negative samples per 
  positive pair
      """
        def __init__(self, temperature=0.1, 
  num_negatives=1000):
            super(SupervisedPixelWiseContrastiveLoss,
  self).__init__()
          self.temperature = temperature
          self.num_negatives = num_negatives

      def forward(self, feat_online, feat_target, 
  mask_online, mask_target):
          """
          Args:
              feat_online: (B, C, H, W) feature map from 
  online encoder
              feat_target: (B, C, H, W) feature map from 
  target encoder  
              mask_online: (B, H, W) binary mask for online 
  (0=bg, 1=fg)
              mask_target: (B, H, W) binary mask for target 
  (0=bg, 1=fg)
          Returns:
              loss: scalar InfoNCE contrastive loss
          """
          B, C, H, W = feat_online.shape
          # Normalize features
          feat_online = F.normalize(feat_online, dim=1)
          feat_target = F.normalize(feat_target, dim=1)

          # Reshape features to (B*H*W, C)
          feat_online_flat = feat_online.permute(0, 2, 3,
  1).reshape(-1, C)  # (B*H*W, C)
          feat_target_flat = feat_target.permute(0, 2, 3,
  1).reshape(-1, C)  # (B*H*W, C)

          # Reshape masks to (B*H*W,) with relaxed threshold
          mask_online_flat = mask_online.reshape(-1)  # 
  (B*H*W,)
          mask_target_flat = mask_target.reshape(-1)  # 
  (B*H*W,)

          # Find foreground pixels with relaxed threshold 
  (0.3 instead of 1.0)
          fg_online = mask_online_flat > 0.3
          fg_target = mask_target_flat > 0.3

          # Find background pixels
          bg_online = mask_online_flat <= 0.3
          bg_target = mask_target_flat <= 0.3

          # Get foreground features
          fg_feat_online = feat_online_flat[fg_online]  # 
  (N_fg_online, C)
          fg_feat_target = feat_target_flat[fg_target]  # 
  (N_fg_target, C)

          # Get background features
          bg_feat_online = feat_online_flat[bg_online]  # 
  (N_bg_online, C)
          bg_feat_target = feat_target_flat[bg_target]  # 
  (N_bg_target, C)

          # Check if we have enough foreground pixels
          if fg_feat_online.shape[0] < 10 or
  fg_feat_target.shape[0] < 10:
              # If too few foreground pixels, return small 
  loss
              return torch.tensor(0.1,
  device=feat_online.device, requires_grad=True)

          # Sample negative features (background features)
          num_neg = min(self.num_negatives,
  bg_feat_online.shape[0], bg_feat_target.shape[0])
          if num_neg > 0:
              neg_feat_online = bg_feat_online[:num_neg]  # 
  (num_neg, C)
              neg_feat_target = bg_feat_target[:num_neg]  # 
  (num_neg, C)
          else:
              # If no background pixels, use random features 
  as negatives
              return torch.tensor(0.1,
  device=feat_online.device, requires_grad=True)

          # Compute positive similarities with numerical 
  stability
          pos_sim = torch.mm(fg_feat_online,
  fg_feat_target.t()) / max(self.temperature, 1e-7)
          pos_sim = torch.clamp(pos_sim, min=-50, max=50)  # 
  Prevent overflow

          # Compute negative similarities (online foreground 
  vs target background)
          neg_sim_online = torch.mm(fg_feat_online,
  neg_feat_target.t()) / max(self.temperature, 1e-7)
          neg_sim_online = torch.clamp(neg_sim_online,
  min=-50, max=50)  # Prevent overflow

          # Compute negative similarities (online background 
  vs target background)
          neg_sim_bg_online = torch.mm(neg_feat_online,
  neg_feat_target.t()) / max(self.temperature, 1e-7)
          neg_sim_bg_online = torch.clamp(neg_sim_bg_online,
  min=-50, max=50)  # Prevent overflow

          # Compute row-wise average of positive similarities
          pos_sim_avg = torch.mean(pos_sim, dim=1,
  keepdim=True)  # (N_fg_online, 1)

          # For InfoNCE, create similarity matrix
          logits_online = torch.cat([pos_sim_avg,
  neg_sim_online], dim=1)  # (N_fg_online, 1 + num_neg)

          # Create labels for InfoNCE (positive pair is at 
  index 0)
          labels_online =
  torch.zeros(fg_feat_online.shape[0],
  device=feat_online.device, dtype=torch.long)

          # Handle size mismatch between online and target 
  features
          min_size = min(fg_feat_online.shape[0],
  fg_feat_target.shape[0])
          if min_size == 0:
              return torch.tensor(0.1,
  device=feat_online.device, requires_grad=True)

          logits_online = logits_online[:min_size]
          labels_online = labels_online[:min_size]

          # Compute InfoNCE loss for online->target direction
          loss_online = F.cross_entropy(logits_online,
  labels_online)

          # Check for NaN in loss_online
          if torch.isnan(loss_online) or
  torch.isinf(loss_online):
              loss_online = torch.tensor(0.0,
  device=feat_online.device, requires_grad=True)

          # Compute InfoNCE loss for target->online direction
   (symmetric)
          neg_sim_target = torch.mm(fg_feat_target,
  neg_feat_online.t()) / max(self.temperature, 1e-7)
          neg_sim_target = torch.clamp(neg_sim_target,
  min=-50, max=50)  # Prevent overflow

          # Compute row-wise average of transposed pos_sim 
  for target direction
          pos_sim_avg_target = torch.mean(pos_sim.t(), dim=1,
   keepdim=True)  # (N_fg_target, 1)

          logits_target = torch.cat([pos_sim_avg_target,
  neg_sim_target], dim=1)  # (N_fg_target, 1 + num_neg)
          labels_target =
  torch.zeros(fg_feat_target.shape[0],
  device=feat_online.device, dtype=torch.long)

          # Use the same min_size for consistency
          logits_target = logits_target[:min_size]
          labels_target = labels_target[:min_size]

          loss_target = F.cross_entropy(logits_target,
  labels_target)

          # Check for NaN in loss_target
          if torch.isnan(loss_target) or
  torch.isinf(loss_target):
              loss_target = torch.tensor(0.0,
  device=feat_online.device, requires_grad=True)

          # Background-background negative pairing loss
          if num_neg > 1:
              bg_labels = torch.arange(num_neg,
  device=feat_online.device)
              loss_bg = F.cross_entropy(neg_sim_bg_online,
  bg_labels)

              # Check for NaN in loss_bg
              if torch.isnan(loss_bg) or
  torch.isinf(loss_bg):
                  loss_bg = torch.tensor(0.0,
  device=feat_online.device, requires_grad=True)
          else:
              loss_bg = torch.tensor(0.0,
  device=feat_online.device, requires_grad=True)

          # Return average of all three losses
          return (loss_online + loss_target + loss_bg) / 3

  class ContrastiveLoss(nn.Module):
      """
      Main contrastive loss - now self-supervised without 
  organ classes
      """
      def __init__(self, temperature=0.5):
          super(ContrastiveLoss, self).__init__()
          self.temperature = temperature
          self.supervised_loss =
  SupervisedPixelWiseContrastiveLoss(temperature=temperature)

      def forward(self, feature_online, feature_target, 
  mask_online, mask_target):
          """
          Self-supervised contrastive loss without organ 
  classes
          """
          return self.supervised_loss(feature_online,
  feature_target, mask_online, mask_target)