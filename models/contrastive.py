import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedPixelWiseContrastiveLoss(nn.Module):
    """
    Supervised pixel-wise InfoNCE contrastive loss for online-target learning.
    Uses binary masks (0/1) with organ class information to create positive pairs 
    (same organ class) and negative pairs (foreground vs background).
    
    Args:
        temperature: Temperature parameter for contrastive learning
        num_negatives: Number of negative samples per positive pair
    """
    def __init__(self, temperature=0.1, num_negatives=1000):
        super(SupervisedPixelWiseContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(self, feat_online, feat_target, mask_online, mask_target, organ_class_online, organ_class_target):
        """
        Args:
            feat_online: (B, C, H, W) feature map from online encoder
            feat_target: (B, C, H, W) feature map from target encoder  
            mask_online: (B, H, W) binary mask for online (0=bg, 1=fg)
            mask_target: (B, H, W) binary mask for target (0=bg, 1=fg)
            organ_class_online: (B,) organ class ID for online mask
            organ_class_target: (B,) organ class ID for target mask
        Returns:
            loss: scalar InfoNCE contrastive loss
        """
        B, C, H, W = feat_online.shape
        # Normalize features
        feat_online = F.normalize(feat_online, dim=1)
        feat_target = F.normalize(feat_target, dim=1)
        
        # Reshape features to (B*H*W, C)
        feat_online_flat = feat_online.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        feat_target_flat = feat_target.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Reshape masks to (B*H*W,)
        mask_online_flat = mask_online.reshape(-1)  # (B*H*W,)
        mask_target_flat = mask_target.reshape(-1)  # (B*H*W,)
        
        # Find foreground pixels (mask == 1)
        fg_online = mask_online_flat == 1
        fg_target = mask_target_flat == 1
        
        # Find background pixels (mask == 0)
        bg_online = mask_online_flat == 0
        bg_target = mask_target_flat == 0
        
        # Get foreground features
        fg_feat_online = feat_online_flat[fg_online]  # (N_fg_online, C)
        fg_feat_target = feat_target_flat[fg_target]  # (N_fg_target, C)
        
        # Get background features
        bg_feat_online = feat_online_flat[bg_online]  # (N_bg_online, C)
        bg_feat_target = feat_target_flat[bg_target]  # (N_bg_target, C)
        
        
        # Check if both masks are for the same organ class (positive pairs)
        same_organ = (organ_class_online == organ_class_target).all()
        
        if not same_organ:
            # Different organs - no positive pairs, return small loss to encourage separation
            return torch.tensor(0.1, device=feat_online.device, requires_grad=True)
        
        # Sample negative features (background features)
        num_neg = min(self.num_negatives, bg_feat_online.shape[0], bg_feat_target.shape[0])
        if num_neg > 0:
            neg_feat_online = bg_feat_online[:num_neg]  # (num_neg, C)
            neg_feat_target = bg_feat_target[:num_neg]  # (num_neg, C)
        else:
            # If no background pixels, use random features as negatives
            return torch.tensor(0.1, device=feat_online.device, requires_grad=True)
            # neg_feat_teacher = torch.randn_like(fg_feat_teacher[:min(100, fg_feat_teacher.shape[0])])
            # neg_feat_student = torch.randn_like(fg_feat_student[:min(100, fg_feat_student.shape[0])])

        # InfoNCE Loss Implementation
        # Positive pairs: online foreground <-> target foreground (same organ)
        # Negative pairs: online foreground <-> target background features
        #                 online background <-> target background features (bg-bg negative pairing)

        # Compute positive similarities with numerical stability
        pos_sim = torch.mm(fg_feat_online, fg_feat_target.t()) / max(self.temperature, 1e-7)
        pos_sim = torch.clamp(pos_sim, min=-50, max=50)  # Prevent overflow in cross_entropy

        # Compute negative similarities (online foreground vs target background)
        neg_sim_online = torch.mm(fg_feat_online, neg_feat_target.t()) / max(self.temperature, 1e-7)
        neg_sim_online = torch.clamp(neg_sim_online, min=-50, max=50)  # Prevent overflow
        # Compute negative similarities (online background vs target background)
        neg_sim_bg_online = torch.mm(neg_feat_online, neg_feat_target.t()) / max(self.temperature, 1e-7)
        neg_sim_bg_online = torch.clamp(neg_sim_bg_online, min=-50, max=50)  # Prevent overflow

        # For InfoNCE, create similarity matrix where each row represents
        # one positive pair and multiple negative pairs
        logits_online = torch.cat([pos_sim, neg_sim_online], dim=1)  # (N_fg_online, N_fg_target + num_neg)
        # Optionally, you can also include bg-bg negatives in the loss if you want to penalize background similarity
        # For now, we keep them separate for clarity

        # Create labels for InfoNCE (the positive pair index for each row)
        labels_online = torch.arange(min(fg_feat_online.shape[0], fg_feat_target.shape[0]), 
                                    device=feat_online.device)

        # If we have more online foreground pixels than target, truncate
        if fg_feat_online.shape[0] > fg_feat_target.shape[0]:
            logits_online = logits_online[:fg_feat_target.shape[0]]

        # Compute InfoNCE loss for online->target direction
        loss_online = F.cross_entropy(logits_online, labels_online)
        
        # Check for NaN in loss_online
        if torch.isnan(loss_online) or torch.isinf(loss_online):
            loss_online = torch.tensor(0.0, device=feat_online.device, requires_grad=True)

        # Compute InfoNCE loss for target->online direction (symmetric)
        neg_sim_target = torch.mm(fg_feat_target, neg_feat_online.t()) / max(self.temperature, 1e-7)
        neg_sim_target = torch.clamp(neg_sim_target, min=-50, max=50)  # Prevent overflow
        logits_target = torch.cat([pos_sim.t(), neg_sim_target], dim=1)  # (N_fg_target, N_fg_online + num_neg)
        labels_target = torch.arange(min(fg_feat_target.shape[0], fg_feat_online.shape[0]), 
                                    device=feat_online.device)

        if fg_feat_target.shape[0] > fg_feat_online.shape[0]:
            logits_target = logits_target[:fg_feat_online.shape[0]]

        loss_target = F.cross_entropy(logits_target, labels_target)
        
        # Check for NaN in loss_target
        if torch.isnan(loss_target) or torch.isinf(loss_target):
            loss_target = torch.tensor(0.0, device=feat_online.device, requires_grad=True)

        # Background-background negative pairing loss (optional, can be weighted)
        # Encourage background features to be consistent between online and target
        # Here, we use InfoNCE-style loss for bg-bg as well
        if num_neg > 1:
            # For bg-bg, treat each online bg as anchor, target bg as positives (diagonal), rest as negatives
            bg_labels = torch.arange(num_neg, device=feat_online.device)
            loss_bg = F.cross_entropy(neg_sim_bg_online, bg_labels)
            
            # Check for NaN in loss_bg
            if torch.isnan(loss_bg) or torch.isinf(loss_bg):
                loss_bg = torch.tensor(0.0, device=feat_online.device, requires_grad=True)
        else:
            loss_bg = torch.tensor(0.0, device=feat_online.device, requires_grad=True)

        # Return average of all three losses (can be weighted if desired)
        return (loss_online + loss_target + loss_bg) / 3

class ContrastiveLoss(nn.Module):
    """
    Legacy contrastive loss - kept for backward compatibility
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.supervised_loss = SupervisedPixelWiseContrastiveLoss(temperature=temperature)

    def forward(self, feature_online, feature_target, mask_online, mask_target, organ_class_online=None, organ_class_target=None):
        """
        Wrapper for supervised contrastive loss
        """
        # If organ class information is not provided, assume same organ
        if organ_class_online is None:
            organ_class_online = torch.ones(feature_online.shape[0], device=feature_online.device)
        if organ_class_target is None:
            organ_class_target = torch.ones(feature_target.shape[0], device=feature_target.device)
            
        return self.supervised_loss(feature_online, feature_target, mask_online, mask_target, 
                                  organ_class_online, organ_class_target)