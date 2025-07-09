import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseContrastiveLoss(nn.Module):
    """
    Pixel-wise contrastive loss for segmentation feature maps.
    Supports teacher-student setups where positive pairs are foreground pixels
    and negatives are background pixels (or vice versa).
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_teacher, feat_student, mask):
        """
        Args:
            feat_teacher: (B, C, H, W) feature map from teacher encoder
            feat_student: (B, C, H, W) feature map from student encoder
            mask: (B, 1, H, W) binary mask, 1=foreground, 0=background
        Returns:
            loss: scalar contrastive loss
        """
        B, C, H, W = feat_teacher[0][0].shape
        #print(feat_teacher.shape)-torch.Size([1, 1, 1, 256, 32, 32])
        #print(feat_teacher[0][0].shape)-torch.Size([1, 256, 32, 32])
        # Flatten spatial dims
        
        feat_teacher = feat_teacher[0][0].permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        feat_student = feat_student[0][0].permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        mask = mask.view(-1)  # (B*H*W,)

        # Normalize features
        feat_teacher = F.normalize(feat_teacher, dim=1)
        feat_student = F.normalize(feat_student, dim=1)

        # Positive indices (foreground)
        pos_idx = mask == 1
        neg_idx = mask == 0

        if pos_idx.sum() == 0 or neg_idx.sum() == 0:
            # Avoid division by zero if mask is empty
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

        # Positive pairs: teacher_fg <-> student_fg
        pos_teacher = feat_teacher[pos_idx]
        pos_student = feat_student[pos_idx]
        # Negative pairs: teacher_fg <-> student_bg
        neg_student = feat_student[neg_idx]

        # Compute logits
        logits_pos = (pos_teacher * pos_student).sum(dim=1) / self.temperature  # (N_fg,)
        logits_neg = torch.mm(pos_teacher, neg_student.t()) / self.temperature  # (N_fg, N_bg)

        # For each positive, logits = [positive, all negatives]
        logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)  # (N_fg, 1+N_bg)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positives are index 0

        loss = F.cross_entropy(logits, labels)
        return loss 