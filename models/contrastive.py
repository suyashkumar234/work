import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedPixelWiseContrastiveLoss(nn.Module):
    """
    Supervised pixel-wise InfoNCE contrastive loss for teacher-student learning.
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

    def forward(self, feat_teacher, feat_student, mask_teacher, mask_student, organ_class_teacher, organ_class_student):
        """
        Args:
            feat_teacher: (B, C, H, W) feature map from teacher encoder
            feat_student: (B, C, H, W) feature map from student encoder  
            mask_teacher: (B, H, W) binary mask for teacher (0=bg, 1=fg)
            mask_student: (B, H, W) binary mask for student (0=bg, 1=fg)
            organ_class_teacher: (B,) organ class ID for teacher mask
            organ_class_student: (B,) organ class ID for student mask
                                                                                                                                                                                                                                     vgcfxw
            loss: scalar InfoNCE contrastive loss
        """
        B, C, H, W = feat_teacher.shape
        #print(organ_class_teacher[0], organ_class_student[0])
        # Normalize features
        feat_teacher = F.normalize(feat_teacher, dim=1)
        feat_student = F.normalize(feat_student, dim=1)
        #print(organ_class_teacher, organ_class_student)
        # Reshape features to (B*H*W, C)
        feat_teacher_flat = feat_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        feat_student_flat = feat_student.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Reshape masks to (B*H*W,)
        mask_teacher_flat = mask_teacher.reshape(-1)  # (B*H*W,)
        mask_student_flat = mask_student.reshape(-1)  # (B*H*W,)
        
        # Find foreground pixels (mask == 1)
        fg_teacher = mask_teacher_flat == 1
        fg_student = mask_student_flat == 1
        
        # Find background pixels (mask == 0)
        bg_teacher = mask_teacher_flat == 0
        bg_student = mask_student_flat == 0
        
        # Get foreground features
        fg_feat_teacher = feat_teacher_flat[fg_teacher]  # (N_fg_teacher, C)
        fg_feat_student = feat_student_flat[fg_student]  # (N_fg_student, C)
        
        # Get background features
        bg_feat_teacher = feat_teacher_flat[bg_teacher]  # (N_bg_teacher, C)
        bg_feat_student = feat_student_flat[bg_student]  # (N_bg_student, C)
        
        
        # Check if both masks are for the same organ class (positive pairs)
        same_organ = (organ_class_teacher == organ_class_student).all()
        
        if not same_organ:
            # Different organs - no positive pairs, return small loss to encourage separation
            return torch.tensor(0.1, device=feat_teacher.device, requires_grad=True)
        
        # Sample negative features (background features)
        num_neg = min(self.num_negatives, bg_feat_teacher.shape[0], bg_feat_student.shape[0])
        if num_neg > 0:
            neg_feat_teacher = bg_feat_teacher[:num_neg]  # (num_neg, C)
            neg_feat_student = bg_feat_student[:num_neg]  # (num_neg, C)
        else:
            # If no background pixels, use random features as negatives
            return torch.tensor(0.1, device=feat_teacher.device, requires_grad=True)
            # neg_feat_teacher = torch.randn_like(fg_feat_teacher[:min(100, fg_feat_teacher.shape[0])])
            # neg_feat_student = torch.randn_like(fg_feat_student[:min(100, fg_feat_student.shape[0])])

        # InfoNCE Loss Implementation
        # Positive pairs: teacher foreground <-> student foreground (same organ)
        # Negative pairs: teacher foreground <-> student background features
        #                 teacher background <-> student background features (bg-bg negative pairing)

        # Compute positive similarities with numerical stability
        pos_sim = torch.mm(fg_feat_teacher, fg_feat_student.t()) / max(self.temperature, 1e-7)
        pos_sim = torch.clamp(pos_sim, min=-50, max=50)  # Prevent overflow in cross_entropy

        # Compute negative similarities (teacher foreground vs student background)
        neg_sim_teacher = torch.mm(fg_feat_teacher, neg_feat_student.t()) / max(self.temperature, 1e-7)
        neg_sim_teacher = torch.clamp(neg_sim_teacher, min=-50, max=50)  # Prevent overflow
        # Compute negative similarities (teacher background vs student background)
        neg_sim_bg_teacher = torch.mm(neg_feat_teacher, neg_feat_student.t()) / max(self.temperature, 1e-7)
        neg_sim_bg_teacher = torch.clamp(neg_sim_bg_teacher, min=-50, max=50)  # Prevent overflow

        # For InfoNCE, create similarity matrix where each row represents
        # one positive pair and multiple negative pairs
        logits_teacher = torch.cat([pos_sim, neg_sim_teacher], dim=1)  # (N_fg_teacher, N_fg_student + num_neg)
        # Optionally, you can also include bg-bg negatives in the loss if you want to penalize background similarity
        # For now, we keep them separate for clarity

        # Create labels for InfoNCE (the positive pair index for each row)
        labels_teacher = torch.arange(min(fg_feat_teacher.shape[0], fg_feat_student.shape[0]), 
                                    device=feat_teacher.device)

        # If we have more teacher foreground pixels than student, truncate
        if fg_feat_teacher.shape[0] > fg_feat_student.shape[0]:
            logits_teacher = logits_teacher[:fg_feat_student.shape[0]]

        # Compute InfoNCE loss for teacher->student direction
        loss_teacher = F.cross_entropy(logits_teacher, labels_teacher)
        
        # Check for NaN in loss_teacher
        if torch.isnan(loss_teacher) or torch.isinf(loss_teacher):
            loss_teacher = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

        # Compute InfoNCE loss for student->teacher direction (symmetric)
        neg_sim_student = torch.mm(fg_feat_student, neg_feat_teacher.t()) / max(self.temperature, 1e-7)
        neg_sim_student = torch.clamp(neg_sim_student, min=-50, max=50)  # Prevent overflow
        logits_student = torch.cat([pos_sim.t(), neg_sim_student], dim=1)  # (N_fg_student, N_fg_teacher + num_neg)
        labels_student = torch.arange(min(fg_feat_student.shape[0], fg_feat_teacher.shape[0]), 
                                    device=feat_teacher.device)

        if fg_feat_student.shape[0] > fg_feat_teacher.shape[0]:
            logits_student = logits_student[:fg_feat_teacher.shape[0]]

        loss_student = F.cross_entropy(logits_student, labels_student)
        
        # Check for NaN in loss_student
        if torch.isnan(loss_student) or torch.isinf(loss_student):
            loss_student = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

        # Background-background negative pairing loss (optional, can be weighted)
        # Encourage background features to be dissimilar between teacher and student
        # Here, we use InfoNCE-style loss for bg-bg as well
        if num_neg > 1:
            # For bg-bg, treat each teacher bg as anchor, student bg as positives (diagonal), rest as negatives
            bg_labels = torch.arange(num_neg, device=feat_teacher.device)
            loss_bg = F.cross_entropy(neg_sim_bg_teacher, bg_labels)
            
            # Check for NaN in loss_bg
            if torch.isnan(loss_bg) or torch.isinf(loss_bg):
                loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        else:
            loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

        # Return average of all three losses (can be weighted if desired)
        # print('')
        # print(loss_teacher, loss_student, loss_bg)
        # print('')
        return (loss_teacher + loss_student + loss_bg) / 3

class ContrastiveLoss(nn.Module):
    """
    Legacy contrastive loss - kept for backward compatibility
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.supervised_loss = SupervisedPixelWiseContrastiveLoss(temperature=temperature)

    def forward(self, feature_teacher, feature_student, mask_teacher, mask_student, organ_class_teacher=None, organ_class_student=None):
        """
        Wrapper for supervised contrastive loss
        """
        # If organ class information is not provided, assume same organ
        #print(organ_class_teacher, organ_class_student)
        if organ_class_teacher is None:
            organ_class_teacher = torch.ones(feature_teacher.shape[0], device=feature_teacher.device)
        if organ_class_student is None:
            organ_class_student = torch.ones(feature_student.shape[0], device=feature_student.device)
            
        return self.supervised_loss(feature_teacher, feature_student, mask_teacher, mask_student, 
                                  organ_class_teacher, organ_class_student)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SupervisedPixelWiseContrastiveLoss(nn.Module):
#     """
#     Supervised pixel-wise InfoNCE contrastive loss for teacher-student learning.
#     Uses binary masks (0/1) with organ class information to create positive pairs 
#     (same organ class) and negative pairs (foreground vs background).
    
#     Args:
#         temperature: Temperature parameter for contrastive learning
#         num_negatives: Number of negative samples per positive pair
#     """
#     def __init__(self, temperature=0.1, num_negatives=1000):
#         super(SupervisedPixelWiseContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.num_negatives = num_negatives

#     def forward(self, feat_teacher, feat_student, mask_teacher, mask_student, organ_class_teacher, organ_class_student):
#         """
#         Args:
#             feat_teacher: (B, C, H, W) feature map from teacher encoder
#             feat_student: (B, C, H, W) feature map from student encoder  
#             mask_teacher: (B, H, W) binary mask for teacher (0=bg, 1=fg)
#             mask_student: (B, H, W) binary mask for student (0=bg, 1=fg)
#             organ_class_teacher: (B,) organ class ID for teacher mask
#             organ_class_student: (B,) organ class ID for student mask
#                                                                                                                                                                                                                                      vgcfxw
#             loss: scalar InfoNCE contrastive loss
#         """
#         B, C, H, W = feat_teacher.shape
#         #print(organ_class_teacher[0], organ_class_student[0])
#         # Normalize features
#         feat_teacher = F.normalize(feat_teacher, dim=1)
#         feat_student = F.normalize(feat_student, dim=1)
#         #print(organ_class_teacher, organ_class_student)
#         # Reshape features to (B*H*W, C)
#         feat_teacher_flat = feat_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
#         feat_student_flat = feat_student.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
#         # Reshape masks to (B*H*W,)
#         mask_teacher_flat = mask_teacher.reshape(-1)  # (B*H*W,)
#         mask_student_flat = mask_student.reshape(-1)  # (B*H*W,)
        
#         # Find foreground pixels (mask == 1)
#         fg_teacher = mask_teacher_flat == 1
#         fg_student = mask_student_flat == 1
        
#         # Find background pixels (mask == 0)
#         bg_teacher = mask_teacher_flat == 0
#         bg_student = mask_student_flat == 0
        
#         # Get foreground features
#         fg_feat_teacher = feat_teacher_flat[fg_teacher]  # (N_fg_teacher, C)
#         fg_feat_student = feat_student_flat[fg_student]  # (N_fg_student, C)
        
#         # Get background features
#         bg_feat_teacher = feat_teacher_flat[bg_teacher]  # (N_bg_teacher, C)
#         bg_feat_student = feat_student_flat[bg_student]  # (N_bg_student, C)
        
#         # # Check if we have enough foreground and background pixels
#         # if fg_feat_teacher.shape[0] < 1 or fg_feat_student.shape[0] < 1:
#         #     return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
#         # if bg_feat_teacher.shape[0] < 1 or bg_feat_student.shape[0] < 1:
#         #     return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
#         # Check if both masks are for the same organ class (positive pairs)
#         same_organ = (organ_class_teacher == organ_class_student).all()
        
#         if not same_organ:
#             # Different organs - no positive pairs, return small loss to encourage separation
#             return torch.tensor(0.1, device=feat_teacher.device, requires_grad=True)
        
#         # Sample negative features (background features)
#         num_neg = min(self.num_negatives, bg_feat_teacher.shape[0], bg_feat_student.shape[0])
#         if num_neg > 0:
#             neg_feat_teacher = bg_feat_teacher[:num_neg]  # (num_neg, C)
#             neg_feat_student = bg_feat_student[:num_neg]  # (num_neg, C)
#         else:
#             # If no background pixels, use random features as negatives
#             neg_feat_teacher = torch.randn_like(fg_feat_teacher[:min(100, fg_feat_teacher.shape[0])])
#             neg_feat_student = torch.randn_like(fg_feat_student[:min(100, fg_feat_student.shape[0])])

#         # InfoNCE Loss Implementation
#         # Positive pairs: teacher foreground <-> student foreground (same organ)
#         # Negative pairs: teacher foreground <-> student background features
#         #                 teacher background <-> student background features (bg-bg negative pairing)

#         # Compute positive similarities
#         pos_sim = torch.mm(fg_feat_teacher, fg_feat_student.t()) / self.temperature  # (N_fg_teacher, N_fg_student)

#         # Compute negative similarities (teacher foreground vs student background)
#         neg_sim_teacher = torch.mm(fg_feat_teacher, neg_feat_student.t()) / self.temperature  # (N_fg_teacher, num_neg)
#         # Compute negative similarities (teacher background vs student background)
#         neg_sim_bg_teacher = torch.mm(neg_feat_teacher, neg_feat_student.t()) / self.temperature  # (num_neg, num_neg)

#         # For InfoNCE, create similarity matrix where each row represents
#         # one positive pair and multiple negative pairs
#         logits_teacher = torch.cat([pos_sim, neg_sim_teacher], dim=1)  # (N_fg_teacher, N_fg_student + num_neg)
#         # Optionally, you can also include bg-bg negatives in the loss if you want to penalize background similarity
#         # For now, we keep them separate for clarity

#         # Create labels for InfoNCE (the positive pair index for each row)
#         labels_teacher = torch.arange(min(fg_feat_teacher.shape[0], fg_feat_student.shape[0]), 
#                                     device=feat_teacher.device)

#         # If we have more teacher foreground pixels than student, truncate
#         if fg_feat_teacher.shape[0] > fg_feat_student.shape[0]:
#             logits_teacher = logits_teacher[:fg_feat_student.shape[0]]

#         # Compute InfoNCE loss for teacher->student direction
#         loss_teacher = F.cross_entropy(logits_teacher, labels_teacher)

#         # Compute InfoNCE loss for student->teacher direction (symmetric)
#         neg_sim_student = torch.mm(fg_feat_student, neg_feat_teacher.t()) / self.temperature  # (N_fg_student, num_neg)
#         logits_student = torch.cat([pos_sim.t(), neg_sim_student], dim=1)  # (N_fg_student, N_fg_teacher + num_neg)
#         labels_student = torch.arange(min(fg_feat_student.shape[0], fg_feat_teacher.shape[0]), 
#                                     device=feat_teacher.device)

#         if fg_feat_student.shape[0] > fg_feat_teacher.shape[0]:
#             logits_student = logits_student[:fg_feat_teacher.shape[0]]

#         loss_student = F.cross_entropy(logits_student, labels_student)

#         # Background-background negative pairing loss (optional, can be weighted)
#         # Encourage background features to be dissimilar between teacher and student
#         # Here, we use InfoNCE-style loss for bg-bg as well
#         if num_neg > 1:
#             # For bg-bg, treat each teacher bg as anchor, student bg as positives (diagonal), rest as negatives
#             bg_labels = torch.arange(num_neg, device=feat_teacher.device)
#             loss_bg = F.cross_entropy(neg_sim_bg_teacher, bg_labels)
#         else:
#             loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

#         # Return average of all three losses (can be weighted if desired)
#         return (loss_teacher + loss_student + loss_bg) / 3

# class ContrastiveLoss(nn.Module):
#     """
#     Legacy contrastive loss - kept for backward compatibility
#     """
#     def __init__(self, temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.supervised_loss = SupervisedPixelWiseContrastiveLoss(temperature=temperature)

#     def forward(self, feature_teacher, feature_student, mask_teacher, mask_student, organ_class_teacher=None, organ_class_student=None):
#         """
#         Wrapper for supervised contrastive loss
#         """
#         # If organ class information is not provided, assume same organ
#         #print(organ_class_teacher, organ_class_student)
#         if organ_class_teacher is None:
#             organ_class_teacher = torch.ones(feature_teacher.shape[0], device=feature_teacher.device)
#         if organ_class_student is None:
#             organ_class_student = torch.ones(feature_student.shape[0], device=feature_student.device)
            
#         return self.supervised_loss(feature_teacher, feature_student, mask_teacher, mask_student, 
#                                   organ_class_teacher, organ_class_student)