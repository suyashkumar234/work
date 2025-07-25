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
        #print("shape of feat_online", feat_online.shape)
        feat_target = F.normalize(feat_target, dim=1)
        #print("shape of feat_target", feat_target.shape)
        
        # Reshape features to (B*H*W, C)
        feat_online_flat = feat_online.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        #print("shape of feat_online_flat", feat_online_flat.shape)
        feat_target_flat = feat_target.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        #print("shape of feat_target_flat", feat_target_flat.shape)
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
        #print("shape of fg_feat_online", fg_feat_online.shape)
        fg_feat_target = feat_target_flat[fg_target]  # (N_fg_target, C)
        #print("shape of fg_feat_target", fg_feat_target.shape)
        
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
        #print("shape of pos_sim", pos_sim.shape)
        pos_sim = torch.clamp(pos_sim, min=-50, max=50)  # Prevent overflow in cross_entropy
        #print("shape of pos_sim after clamp", pos_sim.shape)


        # Compute negative similarities (online foreground vs target background)
        neg_sim_online = torch.mm(fg_feat_online, neg_feat_target.t()) / max(self.temperature, 1e-7)
        #print("shape of neg_sim_online", neg_sim_online.shape)
        neg_sim_online = torch.clamp(neg_sim_online, min=-50, max=50)  # Prevent overflow
        # Compute negative similarities (online background vs target background)
        #print("shape of neg_feat_online", neg_feat_online.shape)
        #print("shape of neg_feat_target", neg_feat_target.shape)
        neg_sim_bg_online = torch.mm(neg_feat_online, neg_feat_target.t()) / max(self.temperature, 1e-7)
        # print("shape of neg_feat_online", neg_feat_online.shape)
        # print("shape of neg_feat_target", neg_feat_target.shape)
        # print("shape of neg_sim_bg_online", neg_sim_bg_online.shape)
        # print("neg_sim_bg_online similarity matrix:\n", neg_sim_bg_online)
        neg_sim_bg_online = torch.clamp(neg_sim_bg_online, min=-50, max=50)  # Prevent overflow
        

        # Compute row-wise average of positive similarities instead of using diagonal
        pos_sim_avg = torch.mean(pos_sim, dim=1, keepdim=True)  # (N_fg_online, 1)
        #print("shape of pos_sim_avg", pos_sim_avg.shape)
        
        # For InfoNCE, create similarity matrix where each row represents
        # one averaged positive pair and multiple negative pairs
        logits_online = torch.cat([pos_sim_avg, neg_sim_online], dim=1)  # (N_fg_online, 1 + num_neg)
        #print("shape of logits_online", logits_online.shape)
        # Optionally, you can also include bg-bg negatives in the loss if you want to penalize background similarity
        # For now, we keep them separate for clarity

        # Create labels for InfoNCE (the positive pair is now at index 0)
        labels_online = torch.zeros(fg_feat_online.shape[0], device=feat_online.device, dtype=torch.long)
        #print("labels", labels_online)

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
        #print("shape of neg_sim_target", neg_sim_target.shape)
        neg_sim_target = torch.clamp(neg_sim_target, min=-50, max=50)  # Prevent overflow
        #print("shape of neg_sim_target after clamp", neg_sim_target.shape)
        
        # Compute row-wise average of transposed pos_sim for target direction
        pos_sim_avg_target = torch.mean(pos_sim.t(), dim=1, keepdim=True)  # (N_fg_target, 1)
        #print("shape of pos_sim_avg_target", pos_sim_avg_target.shape)
        
        logits_target = torch.cat([pos_sim_avg_target, neg_sim_target], dim=1)  # (N_fg_target, 1 + num_neg)
        #print("shape of logits_target", logits_target.shape)
        labels_target = torch.zeros(fg_feat_target.shape[0], device=feat_online.device, dtype=torch.long)
        #print("labels_target", labels_target)
        if fg_feat_target.shape[0] > fg_feat_online.shape[0]:
            logits_target = logits_target[:fg_feat_online.shape[0]]

        loss_target = F.cross_entropy(logits_target, labels_target)
        #print("shape of loss_target", loss_target.shape)
        #print("loss_target", loss_target)
        # Check for NaN in loss_target
        if torch.isnan(loss_target) or torch.isinf(loss_target):
            loss_target = torch.tensor(0.0, device=feat_online.device, requires_grad=True)
        #print("loss_target after check", loss_target)
        # Background-background negative pairing loss (optional, can be weighted)
        # Encourage background features to be consistent between online and target
        # Here, we use InfoNCE-style loss for bg-bg as well
        if num_neg > 1:
            # For bg-bg, treat each online bg as anchor, target bg as positives (diagonal), rest as negatives
            bg_labels = torch.arange(num_neg, device=feat_online.device)
            #print("bg_labels", bg_labels)
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
#             return torch.tensor(0.1, device=feat_teacher.device, requires_grad=True)
#             # neg_feat_teacher = torch.randn_like(fg_feat_teacher[:min(100, fg_feat_teacher.shape[0])])
#             # neg_feat_student = torch.randn_like(fg_feat_student[:min(100, fg_feat_student.shape[0])])

#         # InfoNCE Loss Implementation
#         # Positive pairs: teacher foreground <-> student foreground (same organ)
#         # Negative pairs: teacher foreground <-> student background features
#         #                 teacher background <-> student background features (bg-bg negative pairing)

#         # Compute positive similarities with numerical stability
#         pos_sim = torch.mm(fg_feat_teacher, fg_feat_student.t()) / max(self.temperature, 1e-7)
#         pos_sim = torch.clamp(pos_sim, min=-50, max=50)  # Prevent overflow in cross_entropy

#         # Compute negative similarities (teacher foreground vs student background)
#         neg_sim_teacher = torch.mm(fg_feat_teacher, neg_feat_student.t()) / max(self.temperature, 1e-7)
#         neg_sim_teacher = torch.clamp(neg_sim_teacher, min=-50, max=50)  # Prevent overflow
#         # Compute negative similarities (teacher background vs student background)
#         neg_sim_bg_teacher = torch.mm(neg_feat_teacher, neg_feat_student.t()) / max(self.temperature, 1e-7)
#         neg_sim_bg_teacher = torch.clamp(neg_sim_bg_teacher, min=-50, max=50)  # Prevent overflow

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
        
#         # Check for NaN in loss_teacher
#         if torch.isnan(loss_teacher) or torch.isinf(loss_teacher):
#             loss_teacher = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

#         # Compute InfoNCE loss for student->teacher direction (symmetric)
#         neg_sim_student = torch.mm(fg_feat_student, neg_feat_teacher.t()) / max(self.temperature, 1e-7)
#         neg_sim_student = torch.clamp(neg_sim_student, min=-50, max=50)  # Prevent overflow
#         logits_student = torch.cat([pos_sim.t(), neg_sim_student], dim=1)  # (N_fg_student, N_fg_teacher + num_neg)
#         labels_student = torch.arange(min(fg_feat_student.shape[0], fg_feat_teacher.shape[0]), 
#                                     device=feat_teacher.device)

#         if fg_feat_student.shape[0] > fg_feat_teacher.shape[0]:
#             logits_student = logits_student[:fg_feat_teacher.shape[0]]

#         loss_student = F.cross_entropy(logits_student, labels_student)
        
#         # Check for NaN in loss_student
#         if torch.isnan(loss_student) or torch.isinf(loss_student):
#             loss_student = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

#         # Background-background negative pairing loss (optional, can be weighted)
#         # Encourage background features to be dissimilar between teacher and student
#         # Here, we use InfoNCE-style loss for bg-bg as well
#         if num_neg > 1:
#             # For bg-bg, treat each teacher bg as anchor, student bg as positives (diagonal), rest as negatives
#             bg_labels = torch.arange(num_neg, device=feat_teacher.device)
#             loss_bg = F.cross_entropy(neg_sim_bg_teacher, bg_labels)
            
#             # Check for NaN in loss_bg
#             if torch.isnan(loss_bg) or torch.isinf(loss_bg):
#                 loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
#         else:
#             loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

#         # Return average of all three losses (can be weighted if desired)
#         # print('')
#         # print(loss_teacher, loss_student, loss_bg)
#         # print('')
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
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class SupervisedPixelWiseContrastiveLoss(nn.Module):
# #     """
# #     Supervised pixel-wise InfoNCE contrastive loss for teacher-student learning.
# #     Uses binary masks (0/1) with organ class information to create positive pairs 
# #     (same organ class) and negative pairs (foreground vs background).
    
# #     Args:
# #         temperature: Temperature parameter for contrastive learning
# #         num_negatives: Number of negative samples per positive pair
# #     """
# #     def __init__(self, temperature=0.1, num_negatives=1000):
# #         super(SupervisedPixelWiseContrastiveLoss, self).__init__()
# #         self.temperature = temperature
# #         self.num_negatives = num_negatives

# #     def forward(self, feat_teacher, feat_student, mask_teacher, mask_student, organ_class_teacher, organ_class_student):
# #         """
# #         Args:
# #             feat_teacher: (B, C, H, W) feature map from teacher encoder
# #             feat_student: (B, C, H, W) feature map from student encoder  
# #             mask_teacher: (B, H, W) binary mask for teacher (0=bg, 1=fg)
# #             mask_student: (B, H, W) binary mask for student (0=bg, 1=fg)
# #             organ_class_teacher: (B,) organ class ID for teacher mask
# #             organ_class_student: (B,) organ class ID for student mask
# #                                                                                                                                                                                                                                      vgcfxw
# #             loss: scalar InfoNCE contrastive loss
# #         """
# #         B, C, H, W = feat_teacher.shape
# #         #print(organ_class_teacher[0], organ_class_student[0])
# #         # Normalize features
# #         feat_teacher = F.normalize(feat_teacher, dim=1)
# #         feat_student = F.normalize(feat_student, dim=1)
# #         #print(organ_class_teacher, organ_class_student)
# #         # Reshape features to (B*H*W, C)
# #         feat_teacher_flat = feat_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
# #         feat_student_flat = feat_student.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
# #         # Reshape masks to (B*H*W,)
# #         mask_teacher_flat = mask_teacher.reshape(-1)  # (B*H*W,)
# #         mask_student_flat = mask_student.reshape(-1)  # (B*H*W,)
        
# #         # Find foreground pixels (mask == 1)
# #         fg_teacher = mask_teacher_flat == 1
# #         fg_student = mask_student_flat == 1
        
# #         # Find background pixels (mask == 0)
# #         bg_teacher = mask_teacher_flat == 0
# #         bg_student = mask_student_flat == 0
        
# #         # Get foreground features
# #         fg_feat_teacher = feat_teacher_flat[fg_teacher]  # (N_fg_teacher, C)
# #         fg_feat_student = feat_student_flat[fg_student]  # (N_fg_student, C)
        
# #         # Get background features
# #         bg_feat_teacher = feat_teacher_flat[bg_teacher]  # (N_bg_teacher, C)
# #         bg_feat_student = feat_student_flat[bg_student]  # (N_bg_student, C)
        
# #         # # Check if we have enough foreground and background pixels
# #         # if fg_feat_teacher.shape[0] < 1 or fg_feat_student.shape[0] < 1:
# #         #     return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
# #         # if bg_feat_teacher.shape[0] < 1 or bg_feat_student.shape[0] < 1:
# #         #     return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
# #         # Check if both masks are for the same organ class (positive pairs)
# #         same_organ = (organ_class_teacher == organ_class_student).all()
        
# #         if not same_organ:
# #             # Different organs - no positive pairs, return small loss to encourage separation
# #             return torch.tensor(0.1, device=feat_teacher.device, requires_grad=True)
        
# #         # Sample negative features (background features)
# #         num_neg = min(self.num_negatives, bg_feat_teacher.shape[0], bg_feat_student.shape[0])
# #         if num_neg > 0:
# #             neg_feat_teacher = bg_feat_teacher[:num_neg]  # (num_neg, C)
# #             neg_feat_student = bg_feat_student[:num_neg]  # (num_neg, C)
# #         else:
# #             # If no background pixels, use random features as negatives
# #             neg_feat_teacher = torch.randn_like(fg_feat_teacher[:min(100, fg_feat_teacher.shape[0])])
# #             neg_feat_student = torch.randn_like(fg_feat_student[:min(100, fg_feat_student.shape[0])])

# #         # InfoNCE Loss Implementation
# #         # Positive pairs: teacher foreground <-> student foreground (same organ)
# #         # Negative pairs: teacher foreground <-> student background features
# #         #                 teacher background <-> student background features (bg-bg negative pairing)

# #         # Compute positive similarities
# #         pos_sim = torch.mm(fg_feat_teacher, fg_feat_student.t()) / self.temperature  # (N_fg_teacher, N_fg_student)

# #         # Compute negative similarities (teacher foreground vs student background)
# #         neg_sim_teacher = torch.mm(fg_feat_teacher, neg_feat_student.t()) / self.temperature  # (N_fg_teacher, num_neg)
# #         # Compute negative similarities (teacher background vs student background)
# #         neg_sim_bg_teacher = torch.mm(neg_feat_teacher, neg_feat_student.t()) / self.temperature  # (num_neg, num_neg)

# #         # For InfoNCE, create similarity matrix where each row represents
# #         # one positive pair and multiple negative pairs
# #         logits_teacher = torch.cat([pos_sim, neg_sim_teacher], dim=1)  # (N_fg_teacher, N_fg_student + num_neg)
# #         # Optionally, you can also include bg-bg negatives in the loss if you want to penalize background similarity
# #         # For now, we keep them separate for clarity

# #         # Create labels for InfoNCE (the positive pair index for each row)
# #         labels_teacher = torch.arange(min(fg_feat_teacher.shape[0], fg_feat_student.shape[0]), 
# #                                     device=feat_teacher.device)

# #         # If we have more teacher foreground pixels than student, truncate
# #         if fg_feat_teacher.shape[0] > fg_feat_student.shape[0]:
# #             logits_teacher = logits_teacher[:fg_feat_student.shape[0]]

# #         # Compute InfoNCE loss for teacher->student direction
# #         loss_teacher = F.cross_entropy(logits_teacher, labels_teacher)

# #         # Compute InfoNCE loss for student->teacher direction (symmetric)
# #         neg_sim_student = torch.mm(fg_feat_student, neg_feat_teacher.t()) / self.temperature  # (N_fg_student, num_neg)
# #         logits_student = torch.cat([pos_sim.t(), neg_sim_student], dim=1)  # (N_fg_student, N_fg_teacher + num_neg)
# #         labels_student = torch.arange(min(fg_feat_student.shape[0], fg_feat_teacher.shape[0]), 
# #                                     device=feat_teacher.device)

# #         if fg_feat_student.shape[0] > fg_feat_teacher.shape[0]:
# #             logits_student = logits_student[:fg_feat_teacher.shape[0]]

# #         loss_student = F.cross_entropy(logits_student, labels_student)

# #         # Background-background negative pairing loss (optional, can be weighted)
# #         # Encourage background features to be dissimilar between teacher and student
# #         # Here, we use InfoNCE-style loss for bg-bg as well
# #         if num_neg > 1:
# #             # For bg-bg, treat each teacher bg as anchor, student bg as positives (diagonal), rest as negatives
# #             bg_labels = torch.arange(num_neg, device=feat_teacher.device)
# #             loss_bg = F.cross_entropy(neg_sim_bg_teacher, bg_labels)
# #         else:
# #             loss_bg = torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

# #         # Return average of all three losses (can be weighted if desired)
# #         return (loss_teacher + loss_student + loss_bg) / 3

# # class ContrastiveLoss(nn.Module):
# #     """
# #     Legacy contrastive loss - kept for backward compatibility
# #     """
# #     def __init__(self, temperature=0.5):
# #         super(ContrastiveLoss, self).__init__()
# #         self.temperature = temperature
# #         self.supervised_loss = SupervisedPixelWiseContrastiveLoss(temperature=temperature)

# #     def forward(self, feature_teacher, feature_student, mask_teacher, mask_student, organ_class_teacher=None, organ_class_student=None):
# #         """
# #         Wrapper for supervised contrastive loss
# #         """
# #         # If organ class information is not provided, assume same organ
# #         #print(organ_class_teacher, organ_class_student)
# #         if organ_class_teacher is None:
# #             organ_class_teacher = torch.ones(feature_teacher.shape[0], device=feature_teacher.device)
# #         if organ_class_student is None:
# #             organ_class_student = torch.ones(feature_student.shape[0], device=feature_student.device)
            
# #         return self.supervised_loss(feature_teacher, feature_student, mask_teacher, mask_student, 
# #                                   organ_class_teacher, organ_class_student)
