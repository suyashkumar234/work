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
        # Handle different input formats
        if feat_teacher.dim() == 6:  # (1, 1, 1, 256, 32, 32)
            feat_teacher = feat_teacher[0, 0, 0]  # (256, 32, 32)
            feat_student = feat_student[0, 0, 0]  # (256, 32, 32)
            mask = mask[0, 0, 0]  # (32, 32)
        elif feat_teacher.dim() == 5:  # (1, 1, 256, 32, 32)
            feat_teacher = feat_teacher[0, 0]  # (256, 32, 32)
            feat_student = feat_student[0, 0]  # (256, 32, 32)
            mask = mask[0, 0]  # (32, 32)
        elif feat_teacher.dim() == 4:  # (1, 256, 32, 32)
            feat_teacher = feat_teacher[0]  # (256, 32, 32)
            feat_student = feat_student[0]  # (256, 32, 32)
            mask = mask[0, 0]  # (32, 32)
        else:
            raise ValueError(f"Unexpected input shape: {feat_teacher.shape}")
        
        C, H, W = feat_teacher.shape
        feat_teacher = feat_teacher.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        feat_student = feat_student.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        mask = mask.view(-1)  # (H*W,)

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


class OrganAwareContrastiveLoss(nn.Module):
    """
    Organ-aware contrastive loss that clusters similar organs together.
    Uses spatial proximity and feature similarity to identify organ-level positive pairs.
    """
    def __init__(self, temperature=0.1, spatial_weight=0.3, feature_weight=0.7, 
                 spatial_threshold=5.0, min_organ_size=5):
        super().__init__()
        self.temperature = temperature
        self.spatial_weight = spatial_weight
        self.feature_weight = feature_weight
        self.spatial_threshold = spatial_threshold
        self.min_organ_size = min_organ_size

    def forward(self, feat_teacher, feat_student, mask):
        """
        Args:
            feat_teacher: (B, C, H, W) feature map from teacher encoder
            feat_student: (B, C, H, W) feature map from student encoder
            mask: (B, 1, H, W) binary mask, 1=foreground, 0=background
        Returns:
            loss: scalar contrastive loss
        """
        # Handle different input formats
        #if feat_teacher.dim() == 6:  # (1, 1, 1, 256, 32, 32)
        feat_teacher = feat_teacher[0, 0, 0]  # (256, 32, 32)
        feat_student = feat_student[0, 0, 0]  # (256, 32, 32)
        mask = mask[0, 0, 0]  # (32, 32)
       
        
        C, H, W = feat_teacher.shape
        
        # Get foreground indices
        #print(torch.where(mask == 1))- gives the indices of the foreground pixels where mask is 1. First element is the row indices(y coordinates), second element is the column indices(x coordinates).
        fg_indices = torch.where(mask == 1)
        #print(len(fg_indices[0]))
        #print(len(fg_indices[1]))
        
        if len(fg_indices[0]) < self.min_organ_size:
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
        # Get foreground features
        feat_teacher_fg = feat_teacher[:, fg_indices[0], fg_indices[1]]  # (C, N_fg)
        #print(feat_teacher_fg.shape)
        feat_student_fg = feat_student[:, fg_indices[0], fg_indices[1]]  # (C, N_fg)
        
        # Normalize features
        feat_teacher_fg = F.normalize(feat_teacher_fg, dim=0)  # (C, N_fg)
        feat_student_fg = F.normalize(feat_student_fg, dim=0)  # (C, N_fg)
        
        N_fg = feat_teacher_fg.shape[1]
        # print(N_fg)- number of foreground pixels
        
        # Create spatial coordinates for organ clustering
        y_coords = fg_indices[0].float().unsqueeze(0)  # (1, N_fg)
        #print(y_coords)
        x_coords = fg_indices[1].float().unsqueeze(0)  # (1, N_fg)
        coords = torch.cat([y_coords, x_coords], dim=0)  # (2, N_fg)

        # Compute spatial distances between all foreground pixels
        spatial_dist = torch.cdist(coords.t(), coords.t())  # (N_fg, N_fg)
        # coords.t() is the transpose of the coords tensor. It is (N_fg, 2)
        # spatial_dist is the distance between all foreground pixels. It is (N_fg, N_fg)
        # Compute feature similarities
        feature_sim_teacher = torch.mm(feat_teacher_fg.t(), feat_teacher_fg)  # (N_fg, N_fg)
        feature_sim_student = torch.mm(feat_student_fg.t(), feat_student_fg)  # (N_fg, N_fg)
        feature_sim_cross = torch.mm(feat_teacher_fg.t(), feat_student_fg)  # (N_fg, N_fg)
        
        # Create organ similarity matrix (combining spatial and feature similarity)
        spatial_sim = torch.exp(-spatial_dist / self.spatial_threshold) 
        feature_sim = (feature_sim_teacher + feature_sim_student) / 2
        
        # Combined similarity for organ identification
        organ_sim = (self.spatial_weight * spatial_sim + 
                    self.feature_weight * feature_sim) # organ_sim is the similarity between all foreground pixels. It is (N_fg, N_fg)
        
        # Adaptive threshold based on similarity distribution
        # Use a more lenient threshold early in training

        # will change this later
        similarity_threshold = max(0.5, organ_sim.mean() - 0.1)  # Adaptive threshold
        organ_clusters = organ_sim > similarity_threshold # organ_clusters is a boolean tensor of shape (N_fg, N_fg)
        
        #print(organ_clusters)- True or False for each foreground pixel.
        # Debug: print organ clustering statistics
        # print(f"Organ clustering debug:")
        # print(f"  - Total foreground pixels: {N_fg}")
        # print(f"  - Organ similarity range: {organ_sim.min():.3f} to {organ_sim.max():.3f}")
        # print(f"  - Pixels above threshold: {organ_clusters.sum()}")
        # print(f"  - Average similarity: {organ_sim.mean():.3f}")
        
        # Create positive pairs: pixels from same organ across encoders
        positive_pairs = []
        negative_pairs = []
        #print(organ_clusters.shape)
        # print(organ_clusters)
        # print(organ_clusters[3])
        for i in range(N_fg):
            # Find pixels in the same organ cluster
            same_organ = organ_clusters[i] # same_organ is a boolean tensor of shape (N_fg,)
            same_organ_indices = torch.where(same_organ)[0]# same_organ_indices is a tensor of shape (N_fg,) here y is the row index and x is the column index
            #print(same_organ_indices)
            
            if len(same_organ_indices) > 1:
                # Positive pairs: teacher pixel i with student pixels in same organ
                for j in same_organ_indices:
                    if i != j:  # Don't pair with self
                        positive_pairs.append((i, j))
                
                # Negative pairs: teacher pixel i with student pixels from different organs
                diff_organ_indices = torch.where(~same_organ)[0]
                if len(diff_organ_indices) > 0:
                    # Sample some negative pairs for this anchor
                    #num_neg = min(len(diff_organ_indices), 5)  # Limit negatives
                    num_neg = len(diff_organ_indices)
                    neg_indices = torch.randperm(len(diff_organ_indices))[:num_neg]
                    for idx in neg_indices:
                        negative_pairs.append((i, diff_organ_indices[idx])) # randomly select negative pairs from different organs
        #print(positive_pairs)-[(0, tensor(1, device='mps:0')), (0, tensor(2, device='mps:0')), (0, tensor(3, device='mps:0')), (0, tensor(4, device='mps:0')), (0, tensor(6, device='mps:0')), (0, tensor(7, device='mps:0')), (0, tensor(8, device='mps:0')), (0, tensor(9, device='mps:0')), (0, tensor(10, device='mps:0')), (0, tensor(14, device='mps:0')), (0, tensor(15, device='mps:0')), (0, tensor(16, device='mps:0')), (0, tensor(17, device='mps:0')), (0, tensor(18, device='mps:0')), (0, tensor(20, device='mps:0')), (0, tensor(27, device='mps:0')), (1, tensor(0, device='mps:0')),......
        print(f"  - Positive pairs found: {len(positive_pairs)}")
        print(f"  - Negative pairs found: {len(negative_pairs)}")
        #print(positive_pairs[2])
        if len(positive_pairs) == 0 or len(negative_pairs) == 0:
            print(f"  - WARNING: No valid pairs found, returning zero loss")
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
        # Sample pairs for efficiency and create anchor-based negative pairs
        max_pairs = min(50, len(positive_pairs))
        pos_indices = torch.randperm(len(positive_pairs))[:max_pairs]
        pos_pairs = [positive_pairs[i] for i in pos_indices]
        
        # Compute contrastive loss using all foreground pixels as potential negatives
        loss = 0
        valid_losses = 0
        
        for pos_i, pos_j in pos_pairs:
            # Positive pair: teacher feature i with student feature j
            pos_sim = torch.dot(feat_teacher_fg[:, pos_i], feat_student_fg[:, pos_j]) / self.temperature
            
            # Use all other foreground pixels as negatives (except those in same organ)
            same_organ = organ_clusters[pos_i]
            neg_indices = torch.where(~same_organ)[0]
            
            if len(neg_indices) > 0:
                # Sample negative pairs
                #num_neg = min(len(neg_indices), 10)
                num_neg = len(neg_indices)
                neg_sample = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]
                
                # Compute negative similarities
                neg_sims = torch.matmul(feat_teacher_fg[:, pos_i].unsqueeze(0), feat_student_fg[:, neg_sample])  # shape (1, num_neg)
                neg_sims = neg_sims.squeeze(0) / self.temperature  # shape (num_neg,)
                
                # Combine positive and negative logits
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                valid_losses += 1
        
        #print(f"  - Valid losses computed: {valid_losses}")
        
        if valid_losses > 0:
            return loss / valid_losses
        else:
            print(f"  - WARNING: No valid losses computed, returning zero loss")
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True) 


class TwoLevelContrastiveLoss(nn.Module):
    """
    Two-level contrastive loss for segmentation of feature maps using InfoNCE
    Level 1: Global foreground vs background contrastive loss
    Level 2: Local organ vs organ contrastive loss within foreground
    """
    def __init__(self, temperature=0.1, spatial_weight=0.3, feature_weight=0.7, 
                 spatial_threshold=5.0, min_organ_size=5):
        super().__init__()
        self.temperature = temperature
        self.spatial_weight = spatial_weight
        self.feature_weight = feature_weight
        self.spatial_threshold = spatial_threshold
        self.min_organ_size = min_organ_size

    def forward(self, feat_teacher, feat_student, mask_teacher, mask_student):
        """
        Args:
            feat_teacher: (B, C, H, W) feature map from teacher encoder
            feat_student: (B, C, H, W) feature map from student encoder
            mask_teacher: (B, 1, H, W) binary mask from teacher, 1=foreground, 0=background
            mask_student: (B, 1, H, W) binary mask from student, 1=foreground, 0=background
        Returns:
            loss: scalar contrastive loss
        """
        feat_teacher = feat_teacher[0, 0, 0]
        feat_student = feat_student[0, 0, 0]
        mask_teacher = mask_teacher[0, 0, 0]
        mask_student = mask_student[0, 0, 0]

        C, H, W = feat_teacher.shape

        # Level 1: Global contrastive loss (foreground vs background)
        level1_loss = self._compute_level1_loss(feat_teacher, feat_student, mask_teacher, mask_student)
        
        # Level 2: Local contrastive loss (organ vs organ within foreground)
        level2_loss = self._compute_level2_loss(feat_teacher, feat_student, mask_teacher, mask_student)
        
        # Combine losses (you can adjust weights)
        total_loss = 0.7*level1_loss + 0.3*level2_loss
        
        return total_loss

    def _compute_level1_loss(self, feat_teacher, feat_student, mask_teacher, mask_student):
        """
        Level 1: Global contrastive loss using InfoNCE
        Positive: Teacher FG ↔ Student FG, Teacher BG ↔ Student BG
        Negative: Teacher FG ↔ Student BG, Teacher BG ↔ Student FG
        Also: Student FG ↔ Teacher BG, Student BG ↔ Teacher FG (symmetric)
        """
        C, H, W = feat_teacher.shape
        
        # Flatten features and masks
        feat_teacher_flat = feat_teacher.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        feat_student_flat = feat_student.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        mask_teacher_flat = mask_teacher.view(-1)  # (H*W,)
        mask_student_flat = mask_student.view(-1)  # (H*W,)
        
        # Normalize features
        feat_teacher_flat = F.normalize(feat_teacher_flat, dim=1)
        feat_student_flat = F.normalize(feat_student_flat, dim=1)
        
        # Get indices for different regions
        teacher_fg_idx = mask_teacher_flat == 1
        teacher_bg_idx = mask_teacher_flat == 0
        student_fg_idx = mask_student_flat == 1
        student_bg_idx = mask_student_flat == 0
        
        if teacher_fg_idx.sum() == 0 or teacher_bg_idx.sum() == 0 or student_fg_idx.sum() == 0 or student_bg_idx.sum() == 0:
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        
        # Sample pixels for efficiency
        n_samples = 50 # number of samples to draw from the foreground and background pixels, limiting the number of samples to 50 majorly limiting the number of background pixels

        
        # Teacher foreground features
        teacher_fg_indices = torch.where(teacher_fg_idx)[0][torch.randperm(teacher_fg_idx.sum())[:min(n_samples, teacher_fg_idx.sum())]]
        teacher_fg_feat = feat_teacher_flat[teacher_fg_indices]
        
        # Teacher background features
        teacher_bg_indices = torch.where(teacher_bg_idx)[0][torch.randperm(teacher_bg_idx.sum())[:min(n_samples, teacher_bg_idx.sum())]]
        teacher_bg_feat = feat_teacher_flat[teacher_bg_indices]
        
        # Student foreground features
        student_fg_indices = torch.where(student_fg_idx)[0][torch.randperm(student_fg_idx.sum())[:min(n_samples, student_fg_idx.sum())]]
        student_fg_feat = feat_student_flat[student_fg_indices]
        
        # Student background features
        student_bg_indices = torch.where(student_bg_idx)[0][torch.randperm(student_bg_idx.sum())[:min(n_samples, student_bg_idx.sum())]]
        student_bg_feat = feat_student_flat[student_bg_indices]
        
        loss = 0
        valid_pairs = 0
        
        # InfoNCE: Teacher FG vs Student FG (positive) and Student BG (negative)
        if len(teacher_fg_feat) > 0 and len(student_fg_feat) > 0 and len(student_bg_feat) > 0:
            # Positive pairs: teacher FG with student FG
            pos_sims = torch.mm(teacher_fg_feat, student_fg_feat.t()) / self.temperature  # (n_teacher_fg, n_student_fg)
            
            # Negative pairs: teacher FG with student BG
            neg_sims = torch.mm(teacher_fg_feat, student_bg_feat.t()) / self.temperature  # (n_teacher_fg, n_student_bg)
            
            # For each teacher FG pixel, compute InfoNCE loss
            for i in range(len(teacher_fg_feat)):
                # Positive similarity (best match with student FG)
                pos_sim = pos_sims[i].max() # max similarity between the teacher FG pixel and the student FG pixel acting as an anchor
                
                # Negative similarities (all student BG)
                neg_sim = neg_sims[i]
                
                # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                valid_pairs += 1
        
        # InfoNCE: Teacher BG vs Student BG (positive) and Student FG (negative)
        if len(teacher_bg_feat) > 0 and len(student_bg_feat) > 0 and len(student_fg_feat) > 0:
            # Positive pairs: teacher BG with student BG
            pos_sims = torch.mm(teacher_bg_feat, student_bg_feat.t()) / self.temperature
            
            # Negative pairs: teacher BG with student FG
            neg_sims = torch.mm(teacher_bg_feat, student_fg_feat.t()) / self.temperature
            
            for i in range(len(teacher_bg_feat)):
                pos_sim = pos_sims[i].max() # 
                neg_sim = neg_sims[i]
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                valid_pairs += 1
        
        # InfoNCE: Student FG vs Teacher FG (positive) and Teacher BG (negative) - SYMMETRIC
        if len(student_fg_feat) > 0 and len(teacher_fg_feat) > 0 and len(teacher_bg_feat) > 0:
            # Positive pairs: student FG with teacher FG
            pos_sims = torch.mm(student_fg_feat, teacher_fg_feat.t()) / self.temperature
            
            # Negative pairs: student FG with teacher BG
            neg_sims = torch.mm(student_fg_feat, teacher_bg_feat.t()) / self.temperature
            
            for i in range(len(student_fg_feat)):
                pos_sim = pos_sims[i].max()
                neg_sim = neg_sims[i]
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                valid_pairs += 1
        
        # InfoNCE: Student BG vs Teacher BG (positive) and Teacher FG (negative) - SYMMETRIC
        if len(student_bg_feat) > 0 and len(teacher_bg_feat) > 0 and len(teacher_fg_feat) > 0:
            # Positive pairs: student BG with teacher BG
            pos_sims = torch.mm(student_bg_feat, teacher_bg_feat.t()) / self.temperature
            
            # Negative pairs: student BG with teacher FG
            neg_sims = torch.mm(student_bg_feat, teacher_fg_feat.t()) / self.temperature
            
            for i in range(len(student_bg_feat)):
                pos_sim = pos_sims[i].max()
                neg_sim = neg_sims[i]
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                valid_pairs += 1
        
        return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
    
    def _compute_level2_loss(self, feat_teacher, feat_student, mask_teacher, mask_student):
        """
        Level 2: Local contrastive loss (Organ vs Organ within foreground) using InfoNCE
        Uses pseudo-masks as cluster labels for robust organ-aware contrastive loss.
        """
        if mask_teacher.dim() == 3:
            mask_teacher = mask_teacher.squeeze(0)
        if mask_student.dim() == 3:
            mask_student = mask_student.squeeze(0)
        unique_vals, counts = torch.unique(mask_teacher, return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"Value: {val.item()}, Count: {count.item()}")
        unique_labels = torch.unique(mask_teacher)

        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        if len(unique_labels) == 0:
            return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)
        C, H, W = feat_teacher.shape
        feat_teacher_flat = feat_teacher.view(C, -1).t()  # (H*W, C)
        feat_student_flat = feat_student.view(C, -1).t()  # (H*W, C)
        mask_teacher_flat = mask_teacher.view(-1)
        mask_student_flat = mask_student.view(-1)
        loss = 0
        valid_pairs = 0
        n_samples_per_label = 30  # To avoid OOM, sample per organ
        for label in unique_labels:
            t_idx = torch.where(mask_teacher_flat == label)[0]
            s_idx = torch.where(mask_student_flat == label)[0]
            if len(t_idx) < 2 or len(s_idx) < 2:
                continue
            # Sample for efficiency
            t_idx = t_idx[torch.randperm(len(t_idx))[:min(n_samples_per_label, len(t_idx))]]
            s_idx = s_idx[torch.randperm(len(s_idx))[:min(n_samples_per_label, len(s_idx))]]
            t_feat = F.normalize(feat_teacher_flat[t_idx], dim=1)  # (Nt, C)
            s_feat = F.normalize(feat_student_flat[s_idx], dim=1)  # (Ns, C)
            # Positives: all pairs between t_feat and s_feat
            pos_sims = torch.mm(t_feat, s_feat.t()) / self.temperature  # (Nt, Ns)
            # Negatives: all pairs with different labels
            neg_teacher_idx = torch.where((mask_teacher_flat != label) & (mask_teacher_flat != 0))[0]
            neg_student_idx = torch.where((mask_student_flat != label) & (mask_student_flat != 0))[0]
            if len(neg_teacher_idx) > 0 and len(neg_student_idx) > 0:
                neg_t_feat = F.normalize(feat_teacher_flat[neg_teacher_idx], dim=1)
                neg_s_feat = F.normalize(feat_student_flat[neg_student_idx], dim=1)
                neg_sims_t = torch.mm(t_feat, neg_s_feat.t()) / self.temperature  # (Nt, Nneg_s)
                neg_sims_s = torch.mm(s_feat, neg_t_feat.t()) / self.temperature  # (Ns, Nneg_t)
            else:
                neg_sims_t = None
                neg_sims_s = None
            # InfoNCE for teacher anchors
            for i in range(t_feat.shape[0]):
                pos_sim = pos_sims[i]  # (Ns,)
                if neg_sims_t is not None:
                    neg_sim = neg_sims_t[i]  # (Nneg_s,)
                    logits = torch.cat([pos_sim, neg_sim])
                    labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                    loss += F.cross_entropy(logits.unsqueeze(0), labels)
                    valid_pairs += 1
            # InfoNCE for student anchors
            for i in range(s_feat.shape[0]):
                pos_sim = pos_sims[:, i]  # (Nt,)
                if neg_sims_s is not None:
                    neg_sim = neg_sims_s[i]  # (Nneg_t,)
                    logits = torch.cat([pos_sim, neg_sim])
                    labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                    loss += F.cross_entropy(logits.unsqueeze(0), labels)
                    valid_pairs += 1
        return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=feat_teacher.device, requires_grad=True) 
        
        