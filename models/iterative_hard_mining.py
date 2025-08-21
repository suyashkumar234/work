"""
Iterative Hard Sample Mining Module for SSL Medical Segmentation
Identifies hard positives and negatives based on prediction errors and uses attention for refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssl_attention import SSLAttentionModule, MultiHeadAttention


class HardSampleDetector(nn.Module):
    """
    Detects hard positive and negative samples based on prediction-ground truth discrepancies
    """
    
    def __init__(self, error_threshold=0.5, min_hard_samples=10):
        super(HardSampleDetector, self).__init__()
        self.error_threshold = error_threshold
        self.min_hard_samples = min_hard_samples
    
    def detect_hard_samples(self, prediction_mask, ground_truth_mask):
        """
        Identify hard positive and negative samples based on prediction errors
        
        Args:
            prediction_mask: [B, H, W] - predicted binary mask (0-1)
            ground_truth_mask: [B, H, W] - ground truth binary mask (0-1)
            
        Returns:
            hard_positives: [B, H, W] - binary mask of hard positive regions
            hard_negatives: [B, H, W] - binary mask of hard negative regions
        """
        B, H, W = prediction_mask.shape
        
        # Convert to binary if not already
        pred_binary = (prediction_mask > 0.5).float()
        gt_binary = (ground_truth_mask > 0.5).float()
        
        # Hard negatives: predicted as foreground but actually background
        hard_negatives = (pred_binary == 1) & (gt_binary == 0)  # False Positives
        
        # Hard positives: predicted as background but actually foreground  
        hard_positives = (pred_binary == 0) & (gt_binary == 1)  # False Negatives
        
        return hard_positives.float(), hard_negatives.float()
    
    def get_hard_sample_features(self, features, hard_mask):
        """
        Extract features corresponding to hard sample locations
        
        Args:
            features: [B, C, H, W] - feature maps
            hard_mask: [B, H, W] - binary mask indicating hard sample locations
            
        Returns:
            hard_features: [N_hard, C] - features at hard sample locations
            hard_indices: [N_hard, 3] - (batch_idx, h_idx, w_idx) of hard samples
        """
        B, C, H, W = features.shape
        
        # Find locations of hard samples
        hard_locations = torch.nonzero(hard_mask, as_tuple=False)  # [N_hard, 3]
        
        if hard_locations.size(0) == 0:
            # No hard samples found, return empty tensors
            return torch.empty(0, C, device=features.device), torch.empty(0, 3, device=features.device)
        
        # Extract features at hard sample locations
        batch_indices = hard_locations[:, 0]
        h_indices = hard_locations[:, 1] 
        w_indices = hard_locations[:, 2]
        
        # FIXED: Remove the transpose that was causing dimension mismatch
        hard_features = features[batch_indices, :, h_indices, w_indices]  # [N_hard, C] - CORRECT!
        
        return hard_features, hard_locations


class HardSampleAttention(nn.Module):
    """
    Attention mechanism specifically for hard sample refinement
    """
    
    def __init__(self, feature_dim, n_heads=4, dropout=0.1):
        super(HardSampleAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention = MultiHeadAttention(feature_dim, n_heads, dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hard_features, context_features):
        """
        Apply attention between hard samples and context features
        
        Args:
            hard_features: [N_hard, C] - features of hard samples
            context_features: [N_context, C] - context features to attend to
            
        Returns:
            refined_features: [N_hard, C] - attention-refined hard sample features
            attention_weights: attention weights for analysis
        """
        if hard_features.size(0) == 0:
            return hard_features, None
            
        # Apply cross-attention: hard samples attend to context
        refined_features, attention_weights = self.attention(
            hard_features.unsqueeze(0),  # Add batch dim: [1, N_hard, C]
            context_features.unsqueeze(0),  # Add batch dim: [1, N_context, C] 
            context_features.unsqueeze(0)   # Add batch dim: [1, N_context, C]
        )
        
        # Remove batch dimension and apply residual connection
        refined_features = refined_features.squeeze(0)  # [N_hard, C]
        refined_features = self.norm(hard_features + self.dropout(refined_features))
        
        return refined_features, attention_weights


class IterativeHardMiningModule(nn.Module):
    """
    Complete iterative hard sample mining pipeline
    Integrates with existing SSL attention framework
    """
    
    def __init__(self, feature_dim, n_iterations=3, n_heads=4, dropout=0.1, 
                 error_threshold=0.5, mining_strength=0.8):
        super(IterativeHardMiningModule, self).__init__()
        self.n_iterations = n_iterations
        self.mining_strength = mining_strength
        
        # Components
        self.hard_detector = HardSampleDetector(error_threshold=error_threshold)
        self.hard_attention = HardSampleAttention(feature_dim, n_heads, dropout)
        
        # Feature refinement layers
        self.feature_refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Confidence prediction for mining control
        self.confidence_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, online_features, target_features, initial_prediction, ground_truth_mask=None):
        """
        Perform iterative hard sample mining and attention refinement
        
        Args:
            online_features: [B, C, H, W] - teacher encoder features
            target_features: [B, C, H, W] - student encoder features  
            initial_prediction: [B, H, W] - initial prediction mask
            ground_truth_mask: [B, H, W] - ground truth for mining (optional)
            
        Returns:
            refined_online_features: [B, C, H, W] - refined teacher features
            refined_target_features: [B, C, H, W] - refined student features
            mining_history: dict with mining statistics and attention weights
        """
        B, C, H, W = online_features.shape
        device = online_features.device
        
        refined_online = online_features.clone()
        refined_target = target_features.clone()
        
        mining_history = {
            'iterations': [],
            'hard_positive_counts': [],
            'hard_negative_counts': [],
            'attention_weights': []
        }
        
        current_prediction = initial_prediction.clone()
        
        for iteration in range(self.n_iterations):
            iteration_info = {'iteration': iteration}
            
            if ground_truth_mask is not None:
                # Detect hard samples using ground truth
                hard_positives, hard_negatives = self.hard_detector.detect_hard_samples(
                    current_prediction, ground_truth_mask
                )
            else:
                # Use prediction confidence for pseudo hard sample detection
                confidence_scores = self._compute_prediction_confidence(refined_online, current_prediction)
                hard_positives, hard_negatives = self._detect_hard_samples_from_confidence(
                    current_prediction, confidence_scores
                )
            
            # Count hard samples for monitoring
            hard_pos_count = hard_positives.sum().item()
            hard_neg_count = hard_negatives.sum().item()
            
            mining_history['hard_positive_counts'].append(hard_pos_count)
            mining_history['hard_negative_counts'].append(hard_neg_count)
            
            if hard_pos_count == 0 and hard_neg_count == 0:
                # No hard samples found, break early
                break
                
            # Extract hard sample features
            hard_pos_feats_online, hard_pos_indices = self.hard_detector.get_hard_sample_features(
                refined_online, hard_positives
            )
            hard_neg_feats_online, hard_neg_indices = self.hard_detector.get_hard_sample_features(
                refined_online, hard_negatives
            )
            
            hard_pos_feats_target, _ = self.hard_detector.get_hard_sample_features(
                refined_target, hard_positives
            )
            hard_neg_feats_target, _ = self.hard_detector.get_hard_sample_features(
                refined_target, hard_negatives
            )
            
            # Get context features (all foreground and background features)
            fg_mask = (ground_truth_mask > 0.5).float() if ground_truth_mask is not None else (current_prediction > 0.5).float()
            bg_mask = 1.0 - fg_mask
            
            fg_feats_online, _ = self.hard_detector.get_hard_sample_features(refined_online, fg_mask)
            bg_feats_online, _ = self.hard_detector.get_hard_sample_features(refined_online, bg_mask)
            
            # Refine hard positive features (attend to foreground context)
            if hard_pos_feats_online.size(0) > 0 and fg_feats_online.size(0) > 0:
                # Add robust safety checks to avoid dimension mismatch
                min_samples = 10  # Allow attention with reasonable sample sizes
                if (hard_pos_feats_online.size(0) >= min_samples and 
                    fg_feats_online.size(0) >= min_samples and 
                    hard_pos_feats_online.size(1) == 256 and 
                    fg_feats_online.size(1) == 256):
                    
                    try:
                        refined_hard_pos_online, pos_attn_weights = self.hard_attention(
                            hard_pos_feats_online, fg_feats_online
                        )
                        refined_hard_pos_target, _ = self.hard_attention(
                            hard_pos_feats_target, fg_feats_online
                        )
                        
                        # Update features at hard positive locations
                        self._update_features_at_locations(
                            refined_online, refined_hard_pos_online, hard_pos_indices
                        )
                        self._update_features_at_locations(
                            refined_target, refined_hard_pos_target, hard_pos_indices
                        )
                        
                        iteration_info['pos_attention_weights'] = pos_attn_weights
                    except RuntimeError as e:
                        print(f"Positive attention failed, using feature refinement fallback: {e}")
                        # Fallback to simple refinement with safety check
                        if hard_pos_feats_online.size(0) >= 5:
                            try:
                                refined_hard_pos_online = self.feature_refiner(hard_pos_feats_online)
                                refined_hard_pos_target = self.feature_refiner(hard_pos_feats_target)
                                
                                self._update_features_at_locations(
                                    refined_online, refined_hard_pos_online, hard_pos_indices
                                )
                                self._update_features_at_locations(
                                    refined_target, refined_hard_pos_target, hard_pos_indices
                                )
                            except RuntimeError:
                                print(f"Feature refinement also failed, skipping positive samples")
                                pass
                else:
                    # Skip attention if conditions not met, but still try simple refinement
                    if hard_pos_feats_online.size(0) >= 5:  # Only refine if we have at least 5 samples
                        try:
                            refined_hard_pos_online = self.feature_refiner(hard_pos_feats_online)
                            refined_hard_pos_target = self.feature_refiner(hard_pos_feats_target)
                            
                            # Update with refined features
                            self._update_features_at_locations(
                                refined_online, refined_hard_pos_online, hard_pos_indices
                            )
                            self._update_features_at_locations(
                                refined_target, refined_hard_pos_target, hard_pos_indices
                            )
                            print(f"✅ Applied feature refinement to {hard_pos_feats_online.size(0)} hard positive samples")
                        except RuntimeError as e:
                            # If even feature refinement fails, skip this iteration
                            print(f"Skipping hard positive refinement: feature_refiner failed with {hard_pos_feats_online.size(0)} samples - {e}")
                            pass
                    else:
                        print(f"Skipping hard positive refinement: too few samples ({hard_pos_feats_online.size(0)})")
            
            # Refine hard negative features (attend to background context)
            if hard_neg_feats_online.size(0) > 0 and bg_feats_online.size(0) > 0:
                # Add robust safety checks to avoid dimension mismatch
                min_samples = 10  # Allow attention with reasonable sample sizes
                if (hard_neg_feats_online.size(0) >= min_samples and 
                    bg_feats_online.size(0) >= min_samples and 
                    hard_neg_feats_online.size(1) == 256 and 
                    bg_feats_online.size(1) == 256):
                    
                    try:
                        refined_hard_neg_online, neg_attn_weights = self.hard_attention(
                            hard_neg_feats_online, bg_feats_online
                        )
                        refined_hard_neg_target, _ = self.hard_attention(
                            hard_neg_feats_target, bg_feats_online
                        )
                        
                        # Update features at hard negative locations
                        self._update_features_at_locations(
                            refined_online, refined_hard_neg_online, hard_neg_indices
                        )
                        self._update_features_at_locations(
                            refined_target, refined_hard_neg_target, hard_neg_indices
                        )
                        
                        iteration_info['neg_attention_weights'] = neg_attn_weights
                    except RuntimeError as e:
                        print(f"Negative attention failed, using feature refinement fallback: {e}")
                        # Fallback to simple refinement with safety check
                        if hard_neg_feats_online.size(0) >= 5:
                            try:
                                refined_hard_neg_online = self.feature_refiner(hard_neg_feats_online)
                                refined_hard_neg_target = self.feature_refiner(hard_neg_feats_target)
                                
                                self._update_features_at_locations(
                                    refined_online, refined_hard_neg_online, hard_neg_indices
                                )
                                self._update_features_at_locations(
                                    refined_target, refined_hard_neg_target, hard_neg_indices
                                )
                            except RuntimeError:
                                print(f"Feature refinement also failed, skipping negative samples")
                                pass
                else:
                    # Skip attention if conditions not met, but still try simple refinement
                    if hard_neg_feats_online.size(0) >= 5:  # Only refine if we have at least 5 samples
                        try:
                            refined_hard_neg_online = self.feature_refiner(hard_neg_feats_online)
                            refined_hard_neg_target = self.feature_refiner(hard_neg_feats_target)
                            
                            # Update with refined features
                            self._update_features_at_locations(
                                refined_online, refined_hard_neg_online, hard_neg_indices
                            )
                            self._update_features_at_locations(
                                refined_target, refined_hard_neg_target, hard_neg_indices
                            )
                            print(f"✅ Applied feature refinement to {hard_neg_feats_online.size(0)} hard negative samples")
                        except RuntimeError as e:
                            # If even feature refinement fails, skip this iteration
                            print(f"Skipping hard negative refinement: feature_refiner failed with {hard_neg_feats_online.size(0)} samples - {e}")
                            pass
                    else:
                        print(f"Skipping hard negative refinement: too few samples ({hard_neg_feats_online.size(0)})")
            
            mining_history['iterations'].append(iteration_info)
            
            # Update prediction for next iteration (simplified)
            if iteration < self.n_iterations - 1:
                current_prediction = self._update_prediction(refined_online, current_prediction)
        
        return refined_online, refined_target, mining_history
    
    def _compute_prediction_confidence(self, features, prediction):
        """Compute confidence scores for predictions"""
        B, C, H, W = features.shape
        
        # Pool features and predict confidence
        pooled_features = F.adaptive_avg_pool2d(features, 1).view(B, C)
        confidence = self.confidence_predictor(pooled_features)
        
        # Broadcast confidence to spatial dimensions
        confidence_map = confidence.view(B, 1, 1).expand(B, H, W)
        
        return confidence_map
    
    def _detect_hard_samples_from_confidence(self, prediction, confidence):
        """Detect hard samples using confidence scores"""
        # Low confidence regions are considered hard
        low_confidence = confidence < 0.5
        
        # Hard positives: low confidence + predicted as foreground
        hard_positives = low_confidence & (prediction > 0.5)
        
        # Hard negatives: low confidence + predicted as background  
        hard_negatives = low_confidence & (prediction <= 0.5)
        
        return hard_positives.float(), hard_negatives.float()
    
    def _update_features_at_locations(self, feature_map, new_features, locations):
        """Update feature map at specific locations"""
        if locations.size(0) == 0 or new_features.size(0) == 0:
            return
            
        batch_indices = locations[:, 0]
        h_indices = locations[:, 1]
        w_indices = locations[:, 2]
        
        # FIXED: Remove transpose operations that were causing dimension issues
        # Get original features at locations: [N_hard, C]
        original_features = feature_map[batch_indices, :, h_indices, w_indices]
        
        # Apply mining strength as a blending factor
        blended_features = (
            self.mining_strength * new_features + 
            (1 - self.mining_strength) * original_features
        )
        
        # Update feature map directly: [N_hard, C] -> assign back
        feature_map[batch_indices, :, h_indices, w_indices] = blended_features
    
    def _update_prediction(self, features, current_prediction):
        """Simple prediction update based on refined features"""
        # This is a simplified version - in practice, you'd use the actual segmentation head
        B, C, H, W = features.shape
        
        # Simple feature-based prediction update
        feature_mean = torch.mean(features, dim=1)  # [B, H, W]
        feature_norm = torch.norm(feature_mean, dim=(1, 2), keepdim=True)
        normalized_features = feature_mean / (feature_norm + 1e-8)
        
        # Blend with current prediction
        updated_prediction = 0.7 * current_prediction + 0.3 * torch.sigmoid(normalized_features)
        
        return updated_prediction