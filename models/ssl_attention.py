"""
Self-supervised learning attention module for online-target encoder feature interaction.
Applies self-attention to both online and target support features, then cross-attention between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, seq_len, d_model] or [B, H, W, d_model]
            key: [B, seq_len, d_model] or [B, H, W, d_model]  
            value: [B, seq_len, d_model] or [B, H, W, d_model]
            mask: Optional attention mask
        """
        batch_size = query.size(0)
        seq_len = query.size(-2)
        
        # Linear transformations and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class SelfAttentionBlock(nn.Module):
    """Self-attention block with residual connection and layer normalization"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, H, W, C] or [B, seq_len, C]
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for interaction between online and target features"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_features, key_value_features, mask=None):
        """
        Args:
            query_features: [B, H, W, C] - features that will be updated
            key_value_features: [B, H, W, C] - features to attend to
        """
        # Cross-attention with residual connection
        attn_output, attn_weights = self.attention(
            query_features, key_value_features, key_value_features, mask
        )
        query_features = self.norm1(query_features + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(query_features)
        query_features = self.norm2(query_features + self.dropout(ff_output))
        
        return query_features, attn_weights


class SSLAttentionModule(nn.Module):
    """
    SSL Attention Module that applies:
    1. Self-attention to online support features
    2. Self-attention to target support features  
    3. Cross-attention between online and target features
    """
    
    def __init__(self, feature_dim, n_heads=8, dropout=0.1, n_layers=1):
        super(SSLAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Self-attention blocks for online and target features
        self.online_self_attention = nn.ModuleList([
            SelfAttentionBlock(feature_dim, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        self.target_self_attention = nn.ModuleList([
            SelfAttentionBlock(feature_dim, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Cross-attention blocks
        self.online_to_target_cross = nn.ModuleList([
            CrossAttentionBlock(feature_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.target_to_online_cross = nn.ModuleList([
            CrossAttentionBlock(feature_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, online_features, target_features, online_mask=None, target_mask=None):
        """
        Apply attention mechanism to support features before contrastive learning.
        
        Args:
            online_features: [B, C, H, W] - online encoder support features (teacher, gradient-updated)
            target_features: [B, C, H, W] - target encoder support features (student, momentum-updated)
            online_mask: Optional mask for online features (teacher)
            target_mask: Optional mask for target features (student)
            
        Returns:
            tuple: (enhanced_online_features, enhanced_target_features, attention_weights)
        """
        B, C, H, W = online_features.shape
        
        # Reshape features for attention: [B, C, H, W] -> [B, H*W, C]
        online_flat = online_features.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        target_flat = target_features.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        
        attention_weights = {}
        
        # Apply self-attention to online features
        for i, self_attn in enumerate(self.online_self_attention):
            online_flat, attn_weights = self_attn(online_flat, online_mask)
            attention_weights[f'online_self_attn_{i}'] = attn_weights
            
        # Apply self-attention to target features  
        for i, self_attn in enumerate(self.target_self_attention):
            target_flat, attn_weights = self_attn(target_flat, target_mask)
            attention_weights[f'target_self_attn_{i}'] = attn_weights
        
        # Apply cross-attention: online attends to target
        for i, cross_attn in enumerate(self.online_to_target_cross):
            online_flat, attn_weights = cross_attn(online_flat, target_flat)
            attention_weights[f'online_to_target_cross_{i}'] = attn_weights
            
        # Apply cross-attention: target attends to online
        for i, cross_attn in enumerate(self.target_to_online_cross):
            target_flat, attn_weights = cross_attn(target_flat, online_flat)
            attention_weights[f'target_to_online_cross_{i}'] = attn_weights
        
        # Reshape back to [B, C, H, W]
        enhanced_online = online_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        enhanced_target = target_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return enhanced_online, enhanced_target, attention_weights


class FeatureMaskAttention(nn.Module):
    """
    Mask-aware attention that uses foreground masks to focus attention on relevant regions
    """
    
    def __init__(self, feature_dim, n_heads=8, dropout=0.1):
        super(FeatureMaskAttention, self).__init__()
        self.attention_module = SSLAttentionModule(feature_dim, n_heads, dropout, n_layers=1)
        
    def create_attention_mask(self, fg_mask, threshold=0.5):
        """
        Create attention mask from foreground mask
        Args:
            fg_mask: [B, H, W] - foreground mask
            threshold: threshold for binary mask
        Returns:
            mask: [B, 1, H*W, H*W] - attention mask
        """
        B, H, W = fg_mask.shape
        # Create binary mask
        binary_mask = (fg_mask > threshold).float()  # [B, H, W]
        binary_mask = binary_mask.view(B, H*W)  # [B, H*W]
        
        # Create attention mask: [B, 1, H*W, H*W]
        # Only allow attention between foreground pixels
        mask = binary_mask.unsqueeze(1) * binary_mask.unsqueeze(2)  # [B, H*W, H*W]
        mask = mask.unsqueeze(1)  # [B, 1, H*W, H*W]
        
        return mask
        
    def forward(self, online_features, target_features, online_fg_mask, target_fg_mask):
        """
        Apply mask-aware attention
        
        Args:
            online_features: [B, C, H, W]
            target_features: [B, C, H, W] 
            online_fg_mask: [B, H, W]
            target_fg_mask: [B, H, W]
        """
        # Create attention masks
        online_mask = self.create_attention_mask(online_fg_mask)
        target_mask = self.create_attention_mask(target_fg_mask)
        
        # Apply attention
        enhanced_online, enhanced_target, attention_weights = self.attention_module(
            online_features, target_features, online_mask, target_mask
        )
        
        return enhanced_online, enhanced_target, attention_weights