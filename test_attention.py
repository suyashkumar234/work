"""
Test script for SSL Attention Module
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ssl_attention import SSLAttentionModule

def test_ssl_attention():
    """Test the SSL Attention Module"""
    
    print("Testing SSL Attention Module...")
    
    # Test parameters
    batch_size = 1
    channels = 512  # ResNet101 feature dimension
    height, width = 32, 32  # Feature map size
    
    # Create test data
    online_features = torch.randn(batch_size, channels, height, width)
    target_features = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shapes:")
    print(f"  Online features: {online_features.shape}")
    print(f"  Target features: {target_features.shape}")
    
    # Initialize attention module
    attention_module = SSLAttentionModule(
        feature_dim=channels,
        n_heads=8,
        dropout=0.1,
        n_layers=1
    )
    
    print(f"\nAttention module parameters: {sum(p.numel() for p in attention_module.parameters())}")
    
    # Test forward pass
    try:
        enhanced_online, enhanced_target, attention_weights = attention_module(
            online_features, target_features
        )
        
        print(f"\nOutput shapes:")
        print(f"  Enhanced online: {enhanced_online.shape}")
        print(f"  Enhanced target: {enhanced_target.shape}")
        print(f"  Attention weights keys: {list(attention_weights.keys())}")
        
        # Check if shapes are preserved
        assert enhanced_online.shape == online_features.shape, "Online features shape mismatch!"
        assert enhanced_target.shape == target_features.shape, "Target features shape mismatch!"
        
        print("\n✓ SSL Attention Module test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ SSL Attention Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_masks():
    """Test with foreground masks"""
    
    print("\nTesting with foreground masks...")
    
    batch_size = 1
    channels = 512
    height, width = 32, 32
    
    # Create test data
    online_features = torch.randn(batch_size, channels, height, width)
    target_features = torch.randn(batch_size, channels, height, width)
    
    # Create binary masks (simulate foreground regions)
    online_mask = torch.zeros(batch_size, height, width)
    online_mask[:, 10:20, 10:20] = 1  # Small foreground region
    
    target_mask = torch.zeros(batch_size, height, width) 
    target_mask[:, 12:22, 12:22] = 1  # Slightly offset foreground region
    
    # Test FeatureMaskAttention
    from models.ssl_attention import FeatureMaskAttention
    
    mask_attention = FeatureMaskAttention(feature_dim=channels)
    
    try:
        enhanced_online, enhanced_target, attention_weights = mask_attention(
            online_features, target_features, online_mask, target_mask
        )
        
        print(f"  Enhanced online with mask: {enhanced_online.shape}")
        print(f"  Enhanced target with mask: {enhanced_target.shape}")
        
        assert enhanced_online.shape == online_features.shape, "Masked online features shape mismatch!"
        assert enhanced_target.shape == target_features.shape, "Masked target features shape mismatch!"
        
        print("✓ Mask-aware attention test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Mask-aware attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SSL Attention Module Tests")
    print("=" * 50)
    
    # Test basic attention
    basic_test = test_ssl_attention()
    
    # Test with masks
    mask_test = test_with_masks()
    
    print("=" * 50)
    if basic_test and mask_test:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 50)