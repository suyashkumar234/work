#!/usr/bin/env python3
"""
Test script for contrastive loss and visualization integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Import our modules
from models.contrastive import SupervisedPixelWiseContrastiveLoss
from util.organ_visualization import create_organ_visualizer

def test_contrastive_loss():
    """Test the supervised contrastive loss with binary masks and organ classes."""
    print("Testing SupervisedPixelWiseContrastiveLoss...")
    
    # Create test data
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create random features (with gradients)
    feat_teacher = torch.randn(batch_size, channels, height, width, requires_grad=True)
    feat_student = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    # Create binary masks (0 or 1)
    mask_teacher = torch.randint(0, 2, (batch_size, height, width)).float()
    mask_student = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # Create organ class IDs (same organ for both teacher and student)
    organ_class_teacher = torch.tensor([1, 1])  # Same organ class
    organ_class_student = torch.tensor([1, 1])
    
    # Initialize contrastive loss
    contrastive_loss = SupervisedPixelWiseContrastiveLoss(temperature=0.1)
    
    # Test forward pass
    try:
        loss = contrastive_loss(
            feat_teacher, feat_student, 
            mask_teacher, mask_student,
            organ_class_teacher, organ_class_student
        )
        print(f"✅ Contrastive loss computed successfully: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("✅ Backward pass successful")
        
    except Exception as e:
        print(f"❌ Contrastive loss failed: {e}")
        return False
    
    return True

def test_visualization():
    """Test the organ visualization utility."""
    print("\nTesting OrganContrastiveVisualizer...")
    
    # Create test data
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create random features
    teacher_features = torch.randn(batch_size, channels, height, width)
    student_features = torch.randn(batch_size, channels, height, width)
    
    # Create binary masks
    teacher_masks = torch.randint(0, 2, (batch_size, height, width)).float()
    student_masks = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # Create organ classes
    organ_classes = torch.tensor([1, 4])  # Different organ classes
    
    # Create visualizer
    try:
        visualizer = create_organ_visualizer(dataset_name='SABS', save_dir='test_visualizations')
        print("✅ Visualizer created successfully")
        
        # Test t-SNE visualization with proper indexing
        # Reshape features to (B*H*W, C) and masks to (B*H*W,) for indexing
        teacher_features_flat = teacher_features.permute(0, 2, 3, 1).reshape(-1, channels)
        student_features_flat = student_features.permute(0, 2, 3, 1).reshape(-1, channels)
        teacher_masks_flat = teacher_masks.reshape(-1)
        student_masks_flat = student_masks.reshape(-1)
        
        # Expand organ classes to match flattened dimensions
        organ_classes_expanded = organ_classes.repeat_interleave(height * width)
        
        # Test t-SNE visualization
        tsne_file = visualizer.visualize_tsne_features(
            teacher_features_flat, student_features_flat,
            teacher_masks_flat, student_masks_flat,
            organ_classes_expanded,
            title="Test t-SNE Visualization"
        )
        print(f"✅ t-SNE visualization saved: {tsne_file}")
        
        # Test similarity matrix visualization
        sim_file = visualizer.visualize_similarity_matrix(
            teacher_features_flat, student_features_flat,
            teacher_masks_flat, student_masks_flat,
            organ_classes_expanded,
            title="Test Similarity Matrix"
        )
        print(f"✅ Similarity matrix saved: {sim_file}")
        
        # Test comprehensive analysis
        viz_files = visualizer.visualize_contrastive_analysis(
            teacher_features_flat, student_features_flat,
            teacher_masks_flat, student_masks_flat,
            organ_classes_expanded,
            title="Test Comprehensive Analysis"
        )
        print(f"✅ Comprehensive analysis saved: {viz_files}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_integration():
    """Test integration with the model."""
    print("\nTesting model integration...")
    
    try:
        from models.grid_proto_fewshot import FewShotSeg
        
        # Create a simple model instance with proper config
        model = FewShotSeg(
            in_channels=3,
            pretrained_path=None,
            cfg={
                'align': False, 
                'temperature': 0.1, 
                'use_coco_init': False,
                'proto_grid_size': 8,  # Add missing parameter
                'feature_hw': [32, 32],  # Add missing parameter
                'cls_name': 'grid_proto'  # Fixed parameter name
            }
        )
        
        print("✅ Model created successfully")
        
        # Test that contrastive loss is properly initialized
        if hasattr(model, 'contrastive_loss'):
            print("✅ Contrastive loss initialized in model")
        else:
            print("❌ Contrastive loss not found in model")
            return False
            
    except Exception as e:
        print(f"❌ Model integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Contrastive Learning Integration")
    print("=" * 50)
    
    # Test contrastive loss
    contrastive_success = test_contrastive_loss()
    
    # Test visualization
    viz_success = test_visualization()
    
    # Test model integration
    model_success = test_model_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Contrastive Loss: {'✅ PASS' if contrastive_success else '❌ FAIL'}")
    print(f"   Visualization: {'✅ PASS' if viz_success else '❌ FAIL'}")
    print(f"   Model Integration: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if all([contrastive_success, viz_success, model_success]):
        print("\n🎉 All tests passed! Integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    # Clean up test files
    if os.path.exists('test_visualizations'):
        import shutil
        shutil.rmtree('test_visualizations')
        print("🧹 Cleaned up test files")

if __name__ == "__main__":
    main() 