# Pipeline Summary - Quick Reference

## What This Code Does
This is a **Few-Shot Medical Image Segmentation** system that can learn to segment new organs with just 1 example (1-shot learning). It uses:
- **Self-Supervised Learning (SSL)** to learn from unlabeled data
- **Attention mechanisms** to focus on important features  
- **Prototype-based classification** for few-shot learning

## Key Files to Understand the Pipeline

### 1. Main Pipeline Files
- **`PIPELINE_EXPLANATION.md`** - Complete detailed explanation (READ THIS FIRST)
- **`PIPELINE_VISUAL_DIAGRAM.md`** - Visual flowcharts and diagrams
- **`training.py`** - Main training script
- **`validation.py`** - Testing/evaluation script
- **`config_ssl_upload.py`** - All configuration settings

### 2. Model Architecture
- **`models/grid_proto_fewshot.py`** - Main few-shot segmentation model
- **`models/ssl_attention.py`** - Self-supervised attention modules
- **`models/backbone/torchvision_backbones.py`** - ResNet-101 encoder

### 3. Data Processing
- **`dataloaders/GenericSuperDatasetv2.py`** - Superpixel dataset creation
- **`dataloaders/dev_customized_med.py`** - Medical data utilities
- **Data preprocessing notebooks** in repository snippets

## 3-Step Pipeline Overview

### Step 1: Data Preprocessing
```bash
# Convert raw medical images → normalized format → superpixel pseudolabels
Raw DICOM/NIfTI → Intensity Normalization → Resampling → Superpixel Generation
```

### Step 2: Training  
```bash
python training.py  # Few-shot episodes with SSL attention and contrastive learning
```

### Step 3: Validation
```bash
python validation.py with reload_model_path=model.pth  # Test on new organs
```

## Key Innovation: SSL + Attention + Few-Shot
1. **Self-Supervised Learning**: Uses superpixel pseudolabels to learn without manual annotations
2. **Attention Mechanisms**: Self-attention + cross-attention between online/target encoders  
3. **Few-Shot Learning**: Learns new organs with just 1 support example
4. **Medical Focus**: Specialized for abdominal CT/MRI organ segmentation

## Quick Start
1. Read `PIPELINE_EXPLANATION.md` for complete understanding
2. Check `config_ssl_upload.py` for settings  
3. Run preprocessing notebooks for your dataset
4. Train with `python training.py`
5. Validate with `python validation.py`

## Datasets Supported
- **SABS**: Abdominal CT scans (13 organs)
- **CHAOST2**: Abdominal MRI scans (4 organs)

Both datasets support few-shot learning scenarios where you train on some organs and test on completely different organs.