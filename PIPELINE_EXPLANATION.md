# Medical Image Segmentation Pipeline Explanation

## Overview
This is a **Few-Shot Medical Image Segmentation** system that uses **Self-Supervised Learning (SSL)** with **Attention Mechanisms** for organ segmentation in abdominal CT and MRI images. The pipeline is based on PANet architecture with several enhancements including attention modules and contrastive learning.

## Complete Pipeline Flow

### 1. Data Preprocessing Pipeline

#### **Step 1: Data Download and Initial Setup**
- **Abdominal MRI (CHAOST2)**: Download from CHAOS Challenge
- **Abdominal CT (SABS)**: Download from Synapse Multi-atlas dataset

#### **Step 2: Data Format Conversion**
```
Raw Data → NIfTI Format → Normalized Images → Processed Dataset
```

**For CHAOST2 (MRI):**
1. Convert DICOM to NIfTI: `./data/CHAOST2/dcm_img_to_nii.sh`
2. Convert PNG ground truth to NIfTI: `./data/CHAOST2/png_gth_to_nii.ipynb`
3. Normalize images: `./data/CHAOST2/image_normalize.ipynb`

**For SABS (CT):**
1. Apply intensity windowing: `./data/SABS/intensity_normalization.ipynb`
2. Crop and resample: `./data/SABS/resampling_and_roi.ipynb`

#### **Step 3: Index Generation**
- Build class-slice indexing: `./data/<DATASET>/class_slice_index_gen.ipynb`
- Creates mapping between slices and anatomical structures

#### **Step 4: Pseudolabel Generation (SSL Component)**
- Generate superpixel-based pseudolabels: `./data/pseudolabel_gen.ipynb`
- Uses Felzenszwalb algorithm to create superpixel segmentations
- Creates foreground masks for self-supervised learning

### 2. Model Architecture

#### **Backbone Network**
```
Input Image (3 channels) 
    ↓
ResNet-101 Encoder (from DeepLabV3)
    ↓
Features (2048 dims → 256 dims via localconv)
    ↓
[Optional: SSL Attention Processing]
    ↓
Prototype-based Few-Shot Classifier
```

#### **Key Components:**

**A. Backbone (`models/backbone/torchvision_backbones.py`)**
- **TVDeeplabRes101Encoder**: ResNet-101 from torchvision DeepLabV3
- Extracts 256-dimensional features from input images
- Optional ASPP (Atrous Spatial Pyramid Pooling) - disabled by default

**B. Few-Shot Segmentation (`models/grid_proto_fewshot.py`)**
- **FewShotSeg**: Main model class
- **Prototype Learning**: Creates prototypes from support images
- **Grid-based Prototypes**: Uses spatial grid for local prototypes
- **Alignment Loss**: Ensures prototype consistency

**C. SSL Attention Modules (`models/ssl_attention.py`)**
- **MultiHeadAttention**: Core attention mechanism
- **SelfAttentionBlock**: Self-attention with residual connections
- **CrossAttentionBlock**: Cross-attention between features
- **SSLAttentionModule**: Combines self and cross attention
- **FeatureMaskAttention**: Mask-aware attention for foreground focus

### 3. Training Pipeline (`training.py`)

#### **Data Loading Flow**
```
SuperpixelDataset 
    ↓
Data Augmentation (myaug.augs)
    ↓
Few-Shot Episode Creation
    ↓
Support & Query Image Pairs
    ↓
PyTorch DataLoader
```

#### **Training Episode Structure**
Each training episode contains:
- **Support Images**: 1-shot examples with masks
- **Query Images**: Images to segment
- **Class ID**: Target anatomical structure
- **Superpixel Masks**: For self-supervised learning

#### **Forward Pass Flow**
```python
# 1. Load episode data
support_images, support_masks, query_images, query_labels = batch

# 2. Extract features
support_features = encoder(support_images)
query_features = encoder(query_images)

# 3. [Optional] Apply SSL Attention
if use_ssl_attention:
    enhanced_features = ssl_attention(support_features)

# 4. Create prototypes
prototypes = create_prototypes(enhanced_features, support_masks)

# 5. Classify query
predictions = classify_with_prototypes(query_features, prototypes)

# 6. Compute losses
segmentation_loss = cross_entropy(predictions, query_labels)
alignment_loss = prototype_alignment_loss(prototypes)
contrastive_loss = ssl_contrastive_loss(features)

total_loss = segmentation_loss + alignment_loss + contrastive_loss
```

#### **Loss Components**
1. **Segmentation Loss**: Cross-entropy for pixel classification
2. **Alignment Loss**: Ensures prototype consistency
3. **Contrastive Loss**: Self-supervised learning objective
4. **Weighted Combination**: `λ₁ × seg_loss + λ₂ × align_loss + λ₃ × contrast_loss`

### 4. Self-Supervised Learning Components

#### **SSL Attention Pipeline**
```
Online Encoder Features ←→ Target Encoder Features
        ↓                           ↓
   Self-Attention              Self-Attention
        ↓                           ↓
        ↓────── Cross-Attention ────→↓
        ↓                           ↓
   Enhanced Online            Enhanced Target
     Features                  Features
        ↓                           ↓
    Contrastive Learning Loss
```

#### **Attention Mechanisms**
1. **Self-Attention**: Captures spatial relationships within features
2. **Cross-Attention**: Enables knowledge transfer between encoders
3. **Mask-Aware Attention**: Focuses on foreground regions

#### **Iterative Hard Mining (Optional)**
- Identifies difficult samples during training
- Applies additional attention refinement
- Improves learning on challenging cases

### 5. Validation Pipeline (`validation.py`)

#### **Validation Flow**
```
Pre-trained Model
    ↓
Load Support Images (from training set)
    ↓
Load Query Images (from test set)
    ↓
Extract Support Prototypes
    ↓
Segment Query Images
    ↓
Compute Metrics (Dice, IoU)
```

#### **3D Volume Processing**
- Processes 3D medical volumes slice-by-slice
- Maintains spatial consistency across slices
- Reconstructs 3D segmentation from 2D predictions

### 6. Configuration System (`config_ssl_upload.py`)

#### **Key Configuration Categories**

**A. Model Configuration**
```python
model = {
    'which_model': 'dlfcn_res101',          # Backbone architecture
    'cls_name': 'grid_proto',               # Classifier type
    'use_ssl_attention': True,              # Enable attention
    'ssl_attention_heads': 4,               # Number of heads
    'ssl_attention_layers': 1,              # Number of layers
}
```

**B. Training Configuration**
```python
training = {
    'n_steps': 100100,                      # Training iterations
    'batch_size': 1,                        # Batch size
    'lr': 5e-4,                            # Learning rate
    'max_iters_per_load': 500,             # Dataset reload frequency
}
```

**C. Task Configuration**
```python
task = {
    'n_ways': 1,                           # Number of classes per episode
    'n_shots': 1,                          # Support examples per class
    'n_queries': 1,                        # Query examples per episode
}
```

### 7. Data Flow Summary

#### **Training Data Flow**
```
Raw Medical Images
    ↓ (Preprocessing)
NIfTI Normalized Images
    ↓ (Superpixel Generation)
Pseudolabels + Superpixel Masks
    ↓ (Episode Sampling)
Support-Query Pairs
    ↓ (Model Forward)
Feature Extraction → Attention → Prototypes → Classification
    ↓ (Loss Computation)
Segmentation + Alignment + Contrastive Losses
    ↓ (Backpropagation)
Model Parameter Updates
```

#### **Validation Data Flow**
```
Test Medical Images
    ↓ (Support Set Selection)
Few-Shot Support Examples
    ↓ (Prototype Creation)
Support Prototypes
    ↓ (Query Processing)
Query Feature Extraction
    ↓ (Classification)
Prototype-based Segmentation
    ↓ (Evaluation)
Dice Score, IoU Metrics
```

### 8. Key Innovation Points

#### **A. SSL Attention Integration**
- Enhances feature learning through self and cross attention
- Improves prototype quality for few-shot learning
- Enables better knowledge transfer between online/target encoders

#### **B. Superpixel-based SSL**
- Uses superpixel pseudolabels for self-supervised pretraining
- Reduces annotation requirements
- Improves generalization to new anatomical structures

#### **C. Grid-based Prototypes**
- Spatial grid prototypes capture local patterns
- More robust than global prototypes
- Better handling of anatomical variations

#### **D. Contrastive Learning**
- Symmetric negative pair sampling
- Temperature-scaled similarity
- Improves feature discrimination

### 9. Directory Structure
```
work/
├── training.py              # Main training script
├── validation.py            # Validation/testing script
├── config_ssl_upload.py     # Configuration file
├── models/
│   ├── grid_proto_fewshot.py    # Main model
│   ├── ssl_attention.py         # Attention modules
│   ├── contrastive.py           # Contrastive loss
│   └── backbone/                # Encoder architectures
├── dataloaders/
│   ├── GenericSuperDatasetv2.py # Superpixel dataset
│   ├── dev_customized_med.py    # Medical data utilities
│   └── augutils.py              # Data augmentation
└── data/
    ├── SABS/                    # CT dataset processing
    ├── CHAOST2/                 # MRI dataset processing
    └── pseudolabel_gen.ipynb    # SSL pseudolabel generation
```

### 10. Running the Pipeline

#### **Training**
```bash
python training.py with config_ssl_upload
```

#### **Validation**
```bash
python validation.py with config_ssl_upload reload_model_path=path/to/model.pth
```

#### **Key Features**
- M1 Mac compatibility
- GPU/CPU automatic detection
- Sacred experiment tracking
- Configurable attention mechanisms
- Multiple dataset support (SABS, CHAOST2)

This pipeline provides a complete end-to-end solution for few-shot medical image segmentation with self-supervised learning and attention mechanisms, making it particularly suitable for scenarios with limited annotated medical data.