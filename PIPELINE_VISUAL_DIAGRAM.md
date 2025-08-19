# Visual Pipeline Flow Diagram

## Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA PREPROCESSING                        │
├─────────────────────────────────────────────────────────────────────┤
│ Raw Data (DICOM/NIfTI)                                             │
│       ↓                                                             │
│ 1. Format Conversion (dcm_img_to_nii.sh, png_gth_to_nii.ipynb)     │
│       ↓                                                             │
│ 2. Intensity Normalization (intensity_normalization.ipynb)         │
│       ↓                                                             │
│ 3. Resampling & ROI (resampling_and_roi.ipynb)                     │
│       ↓                                                             │
│ 4. Class-Slice Indexing (class_slice_index_gen.ipynb)              │
│       ↓                                                             │
│ 5. Superpixel Pseudolabels (pseudolabel_gen.ipynb)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │  Support Images │    │  Query Images   │                        │
│  │   (1-shot)      │    │   (unlabeled)   │                        │
│  └─────────────────┘    └─────────────────┘                        │
│           ↓                        ↓                                │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              BACKBONE ENCODER                       │            │
│  │           (ResNet-101 DeepLabV3)                    │            │
│  │        Input: 3×256×256 → Output: 256×32×32        │            │
│  └─────────────────────────────────────────────────────┘            │
│           ↓                        ↓                                │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │ Support Features│    │ Query Features  │                        │
│  │   256×32×32     │    │   256×32×32     │                        │
│  └─────────────────┘    └─────────────────┘                        │
│           ↓                        ↓                                │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              SSL ATTENTION MODULE                   │            │
│  │  ┌──────────────┐  ┌──────────────┐                │            │
│  │  │Self-Attention│  │Cross-Attention│                │            │
│  │  │    (4 heads) │  │   (Online↔Target)             │            │
│  │  └──────────────┘  └──────────────┘                │            │
│  └─────────────────────────────────────────────────────┘            │
│           ↓                        ↓                                │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │Enhanced Support │    │Enhanced Query   │                        │
│  │   Features      │    │   Features      │                        │
│  └─────────────────┘    └─────────────────┘                        │
│           ↓                        ↓                                │
│  ┌─────────────────────────────────────────────────────┐            │
│  │            PROTOTYPE CREATION                       │            │
│  │  Grid-based Prototypes from Support Features       │            │
│  │  Foreground + Background Prototypes (8×8 grid)     │            │
│  └─────────────────────────────────────────────────────┘            │
│           ↓                        ↓                                │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              CLASSIFICATION                         │            │
│  │  Cosine Similarity: Query Features ↔ Prototypes    │            │
│  │  Output: Segmentation Predictions                   │            │
│  └─────────────────────────────────────────────────────┘            │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────┐            │
│  │                LOSS COMPUTATION                     │            │
│  │  • Segmentation Loss (Cross-entropy)               │            │
│  │  • Alignment Loss (Prototype consistency)          │            │
│  │  • Contrastive Loss (SSL objective)                │            │
│  │  Total = λ₁×seg + λ₂×align + λ₃×contrast           │            │
│  └─────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          VALIDATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                                │
│  │ Pre-trained     │                                                │
│  │ Model           │                                                │
│  └─────────────────┘                                                │
│           ↓                                                         │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │Support Examples │    │Test Query Images│                        │
│  │(from train set) │    │(from test set)  │                        │
│  └─────────────────┘    └─────────────────┘                        │
│           ↓                        ↓                                │
│  ┌─────────────────────────────────────────────────────┐            │
│  │            3D VOLUME PROCESSING                     │            │
│  │  • Process slice-by-slice                          │            │
│  │  • Maintain spatial consistency                    │            │
│  │  • Reconstruct 3D segmentation                     │            │
│  └─────────────────────────────────────────────────────┘            │
│           ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              EVALUATION METRICS                     │            │
│  │  • Dice Score                                       │            │
│  │  • Intersection over Union (IoU)                   │            │
│  │  • Sensitivity, Specificity                        │            │
│  └─────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Data Structures Flow

```
Episode Structure:
┌─────────────────────────────────────────────────────┐
│ sample_batched = {                                  │
│   'class_id': [organ_id],                          │
│   'support_images': [[tensor([C,H,W])]],           │
│   'support_fg_mask': [[tensor([H,W])]],            │
│   'support_bg_mask': [[tensor([H,W])]],            │
│   'query_images': [tensor([C,H,W])],               │
│   'query_labels': [tensor([H,W])],                 │
│   'mean': [normalization_mean],                    │
│   'std': [normalization_std]                       │
│ }                                                  │
└─────────────────────────────────────────────────────┘
```

## Attention Module Detail

```
SSL Attention Processing:
┌─────────────────────────────────────────────────────┐
│  Input Features: [B, 256, H, W]                    │
│         ↓                                           │
│  Reshape: [B, H×W, 256] (sequence format)          │
│         ↓                                           │
│  ┌─────────────────────────────────────────────┐    │
│  │           SELF-ATTENTION                    │    │
│  │  Q = Linear(features)                       │    │
│  │  K = Linear(features)                       │    │
│  │  V = Linear(features)                       │    │
│  │  Attention = Softmax(QK^T/√d_k)            │    │
│  │  Output = Attention × V                     │    │
│  └─────────────────────────────────────────────┘    │
│         ↓                                           │
│  ┌─────────────────────────────────────────────┐    │
│  │          CROSS-ATTENTION                    │    │
│  │  Q_online = Linear(online_features)         │    │
│  │  K_target = Linear(target_features)         │    │
│  │  V_target = Linear(target_features)         │    │
│  │  Cross_Attn = Softmax(Q_online×K_target^T)  │    │
│  │  Enhanced = Cross_Attn × V_target           │    │
│  └─────────────────────────────────────────────┘    │
│         ↓                                           │
│  Reshape: [B, 256, H, W] (back to spatial)         │
│         ↓                                           │
│  Output: Enhanced Features                          │
└─────────────────────────────────────────────────────┘
```

## Configuration Hierarchy

```
config_ssl_upload.py
├── Model Config
│   ├── backbone: 'dlfcn_res101'
│   ├── classifier: 'grid_proto'  
│   ├── use_ssl_attention: True/False
│   ├── ssl_attention_heads: 4
│   └── ssl_attention_layers: 1
├── Training Config  
│   ├── n_steps: 100100
│   ├── batch_size: 1
│   ├── lr: 5e-4
│   └── max_iters_per_load: 500
├── Task Config
│   ├── n_ways: 1 (num classes per episode)
│   ├── n_shots: 1 (support examples)
│   └── n_queries: 1 (query examples)
└── Dataset Paths
    ├── SABS: CT data path
    └── CHAOST2: MRI data path
```