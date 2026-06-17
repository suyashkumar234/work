<div align="center">

# Few-Shot Medical Image Segmentation
### Self-Supervised Learning with Superpixels — ISI Kolkata Research Internship

</div>

---

## Overview

This repository contains research code for **few-shot medical image segmentation** using self-supervised learning (SSL), developed during a research internship at the **Indian Statistical Institute, Kolkata** under **Prof. Umapada Pal** and **Dr. Siladittya Manna**.

The project is based on [Self-supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](https://arxiv.org/abs/2007.09886) (PANet/CoWPro), and progressively extends it with attention mechanisms, data augmentation, knowledge distillation, and hard example mining.

**Datasets:** CHAOS (MRI) and SABS (CT abdominal)  
**Backbone:** ResNet101  
**Framework:** PyTorch + Sacred (experiment tracking)

---

## Branch Summary

| Branch | Strategy | Status | Key Innovation |
|---|---|---|---|
| `master` | Base PANet + M1 compatibility | ✅ Stable | Device-agnostic training |
| `ssl-online-target` | Pure SSL with online/target encoders | ✅ Production-ready | Conv projector, no annotations |
| `attention-ssl-online-target` | SSL + attention-guided reconstruction | 🟡 Research | Multi-head cross-attention |
| `aug-ssl-online-target` | SSL + dual augmentation + advanced negative pairs | 🟡 Experimental | Background-background pairing |
| `aug2-ssl-online-target` | Refined augmentation iteration | 🟡 Experimental | Augmentation schedule tuning |
| `clean-aug-ssl-online-target` | Production-clean augmentation | 🟡 Research | Cleaned dual aug pipeline |
| `student_teacher_experiment` | Knowledge distillation (contrastive) | 🟡 Research | Teacher-student + t-SNE analysis |
| `supervised_teacher_student` | Supervised knowledge distillation | 🟡 Research | Class-aware contrastive pairs |
| `iterative-hard-mining` | Curriculum learning via hard examples | 🟠 Early-stage | Difficulty-aware sample selection |
| `ran-ssl` | Recurrent attention + SSL | 🟡 Testing | Sequential attention mechanisms |

---

## Development Timeline

```
2023
└─ Foundation: PANet base + CHAOS/SABS data pipelines (sadimanna, cheng-01037)

Jul 2025
├─ master:                  M1 Mac compatibility, device-agnostic fixes
├─ ssl-online-target:       ResNet101 upgrade, true SSL (no annotations)
├─ student_teacher_*:       Contrastive knowledge distillation integration
└─ aug-ssl-online-target:   Dual augmentation + advanced negative pair strategies

Aug 2025
├─ attention-ssl-ot:        Multi-head attention, mask-guided feature enhancement
├─ aug-ssl:                 Background-background pairing innovation
├─ ran-ssl:                 Recurrent attention experiments + cleanup
└─ master:                  Convolutional projector finalization
```

---

## Branch 1: `master` — Stable Baseline

The production baseline. Core PANet/CoWPro implementation adapted for abdominal medical imaging with full M1 Mac (MPS) compatibility.

**Key components:**
- Grid Proto networks for few-shot support/query matching
- Device-agnostic training: `CUDA / MPS (Apple Silicon) / CPU`
- Data pipeline: DICOM → NIfTI conversion, CHAOS MRI + SABS CT support
- Contrastive loss with convolutional projector
- Visualization and debug infrastructure

**Latest additions (Aug 2025):** convolutional projector for improved feature transformation, updated SLURM training scripts.

---

## Branch 2: `ssl-online-target` — Core SSL Implementation

The primary SSL branch. Implements true self-supervised training using superpixel-based pseudo-labels — **no manual annotations required**.

### Architecture

```
Support Image                    Query Image
      │                                │
  Encoder (ResNet101)           Encoder (ResNet101)
  [Online]                      [Online]
      │                                │
  Conv Projector                Conv Projector
      │                                │
  Superpixel masks ──────────── Alignment Loss
      │
  Momentum update → Target Encoder
```

### Progress timeline

| Date | Commit | Change |
|---|---|---|
| Jul 30 | `48443d49` | ResNet50 → ResNet101; removed organ class IDs for pure SSL |
| Aug 9 | `e2b6bdac` | New training scripts, pipeline refinement |
| Aug 22 | `0d9f4312` | Updated SLURM test scripts |
| Aug 23 | `084b92cb` | Convolutional projector added |
| Aug 23 | `d43c7e50` | Final file uploads |

### Key innovations
- **Online/Target encoder separation:** target encoder updated via momentum (no gradient)
- **Binary superpixel masks:** foreground/background separation without annotations
- **Conv projector:** replaces linear projection for richer feature transformation
- **Pure SSL:** no organ class labels used during training

---

## Branch 3: `attention-ssl-online-target` — Attention-Guided SSL

Extends `ssl-online-target` with attention mechanisms for better feature alignment between teacher and student.

### New modules

**SSLAttentionModule** — self-attention + cross-attention between teacher/student features:
```
Teacher features  ──┐
                    ├─ Cross-Attention (4 heads) ─→ refined student features
Student features  ──┘
```

**FeatureMaskAttention** — foreground-aware feature enhancement:
```
Features + Binary mask (threshold=0.5)
         │
    Mask-guided attention weighting
         │
    Enhanced foreground features
```

### Loss composition
```
Total Loss = Query Loss (×1.0) + Align Loss (×1.0) + Contrastive Loss (×1.0)
```

### Configuration
- Attention heads: 4
- Dropout: 0.1
- Single attention layer (lightweight)
- Device-agnostic (MPS/CUDA/CPU)

---

## Branch 4: `aug-ssl-online-target` — Dual Augmentation + Advanced Negative Pairs

The most extensively developed experimental branch. Combines SSL with dual-view augmentation and progressively more sophisticated contrastive pair strategies.

### Development phases

**Phase 1 — Base augmentation (Jul 7–13):**
- M1 Mac optimizations ported
- Debugging infrastructure added
- Standard augmentation pipeline

**Phase 2 — Supervised contrastive learning (Jul 17):**
```
Commit 3d16817d: organ-specific feature extraction + visualization
Commit cefc2fe7: class ID + pixel negative pairs using student background
```
Class-aware contrastive pairs: same-organ features as positives, cross-organ as negatives.

**Phase 3 — Background-background pairing (Jul 21):**
```
Commit 64691ef4: background-background negative pairing
```
Key innovation — adds background region pairs as additional negatives, improving feature separation:
```
Positive pairs:  foreground(A) ↔ foreground(B)   [same organ]
Negative pairs:  foreground(A) ↔ background(B)   [standard]
                 background(A) ↔ background(B)    [NEW — improves background separation]
```

**Phase 4 — Clean dual augmentation (Aug 4–5):**
- Two augmented views generated per sample
- Contrastive loss between views
- Repository cleanup (removed training_aug.py, test artifacts, SLURM temp files)

### Loss evolution
```
v1: SSL alignment loss
v2: + supervised contrastive (class-aware)
v3: + multi-negative pairing (bg-bg)
v4: + dual augmentation views
```

---

## Branch 5: `student_teacher_experiment` — Knowledge Distillation

Implements a full teacher-student framework with contrastive learning and extensive visualization.

### Architecture
```
Input
  │
  ├─ Teacher Encoder (frozen/momentum)
  │   └─ Strong feature representations
  │
  └─ Student Encoder (trained)
      └─ Learns to match teacher distribution
            │
      Contrastive Projector
            │
      InfoNCE Loss + Cross-entropy
```

### Progress timeline

| Date | Commit | Change |
|---|---|---|
| Jul 7 | `53f8dc44` | M1 Mac optimizations + debug infrastructure |
| Jul 9 | `fb8ca5f5` | Support mask structure fixes, normalization handling |
| Jul 13 | `68ef3e84` | Full contrastive learning + visualization pipeline |

### Visualization & analysis tools (Jul 13)
- t-SNE feature space visualization
- Similarity matrix analysis
- Organ-level feature clustering
- Contrastive pair quality assessment

---

## Branch 6: `supervised_teacher_student` — Supervised Knowledge Distillation

Extends the teacher-student framework with organ class supervision signals.

**Key additions over `student_teacher_experiment`:**
- Class-aware contrastive pairs (organ labels used as supervision)
- Organ-specific feature extraction per class
- Supervised distillation loss alongside unsupervised SSL

---

## Branch 7: `iterative-hard-mining` — Curriculum Learning

Explores difficulty-aware training: the model iteratively identifies the hardest samples and focuses training on them.

**Concept:**
```
Epoch t:
  1. Forward pass all samples → compute per-sample loss
  2. Rank by difficulty (high loss = hard)
  3. Epoch t+1: oversample top-k% hardest samples
  4. Repeat → curriculum gradually focuses on difficult cases
```

Status: Early-stage research. Multiple commits exploring different difficulty metrics and sampling strategies.

---

## Branch 8: `ran-ssl` — Recurrent Attention Network SSL

Experiments with sequential/recurrent attention mechanisms for SSL, exploring temporal dependencies in volumetric medical scans.

**Latest commit (Aug 5):** repository cleanup — removed augmentation scripts, test outputs, SLURM artifacts. Core implementation intact.

---

## Key Technical Themes Across Branches

| Theme | Branches | Technique |
|---|---|---|
| Self-supervised learning | `ssl-*`, `attention-ssl-*`, `aug-ssl-*` | Superpixel masks, online/target encoders |
| Contrastive learning | All SSL branches | InfoNCE, alignment loss, projector heads |
| Knowledge distillation | `student_teacher_*`, `supervised_t_s` | Momentum encoder, teacher-student sync |
| Attention mechanisms | `attention-ssl-*`, `ran-ssl` | Multi-head, cross-attention, mask-guided |
| Hard example mining | `iterative-hard-mining` | Curriculum learning, difficulty ranking |
| Data augmentation | `aug-ssl-*` | Dual views, background pairing |

---

## Repository Structure

```
.
├── train.py / training.py      # Main training loop (varies by branch)
├── models/
│   ├── grid_proto_maml.py      # Core PANet/CoWPro model
│   ├── ssl_modules.py          # SSL encoder, projector, attention modules
│   └── teacher_student.py      # Knowledge distillation framework
├── dataloaders/
│   ├── chaos.py                # CHAOS MRI dataset
│   └── sabs.py                 # SABS CT dataset
├── utils/
│   ├── dicom_to_nifti.py       # Data conversion utilities
│   └── visualization.py        # t-SNE, similarity matrices
├── scripts/
│   └── *.sh                    # SLURM training scripts
└── notebooks/
    └── *.ipynb                 # Data preprocessing, analysis
```

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
# Key deps: torch, sacred, nibabel, scikit-image, monai
```

### Training — SSL baseline

```bash
python train.py with dataset=CHAOS backbone=resnet101 ssl=True
```

### Training — Attention SSL

```bash
git checkout attention-ssl-online-target
python train.py with dataset=CHAOS backbone=resnet101 ssl=True attention=True
```

### Training — Augmentation SSL

```bash
git checkout aug-ssl-online-target
python train.py with dataset=SABS backbone=resnet101 dual_aug=True
```

---

## References

[1] Ouyang et al. "Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation." *ECCV 2020*. [arxiv:2007.09886](https://arxiv.org/abs/2007.09886)

[2] Wang et al. "PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment." *ICCV 2019*.

---

## Acknowledgements

This work was carried out under the supervision of **Prof. Umapada Pal** and **Dr. Siladittya Manna** at the Indian Statistical Institute, Kolkata, as part of a research internship from IIT (BHU) Varanasi (Feb–Sep 2025).
