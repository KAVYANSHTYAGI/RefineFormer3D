# RefineFormer3D

# 🧠 RefineFormer3D - Brain Tumor Segmentation (BraTS 2017)

<div align="center">
    <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-red" alt="PyTorch Badge">
    <img src="https://img.shields.io/badge/Task-Brain%20Tumor%20Segmentation-green" alt="Task Badge">
</div>

---

## 📚 Overview

This project implements **RefineFormer3D**, a powerful 3D Transformer-based architecture for brain tumor segmentation on the **BraTS 2017 dataset**.

We carefully design a complete training pipeline:
- Proper handling of BraTS 2017 HGG+LGG training data
- RefineFormer3D model (Mix Vision Transformer + Refine Decoder)
- Mixed Precision (AMP) training for faster, memory-efficient training
- Dice, IoU, Hausdorff distance metrics tracking
- Best checkpoint saving based on validation Dice

---

## 🧠 Model Architecture: RefineFormer3D

> **Inspired by SegFormer3D but upgraded for better performance**

### Encoder (Mix Vision Transformer 3D):
- 4 hierarchical stages
- Each stage:  
  - 3D patch embedding (Conv3D)
  - Transformer blocks (Self Attention + MLP)
- Spatial reduction ratios for efficient feature aggregation
- Deeper stages capture high-level context

### Decoder (RefineFormer Head):
- Feature fusion from all 4 encoder stages
- MLP-based linear projections
- 3D convolution for feature fusion
- Upsample to original resolution
- Final Conv3D → multi-class segmentation output

<div align="center">
    <b>Input 3D Volume ➔ Transformer Encoder ➔ Refine Decoder ➔ Tumor Segmentation Output</b>
</div>

---

## 📚 Dataset: BraTS 2017

- **Task**: 3D segmentation of brain tumors
- **Data**:
  - 4 MRI modalities per patient:
    - FLAIR
    - T1
    - T1CE
    - T2
  - Ground truth segmentation mask (`seg.nii.gz`)
- **Training Data**:
  - HGG (High-Grade Glioma) patients
  - LGG (Low-Grade Glioma) patients
- **Validation Data**:
  - Flat structure (no HGG/LGG split)

✅ We **merge** HGG + LGG during training  
✅ Labels are **remapped** into 3 classes:
  - 0 = Background
  - 1 = Tumor Core (NCR + Enhancing)
  - 2 = Edema (Whole Tumor)

---

## ⚙️ Training Details

- **Framework**: PyTorch
- **Mixed Precision Training**: ✅ (`torch.cuda.amp`)
- **Loss**: Custom RefineFormer3D Loss
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Cosine Annealing
- **Metrics**:
  - Dice Coefficient (per-class + mean)
  - IoU (Jaccard Index)
  - Hausdorff Distance

---

## 🛠️ Setup Instructions

```bash
# Step 1: Clone the repository
git clone https://github.com/yourname/refineformer3d-brats.git
cd refineformer3d-brats

# Step 2: Install required packages
pip install torch torchvision torchaudio nibabel tqdm numpy scikit-learn

# Step 3: Prepare dataset
# Put your BraTS 2017 data like this:
# ├── dataset/
#     ├── Brats17TrainingData/
#         ├── HGG/
#         ├── LGG/
#     ├── Brats17ValidationData/

# Step 4: Configure your config.py (paths, device, etc.)

# Step 5: Start training
python train.py
