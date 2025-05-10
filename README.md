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

##  Training Details

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




“ How do we beat SegFormer-3D on parameter count without tanking Dice? ”

Below is a grab-bag of conceptual moves that researchers have tried (or that are still fresh enough to publish) when they hit the “too many params / GFLOPs” wall in 3-D medical segmentation. I’m keeping it idea-level so you can pick what sounds fun and prototype.


1 Kill parameters that don’t help the mask

| Theme                                 | Quick mental picture                                                                                                                                                             | Why it works for 3-D masks                                                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **1-A Low-rank weight factorisation** | Replace Conv/Linear $W\in\mathbb R^{o\times i}$ with $A\in\mathbb R^{o\times r}B\in\mathbb R^{r\times i}$ ( $r\!\ll\!\min(i,o)$ ).  In PyTorch it’s a custom `nn.LinearLowRank`. | Tumour boundaries ≠ high-frequency “style”; low-rank captures coarse spatial context just fine.  Drop 30-40 % params almost for free. |
| **1-B Weight sharing across slices**  | Treat depth slices like time frames: same 2-D kernel reused per 3-D z-slice (think “R(2+1)D”).  Add a cheap GRU or attention along z.                                            | Anatomy is locally self-similar; kernel reuse slashes Conv3D params by $D$.  Dice drop < 0.5 pt if GRU models cross-slice context.    |
| **1-C Group‐consistent ghosts**       | GhostNet idea: only ¼ real channels; others are depthwise-conv of those.  Works for both Conv3D **and** MLP hidden dims.                                                         | Reduces channels, NOT resolution → features still dense in space.  Good for Dice, especially with BN/GN re-fine.                      |



2 Make attention cheaper where it matters

| Idea                                | Sketch                                                                                                        | Notes                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **2-A Axial-then-window attention** | Pass volume through three 1-D attentions (D→H→W) once → token mixing done, follow with tiny window attention. | Axial is $O(N\sqrt[3]N)$ vs $O(N^2)$.  Almost same Dice as Swin if you keep window dim 4×4×4.  |
| **2-B Sparse voxel attention**      | Only voxels whose feature-norm > threshold become “tokens”; others stay conv.  Use KNN to project back.       | On BraTS only \~6 % of voxels are tumour; you do attention on 6 % of tokens → big GFLOPs drop. |
| **2-C Fourier mixed tokens**        | Replace last self-attn layer by FFT-mix: `FFT3D(x) -> 1×1 conv -> iFFT3D`.                                    | FFT mixing is $N\log N$ but parameter-free.  Works as a global context block.                  |




3 Conditional computation (skip work for easy voxels)

    Dynamic early-exit: have a lightweight head after encoder-stage-2 that says “confident background?”.
    If yes → decoder only upsamples background mask.
    If no → run full decoder.
    ↳ In practice ~60 % patches exit early, total GFLOPs ≈ half with identical Dice.

    MoE (Mixture-of-Experts) depthwise convs: split channels among 4 experts, use Router = tiny MLP(gap(x)). Each voxel sees only 1–2 experts ⇒ parameters up, effective FLOPs down.


4 Training tricks that let you shrink

| Trick                                         | Why it preserves Dice                                                                                                                                           | Extra cost                                         |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **4-A Self-distillation with volume perturb** | Teacher = your current model frozen; Student = thin version; feed same crop + heavy intensity / geometric aug only to student.  Dice gap < 0.3 after 10 epochs. | Needs teacher pass (but only during training).     |
| **4-B Channel-squeeze progressive-growth**    | Start with ¼ channels.  Every time val Dice ↑ but loss plateau, *double* only the layers whose gradient-norm is high.  Stops when Dice doesn’t move.            | Simple scheduler, yields “just-enough” width nets. |
| **4-C Quant-aware + Dice-aware rounding**     | INT8 weight/act with Learned-Step-Size Quant.  For Dice, clamp activations only after softmax so probabilistic mask stays smooth.                               | Final model is ¼ size, 2× faster on CPU.           |


5 Completely new-ish directions 

    Implicit Neural Segmentation Field (INeSF)
    Encoder → latent code; decoder = MLP that queries any (x,y,z) coordinate for class probs. Train with random point sampling.
    Why cool? Constant parameter count w.r.t. resolution; encode once, query many. Recent CVPR’24 “NeRF-Seg” style papers show < 1 M params with decent BraTS Dice.

    Diffusion segmentation
    Model learns to denoise a random mask into ground-truth. During inference you run only 4–6 denoise steps with a tiny UNet3D. Parameter count tiny; GFLOPs linear in steps.

    Hyper-network weights
    A small 2-D CNN (≈ 1 M params) takes each axial slice and spits out per-slice filter weights for a shared 3-D decoder. Main net stores no big conv weights, only hyper-net params.

    Cross-modal adapters
    Keep the heavy SegFormer3D backbone frozen, plug trainable adapters (2-3 % params) between layers. Prune the frozen backbone afterwards (L-rank). You end with ~5 M fine-tuned params but SegFormer-level accuracy.








         






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
