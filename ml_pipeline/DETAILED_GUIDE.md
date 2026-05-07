# ML Pipeline - Detailed Guide for Intracranial Aneurysm Detection

## Table of Contents
1. [Introduction](#1-introduction)
2. [Medical Background](#2-medical-background)
3. [Pipeline Overview](#3-pipeline-overview)
4. [File-by-File Explanation](#4-file-by-file-explanation)
5. [How to Run Detection](#5-how-to-run-detection)
6. [Understanding the Output](#6-understanding-the-output)
7. [Training Process](#7-training-process)
8. [Technical Deep Dive](#8-technical-deep-dive)
9. [Troubleshooting](#9-troubleshooting)
10. [References](#10-references)

---

## 1. Introduction

### What is This Project?
This is an **automated brain aneurysm detection system** that analyzes 3D CT Angiography (CTA) scans to identify potential intracranial aneurysms. It uses deep learning (specifically, a 3D Convolutional Neural Network) to process medical images and predict aneurysm locations.

### Why is This Important?
- Brain aneurysms affect 3-5% of the population
- Ruptured aneurysms cause life-threatening subarachnoid hemorrhage
- Early detection can save lives through preventive treatment
- Manual analysis is time-consuming and prone to human error

### Model Origin
This code is based on the **7th place solution** from the Kaggle RSNA 2024 competition, which achieved a **public leaderboard score of 0.518** (binary log loss). The original solution uses the nnU-Net framework by MIC-DKFZ.

---

## 2. Medical Background

### What is an Intracranial Aneurysm?
An intracranial aneurysm is a **weak, bulging spot on an artery wall** in the brain. Think of it like a balloon that forms on a weakened section of a garden hose.

### Anatomical Locations (13 Classes)
The model detects aneurysms at 13 specific brain artery locations:

| Code | Location | Description |
|------|----------|-------------|
| R-ICA | Right Internal Carotid Artery | Main artery supplying brain |
| L-ICA | Left Internal Carotid Artery | Main artery supplying brain |
| R-MCA | Right Middle Cerebral Artery | Supplies side of brain |
| L-MCA | Left Middle Cerebral Artery | Supplies side of brain |
| R-ACA | Right Anterior Cerebral Artery | Supplies front of brain |
| L-ACA | Left Anterior Cerebral Artery | Supplies front of brain |
| Acomm | Anterior Communicating Artery | Connects left and right |
| R-Pcomm | Right Posterior Communicating | Connects front and back circulation |
| L-Pcomm | Left Posterior Communicating | Connects front and back circulation |
| R-PCA | Right Posterior Cerebral Artery | Supplies back of brain |
| L-PCA | Left Posterior Cerebral Artery | Supplies back of brain |
| BA | Basilar Artery | Major artery at brain stem |
| 3rd-A2 | Third A2 Segment (rare) | Anatomical variant |

### Circle of Willis
These arteries form the "Circle of Willis" - a circular arrangement of arteries at the base of the brain. This is where most aneurysms occur.

```
                    ┌───────────────────┐
                    │  Anterior Brain   │
                    └───────────────────┘
                            ↑
              R-ACA ○───────●───────○ L-ACA
                   ╱       │       ╲
                  ╱     Acomm      ╲
                 ╱         │         ╲
              R-ICA        │       L-ICA
                │          │          │
            R-Pcomm●       │       ●L-Pcomm
                ╲          │         ╱
                 ╲         │        ╱
              R-PCA ○──────●──────○ L-PCA
                           │
                          BA
                           │
                    └───────────────────┘
                    │  Posterior Brain  │
                    └───────────────────┘
```

---

## 3. Pipeline Overview

### High-Level Flow

```
INPUT                    PROCESSING                     OUTPUT
┌──────────┐    ┌─────────────────────────┐    ┌──────────────────┐
│ DICOM    │───▶│  Deep Learning Model    │───▶│ JSON Results     │
│ Folder   │    │  (61 million params)    │    │ + Probabilities  │
│ (~300    │    │                         │    │                  │
│  slices) │    │  1. Preprocess          │    │ {                │
└──────────┘    │  2. Predict             │    │   "detected":    │
                │  3. Postprocess         │    │      true,       │
                └─────────────────────────┘    │   "prob": 0.85   │
                                               │ }                │
                                               └──────────────────┘
```

### Stage-by-Stage Breakdown

#### Stage 1: Preprocessing
```
📁 DICOM Files (many .dcm files)
         │
         ▼ Read pixel values
    3D Volume (Z × H × W)
         │
         ▼ Clip to [0, 600] HU (blood vessel range)
    Normalized Volume
         │
         ▼ Resample to 0.5mm isotropic
    Standardized Volume (ready for model)
```

#### Stage 2: Model Prediction
```
Input Volume (1 × 64 × 64 × 64)
         │
         ▼ 6 Encoder Stages (extract features)
    Feature Maps (320 × 2 × 2 × 2)
         │
         ▼ 5 Decoder Stages (reconstruct spatial info)
    Output Volume (13 × 64 × 64 × 64)
         │
         ▼ Each channel = probability map for one location
```

#### Stage 3: Postprocessing
```
Probability Maps (13 channels)
         │
         ▼ Take maximum probability per location
    Per-Location Probabilities [0-100%]
         │
         ▼ Apply threshold (50%)
    Binary Detection + JSON Report
```

---

## 4. File-by-File Explanation

### config.py - Configuration Hub

**Purpose**: Central location for all settings so you only need to change one file.

```python
# Key Constants Explained:

# Device selection (GPU preferred for speed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CT value range for blood vessels (Hounsfield Units)
INTENSITY_CLIP_MIN = 0    # Below this is air/fat
INTENSITY_CLIP_MAX = 600  # Above this is bone

# Target spacing (mm) - model trained on this resolution
SPACING = [0.5, 0.5, 0.5]

# Detection threshold - probability above this = aneurysm detected
DETECTION_THRESHOLD = 0.5  # 50%

# Anatomical locations (13 classes)
LOCATION_LABELS = [
    "R-ICA", "L-ICA", "R-MCA", "L-MCA", "R-ACA", "L-ACA",
    "Acomm", "R-Pcomm", "L-Pcomm", "R-PCA", "L-PCA", "BA", "3rd-A2"
]
```

**Why These Values?**
- 0.5mm spacing balances resolution vs memory
- [0, 600] HU range captures contrast-enhanced blood vessels
- 50% threshold balances sensitivity and specificity

---

### 1_preprocessing.py - Data Preparation

**Purpose**: Convert raw DICOM files into a format the neural network can process.

#### Step 1: Load DICOM Files
```python
def load_dicom_series(folder_path):
    """
    DICOM = Digital Imaging and Communications in Medicine
    Each .dcm file = one slice of the CT scan
    """
    # Sort by slice location (Z-axis)
    slices = sorted(dicom_files, key=lambda x: x.ImagePositionPatient[2])
    
    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices])
    
    # Convert to Hounsfield Units (HU)
    # HU = pixel_value * slope + intercept
    volume = volume * slope + intercept
    
    return volume  # Shape: (Z, H, W)
```

#### Step 2: Intensity Normalization
```python
def normalize_intensity(volume):
    """
    Why clip to [0, 600]?
    - Air: -1000 HU (not relevant)
    - Fat: -100 HU (not relevant)
    - Water: 0 HU
    - Blood: 30-45 HU
    - Contrast-enhanced blood: 100-400 HU ← Target
    - Bone: 700-3000 HU (causes artifacts)
    """
    volume = np.clip(volume, 0, 600)
    volume = volume / 600.0  # Scale to [0, 1]
    return volume
```

#### Step 3: Spatial Resampling
```python
def resample_volume(volume, original_spacing, target_spacing=(0.5, 0.5, 0.5)):
    """
    Why resample?
    - Different CT scanners produce different resolutions
    - Model trained on 0.5mm isotropic spacing
    - Must match training data for accurate predictions
    
    Example:
    - Original: 0.4 × 0.4 × 1.0 mm (anisotropic)
    - Target: 0.5 × 0.5 × 0.5 mm (isotropic)
    """
    # Calculate new dimensions
    zoom_factors = original_spacing / target_spacing
    
    # Trilinear interpolation
    resampled = scipy.ndimage.zoom(volume, zoom_factors, order=1)
    
    return resampled
```

---

### 2_model.py - Neural Network Architecture

**Purpose**: Define the deep learning model that "sees" aneurysms in the scans.

#### Architecture: Residual Encoder U-Net

```
This is a specialized 3D version of the famous U-Net architecture,
enhanced with residual connections (like ResNet).

            ┌─────────────────────────────────────────┐
            │           RESIDUAL ENCODER U-NET          │
            │                                           │
            │  INPUT → ENCODER → BOTTLENECK → DECODER → OUTPUT  │
            │                                           │
            └─────────────────────────────────────────┘

Encoder (Feature Extraction):
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 1ch → 32ch    (maintains resolution)                   │
│ Stage 2: 32ch → 64ch   (↓ resolution by 2)                      │
│ Stage 3: 64ch → 128ch  (↓ resolution by 2)                      │
│ Stage 4: 128ch → 256ch (↓ resolution by 2)                      │
│ Stage 5: 256ch → 320ch (↓ resolution by 2)                      │
│ Stage 6: 320ch → 320ch (bottleneck, no downsampling)            │
└─────────────────────────────────────────────────────────────────┘

Decoder (Spatial Reconstruction):
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 320ch → 256ch (↑ resolution by 2, concat skip)         │
│ Stage 2: 256ch → 128ch (↑ resolution by 2, concat skip)         │
│ Stage 3: 128ch → 64ch  (↑ resolution by 2, concat skip)         │
│ Stage 4: 64ch → 32ch   (↑ resolution by 2, concat skip)         │
│ Stage 5: 32ch → 32ch   (↑ resolution by 2, concat skip)         │
│ Final:   32ch → 13ch   (output layer)                           │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Components

##### 1. Residual Block
```python
class ResidualBlock(nn.Module):
    """
    Why residual connections?
    - Solves vanishing gradient problem
    - Allows training very deep networks
    - Output = Input + F(Input)
    
    Visualization:
    
    Input ─────────────────────────────────┐
      │                                    │
      ▼                                    │ (skip connection)
    ┌─────────┐                            │
    │ Conv3D  │                            │
    │ Norm    │                            │
    │ ReLU    │                            │
    └─────────┘                            │
      │                                    │
      ▼                                    │
    ┌─────────┐                            │
    │ Conv3D  │                            │
    │ Norm    │                            │
    └─────────┘                            │
      │                                    │
      ▼                                    │
    ┌───────────────────────────────────────┐
    │              + (addition)             │
    └───────────────────────────────────────┘
      │
      ▼
    Output
    """
```

##### 2. Skip Connections (U-Net Style)
```python
"""
Why skip connections?

Problem: Deep networks lose spatial information
Solution: Connect encoder directly to decoder

    Encoder                    Decoder
    ┌─────┐                   ┌─────┐
    │ E1  │─────────────────▶│ D5  │  (high resolution details)
    └─────┘                   └─────┘
       │                         ↑
       ▼                         │
    ┌─────┐                   ┌─────┐
    │ E2  │─────────────────▶│ D4  │
    └─────┘                   └─────┘
       │                         ↑
       ▼                         │
    ┌─────┐                   ┌─────┐
    │ E3  │─────────────────▶│ D3  │
    └─────┘                   └─────┘
       │                         ↑
       ▼                         │
    ┌─────┐                   ┌─────┐
    │ E4  │─────────────────▶│ D2  │
    └─────┘                   └─────┘
       │                         ↑
       ▼                         │
    ┌─────┐                   ┌─────┐
    │ E5  │─────────────────▶│ D1  │
    └─────┘                   └─────┘
       │                         ↑
       ▼                         │
    ┌─────────────────────────────┐
    │        Bottleneck           │
    └─────────────────────────────┘
"""
```

##### 3. Output Layer
```python
# Final convolution: 32 channels → 13 channels (one per location)
self.output_conv = nn.Conv3d(32, 13, kernel_size=1)

# Each channel is a probability map:
# output[0] = R-ICA probability at each voxel
# output[1] = L-ICA probability at each voxel
# ...
# output[12] = 3rd-A2 probability at each voxel
```

#### Model Statistics
- **Total Parameters**: 61,467,533 (~61 million)
- **Input Shape**: (batch, 1, 64, 64, 64)
- **Output Shape**: (batch, 13, 64, 64, 64)
- **Memory**: ~2.4GB on GPU during inference

---

### 3_inference.py - Sliding Window Prediction

**Purpose**: Handle large 3D volumes that don't fit in GPU memory.

#### The Problem
```
Brain CT Volume:     ~300 × 512 × 512 voxels
GPU Memory:          ~8-24 GB
Single Forward Pass: Would require 50+ GB (impossible!)
```

#### The Solution: Sliding Window
```
Split the large volume into overlapping patches,
process each patch, then combine the results.

    Full Volume
    ┌─────────────────────────────────────────┐
    │  ┌───────┐                              │
    │  │Patch 1│                              │
    │  └───────┘                              │
    │      ┌───────┐                          │
    │      │Patch 2│  (50% overlap)           │
    │      └───────┘                          │
    │          ┌───────┐                      │
    │          │Patch 3│                      │
    │          └───────┘                      │
    │              ...                        │
    └─────────────────────────────────────────┘
```

#### Gaussian Blending
```python
def compute_gaussian_importance(patch_size):
    """
    Why Gaussian blending?
    
    Problem: Edge artifacts when combining patches
    
    Patch 1         Patch 2
    ┌───────┐      ┌───────┐
    │ Good  │Artifact Artifact│ Good  │
    │ here  │  here│  here │ here  │
    └───────┘      └───────┘
    
    Solution: Weight center pixels higher than edges
    
    Gaussian Weight Map:
    ┌─────────────────────────┐
    │  Low    Med    Low      │
    │  Med    HIGH   Med      │  ← Center has highest weight
    │  Low    Med    Low      │
    └─────────────────────────┘
    
    Result: Smooth transitions between patches
    """
```

#### Sliding Window Algorithm
```python
def sliding_window_inference(volume, model, patch_size, overlap):
    """
    1. Create empty output volume (same size as input)
    2. Create weight volume (to track how many times each voxel is predicted)
    3. For each overlapping patch:
       a. Extract patch from volume
       b. Run through model
       c. Multiply by Gaussian weight
       d. Add to output
       e. Add Gaussian weight to weight volume
    4. Divide output by weights (average predictions)
    """
    
    # Calculate step size (50% overlap = step_size = patch_size / 2)
    step_size = patch_size * (1 - overlap)  # e.g., 64 * 0.5 = 32
    
    # Iterate through all positions
    for z in range(0, Z_max, step_size):
        for y in range(0, Y_max, step_size):
            for x in range(0, X_max, step_size):
                patch = volume[z:z+64, y:y+64, x:x+64]
                prediction = model(patch)
                output[z:z+64, y:y+64, x:x+64] += prediction * gaussian_weights
                weights[z:z+64, y:y+64, x:x+64] += gaussian_weights
    
    return output / weights  # Averaged predictions
```

---

### 4_postprocessing.py - Extract Detection Results

**Purpose**: Convert raw model output into human-readable results.

#### From Probabilities to Detections
```python
def extract_detections(prediction_volume, threshold=0.5):
    """
    Input: 13 probability maps (one per location)
    Output: JSON with detection results
    
    Steps:
    1. Apply sigmoid (convert logits to probabilities)
    2. For each location channel:
       - Take maximum probability across all voxels
       - If max > threshold → detection found
    3. Format results
    """
    
    # Apply sigmoid
    probs = torch.sigmoid(prediction_volume)  # [0, 1] range
    
    # Get max probability per location
    for i, location in enumerate(LOCATION_LABELS):
        max_prob = probs[i].max()
        
        if max_prob > threshold:
            detections.append({
                "location": location,
                "probability": max_prob,
                "status": "DETECTED"
            })
```

#### Risk Level Classification
```python
def classify_risk(max_probability):
    """
    Risk levels based on confidence:
    
    | Probability | Risk Level | Recommendation |
    |-------------|------------|----------------|
    | 0-50%       | NEGATIVE   | No aneurysm detected |
    | 50-70%      | LOW        | Monitor, possible false positive |
    | 70-85%      | MEDIUM     | Recommend follow-up imaging |
    | 85-100%     | HIGH       | Urgent review recommended |
    """
    if max_probability < 0.5:
        return "NEGATIVE"
    elif max_probability < 0.7:
        return "LOW"
    elif max_probability < 0.85:
        return "MEDIUM"
    else:
        return "HIGH"
```

#### Output Format
```json
{
    "series_uid": "1.2.3.4.5...",
    "has_aneurysm": true,
    "max_probability": 0.847,
    "risk_level": "MEDIUM",
    "detections": [
        {
            "location": "R-MCA",
            "probability": 0.847,
            "confidence": "84.7%"
        },
        {
            "location": "Acomm",
            "probability": 0.523,
            "confidence": "52.3%"
        }
    ],
    "timestamp": "2026-03-08 10:30:00"
}
```

---

### 5_loss.py - Training Loss Function

**Purpose**: Guide the model during training (not used during inference).

#### Why TopK BCE Loss?

```
Standard BCE Loss Problem:
- 99.9% of voxels are "normal" (no aneurysm)
- 0.1% of voxels contain aneurysms
- Model learns to always predict "normal" (easy!)

TopK BCE Solution:
- Only compute loss on TOP 20% hardest samples
- Forces model to focus on difficult cases
- Ignores easy background voxels
```

#### TopK BCE Algorithm
```python
def topk_bce_loss(prediction, target, k_ratio=0.2):
    """
    1. Compute BCE loss for ALL voxels
    2. Sort losses from highest to lowest
    3. Take only top 20% (hardest samples)
    4. Return mean of top losses
    
    Example:
    losses = [0.01, 0.02, 0.03, 0.8, 0.9]  # 5 samples
    top_20% = [0.9, 0.8]  # 2 hardest samples
    final_loss = mean([0.9, 0.8]) = 0.85
    """
    # Per-voxel BCE loss
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Flatten and sort
    bce_flat = bce.view(-1)
    sorted_loss, _ = torch.sort(bce_flat, descending=True)
    
    # Take top k%
    k = int(len(sorted_loss) * k_ratio)
    topk_loss = sorted_loss[:k]
    
    return topk_loss.mean()
```

#### Blob Regression (EDT-based)

```
Point Annotation → Blob Target

Instead of training on single points (hard to learn),
convert to "blobs" centered on annotation points.

    Point Annotation:           Blob Target (r=65):
    
    ┌───────────────────┐      ┌───────────────────┐
    │                   │      │     ▓▓▓▓▓▓▓       │
    │                   │      │   ▓▓▓▓▓▓▓▓▓▓▓     │
    │         ●         │ ───▶ │  ▓▓▓▓▓▓●▓▓▓▓▓▓   │
    │     (point)       │      │   ▓▓▓▓▓▓▓▓▓▓▓     │
    │                   │      │     ▓▓▓▓▓▓▓       │
    └───────────────────┘      └───────────────────┘
    
EDT = Euclidean Distance Transform
- Voxels at center = 1.0 (highest)
- Voxels further away = lower value
- Beyond radius 65 = 0.0
```

---

### run_detection.py - Main Entry Point

**Purpose**: Orchestrate all stages into a single, easy-to-use script.

```python
def main():
    """
    Complete Pipeline Flow:
    
    1. Parse Arguments
       └── --input: DICOM folder
       └── --output: Results JSON path
    
    2. Load Model
       └── Load checkpoint
       └── Move to GPU
       └── Set evaluation mode
    
    3. Preprocess
       └── Load DICOM files
       └── Normalize intensity
       └── Resample to 0.5mm
    
    4. Predict (Sliding Window)
       └── Split into patches
       └── Forward pass per patch
       └── Combine with Gaussian blending
    
    5. Postprocess
       └── Apply sigmoid
       └── Extract max probabilities
       └── Apply threshold
       └── Format results
    
    6. Save Results
       └── Write JSON file
       └── Print summary
    """
```

---

## 5. How to Run Detection

### Prerequisites
```bash
# 1. Activate conda environment
conda activate pretrained_detect

# 2. Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# 3. Verify checkpoint exists
dir "Dataset004_iarsna_crop_2\Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32\fold_all\checkpoint_epoch_1500.pth"
```

### Basic Usage
```bash
# Navigate to ml_pipeline folder
cd "C:\Users\Rayan\Desktop\Main Project\Code\Pretrained detection\ml_pipeline"

# Run detection
python run_detection.py --input "path\to\dicom\folder" --output "results.json"
```

### Example with Real Data
```bash
# Using the provided test data
python run_detection.py \
    --input "C:\Users\Rayan\Downloads\Dataset\1.2.826.0.1.xxx" \
    --output "detection_results.json" \
    --verbose
```

### Command Line Options
| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to DICOM folder | Required |
| `--output` | Path for results JSON | `results.json` |
| `--threshold` | Detection threshold | `0.5` |
| `--verbose` | Print detailed progress | `False` |

---

## 6. Understanding the Output

### Example Output
```json
{
    "input_path": "C:\\Users\\Rayan\\Downloads\\Dataset\\1.2.826...",
    "timestamp": "2026-03-08T10:30:00",
    "detection": {
        "has_aneurysm": true,
        "max_probability": 0.674,
        "risk_level": "LOW"
    },
    "locations": [
        {"name": "R-MCA", "probability": 0.674, "detected": true},
        {"name": "L-ICA", "probability": 0.342, "detected": false},
        ...
    ],
    "processing_time": "45.2 seconds"
}
```

### Interpreting Results

| Field | Meaning |
|-------|---------|
| `has_aneurysm` | Whether any location exceeds threshold |
| `max_probability` | Highest probability across all locations |
| `risk_level` | NEGATIVE / LOW / MEDIUM / HIGH |
| `locations[].detected` | Whether this specific location has aneurysm |

### Risk Level Guide

| Risk Level | Action |
|------------|--------|
| NEGATIVE | No action needed |
| LOW | Monitor; may be false positive |
| MEDIUM | Recommend radiologist review |
| HIGH | Urgent clinical review recommended |

---

## 7. Training Process

> **Note**: This section explains how the model was trained. You don't need to retrain - use the provided checkpoint!

### Training Configuration (from config.py)
```python
TRAINING_CONFIG = {
    'epochs': 1500,
    'batch_size': 32,
    'optimizer': 'SGD',
    'momentum': 0.99,
    'weight_decay': 3e-5,
    'learning_rate_scheduler': 'PolynomialLR',
    'initial_lr': 0.01,
    'final_lr': 1e-7,
}
```

### Training Pipeline
```
┌──────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                 │
└──────────────────────────────────────────────────────────────────────┘

For each epoch (1 to 1500):
    For each batch (32 samples):
        
        1. Load Data
           └── Random 3D patches from training volumes
           └── Apply augmentations (rotation, flip, scale)
        
        2. Forward Pass
           └── prediction = model(input_patch)
        
        3. Compute Loss
           └── loss = TopK_BCE(prediction, target) + Dice_Loss
        
        4. Backward Pass
           └── loss.backward()  # Compute gradients
        
        5. Update Weights
           └── optimizer.step()  # Apply gradients
        
        6. Learning Rate Decay
           └── scheduler.step()  # Polynomial decay

Save checkpoint every 50 epochs
Final checkpoint: checkpoint_epoch_1500.pth
```

### Data Augmentation
```python
# Applied during training to improve generalization

augmentations = [
    RandomRotation(angles=(-30, 30)),    # Rotate 3D volume
    RandomFlip(axes=[0, 1, 2]),          # Flip along any axis
    RandomScale(factors=(0.85, 1.25)),   # Scale up/down
    RandomElasticDeformation(),           # Slight warping
    GaussianNoise(mean=0, std=0.1),      # Add noise
    GaussianBlur(sigma=(0, 1)),          # Slight blur
]
```

---

## 8. Technical Deep Dive

### Memory Optimization

#### Mixed Precision Training
```python
# Use FP16 instead of FP32 during training
# Reduces memory by ~50%, speeds up training by ~2x

with torch.cuda.amp.autocast():
    prediction = model(input)
    loss = loss_fn(prediction, target)
```

#### Gradient Checkpointing
```python
# Trade compute for memory
# Don't store intermediate activations, recompute during backward pass

from torch.utils.checkpoint import checkpoint
output = checkpoint(self.encoder_block, input)
```

### Why 3D Convolutions?

```
2D Convolution (used for images):
- Processes one slice at a time
- Loses inter-slice information
- Aneurysms spanning multiple slices may be missed

3D Convolution (used here):
- Processes entire volume at once
- Captures 3D spatial relationships
- Better for volumetric structures like aneurysms

    2D Conv                     3D Conv
    ┌─────┐                    ┌─────┐
    │     │                    │╱───╱│
    │ 2D  │ × many slices      ││3D ││ ← processes entire volume
    │     │                    │╲───╲│
    └─────┘                    └─────┘
```

### Network Depth Analysis
```
Stage   Input Ch    Output Ch    Feature Map Size    Purpose
────────────────────────────────────────────────────────────────
Enc 1   1          32           64×64×64            Low-level edges
Enc 2   32         64           32×32×32            Simple patterns
Enc 3   64         128          16×16×16            Texture patterns
Enc 4   128        256          8×8×8               Object parts
Enc 5   256        320          4×4×4               High-level features
Enc 6   320        320          2×2×2               Global context
────────────────────────────────────────────────────────────────
Dec 1   640→256    256          4×4×4               Combine features
Dec 2   512→128    128          8×8×8               Refine location
Dec 3   256→64     64           16×16×16            Add details
Dec 4   128→32     32           32×32×32            Fine-tune
Dec 5   64→32      32           64×64×64            Final resolution
Out     32→13      13           64×64×64            Per-class probs
```

---

## 9. Troubleshooting

### Common Issues

#### Issue 1: "CUDA out of memory"
```bash
# Error: RuntimeError: CUDA out of memory

# Solutions:
# 1. Reduce batch size (if training)
# 2. Use CPU instead
python run_detection.py --input ... --device cpu

# 3. Close other GPU applications
nvidia-smi  # Check GPU usage
```

#### Issue 2: "Checkpoint not found"
```bash
# Error: FileNotFoundError: Checkpoint not found

# Solution: Verify checkpoint path
dir "Dataset004_iarsna_crop_2\Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32\fold_all"

# Should contain: checkpoint_epoch_1500.pth
```

#### Issue 3: "Invalid DICOM"
```bash
# Error: InvalidDICOM or No DICOM files found

# Check:
# 1. Folder contains .dcm files
dir "path\to\folder\*.dcm"

# 2. Files are valid DICOM (not corrupted)
python -c "import pydicom; pydicom.dcmread('path/to/file.dcm')"
```

#### Issue 4: "ModuleNotFoundError"
```bash
# Error: ModuleNotFoundError: No module named 'xxx'

# Solution: Install missing packages
pip install torch numpy scipy pydicom
```

### Verification Commands
```bash
# Test each module independently
cd ml_pipeline

python -c "from config import *; print('Config OK')"
python -c "from 1_preprocessing import *; print('Preprocessing OK')"
python -c "from 2_model import *; print('Model OK')"
python -c "from 3_inference import *; print('Inference OK')"
python -c "from 4_postprocessing import *; print('Postprocessing OK')"
python -c "from 5_loss import *; print('Loss OK')"
```

---

## 10. References

### Original Solution
- **Kaggle Competition**: RSNA 2024 Lumbar Spine Degenerative Classification
- **Solution**: 7th Place by MIC-DKFZ team
- **Framework**: nnU-Net v2

### Papers
1. **nnU-Net**: Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021)

2. **ResNet**: He, K., et al. "Deep residual learning for image recognition." CVPR (2016)

3. **U-Net**: Ronneberger, O., et al. "U-Net: Convolutional networks for biomedical image segmentation." MICCAI (2015)

### Links
- nnU-Net GitHub: https://github.com/MIC-DKFZ/nnUNet
- Kaggle Competition: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine
- Model Checkpoint: https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE QUICK REFERENCE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMMANDS:                                                           │
│  ─────────                                                           │
│  Run Detection:                                                      │
│    python run_detection.py --input <DICOM_FOLDER> --output <JSON>    │
│                                                                      │
│  Test Modules:                                                       │
│    python 2_model.py      # Test model architecture                  │
│    python 5_loss.py       # Test loss functions                      │
│                                                                      │
│  FILES:                                                              │
│  ──────                                                              │
│  config.py           Configuration & constants                       │
│  1_preprocessing.py  DICOM loading, normalization                    │
│  2_model.py          Neural network architecture                     │
│  3_inference.py      Sliding window prediction                       │
│  4_postprocessing.py Result extraction                               │
│  5_loss.py           Training loss (TopK BCE)                        │
│  run_detection.py    Main entry point                                │
│                                                                      │
│  KEY PARAMETERS:                                                     │
│  ───────────────                                                     │
│  Spacing:      0.5mm isotropic                                       │
│  Intensity:    [0, 600] HU                                           │
│  Threshold:    50% probability                                       │
│  Patch Size:   64 × 64 × 64 voxels                                   │
│  Overlap:      50%                                                   │
│                                                                      │
│  MODEL STATS:                                                        │
│  ────────────                                                        │
│  Architecture: ResidualEncoderUNet                                   │
│  Parameters:   61.5 million                                          │
│  Input:        1 channel (CT volume)                                 │
│  Output:       13 channels (location probabilities)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Last Updated: March 8, 2026*
*Author: Generated for College Project*
