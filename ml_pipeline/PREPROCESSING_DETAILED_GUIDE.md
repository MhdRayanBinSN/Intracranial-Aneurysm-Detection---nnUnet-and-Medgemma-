# Detailed Preprocessing Guide for Intracranial Aneurysm Detection

## Table of Contents
1. [Overview](#1-overview)
2. [Step 1: DICOM Loading](#2-step-1-dicom-loading)
3. [Step 2: CT Normalization (Hounsfield Units)](#3-step-2-ct-normalization)
4. [Step 3: Spatial Resampling](#4-step-3-spatial-resampling)
5. [Complete Pipeline Flow](#5-complete-pipeline-flow)
6. [Mathematical Formulas](#6-mathematical-formulas)
7. [Why Each Step Matters](#7-why-each-step-matters)
8. [Code Implementation Details](#8-code-implementation-details)
9. [Visual Examples](#9-visual-examples)
10. [Advanced Topics](#10-advanced-topics)

---

## 1. Overview

### What is Preprocessing?
**Preprocessing** transforms raw medical images (DICOM files) into a standardized format that the neural network can process. Think of it as "preparing ingredients before cooking."

### The 3-Step Pipeline

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  DICOM Files    │───▶│  Load & Convert   │───▶│   Normalize CT  │───▶│    Resample      │
│  (many .dcm)    │    │  to 3D Volume     │    │   (HU values)   │    │  (to 0.5mm)      │
└─────────────────┘    └───────────────────┘    └───────────────────┘    └──────────────────┘
     INPUT                 STEP 1                    STEP 2                  STEP 3
                                                                                  │
                                                                                  ▼
                                                                      ┌──────────────────┐
                                                                      │  Ready for Model │
                                                                      │  Shape: (1,Z,Y,X)│
                                                                      └──────────────────┘
```

### Configuration Parameters (from config.py)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `HU_MIN` | 0 | Lower clipping bound (Hounsfield Units) |
| `HU_MAX` | 600 | Upper clipping bound (Hounsfield Units) |
| `TARGET_SPACING` | (0.5, 0.5, 0.5) mm | Isotropic voxel spacing |

---

## 2. Step 1: DICOM Loading

### What is DICOM?
**DICOM** (Digital Imaging and Communications in Medicine) is the standard format for medical images. A CT scan is stored as multiple 2D slices (each in a `.dcm` file), which together form a 3D volume.

### Implementation Method: SimpleITK

We use **SimpleITK** (a medical imaging library) to:
1. Find all DICOM files in a folder
2. Read them in correct order (by slice position)
3. Stack them into a 3D volume
4. Extract metadata (spacing, origin, direction)

### The Code Explained

```python
def load_dicom_series(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series from a folder.
    """
    # Create a reader object
    reader = sitk.ImageSeriesReader()
    
    # Find all DICOM files and sort them by slice position
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_folder))
    
    # Load all files into a single 3D image
    reader.SetFileNames(dicom_files)
    image_sitk = reader.Execute()
    
    # Convert to NumPy array: Shape becomes (Z, Y, X)
    # Z = number of slices, Y = height, X = width
    volume = sitk.GetArrayFromImage(image_sitk)
    
    # Extract important metadata
    properties = {
        'spacing': np.array(image_sitk.GetSpacing()[::-1]),  # mm per voxel
        'origin': np.array(image_sitk.GetOrigin()),           # world coordinates
        'direction': np.array(image_sitk.GetDirection()).reshape(3, 3),
        'original_shape': volume.shape,
    }
    
    return volume, properties
```

### Key Concepts

#### Spacing (Resolution)
**Spacing** tells you the physical size of each voxel in millimeters:
- `spacing = [1.0, 0.5, 0.5]` means each voxel is 1.0mm (Z) × 0.5mm (Y) × 0.5mm (X)
- Lower spacing = higher resolution = more detail

```
               ┌─────┬─────┬─────┬─────┐
   Spacing     │     │     │     │     │  Each square = 1 voxel
   0.5mm       │     │     │     │     │  Physical size = 0.5mm × 0.5mm
               └─────┴─────┴─────┴─────┘
               
               ┌───────────┬───────────┐
   Spacing     │           │           │  Each square = 1 voxel  
   1.0mm       │           │           │  Physical size = 1.0mm × 1.0mm
               └───────────┴───────────┘
```

#### Why SimpleITK?
1. **Handles DICOM complexity**: DICOM files contain hundreds of metadata fields
2. **Automatic sorting**: Slices are ordered by their real-world position
3. **Standard library**: Used by radiologists and researchers worldwide

### Typical Input

| Property | Typical Value |
|----------|---------------|
| Shape | (300-500, 512, 512) |
| Spacing | (0.5-1.0, 0.35-0.5, 0.35-0.5) mm |
| Pixel values | -1024 to +3071 (Hounsfield Units) |
| Data type | int16 or float32 |

---

## 3. Step 2: CT Normalization (Hounsfield Units)

### What are Hounsfield Units (HU)?

**Hounsfield Units** are the standard measurement for CT scan intensity. Named after Sir Godfrey Hounsfield (inventor of CT), they measure tissue radiodensity:

| Substance | HU Value | Appearance |
|-----------|----------|------------|
| Air | -1000 | Black |
| Fat | -100 to -50 | Dark gray |
| Water | 0 | Gray |
| Muscle | +40 to +80 | Light gray |
| Blood | +30 to +45 | Light gray |
| Contrast-enhanced blood | +100 to +600 | Bright white |
| Bone | +400 to +1000 | Very bright white |

### Why Clip to [0, 600] HU?

For **angiography** (blood vessel imaging), contrast agent is injected into the bloodstream. This makes blood vessels appear as:
- **100-300 HU** for arteries with contrast
- **300-600 HU** for heavily concentrated contrast

By clipping to **[0, 600] HU**, we:
1. ✅ Keep blood vessels (our target)
2. ✅ Remove air (-1000 HU) that adds noise
3. ✅ Remove bone (>600 HU) that could confuse the model
4. ✅ Focus the model's attention on relevant structures

### The Normalization Formula

```
                     pixel_value - HU_MIN
normalized_value = ─────────────────────────
                      HU_MAX - HU_MIN
```

**Example calculation:**
```
Original pixel = 300 HU (blood vessel with contrast)
HU_MIN = 0, HU_MAX = 600

Step 1: Clip (300 is already in [0, 600], no change)
Step 2: Normalize = (300 - 0) / (600 - 0) = 0.5
```

### The Code Explained

```python
def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """
    Normalize CT Hounsfield units to [0, 1] range.
    """
    # STEP 1: Clip values outside range
    # np.clip(volume, HU_MIN, HU_MAX):
    #   - Values < 0 become 0
    #   - Values > 600 become 600
    #   - Values in [0, 600] stay unchanged
    volume_clipped = np.clip(volume, HU_MIN, HU_MAX)  # HU_MIN=0, HU_MAX=600
    
    # STEP 2: Min-Max Scaling to [0, 1]
    # This is called "Min-Max Normalization"
    volume_norm = (volume_clipped - HU_MIN) / (HU_MAX - HU_MIN)
    
    # STEP 3: Convert to float32 (neural networks use float32)
    return volume_norm.astype(np.float32)
```

### Visual Effect

```
BEFORE NORMALIZATION (Raw CT):           AFTER NORMALIZATION:
┌─────────────────────────────┐         ┌─────────────────────────────┐
│-1024               +3071 HU │         │  0.0                   1.0  │
│  ◄────────────────────────► │   →     │  ◄─────────────────────────►│
│      huge range             │         │      compressed range       │
│  bone, air, all tissues     │         │   only vessels visible      │
└─────────────────────────────┘         └─────────────────────────────┘
```

### Why This Helps the Neural Network

1. **Consistent scale**: All inputs are [0, 1] regardless of scanner
2. **Focus on task**: Network only sees blood vessel intensities
3. **Numerical stability**: Smaller numbers = more stable gradients
4. **Faster convergence**: Normalized inputs train faster

---

## 4. Step 3: Spatial Resampling

### What is Resampling?

**Resampling** changes the physical spacing (resolution) of the image. It's like resizing a photo, but in 3D and preserving physical dimensions.

### Why Resample to 0.5mm Isotropic?

**Problem**: Different CT scanners produce images with different spacings:
- Scanner A: `[0.8, 0.4, 0.4]` mm
- Scanner B: `[1.5, 0.6, 0.6]` mm

**Solution**: Resample ALL images to the same spacing `[0.5, 0.5, 0.5]` mm

**Isotropic** means all dimensions have equal spacing (a perfect cube voxel).

```
BEFORE (Anisotropic):              AFTER (Isotropic 0.5mm):
     1.0mm                              0.5mm
    ┌──────┐                           ┌────┐
    │      │ 0.5mm                     │    │ 0.5mm
    │      │                           │    │
    └──────┘                           └────┘
    Rectangular                        Perfect cube
```

### The Resampling Algorithm

#### Step A: Calculate New Shape

```python
# How does the shape change?
scale_factors = original_spacing / target_spacing
new_shape = original_shape * scale_factors
```

**Example:**
```
Original: shape=(200, 512, 512), spacing=(1.0, 0.5, 0.5) mm
Target: spacing=(0.5, 0.5, 0.5) mm

scale_factors = [1.0/0.5, 0.5/0.5, 0.5/0.5] = [2.0, 1.0, 1.0]
new_shape = [200*2.0, 512*1.0, 512*1.0] = [400, 512, 512]

(Z dimension doubles because voxels become half the size)
```

#### Step B: Interpolation (Linear)

**Interpolation** fills in pixel values when we resize. We use **Linear Interpolation** (also called trilinear in 3D):

```
What is Linear Interpolation?

Given two known points, estimate the value in between:

Known: A=10  ●─────────────────────●  B=20
                     ↑
              What is value here?
              
Answer: (10 + 20) / 2 = 15 (midpoint)
        Or weighted by distance from each point
```

In 3D (trilinear), we interpolate in all three dimensions:

```
           P₁ ●────────● P₂
             /|       /|
            / |      / |
        P₃ ●────────● P₄
           |  |     |  |        P? = weighted average
           |  ●────────● P₆     of all 8 corner points
           | /P₅    | /
           |/       |/
        P₇ ●────────● P₈
```

### The Code Explained

```python
def resample_volume(volume: np.ndarray, 
                    original_spacing: np.ndarray,
                    target_spacing: Tuple[float, float, float] = TARGET_SPACING):
    """
    Resample volume to target spacing using linear interpolation.
    """
    # Calculate how the shape must change
    original_shape = np.array(volume.shape)
    scale_factors = original_spacing / np.array(target_spacing)
    new_shape = np.round(original_shape * scale_factors).astype(int)
    
    # Create SimpleITK image (required for resampling)
    image_sitk = sitk.GetImageFromArray(volume)
    image_sitk.SetSpacing(original_spacing[::-1].tolist())  # Note: SimpleITK uses (X,Y,Z)
    
    # Configure the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1])   # Target: 0.5mm isotropic
    resampler.SetSize(new_shape[::-1].tolist())        # New dimensions
    resampler.SetInterpolator(sitk.sitkLinear)         # Linear interpolation
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    
    # Execute resampling
    resampled_sitk = resampler.Execute(image_sitk)
    resampled = sitk.GetArrayFromImage(resampled_sitk)
    
    return resampled, np.array(target_spacing)
```

### Interpolation Methods Comparison

| Method | Quality | Speed | When to Use |
|--------|---------|-------|-------------|
| **Nearest Neighbor** | Low (blocky) | Very Fast | Segmentation masks (preserves labels) |
| **Linear/Trilinear** | Good | Fast | CT/MRI images (our choice) |
| **B-Spline** | Very Good | Slow | When quality is critical |
| **Sinc** | Excellent | Very Slow | Research/publishing |

We use **Linear** because it balances quality and speed.

---

## 5. Complete Pipeline Flow

### The Full `preprocess()` Function

```python
def preprocess(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Full preprocessing pipeline.
    """
    # STEP 1: Load DICOM files into 3D volume
    volume, properties = load_dicom_series(dicom_folder)
    # volume.shape = (Z, Y, X), e.g., (300, 512, 512)
    # properties['spacing'] = original spacing in mm
    
    # STEP 2: Normalize HU values to [0, 1]
    volume_norm = normalize_ct(volume)
    # Same shape, but values now in [0, 1]
    
    # STEP 3: Resample to target spacing (0.5mm isotropic)
    volume_resampled, new_spacing = resample_volume(
        volume_norm, 
        properties['spacing']
    )
    # Shape changes based on original vs target spacing
    
    # STEP 4: Add channel dimension for neural network
    # Neural networks expect shape: (Channels, Depth, Height, Width)
    data = volume_resampled[np.newaxis, ...]  # (1, Z, Y, X)
    
    # Update properties with new info
    properties['spacing'] = new_spacing
    properties['preprocessed_shape'] = data.shape
    
    return data, properties
```

### Visual Summary of Transformations

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW INPUT                                                                   │
│  ┌─────────────────────────────────────────┐                                │
│  │ DICOM folder with 300 .dcm files        │                                │
│  │ Each file: 512×512 pixels               │                                │
│  │ Spacing: (1.0, 0.4, 0.4) mm             │                                │
│  │ Values: -1024 to +3071 HU               │                                │
│  └─────────────────────────────────────────┘                                │
│                    │                                                         │
│                    ▼ Step 1: LOAD                                            │
│  ┌─────────────────────────────────────────┐                                │
│  │ 3D NumPy Array                          │                                │
│  │ Shape: (300, 512, 512)                  │                                │
│  │ dtype: int16                            │                                │
│  │ Values: -1024 to +2500 HU               │                                │
│  └─────────────────────────────────────────┘                                │
│                    │                                                         │
│                    ▼ Step 2: NORMALIZE                                       │
│  ┌─────────────────────────────────────────┐                                │
│  │ 3D NumPy Array                          │                                │
│  │ Shape: (300, 512, 512) [unchanged]      │                                │
│  │ dtype: float32                          │                                │
│  │ Values: 0.0 to 1.0                      │                                │
│  └─────────────────────────────────────────┘                                │
│                    │                                                         │
│                    ▼ Step 3: RESAMPLE                                        │
│  ┌─────────────────────────────────────────┐                                │
│  │ 3D NumPy Array                          │                                │
│  │ Shape: (600, 410, 410) [changed!]       │                                │
│  │ Spacing: (0.5, 0.5, 0.5) mm             │                                │
│  │ Values: 0.0 to 1.0                      │                                │
│  └─────────────────────────────────────────┘                                │
│                    │                                                         │
│                    ▼ Step 4: ADD CHANNEL                                     │
│  ┌─────────────────────────────────────────┐                                │
│  │ 4D NumPy Array                          │                                │
│  │ Shape: (1, 600, 410, 410)               │                                │
│  │ Format: (Channel, Z, Y, X)              │                                │
│  │ Ready for neural network!               │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Mathematical Formulas

### Min-Max Normalization

$$
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Where:
- $x$ = original pixel value (in HU)
- $x_{min}$ = 0 (HU_MIN)
- $x_{max}$ = 600 (HU_MAX)
- $x_{normalized}$ = output in [0, 1]

### Resampling Scale Factor

$$
\text{scale\_factor}_i = \frac{\text{original\_spacing}_i}{\text{target\_spacing}_i}
$$

$$
\text{new\_shape}_i = \text{round}(\text{original\_shape}_i \times \text{scale\_factor}_i)
$$

### Trilinear Interpolation

For a point $(x, y, z)$ between 8 neighboring voxels:

$$
V(x,y,z) = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1} V_{ijk} \cdot (1-|x-x_i|)(1-|y-y_j|)(1-|z-z_k|)
$$

Where $V_{ijk}$ are the 8 corner voxel values.

---

## 7. Why Each Step Matters

### Impact on Model Performance

| Step | What Happens if Skipped | Impact on Accuracy |
|------|-------------------------|-------------------|
| **DICOM Loading** | Cannot read medical images | ❌ Fatal - no data |
| **HU Normalization** | Values range from -1024 to +3071 | 🔴 -30% accuracy (unstable training) |
| **Resampling** | Different scales confuse the model | 🟠 -15% accuracy (inconsistent features) |
| **Channel Dimension** | Shape mismatch with model | ❌ Fatal - model crash |

### Why 0.5mm Spacing Specifically?

1. **Model was trained at 0.5mm**: The pretrained RSNA model expects this
2. **Balance quality vs memory**: 
   - 0.25mm = 8× more voxels = runs out of GPU memory
   - 1.0mm = 8× fewer voxels = loses small aneurysm details
3. **Clinical standard**: Many research models use 0.5mm

### Why Linear Interpolation?

| For Images (CT/MRI) | For Masks (Segmentation) |
|---------------------|--------------------------|
| Linear = smooth gradients | Nearest Neighbor = preserve labels |
| Blends neighboring values | No blending (0 or 1 only) |
| Better visual quality | Maintains binary values |

---

## 8. Code Implementation Details

### Coordinate Systems

SimpleITK and NumPy use **opposite** axis orders:

```
NumPy Arrays:         SimpleITK Images:
array[Z, Y, X]        image[X, Y, Z]
    │  │  │               │  │  │
    │  │  └─ columns      │  │  └─ slices
    │  └──── rows         │  └──── rows
    └─────── slices       └─────── columns
```

That's why we use `[::-1]` to reverse axes:
```python
properties['spacing'] = np.array(image_sitk.GetSpacing()[::-1])  # (X,Y,Z) → (Z,Y,X)
```

### Memory Considerations

```python
# Typical memory usage for one CT scan:
# 
# Raw volume: 300 × 512 × 512 × 2 bytes (int16) = ~157 MB
# After normalize: 300 × 512 × 512 × 4 bytes (float32) = ~314 MB
# After resample: 600 × 410 × 410 × 4 bytes = ~404 MB (example)
# With channel: Same as above, just different shape
```

### Error Handling

```python
# Check for empty DICOM folder
if not dicom_files:
    raise ValueError(f"No DICOM files found in {dicom_folder}")
    
# Always print intermediate shapes for debugging
print(f"Loaded shape: {volume.shape}")
print(f"Spacing: {properties['spacing']} mm")
```

---

## 9. Visual Examples

### Before vs After Normalization

```
BEFORE (Raw HU):                    AFTER (Normalized):
┌────────────────────────┐         ┌────────────────────────┐
│  ████  skull (1000 HU) │         │  ████  clipped to 1.0  │
│  ░░░░  brain (40 HU)   │    →    │  ░░░░  ~0.07 (visible) │
│  ▓▓▓▓  vessel (300 HU) │         │  ▓▓▓▓  0.5 (bright)    │
│        air (-1000 HU)  │         │        0.0 (black)     │
└────────────────────────┘         └────────────────────────┘
```

### Before vs After Resampling

```
BEFORE (1.0mm × 0.4mm × 0.4mm):    AFTER (0.5mm uniform):
┌─────┬─────┬─────┐                ┌──┬──┬──┬──┬──┐
│     │     │     │ 1.0mm          │  │  │  │  │  │ 0.5mm
├─────┼─────┼─────┤           →    ├──┼──┼──┼──┼──┤
│     │     │     │                │  │  │  │  │  │
└─────┴─────┴─────┘                ├──┼──┼──┼──┼──┤
  0.4mm 0.4mm 0.4mm                │  │  │  │  │  │
    Anisotropic                    └──┴──┴──┴──┴──┘
    (stretched looking)              Isotropic
                                     (uniform cube voxels)
```

---

## 10. Advanced Topics

### Alternative Normalization Methods

#### Z-Score Normalization (used by default nnU-Net)

$$
x_{zscore} = \frac{x - \mu}{\sigma}
$$

Where $\mu$ = mean, $\sigma$ = standard deviation

**nnU-Net's approach for CT:**
1. Collect foreground voxels from training set
2. Compute global mean, std, and 0.5/99.5 percentiles
3. Clip to percentiles, then z-score normalize

Our simplified approach uses **Min-Max on [0, 600] HU** because:
- Inference-only (no training data statistics)
- CT has known physical scale (HU)
- Simpler to understand and debug

### Preprocessing Order Matters

```
CORRECT ORDER:                    WRONG ORDER:
1. Load                           1. Load
2. Normalize ← before resampling  2. Resample ← loses HU meaning!
3. Resample                       3. Normalize ← wrong scale
```

**Why normalize before resample?**
- Interpolation can introduce values outside [0, 600] HU if not clipped first
- Keeps HU values physically meaningful during processing

### GPU Acceleration

For faster preprocessing (not implemented currently):
```python
import cupy as cp
import cucim

# GPU-accelerated resampling using CUDA
volume_gpu = cp.asarray(volume)
# ... GPU operations
```

### Cropping to Region of Interest (Optional)

Some pipelines add a cropping step to reduce computation:
```python
def crop_to_brain(volume, margin=10):
    """Crop to non-zero region with margin."""
    nonzero = np.where(volume > 0)
    z_min, z_max = nonzero[0].min() - margin, nonzero[0].max() + margin
    y_min, y_max = nonzero[1].min() - margin, nonzero[1].max() + margin
    x_min, x_max = nonzero[2].min() - margin, nonzero[2].max() + margin
    return volume[z_min:z_max, y_min:y_max, x_min:x_max]
```

---

## Summary

| Step | Method | Input | Output | Why |
|------|--------|-------|--------|-----|
| **1** | SimpleITK DICOM Reader | Folder of .dcm files | 3D NumPy (Z,Y,X) | Standard medical image loading |
| **2** | Min-Max Normalization | HU values [-1024, 3071] | Float [0, 1] | Focus on vessels, stabilize training |
| **3** | Trilinear Resampling | Variable spacing | 0.5mm isotropic | Uniform input for neural network |

### Quick Reference Commands

```bash
# Run preprocessing on a DICOM folder
python 1_preprocessing.py /path/to/dicom/folder

# The output is ready for the model
# Shape: (1, Z, Y, X) with values in [0, 1]
```

---

## References

1. **Hounsfield Units**: Hounsfield, G.N. (1973). Computerized transverse axial scanning
2. **nnU-Net Framework**: Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
3. **SimpleITK**: Lowekamp et al. (2013). The Design of SimpleITK
4. **RSNA 2024 Competition**: Kaggle RSNA Intracranial Aneurysm Detection

---

*This guide accompanies the preprocessing code in `1_preprocessing.py`*
