# ML Pipeline - Intracranial Aneurysm Detection

This folder contains the **simplified, essential ML code** extracted from the nnU-Net solution.

## Folder Structure

```
ml_pipeline/
├── README.md                    # This file
├── config.py                    # Configuration & constants
├── 1_preprocessing.py           # DICOM loading & preprocessing
├── 2_model.py                   # Neural network architecture
├── 3_inference.py               # Sliding window prediction
├── 4_postprocessing.py          # Output processing & visualization
├── 5_loss.py                    # Training loss function (TopK BCE)
└── run_detection.py             # Main entry point
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

 DICOM Folder
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 1_preproc   │───▶│  2_model    │───▶│ 3_inference │───▶│ 4_postproc  │
│             │    │             │    │             │    │             │
│ Load DICOM  │    │ ResEnc UNet │    │ Sliding     │    │ Extract     │
│ Normalize   │    │ (13 classes)│    │ Window      │    │ Detections  │
│ Resample    │    │             │    │ Prediction  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               ▼
                                                         JSON Results


┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Annotations                Training Data              Model
      │                          │                        │
      ▼                          ▼                        ▼
┌─────────────┐           ┌─────────────┐         ┌─────────────┐
│ Point to    │──────────▶│ 5_loss.py   │────────▶│ Backprop    │
│ Blob Conv   │           │             │         │             │
│ (EDT r=65)  │           │ TopK BCE    │         │ Update      │
│             │           │ + Dice Loss │         │ Weights     │
└─────────────┘           └─────────────┘         └─────────────┘
```

## Quick Start

```bash
conda activate pretrained_detect
python run_detection.py --input "path/to/dicom" --output "results.json"
```

## Files Description

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~90 | All constants, paths, labels, hyperparameters |
| `1_preprocessing.py` | ~130 | Load DICOM, normalize HU, resample to 0.5mm |
| `2_model.py` | ~220 | ResidualEncoderUNet architecture (6-stage encoder, 5-stage decoder) |
| `3_inference.py` | ~200 | Sliding window prediction with Gaussian blending |
| `4_postprocessing.py` | ~180 | Extract probabilities, apply threshold, format results |
| `5_loss.py` | ~200 | TopK BCE + Dice loss for training |
| `run_detection.py` | ~230 | Main script combining everything |

## Key Concepts

### 1. Preprocessing (1_preprocessing.py)
```python
# Clip CT values to blood vessel range
volume = np.clip(volume, 0, 600)  # HU units

# Normalize to [0, 1]
volume = (volume - 0) / (600 - 0)

# Resample to isotropic 0.5mm spacing
volume = resample(volume, target_spacing=(0.5, 0.5, 0.5))
```

### 2. Model Architecture (2_model.py)
```
Encoder: 1ch → 32 → 64 → 128 → 256 → 320 → 320 (bottleneck)
                ↓     ↓      ↓      ↓      ↓
                └─────────skip connections─────────┐
                                                   ↓
Decoder:                   320 → 256 → 128 → 64 → 32
                                                   ↓
Output:                                       32 → 13 (locations)
```

### 3. Sliding Window (3_inference.py)
```
Large Volume:  [====================================]
                ↓
Patches:       [====]     (0-128)
                 [====]   (64-192)  50% overlap
                   [====] (128-256)
                ↓
Merge with Gaussian weighting
```

### 4. TopK Loss (5_loss.py)
```
Standard BCE: Average loss over ALL pixels (dominated by easy negatives)
TopK BCE:     Average loss over TOP 20% HARDEST pixels (focus on difficult cases)
```
