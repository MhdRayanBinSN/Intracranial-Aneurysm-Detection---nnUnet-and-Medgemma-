# MedGemma Integration — Build Process & Implementation Guide

> **Project:** Intracranial Aneurysm Detection — MedGemma AI Pipeline  
> **Integrated With:** RSNA Intracranial Aneurysm Detection (nnU-Net / Custom CNN)  
> **Build Status:** ✅ Production-Ready  
> **Last Updated:** May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Environment Setup](#4-environment-setup)
5. [Dependencies](#5-dependencies)
6. [Implementation Details](#6-implementation-details)
7. [Build Problems Solved](#7-build-problems-solved)
8. [API Reference](#8-api-reference)
9. [Integration Guide (For Other Projects)](#9-integration-guide-for-other-projects)
10. [Data Format Reference](#10-data-format-reference)

---

## 1. Project Overview

This pipeline integrates **Google MedGemma 4B** (a multimodal medical LLM) into an existing intracranial aneurysm detection project as a **secondary verification layer**.

### What It Does

```
User uploads DICOM / NIfTI brain scan
        ↓
Intensity-based region detection (finds suspicious high-density areas)
        ↓
Multi-window CLAHE preprocessing (CT: Brain + Vessel + Bone channels)
        ↓
MedGemma 4B LLM Analysis (modality-aware, chain-of-thought reasoning)
        ↓
Annotated images + structured radiological report
```

### Supported Modalities

| Modality | Full Name | Detection Method |
|---|---|---|
| **CTA** | CT Angiography | Multi-window RGB (brain/vessel/bone) |
| **CT** | Plain CT | CTA windows (fallback) |
| **MRA** | MR Angiography | Percentile normalization |
| **MRI T2** | T2-weighted MRI | Percentile normalization (flow voids) |
| **MRI T1 Post** | Gadolinium-enhanced T1 | Percentile normalization (enhancement) |

### Integration Context

This MedGemma module is designed to work **alongside** (not replace) the main nnU-Net/CNN detection pipeline. It provides:
- **LLM-based verification** of suspicious regions found by the primary model
- **Multi-modality support** beyond what the primary model covers
- **Structured radiology reports** with anatomical localisation

---

## 2. Architecture

```
train/
├── backend/          ← FastAPI server (MedGemma-only, port 8000)
├── frontend/         ← React + Vite UI (port 5173)
├── medgemma/         ← MedGemma inference module
│   ├── config.py     ← Multi-modality config, prompts, windows
│   ├── inference.py  ← MedGemmaInference class
│   └── utils/        ← Image loading helpers
├── ml/               ← Legacy CNN/nnU-Net (separate pipeline)
└── .env              ← Secrets (HF_TOKEN)
```

### Request Flow

```
React Frontend (localhost:5173)
    │  POST /medgemma/analyze-upload  (multipart form — DICOM/NIfTI files)
    ▼
FastAPI Backend (localhost:8000)
    │  1. Parse files → pixel arrays
    │  2. Intensity-based region detection (scipy)
    │  3. CLAHE multi-window preprocessing
    │  4. Lazy-load MedGemmaInference (first request only)
    │  5. Chain-of-thought LLM analysis per finding
    │  6. Save annotated PNG to temp dir
    ▼
Response: JSON with findings, report, image URLs
    │  GET /medgemma/finding-image/{id}/{slice}
    ▼
React renders results in MedGemmaResults.jsx
```

---

## 3. Directory Structure

```
train/
│
├── .env                          # ← HF_TOKEN goes here (DO NOT COMMIT)
├── .gitignore                    # Protects .env, checkpoints, cache
│
├── backend/
│   ├── main.py                   # FastAPI app — MedGemma-only endpoints
│   └── requirements.txt          # Backend Python deps
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Router (3 routes: /, /medgemma, /medgemma/results)
│   │   ├── api/client.js         # Axios API client (MedGemma endpoints only)
│   │   ├── components/
│   │   │   └── Navbar.jsx        # Home + MedGemma nav only
│   │   └── pages/
│   │       ├── Home.jsx          # Landing page (MedGemma-focused)
│   │       ├── MedGemma.jsx      # File upload + analysis trigger
│   │       └── MedGemmaResults.jsx  # Results viewer with annotated images
│   └── package.json
│
├── medgemma/
│   ├── config.py                 # All settings, prompts, windows
│   ├── inference.py              # MedGemmaInference class
│   ├── requirements.txt          # MedGemma-specific deps
│   └── utils/
│       ├── __init__.py
│       └── image_loader.py       # DICOM/NIfTI loading utilities
│
└── ml/                           # Existing CNN pipeline (untouched)
    ├── data/
    │   └── advanced_preprocessing.py   # KaggleWinningPreprocessor (used by backend)
    └── ...
```

---

## 4. Environment Setup

### Prerequisites

- **OS:** Windows 10/11
- **GPU:** NVIDIA with CUDA 11.8+ (RTX 3060 tested)
- **RAM:** 16 GB minimum (model uses ~2.5 GB VRAM with 4-bit quantization)
- **Python:** 3.10+ (via Anaconda)
- **Node.js:** 18+

### Step 1 — Create Conda Environment

```bash
conda create -n medgemma python=3.10
conda activate medgemma
```

### Step 2 — Install PyTorch with CUDA 11.8

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ **Critical:** `transformers 5.x` requires PyTorch 2.4+. Use `transformers==4.38.2`
> with PyTorch 2.1.x to avoid incompatibilities.

### Step 3 — Install Backend Dependencies

```bash
pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6
pip install transformers==4.38.2 accelerate==0.27.2 bitsandbytes
pip install pydicom nibabel SimpleITK scipy Pillow python-dotenv
pip install pydicom==3.0.1 numpy pandas
```

### Step 4 — Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 5 — Configure HF Token

Create `train/.env`:
```env
# Hugging Face READ token (required for google/medgemma-4b-it)
# Get token: https://huggingface.co/settings/tokens
# Accept terms: https://huggingface.co/google/medgemma-4b-it
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 6 — Run the Application

**Terminal 1 — Backend:**
```bash
conda activate medgemma
cd train/backend
python main.py
# → http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd train/frontend
npm run dev
# → http://localhost:5173
```

---

## 5. Dependencies

### Backend Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.109.0 | REST API framework |
| `uvicorn[standard]` | 0.27.0 | ASGI server with hot-reload |
| `python-multipart` | 0.0.6 | File upload support |
| `torch` | 2.1.2+cu118 | Deep learning, CUDA 11.8 |
| `transformers` | 4.38.2 | HuggingFace model loading |
| `accelerate` | 0.27.2 | Multi-device model distribution |
| `bitsandbytes` | 0.49.1 | 4-bit quantization (INT4/NF4) |
| `pydicom` | 3.0.1 | DICOM file parsing |
| `nibabel` | ≥5.2.0 | NIfTI file parsing |
| `SimpleITK` | ≥2.3.1 | Medical image resampling |
| `scipy` | ≥1.11.0 | Region detection (ndimage, label) |
| `Pillow` | ≥10.0.0 | Image manipulation |
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `numpy` | 1.26.4 | Array operations |

### Frontend Dependencies

| Package | Purpose |
|---|---|
| `react` + `react-dom` | UI framework |
| `react-router-dom` | Client-side routing |
| `axios` | HTTP API client |
| `framer-motion` | Animations |
| `@heroicons/react` | Icons |
| `tailwindcss` | Utility CSS |
| `clsx` | Conditional class merging |
| `vite` | Build tool & dev server |

---

## 6. Implementation Details

### 6.1 Model Loading — 4-Bit Quantization

The model uses `BitsAndBytesConfig` with NF4 quantization:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # double quant saves ~0.4 bits/param
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — better than FP4
    llm_int8_enable_fp32_cpu_offload=True,
)
```

**Memory reduction:** ~16 GB (full precision) → **~2.5 GB VRAM**  
**Trade-off:** ~3-5% accuracy reduction vs. full precision (acceptable for screening)

### 6.2 Lazy Loading

MedGemma loads only on the **first analysis request**, not at startup:

```python
medgemma_model = None  # global

def get_medgemma():
    global medgemma_model
    if medgemma_model is None:
        medgemma_model = MedGemmaInference()  # loads here
    return medgemma_model
```

**Benefit:** Backend starts instantly. VRAM only consumed when needed.

### 6.3 Multi-Window CT Preprocessing

For CTA/CT scans, three clinically-relevant Hounsfield Unit windows are combined into an RGB image:

```python
CTA_WINDOWS = [
    (40,  80),    # Brain window   → Ch0 (Red)   — gray/white matter
    (300, 600),   # Vessel window  → Ch1 (Green)  — contrast-opacified arteries
    (700, 3000),  # Bone window    → Ch2 (Blue)   — skull base, calcification
]
```

Each window is independently normalised to 0–255 and stacked as a 3-channel image. This encodes **3 different tissue contrasts** that MedGemma sees simultaneously.

### 6.4 Aneurysm-Specific Region Detection

The detection filter pipeline (before MedGemma is called):

```
1. Top 1% intensity threshold (percentile 99) — very strict
2. Binary morphological opening (3 iters) — removes noise
3. Binary morphological closing (2 iters) — fills gaps
4. Connected component labeling (scipy.ndimage.label)
5. Per-region filters:
   ├── Size: 15–120 px width AND height (≈3–25mm at 0.5mm/px)
   ├── Location: center must be within 15–85% of image (skull margin rejection)
   ├── Pixel count: ≥50 px (prevents single-pixel noise)
   ├── Circularity: ≥0.35 (aneurysms are round; vessels are elongated)
   └── Aspect ratio: ≤3.0 (rejects linear vessels)
```

**Only slices with passing regions get sent to MedGemma** — this prevents wasting inference time on empty/background slices.

### 6.5 Chain-of-Thought (CoT) Prompting

Instead of directly asking "is there an aneurysm?", the CoT prompt forces structured reasoning:

```
STEP 1 — VESSELS VISIBLE: List every vessel segment
STEP 2 — BIFURCATION INSPECTION: Check each bifurcation
STEP 3 — CANDIDATE ASSESSMENT: Is it saccular or infundibulum/artifact?
STEP 4 — FINAL ANSWER: Structured output with location, size, confidence
```

**Why this works:** MedGemma (like all LLMs) produces better answers when it reasons before concluding. This reduces false positives significantly.

### 6.6 Modality Detection

**From DICOM metadata:**
```python
# Checks DICOM tags: Modality, SeriesDescription, ContrastBolusAgent
if modality_tag == "CT":
    if "ANGIO" in series_description → CTA
    elif contrast_agent present     → CTA
    else                            → CT

if modality_tag == "MR":
    if "TOF" or "MRA" in description → MRA
    elif "POST" or "GAD" in any tag  → MRI_T1_POST
    elif "T2" in description         → MRI_T2
    else                             → MRI_T2 (safe default)
```

**From pixel statistics (when no metadata):**
```python
has_negatives = np.any(arr < -50)   # CT has negative HU (air = -1000)
p99 = np.percentile(arr, 99)         # CT can exceed +500 HU (bone)

if has_negatives or p99 > 500 → CTA/CT
else                           → MRI_T2
```

### 6.7 NIfTI Canonical Orientation

NIfTI files store voxels in RAS (Right-Anterior-Superior) or other orientations depending on the scanner. Using `nibabel.as_closest_canonical()`:

```python
nii_canonical = nib.as_closest_canonical(nii)
data = nii_canonical.get_fdata(dtype=np.float32)
```

This ensures slices at `data[:, :, z]` are **always axial** and match the radiological convention used by DICOM (left side of image = patient's right side).

### 6.8 Confidence Ensemble

When BiomedCLIP scores are available (optional integration):

| CLIP Score | MedGemma | Result |
|---|---|---|
| ≥0.30 (high) | YES | ✅ HIGH CONFIDENCE |
| <0.30 (low) | YES | ⚠️ MEDIUM — review |
| ≥0.30 (high) | NO | 🔍 FLAG FOR REVIEW |
| <0.30 (low) | NO | 🔵 CONFIRMED NEGATIVE |

---

## 7. Build Problems Solved

### Problem 1 — `HF_TOKEN` Not Loaded (401 Gated Repo Error)

**Error:**
```
401 Client Error. Cannot access gated repo for url
https://huggingface.co/google/medgemma-4b-it/resolve/main/chat_template.json
```

**Root Cause:** The model weights were cached locally (883 shards), but processor metadata files (`chat_template.json`, tokenizer config) were missing. These require authenticated download.

**Fix:** 
- Created `train/.env` with `HF_TOKEN=hf_xxxxx`
- Added `python-dotenv` auto-loading in `config.py`
- Changed `inference.py` to auto-detect mode: if token present → online mode (downloads missing files), if not → offline cache

```python
if offline_mode is None:
    offline_mode = False if HF_TOKEN else True  # smart default
```

---

### Problem 2 — `generation_config` Conflict (max_length=20)

**Error:**
```
Passing `generation_config` together with generation-related arguments
({'max_new_tokens'}) is deprecated...
```
And silently: model output was truncated to 20 tokens because the saved `generation_config.json` had `max_length=20`.

**Root Cause:** Google's MedGemma checkpoint saves a `generation_config.json` with `max_length=20`, which overrides any `max_new_tokens` argument passed at inference time.

**Fix:**
```python
# Override saved generation config after model loads
if hasattr(pipe.model, 'generation_config'):
    pipe.model.generation_config.max_length = None
    pipe.model.generation_config.max_new_tokens = MAX_NEW_TOKENS
```

---

### Problem 3 — `torch_dtype` Deprecated Warning

**Warning:** `torch_dtype is deprecated! Use dtype instead!`

**Fix:** Changed all `pipeline()` calls to use `dtype=torch.bfloat16` instead of `torch_dtype`.

---

### Problem 4 — Model Re-loading on Every Slice

**Symptom:** Log showed "🔄 Loading MedGemma model..." before each finding slice.

**Root Cause:** The `get_medgemma()` lazy loader was correct, but the `medgemma_model_error` global was being set on first failure, causing all subsequent calls to return 500 immediately without retrying. This looked like re-loading but was actually repeated error responses.

**Fix:** Smart retry logic — if offline fails but token is available, retry online:
```python
if offline_mode and HF_TOKEN:
    # Retry online to fetch missing processor files
    os.environ.pop("HF_HUB_OFFLINE", None)
    # ... retry load
```

---

### Problem 5 — NIfTI Orientation Mismatch

**Symptom:** Detections from `.nii` files showed different locations than equivalent `.dcm` files from the same patient. Left/Right sides were swapped.

**Root Cause:** NIfTI files use various orientation conventions (RAS, LAS, etc.), and a simple `np.rot90()` was insufficient for proper axial reorientation.

**Fix:** Use nibabel's canonical reorientation:
```python
nii_canonical = nib.as_closest_canonical(nii)  # always → RAS+
data = nii_canonical.get_fdata(dtype=np.float32)
# Now data[:, :, z] is a proper axial slice
```

---

### Problem 6 — `_cowseg.nii` Binary Mask Uploaded as Scan

**Symptom:** Analysis returned 0 or garbage findings for some NIfTI uploads.

**Root Cause:** The dataset contains two NIfTI files per patient:
- `1.2.826...381.nii` — the actual CT scan (HU values)
- `1.2.826...381_cowseg.nii` — Circle of Willis segmentation mask (0/1 only)

Users accidentally uploading the `_cowseg` file got binary mask values (0 and 1), which the threshold logic handled incorrectly.

**Fix:** Two-stage rejection in the parser:
```python
# Stage 1: filename check
if "_cowseg" in fname or "_seg" in fname or "_mask" in fname:
    print("⏭️ Skipping segmentation mask")
    continue

# Stage 2: value sanity check
if data.max() <= 1.1:
    print("❌ Binary mask detected — not a CT scan")
    continue
```

---

### Problem 7 — Legacy nnU-Net Code Conflicting with MedGemma

**Symptom:** Backend startup tried to load a CNN model from `ml/config/config.yaml` and `ml/checkpoints/best.pt`, failed silently, and left `model = None` which caused the health endpoint to report `model_loaded: false`.

**Fix:** Completely rewrote `main.py` to remove all legacy model code. The file now serves **only MedGemma endpoints**:
- Removed: `LOCATION_NAMES`, `PredictionResult`, `AnalysisResponse`, `model`/`preprocessor` globals, lifespan model loader, `/analyze`, `/demo/predict`, `/locations`, `/medgemma/patients`, `/medgemma/analyze` (series folder), `/medgemma/slice/` endpoints
- Kept: MedGemma lazy loader, `/medgemma/analyze-upload`, `/medgemma/finding-image/`, `/health`

---

### Problem 8 — Frontend Dead Routes After Backend Cleanup

**Symptom:** After removing legacy backend endpoints, the frontend `Analysis.jsx` page called `/analyze` and `/demo/predict` which returned 404.

**Fix:** Cleaned frontend to match:
- Deleted: `Analysis.jsx`, `Results.jsx`
- Updated: `App.jsx` (removed routes), `Navbar.jsx` (removed Analysis nav item), `api/client.js` (removed 7 old API functions), `Home.jsx` (updated all CTAs to `/medgemma`)

---

### Problem 9 — "Paging File Too Small" on Windows (safetensors)

**Error:** CUDA out of memory / Windows paging error when loading model shards.

**Root Cause:** `safetensors` by default uses memory-mapped I/O on Windows, which requires the paging file to be large enough to map all 883 weight shards simultaneously.

**Fix:**
```python
os.environ["SAFETENSORS_FAST_GPU"] = "0"  # Disable mmap — load directly
```
Set before any torch/transformers imports.

---

### Problem 10 — Conda `run -c` Fails with Multi-line Python Scripts

**Symptom:** `conda run -n medgemma python -c "..."` fails with `AssertionError: Support for scripts where arguments contain newlines not implemented`.

**Root Cause:** conda's `run` subcommand cannot handle arguments containing newlines — it's a known conda limitation on Windows PowerShell.

**Fix:** Use a separate `.py` script file instead of `-c` inline scripts. Or activate the environment manually:
```powershell
conda activate medgemma
python -c "..."   # works after activation
```

---

## 8. API Reference

### `POST /medgemma/analyze-upload`

Upload DICOM or NIfTI files for analysis.

**Request:** `multipart/form-data`
```
files: List[UploadFile]   (.dcm or .nii or .nii.gz)
```

**Response:**
```json
{
  "id": "a1b2c3d4",
  "status": "completed",
  "report": "CT SCAN ANALYSIS REPORT\n\nAnalyzed: 228 slice(s)\n...",
  "slices_analyzed": 228,
  "has_findings": true,
  "findings": [
    {
      "slice_index": 31,
      "slice_number": 32,
      "response": "HIGH-INTENSITY REGION DETECTED\n\nLocation: Left MCA\n...",
      "image": "/medgemma/finding-image/a1b2c3d4/31",
      "regions": 1,
      "bbox": [120, 85, 165, 130],
      "location": "Left Middle Cerebral Artery",
      "intensity": 234.5
    }
  ],
  "findings_by_location": {
    "Left Middle Cerebral Artery": [ ... ]
  },
  "num_locations": 2,
  "processing_time": 45.3
}
```

---

### `GET /medgemma/finding-image/{analysis_id}/{slice_index}`

Returns the annotated PNG image for a specific finding.

**Response:** `image/png` — full-quality annotated CT slice with red bounding boxes.

---

### `GET /health`

```json
{
  "status": "healthy",
  "medgemma_loaded": false,
  "device": "cuda",
  "gpu_available": true
}
```

---

## 9. Integration Guide (For Other Projects)

### Importing the MedGemma Module Directly

```python
import sys
sys.path.insert(0, "/path/to/train/medgemma")

from inference import MedGemmaInference, detect_modality_from_pixels

# Initialise once (lazy load on first use)
model = MedGemmaInference()

# Analyse a raw pixel array from your existing pipeline
import numpy as np
pixel_array = ...  # your 2D numpy array (HU values or MRI signal)

# Auto-detect modality from pixel statistics
modality = detect_modality_from_pixels(pixel_array)

# Get MedGemma analysis
result = model.analyze_slice(pixel_array, modality=modality, slice_info="Slice 32")
print(result)
```

### Analysing a DICOM File Directly

```python
result = model.analyze_dicom("/path/to/file.dcm")
# Auto-reads DICOM tags to detect modality
```

### Using the 3-Slice Context Window (Better Accuracy)

```python
all_pixel_arrays = [...]  # list of all slices in the scan

result = model.analyze_context_window(
    pixel_arrays=all_pixel_arrays,
    center_idx=32,         # the suspicious slice
    modality="CTA",
    slice_info="Slice 32",
)
```

### REST API Integration

If calling from another application via HTTP:

```python
import requests

url = "http://localhost:8000/medgemma/analyze-upload"
files = [("files", open("slice_032.dcm", "rb"))]
response = requests.post(url, files=files)
data = response.json()

for finding in data["findings"]:
    print(f"Slice {finding['slice_number']}: {finding['location']}")
    print(finding["response"])
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | None | Hugging Face READ token |
| `OFFLINE_MODE` | No | auto | Force offline (`true`) or online (`false`) |
| `CUDA_VISIBLE_DEVICES` | No | all | Restrict to specific GPU(s) |

---

## 10. Data Format Reference

### DICOM Series (Multiple Files)

```
series/
├── slice_001.dcm   ← one .dcm file per axial slice
├── slice_002.dcm
└── slice_228.dcm
```

- Upload **all `.dcm` files** from a series together
- Backend reads `RescaleSlope` and `RescaleIntercept` tags to convert to HU

### NIfTI Volume (Single File)

```
patient/
├── 1.2.826...381.nii           ← UPLOAD THIS — full 3D CT volume
└── 1.2.826...381_cowseg.nii    ← DO NOT upload — binary segmentation mask
```

- One `.nii` file = **entire 3D volume** (all slices stacked)
- `_cowseg.nii` = Circle of Willis segmentation mask (0/1 values only)
- Backend auto-rejects files containing `_cowseg`, `_seg`, or `_mask` in name
- Backend auto-detects binary masks by checking `max value ≤ 1.1`

### NIfTI Orientation

The dataset uses **RAS+ canonical orientation** after nibabel reorientation. Axial slices are at `data[:, :, z]` with:
- X axis: Left → Right (patient)
- Y axis: Posterior → Anterior  
- Z axis: Inferior → Superior (axial slices increase from skull base to vertex)

### HU Value Reference

| Tissue | HU Range | Relevance |
|---|---|---|
| Air | −1000 | Background/air |
| Fat | −100 to −50 | Scalp |
| Water/CSF | 0 to 15 | Ventricles, cisterns |
| Brain (gray) | 35 to 45 | Gray matter |
| Brain (white) | 25 to 35 | White matter |
| Contrast vessel | 200 to 400 | CTA arteries ← aneurysm territory |
| Acute blood | 50 to 80 | SAH |
| Bone | 700 to 3000 | Skull |

---

## Quick Reference Checklist

### First-Time Setup
- [ ] `conda create -n medgemma python=3.10`
- [ ] Install PyTorch 2.1.2+cu118
- [ ] `pip install transformers==4.38.2 accelerate bitsandbytes`
- [ ] `pip install fastapi uvicorn python-multipart pydicom nibabel scipy`
- [ ] Add `HF_TOKEN` to `train/.env`
- [ ] Accept MedGemma terms at https://huggingface.co/google/medgemma-4b-it
- [ ] `cd frontend && npm install`

### Every Run
```bash
# Terminal 1
conda activate medgemma && cd backend && python main.py

# Terminal 2
cd frontend && npm run dev
```

### Uploading Files
- ✅ DICOM: Upload all `.dcm` files from a series
- ✅ NIfTI: Upload only `patientID.nii` (NOT the `_cowseg` version)
- ❌ Do not mix formats in one upload
