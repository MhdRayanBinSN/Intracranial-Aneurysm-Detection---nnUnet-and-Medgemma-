"""
FastAPI Backend — MedGemma Intracranial Aneurysm Detection.

Run with:
    cd backend
    conda activate medgemma
    python main.py
    (or: uvicorn main:app --reload --host 0.0.0.0 --port 8000)
"""

from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import uvicorn
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import uuid
import signal
import atexit
import csv
import ast

GROUND_TRUTH_LOCALIZERS = {}
train_localizers_path = r"C:\Users\Rayan\Desktop\IA\GorundTurth\train_localizers.csv"
if os.path.exists(train_localizers_path):
    try:
        with open(train_localizers_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sop_uid = row.get("SOPInstanceUID", "")
                if sop_uid:
                    coords = ast.literal_eval(row.get("coordinates", "{}"))
                    loc = row.get("location", "")
                    if sop_uid not in GROUND_TRUTH_LOCALIZERS:
                        GROUND_TRUTH_LOCALIZERS[sop_uid] = []
                    GROUND_TRUTH_LOCALIZERS[sop_uid].append({
                        "coordinates": coords,
                        "location": loc
                    })
    except Exception as e:
        print(f"Failed to load ground truth localizers: {e}")

GROUND_TRUTH_SERIES = {}
train_series_path = r"C:\Users\Rayan\Desktop\IA\GorundTurth\train.csv"
if os.path.exists(train_series_path):
    try:
        with open(train_series_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                series_uid = row.get("SeriesInstanceUID", "")
                if series_uid:
                    # Parse numerical fields to int
                    parsed_row = {k: (int(v) if v.isdigit() else v) for k, v in row.items()}
                    GROUND_TRUTH_SERIES[series_uid] = parsed_row
    except Exception as e:
        print(f"Failed to load ground truth series: {e}")


# ─────────────────────────────────────────────
# Path setup — medgemma module
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "medgemma"))

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# MedGemma model (lazy-loaded on first request)
# ─────────────────────────────────────────────
medgemma_model = None
medgemma_model_error: Optional[str] = None

# In-memory store for analysis results
medgemma_store: Dict[str, Any] = {}


# ─────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    medgemma_loaded: bool
    device: str
    gpu_available: bool


class MedGemmaFinding(BaseModel):
    slice_index: int
    slice_number: int
    response: str


class MedGemmaResponse(BaseModel):
    id: str
    status: str
    report: str
    slices_analyzed: int
    has_findings: bool
    findings: List[MedGemmaFinding]
    processing_time: float


# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
def cleanup_resources():
    global medgemma_model
    try:
        print("🧹 Cleaning up resources...")
        medgemma_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")


atexit.register(cleanup_resources)
signal.signal(signal.SIGINT,  lambda s, f: (cleanup_resources(), exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (cleanup_resources(), exit(0)))


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting MedGemma Backend...")
    print(f"📱 Device: {device} | GPU: {torch.cuda.is_available()}")
    print("ℹ️  MedGemma model will load on first upload request (lazy loading).")
    yield
    print("🛑 Shutting down MedGemma Backend...")
    cleanup_resources()


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="MedGemma Aneurysm Detection API",
    description="Intracranial aneurysm detection powered by Google MedGemma",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MedGemma lazy loader
# ─────────────────────────────────────────────
def get_medgemma():
    """Lazy-load MedGemma. Raises HTTP 500 if it fails."""
    global medgemma_model, medgemma_model_error

    if medgemma_model_error:
        raise HTTPException(
            status_code=500,
            detail=f"MedGemma failed to load: {medgemma_model_error}",
        )

    if medgemma_model is None:
        try:
            print("🔄 Loading MedGemma model (first request)...")
            from inference import MedGemmaInference
            medgemma_model = MedGemmaInference()
            print("✅ MedGemma model loaded!")
        except Exception as e:
            medgemma_model_error = str(e)
            print(f"❌ MedGemma loading failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"MedGemma failed to load: {e}",
            )

    return medgemma_model


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — also shows if MedGemma is already loaded."""
    return HealthResponse(
        status="healthy",
        medgemma_loaded=medgemma_model is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────
# MAIN ENDPOINT: Upload & Analyze
# ─────────────────────────────────────────────
@app.post("/medgemma/analyze-upload")
async def analyze_uploaded_files(files: List[UploadFile] = File(...)):
    """
    Upload DICOM (.dcm) or NIfTI (.nii / .nii.gz) slices.
    Returns bounding-box annotated images + MedGemma text analysis for each finding.
    """
    import time
    from io import BytesIO
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from scipy.ndimage import label, find_objects

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    print(f"📤 Received {len(files)} file(s) for analysis [id={analysis_id}]")

    # Temp folder for annotated images
    analysis_img_dir = os.path.join(tempfile.gettempdir(), "medgemma_images", analysis_id)
    os.makedirs(analysis_img_dir, exist_ok=True)

    # ── 1. Parse uploaded files into pixel arrays ──────────────────────────────
    all_slices: List[Dict] = []

    for file in files:
        fname = file.filename.lower()
        try:
            content = await file.read()

            if fname.endswith((".nii", ".nii.gz")):
                # ⚠️ SKIP segmentation masks — _cowseg.nii files are binary
                # masks (0/1 only), NOT CT scans. Uploading them gives garbage results.
                if "_cowseg" in fname or "_seg" in fname or "_mask" in fname:
                    print(f"⏭️  Skipping segmentation mask: {file.filename}")
                    print(f"   (Upload the plain .nii scan file, not the _cowseg version)")
                    continue

                try:
                    import nibabel as nib

                    # Save to temp file (nibabel needs a real file path)
                    suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name

                    nii = nib.load(tmp_path)

                    # ✅ Reorient to RAS+ canonical space so axial slices
                    # match DICOM radiological convention (Left=right side of image)
                    nii_canonical = nib.as_closest_canonical(nii)
                    data = nii_canonical.get_fdata(dtype=np.float32)
                    os.unlink(tmp_path)

                    if len(data.shape) == 3:
                        num_z = data.shape[2]
                        print(f"📦 NIfTI volume: {file.filename}")
                        print(f"   Shape: {data.shape} | HU range: [{data.min():.0f}, {data.max():.0f}]")

                        # Sanity check: if max value ≤ 1 → it's a binary mask, reject
                        if data.max() <= 1.1:
                            print(f"   ❌ Looks like a binary mask (max={data.max():.2f}) — skipping.")
                            print(f"      Upload the plain scan .nii, not the _cowseg version.")
                            continue

                        for z in range(num_z):
                            arr = data[:, :, z]

                            # Skip blank/air-only slices (background = very negative HU ≈ -1000)
                            # A real brain slice has some tissue above -200 HU
                            if np.percentile(arr, 95) < -200:
                                continue  # All air — skip

                            all_slices.append({
                                "pixel_array": arr,
                                "filename":          f"{file.filename} [z={z+1}/{num_z}]",
                                "original_filename": file.filename,
                            })

                        print(f"   ✅ Extracted {len([s for s in all_slices if file.filename in s['original_filename']])} valid slices")
                    else:
                        print(f"⚠️ Unexpected NIfTI shape {data.shape} — skipping.")

                except ImportError:
                    print("⚠️ nibabel not installed — skipping NIfTI file.")
                    print("   Install with: pip install nibabel")
                except Exception as e:
                    print(f"⚠️ NIfTI error ({fname}): {e}")

            elif fname.endswith(".dcm"):
                try:
                    import pydicom
                    dcm = pydicom.dcmread(BytesIO(content))
                    arr = dcm.pixel_array.astype(np.float32)
                    arr = arr * float(getattr(dcm, "RescaleSlope", 1)) \
                            + float(getattr(dcm, "RescaleIntercept", 0))
                    all_slices.append({
                        "pixel_array": arr,
                        "filename": file.filename,
                        "original_filename": file.filename,
                        "series_uid": str(getattr(dcm, "SeriesInstanceUID", "")),
                        "sop_uid": str(getattr(dcm, "SOPInstanceUID", "")),
                    })
                except Exception as e:
                    print(f"⚠️ DICOM error ({fname}): {e}")
            # Non-medical formats silently skipped

        except Exception as e:
            print(f"Error reading {file.filename}: {e}")

    # ── 2. Process each slice ──────────────────────────────────────────────────
    findings:        List[Dict] = []
    slices_analyzed: int        = 0

    for idx, slice_data in enumerate(all_slices):
        try:
            pixel_array = slice_data["pixel_array"]
            slices_analyzed += 1
            h, w = pixel_array.shape

            # ── Preprocessing (CLAHE multi-window or fallback) ──────────────
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
                from data.advanced_preprocessing import KaggleWinningPreprocessor
                prep = KaggleWinningPreprocessor()

                modality = prep.detect_modality(pixel_array)

                if modality == "CT":
                    rgb = prep.create_multi_window_image(pixel_array)
                    if prep.use_clahe:
                        for c in range(3):
                            rgb[:, :, c] = prep.apply_clahe(rgb[:, :, c])
                    rgb = (rgb * 255).astype(np.uint8)
                    windowed_for_detection = rgb[:, :, 1]   # vessel channel
                    base_threshold = 180
                else:
                    rgb_norm = prep.preprocess_mri(pixel_array)
                    rgb = (rgb_norm * 255).astype(np.uint8)
                    windowed_for_detection = rgb[:, :, 0]
                    base_threshold = float(np.percentile(windowed_for_detection, 95))

            except Exception as e:
                print(f"⚠️ Advanced preprocessing error: {e}. Using fallback.")
                wc, ww = 300, 600
                lo, hi = wc - ww / 2, wc + ww / 2
                arr_w = np.clip(pixel_array, lo, hi)
                arr_w = ((arr_w - lo) / (hi - lo) * 255).astype(np.uint8)
                rgb = np.stack([arr_w] * 3, axis=-1)
                windowed_for_detection = arr_w
                modality = "CT"
                base_threshold = 200

            # ── Aneurysm-specific region detection ─────────────────────────
            detection_threshold = float(np.percentile(windowed_for_detection, 99))
            detection_threshold = max(detection_threshold, base_threshold)

            binary = windowed_for_detection > detection_threshold
            binary = ndimage.binary_opening(binary, iterations=3)
            binary = ndimage.binary_closing(binary, iterations=2)

            labeled_array, _ = label(binary)
            regions = find_objects(labeled_array)

            heatmap = np.zeros((h, w), dtype=np.float32)
            detected_regions: List[Dict] = []

            for i, region in enumerate(regions):
                if region is None:
                    continue

                y_sl, x_sl = region
                y_min, y_max = y_sl.start, y_sl.stop
                x_min, x_max = x_sl.start, x_sl.stop

                rw = x_max - x_min
                rh = y_max - y_min

                # Size filter (aneurysms ≈ 3–25 mm)
                if not (15 <= rw <= 120 and 15 <= rh <= 120):
                    continue

                # Central brain only (Circle of Willis)
                cx_r = (x_min + x_max) / 2
                cy_r = (y_min + y_max) / 2
                if cx_r < w * 0.15 or cx_r > w * 0.85:
                    continue
                if cy_r < h * 0.15 or cy_r > h * 0.85:
                    continue

                region_mask  = labeled_array[y_min:y_max, x_min:x_max] == (i + 1)
                pixel_count  = int(np.sum(region_mask))
                bbox_area    = rw * rh

                if pixel_count < 50:
                    continue

                circularity  = pixel_count / bbox_area if bbox_area > 0 else 0
                if circularity < 0.35:
                    continue

                aspect_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
                if aspect_ratio > 3.0:
                    continue

                region_intensity = float(
                    np.mean(windowed_for_detection[y_min:y_max, x_min:x_max][region_mask])
                )
                if region_intensity <= detection_threshold:
                    continue

                detected_regions.append({
                    "bbox": (x_min, y_min, x_max, y_max),
                    "intensity": region_intensity,
                    "area": pixel_count,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "modality": modality,
                })

                # Gaussian heatmap blob
                cy_c = (y_min + y_max) // 2
                cx_c = (x_min + x_max) // 2
                radius = max(rw, rh)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = cy_c + dy, cx_c + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dist = np.sqrt(dy ** 2 + dx ** 2)
                            if dist <= radius:
                                heatmap[ny, nx] += max(0.0, 1 - dist / radius)

            # Apply heatmap overlay to RGB
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
                rgb[:, :, 0] = np.minimum(
                    255, rgb[:, :, 0].astype(np.float32) + heatmap * 150
                ).astype(np.uint8)
                rgb[:, :, 1] = np.maximum(
                    0, rgb[:, :, 1].astype(np.float32) - heatmap * 50
                ).astype(np.uint8)

            sop_uid = slice_data.get("sop_uid", "")
            ground_truth = []
            if sop_uid and sop_uid in GROUND_TRUTH_LOCALIZERS:
                ground_truth = GROUND_TRUTH_LOCALIZERS[sop_uid]

            # ── Save and analyse only slices with findings OR ground truth ────────────────
            if not detected_regions and not ground_truth:
                continue

            # Draw bounding boxes
            img_pil = Image.fromarray(rgb)
            draw    = ImageDraw.Draw(img_pil)
            for reg in detected_regions:
                x0, y0, x1, y1 = reg["bbox"]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                draw.text((x0, y0 - 12), "ROI", fill=(255, 255, 0))
            
            # Draw ground truth if present
            for gt in ground_truth:
                gt_x = gt["coordinates"].get("x", 0)
                gt_y = gt["coordinates"].get("y", 0)
                if gt_x and gt_y:
                    # Draw a distinct green circle/box for ground truth
                    r = 15
                    draw.ellipse([gt_x-r, gt_y-r, gt_x+r, gt_y+r], outline=(0, 255, 0), width=2)
                    draw.text((gt_x, gt_y - r - 12), "Ground Truth", fill=(0, 255, 0))

            img_path = os.path.join(analysis_img_dir, f"slice_{idx}.png")
            img_pil.save(img_path, format="PNG")
            img_url  = f"/medgemma/finding-image/{analysis_id}/{idx}"

            print(f"  >> Slice {idx+1}: {len(detected_regions)} suspicious region(s) found, {len(ground_truth)} ground truth")

            if detected_regions:
                # Pick top region by intensity
                detected_regions.sort(key=lambda r: r["intensity"], reverse=True)
                top = detected_regions[0]
                x_min, y_min, x_max, y_max = top["bbox"]
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Anatomical localisation (radiological convention)
                side = "Right" if center_x < w * 0.5 else "Left"

                if center_y < h * 0.3:
                    location_name = (
                        "Anterior Communicating Artery"
                        if abs(center_x - w * 0.5) < w * 0.15
                        else f"{side} Anterior Cerebral Artery"
                    )
                elif center_y < h * 0.45:
                    location_name = (
                        f"{side} Middle Cerebral Artery"
                        if abs(center_x - w * 0.5) > w * 0.25
                        else f"{side} Supraclinoid Internal Carotid Artery"
                    )
                elif center_y < h * 0.65:
                    location_name = (
                        f"{side} Infraclinoid Internal Carotid Artery"
                        if abs(center_x - w * 0.5) > w * 0.2
                        else f"{side} Posterior Communicating Artery"
                    )
                else:
                    location_name = (
                        "Basilar Tip"
                        if abs(center_x - w * 0.5) < w * 0.1
                        else "Other Posterior Circulation"
                    )
            else:
                top = {"bbox": (0, 0, 0, 0), "intensity": 0}
                location_name = ground_truth[0]["location"] if ground_truth else "Unknown"

            # ── MedGemma LLM analysis ─────────────────────────────────────
            if not detected_regions:
                response_text = f"False Negative: Ground truth aneurysm present at {location_name} but not detected by the model."
            else:
                try:
                    medgemma = get_medgemma()

                    # Crop around the top region with padding
                    pad = 50
                    crop_box = (
                        max(0, top["bbox"][0] - pad),
                        max(0, top["bbox"][1] - pad),
                        min(w, top["bbox"][2] + pad),
                        min(h, top["bbox"][3] + pad),
                    )
                    cropped_img = Image.fromarray(rgb).crop(crop_box)

                    llm_analysis = medgemma.analyze_for_kaggle(
                        cropped_img,
                        modality=modality,
                        slice_info=f"Slice {idx+1} ({slice_data['original_filename']})",
                    )

                    response_text = (
                        f"HIGH-INTENSITY REGION DETECTED\n\n"
                        f"Location: {location_name}\n"
                        f"Bounding Box: ({x_min}, {y_min}) → ({x_max}, {y_max})\n"
                        f"Max Intensity: {top['intensity']:.0f}\n\n"
                        f"MEDGEMMA ANALYSIS:\n{llm_analysis}\n\n"
                        f"Note: Automated detection — requires radiologist review."
                    )
                except Exception as e:
                    print(f"MedGemma analysis error (slice {idx+1}): {e}")
                    response_text = (
                        f"HIGH-INTENSITY REGION DETECTED\n\n"
                        f"Location: {location_name}\n"
                        f"Bounding Box: ({x_min}, {y_min}) → ({x_max}, {y_max})\n"
                        f"Max Intensity: {top['intensity']:.0f}\n\n"
                        f"(MedGemma analysis unavailable: {e})"
                    )

            findings.append({
                "slice_index":  idx,
                "slice_number": idx + 1,
                "response":     response_text,
                "image":        img_url,
                "regions":      len(detected_regions),
                "bbox":         top["bbox"],
                "location":     location_name,
                "intensity":    top["intensity"],
                "ground_truth": ground_truth,
            })

        except Exception as e:
            print(f"Error processing slice {idx+1}: {e}")
            continue

    # ── 3. Build response ──────────────────────────────────────────────────────
    processing_time = time.time() - start_time

    findings_by_location: Dict = defaultdict(list)
    for f in findings:
        findings_by_location[f["location"]].append({
            "slice_index":  f["slice_index"],
            "slice_number": f["slice_number"],
            "response":     f["response"],
            "image":        f["image"],
            "bbox":         f["bbox"],
            "intensity":    f.get("intensity", 0),
            "regions":      f.get("regions", 0),
            "ground_truth": f.get("ground_truth", []),
        })
    for loc in findings_by_location:
        findings_by_location[loc].sort(key=lambda x: x["slice_number"])

    num_locations          = len(findings_by_location)
    total_detection_slices = len(findings)

    if findings:
        total_regions = sum(f.get("regions", 0) for f in findings)
        report = (
            f"CT SCAN ANALYSIS REPORT\n\n"
            f"Analyzed: {slices_analyzed} slice(s)\n"
            f"Findings: {total_detection_slices} slice(s) with detections across "
            f"{num_locations} location(s)\n"
            f"Total ROIs: {total_regions} region(s) of interest\n\n"
            f"Locations with findings:\n"
            + "\n".join(
                f"  - {loc}: {len(s)} slice(s)"
                for loc, s in findings_by_location.items()
            )
            + "\n\nDetection Method: Intensity-based region segmentation + MedGemma LLM\n"
            + "All findings require verification by a qualified radiologist."
        )
    else:
        report = (
            f"CT SCAN ANALYSIS REPORT\n\n"
            f"Analyzed: {slices_analyzed} slice(s)\n"
            f"No significant high-intensity regions detected.\n\n"
            f"Detection Method: Intensity-based region segmentation\n"
            f"Note: Absence of findings does not rule out pathology."
        )

    series_ground_truth = None
    if all_slices:
        for slice_data in all_slices:
            uid = slice_data.get("series_uid", "")
            if not uid:
                fname = slice_data.get("original_filename", "")
                uid = fname.replace(".nii.gz", "").replace(".nii", "")
            if uid in GROUND_TRUTH_SERIES:
                series_ground_truth = GROUND_TRUTH_SERIES[uid]
                break

    return {
        "id":                    analysis_id,
        "status":                "completed",
        "report":                report,
        "slices_analyzed":       slices_analyzed,
        "has_findings":          len(findings) > 0,
        "findings":              findings,
        "findings_by_location":  dict(findings_by_location),
        "num_locations":         num_locations,
        "processing_time":       processing_time,
        "series_ground_truth":   series_ground_truth,
    }


# ─────────────────────────────────────────────
# Serve annotated finding images
# ─────────────────────────────────────────────
@app.get("/medgemma/finding-image/{analysis_id}/{slice_index}")
async def get_finding_image(analysis_id: str, slice_index: int):
    """Return a saved annotated PNG for a specific finding."""
    img_path = os.path.join(
        tempfile.gettempdir(), "medgemma_images", analysis_id, f"slice_{slice_index}.png"
    )
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(img_path, media_type="image/png")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
