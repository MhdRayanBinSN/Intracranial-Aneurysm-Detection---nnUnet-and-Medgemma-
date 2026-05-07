"""
evaluate_from_zip.py
====================
Runs nnU-Net inference on ALL series from the RSNA dataset ZIP
WITHOUT extracting it to disk. Uses Python's zipfile module with BytesIO
to load DICOM files directly into memory.

Saves results to: backend/evaluation_results.json

Usage:
    cd "Code/Pretrained detection/backend"
    conda activate pretrained_detect
    python evaluate_from_zip.py

    Optional flags:
        --limit 50          # Only process first N series (for testing)
        --output my.json    # Custom output path
"""

import argparse
import json
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent
PROJECT_ROOT = Path("C:/Users/Rayan/Desktop/Main Project")
ZIP_PATH     = PROJECT_ROOT / "rsna-intracranial-aneurysm-detection.zip"
TRAIN_CSV    = PROJECT_ROOT / "train.csv"
OUTPUT_JSON  = BACKEND_DIR  / "evaluation_results.json"

LOCATION_LABELS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
]


# ── Setup sys.path so nnunetv2 is importable ──────────────────────────────
SOL_DIR = PROJECT_ROOT / "Code" / "Pretrained detection" / "sol"
sys.path.insert(0, str(SOL_DIR))

import os
MODELS_DIR     = BACKEND_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR  / "checkpoint"
os.environ['nnUNet_raw']          = str(MODELS_DIR / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(MODELS_DIR / "nnUNet_preprocessed")
os.environ['nnUNet_results']      = str(CHECKPOINT_DIR)


# ── In-memory DICOM → SimpleITK volume ────────────────────────────────────

def dicom_bytes_to_sitk(dicom_bytes_list: list) -> sitk.Image:
    """
    Convert a list of (slice_bytes, z_position) tuples to a SimpleITK volume.
    Sorts slices by z_position (ImagePositionPatient[2]) then stacks them.
    """
    slices = []
    for raw_bytes in dicom_bytes_list:
        ds = pydicom.dcmread(BytesIO(raw_bytes))
        try:
            z = float(ds.ImagePositionPatient[2])
        except Exception:
            z = float(getattr(ds, 'InstanceNumber', 0))
        slices.append((z, ds))

    # Sort by z position
    slices.sort(key=lambda x: x[0])

    if not slices:
        raise ValueError("No DICOM slices found")

    # Extract pixel arrays
    arrays = []
    for _, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale if present
        slope  = float(getattr(ds, 'RescaleSlope',  1))
        intcpt = float(getattr(ds, 'RescaleIntercept', 0))
        arr = arr * slope + intcpt
        arrays.append(arr)

    volume = np.stack(arrays, axis=0)  # shape: (Z, Y, X)

    # Build SimpleITK image
    sitk_img = sitk.GetImageFromArray(volume)

    # Set spacing from first slice
    ds0 = slices[0][1]
    try:
        row_spacing, col_spacing = [float(v) for v in ds0.PixelSpacing]
        try:
            slice_thickness = float(ds0.SliceThickness)
        except Exception:
            slice_thickness = abs(slices[1][0] - slices[0][0]) if len(slices) > 1 else 1.0
        sitk_img.SetSpacing([col_spacing, row_spacing, slice_thickness])
    except Exception:
        pass  # leave default spacing

    # Set origin from first slice ImagePositionPatient
    try:
        ipp = [float(v) for v in ds0.ImagePositionPatient]
        sitk_img.SetOrigin(ipp)
    except Exception:
        pass

    # Set direction from ImageOrientationPatient
    try:
        iop = [float(v) for v in ds0.ImageOrientationPatient]
        F = iop[:3]
        R = iop[3:]
        # Third direction: cross product
        N = [
            R[1]*F[2] - R[2]*F[1],
            R[2]*F[0] - R[0]*F[2],
            R[0]*F[1] - R[1]*F[0],
        ]
        direction = F + R + N
        sitk_img.SetDirection(direction)
    except Exception:
        pass

    return sitk_img


# ── Load model ─────────────────────────────────────────────────────────────

def load_model():
    from inference import get_inference_engine
    engine = get_inference_engine()
    return engine


# ── Run inference on an in-memory SimpleITK image ─────────────────────────

def predict_from_dcm_bytes(engine, dicom_bytes_list: list, series_uid: str) -> dict:
    """
    Run inference on DICOM raw bytes read from the ZIP.
    Writes the bytes as real .dcm files to a temp directory,
    then calls engine.predict(tmp_dir) — which does glob('*.dcm').
    Temp dir is deleted immediately after inference.
    """
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write each DICOM slice as a .dcm file
        for idx, raw_bytes in enumerate(dicom_bytes_list):
            dest = tmp_path / f"slice_{idx:05d}.dcm"
            dest.write_bytes(raw_bytes)

        # Now call inference — process_series will glob('*.dcm') and find them
        result = engine.predict(tmp_path)

    return result


# ── Main evaluation loop ───────────────────────────────────────────────────

def evaluate(limit: int = None, output_path: Path = OUTPUT_JSON):
    print(f"\n{'='*60}")
    print("RSNA Aneurysm Detection — ZIP-based Evaluation")
    print(f"{'='*60}")

    # -- Load ground truth labels --
    print(f"\nLoading train.csv from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df = df.set_index('SeriesInstanceUID')
    print(f"  Loaded {len(df)} series labels")

    # -- Open ZIP (no extraction) --
    print(f"\nOpening ZIP: {ZIP_PATH}")
    zf = zipfile.ZipFile(ZIP_PATH, 'r')

    # -- Discover series in ZIP --
    # Expected structure: train/<SeriesUID>/<file>.dcm
    all_names = zf.namelist()
    series_map = defaultdict(list)  # series_uid -> list of zip paths
    for name in all_names:
        parts = name.replace('\\', '/').split('/')
        if len(parts) >= 3 and name.lower().endswith('.dcm'):
            series_uid = parts[1]  # assumes train/seriesUID/file.dcm
            series_map[series_uid].append(name)

    print(f"  Found {len(series_map)} series in ZIP")

    # -- Limit if requested --
    series_uids = list(series_map.keys())
    if limit:
        series_uids = series_uids[:limit]
        print(f"  Processing first {limit} series (--limit flag)")

    # -- Load model --
    print("\nLoading nnU-Net model...")
    engine = load_model()

    # -- Per-series evaluation --
    results = []
    total = len(series_uids)
    overall_tp = overall_fp = overall_fn = overall_tn = 0

    # Per-location counts
    loc_tp = defaultdict(int)
    loc_fp = defaultdict(int)
    loc_fn = defaultdict(int)
    loc_tn = defaultdict(int)

    start_time = time.time()

    for i, uid in enumerate(series_uids):
        elapsed = time.time() - start_time
        eta = (elapsed / max(i, 1)) * (total - i)
        print(f"\n[{i+1}/{total}] {uid[:40]}... | ETA: {eta/60:.1f} min")

        # Fetch ground truth row
        if uid not in df.index:
            print(f"  SKIP: not in train.csv")
            continue
        gt_row = df.loc[uid]
        gt_overall = int(gt_row.get('Aneurysm Present', 0))

        # Read all DICOM files for this series from ZIP into memory
        dicom_paths = series_map[uid]
        dicom_bytes_list = []
        for zpath in dicom_paths:
            try:
                raw = zf.read(zpath)
                # Validate it's a real DICOM before adding (force=True handles missing headers)
                try:
                    pydicom.dcmread(BytesIO(raw), stop_before_pixels=True, force=True)
                    dicom_bytes_list.append(raw)
                except Exception:
                    pass  # Skip non-DICOM files in the ZIP (e.g. READMEs)
            except Exception as e:
                print(f"  WARN: could not read {zpath}: {e}")

        if not dicom_bytes_list:
            print(f"  SKIP: no readable DICOM files")
            continue

        # Run inference — writes .dcm bytes to tmpdir, calls process_series
        try:
            pred = predict_from_dcm_bytes(engine, dicom_bytes_list, uid)
        except Exception as e:
            print(f"  SKIP: inference failed: {e}")
            continue

        # --- Compare predictions vs ground truth ---
        # Overall aneurysm presence
        pred_overall = 1 if pred.get('overall_risk') in ('High', 'Moderate') else 0
        if   gt_overall == 1 and pred_overall == 1: overall_tp += 1
        elif gt_overall == 0 and pred_overall == 1: overall_fp += 1
        elif gt_overall == 1 and pred_overall == 0: overall_fn += 1
        else:                                       overall_tn += 1

        # Per-location
        loc_results = {}
        for loc in LOCATION_LABELS:
            gt_loc = int(gt_row.get(loc, 0))
            # Find matching prediction
            pred_loc = 0
            for pp in pred.get('predictions', []):
                if pp.get('location') == loc and pp.get('detected', False):
                    pred_loc = 1
                    break

            if   gt_loc == 1 and pred_loc == 1: loc_tp[loc] += 1
            elif gt_loc == 0 and pred_loc == 1: loc_fp[loc] += 1
            elif gt_loc == 1 and pred_loc == 0: loc_fn[loc] += 1
            else:                               loc_tn[loc] += 1

            loc_results[loc] = {'gt': gt_loc, 'pred': pred_loc}

        results.append({
            'series_uid': uid,
            'gt_overall': gt_overall,
            'pred_overall': pred_overall,
            'locations': loc_results,
        })

        # Print running metrics
        processed = i + 1
        acc = (overall_tp + overall_tn) / max(processed, 1)
        print(f"  GT={gt_overall} PRED={pred_overall} | Running: TP={overall_tp} FP={overall_fp} FN={overall_fn} TN={overall_tn} ACC={acc:.2%}")

    zf.close()

    # ── Compute summary metrics ─────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results computed.")
        return

    def safe_div(a, b): return round(a / b, 4) if b > 0 else 0.0

    sensitivity  = safe_div(overall_tp, overall_tp + overall_fn)  # recall
    specificity  = safe_div(overall_tn, overall_tn + overall_fp)
    precision    = safe_div(overall_tp, overall_tp + overall_fp)
    f1           = safe_div(2 * overall_tp, 2 * overall_tp + overall_fp + overall_fn)
    accuracy     = safe_div(overall_tp + overall_tn, n)

    per_location_summary = {}
    for loc in LOCATION_LABELS:
        tp = loc_tp[loc]; fp = loc_fp[loc]; fn = loc_fn[loc]; tn = loc_tn[loc]
        per_location_summary[loc] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'sensitivity': safe_div(tp, tp + fn),
            'precision':   safe_div(tp, tp + fp),
            'f1':          safe_div(2*tp, 2*tp + fp + fn),
        }

    summary = {
        'evaluated_series': n,
        'total_series_in_csv': len(df),
        'evaluation_time_sec': round(time.time() - start_time, 1),
        'overall': {
            'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn, 'tn': overall_tn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision':   precision,
            'f1_score':    f1,
            'accuracy':    accuracy,
        },
        'per_location': per_location_summary,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # ── Save ───────────────────────────────────────────────────────────
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE — {n} series processed")
    print(f"  TP={overall_tp}  FP={overall_fp}  FN={overall_fn}  TN={overall_tn}")
    print(f"  Sensitivity: {sensitivity:.2%}   Specificity: {specificity:.2%}")
    print(f"  Precision:   {precision:.2%}   F1: {f1:.2%}   Accuracy: {accuracy:.2%}")
    print(f"  Results saved → {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate nnU-Net on RSNA ZIP without extracting')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of series (for testing, e.g. --limit 50)')
    parser.add_argument('--output', type=str, default=str(OUTPUT_JSON),
                        help='Output JSON path')
    args = parser.parse_args()
    evaluate(limit=args.limit, output_path=Path(args.output))
