"""
Run Detection - Main Entry Point
=================================
Combines all pipeline stages to run aneurysm detection.

Usage:
    python run_detection.py --input <DICOM_FOLDER> --output <OUTPUT_JSON>
"""

import sys
import json
import argparse
from pathlib import Path

# Add sol folder to path for nnU-Net imports
sys.path.insert(0, str(Path(__file__).parent.parent / "sol"))

import torch
import numpy as np

from config import DEVICE, CHECKPOINT_FILE, LOCATION_LABELS


def load_model():
    """
    Load the trained nnU-Net model.
    
    Returns:
        Loaded model ready for inference
    """
    print("=" * 50)
    print("LOADING MODEL")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            f"Download from: https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint"
        )
    
    # Import nnU-Net predictor
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    
    # Load weights
    from config import CHECKPOINT_DIR, CHECKPOINT_FILE
    checkpoint_dir = CHECKPOINT_FILE.parent.parent  # fold_all/checkpoint.pth → trainer_dir
    
    predictor.initialize_from_trained_model_folder(
        str(checkpoint_dir),
        use_folds=('all',),
        checkpoint_name=CHECKPOINT_FILE.name,
    )
    
    print("✅ Model loaded successfully!")
    return predictor


def run_detection(dicom_folder: Path, output_path: Path = None) -> dict:
    """
    Run the full detection pipeline.
    
    Steps:
        1. Load model
        2. Load and preprocess DICOM
        3. Run inference
        4. Postprocess results
        
    Args:
        dicom_folder: Path to DICOM series folder
        output_path: Optional path to save JSON results
        
    Returns:
        Detection results dictionary
    """
    from datetime import datetime
    
    print("\n" + "=" * 60)
    print("INTRACRANIAL ANEURYSM DETECTION PIPELINE")
    print("=" * 60)
    print(f"Input: {dicom_folder}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # ================================================
    # STEP 1: Load Model
    # ================================================
    model = load_model()
    
    # ================================================
    # STEP 2: Load and Preprocess DICOM
    # ================================================
    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)
    
    # Use nnU-Net's preprocessing
    from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import process_series
    import SimpleITK as sitk
    
    # Load DICOM
    print(f"Loading DICOM from: {dicom_folder}")
    image_sitk = process_series(dicom_folder)
    
    img_array = sitk.GetArrayFromImage(image_sitk)  # (Z, Y, X)
    spacing = np.array(image_sitk.GetSpacing()[::-1])  # (Z, Y, X)
    
    print(f"  Shape: {img_array.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # Prepare properties for nnU-Net preprocessor
    properties = {
        "spacing": spacing,
        "direction": image_sitk.GetDirection(),
        "origin": image_sitk.GetOrigin(),
    }
    
    # Run nnU-Net preprocessing
    preprocessor = model.configuration_manager.preprocessor_class()
    data, _, _ = preprocessor.run_case_npy(
        np.array([img_array]),  # Add channel dimension
        None,
        properties,
        model.plans_manager,
        model.configuration_manager,
        model.dataset_json,
    )
    
    print(f"  Preprocessed shape: {data.shape}")
    
    # ================================================
    # STEP 3: Run Inference
    # ================================================
    print("\n" + "=" * 50)
    print("INFERENCE")
    print("=" * 50)
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        logits = model.predict_logits_from_preprocessed_data(
            torch.from_numpy(data)
        ).cpu()
    
    print(f"  Output shape: {logits.shape}")
    
    # Convert to probabilities
    probs = torch.sigmoid(logits)
    
    # ================================================
    # STEP 4: Postprocess
    # ================================================
    print("\n" + "=" * 50)
    print("POSTPROCESSING")
    print("=" * 50)
    
    # Extract per-location probabilities (take max from each channel)
    per_location = {}
    for i, label in enumerate(LOCATION_LABELS):
        max_prob = probs[0, i].max().item()
        per_location[label] = round(max_prob, 4)
        if max_prob > 0.1:
            print(f"  {label}: {max_prob:.1%}")
    
    # Get max probability and detection status
    max_prob = max(per_location.values())
    max_location = max(per_location.items(), key=lambda x: x[1])[0]
    
    # Get detections above threshold (50%)
    detections = []
    for loc, prob in per_location.items():
        if prob >= 0.50:
            detections.append({
                "location": loc,
                "probability": prob,
            })
    
    detections.sort(key=lambda x: x["probability"], reverse=True)
    
    # Determine risk level
    if max_prob >= 0.70:
        risk_level = "HIGH"
    elif max_prob >= 0.50:
        risk_level = "MODERATE"
    elif max_prob >= 0.30:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    # Compile results
    results = {
        "input_folder": str(dicom_folder),
        "timestamp": datetime.now().isoformat(),
        "aneurysm_detected": len(detections) > 0,
        "max_probability": round(max_prob, 4),
        "max_location": max_location,
        "risk_level": risk_level,
        "detections": detections,
        "per_location_probabilities": per_location,
    }
    
    # ================================================
    # STEP 5: Output Results
    # ================================================
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    if results["aneurysm_detected"]:
        print(f"⚠️  ANEURYSM DETECTED")
        print(f"Risk Level: {risk_level}")
        print(f"Highest Probability: {max_prob:.1%} at {max_location}")
        print("\nDetections:")
        for det in detections:
            print(f"  • {det['location']}: {det['probability']:.1%}")
    else:
        print(f"✓ No aneurysm detected")
        print(f"Maximum probability: {max_prob:.1%}")
    
    print("=" * 60)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Run intracranial aneurysm detection on DICOM images"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to DICOM folder"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input folder not found: {args.input}")
        sys.exit(1)
    
    results = run_detection(args.input, args.output)
    
    return results


if __name__ == "__main__":
    main()
