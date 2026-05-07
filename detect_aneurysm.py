#!/usr/bin/env python
"""
Standalone CLI Script for Intracranial Aneurysm Detection

Uses the MIC-DKFZ 7th place Kaggle solution (nnU-Net) for detection.
No frontend/backend required - just run directly on DICOM folders.

Usage:
    python detect_aneurysm.py --input <DICOM_FOLDER> --output <OUTPUT_FOLDER>

Example:
    python detect_aneurysm.py --input ./my_cta_scan --output ./results
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# =============================================================================
# SETUP nnU-Net ENVIRONMENT
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "backend" / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoint"

# Set nnU-Net environment variables BEFORE importing nnunetv2
os.environ['nnUNet_raw'] = str(MODELS_DIR / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(MODELS_DIR / "nnUNet_preprocessed")
os.environ['nnUNet_results'] = str(CHECKPOINT_DIR)

# Add sol folder to path for nnunetv2
SOL_PATH = SCRIPT_DIR / "sol"
if str(SOL_PATH) not in sys.path:
    sys.path.insert(0, str(SOL_PATH))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
DATASET_NAME = "Dataset004_iarsna_crop"
TRAINER_NAME = "Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32"
CHECKPOINT_FILE = "checkpoint_epoch_1500.pth"
FOLD_DIR = "fold_all"

# Anatomical locations (same order as model training)
LOCATION_LABELS = [
    'Other Posterior Circulation',
    'Basilar Tip',
    'Right Posterior Communicating Artery',
    'Left Posterior Communicating Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Infraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Middle Cerebral Artery',
    'Left Middle Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Anterior Communicating Artery',
]


class AneurysmDetector:
    """
    Standalone aneurysm detector using nnU-Net.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = 'cuda'):
        """Initialize detector with model weights."""
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / DATASET_NAME / TRAINER_NAME)
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)
        
        print(f"🔧 Device: {self.device}")
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the nnU-Net model."""
        checkpoint_file = self.checkpoint_path / FOLD_DIR / CHECKPOINT_FILE
        
        if not checkpoint_file.exists():
            print(f"❌ ERROR: Checkpoint not found at {checkpoint_file}")
            print("   Download from: https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint")
            return
        
        print(f"✓ Found checkpoint: {checkpoint_file.name}")
        
        try:
            # Register custom trainer class
            try:
                from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_rsna.Kaggle2025RSNATrainer import Kaggle2025RSNATrainer
                print("✓ Registered Kaggle2025RSNATrainer")
            except ImportError as e:
                print(f"⚠️  Custom trainer import warning: {e}")
            
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch
            
            print("⏳ Initializing nnU-Net predictor...")
            self.model = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,  # Disable TTA for faster inference
                perform_everything_on_device=False,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )
            
            print("⏳ Loading model weights...")
            self.model.initialize_from_trained_model_folder(
                str(self.checkpoint_path),
                use_folds=('all',),
                checkpoint_name=CHECKPOINT_FILE,
            )
            
            # Get preprocessor
            self.preprocessor = self.model.configuration_manager.preprocessor_class()
            
            self.is_loaded = True
            print("✅ Model loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def detect(self, dicom_folder: Path) -> Dict:
        """
        Run aneurysm detection on a DICOM CTA scan.
        
        Args:
            dicom_folder: Path to folder containing DICOM files
            
        Returns:
            Dictionary with detection results
        """
        import torch
        
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        print(f"📂 Processing: {dicom_folder}")
        
        try:
            # Import required modules
            from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import process_series
            import SimpleITK as sitk
            
            # Step 1: Load DICOM series
            print("   [1/4] Loading DICOM series...")
            image_sitk = process_series(dicom_folder)
            print(f"         Shape: {image_sitk.GetSize()}, Spacing: {image_sitk.GetSpacing()}")
            
            # Handle 2D images
            if image_sitk.GetDimension() == 2:
                print("   ⚠️  2D image detected, promoting to 3D...")
                arr = sitk.GetArrayFromImage(image_sitk)
                arr = arr[np.newaxis, ...]
                new_img = sitk.GetImageFromArray(arr)
                new_img.SetSpacing(image_sitk.GetSpacing() + (1.0,))
                new_img.SetOrigin(image_sitk.GetOrigin() + (0.0,))
                d = image_sitk.GetDirection()
                new_img.SetDirection((d[0], d[1], 0.0, d[2], d[3], 0.0, 0.0, 0.0, 1.0))
                image_sitk = new_img
            
            img_array = sitk.GetArrayFromImage(image_sitk)
            spacing = np.flip(np.array(image_sitk.GetSpacing()))
            
            properties = {
                "spacing": spacing,
                "direction": image_sitk.GetDirection(),
                "origin": image_sitk.GetOrigin()
            }
            
            # Step 2: Preprocess
            print("   [2/4] Running nnU-Net preprocessing...")
            data, _, _ = self.preprocessor.run_case_npy(
                np.array([img_array]),
                None,
                properties,
                self.model.plans_manager,
                self.model.configuration_manager,
                self.model.dataset_json,
            )
            print(f"         Preprocessed shape: {data.shape}")
            
            # Step 3: Run inference
            print("   [3/4] Running neural network inference...")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                logits = self.model.predict_logits_from_preprocessed_data(
                    torch.from_numpy(data)
                ).cpu()
            
            print(f"         Output shape: {logits.shape}")
            
            # Step 4: Extract probabilities
            print("   [4/4] Extracting detection probabilities...")
            probs = torch.sigmoid(logits)
            
            # Apply L/R swap correction (model quirk)
            chk = probs.clone()
            # Swap pairs: PComm (3<->4), Infra ICA (5<->6), Supra ICA (7<->8), MCA (9<->10), ACA (11<->12)
            for a, b in [(3,4), (5,6), (7,8), (9,10), (11,12)]:
                probs[:, a] = chk[:, b]
                probs[:, b] = chk[:, a]
            
            # Get labels from model
            labels_from_model = list(self.model.dataset_json["labels"].keys())[1:]  # Skip background
            
            # Extract per-location max probabilities
            results = {}
            detections = []
            
            model_c, model_z, model_y, model_x = probs.shape
            scale_z = img_array.shape[0] / model_z
            scale_y = img_array.shape[1] / model_y
            scale_x = img_array.shape[2] / model_x
            
            for i, label in enumerate(labels_from_model):
                # Get max probability for this location
                flat = probs[i].view(-1)
                max_val, max_idx = torch.max(flat, dim=0)
                prob = float(max_val)
                
                # Convert to coordinates
                idx = int(max_idx)
                z_m = idx // (model_y * model_x)
                rem = idx % (model_y * model_x)
                y_m = rem // model_x
                x_m = rem % model_x
                
                z_orig = int(z_m * scale_z)
                y_orig = int(y_m * scale_y)
                x_orig = int(x_m * scale_x)
                
                # Apply L/R label correction
                corrected_label = label
                if "Left" in label:
                    corrected_label = label.replace("Left", "Right")
                elif "Right" in label:
                    corrected_label = label.replace("Right", "Left")
                
                results[corrected_label] = {
                    "probability": round(prob, 4),
                    "coordinates": {"x": x_orig, "y": y_orig, "z": z_orig}
                }
                
                # Track significant detections
                if prob > 0.5:
                    detections.append({
                        "location": corrected_label,
                        "probability": round(prob, 4),
                        "slice": z_orig,
                        "position": {"x": x_orig, "y": y_orig}
                    })
            
            # Overall aneurysm presence
            max_prob = max(r["probability"] for r in results.values()) if results else 0.0
            
            # Determine risk level
            if max_prob > 0.7:
                risk_level = "HIGH"
            elif max_prob > 0.4:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            # Sort detections by probability
            detections.sort(key=lambda x: x["probability"], reverse=True)
            
            torch.cuda.empty_cache()
            
            output = {
                "summary": {
                    "aneurysm_probability": round(max_prob, 4),
                    "risk_level": risk_level,
                    "num_detections": len(detections),
                    "scan_shape": list(img_array.shape),
                    "scan_spacing": [round(s, 3) for s in spacing.tolist()],
                },
                "detections": detections,
                "per_location": results,
            }
            
            print(f"\n{'='*50}")
            print(f"📊 RESULTS")
            print(f"{'='*50}")
            print(f"Aneurysm Probability: {max_prob:.1%}")
            print(f"Risk Level: {risk_level}")
            print(f"Detections (>50%): {len(detections)}")
            
            if detections:
                print(f"\n🎯 Top Detections:")
                for i, det in enumerate(detections[:5]):
                    print(f"   {i+1}. {det['location']}: {det['probability']:.1%} (slice {det['slice']})")
            
            print(f"{'='*50}\n")
            
            return output
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Detect intracranial aneurysms in CTA scans using nnU-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_aneurysm.py --input ./dicom_folder --output ./results
  python detect_aneurysm.py -i ./scan -o ./output --device cpu
  python detect_aneurysm.py -i ./patient_001 --json-only
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to DICOM folder containing CTA scan')
    parser.add_argument('-o', '--output', type=str, default='./detection_results',
                        help='Output folder for results (default: ./detection_results)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference (default: cuda)')
    parser.add_argument('--json-only', action='store_true',
                        help='Only save JSON results (no images)')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_path}")
        sys.exit(1)
    
    # Create output folder
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🧠 INTRACRANIAL ANEURYSM DETECTION")
    print("   MIC-DKFZ 7th Place Kaggle Solution (nnU-Net)")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = AneurysmDetector(device=args.device)
    
    if not detector.is_loaded:
        print("❌ Failed to initialize detector")
        sys.exit(1)
    
    # Run detection
    results = detector.detect(input_path)
    
    if "error" in results:
        print(f"❌ Detection failed: {results['error']}")
        sys.exit(1)
    
    # Save results
    output_json = output_path / "detection_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_json}")
    print("\n✅ Detection complete!")


if __name__ == "__main__":
    main()
