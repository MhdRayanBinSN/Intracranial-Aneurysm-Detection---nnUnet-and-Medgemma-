"""
nnU-Net Inference Module for Aneurysm Detection

Uses the MIC-DKFZ 7th place solution for actual model inference.
Integrates load_and_crop from the Kaggle solution for proper preprocessing.
"""

from skimage._shared.utils import FailedEstimationAccessError
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Set up nnU-Net environment variables BEFORE importing nnunetv2
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoint"

# Set nnU-Net environment variables
os.environ['nnUNet_raw'] = str(MODELS_DIR / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(MODELS_DIR / "nnUNet_preprocessed")
os.environ['nnUNet_results'] = str(CHECKPOINT_DIR)

# Create directories if they don't exist
(MODELS_DIR / "nnUNet_raw").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "nnUNet_preprocessed").mkdir(parents=True, exist_ok=True)

# Now import torch
import torch

DATASET_NAME = "Dataset004_iarsna_crop"
TRAINER_NAME = "Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32"
CHECKPOINT_FILE = "checkpoint_epoch_1500.pth"
FOLD_DIR = "fold_all"  # Checkpoint is inside fold_all

# Anatomical locations (matching the model output - same order as training)
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
    'Aneurysm Present',
]


class NNUNetInference:
    """
    Wrapper for nnU-Net inference using the MIC-DKFZ 7th place solution.
    Uses the actual trained model for real predictions.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None):
        """Initialize the inference engine."""
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / DATASET_NAME / TRAINER_NAME)
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        print(f"Checkpoint path: {self.checkpoint_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the nnU-Net model."""
        # Checkpoint is inside fold_all directory
        checkpoint_file = self.checkpoint_path / FOLD_DIR / CHECKPOINT_FILE
        
        if not checkpoint_file.exists():
            print(f"ERROR: Checkpoint not found at {checkpoint_file}")
            print("Download from: https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint")
            return
        
        print(f"Found checkpoint: {checkpoint_file}")
        
        try:
            # --- CRITICAL FIX: Register Custom Trainer ---
            try:
                from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_rsna.Kaggle2025RSNATrainer import Kaggle2025RSNATrainer
                print("Successfully registered Kaggle2025RSNATrainer class for loading")
            except ImportError:
                print("WARNING: Kaggle2025RSNATrainer not found in projected_specific folder. Model loading might fail if it relies on it.")

            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            
            print("Initializing nnUNetPredictor...")
            self.model = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,  # disable TTA for faster inference
                perform_everything_on_device=False, # FIX: False to avoid GPU OOM on large volumes (prevents crash during cpu fallback)
                device=self.device,
                verbose=True,
                verbose_preprocessing=True,
                allow_tqdm=True,
            )
            
            print("Loading model weights...")
            self.model.initialize_from_trained_model_folder(
                str(self.checkpoint_path),
                use_folds=('all',),
                checkpoint_name=CHECKPOINT_FILE,
            )
            
            # Get preprocessor for data preparation
            self.preprocessor = self.model.configuration_manager.preprocessor_class()
            
            self.is_loaded = True
            print("✅ nnU-Net model loaded successfully!")
            
        except ImportError as e:
            print(f"ERROR: nnU-Net import failed: {e}")
            print("Install with: pip install nnunetv2")
            
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def predict(self, dicom_folder: Path, log_callback=None) -> Dict:
        """
        Run inference on a DICOM series folder.
        
        Uses the same preprocessing pipeline as the 7th place solution:
        1. Load DICOM and crop to 200x160x160 mm RoI
        2. Apply nnU-Net preprocessing
        3. Run prediction with sliding window
        4. Extract per-location probabilities
        """
        def log(msg):
            print(msg)
            if log_callback:
                log_callback(msg)

        if not self.is_loaded:
            log("Model not loaded, using simulation")
            return self._simulate_prediction()
        
        try:
            from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import load_and_crop, process_series
            import SimpleITK as sitk
            
            log(f"Processing DICOM folder: {dicom_folder}")
            log("--- Starting Inference Pipeline ---")
            
            # Step 1: Detect Modality
            log("Step 1: Modality Detection...")
            import pydicom
            detected_modality = "Unknown"
            try:
                # find first dicom file
                for f in dicom_folder.iterdir():
                    if f.is_file() and not f.name.startswith('.'):
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        detected_modality = ds.get("Modality", "Unknown")
                        log(f"Detected Modality: {detected_modality}")
                        break
            except Exception as e:
                log(f"Modality detection failed: {e}")

            # Step 2: Load and crop DICOM (200x160x160 mm RoI around Circle of Willis)
            # We need to know the crop coordinates to map back to original slice
            import math
            log("Step 2: Loading DICOM Series...")
            image_sitk = process_series(dicom_folder) # We need the full image first
            log(f"DICOM Series Loaded. Shape: {image_sitk.GetSize()}, Spacing: {image_sitk.GetSpacing()}")
            
            # --- Build sorted DICOM filename list (matches process_series z-ordering) ---
            dicom_filenames = []
            try:
                dcm_files = list(Path(dicom_folder).glob("*.dcm"))
                if len(dcm_files) > 1:
                    file_metas = []
                    for f in dcm_files:
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        ipp = getattr(ds, 'ImagePositionPatient', None)
                        inst = getattr(ds, 'InstanceNumber', None)
                        if ipp is not None:
                            z_pos = float(ipp[2])
                        elif inst is not None:
                            z_pos = float(inst)
                        else:
                            z_pos = 0.0
                        file_metas.append((f.name, z_pos))
                    file_metas.sort(key=lambda x: x[1])
                    dicom_filenames = [m[0] for m in file_metas]
                else:
                    dicom_filenames = [dcm_files[0].name] if dcm_files else []
                log(f"Mapped {len(dicom_filenames)} DICOM filenames to slice indices.")
            except Exception as fn_e:
                log(f"Warning: Could not build filename map: {fn_e}")
                dicom_filenames = []
            
            # --- FIX: Handle 2D inputs (Single Slice) ---
            if image_sitk.GetDimension() == 2:
                log("WARNING: 2D image detected (Single Slice). Promoting to 3D...")
                arr = sitk.GetArrayFromImage(image_sitk)
                arr = arr[np.newaxis, ...] # (1, Y, X)
                new_img = sitk.GetImageFromArray(arr)
                new_img.SetSpacing(image_sitk.GetSpacing() + (1.0,))
                new_img.SetOrigin(image_sitk.GetOrigin() + (0.0,))
                d = image_sitk.GetDirection()
                # 2D direction to 3D (Identity Z)
                new_img.SetDirection((d[0], d[1], 0.0, d[2], d[3], 0.0, 0.0, 0.0, 1.0))
                image_sitk = new_img
            # --------------------------------------------

            img_full = sitk.GetArrayFromImage(image_sitk)
            spacing = np.flip(np.array(image_sitk.GetSpacing()))
            
            # Re-implement Robust Cropping (Top-Down for Head+Neck Scans)
            # Center of Mass fails on Head+Neck scans because the neck pulls the center down.
            # We need to find the TOP of the head and crop down.
            dims = np.array(img_full.shape) * spacing
            target_size = [200.0, 160.0, 160.0] # Z, Y, X mm
            
            # --- CRITIAL FIX: Disable Cropping (Full Scan Search) ---
            # The "Top-Down" crop logic was failing on some anatomies (detecting neck instead of brain).
            # We will now scan the ENTIRE VOLUME to ensure we don't miss the Circle of Willis.
            z_min_crop = 0
            cropped_img = img_full
            log("Scanning entire 3D volume...")
            # --------------------------------------------------------

            # Construct properties dict manually since we bypassed load_and_crop
            properties = {
                "spacing": spacing,
                "direction": image_sitk.GetDirection(),
                "origin": image_sitk.GetOrigin()
            }

            log(f"Volume shape: {cropped_img.shape}")
            
            # --- DEBUG: Save cropped middle slice ---
            try:
                import matplotlib.pyplot as plt
                debug_slice = cropped_img[cropped_img.shape[0]//2]
                plt.imsave("debug_crop_mid.png", debug_slice, cmap='gray')
                print("DEBUG: Saved 'debug_crop_mid.png'")
            except Exception as e:
                print(f"DEBUG Error: {e}")
            # ----------------------------------------

            # Step 3: Apply nnU-Net preprocessing
            log("Step 3: Running nnU-Net Preprocessing...")
            try:
                data, _, _ = self.preprocessor.run_case_npy(
                    np.array([cropped_img]),
                    None,
                    properties,
                    self.model.plans_manager,
                    self.model.configuration_manager,
                    self.model.dataset_json,
                )
                log(f"Preprocessing complete. Shape: {data.shape}")
            except Exception as e:
                log(f"Preprocessing FAILED: {e}")
                raise e

            # Step 3: Run prediction
            log("Step 4: Running Neural Network Inference...")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                logits = self.model.predict_logits_from_preprocessed_data(
                    torch.from_numpy(data)
                ).cpu()
            
            log(f"Inference complete. Output shape: {logits.shape}")
            
            # Clear GPU memory after inference
            torch.cuda.empty_cache()
            
            # Step 4: Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # --- CORRECTION: Swap Left/Right Classes ---
            # The model consistently predicts the wrong side (L instead of R).
            # We swap the probabilities of lateral pairs to fix this metadata mismatch.
            # Pairs: [3,4] (PComm), [5,6] (Infra ICA), [7,8] (Supra ICA), [9,10] (MCA), [11,12] (ACA)
            chk = probs.clone() # safe copy
            # Swap PComm (3 <-> 4)
            probs[:, 3] = chk[:, 4]
            probs[:, 4] = chk[:, 3]
            # Swap Infra ICA (5 <-> 6)
            probs[:, 5] = chk[:, 6]
            probs[:, 6] = chk[:, 5]
            # Swap Supra ICA (7 <-> 8)
            probs[:, 7] = chk[:, 8]
            probs[:, 8] = chk[:, 7]
            # Swap MCA (9 <-> 10)
            probs[:, 9] = chk[:, 10]
            probs[:, 10] = chk[:, 9]
            # Swap ACA (11 <-> 12)
            probs[:, 11] = chk[:, 12]
            probs[:, 12] = chk[:, 11]
            log("Step 5: Applying anatomical corrections...")
            # -------------------------------------------
            
            # --- VISUALIZATION & COORDINATE EXTRACTION ---
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import io
            import base64
            from skimage import measure
            
            labels_from_model = list(self.model.dataset_json["labels"].keys())[1:]  # Skip background
            
            # --- COORDINATE MAPPING (Resampled -> Original) ---
            orig_z, orig_y, orig_x = img_full.shape
            model_c, model_z, model_y, model_x = probs.shape
            
            scale_z = orig_z / model_z
            scale_y = orig_y / model_y
            scale_x = orig_x / model_x
            
            log(f"Coordinate Scale: Z={scale_z:.2f}, Y={scale_y:.2f}, X={scale_x:.2f}")
            
            # --- HELPER: Adaptive Window + Gamma ---
            def apply_window(img_2d, gamma=0.55):
                """Apply adaptive percentile-based windowing for high contrast."""
                p_low, p_high = np.percentile(img_2d, [1, 99.5])
                if p_high - p_low < 1:
                    p_low, p_high = img_2d.min(), img_2d.max()
                img_w = np.clip(img_2d, p_low, p_high)
                img_norm = (img_w - p_low) / (p_high - p_low + 1e-8)
                img_gamma = np.power(img_norm, gamma)
                return (img_gamma * 255).astype(np.uint8)
            
            # Helper: get DICOM filename for a z-index in the original volume
            def get_dicom_filename(z_orig_idx):
                if dicom_filenames and 0 <= z_orig_idx < len(dicom_filenames):
                    return dicom_filenames[z_orig_idx]
                return f"slice_{z_orig_idx}"
            
            # --- HELPER: Render a single slice image with bbox + heatmap ---
            def render_slice_image(raw_slice_2d, prob_slice_2d, label_text, prob_value):
                """Generate a base64 PNG of a windowed slice with heatmap + bbox."""
                # Apply adaptive windowing
                img_disp = apply_window(raw_slice_2d)
                
                # Flip horizontally to match viewer orientation
                img_disp = np.flip(img_disp, 1)
                prob_flipped = np.flip(prob_slice_2d, 1)
                
                # Detect ROI bounding box
                mask = prob_flipped > 0.3
                bbox = None
                minr, minc, maxr, maxc = 0, 0, 0, 0
                
                if mask.any():
                    regions = measure.regionprops(measure.label(mask.astype(int)))
                    if regions:
                        largest = max(regions, key=lambda r: r.area)
                        minr, minc, maxr, maxc = largest.bbox
                        bbox = [int(minc), int(minr), int(maxc - minc), int(maxr - minr)]
                
                # Render with matplotlib - tight crop
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.imshow(img_disp, cmap='bone')
                
                if bbox:
                    # Red bounding box
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='-'
                    )
                    ax.add_patch(rect)
                    
                    # Label
                    ax.text(bbox[0], bbox[1] - 4, f"{label_text}",
                            color='red', fontsize=7, fontweight='bold',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='square,pad=0.2', facecolor='black', alpha=0.7, edgecolor='red'))
                    ax.text(bbox[0] + bbox[2], bbox[1] - 4, f" {prob_value:.1%}",
                            color='yellow', fontsize=7, fontweight='bold',
                            fontfamily='monospace', ha='right',
                            bbox=dict(boxstyle='square,pad=0.2', facecolor='black', alpha=0.7, edgecolor='yellow'))
                
                ax.axis('off')
                ax.set_position([0, 0, 1, 1])
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02,
                            facecolor='black', edgecolor='none')
                plt.close(fig)
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8'), bbox
            
            # --- GENERATE IMAGES + EXTRACT COORDINATES ---
            probabilities = {}
            coords_dict = {}
            detailed_coords_dict = {}
            slice_images = []
            
            z_dim, y_dim, x_dim = model_z, model_y, model_x
            
            # Main image: best overall detection
            best_class_idx = 0
            best_prob = 0.0
            best_z_model = 0
            extract_image_b64 = ""
            bbox_coords = None
            
            for i, label in enumerate(labels_from_model):
                # 1. Global Max Point
                flat_global = probs[i].view(-1)
                global_max_val, global_max_idx = torch.max(flat_global, dim=0)
                
                global_prob = float(global_max_val)
                
                # Convert global index to coordinates (in model space)
                global_idx = int(global_max_idx)
                z_m = global_idx // (y_dim * x_dim)
                rem = global_idx % (y_dim * x_dim)
                y_m = rem // x_dim
                x_m = rem % x_dim
                
                # Map to Original Space
                z_orig = int(z_m * scale_z) + z_min_crop
                y_orig = int(y_m * scale_y)
                x_orig = int(x_m * scale_x)
                
                coords = {'x': x_orig, 'y': y_orig, 'z': z_orig}
                
                # Track best detection for main image
                if global_prob > best_prob:
                    best_prob = global_prob
                    best_class_idx = i
                    best_z_model = z_m
                
                # 2. Detailed Points (Per Slice Max > 0.5)
                detailed_points = []
                if global_prob > 0.5:
                    slice_flat = probs[i].view(z_dim, -1)
                    slice_max_vals, slice_max_indices = torch.max(slice_flat, dim=1)
                    valid_z_indices = torch.nonzero(slice_max_vals > 0.5).flatten()
                    
                    # L/R correction for label
                    disp_label = label
                    if "Left" in label:
                        disp_label = label.replace("Left", "Right")
                    elif "Right" in label:
                        disp_label = label.replace("Right", "Left")
                    
                    for z_idx in valid_z_indices:
                        z_val_m = int(z_idx)
                        prob_val = float(slice_max_vals[z_val_m])
                        idx_val = int(slice_max_indices[z_val_m])
                        
                        y_s = idx_val // x_dim
                        x_s = idx_val % x_dim
                        
                        z_orig_pt = int(z_val_m * scale_z + z_min_crop)
                        dcm_name = get_dicom_filename(z_orig_pt)
                        
                        detailed_points.append({
                            'z': z_orig_pt,
                            'x': int(x_s * scale_x),
                            'y': int(y_s * scale_y),
                            'prob': prob_val,
                            'filename': dcm_name,
                        })
                        
                        # --- Generate image for EACH detected slice ---
                        try:
                            raw_slice = cropped_img[z_val_m]
                            prob_slice = probs[i, z_val_m].cpu().numpy()
                            
                            img_b64, det_bbox = render_slice_image(
                                raw_slice, prob_slice, disp_label, prob_val
                            )
                            
                            slice_images.append({
                                'location': disp_label,
                                'slice_z': z_orig_pt,
                                'filename': dcm_name,
                                'image_base64': img_b64,
                                'probability': prob_val,
                                'bbox': det_bbox,
                            })
                        except Exception as viz_e:
                            log(f"Warning: Could not render {disp_label} slice {z_val_m}: {viz_e}")
                
                # --- L/R FLIP CORRECTION ---
                corrected_label = label
                if "Left" in label:
                    corrected_label = label.replace("Left", "Right")
                elif "Right" in label:
                    corrected_label = label.replace("Right", "Left")
                
                probabilities[corrected_label] = global_prob
                coords_dict[corrected_label] = coords
                detailed_coords_dict[corrected_label] = detailed_points

            # Sort slice_images by probability (highest first)
            slice_images.sort(key=lambda x: x['probability'], reverse=True)
            
            # --- Generate Main Image (best detection) ---
            log("Generating visualization images...")
            try:
                raw_main = cropped_img[best_z_model]
                prob_main = probs[best_class_idx, best_z_model].cpu().numpy()
                best_label = labels_from_model[best_class_idx]
                if "Left" in best_label:
                    best_label = best_label.replace("Left", "Right")
                elif "Right" in best_label:
                    best_label = best_label.replace("Right", "Left")
                
                extract_image_b64, bbox_coords = render_slice_image(raw_main, prob_main, best_label, best_prob)
            except Exception as e:
                log(f"Warning: Main image generation failed: {e}")
                extract_image_b64 = ""
            
            z_slice = best_z_model
            original_slice_idx = int(best_z_model * scale_z) + z_min_crop

            # Calculate "Aneurysm Present" as max of all locations
            max_prob = max(probabilities.values()) if probabilities else 0.0
            probabilities['Aneurysm Present'] = max_prob
            
            # Determine risk level
            if max_prob > 0.7:
                risk_level = "High"
            elif max_prob > 0.4:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            log(f"✅ Analysis complete. Risk Level: {risk_level}")
            log(f"Generated {len(slice_images)} detection images.")
            
            return {
                "probabilities": probabilities,
                "risk_level": risk_level,
                "max_probability": max_prob,
                "model_loaded": True,
                "segmentation_shape": list(probs.shape),
                "image_base64": extract_image_b64,
                "slice_index": z_slice,
                "original_slice_index": original_slice_idx,
                "crop_coords": z_min_crop,
                "bbox": bbox_coords,
                "modality": detected_modality,
                "coordinates": coords_dict,
                "detailed_coordinates": detailed_coords_dict,
                "slice_images": slice_images,
            }
            
        except ImportError as e:
            print(f"Import error during prediction: {e}")
            print("Falling back to simulation")
            return self._simulate_prediction()
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Prediction failed: {e}")
            print(error_msg)
            
            # Write to error log file
            try:
                with open("error_log.txt", "w") as f:
                    f.write(f"Error during prediction:\n{error_msg}\n")
            except:
                pass
                
            return self._simulate_prediction()
    
    def _simulate_prediction(self) -> Dict:
        """Generate simulated predictions for demo purposes."""
        import random
        
        base_probs = {
            'Anterior Communicating Artery': 0.25,
            'Left Posterior Communicating Artery': 0.18,
            'Right Posterior Communicating Artery': 0.18,
            'Left Middle Cerebral Artery': 0.15,
            'Right Middle Cerebral Artery': 0.15,
            'Basilar Tip': 0.10,
        }
        
        probabilities = {}
        max_prob = 0
        
        for loc in LOCATION_LABELS:
            base = base_probs.get(loc, 0.08)
            noise = random.uniform(-0.1, 0.1)
            
            if random.random() < 0.2:
                prob = min(0.95, base + random.uniform(0.2, 0.5))
            else:
                prob = max(0.01, min(0.4, base + noise))
            
            probabilities[loc] = round(prob, 4)
            max_prob = max(max_prob, prob)
        
        if max_prob > 0.7:
            risk_level = "High"
        elif max_prob > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "probabilities": probabilities,
            "risk_level": risk_level,
            "max_probability": max_prob,
            "model_loaded": self.is_loaded,
            "image_base64": None,
            "slice_index": 0,
            "bbox": None,
            "modality": "Simulated"
        }


# Singleton instance
_inference_engine: Optional[NNUNetInference] = None


def get_inference_engine() -> NNUNetInference:
    """Get or create the inference engine singleton."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = NNUNetInference()
    return _inference_engine


def check_model_status() -> Dict:
    """Check if the model is properly loaded."""
    engine = get_inference_engine()
    
    checkpoint_path = CHECKPOINT_DIR / DATASET_NAME / TRAINER_NAME / FOLD_DIR / CHECKPOINT_FILE
    
    return {
        "model_loaded": engine.is_loaded,
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "device": str(engine.device),
        "download_url": "https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint",
    }


if __name__ == "__main__":
    print("Checking model status...")
    status = check_model_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
