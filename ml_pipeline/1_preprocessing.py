"""
1. Preprocessing Module
=======================
Loads DICOM images and prepares them for the neural network.

Pipeline:
    DICOM files → Load → Crop ROI → Normalize → Resample → 3D Volume
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import pydicom
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from config import TARGET_SPACING


def load_dicom_series(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series from a folder.
    
    Args:
        dicom_folder: Path to folder containing .dcm files
        
    Returns:
        volume: 3D numpy array (Z, Y, X)
        properties: Dict with spacing, origin, direction
    """
    print(f"Loading DICOM from: {dicom_folder}")
    
    # Read DICOM series using SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_folder))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    reader.SetFileNames(dicom_files)
    image_sitk = reader.Execute()
    
    # Convert to numpy array 3d constructyion
    volume = sitk.GetArrayFromImage(image_sitk)  # Shape: (Z, Y, X)
    
    # Get metadata
    properties = {
        'spacing': np.array(image_sitk.GetSpacing()[::-1]),  # (Z, Y, X)
        'origin': np.array(image_sitk.GetOrigin()),
        'direction': np.array(image_sitk.GetDirection()).reshape(3, 3),
        'original_shape': volume.shape,
    }
    
    print(f"  Loaded shape: {volume.shape}")
    print(f"  Spacing: {properties['spacing']} mm")
    
    return volume, properties


def crop_to_nonzero(volume: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Crop volume to bounding box of non-zero voxels (remove background/air).
    
    This is standard nnU-Net preprocessing that:
        1. Creates a mask of non-zero voxels
        2. Fills holes in the mask
        3. Finds bounding box
        4. Crops to bounding box
    
    Why crop?
        - Removes black background (air padding)
        - Reduces computation (smaller volume)
        - Focuses on actual tissue
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        
    Returns:
        Cropped volume, crop_info dict with bounding box
    """
    print(f"Cropping to non-zero region...")
    original_shape = volume.shape
    
    # Step 1: Create non-zero mask
    nonzero_mask = volume != 0
    
    # Step 2: Fill holes (handle small gaps inside tissue)
    nonzero_mask = binary_fill_holes(nonzero_mask)
    
    # Step 3: Find bounding box of non-zero region
    coords = np.argwhere(nonzero_mask)
    
    if coords.size == 0:
        print("  Warning: Volume is all zeros! No cropping applied.")
        crop_info = {
            'bbox_start': (0, 0, 0),
            'bbox_end': volume.shape,
            'original_shape': original_shape,
            'cropped_shape': volume.shape,
        }
        return volume, crop_info
    
    # Get min/max coordinates (bounding box)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1  # +1 because slice is exclusive
    
    # Step 4: Crop volume
    volume_cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    crop_info = {
        'bbox_start': (z_min, y_min, x_min),
        'bbox_end': (z_max, y_max, x_max),
        'original_shape': original_shape,
        'cropped_shape': volume_cropped.shape,
    }
    
    print(f"  Original shape: {original_shape}")
    print(f"  Bounding box: [{z_min}:{z_max}, {y_min}:{y_max}, {x_min}:{x_max}]")
    print(f"  Cropped shape: {volume_cropped.shape}")
    reduction = 100 * (1 - np.prod(volume_cropped.shape) / np.prod(original_shape))
    print(f"  Volume reduced by: {reduction:.1f}%")
    
    return volume_cropped, crop_info


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """
    Normalize CT using Z-Score normalization (matches nnU-Net training).
    
    Steps:
        1. Compute mean and standard deviation
        2. Subtract mean and divide by std: (x - mean) / std
    
    This is what the model was trained with (see plans.json: "ZScoreNormalization")
    
    Args:
        volume: 3D CT volume
        
    Returns:
        Z-score normalized volume (centered around 0)
    """
    print(f"Normalizing CT (Z-Score): [{volume.min():.0f}, {volume.max():.0f}] HU")
    
    # Step 1: Compute statistics
    mean = volume.mean()
    std = volume.std()
    
    # Step 2: Z-Score normalize: (x - mean) / std
    volume_norm = (volume - mean) / max(std, 1e-8)  # Prevent division by zero
    
    print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"  Result: [{volume_norm.min():.3f}, {volume_norm.max():.3f}]")
    
    return volume_norm.astype(np.float32)


def resample_volume(volume: np.ndarray, 
                    original_spacing: np.ndarray,
                    target_spacing: Tuple[float, float, float] = TARGET_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample volume to target spacing using linear interpolation.
    
    Args:
        volume: 3D numpy array
        original_spacing: (Z, Y, X) spacing in mm
        target_spacing: Target spacing in mm
        
    Returns:
        Resampled volume, new spacing
    """
    print(f"Resampling: {original_spacing} → {target_spacing} mm")
    
    # Calculate new shape
    original_shape = np.array(volume.shape)
    scale_factors = original_spacing / np.array(target_spacing)
    new_shape = np.round(original_shape * scale_factors).astype(int)
    
    # Create SimpleITK image for resampling
    image_sitk = sitk.GetImageFromArray(volume)
    image_sitk.SetSpacing(original_spacing[::-1].tolist())  # SimpleITK uses (X, Y, Z)
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1])  # (X, Y, Z)
    resampler.SetSize(new_shape[::-1].tolist())       # (X, Y, Z)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    
    resampled_sitk = resampler.Execute(image_sitk)
    resampled = sitk.GetArrayFromImage(resampled_sitk)
    
    print(f"  Result shape: {resampled.shape}")
    
    return resampled, np.array(target_spacing)


def preprocess(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Full preprocessing pipeline.
    
    Steps:
        1. Load DICOM series
        2. Crop to non-zero region (ROI)
        3. Normalize CT values (Z-Score)
        4. Resample to target spacing
        
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        Preprocessed 4D array (1, Z, Y, X) ready for model
        Properties dict with metadata
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load DICOM
    volume, properties = load_dicom_series(dicom_folder)
    
    # Step 2: Crop to non-zero region (remove background)
    volume_cropped, crop_info = crop_to_nonzero(volume)
    properties['crop_info'] = crop_info
    
    # Step 3: Normalize (Z-Score)
    volume_norm = normalize_ct(volume_cropped)
    
    # Step 4: Resample
    volume_resampled, new_spacing = resample_volume(
        volume_norm, 
        properties['spacing']
    )
    
    # Add channel dimension (C, Z, Y, X)
    data = volume_resampled[np.newaxis, ...]  # (1, Z, Y, X)
    
    # Update properties
    properties['spacing'] = new_spacing
    properties['preprocessed_shape'] = data.shape
    
    print("=" * 50)
    print(f"Preprocessing complete. Shape: {data.shape}")
    print("=" * 50)
    
    return data, properties


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <dicom_folder>")
        sys.exit(1)
    
    dicom_path = Path(sys.argv[1])
    data, props = preprocess(dicom_path)
    
    print(f"\nOutput shape: {data.shape}")
    print(f"Ready for model input!")
