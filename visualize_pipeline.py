
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes

# ==========================================
# 1. SETUP: Input Handling
# ==========================================
DEFAULT_FILE = r"C:\Users\Rayan\Desktop\Main Project\series\1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381\1.2.826.0.1.3680043.8.498.10514293279013411121652430715824990591.dcm"

if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = DEFAULT_FILE

TARGET_SPACING = (1.0, 1.0, 1.0) # Goal: 1mm isotropic

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def crop_to_nonzero(data):
    """
    Standard nnU-Net Step 1: Remove black background.
    """
    print("   -> Calculating non-zero mask...")
    mask = data != 0
    
    # Get bounding box of non-zero region
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("   -> Warning: Image is all zeros!")
        return data, mask
        
    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1 # Slice is exclusive at end

    # Slice dynamically based on dimensions
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    return data[slices], mask

def z_score_normalization(data):
    """ Standardize: (Pixel - Mean) / Std """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / max(std, 1e-8)

def resample_image(data, original_spacing, target_spacing):
    """ Resize image to match target spacing. """
    ndim = data.ndim
    org_spacing = original_spacing[:ndim]
    tgt_spacing = target_spacing[:ndim]
    
    org_spacing_array = np.array(org_spacing)[::-1] 
    tgt_spacing_array = np.array(tgt_spacing)[::-1]

    resize_factor = org_spacing_array / tgt_spacing_array
    new_shape = np.round(data.shape * resize_factor).astype(int)
    
    # Interpolate (Bicubic)
    data_resampled = resize(data, new_shape, order=3, mode='edge', anti_aliasing=False)
    return data_resampled

# ==========================================
# 3. LOAD DICOM
# ==========================================
if not os.path.exists(INPUT_FILE):
    print(f"❌ File not found: {INPUT_FILE}")
    exit()

print(f"✅ Loading: {INPUT_FILE}")
try:
    img = sitk.ReadImage(INPUT_FILE)
    data_raw = sitk.GetArrayFromImage(img) # (Z, Y, X) or (Y, X)
    spacing_raw = img.GetSpacing()         # (X, Y, Z)
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit()

if data_raw.ndim == 3 and data_raw.shape[0] == 1:
    data_raw = data_raw.squeeze(0)

# ==========================================
# 4. APPLY PIPELINE (Full 4 Steps)
# ==========================================

# --- STEP 1: CROPPING ---
print("\nStep 1: Cropping (Removing black space)...")
data_cropped, _ = crop_to_nonzero(data_raw)
print(f"   -> Old Shape: {data_raw.shape}")
print(f"   -> New Shape: {data_cropped.shape}")

# --- STEP 2: NORMALIZATION ---
print("\nStep 2: Normalizing (Z-Score)...")
data_norm = z_score_normalization(data_cropped)
print(f"   -> Mean: {np.mean(data_norm):.2f}, Std: {np.std(data_norm):.2f}")

# --- STEP 3: RESAMPLING ---
print("\nStep 3: Resampling to 1mm...")
data_resampled = resample_image(data_norm, spacing_raw, TARGET_SPACING)
print(f"   -> Final Shape: {data_resampled.shape}")

# ==========================================
# 5. VISUALIZATION (4 Panels)
# ==========================================
plt.figure(figsize=(20, 5))

# Panel 1: Original
plt.subplot(1, 4, 1)
plt.title(f"1. Raw Input\n{data_raw.shape}")
plt.imshow(data_raw if data_raw.ndim==2 else data_raw[data_raw.shape[0]//2], cmap='gray')
plt.axis('off')

# Panel 2: Cropped
plt.subplot(1, 4, 2)
plt.title(f"2. Cropped (Non-Zero)\n{data_cropped.shape}\n(Removed black borders)")
plt.imshow(data_cropped if data_cropped.ndim==2 else data_cropped[data_cropped.shape[0]//2], cmap='gray')
plt.axis('off')

# Panel 3: Normalized
plt.subplot(1, 4, 3)
plt.title(f"3. Normalized (Z-Score)\n{data_norm.shape} (Unchanged)")
plt.imshow(data_norm if data_norm.ndim==2 else data_norm[data_norm.shape[0]//2], cmap='gray')
plt.axis('off')

# Panel 4: Resampled
plt.subplot(1, 4, 4)
plt.title(f"4. Resampled (1mm)\n{data_resampled.shape}")
plt.imshow(data_resampled if data_resampled.ndim==2 else data_resampled[data_resampled.shape[0]//2], cmap='gray')
plt.axis('off')

plt.tight_layout()
output_path = os.path.abspath("visualization_full_pipeline.png")
plt.savefig(output_path)
print(f"\n✅ Saved Full Pipeline Visualization to:\n   ➡ {output_path}")

try:
    plt.show()
except:
    pass
