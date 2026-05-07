
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ==========================================
# 1. SETUP: Import ACTUAL Project Code
# ==========================================
# Add 'sol' to python path so we can import nnunetv2
current_dir = os.getcwd()
sol_path = os.path.join(current_dir, 'sol')
sys.path.append(sol_path)

print(f"🔹 Added to Python Path: {sol_path}")

try:
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.preprocessing.preprocessors.rsna_aneurysm_preprocessor import RSNA_Aneurysm_Preprocessor
    print("✅ Successfully imported REAL project classes!")
except ImportError as e:
    print(f"❌ Failed to import project code: {e}")
    print("Make sure you are running this from the 'Pretrained detection' folder.")
    exit()

# ==========================================
# 2. CONFIGURATION: Paths
# ==========================================
# Path to the specific Plans file we found earlier
PLANS_FILE = r"Dataset004_iarsna_crop_2\Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32\plans.json"
DATASET_JSON = r"Dataset004_iarsna_crop_2\Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32\dataset.json"

# User's Input File (Passed as argument or default)
DEFAULT_INPUT = r"C:\Users\Rayan\Desktop\Main Project\series\1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381\1.2.826.0.1.3680043.8.498.10514293279013411121652430715824990591.dcm"
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

# ==========================================
# 3. RUN THE REAL PREPROCESSOR
# ==========================================
def run_real_code():
    print(f"\n🚀 Initializing Project's Preprocessor...")
    
    # 1. Load Plans
    if not os.path.exists(PLANS_FILE):
        print(f"❌ Plans file not found: {PLANS_FILE}")
        return

    plans_manager = PlansManager(PLANS_FILE)
    
    # 2. Get Configuration (3d_fullres)
    config_manager = plans_manager.get_configuration("3d_fullres")
    print(f"   -> Loaded Configuration: 3d_fullres")
    print(f"   -> Target Spacing: {config_manager.spacing}")

    # 3. Instantiate the Custom Preprocessor
    # This uses the EXACT class defined in sol/nnunetv2/preprocessing/preprocessors/rsna_aneurysm_preprocessor.py
    preprocessor = RSNA_Aneurysm_Preprocessor(verbose=True)
    print(f"   -> Instantiated Class: {preprocessor.__class__.__name__}")

    # 4. Run it on the image
    print(f"\n🖼 Processing Image: {os.path.basename(INPUT_FILE)}")
    # We need to pass a list of images (just one for specific case)
    # run_case signature: (image_files, seg_file, plans_manager, configuration_manager, dataset_json)
    data, seg, properties = preprocessor.run_case(
        image_files=[INPUT_FILE], 
        seg_file=None, 
        plans_manager=plans_manager, 
        configuration_manager=config_manager,
        dataset_json=DATASET_JSON
    )
    
    print(f"✅ Processing Complete!")
    print(f"   -> Final Shape: {data.shape}")
    print(f"   -> Data Type: {data.dtype}")
    print(f"   -> Max Value: {np.max(data):.2f}, Min Value: {np.min(data):.2f}")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    # Load raw for comparison
    raw_img = sitk.ReadImage(INPUT_FILE)
    raw_arr = sitk.GetArrayFromImage(raw_img)
    if raw_arr.ndim == 3: raw_arr = raw_arr[raw_arr.shape[0]//2]

    # Get middle slice of processed data (Channel 0)
    proc_arr = data[0, :, :, data.shape[3]//2]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("RAW Input (Loaded directly)")
    plt.imshow(raw_arr, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"ACTUAL Pipeline Output\n(RSNA_Aneurysm_Preprocessor)")
    plt.imshow(proc_arr, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.abspath("real_code_result.png")
    plt.savefig(output_path)
    print(f"\n✅ Result saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_real_code()
