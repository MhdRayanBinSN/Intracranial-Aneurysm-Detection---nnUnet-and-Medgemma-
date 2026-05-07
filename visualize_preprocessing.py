
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Try to find the preprocessed data folder
possible_paths = [
    r"c:\Users\Rayan\Desktop\Main Project\Code\Pretrained detection\backend\models\nnUNet_preprocessed\Dataset004_iarsna_crop\nnUNetPlans_3d_fullres",
    r"c:\Users\Rayan\Desktop\Main Project\Code\Pretrained detection\Dataset004_iarsna_crop_2",
    # Add any other potential paths here
]

data_path = None
for path in possible_paths:
    if os.path.exists(path) and len(glob.glob(os.path.join(path, "*.npz"))) > 0:
        data_path = path
        break

if data_path is None:
    print("❌ Could not find preprocessed .npz files.")
    print("This is normal if you only downloaded the weights but not the training data.")
    print("To generate them, you would typically run:")
    print("  nnUNetv2_plan_and_preprocess -d 004 -c 3d_fullres")
    exit()

# Load a random file
files = glob.glob(os.path.join(data_path, "*.npz"))
random_file = files[0] # Just pick the first one for now
print(f"✅ Loading: {random_file}")

data = np.load(random_file)['data'] # Shape: (C, X, Y, Z)
print(f"Shape: {data.shape}")

# Visualize Middle Slice of Channel 0
mid_slice = data.shape[3] // 2
plt.imshow(data[0, :, :, mid_slice], cmap='gray')
plt.title(f"Preprocessed Slice (Z={mid_slice})")
plt.axis('off')
plt.show()
