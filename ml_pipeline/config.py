"""
Configuration & Constants
=========================
All paths, labels, and hyperparameters for the ML pipeline.
"""

from pathlib import Path
import torch

# ============================================
# PATHS
# ============================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
SOL_DIR = BASE_DIR / "sol"
CHECKPOINT_DIR = BASE_DIR / "Dataset004_iarsna_crop_2" / "Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32"
CHECKPOINT_FILE = CHECKPOINT_DIR / "fold_all" / "checkpoint_epoch_1500.pth"

# ============================================
# MODEL CONFIGURATION
# ============================================

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detection threshold
DETECTION_THRESHOLD = 0.50  # 50% probability = positive detection

# Sliding window parameters
TILE_STEP_SIZE = 0.5        # 50% overlap between patches
USE_GAUSSIAN = True         # Gaussian weighting for patch borders
USE_MIRRORING = False       # Test-time augmentation (off for speed)

# ============================================
# ANATOMICAL LOCATIONS (13 classes)
# ============================================

LOCATION_LABELS = [
    'Other Posterior Circulation',              # 0
    'Basilar Tip',                             # 1
    'Right Posterior Communicating Artery',    # 2
    'Left Posterior Communicating Artery',     # 3
    'Right Infraclinoid Internal Carotid Artery',  # 4
    'Left Infraclinoid Internal Carotid Artery',   # 5
    'Right Supraclinoid Internal Carotid Artery',  # 6
    'Left Supraclinoid Internal Carotid Artery',   # 7
    'Right Middle Cerebral Artery',            # 8
    'Left Middle Cerebral Artery',             # 9
    'Right Anterior Cerebral Artery',          # 10
    'Left Anterior Cerebral Artery',           # 11
    'Anterior Communicating Artery',           # 12
]

# Short names for display
LOCATION_SHORT = {
    'Other Posterior Circulation': 'Other Post',
    'Basilar Tip': 'Basilar',
    'Right Posterior Communicating Artery': 'R-PComm',
    'Left Posterior Communicating Artery': 'L-PComm',
    'Right Infraclinoid Internal Carotid Artery': 'R-Infraclinoid',
    'Left Infraclinoid Internal Carotid Artery': 'L-Infraclinoid',
    'Right Supraclinoid Internal Carotid Artery': 'R-Supra ICA',
    'Left Supraclinoid Internal Carotid Artery': 'L-Supra ICA',
    'Right Middle Cerebral Artery': 'R-MCA',
    'Left Middle Cerebral Artery': 'L-MCA',
    'Right Anterior Cerebral Artery': 'R-ACA',
    'Left Anterior Cerebral Artery': 'L-ACA',
    'Anterior Communicating Artery': 'A-Comm',
}

# ============================================
# PREPROCESSING PARAMETERS
# ============================================

# CT Hounsfield Unit clipping (focus on blood vessels)
HU_MIN = 0
HU_MAX = 600

# Target spacing (mm) - model trained with this
TARGET_SPACING = (0.5, 0.5, 0.5)  # isotropic 0.5mm

# ============================================
# TRAINING PARAMETERS (for reference)
# ============================================

TRAINING_CONFIG = {
    'epochs': 1500,
    'batch_size': 32,
    'optimizer': 'SGD',
    'momentum': 0.99,
    'initial_lr': 0.01,
    'lr_schedule': 'polynomial_decay',
    'loss_function': 'TopK_BCE',
    'topk_percentage': 0.2,  # Focus on top 20% hardest samples
    'blob_radius': 65,       # EDT radius in voxels
}
