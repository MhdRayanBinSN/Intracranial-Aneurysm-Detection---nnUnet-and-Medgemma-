"""
3. Inference Module
===================
Runs the neural network on preprocessed data using sliding window.

Why Sliding Window?
    - Brain CT scans are large (e.g., 512×512×300)
    - GPU memory cannot fit entire volume
    - Process in overlapping patches, then merge
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from scipy.ndimage import gaussian_filter

from config import DEVICE, TILE_STEP_SIZE, USE_GAUSSIAN


def compute_gaussian_importance(tile_size: Tuple[int, ...], sigma_scale: float = 0.125) -> torch.Tensor:
    """
    Create a Gaussian weighting mask for blending overlapping patches.
    
    Why? Patches overlap, so edge pixels are predicted multiple times.
    Use Gaussian weights to give more importance to center predictions.
    
    ```
    Without Gaussian:          With Gaussian:
    ┌─────┬─────┐             ┌─────┬─────┐
    │ 1 1 │ 1 1 │             │.2 .8│.8 .2│
    │ 1 1 │ 1 1 │  Equal  →   │.8  1│1  .8│  Weighted
    │ 1 1 │ 1 1 │  weight     │.8  1│1  .8│  by center
    └─────┴─────┘             └─────┴─────┘
           ↓                         ↓
       Seam visible            Smooth blend
    ```
    """
    # Create array with 1 at center
    tmp = np.zeros(tile_size)
    center = [s // 2 for s in tile_size]
    sigmas = [s * sigma_scale for s in tile_size]
    tmp[tuple(center)] = 1
    
    # Apply Gaussian blur
    gaussian_map = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    
    # Normalize
    gaussian_map /= gaussian_map.max()
    
    return torch.from_numpy(gaussian_map).float()


def compute_sliding_window_steps(image_size: Tuple[int, ...], 
                                  tile_size: Tuple[int, ...], 
                                  step_size: float = TILE_STEP_SIZE) -> List[List[int]]:
    """
    Calculate starting positions for sliding window patches.
    
    Example:
        Image: 200 pixels, Tile: 64 pixels, Step: 0.5 (50% overlap)
        
        Step size = 64 * 0.5 = 32 pixels
        
        Positions: [0, 32, 64, 96, 136]
                    └──────────────────────────────────────┐
        Image:     |████████████████████████████████████████|
        Patch 1:   |████████|                               (0-64)
        Patch 2:       |████████|                           (32-96)
        Patch 3:           |████████|                       (64-128)
        Patch 4:               |████████|                   (96-160)
        Patch 5:                       |████████|           (136-200)
    """
    steps = []
    
    for img_s, tile_s in zip(image_size, tile_size):
        # Calculate number of steps needed
        target_step = tile_s * step_size
        num_steps = int(np.ceil((img_s - tile_s) / target_step)) + 1
        
        if num_steps > 1:
            # Evenly space the patches
            max_start = img_s - tile_s
            actual_step = max_start / (num_steps - 1)
            positions = [int(np.round(actual_step * i)) for i in range(num_steps)]
        else:
            positions = [0]
        
        steps.append(positions)
    
    return steps


def sliding_window_inference(model: torch.nn.Module,
                              data: np.ndarray,
                              tile_size: Tuple[int, int, int] = (128, 128, 128),
                              step_size: float = TILE_STEP_SIZE,
                              use_gaussian: bool = USE_GAUSSIAN) -> torch.Tensor:
    """
    Run inference using sliding window approach.
    
    Algorithm:
        1. Pad image if smaller than tile size
        2. Calculate patch positions
        3. For each patch:
           a. Extract patch from image
           b. Run through model
           c. Multiply by Gaussian weights
           d. Accumulate predictions
        4. Divide by accumulated weights
        5. Return final predictions
    
    Args:
        model: Neural network
        data: Preprocessed image (C, Z, Y, X)
        tile_size: Size of each patch
        step_size: Overlap ratio (0.5 = 50% overlap)
        use_gaussian: Whether to use Gaussian blending
        
    Returns:
        Logits tensor (num_classes, Z, Y, X)
    """
    print(f"Running sliding window inference...")
    print(f"  Input shape: {data.shape}")
    print(f"  Tile size: {tile_size}")
    print(f"  Step size: {step_size}")
    
    model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    data = data.to(DEVICE)
    
    # Get dimensions
    assert data.ndim == 4, f"Expected (C, Z, Y, X), got {data.shape}"
    _, z, y, x = data.shape
    image_size = (z, y, x)
    
    # Pad if needed (image smaller than tile)
    padding = []
    for img_s, tile_s in zip(image_size, tile_size):
        if img_s < tile_s:
            pad_total = tile_s - img_s
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding.extend([pad_before, pad_after])
        else:
            padding.extend([0, 0])
    
    # PyTorch padding is reversed: (x_before, x_after, y_before, y_after, z_before, z_after)
    padding = padding[::-1]
    
    if any(p > 0 for p in padding):
        data = F.pad(data, padding, mode='constant', value=0)
        print(f"  Padded to: {data.shape}")
    
    # Get new dimensions after padding
    _, z_pad, y_pad, x_pad = data.shape
    padded_size = (z_pad, y_pad, x_pad)
    
    # Calculate sliding window positions
    steps = compute_sliding_window_steps(padded_size, tile_size, step_size)
    
    # Count total patches
    total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
    print(f"  Total patches: {total_patches}")
    
    # Get Gaussian weights
    if use_gaussian:
        gaussian = compute_gaussian_importance(tile_size).to(DEVICE)
    else:
        gaussian = torch.ones(tile_size, device=DEVICE)
    
    # Initialize accumulators
    # We need to infer number of classes from first prediction
    first_patch = data[:, :tile_size[0], :tile_size[1], :tile_size[2]]
    with torch.no_grad():
        first_output = model(first_patch.unsqueeze(0))
    num_classes = first_output.shape[1]
    
    accumulated = torch.zeros((num_classes,) + padded_size, device=DEVICE)
    weights = torch.zeros(padded_size, device=DEVICE)
    
    # Process each patch
    patch_count = 0
    with torch.no_grad():
        for z_start in steps[0]:
            for y_start in steps[1]:
                for x_start in steps[2]:
                    patch_count += 1
                    
                    # Extract patch
                    z_end = z_start + tile_size[0]
                    y_end = y_start + tile_size[1]
                    x_end = x_start + tile_size[2]
                    
                    patch = data[:, z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Run model (add batch dimension)
                    output = model(patch.unsqueeze(0))  # (1, C, Z, Y, X)
                    output = output.squeeze(0)          # (C, Z, Y, X)
                    
                    # Accumulate weighted predictions
                    accumulated[:, z_start:z_end, y_start:y_end, x_start:x_end] += output * gaussian
                    weights[z_start:z_end, y_start:y_end, x_start:x_end] += gaussian
                    
                    if patch_count % 10 == 0:
                        print(f"  Processed {patch_count}/{total_patches} patches")
    
    # Divide by weights
    logits = accumulated / weights.unsqueeze(0).clamp(min=1e-8)
    
    # Remove padding
    if any(p > 0 for p in padding):
        z_slice = slice(padding[5], -padding[4] if padding[4] > 0 else None)
        y_slice = slice(padding[3], -padding[2] if padding[2] > 0 else None)
        x_slice = slice(padding[1], -padding[0] if padding[0] > 0 else None)
        logits = logits[:, z_slice, y_slice, x_slice]
    
    print(f"  Output shape: {logits.shape}")
    
    return logits.cpu()


def predict(model: torch.nn.Module, preprocessed_data: np.ndarray) -> torch.Tensor:
    """
    Run prediction on preprocessed data.
    
    Args:
        model: Loaded neural network
        preprocessed_data: Array from preprocessing (C, Z, Y, X)
        
    Returns:
        Probability tensor (num_classes, Z, Y, X)
    """
    # Run sliding window inference
    logits = sliding_window_inference(model, preprocessed_data)
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)
    
    return probabilities


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    from config import DEVICE
    
    print(f"Device: {DEVICE}")
    
    # Example with dummy model and data
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 13, 1)
            
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel().to(DEVICE)
    dummy_data = np.random.randn(1, 64, 64, 64).astype(np.float32)
    
    probs = predict(model, dummy_data)
    print(f"Output probabilities shape: {probs.shape}")
