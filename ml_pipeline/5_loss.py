"""
5. Loss Function - TopK Binary Cross Entropy
=============================================
The loss function used during training.

Why TopK?
    - Medical images are 99% background (no aneurysm)
    - Standard BCE would be dominated by easy negative samples
    - TopK focuses on the HARDEST 20% of samples

This is the KEY INNOVATION that makes training effective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TopKBCELoss(nn.Module):
    """
    Top-K Binary Cross Entropy Loss.
    
    Only backpropagates through the K% hardest (highest loss) samples.
    This forces the model to focus on difficult cases.
    
    Example:
        - If k=0.2 (20%), and we have 1000 pixels
        - Calculate loss for all 1000 pixels
        - Keep only the 200 pixels with HIGHEST loss
        - Average those 200 losses
        - Ignore the 800 "easy" pixels
    
    ```
    All Pixels Loss:      TopK (20%):
    ┌─────────────────┐   ┌─────────────────┐
    │ 0.01 0.02 0.90 │   │ ──── ──── 0.90 │ ← Keep (hard)
    │ 0.05 0.03 0.85 │ → │ ──── ──── 0.85 │ ← Keep (hard)
    │ 0.02 0.01 0.80 │   │ ──── ──── 0.80 │ ← Keep (hard)
    │ 0.01 0.02 0.01 │   │ ──── ──── ──── │
    │ ... (mostly 0) │   │ ... (ignored)   │
    └─────────────────┘   └─────────────────┘
       1000 pixels           200 pixels
       Average: ~0.05        Average: ~0.85
    ```
    """
    
    def __init__(self, k: float = 0.2, reduction: str = 'mean'):
        """
        Args:
            k: Fraction of hardest samples to keep (0.2 = top 20%)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.k = k
        self.reduction = reduction
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate TopK BCE Loss.
        
        Args:
            logits: Model output before sigmoid (B, C, ...)
            targets: Ground truth (same shape as logits)
            mask: Optional mask for valid regions
            
        Returns:
            Scalar loss value
        """
        # Calculate BCE loss per pixel (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply mask if provided
        if mask is not None:
            bce_loss = bce_loss * mask
        
        # Flatten to 1D for sorting
        flat_loss = bce_loss.view(-1)
        
        # Calculate how many samples to keep
        num_samples = flat_loss.numel()
        k_samples = max(1, int(num_samples * self.k))
        
        # Get top-k highest losses
        topk_losses, _ = torch.topk(flat_loss, k_samples)
        
        # Reduce
        if self.reduction == 'mean':
            return topk_losses.mean()
        elif self.reduction == 'sum':
            return topk_losses.sum()
        else:
            return topk_losses


class BlobRegressionLoss(nn.Module):
    """
    Combined loss for blob regression training.
    
    Components:
        1. TopK BCE Loss - Focus on hard samples
        2. Dice Loss - Encourage overlap with ground truth
        
    The blob regression approach converts point annotations
    to soft probability spheres (blobs) for easier learning.
    
    Ground Truth Creation:
        1. Take annotation point (x, y, z)
        2. Create sphere of radius R (65 voxels)
        3. Apply Euclidean Distance Transform
        4. Result: Soft probability that decays from center
        
    ```
    Point Annotation:           Blob Target:
           ●                    ░░███░░
         (single point)         ░█████░
                         →      ███████
                                ░█████░
                                ░░███░░
                            (probability sphere)
    ```
    """
    
    def __init__(self, 
                 topk_ratio: float = 0.2,
                 dice_weight: float = 0.5):
        super().__init__()
        self.topk_bce = TopKBCELoss(k=topk_ratio)
        self.dice_weight = dice_weight
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Soft Dice Loss.
        
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        Loss = 1 - Dice
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # Calculate dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        return 1 - dice
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            logits: Model output (B, C, Z, Y, X)
            targets: Blob ground truth (B, C, Z, Y, X)
            
        Returns:
            Combined loss
        """
        # TopK BCE
        bce = self.topk_bce(logits, targets)
        
        # Dice
        dice = self.dice_loss(logits, targets)
        
        # Combine
        total_loss = bce + self.dice_weight * dice
        
        return total_loss


def create_blob_target(center: tuple, 
                       volume_shape: tuple,
                       radius: int = 65) -> torch.Tensor:
    """
    Create a soft blob target from a center point.
    
    This is how training targets are created from point annotations.
    
    Args:
        center: (z, y, x) coordinates
        volume_shape: (Z, Y, X) shape of output
        radius: Blob radius in voxels
        
    Returns:
        3D tensor with probability blob
    """
    z, y, x = center
    Z, Y, X = volume_shape
    
    # Create coordinate grids
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z),
        torch.arange(Y),
        torch.arange(X),
        indexing='ij'
    )
    
    # Calculate distance from center
    distance = torch.sqrt(
        (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
    ).float()
    
    # Convert to probability (1 at center, 0 at radius)
    blob = 1 - (distance / radius).clamp(0, 1)
    
    # Square for sharper falloff
    blob = blob ** 2
    
    return blob


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Create dummy data
    batch_size = 2
    num_classes = 13
    size = 64
    
    logits = torch.randn(batch_size, num_classes, size, size, size)
    targets = torch.rand(batch_size, num_classes, size, size, size)
    
    # Test TopK BCE
    topk_loss = TopKBCELoss(k=0.2)
    loss1 = topk_loss(logits, targets)
    print(f"TopK BCE Loss: {loss1.item():.4f}")
    
    # Test full loss
    blob_loss = BlobRegressionLoss(topk_ratio=0.2, dice_weight=0.5)
    loss2 = blob_loss(logits, targets)
    print(f"Blob Regression Loss: {loss2.item():.4f}")
    
    # Test blob creation
    blob = create_blob_target(
        center=(32, 32, 32),
        volume_shape=(64, 64, 64),
        radius=15
    )
    print(f"Blob shape: {blob.shape}")
    print(f"Blob max: {blob.max():.3f}, min: {blob.min():.3f}")
