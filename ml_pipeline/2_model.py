"""
2. Model Architecture
=====================
ResidualEncoderUNet - The neural network used for aneurysm detection.

Architecture Overview:
    - Encoder: 6 stages with residual blocks
    - Decoder: 5 stages with skip connections
    - Output: 13 channels (one per anatomical location)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    """
    Basic convolution block: Conv3D → InstanceNorm → LeakyReLU
    
    This is the building block of the network.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            bias=False
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.
    
    Architecture:
        Input ────────────────────────────┐
          ↓                               │ (skip connection)
        Conv3D → Norm → ReLU              │
          ↓                               │
        Conv3D → Norm                     │
          ↓                               │
        Add ←─────────────────────────────┘
          ↓
        ReLU
          ↓
        Output
    
    Why? Prevents vanishing gradient problem in deep networks.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Save input for skip connection
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual  # Skip connection (add input)
        out = self.activation(out)
        
        return out


class EncoderStage(nn.Module):
    """
    One stage of the encoder.
    
    Structure:
        Input → ConvBlock → ResidualBlock → ResidualBlock → Output
              ↓
            Downsample (stride=2 conv)
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_residuals: int = 2, downsample: bool = True):
        super().__init__()
        
        # Initial convolution (may change channels)
        self.initial_conv = ConvBlock(in_channels, out_channels)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_residuals)]
        )
        
        # Downsampling (stride 2) - optional for last stage
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv3d(
                out_channels, out_channels,
                kernel_size=2, stride=2, bias=False
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process at current resolution
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        
        # Skip connection (before downsampling)
        skip = x
        
        # Downsample for next stage
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skip  # Return both: downsampled and skip connection


class DecoderStage(nn.Module):
    """
    One stage of the decoder.
    
    Structure:
        Upsample → Concatenate with skip → ConvBlock → ResidualBlock
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )
        
        # Convolution after concatenation
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x)
        x = self.residual(x)
        return x


class ResidualEncoderUNet(nn.Module):
    """
    ResidualEncoderUNet - The main architecture.
    
    Architecture Diagram:
    
        Input (1ch)
            ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │  ENCODER                                                        │
        │  Stage 1: 32ch  ──skip1──→                                     │
        │  Stage 2: 64ch  ──skip2──→                                     │
        │  Stage 3: 128ch ──skip3──→                                     │
        │  Stage 4: 256ch ──skip4──→                                     │
        │  Stage 5: 320ch ──skip5──→                                     │
        │  Stage 6: 320ch (Bottleneck)                                   │
        └─────────────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │  DECODER                                                        │
        │  Stage 5: ←──skip5── + 320ch → 256ch                           │
        │  Stage 4: ←──skip4── + 256ch → 128ch                           │
        │  Stage 3: ←──skip3── + 128ch → 64ch                            │
        │  Stage 2: ←──skip2── + 64ch  → 32ch                            │
        │  Stage 1: ←──skip1── + 32ch  → 32ch                            │
        └─────────────────────────────────────────────────────────────────┘
            ↓
        Output Head (32ch → 13ch)
            ↓
        Output (13 probability maps)
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 13,
                 features: List[int] = [32, 64, 128, 256, 320, 320]):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        
        # ============================================
        # ENCODER
        # ============================================
        self.encoder_stages = nn.ModuleList()
        
        in_ch = in_channels
        for i, out_ch in enumerate(features):
            # Last encoder stage doesn't downsample (it's the bottleneck)
            downsample = (i < len(features) - 1)
            self.encoder_stages.append(EncoderStage(in_ch, out_ch, downsample=downsample))
            in_ch = out_ch
        
        # ============================================
        # DECODER
        # ============================================
        self.decoder_stages = nn.ModuleList()
        
        # Decoder goes in reverse order
        for i in range(len(features) - 1, 0, -1):
            in_ch = features[i]
            skip_ch = features[i - 1]
            out_ch = features[i - 1]
            self.decoder_stages.append(DecoderStage(in_ch, skip_ch, out_ch))
        
        # ============================================
        # OUTPUT HEAD
        # ============================================
        self.output_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (B, C, Z, Y, X)
            
        Returns:
            Logits tensor (B, num_classes, Z, Y, X)
        """
        # ============================================
        # ENCODER: Extract features at multiple scales
        # ============================================
        skips = []
        
        for i, encoder_stage in enumerate(self.encoder_stages):
            x, skip = encoder_stage(x)
            # Don't store bottleneck as skip (last stage)
            if i < len(self.encoder_stages) - 1:
                skips.append(skip)
        
        # x is now the bottleneck features
        
        # ============================================
        # DECODER: Upsample and combine with skips
        # ============================================
        for decoder_stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = decoder_stage(x, skip)
        
        # ============================================
        # OUTPUT: Generate probability maps
        # ============================================
        logits = self.output_conv(x)
        
        return logits


def get_model_summary(model: nn.Module) -> str:
    """Get a summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return f"""
    Model: ResidualEncoderUNet
    ==========================
    Total Parameters:     {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Input Channels:       {model.in_channels}
    Output Classes:       {model.num_classes}
    Feature Stages:       {model.features}
    """


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Create model
    model = ResidualEncoderUNet(
        in_channels=1,
        num_classes=13,
        features=[32, 64, 128, 256, 320, 320]
    )
    
    print(get_model_summary(model))
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
