"""
PatchGAN Discriminator for GAN-based Image Steganography.

Distinguishes between cover images (real) and stego images (fake).
Uses spectral normalization for training stability.

Architecture:
- 4 convolutional layers with stride-2 downsampling
- Spectral normalization (no BatchNorm)
- LeakyReLU activations
- Outputs patch-level predictions averaged to scalar
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for adversarial training.
    
    Classifies images as real (cover) or fake (stego) using
    patch-level predictions for improved spatial awareness.
    
    Features:
    - Spectral normalization for stable training
    - LeakyReLU activations
    - ~1.5M parameters (VRAM-friendly)
    - Works with 256x256 input images
    
    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of filters (default: 64)
        use_sigmoid: Whether to apply sigmoid to output (default: True for BCE)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        use_sigmoid: bool = True
    ):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        
        # Layer 1: 256x256 -> 128x128
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, base_channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2: 128x128 -> 64x64
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3: 64x64 -> 32x32
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4: 32x32 -> 16x16
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer: 16x16 -> 16x16 (1 channel)
        self.conv_out = spectral_norm(nn.Conv2d(base_channels * 8, 1, 3, 1, 1, bias=False))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            image: Input image tensor [B, 3, 256, 256]
            
        Returns:
            Probability of being real [B, 1] (averaged over patches)
        """
        # Feature extraction
        x = self.conv1(image)   # [B, 64, 128, 128]
        x = self.conv2(x)       # [B, 128, 64, 64]
        x = self.conv3(x)       # [B, 256, 32, 32]
        x = self.conv4(x)       # [B, 512, 16, 16]
        
        # Patch predictions
        x = self.conv_out(x)    # [B, 1, 16, 16]
        
        # Average over spatial dimensions
        x = x.mean(dim=[2, 3])  # [B, 1]
        
        # Apply sigmoid if using BCE loss
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        
        return x


# Alias for backward compatibility
Discriminator = PatchDiscriminator


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    disc = PatchDiscriminator()
    print(f"Discriminator parameters: {count_parameters(disc):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    out = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
