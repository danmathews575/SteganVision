"""
CNN-based Encoder-Decoder for Deep Image Steganography.

Inspired by the HiDDeN paper architecture. This module provides:
- Encoder: Embeds a secret image into a cover image to produce a stego image
- Decoder: Extracts the hidden secret from the stego image

Architecture follows a U-Net style design for the encoder to preserve
spatial resolution and fine details during the embedding process.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block with Conv -> BatchNorm -> ReLU.
    
    This block preserves spatial dimensions using padding.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel (default: 3)
        """
        super().__init__()
        
        # Calculate padding to preserve spatial dimensions
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder path.
    
    Applies two conv blocks followed by max pooling for spatial reduction.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (pooled features, skip connection features)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder path.
    
    Upsamples features and concatenates with skip connections,
    followed by two conv blocks.
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels from previous layer
            skip_channels: Number of channels in skip connection
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Transposed convolution for upsampling (maintains channel count, doubles spatial)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # After concatenation with skip: in_channels + skip_channels
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features from previous layer
            skip: Skip connection features from encoder
        
        Returns:
            Upsampled and refined features
        """
        x = self.up(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    """
    U-Net style encoder for image steganography.
    
    Takes a cover image and secret image, embeds the secret into
    the cover to produce a stego image that looks similar to the cover.
    
    Architecture:
        - Input: Concatenated cover (3 channels) + secret (1 channel) = 4 channels
        - U-Net encoder-decoder structure with skip connections
        - Output: Stego image (3 channels) with Tanh activation [-1, 1]
    """
    
    def __init__(self, base_channels: int = 64):
        """
        Args:
            base_channels: Number of channels in first conv layer (default: 64)
        """
        super().__init__()
        
        # Initial convolution to process concatenated input (4 channels)
        self.init_conv = ConvBlock(4, base_channels)
        
        # Encoder path (downsampling)
        self.down1 = DownBlock(base_channels, base_channels * 2)      # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 256 -> 512
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        
        # Decoder path (upsampling with skip connections)
        # skip3 has 512 channels (from down3), skip2 has 256 (from down2), skip1 has 128 (from down1)
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)   # 512 + 512 -> 256
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)   # 256 + 256 -> 128
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels)       # 128 + 128 -> 64
        
        # Final convolution to produce stego image
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=1),  # 1x1 conv to get 3 channels
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        Embed secret image into cover image.
        
        Args:
            cover: Cover image tensor of shape (B, 3, 256, 256)
            secret: Secret image tensor of shape (B, 1, 256, 256)
        
        Returns:
            Stego image tensor of shape (B, 3, 256, 256)
        """
        # Concatenate cover and secret along channel dimension
        x = torch.cat([cover, secret], dim=1)  # (B, 4, 256, 256)
        
        # Initial conv
        x = self.init_conv(x)  # (B, 64, 256, 256)
        
        # Encoder path with skip connections
        x, skip1 = self.down1(x)   # (B, 128, 128, 128), skip1: (B, 128, 256, 256)
        x, skip2 = self.down2(x)   # (B, 256, 64, 64), skip2: (B, 256, 128, 128)
        x, skip3 = self.down3(x)   # (B, 512, 32, 32), skip3: (B, 512, 64, 64)
        
        # Bottleneck
        x = self.bottleneck(x)  # (B, 512, 32, 32)
        
        # Decoder path with skip connections
        x = self.up1(x, skip3)  # (B, 256, 64, 64)
        x = self.up2(x, skip2)  # (B, 128, 128, 128)
        x = self.up3(x, skip1)  # (B, 64, 256, 256)
        
        # Generate stego image
        stego = self.final_conv(x)  # (B, 3, 256, 256)
        
        return stego


class Decoder(nn.Module):
    """
    CNN-based decoder for extracting hidden secret from stego image.
    
    Takes a stego image and attempts to reconstruct the original
    secret image that was embedded by the encoder.
    
    Architecture:
        - Input: Stego image (3 channels)
        - Series of convolutional blocks
        - Output: Reconstructed secret (1 channel) with Tanh activation [-1, 1]
    """
    
    def __init__(self, base_channels: int = 64):
        """
        Args:
            base_channels: Number of channels in first conv layer (default: 64)
        """
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Initial feature extraction
            ConvBlock(3, base_channels),                   # 3 -> 64
            ConvBlock(base_channels, base_channels),       # 64 -> 64
            
            # Deeper feature extraction
            ConvBlock(base_channels, base_channels * 2),   # 64 -> 128
            ConvBlock(base_channels * 2, base_channels * 2),  # 128 -> 128
            
            # Further processing
            ConvBlock(base_channels * 2, base_channels * 4),  # 128 -> 256
            ConvBlock(base_channels * 4, base_channels * 4),  # 256 -> 256
            
            # Refinement
            ConvBlock(base_channels * 4, base_channels * 2),  # 256 -> 128
            ConvBlock(base_channels * 2, base_channels),      # 128 -> 64
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),  # 64 -> 1
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden secret from stego image.
        
        Args:
            stego: Stego image tensor of shape (B, 3, 256, 256)
        
        Returns:
            Reconstructed secret tensor of shape (B, 1, 256, 256)
        """
        x = self.features(stego)
        secret = self.output(x)
        return secret


class EncoderDecoder(nn.Module):
    """
    Combined Encoder-Decoder model for end-to-end image steganography.
    
    This model performs both encoding (hiding) and decoding (revealing)
    operations for deep image steganography.
    
    Usage:
        model = EncoderDecoder()
        stego, reconstructed_secret = model(cover_image, secret_image)
    """
    
    def __init__(self, base_channels: int = 64):
        """
        Args:
            base_channels: Number of base channels for both encoder and decoder
        """
        super().__init__()
        
        self.encoder = Encoder(base_channels=base_channels)
        self.decoder = Decoder(base_channels=base_channels)
    
    def forward(
        self, cover: torch.Tensor, secret: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform full steganography pipeline: encode then decode.
        
        Args:
            cover: Cover image tensor of shape (B, 3, 256, 256)
            secret: Secret image tensor of shape (B, 1, 256, 256)
        
        Returns:
            Tuple of:
                - stego: Stego image of shape (B, 3, 256, 256)
                - reconstructed_secret: Decoded secret of shape (B, 1, 256, 256)
        """
        # Encode: embed secret into cover
        stego = self.encoder(cover, secret)
        
        # Decode: extract secret from stego
        reconstructed_secret = self.decoder(stego)
        
        return stego, reconstructed_secret
    
    def encode(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        Encode only: embed secret into cover image.
        
        Args:
            cover: Cover image tensor of shape (B, 3, 256, 256)
            secret: Secret image tensor of shape (B, 1, 256, 256)
        
        Returns:
            Stego image of shape (B, 3, 256, 256)
        """
        return self.encoder(cover, secret)
    
    def decode(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Decode only: extract hidden secret from stego image.
        
        Args:
            stego: Stego image tensor of shape (B, 3, 256, 256)
        
        Returns:
            Reconstructed secret of shape (B, 1, 256, 256)
        """
        return self.decoder(stego)
