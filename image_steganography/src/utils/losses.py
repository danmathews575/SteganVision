"""
Loss Functions for CNN-based Image Steganography

This module provides loss functions for training the encoder-decoder
steganography model. Uses L1 loss for both cover and secret image reconstruction.

Total Loss = Cover Loss + Secret Loss + SSIM Loss (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# SSIM Loss (Structural Similarity)
# =============================================================================

def gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D kernel
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]
    kernel = kernel.repeat(channels, 1, 1, 1)  # [channels, 1, size, size]
    
    return kernel


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,  # For [-1, 1] normalized images
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute SSIM between two images.
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of Gaussian window
        sigma: Std dev of Gaussian window
        data_range: Dynamic range (2.0 for [-1, 1] images)
        size_average: If True, return mean SSIM
        
    Returns:
        SSIM value (higher is better, max 1.0)
    """
    channels = img1.size(1)
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma, channels).to(img1.device, img1.dtype)
    
    # Compute means
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channels) - mu1_mu2
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3))


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM loss (1 - SSIM).
    
    Args:
        img1: First image [B, C, H, W] in range [-1, 1]
        img2: Second image [B, C, H, W] in range [-1, 1]
        
    Returns:
        SSIM loss (lower is better, min 0.0)
    """
    return 1.0 - compute_ssim(img1, img2)


class SteganographyLoss(nn.Module):
    """
    Combined loss function for image steganography.
    
    Computes:
        - Cover loss: L1 distance between cover image and stego image
        - Secret loss: L1 distance between secret image and reconstructed secret
        - Total loss: Sum of cover loss and secret loss
    """
    
    def __init__(self):
        super(SteganographyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        cover_image: torch.Tensor,
        stego_image: torch.Tensor,
        secret_image: torch.Tensor,
        reconstructed_secret: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute steganography losses.
        
        Args:
            cover_image: Original cover image [B, C, H, W]
            stego_image: Encoded stego image (cover + hidden secret) [B, C, H, W]
            secret_image: Original secret image [B, C, H, W]
            reconstructed_secret: Decoded secret image [B, C, H, W]
        
        Returns:
            Tuple of (total_loss, cover_loss, secret_loss)
        """
        # Cover loss: stego should look like cover
        cover_loss = self.l1_loss(stego_image, cover_image)
        
        # Secret loss: reconstructed secret should match original secret
        secret_loss = self.l1_loss(reconstructed_secret, secret_image)
        
        # Total loss
        total_loss = cover_loss + secret_loss
        
        return total_loss, cover_loss, secret_loss


def compute_cover_loss(
    cover_image: torch.Tensor,
    stego_image: torch.Tensor
) -> torch.Tensor:
    """
    Compute L1 loss between cover image and stego image.
    
    Args:
        cover_image: Original cover image [B, C, H, W]
        stego_image: Encoded stego image [B, C, H, W]
    
    Returns:
        L1 loss between cover and stego images
    """
    return nn.functional.l1_loss(stego_image, cover_image)


def compute_secret_loss(
    secret_image: torch.Tensor,
    reconstructed_secret: torch.Tensor
) -> torch.Tensor:
    """
    Compute L1 loss between secret image and reconstructed secret.
    
    Args:
        secret_image: Original secret image [B, C, H, W]
        reconstructed_secret: Decoded secret image [B, C, H, W]
    
    Returns:
        L1 loss between original and reconstructed secret images
    """
    return nn.functional.l1_loss(reconstructed_secret, secret_image)


def compute_secret_loss_bce(
    secret_image: torch.Tensor,
    reconstructed_secret: torch.Tensor
) -> torch.Tensor:
    """
    Compute BCE loss for BINARY secret images (e.g., MNIST digits).
    
    This loss is superior to L1 for binary/high-contrast secrets because:
    - Forces predictions toward 0 or 1 (not gray values like 0.3, 0.7)
    - Eliminates speckle noise and broken strokes
    - Produces clean, readable digits
    
    Args:
        secret_image: Original secret image [B, C, H, W] in range [-1, 1]
        reconstructed_secret: Decoded secret image [B, C, H, W] in range [-1, 1]
    
    Returns:
        BCE loss (lower is better)
    """
    # Convert from [-1, 1] to [0, 1] for BCE
    secret_01 = (secret_image + 1) / 2
    
    # Use BCEWithLogitsLoss for AMP safety (no need to clamp)
    # Note: reconstructed_secret is treated as logits here
    # We need to scale it appropriately
    logits = reconstructed_secret * 3  # Scale to reasonable logit range
    
    return nn.functional.binary_cross_entropy_with_logits(logits, secret_01)


def compute_secret_loss_hybrid(
    secret_image: torch.Tensor,
    reconstructed_secret: torch.Tensor,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Hybrid loss: BCE (for binary structure) + L1 (for smooth gradients).
    
    Best of both worlds:
    - BCE: Forces binary predictions, eliminates noise
    - L1: Provides smooth gradients for training stability
    
    Args:
        secret_image: Original secret image [B, C, H, W] in range [-1, 1]
        reconstructed_secret: Decoded secret image [B, C, H, W] in range [-1, 1]
        alpha: Weight for BCE (default 0.7), (1-alpha) for L1
    
    Returns:
        Hybrid loss
    """
    bce_loss = compute_secret_loss_bce(secret_image, reconstructed_secret)
    l1_loss = compute_secret_loss(secret_image, reconstructed_secret)
    
    return alpha * bce_loss + (1 - alpha) * l1_loss

def compute_total_loss(
    cover_image: torch.Tensor,
    stego_image: torch.Tensor,
    secret_image: torch.Tensor,
    reconstructed_secret: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all steganography losses.
    
    Args:
        cover_image: Original cover image [B, C, H, W]
        stego_image: Encoded stego image [B, C, H, W]
        secret_image: Original secret image [B, C, H, W]
        reconstructed_secret: Decoded secret image [B, C, H, W]
    
    Returns:
        Tuple of (total_loss, cover_loss, secret_loss)
    """
    cover_loss = compute_cover_loss(cover_image, stego_image)
    secret_loss = compute_secret_loss(secret_image, reconstructed_secret)
    total_loss = cover_loss + secret_loss
    
    return total_loss, cover_loss, secret_loss


# Convenience alias for quick usage
def stego_loss(
    cover: torch.Tensor,
    stego: torch.Tensor,
    secret: torch.Tensor,
    recovered: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function for computing steganography losses.
    
    Args:
        cover: Original cover image [B, C, H, W]
        stego: Encoded stego image [B, C, H, W]
        secret: Original secret image [B, C, H, W]
        recovered: Decoded/reconstructed secret image [B, C, H, W]
    
    Returns:
        Tuple of (total_loss, cover_loss, secret_loss)
    """
    return compute_total_loss(cover, stego, secret, recovered)


# =============================================================================
# GAN Loss Functions
# =============================================================================

def adversarial_loss_lsgan(
    predictions: torch.Tensor,
    is_real: bool
) -> torch.Tensor:
    """
    Compute LSGAN adversarial loss (MSE-based).
    
    LSGAN uses MSE loss with targets:
    - Real: target = 1.0
    - Fake: target = 0.0
    
    Args:
        predictions: Discriminator output [B, 1]
        is_real: Whether these should be classified as real
        
    Returns:
        LSGAN loss value
    """
    target_value = 1.0 if is_real else 0.0
    target = torch.full_like(predictions, target_value)
    return nn.functional.mse_loss(predictions, target)


def adversarial_loss_bce(
    predictions: torch.Tensor,
    is_real: bool
) -> torch.Tensor:
    """
    Compute BCE adversarial loss (vanilla GAN).
    
    BCE loss with targets:
    - Real: target = 1.0
    - Fake: target = 0.0
    
    Args:
        predictions: Discriminator output [B, 1] (after sigmoid)
        is_real: Whether these should be classified as real
        
    Returns:
        BCE loss value
    """
    target_value = 1.0 if is_real else 0.0
    target = torch.full_like(predictions, target_value)
    return nn.functional.binary_cross_entropy(predictions, target)


def discriminator_loss(
    real_preds: torch.Tensor,
    fake_preds: torch.Tensor,
    loss_type: str = 'lsgan'
) -> torch.Tensor:
    """
    Compute discriminator loss for GAN training.
    
    D wants to: classify real as 1, fake as 0
    
    Args:
        real_preds: Discriminator output for real (cover) images [B, 1]
        fake_preds: Discriminator output for fake (stego) images [B, 1]
        loss_type: 'lsgan' or 'bce'
        
    Returns:
        Total discriminator loss
    """
    if loss_type == 'lsgan':
        loss_fn = adversarial_loss_lsgan
    else:
        loss_fn = adversarial_loss_bce
    
    real_loss = loss_fn(real_preds, is_real=True)
    fake_loss = loss_fn(fake_preds, is_real=False)
    
    return (real_loss + fake_loss) / 2


def generator_adversarial_loss(
    fake_preds: torch.Tensor,
    loss_type: str = 'lsgan'
) -> torch.Tensor:
    """
    Compute generator adversarial loss.
    
    G wants to: make discriminator classify fake as real (1)
    
    Args:
        fake_preds: Discriminator output for fake (stego) images [B, 1]
        loss_type: 'lsgan' or 'bce'
        
    Returns:
        Generator adversarial loss
    """
    if loss_type == 'lsgan':
        return adversarial_loss_lsgan(fake_preds, is_real=True)
    else:
        return adversarial_loss_bce(fake_preds, is_real=True)


def generator_loss_gan(
    cover: torch.Tensor,
    stego: torch.Tensor,
    secret: torch.Tensor,
    recovered: torch.Tensor,
    fake_preds: torch.Tensor,
    lambda_cover: float = 1.0,
    lambda_secret: float = 1.0,
    lambda_adv: float = 0.01,
    lambda_ssim: float = 0.5,
    loss_type: str = 'lsgan'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total generator loss for GAN training with SSIM.
    
    L_G = λ1 * L1(cover, stego) + λ2 * L1(secret, recovered) + λ3 * L_adv + λ4 * SSIM_loss
    
    Args:
        cover: Original cover image [B, C, H, W]
        stego: Generated stego image [B, C, H, W]
        secret: Original secret image [B, C, H, W]
        recovered: Decoded secret image [B, C, H, W]
        fake_preds: Discriminator output for stego images [B, 1]
        lambda_cover: Weight for cover loss (default: 1.0)
        lambda_secret: Weight for secret loss (default: 1.0)
        lambda_adv: Weight for adversarial loss (default: 0.01)
        lambda_ssim: Weight for SSIM loss (default: 0.5)
        loss_type: 'lsgan' or 'bce'
        
    Returns:
        Tuple of (total_loss, cover_loss, secret_loss, adv_loss, ssim_loss_val)
    """
    # Reconstruction losses
    cover_loss = compute_cover_loss(cover, stego)
    secret_loss = compute_secret_loss_hybrid(secret, recovered)  # BCE+L1 for binary secrets
    
    # SSIM loss for perceptual quality
    ssim_loss_val = ssim_loss(cover, stego)
    
    # Adversarial loss (generator wants fake to be classified as real)
    adv_loss = generator_adversarial_loss(fake_preds, loss_type)
    
    # Weighted total loss
    total_loss = (
        lambda_cover * cover_loss +
        lambda_secret * secret_loss +
        lambda_adv * adv_loss +
        lambda_ssim * ssim_loss_val
    )
    
    return total_loss, cover_loss, secret_loss, adv_loss, ssim_loss_val


# Backward compatible version (no SSIM)
def generator_loss_gan_legacy(
    cover: torch.Tensor,
    stego: torch.Tensor,
    secret: torch.Tensor,
    recovered: torch.Tensor,
    fake_preds: torch.Tensor,
    lambda_cover: float = 1.0,
    lambda_secret: float = 1.0,
    lambda_adv: float = 0.01,
    loss_type: str = 'lsgan'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy version without SSIM for backward compatibility."""
    cover_loss = compute_cover_loss(cover, stego)
    secret_loss = compute_secret_loss(secret, recovered)
    adv_loss = generator_adversarial_loss(fake_preds, loss_type)
    
    total_loss = (
        lambda_cover * cover_loss +
        lambda_secret * secret_loss +
        lambda_adv * adv_loss
    )
    
    return total_loss, cover_loss, secret_loss, adv_loss
