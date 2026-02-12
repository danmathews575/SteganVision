"""
Post-processing utilities for cleaner secret recovery
"""
import torch
import numpy as np


def binarize_secret(recovered_secret: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Binarize recovered secret image to eliminate noise.
    
    For MNIST-style binary secrets, this converts gray predictions to clean 0/1.
    
    Args:
        recovered_secret: Decoded secret [B, C, H, W] in range [-1, 1]
        threshold: Binarization threshold in [-1, 1] range (default: 0.0)
    
    Returns:
        Binarized secret (values are exactly -1 or 1)
    """
    return torch.where(recovered_secret > threshold, 
                      torch.ones_like(recovered_secret), 
                      -torch.ones_like(recovered_secret))


def adaptive_binarize_secret(recovered_secret: torch.Tensor) -> torch.Tensor:
    """
    Adaptive binarization using Otsu's method (per-image threshold).
    
    Better than fixed threshold for varying image conditions.
    
    Args:
        recovered_secret: Decoded secret [B, C, H, W] in range [-1, 1]
    
    Returns:
        Binarized secret
    """
    # Convert to [0, 1] for easier processing
    secret_01 = (recovered_secret + 1) / 2
    
    # Compute per-image threshold (Otsu-like)
    batch_size = secret_01.shape[0]
    result = torch.zeros_like(recovered_secret)
    
    for i in range(batch_size):
        img = secret_01[i]
        # Use median as simple adaptive threshold
        threshold = img.median()
        result[i] = torch.where(secret_01[i] > threshold, 
                               torch.ones_like(secret_01[i]), 
                               -torch.ones_like(secret_01[i]))
    
    return result


def denoise_secret(recovered_secret: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Apply median filter to remove speckle noise before binarization.
    
    Args:
        recovered_secret: Decoded secret [B, C, H, W] in range [-1, 1]
        kernel_size: Median filter kernel size (default: 3)
    
    Returns:
        Denoised secret
    """
    import torch.nn.functional as F
    
    # Simple median filter approximation using max pooling
    # (True median filter requires scipy or custom CUDA kernel)
    
    # Convert to [0, 1]
    secret_01 = (recovered_secret + 1) / 2
    
    # Apply slight Gaussian blur to reduce noise
    padding = kernel_size // 2
    blurred = F.avg_pool2d(secret_01, kernel_size, stride=1, padding=padding)
    
    # Convert back to [-1, 1]
    return blurred * 2 - 1
