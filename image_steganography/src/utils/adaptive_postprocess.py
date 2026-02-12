"""
Adaptive Post-Processing for Image Steganography

Provides different post-processing methods optimized for:
1. Binary content (MNIST digits, QR codes, logos) - Morphological cleaning
2. Grayscale photos (continuous-tone images) - Denoising filters

Auto-detects content type and applies appropriate method.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Literal


def denorm(x: torch.Tensor) -> np.ndarray:
    """Convert tensor from [-1, 1] to [0, 255] numpy array"""
    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
    x = (x * 255).clamp(0, 255).byte()
    return x.cpu().numpy()


def renorm(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array from [0, 255] to [-1, 1] tensor"""
    x = torch.from_numpy(x).float() / 255.0  # [0, 255] -> [0, 1]
    x = x * 2 - 1  # [0, 1] -> [-1, 1]
    return x


# =============================================================================
# Binary Content Post-Processing (for MNIST, QR codes, logos)
# =============================================================================

def postprocess_binary(
    recovered: torch.Tensor,
    aggressive: bool = True
) -> torch.Tensor:
    """
    Post-processing optimized for BINARY content (digits, QR codes, logos).
    
    Uses morphological operations and connected components to eliminate
    noise while preserving binary structure.
    
    Args:
        recovered: Recovered secret [B, C, H, W] in [-1, 1]
        aggressive: Use stronger noise removal
    
    Returns:
        Cleaned binary secret [B, C, H, W] in [-1, 1]
    """
    batch_size, channels, height, width = recovered.shape
    cleaned_batch = []
    
    for b in range(batch_size):
        cleaned_channels = []
        
        for c in range(channels):
            img = denorm(recovered[b, c])
            
            # Median filter
            kernel_size = 5 if aggressive else 3
            img = cv2.medianBlur(img, kernel_size)
            
            # Gaussian blur + Otsu binarization
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            iterations = 2 if aggressive else 1
            
            # Opening: Remove white noise
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
            
            # Closing: Fill black holes
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Keep largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            output = np.zeros_like(closed)
            
            if num_labels > 1:
                sizes = stats[1:, cv2.CC_STAT_AREA]
                largest_label = np.argmax(sizes) + 1
                if stats[largest_label, cv2.CC_STAT_AREA] >= 50:
                    output[labels == largest_label] = 255
            
            cleaned_tensor = renorm(output)
            cleaned_channels.append(cleaned_tensor)
        
        cleaned_img = torch.stack(cleaned_channels, dim=0)
        cleaned_batch.append(cleaned_img)
    
    result = torch.stack(cleaned_batch, dim=0)
    return result.to(recovered.device)


# =============================================================================
# Grayscale Photo Post-Processing (for continuous-tone images)
# =============================================================================

def postprocess_grayscale(
    recovered: torch.Tensor,
    denoise_strength: float = 5.0
) -> torch.Tensor:
    """
    Post-processing optimized for GRAYSCALE PHOTOS (continuous-tone images).
    
    Uses bilateral filtering to preserve edges while reducing noise.
    Does NOT binarize - preserves grayscale gradients.
    
    Args:
        recovered: Recovered secret [B, C, H, W] in [-1, 1]
        denoise_strength: Strength of denoising (3-10, lower = less smoothing)
    
    Returns:
        Denoised grayscale secret [B, C, H, W] in [-1, 1]
    """
    batch_size, channels, height, width = recovered.shape
    cleaned_batch = []
    
    for b in range(batch_size):
        cleaned_channels = []
        
        for c in range(channels):
            img = denorm(recovered[b, c])
            
            # Bilateral filter: Edge-preserving smoothing
            # d: Diameter of pixel neighborhood
            # sigmaColor: Filter sigma in color space (larger = more colors mixed)
            # sigmaSpace: Filter sigma in coordinate space (larger = farther pixels influence)
            denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            
            # Optional: Very light Non-Local Means for additional noise reduction
            if denoise_strength > 5.0:
                denoised = cv2.fastNlMeansDenoising(
                    denoised,
                    None,
                    h=denoise_strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            
            # Light unsharp masking to recover edge details
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # Blend to avoid over-sharpening
            output = cv2.addWeighted(denoised, 0.7, unsharp, 0.3, 0)
            
            cleaned_tensor = renorm(output)
            cleaned_channels.append(cleaned_tensor)
        
        cleaned_img = torch.stack(cleaned_channels, dim=0)
        cleaned_batch.append(cleaned_img)
    
    result = torch.stack(cleaned_batch, dim=0)
    return result.to(recovered.device)



# =============================================================================
# Auto-Detection and Adaptive Processing
# =============================================================================

def detect_content_type(recovered: torch.Tensor) -> Literal['binary', 'grayscale']:
    """
    Auto-detect if recovered secret is binary or grayscale content.
    
    Heuristic: Binary content has bimodal histogram (peaks at 0 and 1),
    while grayscale photos have more uniform distribution.
    
    Args:
        recovered: Recovered secret [B, C, H, W] in [-1, 1]
    
    Returns:
        'binary' or 'grayscale'
    """
    # Convert to [0, 1] range
    img_01 = (recovered + 1) / 2
    
    # Compute histogram
    hist = torch.histc(img_01, bins=256, min=0, max=1)
    
    # Check if bimodal (peaks at extremes)
    # Binary content has most pixels near 0 or 1
    edge_pixels = hist[:50].sum() + hist[-50:].sum()
    total_pixels = hist.sum()
    edge_ratio = edge_pixels / total_pixels
    
    # If >60% of pixels are near edges, it's likely binary
    if edge_ratio > 0.6:
        return 'binary'
    else:
        return 'grayscale'


def adaptive_postprocess(
    recovered: torch.Tensor,
    content_type: Literal['binary', 'grayscale', 'auto'] = 'auto',
    **kwargs
) -> torch.Tensor:
    """
    Adaptive post-processing that automatically selects the best method.
    
    Args:
        recovered: Recovered secret [B, C, H, W] in [-1, 1]
        content_type: 'binary', 'grayscale', or 'auto' (auto-detect)
        **kwargs: Additional arguments for specific post-processing methods
    
    Returns:
        Cleaned secret [B, C, H, W] in [-1, 1]
    """
    if content_type == 'auto':
        content_type = detect_content_type(recovered)
        print(f"Auto-detected content type: {content_type}")
    
    if content_type == 'binary':
        aggressive = kwargs.get('aggressive', True)
        return postprocess_binary(recovered, aggressive=aggressive)
    else:  # grayscale
        denoise_strength = kwargs.get('denoise_strength', 10.0)
        return postprocess_grayscale(recovered, denoise_strength=denoise_strength)


# =============================================================================
# Backward Compatibility
# =============================================================================

def perfect_clean_secret(
    recovered: torch.Tensor,
    aggressive: bool = True,
    content_type: Literal['binary', 'grayscale', 'auto'] = 'auto'
) -> torch.Tensor:
    """
    Backward-compatible wrapper for adaptive post-processing.
    
    Automatically detects content type and applies appropriate method.
    """
    return adaptive_postprocess(
        recovered,
        content_type=content_type,
        aggressive=aggressive,
        denoise_strength=5.0
    )


if __name__ == '__main__':
    # Test the adaptive system
    print("Testing adaptive post-processing system...")
    
    # Test binary content
    binary_test = torch.randn(1, 1, 256, 256) * 0.3
    binary_test[0, 0, 50:200, 50:200] = 1.0
    noise_mask = torch.rand_like(binary_test) > 0.95
    binary_test[noise_mask] = torch.randn_like(binary_test[noise_mask])
    
    detected = detect_content_type(binary_test)
    print(f"Binary test detected as: {detected}")
    
    # Test grayscale content
    grayscale_test = torch.randn(1, 1, 256, 256) * 0.5
    detected = detect_content_type(grayscale_test)
    print(f"Grayscale test detected as: {detected}")
    
    print("\n✅ Adaptive post-processing system ready!")
