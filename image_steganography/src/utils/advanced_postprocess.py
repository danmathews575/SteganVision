"""
Advanced Post-Processing for Perfect Secret Recovery

Eliminates noise artifacts using morphological operations, 
connected components, and adaptive filtering.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple


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


def remove_noise_morphology(
    img: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Remove noise using morphological operations.
    
    - Opening: Removes small white noise (erosion + dilation)
    - Closing: Fills small black holes (dilation + erosion)
    
    Args:
        img: Binary image [0, 255]
        kernel_size: Size of morphological kernel (odd number)
        iterations: Number of times to apply operations
    
    Returns:
        Cleaned image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: Remove small white noise
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing: Fill small black holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return closed


def remove_noise_median(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Remove salt-and-pepper noise using median filter.
    
    Args:
        img: Grayscale image [0, 255]
        kernel_size: Size of median filter kernel (odd number)
    
    Returns:
        Filtered image
    """
    return cv2.medianBlur(img, kernel_size)


def keep_largest_component(img: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Keep only the largest connected component (the digit).
    Removes isolated noise pixels.
    
    Args:
        img: Binary image [0, 255]
        min_size: Minimum component size to keep (pixels)
    
    Returns:
        Image with only largest component
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    
    # Create output image
    output = np.zeros_like(img)
    
    # Skip label 0 (background)
    if num_labels > 1:
        # Find largest component (excluding background)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(sizes) + 1
        
        # Keep only if larger than min_size
        if stats[largest_label, cv2.CC_STAT_AREA] >= min_size:
            output[labels == largest_label] = 255
    
    return output


def adaptive_binarize(
    img: np.ndarray,
    blur_kernel: int = 5,
    method: str = 'otsu'
) -> np.ndarray:
    """
    Adaptive binarization with Gaussian smoothing.
    
    Args:
        img: Grayscale image [0, 255]
        blur_kernel: Gaussian blur kernel size (odd number)
        method: 'otsu' or 'adaptive'
    
    Returns:
        Binary image [0, 255]
    """
    # Gaussian smoothing to reduce noise before binarization
    blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    
    if method == 'otsu':
        # Otsu's method: Automatic optimal threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive threshold: Local thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    return binary


def perfect_clean_secret(
    recovered: torch.Tensor,
    aggressive: bool = True
) -> torch.Tensor:
    """
    Apply all post-processing steps for perfectly clean secret recovery.
    
    Pipeline:
    1. Denormalize to [0, 255]
    2. Median filter (remove salt-and-pepper)
    3. Gaussian blur + Otsu binarization
    4. Morphological opening (remove white noise)
    5. Morphological closing (fill black holes)
    6. Keep largest connected component
    7. Renormalize to [-1, 1]
    
    Args:
        recovered: Recovered secret tensor [B, C, H, W] in [-1, 1]
        aggressive: If True, use stronger noise removal
    
    Returns:
        Cleaned secret tensor [B, C, H, W] in [-1, 1]
    """
    batch_size, channels, height, width = recovered.shape
    cleaned_batch = []
    
    for b in range(batch_size):
        cleaned_channels = []
        
        for c in range(channels):
            # Convert to numpy [0, 255]
            img = denorm(recovered[b, c])
            
            # Step 1: Median filter (remove salt-and-pepper)
            if aggressive:
                img = remove_noise_median(img, kernel_size=5)
            else:
                img = remove_noise_median(img, kernel_size=3)
            
            # Step 2: Adaptive binarization
            binary = adaptive_binarize(img, blur_kernel=5, method='otsu')
            
            # Step 3: Morphological operations
            if aggressive:
                # Stronger noise removal
                cleaned = remove_noise_morphology(binary, kernel_size=3, iterations=2)
            else:
                cleaned = remove_noise_morphology(binary, kernel_size=3, iterations=1)
            
            # Step 4: Keep largest component (the digit)
            cleaned = keep_largest_component(cleaned, min_size=50)
            
            # Convert back to tensor [-1, 1]
            cleaned_tensor = renorm(cleaned)
            cleaned_channels.append(cleaned_tensor)
        
        # Stack channels
        cleaned_img = torch.stack(cleaned_channels, dim=0)
        cleaned_batch.append(cleaned_img)
    
    # Stack batch
    result = torch.stack(cleaned_batch, dim=0)
    
    return result.to(recovered.device)


def demo_comparison(recovered: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate before/after comparison for demo.
    
    Args:
        recovered: Raw recovered secret [B, C, H, W]
    
    Returns:
        (raw_binary, perfect_clean) tuple
    """
    # Simple binarization (before)
    raw_binary = (recovered > 0.0).float() * 2 - 1
    
    # Perfect cleaning (after)
    perfect = perfect_clean_secret(recovered, aggressive=True)
    
    return raw_binary, perfect


if __name__ == '__main__':
    # Test the post-processing
    import matplotlib.pyplot as plt
    
    # Create test image with noise
    test = torch.randn(1, 1, 256, 256) * 0.3
    test[0, 0, 50:200, 50:200] = 1.0  # White square (digit)
    
    # Add salt-and-pepper noise
    noise_mask = torch.rand_like(test) > 0.95
    test[noise_mask] = torch.randn_like(test[noise_mask])
    
    # Clean
    cleaned = perfect_clean_secret(test, aggressive=True)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denorm(test[0, 0]), cmap='gray')
    axes[0].set_title('Noisy Input')
    axes[1].imshow(denorm(cleaned[0, 0]), cmap='gray')
    axes[1].set_title('Perfect Clean')
    plt.tight_layout()
    plt.savefig('outputs/postprocess_test.png')
    print("✅ Test visualization saved to outputs/postprocess_test.png")
