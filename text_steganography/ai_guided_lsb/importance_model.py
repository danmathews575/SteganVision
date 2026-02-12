"""
AI-Guided Adaptive LSB Steganography - Importance Model

This module generates an AI-inspired importance map using edge detection
and texture analysis to guide adaptive LSB embedding.

ENGINEERING JUSTIFICATION:
---------------------------------------------------------------------------
WHY AI-INSPIRED IMPORTANCE MAPS IMPROVE IMPERCEPTIBILITY:

LSB changes are INVISIBLE in high-texture regions but VISIBLE in smooth regions.
The human visual system (HVS) is less sensitive to changes in:
  - Edges (rapid intensity transitions)
  - Textured areas (high local variance)
  - Noisy regions

By embedding bits ONLY in high-importance (high-texture/edge) pixels,
we make the stego image visually indistinguishable from the original.

METHOD: Sobel Edge Detection + Laplacian Variance
  - Sobel: Detects strong intensity gradients (edges)
  - Laplacian: Measures local texture/contrast
  - Combined: Robust importance score

This is a hybrid intelligent system that uses signal processing heuristics
to optimize embedding locations - achieving the goal of "AI-guided" without
requiring ML training.
---------------------------------------------------------------------------
"""

import cv2
import numpy as np
from typing import Tuple


def compute_importance_map(image: np.ndarray) -> np.ndarray:
    """
    Compute an AI-inspired importance map for adaptive LSB embedding.
    
    High importance = safe to embed (edges, textures)
    Low importance = avoid (smooth regions)
    
    Args:
        image: RGB image as numpy array (H, W, 3), dtype uint8
        
    Returns:
        Importance map (H, W), normalized to [0, 1], dtype float32
        
    Note:
        This function is DETERMINISTIC - same input always produces same output.
        
    CRITICAL DESIGN NOTE:
        To ensure encoder and decoder produce IDENTICAL importance maps,
        we quantize pixel values to their MSBs (clear the lower 2 bits).
        This makes the importance map INVARIANT to LSB modifications.
    """
    # CRITICAL: Quantize RGB image BEFORE grayscale conversion
    # This ensures the importance map is completely LSB-invariant
    # Mask: 0xF8 = 11111000 (clears lowest 3 bits from each channel)
    # We use 3 bits because grayscale = 0.299*R + 0.587*G + 0.114*B
    # and the weighted sum of LSB changes could affect grayscale values
    image_quantized = (image & 0xF8).astype(np.uint8)
    
    # Convert quantized image to grayscale for analysis
    if len(image_quantized.shape) == 3:
        gray = cv2.cvtColor(image_quantized, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_quantized.copy()
    
    gray = gray.astype(np.float32)
    
    # --- Component 1: Sobel Edge Detection ---
    # Detects strong horizontal and vertical edges
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize Sobel to [0, 1]
    sobel_norm = sobel_magnitude / (sobel_magnitude.max() + 1e-8)
    
    # --- Component 2: Laplacian Variance (Local Texture) ---
    # High variance = textured region = good for embedding
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    
    # Normalize Laplacian to [0, 1]
    laplacian_norm = laplacian_abs / (laplacian_abs.max() + 1e-8)
    
    # --- Component 3: Local Variance (Sliding Window) ---
    # Measures texture density in 5x5 neighborhoods
    kernel_size = 5
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = sqr_mean - mean**2
    variance = np.maximum(variance, 0)  # Ensure non-negative
    
    # Normalize variance to [0, 1]
    variance_norm = variance / (variance.max() + 1e-8)
    
    # --- Combine Components ---
    # Weighted combination: edges + texture + variance
    importance = (0.4 * sobel_norm) + (0.35 * laplacian_norm) + (0.25 * variance_norm)
    
    # Final normalization to [0, 1]
    importance = importance / (importance.max() + 1e-8)
    
    return importance.astype(np.float32)


def get_pixel_order(
    importance_map: np.ndarray, 
    image_shape: Tuple[int, ...],
    min_importance: float = 0.0
) -> np.ndarray:
    """
    Get deterministic pixel ordering based on importance map.
    
    Pixels are sorted by:
      1. Importance (descending) - embed in high-importance first
      2. Row index (ascending) - tie-breaker for stability
      3. Column index (ascending) - tie-breaker for stability
      4. Channel index (ascending) - R, G, B order
    
    This STABLE SORT guarantees encoder and decoder pick pixels in EXACT SAME ORDER.
    
    CRITICAL: We quantize importance to 16-bit integers before sorting to avoid
    floating-point comparison issues that could cause different orderings.
    
    Args:
        importance_map: (H, W) importance values
        image_shape: (H, W) or (H, W, C)
        min_importance: Minimum importance threshold (0-1).
            Pixels with importance below this are EXCLUDED from the result.
            This improves imperceptibility by avoiding smooth regions.
        
    Returns:
        Array of shape (N, 3) where each row is (row, col, channel)
        N = number of pixels with importance >= min_importance, times channels
    """
    if len(image_shape) == 3:
        height, width, channels = image_shape
    else:
        height, width = image_shape
        channels = 1
    
    # CRITICAL: Quantize importance to 16-bit integers for EXACT reproducibility
    # This eliminates floating-point comparison issues
    # We use 65535 levels which is more than enough for sorting
    importance_quantized = (importance_map * 65535).astype(np.int32)
    min_importance_quantized = int(min_importance * 65535)
    
    # Create index arrays for all pixel-channel combinations
    rows, cols, chans = np.meshgrid(
        np.arange(height),
        np.arange(width),
        np.arange(channels),
        indexing='ij'
    )
    
    rows = rows.flatten()
    cols = cols.flatten()
    chans = chans.flatten()
    
    # Get quantized importance value for each pixel (same for all channels)
    importance_values = importance_quantized[rows, cols]
    
    # IMPERCEPTIBILITY IMPROVEMENT: Filter out low-importance pixels
    # This prevents embedding in smooth regions where LSB changes are visible
    if min_importance > 0:
        mask = importance_values >= min_importance_quantized
        rows = rows[mask]
        cols = cols[mask]
        chans = chans[mask]
        importance_values = importance_values[mask]
    
    # Create stable sort using lexsort
    # Sort by: (-importance, row, col, channel)
    # Negative importance = descending order (high importance first)
    sort_keys = np.lexsort((chans, cols, rows, -importance_values))
    
    # Build ordered pixel indices
    pixel_order = np.column_stack((rows[sort_keys], cols[sort_keys], chans[sort_keys]))
    
    return pixel_order


def visualize_importance_map(importance_map: np.ndarray) -> np.ndarray:
    """
    Convert importance map to a visualizable heatmap.
    
    Args:
        importance_map: (H, W) normalized importance values
        
    Returns:
        RGB heatmap image (H, W, 3) as uint8
    """
    # Scale to 0-255
    scaled = (importance_map * 255).astype(np.uint8)
    
    # Apply colormap (VIRIDIS: dark = low importance, bright = high)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap
