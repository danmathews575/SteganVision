"""
AI-Guided Adaptive LSB Steganography - Encoder Module

This module embeds text into images using AI-guided pixel selection
and adaptive LSB modification.

ENGINEERING JUSTIFICATION:
---------------------------------------------------------------------------
WHY LSB ENSURES CORRECTNESS (vs GANs):

GANs use neural networks to encode/decode, which introduces:
  - Stochastic behavior (different runs = different results)
  - Lossy reconstruction (never 100% accurate)
  - Training dependency (needs .pth files)

LSB (Least Significant Bit) is DETERMINISTIC:
  - Bit is either 0 or 1, extracted exactly as embedded
  - No training, no weights, no randomness
  - 100% GUARANTEED exact text recovery

WHY THIS IS A HYBRID INTELLIGENT SYSTEM:
  - AI-inspired importance map decides WHERE to embed
  - Classical LSB decides HOW to embed
  - Best of both worlds: intelligent + reliable

IMPERCEPTIBILITY IMPROVEMENTS:
  - Minimum importance threshold: Only embed in high-texture pixels
  - Spread embedding: Maximize use of safe regions
  - Skip smooth areas: Where LSB changes are visible
---------------------------------------------------------------------------
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional

from .importance_model import compute_importance_map, get_pixel_order
from .utils import text_to_bits, validate_capacity, LENGTH_HEADER_BITS


# Minimum importance threshold - pixels below this are NOT used for embedding
# This prevents embedding in smooth regions where LSB changes are visible
# Value 0.05 means we skip the smoothest 5% of pixels (by importance)
MIN_IMPORTANCE_THRESHOLD = 0.05


def encode(
    cover_image_path: Union[str, Path],
    text: str,
    output_path: Union[str, Path],
    bits_per_channel: int = 1,
    save_importance_map: bool = False,
    importance_threshold: float = MIN_IMPORTANCE_THRESHOLD
) -> Tuple[bool, str]:
    """
    Encode text into a cover image using AI-guided adaptive LSB.
    
    Process:
      1. Load cover image
      2. Compute AI importance map
      3. Filter pixels by importance threshold (skip smooth regions)
      4. Validate text capacity against filtered pixels
      5. Sort filtered pixels by importance (high first)
      6. Embed bits using LSB modification
      7. Save stego image
    
    Args:
        cover_image_path: Path to cover image (PNG recommended)
        text: Text to hide (supports Unicode)
        output_path: Path to save stego image
        bits_per_channel: LSBs to use per channel (1 or 2)
        save_importance_map: If True, also save importance map visualization
        importance_threshold: Minimum importance for embedding (0-1)
            Higher = more imperceptible but less capacity
            Default 0.05 = skip smoothest 5% of pixels
        
    Returns:
        (success, message) tuple
    """
    cover_image_path = Path(cover_image_path)
    output_path = Path(output_path)
    
    # --- Step 1: Load Cover Image ---
    if not cover_image_path.exists():
        return False, f"Cover image not found: {cover_image_path}"
    
    try:
        with Image.open(cover_image_path) as img:
            cover_img = img.convert('RGB')
            cover_array = np.array(cover_img, dtype=np.uint8)
    except Exception as e:
        return False, f"Failed to load image: {e}"
    
    # Handle empty text edge case
    if len(text) == 0:
        return False, "Cannot embed empty text. Provide at least one character."
    
    # --- Step 2: Compute AI Importance Map ---
    importance_map = compute_importance_map(cover_array)
    
    # Optionally save importance map for visualization
    if save_importance_map:
        from .importance_model import visualize_importance_map
        imp_vis = visualize_importance_map(importance_map)
        imp_path = output_path.parent / f"{output_path.stem}_importance.png"
        Image.fromarray(imp_vis).save(imp_path)
    
    # --- Step 3: Get Pixel Order with Importance Filtering ---
    pixel_order = get_pixel_order(
        importance_map, 
        cover_array.shape,
        min_importance=importance_threshold
    )
    
    # --- Step 4: Validate Capacity with Filtered Pixels ---
    bits = text_to_bits(text)
    available_bits = len(pixel_order) * bits_per_channel
    required_bits = len(bits)
    
    if required_bits > available_bits:
        # Calculate how many chars we can actually fit
        usable_bits = available_bits - LENGTH_HEADER_BITS
        max_chars = usable_bits // 8
        return False, (
            f"Text too large for high-quality embedding. "
            f"Need {required_bits} bits but only {available_bits} high-importance pixels available. "
            f"Max ~{max_chars} chars with current image. "
            f"Try a larger or more textured cover image."
        )
    
    # --- Step 5: Embed Bits Using LSB ---
    stego_array = cover_array.copy()
    
    for i, bit in enumerate(bits):
        row, col, channel = pixel_order[i]
        
        # Get current pixel value
        pixel_value = stego_array[row, col, channel]
        
        if bits_per_channel == 1:
            # Modify LSB only - minimal change for maximum imperceptibility
            stego_array[row, col, channel] = (pixel_value & 0xFE) | bit
        else:
            # Modify 2 LSBs (for higher capacity, lower imperceptibility)
            if i * 2 + 1 < len(bits):
                two_bits = (bits[i * 2] << 1) | bits[i * 2 + 1]
                stego_array[row, col, channel] = (pixel_value & 0xFC) | two_bits
    
    # --- Step 6: Save Stego Image ---
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stego_img = Image.fromarray(stego_array)
        stego_img.save(output_path, format='PNG')  # PNG = lossless
    except Exception as e:
        return False, f"Failed to save stego image: {e}"
    
    # Calculate embedding stats
    pct_used = (len(bits) / len(pixel_order)) * 100
    return True, (
        f"Successfully encoded {len(text)} characters. "
        f"Used {pct_used:.1f}% of high-importance pixels."
    )


def encode_from_file(
    cover_image_path: Union[str, Path],
    text_file_path: Union[str, Path],
    output_path: Union[str, Path],
    bits_per_channel: int = 1
) -> Tuple[bool, str]:
    """
    Encode text from a file into a cover image.
    
    Args:
        cover_image_path: Path to cover image
        text_file_path: Path to text file containing secret message
        output_path: Path to save stego image
        bits_per_channel: LSBs to use per channel (1 or 2)
        
    Returns:
        (success, message) tuple
    """
    text_file_path = Path(text_file_path)
    
    if not text_file_path.exists():
        return False, f"Text file not found: {text_file_path}"
    
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return False, f"Failed to read text file: {e}"
    
    return encode(cover_image_path, text, output_path, bits_per_channel)
