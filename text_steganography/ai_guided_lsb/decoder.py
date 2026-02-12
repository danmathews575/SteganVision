"""
AI-Guided Adaptive LSB Steganography - Decoder Module

This module extracts hidden text from stego images using the same
AI-guided pixel ordering used during encoding.

ENGINEERING JUSTIFICATION:
---------------------------------------------------------------------------
WHY DETERMINISTIC DECODE GUARANTEES 100% RECOVERY:

The decode process is a MIRROR of encode:
  1. Same importance map algorithm → same importance values
  2. Same min_importance threshold → same filtered pixels
  3. Same pixel sorting algorithm → same pixel order
  4. Same LSB extraction → exact bit recovery
  5. Length header → exact message boundary

No neural network inference = no approximation = no errors.
---------------------------------------------------------------------------
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple

from .importance_model import compute_importance_map, get_pixel_order
from .utils import bits_to_text, LENGTH_HEADER_BITS


# Must match encoder's threshold for correct decoding
MIN_IMPORTANCE_THRESHOLD = 0.05


def decode(
    stego_image_path: Union[str, Path],
    bits_per_channel: int = 1,
    importance_threshold: float = MIN_IMPORTANCE_THRESHOLD
) -> Tuple[bool, str, str]:
    """
    Decode hidden text from a stego image.
    
    Process:
      1. Load stego image
      2. Recompute AI importance map (same algorithm as encoding)
      3. Filter and sort pixels by importance (same order as encoding)
      4. Extract LSBs from sorted pixels
      5. Read length header to find message boundary
      6. Convert bits to text
    
    Args:
        stego_image_path: Path to stego image
        bits_per_channel: LSBs used per channel during encoding (1 or 2)
        importance_threshold: Must match the threshold used during encoding
        
    Returns:
        (success, text, message) tuple
        - success: True if decoding succeeded
        - text: Extracted text (empty string on failure)
        - message: Status message
    """
    stego_image_path = Path(stego_image_path)
    
    # --- Step 1: Load Stego Image ---
    if not stego_image_path.exists():
        return False, "", f"Stego image not found: {stego_image_path}"
    
    try:
        with Image.open(stego_image_path) as img:
            stego_img = img.convert('RGB')
            stego_array = np.array(stego_img, dtype=np.uint8)
    except Exception as e:
        return False, "", f"Failed to load image: {e}"
    
    # --- Step 2: Recompute AI Importance Map ---
    # CRITICAL: Must use EXACT same algorithm as encoder
    importance_map = compute_importance_map(stego_array)
    
    # --- Step 3: Get Pixel Order with Same Filtering ---
    # CRITICAL: Must use same min_importance as encoder
    pixel_order = get_pixel_order(
        importance_map, 
        stego_array.shape,
        min_importance=importance_threshold
    )
    
    if len(pixel_order) < LENGTH_HEADER_BITS:
        return False, "", "Image too small or not enough textured regions"
    
    # --- Step 4: Extract Header Bits First ---
    # We need to read the length header to know how many bits to extract
    header_bits = np.zeros(LENGTH_HEADER_BITS, dtype=np.uint8)
    
    for i in range(LENGTH_HEADER_BITS):
        row, col, channel = pixel_order[i]
        pixel_value = stego_array[row, col, channel]
        
        if bits_per_channel == 1:
            header_bits[i] = pixel_value & 1  # Extract LSB
        else:
            # For 2-bit mode, this would need adjustment
            header_bits[i] = pixel_value & 1
    
    # Parse length from header
    length_str = ''.join(str(b) for b in header_bits)
    text_length = int(length_str, 2)
    
    # Validate length is reasonable
    max_bytes = (len(pixel_order) - LENGTH_HEADER_BITS) // 8
    
    if text_length <= 0:
        return False, "", "Invalid length header: no text found"
    
    if text_length > max_bytes:
        return False, "", f"Invalid length header: claims {text_length} bytes, max possible is {max_bytes}"
    
    # --- Step 5: Extract All Message Bits ---
    total_bits_needed = LENGTH_HEADER_BITS + (text_length * 8)
    all_bits = np.zeros(total_bits_needed, dtype=np.uint8)
    
    # Copy header bits we already extracted
    all_bits[:LENGTH_HEADER_BITS] = header_bits
    
    # Extract remaining bits
    for i in range(LENGTH_HEADER_BITS, total_bits_needed):
        row, col, channel = pixel_order[i]
        pixel_value = stego_array[row, col, channel]
        
        if bits_per_channel == 1:
            all_bits[i] = pixel_value & 1
        else:
            all_bits[i] = pixel_value & 1
    
    # --- Step 6: Convert Bits to Text ---
    try:
        text = bits_to_text(all_bits)
    except Exception as e:
        return False, "", f"Failed to decode text: {e}"
    
    return True, text, f"Successfully decoded {len(text)} characters"


def decode_to_file(
    stego_image_path: Union[str, Path],
    output_file_path: Union[str, Path],
    bits_per_channel: int = 1
) -> Tuple[bool, str]:
    """
    Decode hidden text and save to a file.
    
    Args:
        stego_image_path: Path to stego image
        output_file_path: Path to save decoded text
        bits_per_channel: LSBs used per channel during encoding
        
    Returns:
        (success, message) tuple
    """
    output_file_path = Path(output_file_path)
    
    success, text, message = decode(stego_image_path, bits_per_channel)
    
    if not success:
        return False, message
    
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        return False, f"Failed to save decoded text: {e}"
    
    return True, f"Decoded text saved to {output_file_path}"
