"""
AI-Guided Adaptive LSB Steganography - Utilities Module

This module provides text preprocessing utilities for deterministic
text-to-binary and binary-to-text conversion.

ENGINEERING JUSTIFICATION:
- 32-bit length header ensures exact message boundary detection
- UTF-8 encoding ensures Unicode safety (supports all languages)
- No randomness = deterministic decode = 100% exact recovery
"""

import numpy as np
from typing import Tuple


# Fixed length header size (32 bits = supports up to 4GB text)
LENGTH_HEADER_BITS = 32


def text_to_bits(text: str) -> np.ndarray:
    """
    Convert text to binary representation with length header.
    
    Format: [32-bit length header] + [UTF-8 encoded text bits]
    
    Args:
        text: Input text string (supports Unicode)
        
    Returns:
        numpy array of bits (0s and 1s) as uint8
        
    Example:
        >>> bits = text_to_bits("Hi")
        >>> len(bits)  # 32 (header) + 16 (2 ASCII chars * 8 bits)
        48
    """
    # Encode text to UTF-8 bytes
    text_bytes = text.encode('utf-8')
    text_length = len(text_bytes)
    
    # Create length header (32-bit, big-endian)
    length_bits = format(text_length, f'0{LENGTH_HEADER_BITS}b')
    
    # Convert text bytes to bits
    text_bits = ''.join(format(byte, '08b') for byte in text_bytes)
    
    # Combine: header + text
    all_bits = length_bits + text_bits
    
    # Convert to numpy array
    return np.array([int(b) for b in all_bits], dtype=np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    """
    Convert binary representation back to text.
    
    Reads length header to determine exact message boundary.
    
    Args:
        bits: numpy array of extracted bits (0s and 1s)
        
    Returns:
        Decoded text string
        
    Raises:
        ValueError: If bits array is too short or corrupted
    """
    if len(bits) < LENGTH_HEADER_BITS:
        raise ValueError(f"Insufficient bits: need at least {LENGTH_HEADER_BITS} for header")
    
    # Extract length header
    header_bits = bits[:LENGTH_HEADER_BITS]
    length_str = ''.join(str(int(b)) for b in header_bits)
    text_length = int(length_str, 2)
    
    # Validate we have enough bits for the text
    required_bits = LENGTH_HEADER_BITS + (text_length * 8)
    if len(bits) < required_bits:
        raise ValueError(f"Insufficient bits: need {required_bits}, got {len(bits)}")
    
    # Extract text bits
    text_bits = bits[LENGTH_HEADER_BITS:required_bits]
    
    # Convert bits to bytes
    text_bytes = bytearray()
    for i in range(0, len(text_bits), 8):
        byte_bits = text_bits[i:i+8]
        byte_str = ''.join(str(int(b)) for b in byte_bits)
        text_bytes.append(int(byte_str, 2))
    
    # Decode UTF-8
    return bytes(text_bytes).decode('utf-8')


def calculate_capacity(image_shape: Tuple[int, ...], bits_per_pixel: int = 1) -> int:
    """
    Calculate maximum text capacity for an image.
    
    Args:
        image_shape: (height, width) or (height, width, channels)
        bits_per_pixel: LSBs used per channel (1 or 2)
        
    Returns:
        Maximum number of characters that can be embedded
    """
    if len(image_shape) == 3:
        height, width, channels = image_shape
    else:
        height, width = image_shape
        channels = 1
        
    total_bits = height * width * channels * bits_per_pixel
    usable_bits = total_bits - LENGTH_HEADER_BITS
    max_bytes = usable_bits // 8
    return max_bytes


def validate_capacity(text: str, image_shape: Tuple[int, ...], bits_per_pixel: int = 1) -> Tuple[bool, str]:
    """
    Check if text fits in the image.
    
    Args:
        text: Text to embed
        image_shape: (height, width) or (height, width, channels)
        bits_per_pixel: LSBs used per channel
        
    Returns:
        (can_fit, message) tuple
    """
    text_bytes = text.encode('utf-8')
    required_bits = LENGTH_HEADER_BITS + (len(text_bytes) * 8)
    
    if len(image_shape) == 3:
        height, width, channels = image_shape
    else:
        height, width = image_shape
        channels = 1
        
    available_bits = height * width * channels * bits_per_pixel
    
    if required_bits <= available_bits:
        return True, f"Text fits: {required_bits} bits needed, {available_bits} available"
    else:
        max_chars = calculate_capacity(image_shape, bits_per_pixel)
        return False, f"Text too large: needs {required_bits} bits, only {available_bits} available. Max ~{max_chars} chars."
