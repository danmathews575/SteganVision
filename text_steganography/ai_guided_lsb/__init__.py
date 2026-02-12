"""
AI-Guided Adaptive LSB Steganography Package

A production-ready, deterministic text-to-image steganography system
that uses AI-inspired importance maps for embedding location optimization.
"""

from .encoder import encode, encode_from_file
from .decoder import decode, decode_to_file
from .importance_model import compute_importance_map, get_pixel_order, visualize_importance_map
from .utils import text_to_bits, bits_to_text, calculate_capacity, validate_capacity

__all__ = [
    # Encoder
    'encode',
    'encode_from_file',
    # Decoder
    'decode',
    'decode_to_file',
    # Importance Model
    'compute_importance_map',
    'get_pixel_order', 
    'visualize_importance_map',
    # Utils
    'text_to_bits',
    'bits_to_text',
    'calculate_capacity',
    'validate_capacity',
]
