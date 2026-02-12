"""
AI-Guided LSB Audio Steganography Module

A hybrid intelligent steganography system that uses:
- AI (psychoacoustic heuristics) to determine WHERE to embed
- LSB to perform EXACT bit embedding

This achieves both:
- Imperceptibility (AI-guided placement in masked regions)
- Exactness (100% bit-accurate recovery)
"""

from .encoder import encode, encode_audio_to_audio
from .decoder import decode, decode_audio_from_audio
from .importance_model import compute_importance_map, get_embedding_order
from .utils import calculate_snr, calculate_mse

__all__ = [
    'encode',
    'decode',
    'encode_audio_to_audio',
    'decode_audio_from_audio',
    'compute_importance_map',
    'get_embedding_order',
    'calculate_snr',
    'calculate_mse',
]
