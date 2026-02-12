"""
AI-Guided LSB Audio Steganography - Decoder

CORE PRINCIPLE:
===============
The decoder MUST recompute the IDENTICAL importance map.

Since LSB changes are minimal (1-2 bits per sample), the importance map
computed from the stego audio is nearly identical to the original.

In fact, for EXACT recovery:
- Importance is computed from the UPPER bits (not LSB)
- LSB changes do not affect the importance calculation
- Therefore: encoder importance map == decoder importance map

This guarantees 100% exact recovery.

Author: AI-Guided Steganography System
"""

import os
import numpy as np
import soundfile as sf

from .importance_model import compute_importance_map, get_embedding_order
from .utils import (
    extract_payload,
    flatten_audio,
    HEADER_SIZE,
)


def decode(
    stego_path: str,
    output_path: str,
    bits_per_sample: int = 1
) -> dict:
    """
    Decode secret from stego audio using AI-guided LSB steganography.
    
    The decoder:
    1. Loads stego audio
    2. Recomputes the IDENTICAL importance map
    3. Extracts bits in the same order as embedding
    4. Recovers the exact secret data
    
    Args:
        stego_path: Path to stego audio file (WAV)
        output_path: Path to save recovered secret
        bits_per_sample: Number of LSB bits used (must match encoder)
        
    Returns:
        Dictionary with decoding metadata:
        - status: Success/failure message
        - secret_size: Size of recovered secret in bytes
        - checksum_valid: Whether CRC32 matched
        
    Raises:
        ValueError: If magic bytes mismatch or checksum fails
        FileNotFoundError: If stego file doesn't exist
    """
    # Validate input
    if not os.path.exists(stego_path):
        raise FileNotFoundError(f"Stego file not found: {stego_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load stego audio as int16
    # =========================================================================
    stego_audio, sample_rate = sf.read(stego_path, dtype='int16')
    
    # CRITICAL DETAIL #3: Flatten identically to encoder
    stego_flat, _ = flatten_audio(stego_audio)
    n_samples = len(stego_flat)
    
    # =========================================================================
    # STEP 2: Recompute IDENTICAL importance map
    # =========================================================================
    # IMPORTANT: We compute importance from the stego audio.
    # Since we use LSB masking, the importance map is identical.
    # The psychoacoustic features (energy, flatness, ZCR) depend on
    # the upper bits which are unchanged.
    
    stego_float = stego_flat.astype(np.float64)
    # CRITICAL: Pass bits_per_sample to mask out LSBs for identical results
    importance_map = compute_importance_map(stego_float, sample_rate, bits_per_sample=bits_per_sample)
    
    # Get extraction order (MUST match encoder's embedding order)
    # CRITICAL DETAIL #2: Stable sorting ensures identical order
    extraction_order = get_embedding_order(importance_map)
    
    # =========================================================================
    # STEP 3: Extract bits from LSB in importance order
    # =========================================================================
    # First, extract enough bits to read the header
    header_bits_needed = HEADER_SIZE * 8
    samples_for_header = (header_bits_needed + bits_per_sample - 1) // bits_per_sample
    
    # Extract header bits
    header_bits = []
    for i in range(samples_for_header):
        sample_idx = extraction_order[i]
        sample_value = stego_flat[sample_idx]
        
        for b in range(bits_per_sample):
            if len(header_bits) < header_bits_needed:
                bit = (sample_value >> (bits_per_sample - 1 - b)) & 1
                header_bits.append(bit)
    
    header_bits = np.array(header_bits[:header_bits_needed], dtype=np.uint8)
    
    # Parse header to get payload length
    header_bytes = np.packbits(header_bits).tobytes()
    
    # Validate magic
    magic = header_bytes[:4]
    if magic != b'AISG':
        raise ValueError(
            f"❌ Invalid magic bytes. Expected 'AISG', got {magic}. "
            "This may not be an AI-guided stego audio or wrong bits_per_sample."
        )
    
    import struct
    flags = header_bytes[4]
    is_compressed = bool(flags & 0x01)
    payload_length = struct.unpack('>I', header_bytes[5:9])[0]
    expected_checksum = struct.unpack('>I', header_bytes[9:13])[0]
    
    # Calculate total bits needed
    total_bits_needed = (HEADER_SIZE + payload_length) * 8
    samples_needed = (total_bits_needed + bits_per_sample - 1) // bits_per_sample
    
    if samples_needed > n_samples:
        raise ValueError(
            f"Insufficient samples. Need {samples_needed}, have {n_samples}. "
            "Stego audio may be corrupted or wrong bits_per_sample."
        )
    
    # =========================================================================
    # STEP 4: Extract all payload bits
    # =========================================================================
    all_bits = []
    for i in range(samples_needed):
        sample_idx = extraction_order[i]
        sample_value = stego_flat[sample_idx]
        
        for b in range(bits_per_sample):
            if len(all_bits) < total_bits_needed:
                bit = (sample_value >> (bits_per_sample - 1 - b)) & 1
                all_bits.append(bit)
    
    all_bits = np.array(all_bits[:total_bits_needed], dtype=np.uint8)
    
    # =========================================================================
    # STEP 5: Extract and verify payload
    # =========================================================================
    secret_data, metadata = extract_payload(all_bits)
    
    # =========================================================================
    # STEP 6: Save recovered secret
    # =========================================================================
    with open(output_path, 'wb') as f:
        f.write(secret_data)
    
    return {
        'status': '✅ Decoding successful (AI-Guided LSB)',
        'secret_size_bytes': len(secret_data),
        'checksum_valid': metadata['checksum_valid'],
        'checksum': metadata['checksum'],
        'compressed': metadata.get('compressed', False),
        'compression_ratio': metadata.get('compression_ratio', 1.0),
        'output_path': output_path,
    }


def decode_audio_from_audio(
    stego_path: str,
    output_path: str,
    bits_per_sample: int = 1
) -> dict:
    """
    Convenience wrapper for audio-from-audio steganography.
    
    Args:
        stego_path: Path to stego audio (WAV)
        output_path: Path to save recovered audio (WAV)
        bits_per_sample: LSB bits per sample (must match encoder)
        
    Returns:
        Decoding metadata dictionary
    """
    return decode(stego_path, output_path, bits_per_sample)
