"""
AI-Guided LSB Audio Steganography - Encoder

CORE PRINCIPLE:
===============
AI does NOT encode bits.
AI ONLY decides WHERE embedding is safest.

This encoder:
1. Loads cover audio
2. Computes psychoacoustic importance map (AI guidance)
3. Sorts samples by importance (safest regions first)
4. Embeds payload bits into LSB of high-importance samples
5. Saves stego audio with identical format to cover

The result is IMPERCEPTIBLE embedding because:
- LSB changes occur in loud, complex regions (masked by audio)
- Silent/quiet regions are avoided (no artifacts)
- Deterministic process ensures exact recovery

Author: AI-Guided Steganography System
"""

import os
import numpy as np
import soundfile as sf

from .importance_model import compute_importance_map, get_embedding_order, analyze_importance_distribution
from .utils import (
    create_payload,
    validate_capacity,
    flatten_audio,
    unflatten_audio,
    calculate_snr,
    calculate_mse,
    HEADER_SIZE,
)


def encode(
    cover_path: str,
    secret_path: str,
    output_path: str,
    bits_per_sample: int = 1,
    use_compression: bool = True
) -> dict:
    """
    Encode secret into cover audio using AI-guided LSB steganography.
    
    The AI (importance model) determines WHERE to embed.
    The LSB algorithm performs the actual embedding.
    Compression reduces payload size for larger secrets.
    
    Args:
        cover_path: Path to cover audio file (WAV)
        secret_path: Path to secret file (any binary file, typically WAV)
        output_path: Path to save stego audio (WAV)
        bits_per_sample: Number of LSB bits to use (default 1, max 2)
        use_compression: Whether to compress secret data (default True)
        
    Returns:
        Dictionary with encoding metadata:
        - status: Success/failure message
        - snr: Signal-to-Noise ratio (dB)
        - mse: Mean squared error
        - capacity_utilization: Percentage of capacity used
        - compression_ratio: Compression ratio (if compression enabled)
        
    Raises:
        ValueError: If secret exceeds cover capacity
        FileNotFoundError: If input files don't exist
    """
    # Validate inputs
    if not os.path.exists(cover_path):
        raise FileNotFoundError(f"Cover file not found: {cover_path}")
    if not os.path.exists(secret_path):
        raise FileNotFoundError(f"Secret file not found: {secret_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load cover audio as int16
    # =========================================================================
    cover_audio, sample_rate = sf.read(cover_path, dtype='int16')
    original_shape = cover_audio.shape
    
    # CRITICAL DETAIL #3: Flatten for consistent multi-channel handling
    cover_flat, _ = flatten_audio(cover_audio)
    n_samples = len(cover_flat)
    
    # =========================================================================
    # STEP 2: Load secret and create payload (with compression)
    # =========================================================================
    with open(secret_path, 'rb') as f:
        secret_data = f.read()
    
    original_secret_size = len(secret_data)
    
    # Create payload with optional compression
    payload_bits = create_payload(secret_data, use_compression=use_compression)
    payload_length = len(payload_bits)
    payload_bytes = (payload_length + 7) // 8  # Round up to bytes
    
    # Validate capacity AFTER compression
    capacity_info = validate_capacity(payload_bytes, n_samples, bits_per_sample)
    
    if not capacity_info['fits']:
        compression_note = " (with compression)" if use_compression else " (without compression)"
        raise ValueError(
            f"❌ CAPACITY EXCEEDED{compression_note}\n\n"
            f"Original secret size: {original_secret_size:,} bytes\n"
            f"Payload size{compression_note}: {payload_bytes:,} bytes\n"
            f"Cover capacity: {capacity_info['capacity_bytes']:,} bytes\n\n"
            f"SOLUTIONS:\n"
            f"1. Use a longer cover audio file\n"
            f"2. Use a smaller secret file\n"
            f"3. Increase bits_per_sample (currently {bits_per_sample})\n"
            f"4. {'Compression is already enabled' if use_compression else 'Enable compression (use_compression=True)'}"
        )
    
    # =========================================================================
    # STEP 3: Compute AI importance map
    # =========================================================================
    # Convert to float for importance computation (preserves original audio)
    cover_float = cover_flat.astype(np.float64)
    # CRITICAL: Pass bits_per_sample to mask out LSBs for identical results
    importance_map = compute_importance_map(cover_float, sample_rate, bits_per_sample=bits_per_sample)
    
    # Get embedding order (sorted by importance, stable tie-break)
    embedding_order = get_embedding_order(importance_map)
    
    # =========================================================================
    # STEP 4: Embed payload bits using LSB
    # =========================================================================
    stego_flat = cover_flat.copy()
    
    # Masks for LSB manipulation
    if bits_per_sample == 1:
        clear_mask = np.int16(0xFFFE)  # Clear LSB
    elif bits_per_sample == 2:
        clear_mask = np.int16(0xFFFC)  # Clear 2 LSBs
    else:
        raise ValueError(f"bits_per_sample must be 1 or 2, got {bits_per_sample}")
    
    # Embed bits in importance order
    bit_idx = 0
    for sample_idx in embedding_order:
        if bit_idx >= payload_length:
            break
        
        # Get bits to embed
        bits_to_embed = 0
        for b in range(bits_per_sample):
            if bit_idx + b < payload_length:
                bits_to_embed |= (int(payload_bits[bit_idx + b]) << (bits_per_sample - 1 - b))
        
        # Clear and set LSB(s)
        stego_flat[sample_idx] = (stego_flat[sample_idx] & clear_mask) | np.int16(bits_to_embed)
        bit_idx += bits_per_sample
    
    # =========================================================================
    # STEP 5: Reshape and save stego audio
    # =========================================================================
    stego_audio = unflatten_audio(stego_flat, original_shape)
    sf.write(output_path, stego_audio, sample_rate, subtype='PCM_16')
    
    # =========================================================================
    # STEP 6: Calculate quality metrics
    # =========================================================================
    snr = calculate_snr(cover_flat, stego_flat)
    mse = calculate_mse(cover_flat, stego_flat)
    importance_stats = analyze_importance_distribution(importance_map)
    
    return {
        'status': '✅ Encoding successful (AI-Guided LSB with Compression)' if use_compression else '✅ Encoding successful (AI-Guided LSB)',
        'cover_samples': n_samples,
        'secret_size_bytes': original_secret_size,
        'payload_bytes': payload_bytes,
        'payload_bits': payload_length,
        'bits_per_sample': bits_per_sample,
        'capacity_utilization': capacity_info['utilization_percent'],
        'compression_enabled': use_compression,
        'compression_ratio': original_secret_size / payload_bytes if use_compression and payload_bytes > 0 else 1.0,
        'snr_db': snr,
        'mse': mse,
        'importance_stats': importance_stats,
        'output_path': output_path,
    }


def encode_audio_to_audio(
    cover_path: str,
    secret_audio_path: str,
    output_path: str,
    bits_per_sample: int = 1
) -> dict:
    """
    Convenience wrapper for audio-in-audio steganography.
    
    Simply embeds the raw audio file as binary data.
    
    Args:
        cover_path: Path to cover audio (WAV)
        secret_audio_path: Path to secret audio (WAV)
        output_path: Path to save stego audio (WAV)
        bits_per_sample: LSB bits per sample
        
    Returns:
        Encoding metadata dictionary
    """
    return encode(cover_path, secret_audio_path, output_path, bits_per_sample)
