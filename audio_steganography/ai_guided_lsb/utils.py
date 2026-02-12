"""
AI-Guided LSB Audio Steganography - Utilities

Binary conversion, header management, checksum verification, and compression.
Guarantees EXACT byte-accurate recovery.
"""

import numpy as np
import struct
import zlib
import lzma


# =============================================================================
# CONSTANTS
# =============================================================================

# Magic bytes to identify AI-guided stego audio (4 bytes)
MAGIC = b'AISG'

# Header structure (total 13 bytes):
# - 4 bytes: Magic (AISG)
# - 1 byte:  Flags (bit 0: compression enabled)
# - 4 bytes: Payload length in bytes (big-endian uint32)
# - 4 bytes: CRC32 checksum of payload
HEADER_SIZE = 13

# Compression settings
COMPRESSION_PRESET = 6  # LZMA preset (0-9, higher = better compression, slower)


# =============================================================================
# BINARY CONVERSION
# =============================================================================

def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to bit array.
    
    Each byte is converted to 8 bits (MSB first).
    
    Args:
        data: Input bytes
        
    Returns:
        1D numpy array of bits (0 or 1), dtype=uint8
    """
    if len(data) == 0:
        return np.array([], dtype=np.uint8)
    
    # Convert bytes to numpy array
    byte_array = np.frombuffer(data, dtype=np.uint8)
    
    # Unpack each byte to 8 bits (MSB first)
    bits = np.unpackbits(byte_array)
    
    return bits


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array back to bytes.
    
    Bits are packed 8 at a time (MSB first).
    If bit count is not multiple of 8, pads with zeros.
    
    Args:
        bits: 1D array of bits (0 or 1)
        
    Returns:
        Bytes object
    """
    if len(bits) == 0:
        return b''
    
    # Ensure bits is uint8
    bits = np.asarray(bits, dtype=np.uint8)
    
    # Pad to multiple of 8 if needed
    remainder = len(bits) % 8
    if remainder != 0:
        padding = np.zeros(8 - remainder, dtype=np.uint8)
        bits = np.concatenate([bits, padding])
    
    # Pack bits to bytes
    byte_array = np.packbits(bits)
    
    return byte_array.tobytes()


# =============================================================================
# COMPRESSION
# =============================================================================

def compress_data(data: bytes) -> bytes:
    """
    Compress data using LZMA algorithm.
    
    Args:
        data: Raw bytes to compress
        
    Returns:
        Compressed bytes
    """
    return lzma.compress(data, preset=COMPRESSION_PRESET)


def decompress_data(compressed_data: bytes) -> bytes:
    """
    Decompress LZMA-compressed data.
    
    Args:
        compressed_data: LZMA-compressed bytes
        
    Returns:
        Decompressed bytes
    """
    return lzma.decompress(compressed_data)


# =============================================================================
# PAYLOAD CREATION AND EXTRACTION
# =============================================================================

def create_payload(secret_data: bytes, use_compression: bool = True) -> np.ndarray:
    """
    Create payload with header for embedding.
    
    Payload structure:
    [MAGIC (4B)] [FLAGS (1B)] [LENGTH (4B)] [CRC32 (4B)] [SECRET DATA (N bytes)]
    
    Args:
        secret_data: Raw secret bytes to embed
        use_compression: Whether to compress secret data (default True)
        
    Returns:
        1D array of bits (header + secret)
    """
    # Compress if enabled
    if use_compression:
        processed_data = compress_data(secret_data)
        flags = 0x01  # Bit 0 set = compression enabled
    else:
        processed_data = secret_data
        flags = 0x00
    
    # Calculate CRC32 checksum of PROCESSED data (compressed or raw)
    checksum = zlib.crc32(processed_data) & 0xFFFFFFFF
    
    # Build header
    header = bytearray()
    header.extend(MAGIC)                                      # 4 bytes: Magic
    header.append(flags)                                      # 1 byte:  Flags
    header.extend(struct.pack('>I', len(processed_data)))    # 4 bytes: Length (big-endian)
    header.extend(struct.pack('>I', checksum))                # 4 bytes: CRC32
    
    # Combine header + processed data
    full_payload = bytes(header) + processed_data
    
    # Convert to bits
    payload_bits = bytes_to_bits(full_payload)
    
    return payload_bits


def extract_payload(bits: np.ndarray) -> tuple:
    """
    Extract payload from bit array.
    
    Reads header, validates magic and checksum, decompresses if needed, returns secret data.
    
    Args:
        bits: 1D array of bits extracted from stego audio
        
    Returns:
        Tuple of (secret_data: bytes, metadata: dict)
        
    Raises:
        ValueError: If magic mismatch or checksum failure
    """
    # Convert header bits to bytes
    header_bits = bits[:HEADER_SIZE * 8]
    header_bytes = bits_to_bytes(header_bits)
    
    # Parse header
    magic = header_bytes[:4]
    if magic != MAGIC:
        raise ValueError(
            f"Invalid magic bytes. Expected {MAGIC}, got {magic}. "
            "This file may not be an AI-guided stego audio."
        )
    
    flags = header_bytes[4]
    is_compressed = bool(flags & 0x01)
    payload_length = struct.unpack('>I', header_bytes[5:9])[0]
    expected_checksum = struct.unpack('>I', header_bytes[9:13])[0]
    
    # Extract payload data bits
    total_bits_needed = (HEADER_SIZE + payload_length) * 8
    if len(bits) < total_bits_needed:
        raise ValueError(
            f"Insufficient bits. Need {total_bits_needed}, have {len(bits)}. "
            "Stego audio may be corrupted."
        )
    
    payload_bits = bits[HEADER_SIZE * 8 : total_bits_needed]
    payload_data = bits_to_bytes(payload_bits)[:payload_length]  # Trim any padding
    
    # Verify checksum of payload (compressed or raw)
    actual_checksum = zlib.crc32(payload_data) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Checksum mismatch! Expected {expected_checksum:08X}, got {actual_checksum:08X}. "
            "Payload data is corrupted."
        )
    
    # Decompress if needed
    if is_compressed:
        try:
            secret_data = decompress_data(payload_data)
        except Exception as e:
            raise ValueError(f"Decompression failed: {e}")
    else:
        secret_data = payload_data
    
    metadata = {
        'payload_length': payload_length,
        'checksum': f'{expected_checksum:08X}',
        'checksum_valid': True,
        'compressed': is_compressed,
        'original_size': len(secret_data),
        'compression_ratio': len(secret_data) / payload_length if is_compressed and payload_length > 0 else 1.0,
    }
    
    return secret_data, metadata


# =============================================================================
# CAPACITY CALCULATION
# =============================================================================

def calculate_capacity(n_samples: int, bits_per_sample: int = 1) -> int:
    """
    Calculate maximum payload capacity in bytes.
    
    Args:
        n_samples: Number of audio samples available
        bits_per_sample: LSB bits used per sample (default 1)
        
    Returns:
        Maximum payload size in bytes (excluding header)
    """
    total_bits = n_samples * bits_per_sample
    total_bytes = total_bits // 8
    max_payload = total_bytes - HEADER_SIZE
    return max(0, max_payload)


def validate_capacity(secret_size: int, n_samples: int, bits_per_sample: int = 1) -> dict:
    """
    Validate that secret fits in cover audio.
    
    Args:
        secret_size: Size of secret in bytes
        n_samples: Number of cover audio samples
        bits_per_sample: LSB bits used per sample
        
    Returns:
        Dictionary with validation results
    """
    capacity = calculate_capacity(n_samples, bits_per_sample)
    required = secret_size
    
    fits = required <= capacity
    utilization = (required / capacity * 100) if capacity > 0 else float('inf')
    
    return {
        'fits': fits,
        'capacity_bytes': capacity,
        'required_bytes': required,
        'utilization_percent': utilization,
        'remaining_bytes': capacity - required if fits else 0,
    }


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def flatten_audio(audio: np.ndarray) -> tuple:
    """
    CRITICAL DETAIL #3: Multi-Channel Audio Handling
    =================================================
    For consistent encoding/decoding, we flatten all channels
    into a single stream, embed, then reshape back.
    
    Both encoder and decoder MUST use this identical approach.
    
    Args:
        audio: Audio array (can be mono 1D or stereo 2D)
        
    Returns:
        Tuple of (flattened_audio, original_shape)
    """
    original_shape = audio.shape
    flattened = audio.flatten()
    return flattened, original_shape


def unflatten_audio(audio: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Restore audio to original shape after embedding.
    
    Args:
        audio: Flattened audio array
        original_shape: Original shape before flattening
        
    Returns:
        Audio array reshaped to original dimensions
    """
    return audio.reshape(original_shape)


# =============================================================================
# QUALITY METRICS
# =============================================================================

def calculate_snr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    SNR = 10 * log10(signal_power / noise_power)
    
    Args:
        original: Original audio samples
        modified: Modified (stego) audio samples
        
    Returns:
        SNR in decibels
    """
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    
    signal_power = np.mean(original ** 2)
    noise = original - modified
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return float('inf')  # No difference
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_mse(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        original: Original audio samples
        modified: Modified (stego) audio samples
        
    Returns:
        MSE value
    """
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    
    mse = np.mean((original - modified) ** 2)
    return mse
