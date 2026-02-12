"""
AI-Guided LSB Audio Steganography - Importance Model

ENGINEERING JUSTIFICATION:
==========================
Why Pure LSB is Suboptimal:
- Sequential embedding modifies ALL samples starting from index 0
- Silent/quiet regions have very low amplitude → LSB changes are proportionally larger
- Human hearing is more sensitive to changes in quiet regions (no masking)
- Result: Audible artifacts, especially in quiet passages

How AI-Guided Masking Improves Imperceptibility:
- Loud, complex audio regions exhibit "temporal masking" and "spectral masking"
- A loud sound masks nearby quieter sounds (psychoacoustic phenomenon)
- Small LSB changes in loud regions are completely inaudible
- We prioritize embedding in these "safe" regions

Why Deterministic Decoding is Preserved:
- The importance map is computed from the COVER audio samples
- The stego audio differs only in LSBs → importance map is IDENTICAL
- Same map → same sorting → same extraction order → 100% recovery

This is what makes this a HYBRID INTELLIGENT SYSTEM:
- AI (heuristic intelligence) decides WHERE to embed
- LSB (exact algorithm) performs the actual embedding
- Best of both worlds: imperceptibility + exactness

Author: AI-Guided Steganography System
"""

import numpy as np
from scipy import signal


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_FRAME_SIZE = 512  # Samples per frame for analysis
DEFAULT_HOP_SIZE = 256    # Hop between frames (50% overlap)


# =============================================================================
# CORE IMPORTANCE COMPUTATION
# =============================================================================

def compute_short_time_energy(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    Compute short-time energy per frame.
    
    Higher energy = louder region = better masking = safer to embed.
    
    Args:
        audio: 1D audio samples (flattened)
        frame_size: Number of samples per frame
        hop_size: Hop between frames
        
    Returns:
        1D array of energy values per frame
    """
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop_size + 1)
    
    energy = np.zeros(n_frames, dtype=np.float64)
    
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n_samples)
        frame = audio[start:end].astype(np.float64)
        # RMS energy (more perceptually relevant than sum of squares)
        energy[i] = np.sqrt(np.mean(frame ** 2)) + 1e-10  # Avoid log(0)
    
    return energy


def compute_spectral_flatness(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    Compute spectral flatness per frame.
    
    Spectral Flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)
    
    High flatness (close to 1) = noise-like = complex spectrum = better masking
    Low flatness (close to 0) = tonal = simple spectrum = worse masking
    
    Args:
        audio: 1D audio samples
        frame_size: FFT window size
        hop_size: Hop between frames
        
    Returns:
        1D array of spectral flatness per frame (0 to 1)
    """
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop_size + 1)
    
    flatness = np.zeros(n_frames, dtype=np.float64)
    window = np.hanning(frame_size)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        
        if end > n_samples:
            # Pad last frame with zeros
            frame = np.zeros(frame_size)
            frame[:n_samples - start] = audio[start:n_samples]
        else:
            frame = audio[start:end].astype(np.float64)
        
        # Apply window and compute FFT
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed)) + 1e-10
        
        # Geometric mean (using log for numerical stability)
        log_spectrum = np.log(spectrum)
        geometric_mean = np.exp(np.mean(log_spectrum))
        arithmetic_mean = np.mean(spectrum)
        
        flatness[i] = geometric_mean / arithmetic_mean
    
    return flatness


def compute_zero_crossing_rate(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    Compute zero-crossing rate per frame.
    
    High ZCR = noisy/complex signal = better masking
    Low ZCR = smooth/tonal signal = worse masking
    
    Args:
        audio: 1D audio samples
        frame_size: Number of samples per frame
        hop_size: Hop between frames
        
    Returns:
        1D array of ZCR per frame (0 to 1)
    """
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop_size + 1)
    
    zcr = np.zeros(n_frames, dtype=np.float64)
    
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n_samples)
        frame = audio[start:end].astype(np.float64)
        
        if len(frame) > 1:
            # Count sign changes
            signs = np.sign(frame)
            sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
            zcr[i] = sign_changes / (len(frame) - 1)
        else:
            zcr[i] = 0.0
    
    return zcr


def expand_frame_importance_to_samples(
    frame_importance: np.ndarray,
    n_samples: int,
    frame_size: int,
    hop_size: int
) -> np.ndarray:
    """
    CRITICAL DETAIL #1: Frame → Sample Mapping
    ==========================================
    We compute importance per FRAME, but embed per SAMPLE.
    This function expands frame importance to per-sample importance.
    
    Each sample gets the importance of the frame that contains its center.
    For overlapping frames, we use the maximum importance (conservative choice).
    
    Both encoder and decoder MUST use this identical function.
    
    Args:
        frame_importance: 1D array of importance per frame
        n_samples: Total number of samples in audio
        frame_size: Samples per frame
        hop_size: Hop between frames
        
    Returns:
        1D array of importance per sample (same length as audio)
    """
    sample_importance = np.zeros(n_samples, dtype=np.float64)
    n_frames = len(frame_importance)
    
    # For each frame, assign its importance to all samples in that frame
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n_samples)
        # Use maximum for overlapping regions (more conservative = safer)
        sample_importance[start:end] = np.maximum(
            sample_importance[start:end],
            frame_importance[i]
        )
    
    return sample_importance


def compute_importance_map(
    audio: np.ndarray,
    sr: int,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    bits_per_sample: int = 1
) -> np.ndarray:
    """
    Compute per-sample importance scores using psychoacoustic features.
    
    HIGHER SCORE = SAFER TO EMBED (better masking)
    
    Combined features:
    - Short-time Energy (STE): Louder regions mask better
    - Spectral Flatness (SF): Noisy regions mask better
    - Zero-Crossing Rate (ZCR): Complex signals mask better
    
    CRITICAL FOR DETERMINISM:
    =========================
    We MUST compute importance from the UPPER BITS only, ignoring LSBs.
    This ensures that encoder and decoder produce IDENTICAL importance maps,
    regardless of whether the audio has been modified by LSB embedding.
    
    The mask clears the lower N bits (where N = bits_per_sample).
    
    All computations are FULLY DETERMINISTIC → enables exact decoding.
    
    Args:
        audio: Audio samples (can be mono or stereo, will be flattened)
        sr: Sample rate (not used in current implementation, reserved)
        frame_size: Analysis frame size in samples
        hop_size: Hop size between frames
        bits_per_sample: Number of LSB bits to ignore (default 1)
        
    Returns:
        1D array of importance scores per sample (same length as flattened audio)
    """
    # Flatten to mono for consistent analysis
    audio_flat = audio.flatten()
    n_samples = len(audio_flat)
    
    if n_samples == 0:
        return np.array([])
    
    # =========================================================================
    # CRITICAL: Mask out LSB(s) to ensure identical importance for cover & stego
    # =========================================================================
    # Create mask that clears the lower N bits
    # For bits_per_sample=1: mask = 0xFFFE (-2 in int16)
    # For bits_per_sample=2: mask = 0xFFFC (-4 in int16)
    lsb_mask = np.int16(~((1 << bits_per_sample) - 1))
    
    # Apply mask to get upper bits only
    audio_masked = (audio_flat.astype(np.int16) & lsb_mask).astype(np.float64)
    
    # Compute frame-level features on MASKED audio
    energy = compute_short_time_energy(audio_masked, frame_size, hop_size)
    flatness = compute_spectral_flatness(audio_masked, frame_size, hop_size)
    zcr = compute_zero_crossing_rate(audio_masked, frame_size, hop_size)
    
    # Normalize each feature to [0, 1] range
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-10:
            return np.ones_like(x) * 0.5  # Constant signal
        return (x - x_min) / (x_max - x_min)
    
    energy_norm = normalize(energy)
    flatness_norm = normalize(flatness)
    zcr_norm = normalize(zcr)
    
    # Combine features with weights
    # Energy is most important (loud = safe), followed by flatness and ZCR
    WEIGHT_ENERGY = 0.5
    WEIGHT_FLATNESS = 0.3
    WEIGHT_ZCR = 0.2
    
    frame_importance = (
        WEIGHT_ENERGY * energy_norm +
        WEIGHT_FLATNESS * flatness_norm +
        WEIGHT_ZCR * zcr_norm
    )
    
    # Expand frame importance to per-sample importance
    sample_importance = expand_frame_importance_to_samples(
        frame_importance, n_samples, frame_size, hop_size
    )
    
    return sample_importance


def get_embedding_order(importance_map: np.ndarray) -> np.ndarray:
    """
    Get the order of samples for embedding, sorted by importance.
    
    CRITICAL DETAIL #2: Stable Sorting
    ===================================
    When sorting samples by importance, we MUST enforce a stable tie-break.
    
    Sort key: (-importance, sample_index)
    
    Why? Many samples may have identical importance scores.
    Without stable sorting, encoder and decoder order DIVERGES.
    Result: Silent corruption, unrecoverable secret.
    
    This function returns indices sorted by:
    1. Descending importance (embed in safest regions first)
    2. Ascending sample index (stable tie-break for determinism)
    
    Args:
        importance_map: Per-sample importance scores
        
    Returns:
        1D array of sample indices in embedding order
    """
    n_samples = len(importance_map)
    
    # Create array of (negative_importance, index) for sorting
    # Negative importance because we want descending order
    # Index as secondary key ensures stable tie-breaking
    sort_keys = np.zeros(n_samples, dtype=[('neg_importance', np.float64), ('index', np.int64)])
    sort_keys['neg_importance'] = -importance_map
    sort_keys['index'] = np.arange(n_samples)
    
    # Sort by (neg_importance, index) - this gives us stable sorting
    sorted_indices = np.argsort(sort_keys, order=('neg_importance', 'index'))
    
    return sorted_indices


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_importance_distribution(importance_map: np.ndarray) -> dict:
    """
    Analyze the importance distribution for debugging/logging.
    
    Args:
        importance_map: Per-sample importance scores
        
    Returns:
        Dictionary with statistics
    """
    return {
        'min': float(np.min(importance_map)),
        'max': float(np.max(importance_map)),
        'mean': float(np.mean(importance_map)),
        'std': float(np.std(importance_map)),
        'percentile_10': float(np.percentile(importance_map, 10)),
        'percentile_90': float(np.percentile(importance_map, 90)),
    }
