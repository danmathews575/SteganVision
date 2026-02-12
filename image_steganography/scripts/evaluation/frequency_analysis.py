"""
Frequency-Domain Analysis for Steganography Evaluation

Analyzes spectral characteristics of cover vs stego images using:
- FFT (Fast Fourier Transform)
- DCT (Discrete Cosine Transform)
- Radially Averaged Power Spectrum

GAN stego should have spectrum ≈ cover spectrum
CNN stego may show artifacts in high frequencies
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import zoom


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D FFT magnitude spectrum.
    
    Args:
        image: Grayscale or RGB image [H, W] or [C, H, W]
        
    Returns:
        Log-magnitude spectrum [H, W]
    """
    # Convert to grayscale if RGB
    if image.ndim == 3:
        if image.shape[0] in [1, 3]:  # CHW format
            image = np.mean(image, axis=0)
        else:  # HWC format
            image = np.mean(image, axis=2)
    
    # Compute 2D FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Log magnitude spectrum
    magnitude = np.abs(f_shift)
    log_magnitude = np.log1p(magnitude)  # log(1 + x) for stability
    
    return log_magnitude


def compute_dct_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D DCT magnitude.
    
    Args:
        image: Grayscale or RGB image [H, W] or [C, H, W]
        
    Returns:
        Log-DCT magnitude [H, W]
    """
    # Convert to grayscale if RGB
    if image.ndim == 3:
        if image.shape[0] in [1, 3]:  # CHW format
            image = np.mean(image, axis=0)
        else:  # HWC format
            image = np.mean(image, axis=2)
    
    # Compute 2D DCT
    dct_result = fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
    
    # Log magnitude
    log_magnitude = np.log1p(np.abs(dct_result))
    
    return log_magnitude


def radial_average_spectrum(spectrum: np.ndarray, num_bins: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radially averaged power spectrum.
    
    Compresses 2D spectrum into 1D curve for cleaner comparison.
    
    Args:
        spectrum: 2D magnitude spectrum [H, W]
        num_bins: Number of radial bins
        
    Returns:
        Tuple of (frequencies, radial_profile)
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Create radial coordinates
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Maximum radius
    max_radius = min(center_x, center_y)
    
    # Bin the spectrum by radius
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    radial_profile = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = np.mean(spectrum[mask])
    
    # Normalize frequencies to [0, 1]
    frequencies = (bin_edges[:-1] + bin_edges[1:]) / 2 / max_radius
    
    return frequencies, radial_profile


def compute_spectral_difference(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute spectral difference between two images.
    
    Args:
        spectrum1: First magnitude spectrum
        spectrum2: Second magnitude spectrum
        
    Returns:
        Mean absolute difference
    """
    return np.mean(np.abs(spectrum1 - spectrum2))


def analyze_batch_spectra(
    covers: List[torch.Tensor],
    cnn_stegos: List[torch.Tensor],
    gan_stegos: List[torch.Tensor],
    num_samples: int = 100,
    num_bins: int = 128
) -> Dict:
    """
    Compute batch spectral analysis.
    
    Args:
        covers: List of cover tensors [C, H, W]
        cnn_stegos: List of CNN stego tensors
        gan_stegos: List of GAN stego tensors
        num_samples: Number of samples to analyze
        num_bins: Number of radial frequency bins
        
    Returns:
        Dictionary with spectral analysis results
    """
    n = min(num_samples, len(covers))
    
    # Initialize accumulators
    cover_profiles = []
    cnn_profiles = []
    gan_profiles = []
    
    cnn_diffs = []
    gan_diffs = []
    
    print(f"Analyzing spectral characteristics of {n} samples...")
    
    for i in range(n):
        if (i + 1) % 20 == 0:
            print(f"  Processing sample {i + 1}/{n}")
        
        # Convert tensors to numpy
        cover_np = covers[i].cpu().numpy()
        cnn_np = cnn_stegos[i].cpu().numpy()
        gan_np = gan_stegos[i].cpu().numpy()
        
        # Compute FFT magnitude spectra
        cover_fft = compute_fft_magnitude(cover_np)
        cnn_fft = compute_fft_magnitude(cnn_np)
        gan_fft = compute_fft_magnitude(gan_np)
        
        # Radially averaged profiles
        freqs, cover_profile = radial_average_spectrum(cover_fft, num_bins)
        _, cnn_profile = radial_average_spectrum(cnn_fft, num_bins)
        _, gan_profile = radial_average_spectrum(gan_fft, num_bins)
        
        cover_profiles.append(cover_profile)
        cnn_profiles.append(cnn_profile)
        gan_profiles.append(gan_profile)
        
        # Spectral differences
        cnn_diffs.append(compute_spectral_difference(cover_fft, cnn_fft))
        gan_diffs.append(compute_spectral_difference(cover_fft, gan_fft))
    
    # Average profiles
    cover_profile_mean = np.mean(cover_profiles, axis=0)
    cover_profile_std = np.std(cover_profiles, axis=0)
    cnn_profile_mean = np.mean(cnn_profiles, axis=0)
    cnn_profile_std = np.std(cnn_profiles, axis=0)
    gan_profile_mean = np.mean(gan_profiles, axis=0)
    gan_profile_std = np.std(gan_profiles, axis=0)
    
    # Spectral difference statistics
    cnn_diff_mean = np.mean(cnn_diffs)
    cnn_diff_std = np.std(cnn_diffs)
    gan_diff_mean = np.mean(gan_diffs)
    gan_diff_std = np.std(gan_diffs)
    
    return {
        'frequencies': freqs,
        'cover_profile_mean': cover_profile_mean,
        'cover_profile_std': cover_profile_std,
        'cnn_profile_mean': cnn_profile_mean,
        'cnn_profile_std': cnn_profile_std,
        'gan_profile_mean': gan_profile_mean,
        'gan_profile_std': gan_profile_std,
        'cnn_spectral_diff_mean': cnn_diff_mean,
        'cnn_spectral_diff_std': cnn_diff_std,
        'gan_spectral_diff_mean': gan_diff_mean,
        'gan_spectral_diff_std': gan_diff_std
    }


def plot_spectral_comparison(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot spectral comparison between cover, CNN stego, and GAN stego.
    
    Args:
        results: Dictionary from analyze_batch_spectra
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    freqs = results['frequencies']
    
    # Plot 1: Radially averaged power spectra
    ax1 = axes[0]
    ax1.plot(freqs, results['cover_profile_mean'], 'b-', label='Cover', linewidth=2)
    ax1.fill_between(
        freqs,
        results['cover_profile_mean'] - results['cover_profile_std'],
        results['cover_profile_mean'] + results['cover_profile_std'],
        alpha=0.2, color='blue'
    )
    ax1.plot(freqs, results['cnn_profile_mean'], 'r-', label='CNN Stego', linewidth=2)
    ax1.plot(freqs, results['gan_profile_mean'], 'g-', label='GAN Stego', linewidth=2)
    ax1.set_xlabel('Normalized Frequency', fontsize=12)
    ax1.set_ylabel('Log Magnitude', fontsize=12)
    ax1.set_title('Radially Averaged Power Spectrum', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference from cover spectrum
    ax2 = axes[1]
    cnn_diff = np.abs(results['cnn_profile_mean'] - results['cover_profile_mean'])
    gan_diff = np.abs(results['gan_profile_mean'] - results['cover_profile_mean'])
    ax2.plot(freqs, cnn_diff, 'r-', label='CNN − Cover', linewidth=2)
    ax2.plot(freqs, gan_diff, 'g-', label='GAN − Cover', linewidth=2)
    ax2.set_xlabel('Normalized Frequency', fontsize=12)
    ax2.set_ylabel('Absolute Difference', fontsize=12)
    ax2.set_title('Spectral Deviation from Cover', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of mean spectral differences
    ax3 = axes[2]
    methods = ['CNN Stego', 'GAN Stego']
    means = [results['cnn_spectral_diff_mean'], results['gan_spectral_diff_mean']]
    stds = [results['cnn_spectral_diff_std'], results['gan_spectral_diff_std']]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax3.bar(methods, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax3.set_ylabel('Mean Spectral Difference ↓', fontsize=12)
    ax3.set_title('Overall Spectral Distance from Cover', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        # Save as both PNG and SVG
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        svg_path = save_path.replace('.png', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Spectral analysis saved to: {save_path}")
    
    return fig


def compute_high_frequency_energy(spectrum: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute proportion of energy in high frequencies.
    
    Steganography artifacts often appear in high frequencies.
    
    Args:
        spectrum: 2D magnitude spectrum
        threshold: Fraction of max radius to consider "high frequency"
        
    Returns:
        Ratio of high-frequency energy to total energy
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    max_radius = min(center_x, center_y)
    high_freq_mask = r > (threshold * max_radius)
    
    total_energy = np.sum(spectrum ** 2)
    high_freq_energy = np.sum(spectrum[high_freq_mask] ** 2)
    
    return high_freq_energy / total_energy if total_energy > 0 else 0


if __name__ == '__main__':
    # Test the frequency analysis module
    print("Testing Frequency Analysis...")
    
    # Create test images
    np.random.seed(42)
    h, w = 256, 256
    
    # Simulated cover (smooth image with some structure)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    cover = np.sin(5 * X) * np.cos(5 * Y) + 0.1 * np.random.randn(h, w)
    
    # Simulated stego (cover with high-frequency noise)
    cnn_stego = cover + 0.05 * np.random.randn(h, w)
    gan_stego = cover + 0.02 * np.random.randn(h, w)  # Less noise
    
    # Compute FFT
    cover_fft = compute_fft_magnitude(cover)
    cnn_fft = compute_fft_magnitude(cnn_stego)
    gan_fft = compute_fft_magnitude(gan_stego)
    
    # Radial profiles
    freqs, cover_profile = radial_average_spectrum(cover_fft)
    _, cnn_profile = radial_average_spectrum(cnn_fft)
    _, gan_profile = radial_average_spectrum(gan_fft)
    
    print(f"Cover spectrum shape: {cover_fft.shape}")
    print(f"Radial profile length: {len(cover_profile)}")
    
    # Spectral differences
    cnn_diff = compute_spectral_difference(cover_fft, cnn_fft)
    gan_diff = compute_spectral_difference(cover_fft, gan_fft)
    
    print(f"CNN spectral difference: {cnn_diff:.4f}")
    print(f"GAN spectral difference: {gan_diff:.4f}")
    
    # High-frequency energy
    cover_hf = compute_high_frequency_energy(cover_fft)
    cnn_hf = compute_high_frequency_energy(cnn_fft)
    gan_hf = compute_high_frequency_energy(gan_fft)
    
    print(f"High-freq energy - Cover: {cover_hf:.4f}, CNN: {cnn_hf:.4f}, GAN: {gan_hf:.4f}")
    
    print("\nFrequency analysis test complete!")
