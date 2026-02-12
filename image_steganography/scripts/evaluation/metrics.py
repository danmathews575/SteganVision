"""
Image Quality Metrics for Steganography Evaluation

Provides comprehensive metrics for comparing CNN vs GAN steganography:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- BER (Bit Error Rate)
- LPIPS (Learned Perceptual Image Patch Similarity) - optional

All metrics support batch computation with mean ± std reporting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
import warnings

# Try to import LPIPS (optional dependency)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips")


class MetricsCalculator:
    """
    Comprehensive metrics calculator for steganography evaluation.
    
    Computes image quality metrics between cover/stego pairs and
    secret/recovered pairs with statistical reporting.
    """
    
    def __init__(self, device: torch.device = None, use_lpips: bool = True):
        """
        Initialize metrics calculator.
        
        Args:
            device: Torch device for LPIPS computation
            use_lpips: Whether to use LPIPS (requires lpips package)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        
        if self.use_lpips:
            print("Loading LPIPS model (VGG backbone)...")
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_model.eval()
            print("LPIPS model loaded.")
        else:
            self.lpips_model = None
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor [-1, 1] to numpy [0, 1]."""
        img = tensor.detach().cpu().numpy()
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        return np.clip(img, 0, 1)
    
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            img1: First image [C, H, W] or [H, W] in range [0, 1]
            img2: Second image [C, H, W] or [H, W] in range [0, 1]
            data_range: Data range of images (default: 1.0)
            
        Returns:
            PSNR value in dB (higher is better)
        """
        # Handle channel dimension
        if img1.ndim == 3:
            img1 = np.transpose(img1, (1, 2, 0))  # CHW -> HWC
            img2 = np.transpose(img2, (1, 2, 0))
        
        return psnr_skimage(img1, img2, data_range=data_range)
    
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
        """
        Compute Structural Similarity Index.
        
        Args:
            img1: First image [C, H, W] or [H, W] in range [0, 1]
            img2: Second image [C, H, W] or [H, W] in range [0, 1]
            data_range: Data range of images (default: 1.0)
            
        Returns:
            SSIM value in [0, 1] (higher is better)
        """
        # Handle channel dimension
        if img1.ndim == 3:
            img1 = np.transpose(img1, (1, 2, 0))  # CHW -> HWC
            img2 = np.transpose(img2, (1, 2, 0))
            channel_axis = 2
        else:
            channel_axis = None
        
        return ssim_skimage(img1, img2, data_range=data_range, channel_axis=channel_axis)
    
    @staticmethod
    def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        
        Args:
            img1: First image [C, H, W] or [H, W] in range [0, 1]
            img2: Second image [C, H, W] or [H, W] in range [0, 1]
            
        Returns:
            MSE value (lower is better)
        """
        return np.mean((img1 - img2) ** 2)
    
    @staticmethod
    def compute_ber(secret: np.ndarray, recovered: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute Bit Error Rate for secret images.
        
        Binarizes images using fixed threshold and computes error rate.
        
        Args:
            secret: Original secret [C, H, W] or [H, W] in range [0, 1]
            recovered: Recovered secret [C, H, W] or [H, W] in range [0, 1]
            threshold: Binarization threshold (default: 0.5)
            
        Returns:
            BER value in [0, 1] (lower is better, 0 = perfect recovery)
        """
        # Binarize using fixed threshold
        secret_binary = (secret > threshold).astype(np.float32)
        recovered_binary = (recovered > threshold).astype(np.float32)
        
        # Compute bit error rate
        errors = np.abs(secret_binary - recovered_binary)
        ber = np.mean(errors)
        
        return ber
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).
        
        Args:
            img1: First image tensor [1, C, H, W] in range [-1, 1]
            img2: Second image tensor [1, C, H, W] in range [-1, 1]
            
        Returns:
            LPIPS value (lower is better, 0 = identical)
        """
        if not self.use_lpips:
            return float('nan')
        
        with torch.no_grad():
            # Ensure 4D tensor
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            
            # Handle grayscale by repeating channels
            if img1.shape[1] == 1:
                img1 = img1.repeat(1, 3, 1, 1)
            if img2.shape[1] == 1:
                img2 = img2.repeat(1, 3, 1, 1)
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            lpips_val = self.lpips_model(img1, img2)
            
        return lpips_val.item()
    
    def compute_single_pair_metrics(
        self,
        cover: torch.Tensor,
        stego: torch.Tensor,
        secret: torch.Tensor,
        recovered: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single image pair.
        
        Args:
            cover: Cover image tensor [C, H, W] in [-1, 1]
            stego: Stego image tensor [C, H, W] in [-1, 1]
            secret: Secret image tensor [C, H, W] in [-1, 1]
            recovered: Recovered secret tensor [C, H, W] in [-1, 1]
            
        Returns:
            Dictionary with all metric values
        """
        # Convert to numpy [0, 1]
        cover_np = self.tensor_to_numpy(cover)
        stego_np = self.tensor_to_numpy(stego)
        secret_np = self.tensor_to_numpy(secret)
        recovered_np = self.tensor_to_numpy(recovered)
        
        metrics = {}
        
        # Imperceptibility metrics (cover vs stego)
        metrics['psnr'] = self.compute_psnr(cover_np, stego_np)
        metrics['ssim'] = self.compute_ssim(cover_np, stego_np)
        metrics['mse'] = self.compute_mse(cover_np, stego_np)
        
        # LPIPS (if available)
        if self.use_lpips:
            metrics['lpips'] = self.compute_lpips(cover, stego)
        
        # Secret recovery metrics
        metrics['secret_psnr'] = self.compute_psnr(secret_np, recovered_np)
        metrics['secret_mse'] = self.compute_mse(secret_np, recovered_np)
        metrics['ber'] = self.compute_ber(secret_np, recovered_np, threshold=0.5)
        
        return metrics
    
    def compute_batch_metrics(
        self,
        covers: List[torch.Tensor],
        stegos: List[torch.Tensor],
        secrets: List[torch.Tensor],
        recovereds: List[torch.Tensor],
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for a batch of images with mean ± std.
        
        Args:
            covers: List of cover tensors
            stegos: List of stego tensors
            secrets: List of secret tensors
            recovereds: List of recovered tensors
            verbose: Whether to print progress
            
        Returns:
            Dictionary with 'mean', 'std', and 'values' for each metric
        """
        all_metrics = {
            'psnr': [], 'ssim': [], 'mse': [],
            'secret_psnr': [], 'secret_mse': [], 'ber': []
        }
        if self.use_lpips:
            all_metrics['lpips'] = []
        
        n_samples = len(covers)
        
        for i in range(n_samples):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processing sample {i + 1}/{n_samples}")
            
            metrics = self.compute_single_pair_metrics(
                covers[i], stegos[i], secrets[i], recovereds[i]
            )
            
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)
        
        # Compute mean ± std
        results = {}
        for key, values in all_metrics.items():
            values_arr = np.array(values)
            results[key] = {
                'mean': np.mean(values_arr),
                'std': np.std(values_arr),
                'values': values_arr
            }
        
        return results
    
    @staticmethod
    def format_metric(mean: float, std: float, precision: int = 4) -> str:
        """Format metric as 'mean ± std'."""
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    
    def print_results(self, results: Dict[str, Dict[str, float]], title: str = "Metrics"):
        """Print formatted results table."""
        print(f"\n{'=' * 50}")
        print(f"{title}")
        print('=' * 50)
        
        metric_names = {
            'psnr': 'PSNR (dB) ↑',
            'ssim': 'SSIM ↑',
            'mse': 'MSE ↓',
            'lpips': 'LPIPS ↓',
            'secret_psnr': 'Secret PSNR (dB) ↑',
            'secret_mse': 'Secret MSE ↓',
            'ber': 'BER ↓'
        }
        
        for key in ['psnr', 'ssim', 'mse', 'lpips', 'secret_psnr', 'secret_mse', 'ber']:
            if key in results:
                name = metric_names.get(key, key)
                formatted = self.format_metric(results[key]['mean'], results[key]['std'])
                print(f"  {name:20s}: {formatted}")
        
        print('=' * 50)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d value (0.2=small, 0.5=medium, 0.8=large)
    """
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1)


def paired_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test between two groups.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Tuple of (t-statistic, p-value)
    """
    from scipy import stats
    return stats.ttest_rel(group1, group2)


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of tuples (corrected_threshold, is_significant)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return [(corrected_alpha, p < corrected_alpha) for p in p_values]


if __name__ == '__main__':
    # Test the metrics module
    print("Testing MetricsCalculator...")
    
    calc = MetricsCalculator(use_lpips=LPIPS_AVAILABLE)
    
    # Create dummy tensors
    cover = torch.randn(3, 256, 256)
    stego = cover + torch.randn_like(cover) * 0.1  # Add small noise
    secret = torch.randn(1, 256, 256)
    recovered = secret + torch.randn_like(secret) * 0.05
    
    metrics = calc.compute_single_pair_metrics(cover, stego, secret, recovered)
    
    print("\nSingle pair metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMetrics module test complete!")
