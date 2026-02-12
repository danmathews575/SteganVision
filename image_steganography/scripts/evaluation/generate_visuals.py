"""
Generate Publication-Quality Visualizations for GAN vs CNN Evaluation

Creates:
1. Visual comparison grid (Cover | CNN Stego | GAN Stego | CNN Diff | GAN Diff)
2. ROC curves for steganalysis
3. Box plots for metric distributions
4. Annotated difference maps with mean pixel error

All outputs saved as both PNG (raster) and SVG (vector) formats.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_images_as_numpy(image_dir: Path, indices: List[int]) -> List[np.ndarray]:
    """Load specific images as numpy arrays [0, 1]."""
    image_paths = sorted(image_dir.glob('*.png'))
    images = []
    
    for idx in indices:
        if idx < len(image_paths):
            img = Image.open(image_paths[idx]).convert('RGB')
            img_np = np.array(img) / 255.0
            images.append(img_np)
    
    return images


def load_grayscale_as_numpy(image_dir: Path, indices: List[int]) -> List[np.ndarray]:
    """Load specific grayscale images as numpy arrays [0, 1]."""
    image_paths = sorted(image_dir.glob('*.png'))
    images = []
    
    for idx in indices:
        if idx < len(image_paths):
            img = Image.open(image_paths[idx]).convert('L')
            img_np = np.array(img) / 255.0
            images.append(img_np)
    
    return images


def compute_diff_map(img1: np.ndarray, img2: np.ndarray, amplify: float = 10.0) -> np.ndarray:
    """Compute amplified absolute difference map."""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_mean = np.mean(diff, axis=2) if diff.ndim == 3 else diff
    diff_amplified = np.clip(diff_mean * amplify, 0, 1)
    return diff_amplified


def generate_visual_comparison(
    data_dir: Path,
    output_path: Path,
    num_samples: int = 5,
    sample_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (20, 4)
):
    """
    Generate 5-column comparison grid.
    
    Columns: Cover | CNN Stego | GAN Stego | CNN Diff ×10 | GAN Diff ×10
    """
    print("\nGenerating visual comparison grid...")
    
    if sample_indices is None:
        sample_indices = list(range(num_samples))
    
    n = len(sample_indices)
    
    # Load images
    covers = load_images_as_numpy(data_dir / 'cover', sample_indices)
    cnn_stegos = load_images_as_numpy(data_dir / 'cnn_stego', sample_indices)
    gan_stegos = load_images_as_numpy(data_dir / 'gan_stego', sample_indices)
    
    # Create figure
    fig_height = 3 * n + 1
    fig, axes = plt.subplots(n, 5, figsize=(figsize[0], fig_height))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    col_titles = ['Cover Image', 'CNN Stego', 'GAN Stego', 'CNN |Diff| ×10', 'GAN |Diff| ×10']
    
    for i in range(n):
        cover = covers[i]
        cnn_stego = cnn_stegos[i]
        gan_stego = gan_stegos[i]
        
        # Compute difference maps
        cnn_diff = compute_diff_map(cover, cnn_stego, amplify=10.0)
        gan_diff = compute_diff_map(cover, gan_stego, amplify=10.0)
        
        # Compute mean pixel errors
        cnn_mpe = np.mean(np.abs(cover - cnn_stego))
        gan_mpe = np.mean(np.abs(cover - gan_stego))
        
        # Column 1: Cover
        axes[i, 0].imshow(cover)
        axes[i, 0].axis('off')
        
        # Column 2: CNN Stego
        axes[i, 1].imshow(cnn_stego)
        axes[i, 1].axis('off')
        
        # Column 3: GAN Stego
        axes[i, 2].imshow(gan_stego)
        axes[i, 2].axis('off')
        
        # Column 4: CNN Diff with annotation
        im_cnn = axes[i, 3].imshow(cnn_diff, cmap='inferno', vmin=0, vmax=1)
        axes[i, 3].axis('off')
        axes[i, 3].text(
            0.5, -0.05, f'MPE: {cnn_mpe:.4f}',
            transform=axes[i, 3].transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold'
        )
        
        # Column 5: GAN Diff with annotation
        im_gan = axes[i, 4].imshow(gan_diff, cmap='inferno', vmin=0, vmax=1)
        axes[i, 4].axis('off')
        axes[i, 4].text(
            0.5, -0.05, f'MPE: {gan_mpe:.4f}',
            transform=axes[i, 4].transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold'
        )
        
        # Set column titles for first row
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('GAN vs CNN Steganography — Visual Comparison', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save as PNG and SVG
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Visual comparison saved to: {output_path}")
    print(f"SVG version saved to: {svg_path}")
    
    plt.close(fig)
    return fig


def generate_secret_comparison(
    data_dir: Path,
    output_path: Path,
    num_samples: int = 5,
    sample_indices: Optional[List[int]] = None
):
    """
    Generate secret recovery comparison grid.
    
    Columns: Original Secret | CNN Recovered | GAN Recovered
    """
    print("\nGenerating secret recovery comparison...")
    
    if sample_indices is None:
        sample_indices = list(range(num_samples))
    
    n = len(sample_indices)
    
    # Load images
    secrets = load_grayscale_as_numpy(data_dir / 'secret', sample_indices)
    cnn_recovered = load_grayscale_as_numpy(data_dir / 'cnn_recovered', sample_indices)
    gan_recovered = load_grayscale_as_numpy(data_dir / 'gan_recovered', sample_indices)
    
    # Create figure
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n + 1))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = ['Original Secret', 'CNN Recovered', 'GAN Recovered']
    
    for i in range(n):
        # Column 1: Original Secret
        axes[i, 0].imshow(secrets[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        
        # Column 2: CNN Recovered
        axes[i, 1].imshow(cnn_recovered[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        cnn_mse = np.mean((secrets[i] - cnn_recovered[i]) ** 2)
        axes[i, 1].text(0.5, -0.05, f'MSE: {cnn_mse:.4f}',
                        transform=axes[i, 1].transAxes, ha='center', va='top', fontsize=10)
        
        # Column 3: GAN Recovered
        axes[i, 2].imshow(gan_recovered[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        gan_mse = np.mean((secrets[i] - gan_recovered[i]) ** 2)
        axes[i, 2].text(0.5, -0.05, f'MSE: {gan_mse:.4f}',
                        transform=axes[i, 2].transAxes, ha='center', va='top', fontsize=10)
        
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    fig.suptitle('Secret Recovery Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Secret comparison saved to: {output_path}")
    
    plt.close(fig)


def generate_roc_curves(
    cnn_fpr: np.ndarray,
    cnn_tpr: np.ndarray,
    cnn_auc: float,
    gan_fpr: np.ndarray,
    gan_tpr: np.ndarray,
    gan_auc: float,
    output_path: Path
):
    """Generate ROC curve comparison plot."""
    print("\nGenerating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curves
    ax.plot(cnn_fpr, cnn_tpr, 'r-', linewidth=2, label=f'CNN Stego (AUC = {cnn_auc:.4f})')
    ax.plot(gan_fpr, gan_tpr, 'g-', linewidth=2, label=f'GAN Stego (AUC = {gan_auc:.4f})')
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    # Perfect undetectable steganography region
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color='green', label='Ideal (undetectable)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Steganalysis ROC Curves\n(Lower AUC = Better Steganography)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"ROC curves saved to: {output_path}")
    
    plt.close(fig)


def generate_metrics_boxplots(
    cnn_metrics: Dict,
    gan_metrics: Dict,
    output_path: Path
):
    """Generate box plots comparing metric distributions."""
    print("\nGenerating metrics box plots...")
    
    metrics_to_plot = ['psnr', 'ssim', 'secret_psnr', 'ber']
    titles = ['PSNR (dB) ↑', 'SSIM ↑', 'Secret PSNR (dB) ↑', 'BER ↓']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    colors = {'CNN': '#e74c3c', 'GAN': '#27ae60'}
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        if metric in cnn_metrics and metric in gan_metrics:
            cnn_vals = cnn_metrics[metric]['values']
            gan_vals = gan_metrics[metric]['values']
            
            bp = ax.boxplot(
                [cnn_vals, gan_vals],
                labels=['CNN', 'GAN'],
                patch_artist=True
            )
            
            bp['boxes'][0].set_facecolor(colors['CNN'])
            bp['boxes'][1].set_facecolor(colors['GAN'])
            
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title.split()[0], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Metric Distributions: CNN vs GAN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Box plots saved to: {output_path}")
    
    plt.close(fig)


def main():
    """Generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    parser.add_argument('--data_dir', type=str, default='outputs/evaluation/data',
                        help='Directory with test data')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation/plots',
                        help='Output directory for plots')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Evaluation Visualizations")
    print("=" * 60)
    
    # Check if data exists
    if not (data_dir / 'cover').exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run prepare_test_data.py first.")
        return
    
    # Count available samples
    n_available = len(list((data_dir / 'cover').glob('*.png')))
    n_samples = min(args.num_samples, n_available)
    print(f"Using {n_samples} samples from {n_available} available")
    
    # Generate visual comparison
    generate_visual_comparison(
        data_dir,
        output_dir / 'visual_comparison.png',
        num_samples=n_samples
    )
    
    # Generate secret comparison
    generate_secret_comparison(
        data_dir,
        output_dir / 'secret_comparison.png',
        num_samples=n_samples
    )
    
    print("\n" + "=" * 60)
    print("Visualization Generation Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
