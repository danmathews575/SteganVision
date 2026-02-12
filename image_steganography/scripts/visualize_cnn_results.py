"""
Visualization Script for CNN-based Image Steganography Results

This script loads a trained CNN encoder and decoder from a checkpoint
and generates qualitative visualizations of the steganography results.

For each sample, it displays:
- Cover image (original)
- Stego image (cover with hidden secret)
- |Cover - Stego| difference map (amplified for visibility)
- Secret image (original)
- Recovered secret (extracted from stego)

Usage:
    python scripts/visualize_cnn_results.py
    python scripts/visualize_cnn_results.py --checkpoint checkpoints/best_model.pth
    python scripts/visualize_cnn_results.py --num_samples 10
    python scripts/visualize_cnn_results.py --save_path results/visualization.png
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder
from data.dataset import SteganographyDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize CNN Steganography Results'
    )
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    
    # Data parameters
    parser.add_argument('--celeba_dir', type=str, default='data/celeba',
                        help='Path to CelebA images directory')
    parser.add_argument('--mnist_root', type=str, default='data/mnist',
                        help='Path to MNIST root directory')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of samples to visualize (default: 6, max: 10)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for encoder/decoder (must match training)')
    parser.add_argument('--diff_amplify', type=float, default=10.0,
                        help='Amplification factor for difference map (default: 10.0)')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization (optional, displays if not set)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure (default: 150)')
    parser.add_argument('--figsize_scale', type=float, default=2.5,
                        help='Scale factor for figure size (default: 2.5)')
    
    return parser.parse_args()


def setup_device():
    """
    Setup and return the best available device.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_models(checkpoint_path, base_channels, device):
    """
    Load encoder and decoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        base_channels: Number of base channels (must match training)
        device: Target device
        
    Returns:
        Tuple[Encoder, Decoder]: Loaded encoder and decoder models
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize models
    encoder = Encoder(base_channels=base_channels).to(device)
    decoder = Decoder(base_channels=base_channels).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Get training info from checkpoint
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    print(f"  Epoch: {epoch}")
    if isinstance(loss, float):
        print(f"  Loss: {loss:.6f}")
    
    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


def load_samples(args, num_samples):
    """
    Load samples from the dataset.
    
    Args:
        args: Parsed command line arguments
        num_samples: Number of samples to load
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cover and secret image batches
    """
    print(f"\nLoading {num_samples} samples from dataset...")
    
    # Create dataset (use test split for visualization)
    dataset = SteganographyDataset(
        celeba_dir=args.celeba_dir,
        mnist_root=args.mnist_root,
        image_size=256,
        split='test',  # Use test split for evaluation
        max_samples=num_samples
    )
    
    # Collect samples
    covers = []
    secrets = []
    
    for i in range(num_samples):
        cover, secret = dataset[i]
        covers.append(cover)
        secrets.append(secret)
    
    # Stack into batches
    covers = torch.stack(covers)
    secrets = torch.stack(secrets)
    
    return covers, secrets


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Input tensor in range [-1, 1]
        
    Returns:
        Tensor in range [0, 1], clipped to valid range
    """
    return torch.clamp((tensor + 1) / 2, 0, 1)


def tensor_to_numpy(tensor):
    """
    Convert a single image tensor to numpy array for matplotlib.
    
    Args:
        tensor: Image tensor of shape (C, H, W)
        
    Returns:
        numpy array of shape (H, W, C) or (H, W) for grayscale
    """
    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()
    
    # Handle grayscale (1, H, W) -> (H, W)
    if img.shape[0] == 1:
        return img[0]
    
    # Handle RGB (3, H, W) -> (H, W, 3)
    return np.transpose(img, (1, 2, 0))


def compute_difference_map(cover, stego, amplify=10.0):
    """
    Compute the absolute difference between cover and stego images.
    
    Args:
        cover: Cover image tensor (C, H, W)
        stego: Stego image tensor (C, H, W)
        amplify: Amplification factor for visibility
        
    Returns:
        Difference map as numpy array
    """
    # Compute absolute difference
    diff = torch.abs(cover - stego)
    
    # Convert to grayscale by averaging channels
    diff_gray = diff.mean(dim=0)
    
    # Amplify for visibility (differences are usually very subtle)
    diff_amplified = torch.clamp(diff_gray * amplify, 0, 1)
    
    return diff_amplified.cpu().numpy()


def create_visualization(covers, stegos, secrets, recovered, args):
    """
    Create a matplotlib visualization grid.
    
    Layout:
        Row 0: Labels
        Rows 1-N: [Cover | Stego | Difference | Secret | Recovered] for each sample
    
    Args:
        covers: Cover image tensors (N, 3, H, W)
        stegos: Stego image tensors (N, 3, H, W)
        secrets: Secret image tensors (N, 1, H, W)
        recovered: Recovered secret tensors (N, 1, H, W)
        args: Parsed command line arguments
        
    Returns:
        matplotlib figure
    """
    num_samples = covers.shape[0]
    num_cols = 5  # Cover, Stego, Difference, Secret, Recovered
    
    # Calculate figure size
    fig_width = num_cols * args.figsize_scale
    fig_height = num_samples * args.figsize_scale + 0.5
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    titles = ['Cover Image', 'Stego Image', f'|Cover - Stego| × {args.diff_amplify}', 
              'Secret Image', 'Recovered Secret']
    
    for sample_idx in range(num_samples):
        # Denormalize all images
        cover = denormalize(covers[sample_idx])
        stego = denormalize(stegos[sample_idx])
        secret = denormalize(secrets[sample_idx])
        rec = denormalize(recovered[sample_idx])
        
        # Compute difference map
        diff_map = compute_difference_map(covers[sample_idx], stegos[sample_idx], 
                                          amplify=args.diff_amplify)
        
        # Row images
        row_images = [
            tensor_to_numpy(cover),
            tensor_to_numpy(stego),
            diff_map,
            tensor_to_numpy(secret),
            tensor_to_numpy(rec)
        ]
        
        # Colormaps for each column
        cmaps = [None, None, 'hot', 'gray', 'gray']
        
        for col_idx, (img, cmap) in enumerate(zip(row_images, cmaps)):
            ax = axes[sample_idx, col_idx]
            
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(img)
            
            ax.axis('off')
            
            # Add column titles to first row only
            if sample_idx == 0:
                ax.set_title(titles[col_idx], fontsize=10, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add main title
    fig.suptitle('CNN Steganography Results', fontsize=14, fontweight='bold', y=1.02)
    
    return fig


def compute_metrics(covers, stegos, secrets, recovered):
    """
    Compute quantitative metrics for the steganography results.
    
    Args:
        covers: Cover image tensors (N, 3, H, W)
        stegos: Stego image tensors (N, 3, H, W)
        secrets: Secret image tensors (N, 1, H, W)
        recovered: Recovered secret tensors (N, 1, H, W)
        
    Returns:
        dict: Dictionary of metrics
    """
    # Cover loss (L1 between cover and stego)
    cover_loss = torch.mean(torch.abs(covers - stegos)).item()
    
    # Secret loss (L1 between secret and recovered)
    secret_loss = torch.mean(torch.abs(secrets - recovered)).item()
    
    # PSNR for stego (how similar is stego to cover)
    mse_stego = torch.mean((covers - stegos) ** 2).item()
    psnr_stego = 10 * np.log10(4.0 / mse_stego) if mse_stego > 0 else float('inf')
    
    # PSNR for secret recovery
    mse_secret = torch.mean((secrets - recovered) ** 2).item()
    psnr_secret = 10 * np.log10(4.0 / mse_secret) if mse_secret > 0 else float('inf')
    
    return {
        'cover_l1': cover_loss,
        'secret_l1': secret_loss,
        'psnr_stego': psnr_stego,
        'psnr_secret': psnr_secret
    }


def main():
    """Main visualization function."""
    args = parse_args()
    
    print("=" * 60)
    print("CNN Steganography Visualization")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Clamp number of samples
    num_samples = min(max(args.num_samples, 1), 10)
    if num_samples != args.num_samples:
        print(f"\nNote: Clamped num_samples to {num_samples}")
    
    # Load models
    encoder, decoder = load_models(args.checkpoint, args.base_channels, device)
    
    # Load samples
    covers, secrets = load_samples(args, num_samples)
    covers = covers.to(device)
    secrets = secrets.to(device)
    
    # Generate stego images and recover secrets
    print("\nGenerating steganography results...")
    with torch.no_grad():
        stegos = encoder(covers, secrets)
        recovered = decoder(stegos)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(covers, stegos, secrets, recovered)
    
    print(f"\nQuantitative Results:")
    print(f"  Cover L1 Loss:    {metrics['cover_l1']:.6f}")
    print(f"  Secret L1 Loss:   {metrics['secret_l1']:.6f}")
    print(f"  Stego PSNR:       {metrics['psnr_stego']:.2f} dB")
    print(f"  Secret PSNR:      {metrics['psnr_secret']:.2f} dB")
    
    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(covers, stegos, secrets, recovered, args)
    
    # Save or display
    if args.save_path:
        # Create directory if needed
        save_dir = Path(args.save_path).parent
        if save_dir and not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        fig.savefig(args.save_path, dpi=args.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"\nVisualization saved to: {args.save_path}")
    else:
        print("\nDisplaying visualization (close window to exit)...")
        plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
