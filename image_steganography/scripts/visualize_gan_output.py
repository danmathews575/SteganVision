"""
GAN Steganography Model Visualization Script

Loads the best GAN checkpoint and visualizes model outputs on fixed samples.
Creates a 5-column grid: Cover | Stego | Difference | Secret | Recovered

Usage:
    python scripts/visualize_gan_output.py
"""

import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder
from data.dataset import SteganographyDataset


def load_checkpoint(checkpoint_path, encoder, decoder, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    losses = checkpoint.get('losses', {})
    
    print(f"  Loaded from epoch {epoch}")
    if losses:
        print(f"  G Loss: {losses.get('g_loss', 'N/A'):.6f}")
        print(f"  Cover Loss: {losses.get('cover_loss', 'N/A'):.6f}")
        print(f"  Secret Loss: {losses.get('secret_loss', 'N/A'):.6f}")
    
    return encoder, decoder, epoch


def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for visualization."""
    return (tensor + 1) / 2


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for matplotlib."""
    # Move to CPU, denormalize, convert to numpy
    img = denormalize(tensor).cpu().numpy()
    
    # Handle different channel counts
    if img.shape[0] == 1:
        # Grayscale: (1, H, W) -> (H, W)
        img = img.squeeze(0)
    elif img.shape[0] == 3:
        # RGB: (3, H, W) -> (H, W, 3)
        img = np.transpose(img, (1, 2, 0))
    
    return np.clip(img, 0, 1)


def compute_difference_map(cover, stego):
    """Compute amplified absolute difference between cover and stego."""
    # Absolute difference, amplified by 10x
    diff = torch.abs(cover - stego).mean(dim=0)  # Average across channels -> (H, W)
    diff_amplified = torch.clamp(diff * 10, 0, 1)
    return diff_amplified.cpu().numpy()


def visualize_outputs(encoder, decoder, dataset, device, num_samples=4, output_path='outputs/current_model_result.png'):
    """
    Visualize model outputs in a 5-column grid.
    
    Columns: Cover | Stego | Difference Map | Secret | Recovered
    """
    encoder.eval()
    decoder.eval()
    
    # Create figure with 5 columns and num_samples rows
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
    
    # Column titles
    col_titles = ['Cover Image', 'Stego Image', '|Cover − Stego| × 10', 'Original Secret', 'Recovered Secret']
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get fixed sample (deterministic - same samples each run)
            cover, secret = dataset[i]
            
            # Add batch dimension and move to device
            cover = cover.unsqueeze(0).to(device)
            secret = secret.unsqueeze(0).to(device)
            
            # Forward pass
            stego = encoder(cover, secret)
            recovered = decoder(stego)
            
            # Remove batch dimension for visualization
            cover = cover.squeeze(0)
            secret = secret.squeeze(0)
            stego = stego.squeeze(0)
            recovered = recovered.squeeze(0)
            
            # Convert to numpy for plotting
            cover_np = tensor_to_numpy(cover)
            stego_np = tensor_to_numpy(stego)
            secret_np = tensor_to_numpy(secret)
            recovered_np = tensor_to_numpy(recovered)
            diff_np = compute_difference_map(cover, stego)
            
            # Plot row
            row_axes = axes[i] if num_samples > 1 else axes
            
            # Column 1: Cover
            row_axes[0].imshow(cover_np)
            row_axes[0].axis('off')
            if i == 0:
                row_axes[0].set_title(col_titles[0], fontsize=12, fontweight='bold')
            
            # Column 2: Stego
            row_axes[1].imshow(stego_np)
            row_axes[1].axis('off')
            if i == 0:
                row_axes[1].set_title(col_titles[1], fontsize=12, fontweight='bold')
            
            # Column 3: Difference Map (heatmap)
            row_axes[2].imshow(diff_np, cmap='inferno')
            row_axes[2].axis('off')
            if i == 0:
                row_axes[2].set_title(col_titles[2], fontsize=12, fontweight='bold')
            
            # Column 4: Original Secret
            row_axes[3].imshow(secret_np, cmap='gray')
            row_axes[3].axis('off')
            if i == 0:
                row_axes[3].set_title(col_titles[3], fontsize=12, fontweight='bold')
            
            # Column 5: Recovered Secret
            row_axes[4].imshow(recovered_np, cmap='gray')
            row_axes[4].axis('off')
            if i == 0:
                row_axes[4].set_title(col_titles[4], fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('GAN Steganography — Current Model Output', fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nVisualization saved to: {output_path}")
    
    # Display inline (works in notebooks)
    plt.show()
    
    return fig


def main():
    """Main visualization function."""
    print("=" * 60)
    print("GAN Steganography Model Visualization")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Find best checkpoint
    checkpoint_dir = Path('checkpoints/gan')
    best_checkpoint = checkpoint_dir / 'best_gan_model.pth'
    
    if not best_checkpoint.exists():
        # Fallback: find latest epoch checkpoint
        epoch_checkpoints = sorted(checkpoint_dir.glob('gan_checkpoint_epoch_*.pth'))
        if epoch_checkpoints:
            best_checkpoint = epoch_checkpoints[-1]
            print(f"\nbest_gan_model.pth not found, using: {best_checkpoint.name}")
        else:
            raise FileNotFoundError(f"No GAN checkpoints found in {checkpoint_dir}")
    
    # Initialize models
    print("\nInitializing models...")
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    # Load checkpoint
    encoder, decoder, epoch = load_checkpoint(best_checkpoint, encoder, decoder, device)
    
    # Load dataset (only need a few samples)
    print("\nLoading dataset...")
    dataset = SteganographyDataset(
        celeba_dir='data/celeba',
        mnist_root='data/mnist',
        image_size=256,
        split='train',
        max_samples=10  # Only load a few samples
    )
    
    # Visualize outputs
    print("\nGenerating visualization...")
    num_samples = 4  # Number of rows
    output_path = 'outputs/current_model_result.png'
    
    visualize_outputs(encoder, decoder, dataset, device, num_samples, output_path)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
