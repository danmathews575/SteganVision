"""
Training Script for CNN-based Image Steganography

Trains the Encoder and Decoder jointly using L1 losses for:
- Cover loss: Stego image should look like cover image
- Secret loss: Reconstructed secret should match original secret

No GAN, discriminator, or adversarial training involved.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image

torch.cuda.empty_cache()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder_decoder import Encoder, Decoder
from data.dataset import SteganographyDataset
from utils.losses import stego_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN-based Image Steganography Model'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer (default: 1e-4)')
    
    # Data parameters
    parser.add_argument('--celeba_dir', type=str, 
                        default='data/celeba',
                        help='Path to CelebA images directory')
    parser.add_argument('--mnist_root', type=str, default='data/mnist',
                        help='Path to MNIST root directory')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2, optimized for Windows)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset to first N samples for faster iteration (default: None, use full dataset)')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for encoder/decoder (default: 64)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    return device


def create_dataloader(args):
    """
    Create and return the training DataLoader.
    
    Optimized for Windows + CUDA:
    - num_workers=2: Safe value for Windows multiprocessing
    - pin_memory=True when CUDA available: Faster CPU->GPU transfer
    - persistent_workers=True: Avoids worker restart overhead between epochs
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        DataLoader: Training data loader
    
    Example usage with subset:
        # Full dataset (default)
        python train_cnn.py --epochs 100
        
        # Quick test with 1000 samples
        python train_cnn.py --max_samples 1000 --epochs 5
        
        # Development iteration with 5000 samples
        python train_cnn.py --max_samples 5000 --epochs 20
    """
    # Create dataset with optional max_samples limit
    # max_samples selects the first N samples deterministically (no random sampling)
    dataset = SteganographyDataset(
        celeba_dir=args.celeba_dir,
        mnist_root=args.mnist_root,
        image_size=256,
        split='train',
        max_samples=args.max_samples  # None = full dataset, N = first N samples
    )
    
    # Optimize DataLoader settings based on environment
    use_cuda = torch.cuda.is_available()
    num_workers = args.num_workers
    
    # Log resolved dataset paths for verification
    print(f"\nDataLoader Configuration:")
    print(f"  - CelebA path: {Path(args.celeba_dir).resolve()}")
    print(f"  - MNIST path: {Path(args.mnist_root).resolve()}")
    print(f"  - num_workers: {num_workers}")
    print(f"  - pin_memory: {use_cuda}")
    print(f"  - persistent_workers: {num_workers > 0}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Enable only when CUDA is available
        drop_last=True,  # Drop incomplete batches
        persistent_workers=(num_workers > 0)  # Keep workers alive between epochs
    )
    
    return dataloader


def create_models(args, device):
    """
    Create and return encoder and decoder models.
    
    Args:
        args: Parsed command line arguments
        device: Target device (CPU/GPU)
        
    Returns:
        Tuple[Encoder, Decoder]: Encoder and decoder models on device
    """
    encoder = Encoder(base_channels=args.base_channels).to(device)
    decoder = Decoder(base_channels=args.base_channels).to(device)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"\nModel Parameters:")
    print(f"  - Encoder: {encoder_params:,}")
    print(f"  - Decoder: {decoder_params:,}")
    print(f"  - Total: {total_params:,}")
    
    return encoder, decoder


def save_checkpoint(encoder, decoder, optimizer, epoch, loss, checkpoint_dir, filename=None):
    """
    Save model checkpoint.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
        filename: Optional specific filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch:04d}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, encoder, decoder, optimizer, device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        device: Target device
        
    Returns:
        int: Starting epoch number
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
    
    return start_epoch


def train_epoch(encoder, decoder, dataloader, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Target device
        
    Returns:
        Tuple[float, float, float]: Average (total_loss, cover_loss, secret_loss)
    """
    encoder.train()
    decoder.train()
    
    total_loss_sum = 0.0
    cover_loss_sum = 0.0
    secret_loss_sum = 0.0
    num_batches = 0
    
    # Progress bar for batches
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for cover, secret in pbar:
        # Move data to device
        cover = cover.to(device)
        secret = secret.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through encoder
        # Encoder takes cover (B, 3, H, W) and secret (B, 1, H, W)
        # Returns stego (B, 3, H, W)
        stego = encoder(cover, secret)
        
        # Forward pass through decoder
        # Decoder takes stego (B, 3, H, W)
        # Returns recovered secret (B, 1, H, W)
        recovered = decoder(stego)
        
        # Compute losses
        total_loss, cover_loss, secret_loss = stego_loss(
            cover, stego, secret, recovered
        )
        
        # Backward pass
        total_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate losses
        total_loss_sum += total_loss.item()
        cover_loss_sum += cover_loss.item()
        secret_loss_sum += secret_loss.item()
        num_batches += 1
        
        # Update progress bar with current losses
        pbar.set_postfix({
            'Total': f'{total_loss.item():.4f}',
            'Cover': f'{cover_loss.item():.4f}',
            'Secret': f'{secret_loss.item():.4f}'
        })
    
    # Calculate averages
    avg_total = total_loss_sum / num_batches
    avg_cover = cover_loss_sum / num_batches
    avg_secret = secret_loss_sum / num_batches
    
    return avg_total, avg_cover, avg_secret


def save_example_outputs(encoder, decoder, dataloader, device, checkpoint_dir, num_examples=4):
    """
    Save example outputs for qualitative comparison.
    
    Saves comparison images showing:
    - Cover image (original)
    - Stego image (cover + hidden secret)
    - Secret image (original)
    - Recovered secret (decoded from stego)
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        dataloader: DataLoader to get samples from
        device: Target device
        checkpoint_dir: Directory to save examples
        num_examples: Number of example sets to save
    """
    encoder.eval()
    decoder.eval()
    
    examples_dir = Path(checkpoint_dir) / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {num_examples} example outputs...")
    
    # Get a batch of samples
    cover_batch, secret_batch = next(iter(dataloader))
    cover_batch = cover_batch[:num_examples].to(device)
    secret_batch = secret_batch[:num_examples].to(device)
    
    with torch.no_grad():
        # Generate stego images and recover secrets
        stego_batch = encoder(cover_batch, secret_batch)
        recovered_batch = decoder(stego_batch)
    
    # Denormalize images from [-1, 1] to [0, 1] for saving
    def denorm(x):
        return (x + 1) / 2
    
    for i in range(num_examples):
        # Save individual images
        save_image(denorm(cover_batch[i]), examples_dir / f'example_{i+1}_cover.png')
        save_image(denorm(stego_batch[i]), examples_dir / f'example_{i+1}_stego.png')
        save_image(denorm(secret_batch[i]).repeat(3, 1, 1), examples_dir / f'example_{i+1}_secret.png')
        save_image(denorm(recovered_batch[i]).repeat(3, 1, 1), examples_dir / f'example_{i+1}_recovered.png')
        
        # Save comparison grid (cover | stego | secret | recovered)
        # Expand grayscale to RGB for consistent grid
        secret_rgb = denorm(secret_batch[i]).repeat(3, 1, 1)
        recovered_rgb = denorm(recovered_batch[i]).repeat(3, 1, 1)
        
        comparison = torch.stack([
            denorm(cover_batch[i]),
            denorm(stego_batch[i]),
            secret_rgb,
            recovered_rgb
        ])
        save_image(comparison, examples_dir / f'example_{i+1}_comparison.png', nrow=4, padding=2)
    
    print(f"  Examples saved to: {examples_dir}")
    
    encoder.train()
    decoder.train()


def train(args):
    """
    Main training function.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("CNN-based Image Steganography Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = setup_device()
    
    # Create dataloader
    print("\nLoading dataset...")
    dataloader = create_dataloader(args)
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create models
    print("\nInitializing models...")
    encoder, decoder = create_models(args, device)
    
    # Create optimizer (joint training)
    # Combine parameters from both encoder and decoder
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(all_params, lr=args.lr)
    print(f"\nOptimizer: Adam (lr={args.lr})")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, encoder, decoder, optimizer, device
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)
        
        # Train for one epoch
        avg_total, avg_cover, avg_secret = train_epoch(
            encoder, decoder, dataloader, optimizer, device
        )
        
        # Log epoch results
        print(f"\n  Epoch Summary:")
        print(f"    Total Loss:  {avg_total:.6f}")
        print(f"    Cover Loss:  {avg_cover:.6f}")
        print(f"    Secret Loss: {avg_secret:.6f}")
        
        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            save_checkpoint(
                encoder, decoder, optimizer, epoch, avg_total,
                args.checkpoint_dir, filename='best_model.pth'
            )
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                encoder, decoder, optimizer, epoch, avg_total,
                args.checkpoint_dir
            )
    
    # Save final model
    save_checkpoint(
        encoder, decoder, optimizer, args.epochs, avg_total,
        args.checkpoint_dir, filename='final_model.pth'
    )
    
    # Save example outputs for qualitative comparison
    save_example_outputs(encoder, decoder, dataloader, device, args.checkpoint_dir, num_examples=4)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print("Example outputs saved to: {}/examples/".format(args.checkpoint_dir))
    print("\nCNN baseline is ready for GAN integration.")


if __name__ == '__main__':
    args = parse_args()
    train(args)
