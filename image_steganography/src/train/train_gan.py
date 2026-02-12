"""
GAN Training Script for Image Steganography (GPU Optimized)

Optimized for NVIDIA RTX 3050 6GB VRAM with:
- Mixed Precision Training (AMP) with single shared GradScaler
- Gradient Accumulation (physical batch 4, effective batch 8)
- Memory-safe DataLoader configuration for Windows
- Proper GAN gradient control (freeze/unfreeze)
- Safe pause/resume with scaler state preservation

Usage:
    python src/train/train_gan.py --resume checkpoints/gan/best_gan_model.pth --epochs 15
    python src/train/train_gan.py --cnn_checkpoint checkpoints/best_model.pth --epochs 15
"""

import os
import sys
import gc
import signal
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision.utils import save_image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder_decoder import Encoder, Decoder
from models.discriminator import PatchDiscriminator
from data.dataset import SteganographyDataset
from utils.losses import (
    discriminator_loss, 
    generator_loss_gan,
    compute_cover_loss,
    compute_secret_loss
)


# =============================================================================
# Global Training State for Ctrl+C Handler
# =============================================================================

training_state = {
    'encoder': None,
    'decoder': None,
    'discriminator': None,
    'opt_g': None,
    'opt_d': None,
    'scaler': None,  # Single shared scaler
    'epoch': 0,
    'step': 0,
    'losses': {},
    'checkpoint_dir': 'checkpoints/gan',
}


def signal_handler(signum, frame):
    """Handle Ctrl+C by saving checkpoint before exiting."""
    print("\n\n" + "=" * 60)
    print("INTERRUPT DETECTED - Saving checkpoint...")
    print("=" * 60)
    
    if training_state['encoder'] is not None and training_state['epoch'] > 0:
        checkpoint_dir = training_state['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'interrupted_checkpoint.pth')
        
        checkpoint = {
            'epoch': training_state['epoch'],
            'step': training_state['step'],
            'encoder_state_dict': training_state['encoder'].state_dict(),
            'decoder_state_dict': training_state['decoder'].state_dict(),
            'discriminator_state_dict': training_state['discriminator'].state_dict(),
            'optimizer_g_state_dict': training_state['opt_g'].state_dict(),
            'optimizer_d_state_dict': training_state['opt_d'].state_dict(),
            'scaler_state_dict': training_state['scaler'].state_dict(),  # Save scaler state
            'losses': training_state['losses'],
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved to: {checkpoint_path}")
        print(f"Resume with: --resume {checkpoint_path}")
    else:
        print("No training progress to save.")
    
    print("\nExiting...")
    sys.exit(0)


# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# Utility Functions
# =============================================================================

def set_requires_grad(models, requires_grad):
    """
    Set requires_grad for all parameters in the given models.
    
    CRITICAL for GAN training:
    - Freeze G when training D to prevent unnecessary gradient computation
    - Freeze D when training G to avoid backward through D
    """
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GAN-based Image Steganography Model (GPU Optimized)'
    )
    
    # CNN checkpoint (required for fresh start)
    parser.add_argument('--cnn_checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to pretrained CNN checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of GAN training epochs (default: 3)')
    parser.add_argument('--physical_batch_size', type=int, default=4,
                        help='Physical batch size per GPU step (default: 4)')
    parser.add_argument('--accum_steps', type=int, default=2,
                        help='Gradient accumulation steps (default: 2, effective batch = 8)')
    parser.add_argument('--lr_g', type=float, default=5e-5,
                        help='Learning rate for generator (default: 5e-5)')
    parser.add_argument('--lr_d', type=float, default=5e-5,
                        help='Learning rate for discriminator (default: 5e-5)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 (default: 0.999)')
    
    # Loss weights
    parser.add_argument('--lambda_cover', type=float, default=1.5,
                        help='Weight for cover loss (default: 1.5)')
    parser.add_argument('--lambda_secret', type=float, default=1.0,
                        help='Weight for secret loss (default: 1.0)')
    parser.add_argument('--lambda_adv', type=float, default=0.003,
                        help='Weight for adversarial loss (default: 0.003)')
    parser.add_argument('--lambda_ssim', type=float, default=0.5,
                        help='Weight for SSIM loss (default: 0.5)')
    parser.add_argument('--loss_type', type=str, default='lsgan',
                        choices=['lsgan', 'bce'],
                        help='Adversarial loss type (default: lsgan)')
    
    # Data parameters
    parser.add_argument('--celeba_dir', type=str, default='data/celeba',
                        help='Path to CelebA images directory')
    parser.add_argument('--mnist_root', type=str, default='data/mnist',
                        help='Path to MNIST root directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size for testing')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for models')
    
    # Generalization & Robustness
    parser.add_argument('--cover_dirs', type=str, nargs='+', default=None,
                        help='List of additional directories for cover images')
    parser.add_argument('--secret_type', type=str, default='fashion_mnist',
                        help='Type of secret to use (mnist, fashion_mnist)')
    parser.add_argument('--noise_std', type=float, default=0.05,
                        help='Std dev of Gaussian noise injected before decoding (default: 0.05)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/gan',
                        help='Directory to save GAN checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to GAN checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Safety check: clamp lambda_adv
    if args.lambda_adv > 0.05:
        print(f"WARNING: lambda_adv={args.lambda_adv} exceeds max 0.05, clamping to 0.05")
        args.lambda_adv = 0.05
    
    # Compute effective batch size for logging
    args.effective_batch_size = args.physical_batch_size * args.accum_steps
    
    return args


def compute_texture_mask(img):
    """Compute texture mask for adaptive embedding."""
    # Simple edge detection
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    # Pad to match original size
    dx = torch.cat([dx, torch.zeros_like(img[:, :, :, :1])], dim=3)
    dy = torch.cat([dy, torch.zeros_like(img[:, :, :1, :])], dim=2)
    
    edge_mag = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
    mask = edge_mag.mean(dim=1, keepdim=True)  # Convert to 1 channel
    
    # Normalize mask [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    # Scale to [0.5, 1.0] to keep minimum embedding strength
    mask = 0.5 + 0.5 * mask
    return mask


def setup_device():
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Clear any cached memory from previous runs
        torch.cuda.empty_cache()
        gc.collect()
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU (training will be slow).")
    return device


def create_dataloader(args):
    """
    Create memory-safe training DataLoader.
    
    GPU OPTIMIZATION:
    - num_workers=0: Windows-safe, avoids multiprocessing memory issues
    - pin_memory=False: Reduces memory fragmentation on low-VRAM GPUs
    - persistent_workers=False: No worker processes to manage
    - drop_last=True: Ensures consistent batch sizes for gradient accumulation
    """
    dataset = SteganographyDataset(
        celeba_dir=args.celeba_dir,
        cover_dirs=args.cover_dirs,
        mnist_root=args.mnist_root,
        secret_type=args.secret_type,
        image_size=256,
        split='train',
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.physical_batch_size,
        shuffle=True,
        num_workers=0,              # Windows safe - no multiprocessing
        pin_memory=False,           # Avoid memory fragmentation
        drop_last=True,             # Consistent batch sizes for accumulation
        persistent_workers=False    # No worker processes
    )
    
    return dataloader


def load_pretrained_cnn(checkpoint_path, encoder, decoder, device):
    """Load pretrained CNN weights into encoder and decoder."""
    print(f"\nLoading pretrained CNN: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    print(f"  Loaded from epoch {epoch}, loss: {loss:.6f}" if isinstance(loss, float) else f"  Loaded from epoch {epoch}")
    
    return encoder, decoder


def load_gan_checkpoint(checkpoint_path, encoder, decoder, discriminator, opt_g, opt_d, scaler, device):
    """
    Load GAN checkpoint to resume training.
    
    GPU OPTIMIZATION:
    - Loads scaler state for proper AMP resume
    - Uses map_location to prevent VRAM spike
    """
    print(f"\nResuming from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GAN checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    # Load scaler state if available (for AMP resume)
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("  Loaded AMP scaler state")
    
    start_epoch = checkpoint['epoch'] + 1
    losses = checkpoint.get('losses', {})
    
    print(f"  Loaded epoch {checkpoint['epoch']}")
    if losses:
        print(f"  Previous G Loss: {losses.get('g_loss', 'N/A'):.6f}")
        print(f"  Previous D Loss: {losses.get('d_loss', 'N/A'):.6f}")
    
    return start_epoch, losses


def save_checkpoint(encoder, decoder, discriminator, opt_g, opt_d, scaler, epoch, step, losses, checkpoint_dir, filename=None):
    """
    Save GAN checkpoint with all states for safe resume.
    
    Includes:
    - Model weights (G & D)
    - Optimizer states
    - AMP scaler state
    - Current epoch and step
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'gan_checkpoint_epoch_{epoch:04d}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  # Save scaler for AMP resume
        'losses': losses,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


# =============================================================================
# Training Loop (GPU Optimized)
# =============================================================================

def train_epoch(encoder, decoder, discriminator, dataloader, opt_g, opt_d, scaler, device, args):
    """
    Train for one epoch with GPU optimizations.
    
    OPTIMIZATIONS:
    1. Mixed Precision (AMP) - Reduces VRAM by ~40%
    2. Gradient Accumulation - Simulates larger batch size
    3. Proper gradient control - Freezes D/G during respective steps
    4. Memory hygiene - Periodic cache clearing
    
    Training loop per accumulation cycle:
    1. Discriminator step: D(cover) vs D(stego.detach())
    2. Generator step: L1 losses + adversarial loss from D(stego)
    """
    encoder.train()
    decoder.train()
    discriminator.train()
    
    d_loss_sum = 0.0
    g_loss_sum = 0.0
    cover_loss_sum = 0.0
    secret_loss_sum = 0.0
    adv_loss_sum = 0.0
    ssim_loss_sum = 0.0
    num_updates = 0
    
    accum_steps = args.accum_steps
    
    # Zero gradients at start
    opt_d.zero_grad()
    opt_g.zero_grad()
    
    
    pbar = tqdm(dataloader, desc="GAN Training", leave=False)
    
    for i, (cover, secret) in enumerate(pbar):
        # Move to GPU (non_blocking for slight speedup)
        cover = cover.to(device, non_blocking=True)
        secret = secret.to(device, non_blocking=True)
        
        # =====================================================================
        # DISCRIMINATOR STEP
        # =====================================================================
        # Freeze Generator, Unfreeze Discriminator
        set_requires_grad([encoder, decoder], False)
        set_requires_grad([discriminator], True)
        
        # Generate stego WITHOUT gradients (saves memory)
        # CRITICAL: torch.no_grad() OUTSIDE autocast(device_type='cuda') for proper memory savings
        with torch.no_grad():
            with autocast(device_type='cuda'):
                stego_raw = encoder(cover, secret)
                
                # Texture-aware masking
                mask = compute_texture_mask(cover)
                stego_masked = cover + (stego_raw - cover) * mask
                stego_detached = stego_masked.detach()
                
                # Discriminator noise injection (lighter noise)
                noise_d = torch.randn_like(stego_detached) * 0.001
                stego_for_d = stego_detached + noise_d
        
        # D forward pass with AMP
        with autocast(device_type='cuda'):
            real_preds = discriminator(cover)
            fake_preds = discriminator(stego_for_d)
            
            # Scale loss for gradient accumulation
            d_loss = discriminator_loss(real_preds, fake_preds, loss_type=args.loss_type)
            d_loss_scaled = d_loss / accum_steps
        
        # Backward with gradient scaling
        scaler.scale(d_loss_scaled).backward()
        
        # =====================================================================
        # GENERATOR STEP
        # =====================================================================
        # Freeze Discriminator, Unfreeze Generator
        set_requires_grad([discriminator], False)
        set_requires_grad([encoder, decoder], True)
        
        # G forward pass with AMP
        with autocast(device_type='cuda'):
            stego_raw = encoder(cover, secret)
            
            # Texture-aware masking (differentiable)
            mask = compute_texture_mask(cover)
            stego = cover + (stego_raw - cover) * mask
            
            # Channel Energy Balancing Loss
            # Penalize mean shift per channel to prevent color bias
            balance_loss = (stego - cover).mean(dim=(2,3)).pow(2).mean()
            lambda_balance = 0.05
            
            # Robustness: Inject noise before decoding (stronger noise)
            # Use fixed sigma=0.002 as requested
            noise_g = torch.randn_like(stego) * 0.002
            stego_noisy = stego + noise_g
            recovered = decoder(stego_noisy)
            
            # Discriminator sees clean(er) stego for G feedback (or same 0.001 noise?)
            # User said: "Use noisy stego for the DECODER"
            # Standard practice: D sees same stego as it saw during D step, 
            # but usually G wants to fool D on the *actual* output.
            # Keeping it consistent with D training: inject 0.001 noise for D check too?
            # User said: "clean stego OR lightly noisy stego for the DISCRIMINATOR"
            # Let's use lightly noisy (0.001) to match D training distribution
            noise_for_d_check = torch.randn_like(stego) * 0.001
            stego_for_d_check = stego + noise_for_d_check
            
            fake_preds_for_g = discriminator(stego_for_d_check)
            
            # Generator loss (reconstruction + adversarial + SSIM)
            g_loss_base, c_loss, s_loss, a_loss, ssim_loss_val = generator_loss_gan(
                cover, stego, secret, recovered, fake_preds_for_g,
                lambda_cover=args.lambda_cover,
                lambda_secret=args.lambda_secret,
                lambda_adv=args.lambda_adv,
                lambda_ssim=args.lambda_ssim,
                loss_type=args.loss_type
            )
            
            # Add balance loss
            g_loss = g_loss_base + lambda_balance * balance_loss
            
            g_loss_scaled = g_loss / accum_steps
        
        # Backward with gradient scaling
        scaler.scale(g_loss_scaled).backward()
        
        # =====================================================================
        # OPTIMIZER STEP (every accum_steps iterations)
        # =====================================================================
        if (i + 1) % accum_steps == 0:
            # Step D optimizer
            scaler.step(opt_d)
            opt_d.zero_grad()
            
            # Step G optimizer
            scaler.step(opt_g)
            opt_g.zero_grad()
            
            # Update scaler ONCE after both optimizers (single shared scaler)
            scaler.update()
            
            # Accumulate losses (use unscaled values)
            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            cover_loss_sum += c_loss.item()
            secret_loss_sum += s_loss.item()
            adv_loss_sum += a_loss.item()
            ssim_loss_sum += ssim_loss_val.item()
            num_updates += 1
            
            # Update global state for Ctrl+C handler
            training_state['step'] = i
            training_state['losses'] = {
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'cover_loss': c_loss.item(),
                'secret_loss': s_loss.item(),
                'adv_loss': a_loss.item(),
                'ssim_loss': ssim_loss_val.item(),
            }
        
        # Update progress bar
        pbar.set_postfix({
            'D': f'{d_loss.item():.4f}',
            'G': f'{g_loss.item():.4f}',
            'Cov': f'{c_loss.item():.4f}',
            'Sec': f'{s_loss.item():.4f}'
        })
        
        # =====================================================================
        # MEMORY HYGIENE (every 100 iterations)
        # =====================================================================
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    # Handle remaining gradients if not divisible by accum_steps
    if (i + 1) % accum_steps != 0:
        scaler.step(opt_d)
        scaler.step(opt_g)
        scaler.update()
        opt_d.zero_grad()
        opt_g.zero_grad()
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Average losses
    if num_updates > 0:
        losses = {
            'd_loss': d_loss_sum / num_updates,
            'g_loss': g_loss_sum / num_updates,
            'cover_loss': cover_loss_sum / num_updates,
            'secret_loss': secret_loss_sum / num_updates,
            'adv_loss': adv_loss_sum / num_updates,
            'ssim_loss': ssim_loss_sum / num_updates,
        }
    else:
        losses = {'d_loss': 0, 'g_loss': 0, 'cover_loss': 0, 'secret_loss': 0, 'adv_loss': 0, 'ssim_loss': 0}
    
    return losses



def save_example_outputs(encoder, decoder, dataloader, device, checkpoint_dir, num_examples=10):
    """Save example outputs for qualitative comparison."""
    encoder.eval()
    decoder.eval()
    
    examples_dir = Path(checkpoint_dir) / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {num_examples} example outputs...")
    
    cover_batch, secret_batch = next(iter(dataloader))
    cover_batch = cover_batch[:num_examples].to(device)
    secret_batch = secret_batch[:num_examples].to(device)
    
    with torch.no_grad():
        with autocast(device_type='cuda'):
            stego_raw = encoder(cover_batch, secret_batch)
            
            # Apply texture mask for visualization
            mask = compute_texture_mask(cover_batch)
            stego_batch = cover_batch + (stego_raw - cover_batch) * mask
            
            recovered_batch = decoder(stego_batch)
    
    def denorm(x):
        return (x.float() + 1) / 2  # Ensure float for save_image
    
    for i in range(num_examples):
        save_image(denorm(cover_batch[i]), examples_dir / f'example_{i+1}_cover.png')
        save_image(denorm(stego_batch[i]), examples_dir / f'example_{i+1}_stego.png')
        save_image(denorm(secret_batch[i]).repeat(3, 1, 1), examples_dir / f'example_{i+1}_secret.png')
        save_image(denorm(recovered_batch[i]).repeat(3, 1, 1), examples_dir / f'example_{i+1}_recovered.png')
        
        # Difference maps
        diff = torch.abs(cover_batch[i].float() - stego_batch[i].float()).mean(dim=0, keepdim=True)
        
        # x1 Diff (Unamplified)
        save_image(diff, examples_dir / f'example_{i+1}_diff_x1.png')
        
        # x10 Diff (Amplified)
        diff_amplified = torch.clamp(diff * 10, 0, 1).repeat(3, 1, 1)
        save_image(diff_amplified, examples_dir / f'example_{i+1}_diff_x10.png')
    
    print(f"  Examples saved to: {examples_dir}")
    
    encoder.train()
    decoder.train()


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    """Main GAN training function with GPU optimizations."""
    print("=" * 60)
    print("GAN-based Image Steganography Training (GPU Optimized)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nGPU Optimizations:")
    print(f"  - Mixed Precision (AMP): Enabled")
    print(f"  - Physical batch size: {args.physical_batch_size}")
    print(f"  - Gradient accumulation: {args.accum_steps} steps")
    print(f"  - Effective batch size: {args.effective_batch_size}")
    print(f"\nLoss weights: λ_cover={args.lambda_cover}, λ_secret={args.lambda_secret}, λ_adv={args.lambda_adv}")
    print(f"Loss type: {args.loss_type.upper()}")
    
    # Setup device
    device = setup_device()
    
    # Create dataloader
    print("\nLoading dataset...")
    dataloader = create_dataloader(args)
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Optimizer steps per epoch: ~{len(dataloader) // args.accum_steps}")
    
    # Create models
    print("\nInitializing models...")
    encoder = Encoder(base_channels=args.base_channels).to(device)
    decoder = Decoder(base_channels=args.base_channels).to(device)
    discriminator = PatchDiscriminator(base_channels=args.base_channels).to(device)
    
    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  - Encoder: {enc_params:,}")
    print(f"  - Decoder: {dec_params:,}")
    print(f"  - Discriminator: {disc_params:,}")
    print(f"  - Total: {enc_params + dec_params + disc_params:,}")
    
    # Create optimizers
    g_params = list(encoder.parameters()) + list(decoder.parameters())
    opt_g = Adam(g_params, lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    # Create SINGLE shared scaler for AMP (NOT separate scalers)
    scaler = GradScaler()
    
    print(f"\nOptimizers:")
    print(f"  - Generator: Adam (lr={args.lr_g}, betas=({args.beta1}, {args.beta2}))")
    print(f"  - Discriminator: Adam (lr={args.lr_d}, betas=({args.beta1}, {args.beta2}))")
    print(f"  - AMP Scaler: Single shared GradScaler")
    
    # Resume from checkpoint or load pretrained CNN
    start_epoch = 1
    best_g_loss = float('inf')
    
    if args.resume:
        start_epoch, prev_losses = load_gan_checkpoint(
            args.resume, encoder, decoder, discriminator, opt_g, opt_d, scaler, device
        )
        if prev_losses and 'g_loss' in prev_losses:
            best_g_loss = prev_losses['g_loss']
    else:
        encoder, decoder = load_pretrained_cnn(args.cnn_checkpoint, encoder, decoder, device)
    
    # Store training state for Ctrl+C handler
    training_state['encoder'] = encoder
    training_state['decoder'] = decoder
    training_state['discriminator'] = discriminator
    training_state['opt_g'] = opt_g
    training_state['opt_d'] = opt_d
    training_state['scaler'] = scaler
    training_state['checkpoint_dir'] = args.checkpoint_dir
    
    # Training loop
    print("\n" + "=" * 60)
    print(f"{'Resuming' if args.resume else 'Starting'} GAN Training")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)
        
        # Update global state
        training_state['epoch'] = epoch
        
        # Train epoch
        losses = train_epoch(
            encoder, decoder, discriminator,
            dataloader, opt_g, opt_d, scaler, device, args
        )
        
        # Update training state
        training_state['losses'] = losses
        
        # Log epoch results
        print(f"\n  Epoch Summary:")
        print(f"    D Loss:      {losses['d_loss']:.6f}")
        print(f"    G Loss:      {losses['g_loss']:.6f}")
        print(f"    Cover Loss:  {losses['cover_loss']:.6f}")
        print(f"    Secret Loss: {losses['secret_loss']:.6f}")
        print(f"    Adv Loss:    {losses['adv_loss']:.6f}")
        
        # Report GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU Memory:  {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Save best model
        if losses['g_loss'] < best_g_loss:
            best_g_loss = losses['g_loss']
            save_checkpoint(
                encoder, decoder, discriminator, opt_g, opt_d, scaler,
                epoch, 0, losses, args.checkpoint_dir, filename='best_gan_model.pth'
            )
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                encoder, decoder, discriminator, opt_g, opt_d, scaler,
                epoch, 0, losses, args.checkpoint_dir
            )
    
    # Save final model
    save_checkpoint(
        encoder, decoder, discriminator, opt_g, opt_d, scaler,
        args.epochs, 0, losses, args.checkpoint_dir, filename='final_gan_model.pth'
    )
    
    # Save example outputs
    save_example_outputs(encoder, decoder, dataloader, device, args.checkpoint_dir, num_examples=4)
    
    print("\n" + "=" * 60)
    print("GAN Training Complete!")
    print(f"Best G Loss: {best_g_loss:.6f}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
