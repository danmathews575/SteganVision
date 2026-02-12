import sys
import os
import torch
import argparse
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torch.amp import autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoder_decoder import Encoder, Decoder
from src.train.train_gan import create_dataloader, compute_texture_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    checkpoint_path = 'checkpoints/gan/interrupted_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/gan/best_gan_model.pth'
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    # Create dataloader
    class Args:
        celeba_dir = 'data/celeba'
        cover_dirs = None
        mnist_root = 'data/mnist'
        secret_type = 'fashion_mnist'
        physical_batch_size = 4
        max_samples = 100
    
    args = Args()
    dataloader = create_dataloader(args)
    
    # Get one batch
    cover, secret = next(iter(dataloader))
    cover = cover.to(device)
    secret = secret.to(device)
    
    batch_size = cover.size(0)
    print(f"Visualizing {batch_size} samples...")
    
    with torch.no_grad():
        with autocast(device_type='cuda'):
            stego_raw = encoder(cover, secret)
            
            # Apply masking
            mask = compute_texture_mask(cover)
            stego = cover + (stego_raw - cover) * mask
            
            recovered = decoder(stego)
            
            # Compute diffs
            diff = torch.abs(cover - stego)
            diff_amplified = torch.clamp(diff * 10, 0, 1)
            
            # Expand secret to 3 channels for grid
            secret_3ch = secret.repeat(1, 3, 1, 1)
            recovered_3ch = recovered.repeat(1, 3, 1, 1)
            
            # Mask visualization (1ch -> 3ch)
            mask_3ch = mask.repeat(1, 3, 1, 1)
            
    # Prepare grid
    # We want rows: [Cover, Stego, Diff x1, Diff x10, Secret, Recovered]
    # We have batches of each.
    
    # Denormalize images [-1, 1] -> [0, 1]
    def denorm(x):
        return (x + 1) / 2
    
    # Diffs and masks are already [0, 1] or positive.
    # Diff is absolute difference, so [0, 2] technically but small.
    # Mask is [0.5, 1].
    
    images_flat = []
    for i in range(batch_size):
        images_flat.extend([
            denorm(cover[i]),
            denorm(stego[i]),
            diff[i],                   # Diff x1
            diff_amplified[i],         # Diff x10
            mask_3ch[i],               # Texture Mask
            denorm(secret_3ch[i]),
            denorm(recovered_3ch[i])
        ])
    
    # Stack
    grid_tensor = torch.stack(images_flat)
    
    # Make grid (7 columns)
    grid = make_grid(grid_tensor, nrow=7, padding=2, normalize=False)
    
    output_path = 'results/final_grid.png'
    os.makedirs('results', exist_ok=True)
    save_image(grid, output_path)
    print(f"Grid saved to {output_path}")

if __name__ == "__main__":
    main()
