"""
Prepare Test Data for GAN vs CNN Evaluation

Generates aligned test datasets with STRICT SAMPLE ALIGNMENT:
- Same cover + same secret → CNN & GAN
- This ensures fair, valid comparison

Output Structure:
    outputs/evaluation/data/
    ├── cover/           # Original CelebA images
    ├── secret/          # Original MNIST secrets
    ├── cnn_stego/       # CNN-generated stego images
    ├── gan_stego/       # GAN-generated stego images
    ├── cnn_recovered/   # CNN-recovered secrets
    └── gan_recovered/   # GAN-recovered secrets
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.amp import autocast
from torchvision.utils import save_image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder
from data.dataset import SteganographyDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare test data for evaluation')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of test samples to generate')
    parser.add_argument('--cnn_checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to CNN model checkpoint')
    parser.add_argument('--gan_checkpoint', type=str, default='checkpoints/gan/best_gan_model.pth',
                        help='Path to GAN model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation/data',
                        help='Output directory for test data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--celeba_dir', type=str, default='data/celeba',
                        help='CelebA directory')
    parser.add_argument('--mnist_root', type=str, default='data/mnist',
                        help='MNIST root directory')
    return parser.parse_args()


def setup_device():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_model(checkpoint_path: str, device: torch.device, base_channels: int = 64):
    """Load encoder-decoder from checkpoint."""
    encoder = Encoder(base_channels=base_channels).to(device)
    decoder = Decoder(base_channels=base_channels).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch {epoch}")
    
    return encoder, decoder


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def generate_test_data(args):
    """Generate aligned test data for both CNN and GAN models."""
    print("=" * 60)
    print("Preparing Test Data for GAN vs CNN Evaluation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {args.seed}")
    print(f"Number of samples: {args.num_samples}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = setup_device()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    subdirs = ['cover', 'secret', 'cnn_stego', 'gan_stego', 'cnn_recovered', 'gan_recovered']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = SteganographyDataset(
        celeba_dir=args.celeba_dir,
        mnist_root=args.mnist_root,
        image_size=256,
        split='train',  # Using train split (could also use test split if available)
        max_samples=args.num_samples
    )
    
    # Load CNN model
    print(f"\nLoading CNN model: {args.cnn_checkpoint}")
    cnn_encoder, cnn_decoder = load_model(args.cnn_checkpoint, device)
    
    # Load GAN model
    print(f"\nLoading GAN model: {args.gan_checkpoint}")
    gan_encoder, gan_decoder = load_model(args.gan_checkpoint, device)
    
    # Generate data
    print(f"\nGenerating {args.num_samples} aligned test samples...")
    
    for i in tqdm(range(args.num_samples), desc="Generating test data"):
        # Get same cover and secret for both models
        cover, secret = dataset[i]
        cover = cover.unsqueeze(0).to(device)
        secret = secret.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # CNN forward pass
                cnn_stego = cnn_encoder(cover, secret)
                cnn_recovered = cnn_decoder(cnn_stego)
                
                # GAN forward pass (same cover, same secret)
                gan_stego = gan_encoder(cover, secret)
                gan_recovered = gan_decoder(gan_stego)
        
        # Convert to float32 for saving
        cover = cover.float()
        secret = secret.float()
        cnn_stego = cnn_stego.float()
        gan_stego = gan_stego.float()
        cnn_recovered = cnn_recovered.float()
        gan_recovered = gan_recovered.float()
        
        # Save images (denormalized to [0, 1])
        idx_str = f'{i:05d}'
        
        save_image(denormalize(cover), output_dir / 'cover' / f'{idx_str}.png')
        save_image(denormalize(secret).repeat(1, 3, 1, 1), output_dir / 'secret' / f'{idx_str}.png')
        save_image(denormalize(cnn_stego), output_dir / 'cnn_stego' / f'{idx_str}.png')
        save_image(denormalize(gan_stego), output_dir / 'gan_stego' / f'{idx_str}.png')
        save_image(denormalize(cnn_recovered).repeat(1, 3, 1, 1), output_dir / 'cnn_recovered' / f'{idx_str}.png')
        save_image(denormalize(gan_recovered).repeat(1, 3, 1, 1), output_dir / 'gan_recovered' / f'{idx_str}.png')
    
    # Save configuration for reproducibility
    config = {
        'num_samples': args.num_samples,
        'seed': args.seed,
        'cnn_checkpoint': args.cnn_checkpoint,
        'gan_checkpoint': args.gan_checkpoint,
        'celeba_dir': args.celeba_dir,
        'mnist_root': args.mnist_root,
        'timestamp': datetime.now().isoformat(),
        'image_size': 256,
        'normalization': '[-1, 1] -> [0, 1]'
    }
    
    results_dir = Path(args.output_dir).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    with open(results_dir / 'random_seed.txt', 'w') as f:
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    print("\n" + "=" * 60)
    print("Test Data Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Generated {args.num_samples} aligned sample triplets")
    print("\nDirectory structure:")
    for subdir in subdirs:
        count = len(list((output_dir / subdir).glob('*.png')))
        print(f"  {subdir}/: {count} images")
    print(f"\nConfig saved to: {results_dir / 'config_used.yaml'}")
    
    return output_dir


if __name__ == '__main__':
    args = parse_args()
    generate_test_data(args)
