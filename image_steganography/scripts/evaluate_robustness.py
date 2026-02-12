import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder
from data.transforms import get_secret_transforms, get_cover_transforms

def compute_metrics(original, generated, grayscale=False):
    """Compute PSNR and SSIM."""
    # Convert tensors to numpy images [0, 255]
    if torch.is_tensor(original):
        original = original.squeeze().cpu().detach().numpy()
        generated = generated.squeeze().cpu().detach().numpy()
        
    # Denormalize [-1, 1] -> [0, 255]
    img1 = (original * 0.5 + 0.5) * 255.0
    img2 = (generated * 0.5 + 0.5) * 255.0
    
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    if grayscale:
        # (H, W) or (1, H, W) -> (H, W)
        if img1.ndim == 3: img1 = img1.squeeze()
        if img2.ndim == 3: img2 = img2.squeeze()
        data_range = 255
        p = psnr_metric(img1, img2, data_range=data_range)
        s = ssim_metric(img1, img2, data_range=data_range)
    else:
        # (3, H, W) -> (H, W, 3)
        if img1.shape[0] == 3:
            img1 = img1.transpose(1, 2, 0)
            img2 = img2.transpose(1, 2, 0)
        
        p = psnr_metric(img1, img2)
        s = ssim_metric(img1, img2, channel_axis=2)
        
    return p, s

def evaluate(args):
    print(f"Evaluating robustness on {args.unseen_dir}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    checkpoint_path = args.checkpoint or 'checkpoints/gan/best_gan_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    
    # Data Setup
    unseen_dir = Path(args.unseen_dir)
    cover_paths = sorted(list(unseen_dir.glob('*.jpg')) + list(unseen_dir.glob('*.png')))
    
    if not cover_paths:
        print("No images found in unseen dir.")
        return

    # FashionMNIST Test Set
    secret_dataset = datasets.FashionMNIST(
        root='data/mnist', train=False, download=True,
        transform=None # Manual transform
    )
    
    cover_transform = get_cover_transforms(image_size=256, split='test')
    secret_transform = get_secret_transforms(image_size=256, channels=1)
    
    metrics = {
        'cover_psnr': [], 'cover_ssim': [],
        'secret_psnr': [], 'secret_ssim': []
    }
    
    results_dir = Path('results/robustness')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, cover_path in enumerate(cover_paths):
            # Load Pair
            try:
                cover = Image.open(cover_path).convert('RGB')
                cover = cover_transform(cover).unsqueeze(0).to(device)
                
                secret_data, _ = secret_dataset[i % len(secret_dataset)]
                secret = secret_transform(secret_data).unsqueeze(0).to(device)
                
                # Inference
                stego = encoder(cover, secret)
                recovered = decoder(stego)
                
                # Metrics
                cp, cs = compute_metrics(cover, stego, grayscale=False)
                sp, ss = compute_metrics(secret, recovered, grayscale=True)
                
                metrics['cover_psnr'].append(cp)
                metrics['cover_ssim'].append(cs)
                metrics['secret_psnr'].append(sp)
                metrics['secret_ssim'].append(ss)
                
                print(f"Sample {i} ({cover_path.name}):")
                print(f"  Cover: {cp:.2f} dB / {cs:.4f}")
                print(f"  Secret: {sp:.2f} dB / {ss:.4f}")
                
            except Exception as e:
                print(f"Error processing {cover_path.name}: {e}")

    # Summary
    print("\n" + "="*40)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*40)
    print(f"Evaluated on {len(metrics['cover_psnr'])} samples.")
    print(f"Avg Cover PSNR:  {np.mean(metrics['cover_psnr']):.2f} dB")
    print(f"Avg Cover SSIM:  {np.mean(metrics['cover_ssim']):.4f}")
    print(f"Avg Secret PSNR: {np.mean(metrics['secret_psnr']):.2f} dB")
    print(f"Avg Secret SSIM: {np.mean(metrics['secret_ssim']):.4f}")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen_dir', type=str, default='data/unseen')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    evaluate(args)
