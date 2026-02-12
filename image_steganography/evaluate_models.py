"""
GAN Model Evaluation Script
Compare checkpoints and measure metrics for steganography quality.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.encoder_decoder import Encoder, Decoder

def load_model(checkpoint_path, device):
    """Load encoder and decoder from checkpoint."""
    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    encoder.load_state_dict(cp['encoder_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, cp

def preprocess_image(img_path, size=256):
    """Load and preprocess image."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

def preprocess_secret(img_path, size=256):
    """Load and preprocess grayscale secret."""
    img = Image.open(img_path).convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor

def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(4.0 / mse)  # data_range = 2 for [-1,1], max diff = 2, so 2^2 = 4

def calculate_ssim(img1, img2):
    """Simplified SSIM calculation."""
    C1 = 0.01 ** 2 * 4  # (K1 * data_range)^2
    C2 = 0.03 ** 2 * 4
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()

def evaluate_model(encoder, decoder, cover_path, secret_path, device):
    """Evaluate a model on a single image pair."""
    cover = preprocess_image(cover_path).to(device)
    secret = preprocess_secret(secret_path).to(device)
    
    with torch.no_grad():
        stego = encoder(cover, secret)
        recovered = decoder(stego)
    
    # Metrics
    cover_psnr = calculate_psnr(cover, stego)
    cover_ssim = calculate_ssim(cover, stego)
    secret_psnr = calculate_psnr(secret, recovered)
    secret_ssim = calculate_ssim(secret, recovered)
    
    # Max difference (detectability)
    max_diff = torch.abs(cover - stego).max().item()
    mean_diff = torch.abs(cover - stego).mean().item()
    
    return {
        'cover_psnr': cover_psnr,
        'cover_ssim': cover_ssim,
        'secret_psnr': secret_psnr,
        'secret_ssim': secret_ssim,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Checkpoints to compare
    checkpoints = [
        ('final_gan_model.pth', 'checkpoints/gan/final_gan_model.pth'),
        ('best_gan_model.pth', 'checkpoints/gan/best_gan_model.pth'),
        ('interrupted_checkpoint.pth', 'checkpoints/gan/interrupted_checkpoint.pth'),
        ('epoch_0007', 'checkpoints/gan/gan_checkpoint_epoch_0007.pth'),
    ]
    
    # Test images
    test_cover = 'data/celeba/000001.jpg'  # Adjust if needed
    test_secret = 'data/mnist/MNIST/raw/t10k-images-idx3-ubyte'  # We'll create a simple test
    
    # Create a simple test secret
    secret_tensor = torch.randn(1, 1, 256, 256).to(device) * 0.5
    
    print("\n" + "=" * 80)
    print("GAN MODEL COMPARISON")
    print("=" * 80)
    
    for name, path in checkpoints:
        if not Path(path).exists():
            print(f"\n{name}: NOT FOUND")
            continue
            
        try:
            encoder, decoder, cp = load_model(path, device)
            epoch = cp.get('epoch', 'N/A')
            losses = cp.get('losses', {})
            
            # Use random test data for quick comparison
            cover_tensor = torch.randn(1, 3, 256, 256).to(device) * 0.5
            
            with torch.no_grad():
                stego = encoder(cover_tensor, secret_tensor)
                recovered = decoder(stego)
            
            # Metrics
            cover_psnr = calculate_psnr(cover_tensor, stego)
            cover_ssim = calculate_ssim(cover_tensor, stego)
            secret_psnr = calculate_psnr(secret_tensor, recovered)
            secret_ssim = calculate_ssim(secret_tensor, recovered)
            max_diff = torch.abs(cover_tensor - stego).max().item()
            
            print(f"\n{name}")
            print("-" * 40)
            print(f"  Epoch: {epoch}")
            print(f"  Losses: {losses}")
            print(f"  Cover PSNR: {cover_psnr:.2f} dB")
            print(f"  Cover SSIM: {cover_ssim:.4f}")
            print(f"  Secret PSNR: {secret_psnr:.2f} dB")
            print(f"  Secret SSIM: {secret_ssim:.4f}")
            print(f"  Max Diff: {max_diff:.4f}")
            
            # Quality assessment
            if cover_psnr > 40 and cover_ssim > 0.95:
                print(f"  Quality: ✅ EXCELLENT (undetectable)")
            elif cover_psnr > 35 and cover_ssim > 0.90:
                print(f"  Quality: ⚠️ GOOD (minor artifacts)")
            else:
                print(f"  Quality: ❌ NEEDS IMPROVEMENT")
                
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
