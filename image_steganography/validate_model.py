"""
GAN Steganography Model Validation Script
Comprehensive evaluation of model performance with visual outputs.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from datetime import datetime
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.encoder_decoder import Encoder, Decoder

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoints/gan/best_gan_model.pth'
OUTPUT_DIR = Path('outputs/validation')
NUM_TEST_IMAGES = 4

# =============================================================================
# Helper Functions
# =============================================================================

def preprocess_cover(img_path, size=256):
    """Load and preprocess cover image."""
    img = Image.open(img_path).convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def preprocess_secret(img_path, size=256):
    """Load and preprocess grayscale secret."""
    img = Image.open(img_path).convert('L').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def tensor_to_image(tensor):
    """Convert tensor [-1,1] to numpy [0,255]."""
    arr = (tensor.squeeze().cpu().numpy() + 1) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return arr

def calc_psnr(img1, img2):
    """Calculate PSNR between two tensors."""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(4.0 / mse)

def calc_ssim(img1, img2):
    """Simplified SSIM calculation."""
    C1 = 0.01 ** 2 * 4
    C2 = 0.03 ** 2 * 4
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.item()

# =============================================================================
# Main Validation
# =============================================================================

def main():
    print("=" * 70)
    print("GAN STEGANOGRAPHY MODEL VALIDATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    cp = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    encoder = Encoder(base_channels=64).to(DEVICE).eval()
    decoder = Decoder(base_channels=64).to(DEVICE).eval()
    encoder.load_state_dict(cp['encoder_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    
    epoch = cp.get('epoch', '?')
    losses = cp.get('losses', {})
    mtime = datetime.fromtimestamp(Path(CHECKPOINT_PATH).stat().st_mtime)
    
    print(f"  Epoch: {epoch}")
    print(f"  Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
    if losses:
        print(f"  G Loss: {losses.get('g_loss', 'N/A'):.6f}")
        print(f"  Cover Loss: {losses.get('cover_loss', 'N/A'):.6f}")
        print(f"  Secret Loss: {losses.get('secret_loss', 'N/A'):.6f}")
    
    # Find test images
    celeba_dir = Path('data/celeba')
    cover_images = sorted(celeba_dir.glob('*.jpg'))[:NUM_TEST_IMAGES]
    
    # Find MNIST images for secrets
    mnist_dir = Path('data/fashion_mnist')
    if not mnist_dir.exists():
        mnist_dir = Path('data/mnist')
    
    # Create simple test secrets if MNIST not available
    secrets = []
    for i in range(NUM_TEST_IMAGES):
        secret = np.zeros((256, 256), dtype=np.float32)
        # Create digit-like patterns
        if i == 0:  # "5"
            secret[40:80, 60:180] = 1
            secret[80:140, 40:80] = 1
            secret[140:180, 60:180] = 1
            secret[180:220, 40:80] = 1
            secret[220:256, 60:180] = 1
        elif i == 1:  # "0"
            for y in range(60, 200):
                for x in range(80, 180):
                    if 60 <= ((x-130)**2 + (y-130)**2)**0.5 <= 80:
                        secret[y, x] = 1
        elif i == 2:  # "4"
            secret[40:160, 40:80] = 1
            secret[140:180, 40:220] = 1
            secret[40:220, 160:200] = 1
        else:  # "1"
            secret[40:220, 110:150] = 1
        
        secret = secret / 0.5 - 1.0  # Normalize to [-1, 1]
        secrets.append(torch.from_numpy(secret).unsqueeze(0).unsqueeze(0))
    
    print(f"\nTesting on {len(cover_images)} images...")
    
    # Metrics storage
    results = []
    
    for i, cover_path in enumerate(cover_images):
        cover = preprocess_cover(cover_path).to(DEVICE)
        secret = secrets[i].to(DEVICE)
        
        with torch.no_grad():
            stego = encoder(cover, secret)
            recovered = decoder(stego)
        
        # Calculate metrics
        cover_psnr = calc_psnr(cover, stego)
        cover_ssim = calc_ssim(cover, stego)
        secret_psnr = calc_psnr(secret, recovered)
        secret_ssim = calc_ssim(secret, recovered)
        max_diff = torch.abs(cover - stego).max().item()
        mean_diff = torch.abs(cover - stego).mean().item()
        
        results.append({
            'cover_psnr': cover_psnr,
            'cover_ssim': cover_ssim,
            'secret_psnr': secret_psnr,
            'secret_ssim': secret_ssim,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
        })
        
        # Save visual outputs
        def denorm(x):
            return (x.float() + 1) / 2
        
        save_image(denorm(cover), OUTPUT_DIR / f'test_{i+1}_cover.png')
        save_image(denorm(stego), OUTPUT_DIR / f'test_{i+1}_stego.png')
        save_image(denorm(secret).repeat(1, 3, 1, 1), OUTPUT_DIR / f'test_{i+1}_secret.png')
        save_image(denorm(recovered).repeat(1, 3, 1, 1), OUTPUT_DIR / f'test_{i+1}_recovered.png')
        
        # Difference map (amplified)
        diff = torch.abs(cover - stego).mean(dim=1, keepdim=True)
        diff_amp = torch.clamp(diff * 10, 0, 1)
        save_image(diff_amp, OUTPUT_DIR / f'test_{i+1}_diff_x10.png')
        
        print(f"\n  Image {i+1}: {cover_path.name}")
        print(f"    Cover PSNR:  {cover_psnr:.2f} dB")
        print(f"    Cover SSIM:  {cover_ssim:.4f}")
        print(f"    Secret PSNR: {secret_psnr:.2f} dB")
        print(f"    Max Diff:    {max_diff:.4f}")
    
    # Summary
    avg_cover_psnr = np.mean([r['cover_psnr'] for r in results])
    avg_cover_ssim = np.mean([r['cover_ssim'] for r in results])
    avg_secret_psnr = np.mean([r['secret_psnr'] for r in results])
    avg_max_diff = np.mean([r['max_diff'] for r in results])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Average Cover PSNR:  {avg_cover_psnr:.2f} dB")
    print(f"  Average Cover SSIM:  {avg_cover_ssim:.4f}")
    print(f"  Average Secret PSNR: {avg_secret_psnr:.2f} dB")
    print(f"  Average Max Diff:    {avg_max_diff:.4f}")
    
    # Quality Assessment
    print("\n" + "-" * 70)
    print("QUALITY ASSESSMENT")
    print("-" * 70)
    
    # Cover quality
    if avg_cover_psnr >= 40:
        cover_grade = "✅ EXCELLENT (nearly undetectable)"
    elif avg_cover_psnr >= 35:
        cover_grade = "⚠️ GOOD (minor artifacts)"
    elif avg_cover_psnr >= 30:
        cover_grade = "⚠️ FAIR (visible artifacts)"
    else:
        cover_grade = "❌ POOR (obvious artifacts)"
    
    # Secret quality
    if avg_secret_psnr >= 20:
        secret_grade = "✅ EXCELLENT (clear recovery)"
    elif avg_secret_psnr >= 15:
        secret_grade = "⚠️ GOOD (readable)"
    else:
        secret_grade = "❌ POOR (degraded)"
    
    print(f"\n  Cover Imperceptibility: {cover_grade}")
    print(f"  Secret Recovery:        {secret_grade}")
    
    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    
    if avg_cover_psnr >= 38 and avg_secret_psnr >= 18:
        print("\n  ✅ Model is READY for demo!")
    elif avg_cover_psnr < 35:
        print(f"\n  ⚠️ Cover PSNR ({avg_cover_psnr:.1f} dB) below target (40 dB)")
        print("     Consider more training epochs.")
    elif avg_secret_psnr < 15:
        print(f"\n  ⚠️ Secret PSNR ({avg_secret_psnr:.1f} dB) too low")
        print("     Increase λ_secret weight.")
    else:
        print("\n  ⚠️ Close to target. A few more epochs may help.")
    
    print(f"\n  Visual outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
