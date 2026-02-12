"""
Quick GAN Model Comparison - Uses actual CelebA images
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.encoder_decoder import Encoder, Decoder
from utils.advanced_postprocess import perfect_clean_secret

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Find test images
celeba_dir = Path('data/celeba')
celeba_images = list(celeba_dir.glob('*.jpg'))[:4]
print(f"Found {len(celeba_images)} test images")

def preprocess_cover(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

def preprocess_secret():
    # Simple MNIST-like pattern
    secret = np.zeros((256, 256), dtype=np.float32)
    secret[50:200, 80:180] = 1.0  # White rectangle
    secret = secret / 0.5 - 1.0  # Normalize to [-1, 1]
    return torch.from_numpy(secret).unsqueeze(0).unsqueeze(0).to(device)

def calc_psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(4.0 / mse)

checkpoints = [
    ('final_gan_model', 'checkpoints/gan/final_gan_model.pth'),
    ('best_gan_model', 'checkpoints/gan/best_gan_model.pth'),  
    ('interrupted', 'checkpoints/gan/interrupted_checkpoint.pth'),
]

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = {}

for name, path in checkpoints:
    if not Path(path).exists():
        print(f"{name}: NOT FOUND")
        continue
    
    # Load model
    cp = torch.load(path, map_location=device, weights_only=False)
    encoder = Encoder(base_channels=64).to(device).eval()
    decoder = Decoder(base_channels=64).to(device).eval()
    encoder.load_state_dict(cp['encoder_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    
    epoch = cp.get('epoch', '?')
    mtime = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    
    # Test on real images
    cover_psnrs = []
    secret_psnrs = []
    max_diffs = []
    
    secret = preprocess_secret()
    
    for img_path in celeba_images:
        cover = preprocess_cover(img_path)
        
        with torch.no_grad():
            stego = encoder(cover, secret)
            recovered_raw = decoder(stego)
            # Apply advanced post-processing for perfect clean recovery
            recovered = perfect_clean_secret(recovered_raw, aggressive=True)
        
        cover_psnrs.append(calc_psnr(cover, stego))
        secret_psnrs.append(calc_psnr(secret, recovered))
        max_diffs.append(torch.abs(cover - stego).max().item())
    
    avg_cover_psnr = np.mean(cover_psnrs)
    avg_secret_psnr = np.mean(secret_psnrs)
    avg_max_diff = np.mean(max_diffs)
    
    results[name] = {
        'epoch': epoch,
        'cover_psnr': avg_cover_psnr,
        'secret_psnr': avg_secret_psnr,
        'max_diff': avg_max_diff,
    }
    
    # Quality assessment
    if avg_cover_psnr > 40:
        quality = "✅ EXCELLENT"
    elif avg_cover_psnr > 35:
        quality = "⚠️ GOOD"
    elif avg_cover_psnr > 30:
        quality = "⚠️ FAIR"
    else:
        quality = "❌ POOR"
    
    print(f"\n{name} (epoch {epoch}, {mtime})")
    print(f"  Cover PSNR:  {avg_cover_psnr:.2f} dB  {quality}")
    print(f"  Secret PSNR: {avg_secret_psnr:.2f} dB")
    print(f"  Max Diff:    {avg_max_diff:.4f}")

# Summary
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

best = max(results.items(), key=lambda x: x[1]['cover_psnr'])
print(f"\nBest model: {best[0]} (Cover PSNR: {best[1]['cover_psnr']:.2f} dB)")

if best[1]['cover_psnr'] < 40:
    print("\n⚠️ ADDITIONAL TRAINING RECOMMENDED")
    print("   Current PSNR < 40 dB means artifacts may be visible.")
    print("   Target: PSNR > 40 dB for undetectable steganography.")
else:
    print("\n✅ Model quality is sufficient for demo.")
