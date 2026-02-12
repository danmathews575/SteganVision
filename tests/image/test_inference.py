"""
Recreate the original test grid with the latest model
Uses the same 4 test images and MNIST secrets as the original test
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import os

# Fix path and encoding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

# Import from package
from image_steganography.src.models.encoder_decoder import Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use best_model.pth instead of deleted checkpoint
checkpoint_path = Path('image_steganography/checkpoints/best_model.pth')
if not checkpoint_path.exists():
    print(f"Skipping test: Checkpoint not found at {checkpoint_path}")
    sys.exit(0)

print(f"Loading checkpoint: {checkpoint_path}")
cp = torch.load(checkpoint_path, map_location=device, weights_only=False)

encoder = Encoder(base_channels=64).to(device).eval()
decoder = Decoder(base_channels=64).to(device).eval()
encoder.load_state_dict(cp['encoder_state_dict'])
decoder.load_state_dict(cp['decoder_state_dict'])

epoch = cp.get('epoch', '?')
print(f"Testing Model: Epoch {epoch}")

# Use the same 4 cover images (first 4 from CelebA)
data_dir = Path('image_steganography/data/celeba')
if not data_dir.exists():
    print(f"Skipping test: Data directory not found at {data_dir}")
    sys.exit(0)

cover_paths = sorted(data_dir.glob('*.jpg'))[:4]
if not cover_paths:
    print("No images found in data directory.")
    sys.exit(0)

# Create the same 4 MNIST-like secrets as original: 5, 0, 4, 1
def create_digit(digit_num):
    secret = np.zeros((256, 256), dtype=np.float32)
    if digit_num == 5:  # "5"
        secret[40:80, 60:180] = 1
        secret[80:140, 40:80] = 1
        secret[140:180, 60:180] = 1
        secret[180:220, 160:200] = 1
        secret[220:256, 60:180] = 1
    elif digit_num == 0:  # "0"
        for y in range(60, 200):
            for x in range(80, 180):
                dist = ((x-130)**2 + (y-130)**2)**0.5
                if 60 <= dist <= 80:
                    secret[y, x] = 1
    elif digit_num == 4:  # "4"
        secret[40:160, 40:80] = 1
        secret[140:180, 40:220] = 1
        secret[40:220, 160:200] = 1
    elif digit_num == 1:  # "1"
        secret[40:220, 110:150] = 1
    return torch.from_numpy(secret / 0.5 - 1).unsqueeze(0).unsqueeze(0).to(device)

secrets = [create_digit(d) for d in [5, 0, 4, 1]]

# Process all images
results = []
for i, (cover_path, secret) in enumerate(zip(cover_paths, secrets)):
    img = Image.open(cover_path).convert('RGB').resize((256, 256))
    cover = torch.from_numpy(np.array(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stego = encoder(cover, secret)
        recovered = decoder(stego)
    
    # Calculate metrics
    mse = ((cover - stego) ** 2).mean().item()
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(4.0 / mse)
    
    results.append({
        'cover': cover,
        'stego': stego,
        'secret': secret,
        'recovered': recovered,
        'psnr': psnr
    })
    print(f"Image {i+1}: PSNR = {psnr:.2f} dB")

# Create composite grid (same layout as original)
def denorm(x):
    return ((x.squeeze().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

# Create 5x4 grid
grid = Image.new('RGB', (256*5, 256*4 + 40), (255, 255, 255))
draw = ImageDraw.Draw(grid)

# Add title
draw.text((10, 10), f"GAN Steganography - Epoch {epoch} Model Output", fill=(0, 0, 0))

# Column headers
headers = ['Cover Image', 'Stego Image', '|Cover - Stego| x 10', 'Original Secret', 'Recovered Secret']
for col, header in enumerate(headers):
    draw.text((col*256 + 10, 25), header, fill=(0, 0, 0))

# Fill grid
for row, result in enumerate(results):
    y_offset = row * 256 + 40
    
    # Cover
    cover_img = Image.fromarray(denorm(result['cover']).transpose(1, 2, 0))
    grid.paste(cover_img, (0, y_offset))
    
    # Stego
    stego_img = Image.fromarray(denorm(result['stego']).transpose(1, 2, 0))
    grid.paste(stego_img, (256, y_offset))
    
    # Difference (amplified 10x)
    diff = torch.abs(result['cover'] - result['stego']).mean(dim=1, keepdim=True)
    diff_amp = (diff * 10).clamp(0, 1)
    diff_img = Image.fromarray((diff_amp.squeeze().cpu().numpy() * 255).astype(np.uint8))
    # Convert to RGB with purple colormap for comparison
    diff_rgb = Image.new('RGB', (256, 256))
    diff_arr = np.array(diff_img)
    diff_colored = np.zeros((256, 256, 3), dtype=np.uint8)
    diff_colored[:, :, 0] = diff_arr  # Red channel
    diff_colored[:, :, 2] = diff_arr  # Blue channel (makes purple)
    grid.paste(Image.fromarray(diff_colored), (512, y_offset))
    
    # Secret
    secret_img = Image.fromarray(denorm(result['secret']))
    grid.paste(secret_img.convert('RGB'), (768, y_offset))
    
    # Recovered
    recovered_img = Image.fromarray(denorm(result['recovered']))
    grid.paste(recovered_img.convert('RGB'), (1024, y_offset))

# Save
output_path = Path('tests/outputs/image/latest_model_result.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
grid.save(output_path)

avg_psnr = np.mean([r['psnr'] for r in results])
print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
print(f"Saved to: {output_path}")
