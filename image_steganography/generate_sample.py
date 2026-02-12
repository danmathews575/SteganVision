"""Generate sample output from latest GAN model"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from models.encoder_decoder import Encoder, Decoder

device = torch.device('cuda')
cp = torch.load('checkpoints/gan/interrupted_checkpoint.pth', map_location=device, weights_only=False)

encoder = Encoder(base_channels=64).to(device).eval()
decoder = Decoder(base_channels=64).to(device).eval()
encoder.load_state_dict(cp['encoder_state_dict'])
decoder.load_state_dict(cp['decoder_state_dict'])

print(f"Model: Epoch {cp.get('epoch', '?')}")

# Load test image
cover_path = list(Path('data/celeba').glob('*.jpg'))[0]
img = Image.open(cover_path).convert('RGB').resize((256, 256))
cover = torch.from_numpy(np.array(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

# Create secret pattern (digit "5")
secret = np.zeros((256, 256), dtype=np.float32)
secret[40:80, 60:180] = 1      # Top bar
secret[80:140, 40:80] = 1      # Left vertical
secret[140:180, 60:180] = 1    # Middle bar
secret[180:220, 160:200] = 1   # Right vertical
secret[220:256, 60:180] = 1    # Bottom bar
secret = torch.from_numpy(secret / 0.5 - 1).unsqueeze(0).unsqueeze(0).to(device)

# Generate stego
with torch.no_grad():
    stego = encoder(cover, secret)
    recovered = decoder(stego)

# Calculate metrics
mse = ((cover - stego) ** 2).mean().item()
psnr = 10 * np.log10(4.0 / mse)
max_diff = torch.abs(cover - stego).max().item()

print(f"Cover PSNR: {psnr:.2f} dB")
print(f"Max Diff: {max_diff:.4f}")

# Save outputs
def denorm(x):
    return ((x.squeeze().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

output_dir = Path('outputs/sample')
output_dir.mkdir(parents=True, exist_ok=True)

Image.fromarray(denorm(cover).transpose(1, 2, 0)).save(output_dir / 'cover.png')
Image.fromarray(denorm(stego).transpose(1, 2, 0)).save(output_dir / 'stego.png')
Image.fromarray(denorm(secret)).save(output_dir / 'secret.png')
Image.fromarray(denorm(recovered)).save(output_dir / 'recovered.png')

# Difference map
diff = torch.abs(cover - stego).mean(dim=1, keepdim=True)
diff_amp = (diff * 10).clamp(0, 1)
diff_img = (diff_amp.squeeze().cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(diff_img).save(output_dir / 'diff_x10.png')

# Create composite
from PIL import ImageDraw, ImageFont

composite = Image.new('RGB', (256*5, 256), (255, 255, 255))
composite.paste(Image.fromarray(denorm(cover).transpose(1, 2, 0)), (0, 0))
composite.paste(Image.fromarray(denorm(stego).transpose(1, 2, 0)), (256, 0))
composite.paste(Image.fromarray(diff_img).convert('RGB'), (512, 0))
composite.paste(Image.fromarray(denorm(secret)).convert('RGB'), (768, 0))
composite.paste(Image.fromarray(denorm(recovered)).convert('RGB'), (1024, 0))

# Add labels
draw = ImageDraw.Draw(composite)
labels = ['Cover', 'Stego', 'Diff x10', 'Secret', 'Recovered']
for i, label in enumerate(labels):
    draw.text((i*256 + 10, 10), label, fill=(255, 255, 0))

composite.save(output_dir / 'composite.png')
print(f"\nSaved to: {output_dir}")
print("Files: cover.png, stego.png, diff_x10.png, secret.png, recovered.png, composite.png")
