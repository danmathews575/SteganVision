import sys
import os
import torch
import glob
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid, save_image
from torch.amp import autocast
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoder_decoder import Encoder, Decoder
from src.train.train_gan import compute_texture_mask

def transform_image(path, device):
    img = Image.open(path).convert('RGB')
    img = img.resize((256, 256))
    img_np = np.array(img).astype(np.float32) / 255.0
    # (H, W, 3) -> (1, 3, H, W) -> Normalize [-1, 1]
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor.to(device)

def generate_secret(device):
    # Generate random secret
    secret = torch.rand(1, 1, 256, 256).to(device)
    # Normalize [-1, 1] (rand is [0, 1])
    secret = (secret - 0.5) / 0.5
    return secret

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    checkpoint_path = 'checkpoints/gan/final_gan_model.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/gan/best_gan_model.pth'
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    # Load unseen images
    unseen_dir = Path('data/unseen')
    image_paths = sorted(list(unseen_dir.glob('*.jpg')) + list(unseen_dir.glob('*.png')))
    
    if not image_paths:
        print("No images found in data/unseen")
        return

    print(f"Found {len(image_paths)} images: {[p.name for p in image_paths]}")
    
    images_flat = []
    
    # Denormalize images [-1, 1] -> [0, 1]
    def denorm(x):
        return (x + 1) / 2
        
    with torch.no_grad():
        for path in image_paths:
            cover = transform_image(path, device)
            secret = generate_secret(device)
            
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
            
            # Add to list
            images_flat.extend([
                denorm(cover[0]),
                denorm(stego[0]),
                diff[0],                   # Diff x1
                diff_amplified[0],         # Diff x10
                mask_3ch[0],               # Texture Mask
                denorm(secret_3ch[0]),
                denorm(recovered_3ch[0])
            ])
    
    # Stack
    grid_tensor = torch.stack(images_flat)
    
    # Make grid (7 columns)
    # Make grid (7 columns)
    grid = make_grid(grid_tensor, nrow=7, padding=2, normalize=False)
    
    # Add labels
    # Convert to PIL
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    # Create new image with header
    header_height = 50
    new_im = Image.new('RGB', (im.width, im.height + header_height), (255, 255, 255))
    new_im.paste(im, (0, header_height))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(new_im)
    
    # Try to load a font, otherwise default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    labels = ["Cover", "Stego", "Diff x1", "Diff x10", "Mask", "Secret", "Recovered"]
    col_width = im.width // 7
    
    for i, label in enumerate(labels):
        # Calculate centered position
        # bbox = draw.textbbox((0, 0), label, font=font) # standard for newer Pillow
        # text_width = bbox[2] - bbox[0]
        # For compatibility with older PIL if needed (though new environment likely has new PIL)
        text_width = draw.textlength(label, font=font)
        x = i * col_width + (col_width - text_width) / 2
        y = (header_height - 20) / 2 
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    
    output_path = 'results/unseen_grid.png'
    os.makedirs('results', exist_ok=True)
    new_im.save(output_path)
    print(f"Grid saved to {output_path}")

if __name__ == "__main__":
    main()
