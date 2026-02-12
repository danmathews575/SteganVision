import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import datasets
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder
from data.transforms import get_secret_transforms, get_cover_transforms

def denorm(tensor):
    """Denormalize tensor [-1, 1] -> [0, 1] for plotting."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    encoder = Encoder(base_channels=64).to(device)
    decoder = Decoder(base_channels=64).to(device)
    
    checkpoint_path = 'checkpoints/gan/interrupted_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        # Fallback to best if interrupted doesn't exist (though it should)
        checkpoint_path = 'checkpoints/gan/best_gan_model.pth'

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    # Data
    unseen_dir = Path('data/unseen')
    cover_paths = sorted(list(unseen_dir.glob('*.jpg')) + list(unseen_dir.glob('*.png')))[:4] # Take top 4
    
    if not cover_paths:
        print("No images found in data/unseen")
        return

    # FashionMNIST Secret
    secret_dataset = datasets.FashionMNIST(
        root='data/mnist', train=False, download=True, transform=None
    )
    
    cover_transform = get_cover_transforms(image_size=256, split='test')
    secret_transform = get_secret_transforms(image_size=256, channels=1)

    # Plotting
    num_samples = len(cover_paths)
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    cols = ['Cover', 'Secret', 'Stego', 'Recovered', 'Stego Diff (x10)', 'Sec Diff']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, pad=10)

    print("Generating grid...")
    
    with torch.no_grad():
        for i, cover_path in enumerate(cover_paths):
            # Prepare Inputs
            cover_pil = Image.open(cover_path).convert('RGB')
            cover = cover_transform(cover_pil).unsqueeze(0).to(device)
            
            secret_data, _ = secret_dataset[i % len(secret_dataset)]
            secret = secret_transform(secret_data).unsqueeze(0).to(device)
            
            # Inference
            stego = encoder(cover, secret)
            recovered = decoder(stego)
            
            # Post-process for display
            cover_img = denorm(cover).squeeze().permute(1, 2, 0).cpu().numpy()
            secret_img = denorm(secret).squeeze().cpu().numpy()
            stego_img = denorm(stego).squeeze().permute(1, 2, 0).cpu().numpy()
            recovered_img = denorm(recovered).squeeze().cpu().numpy()
            
            # Differences
            stego_diff = np.abs(cover_img - stego_img)
            stego_diff_viz = np.clip(stego_diff * 10, 0, 1) # Amplify x10
            
            sec_diff = np.abs(secret_img - recovered_img)
            
            # Plot
            axes[i, 0].imshow(cover_img)
            axes[i, 1].imshow(secret_img, cmap='gray')
            axes[i, 2].imshow(stego_img)
            axes[i, 3].imshow(recovered_img, cmap='gray')
            axes[i, 4].imshow(stego_diff_viz)
            axes[i, 5].imshow(sec_diff, cmap='inferno')
            
            for ax in axes[i]:
                ax.axis('off')
                
            # Add row label
            axes[i, 0].text(-0.1, 0.5, cover_path.stem, transform=axes[i, 0].transAxes, 
                           va='center', ha='right', fontsize=10, rotation=90)

    output_path = 'results/robustness_grid.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Grid saved to {output_path}")

if __name__ == '__main__':
    main()
