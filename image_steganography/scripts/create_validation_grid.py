import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

def create_grid():
    base_dir = Path('test_results/image_image')
    output_path = base_dir / 'validation_summary_grid.png'
    
    samples = [0, 1, 2]
    cols = ['Cover', 'Secret', 'Stego', 'Decoded']
    rows = len(samples)
    
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3*rows))
    fig.suptitle('Image GAN Steganography Validation Results', fontsize=16, y=1.02)
    
    for i, sample_idx in enumerate(samples):
        # Paths
        paths = [
            base_dir / f'covers/cover_{sample_idx}.png',
            base_dir / f'secrets/secret_{sample_idx}.png',
            base_dir / f'stego/stego_{sample_idx}.png',
            base_dir / f'decoded/decoded_{sample_idx}.png'
        ]
        
        for j, p in enumerate(paths):
            ax = axes[i, j]
            if p.exists():
                img = Image.open(p)
                ax.imshow(img, cmap='gray' if 'secret' in str(p) or 'decoded' in str(p) else None)
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center')
            
            ax.axis('off')
            
            # Column titles
            if i == 0:
                ax.set_title(cols[j], fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved grid to {output_path}")

if __name__ == '__main__':
    create_grid()
