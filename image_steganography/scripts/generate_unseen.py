import os
from pathlib import Path
from skimage import data
from PIL import Image
import numpy as np

def save_image(img, path):
    # Ensure rgb
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:,:,:3]
    Image.fromarray(img).save(path)

def main():
    output_dir = Path('data/unseen')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating unseen test images in {output_dir}...")

    try:
        images = {
            'astronaut': data.astronaut(),
            'coffee': data.coffee(),
            'cat': data.chelsea(),
            'rocket': data.rocket()
        }

        for name, img in images.items():
            save_image(img, output_dir / f'{name}.jpg')
            print(f"Saved {name}.jpg")
            
    except Exception as e:
        print(f"Error generating images: {e}")
        # Fallback if skimage data not available
        print("Creating random noise images instead...")
        for name in ['noise1', 'noise2', 'noise3']:
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            save_image(img, output_dir / f'{name}.jpg')

if __name__ == '__main__':
    main()
