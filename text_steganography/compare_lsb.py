"""
Comparison Script: Standard LSB vs AI-Guided LSB
Generates empirical metrics (PSNR, SSIM) for conference paper.
"""

import cv2
import numpy as np
import math
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Import AI-Guided implementation
import sys
sys.path.append(str(Path.cwd()))
from text_steganography.ai_guided_lsb.encoder import encode
from text_steganography.ai_guided_lsb.utils import text_to_bits

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def standard_lsb_encode(img, text):
    """Simple sequential LSB encoding (Top-left to Bottom-right)"""
    bits = text_to_bits(text)
    flat = img.flatten()
    
    if len(bits) > len(flat):
        raise ValueError("Text too long for image")
        
    encoded = flat.copy()
    encoded[:len(bits)] = (encoded[:len(bits)] & 0xFE) | bits
    
    return encoded.reshape(img.shape)

def main():
    print("="*60)
    print("TEXT STEGANOGRAPHY COMPARISON")
    print("="*60)
    
    # Setup paths
    base_dir = Path("text_steganography")
    cover_dir = base_dir / "test_data/cover"
    
    # Try to find a cover image
    cover_files = list(cover_dir.glob("*.png")) + list(cover_dir.glob("*.jpg"))
    if not cover_files:
        print("No cover images found!")
        # Create synthetic image if needed
        cover_path = base_dir / "synthetic_cover.png"
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img).save(cover_path)
    else:
        cover_path = cover_files[0]
        
    print(f"Using cover image: {cover_path}")
    
    # Secret text (Conference Abstract length)
    secret_text = "This is a test of the AI-Guided Steganography system. " * 50
    print(f"Secret Length: {len(secret_text)} characters")
    
    # Load original
    original = np.array(Image.open(cover_path).convert('RGB'))
    
    # 1. Run Standard LSB
    print("\nRunning Standard LSB...")
    stego_std = standard_lsb_encode(original, secret_text)
    
    # 2. Run AI-Guided LSB
    print("Running AI-Guided LSB...")
    ai_output_path = base_dir / "temp_ai_stego.png"
    success, msg = encode(
        cover_image_path=cover_path,
        text=secret_text,
        output_path=ai_output_path,
        importance_threshold=0.05  # Skip smooth areas
    )
    if not success:
        print(f"AI Encode Failed: {msg}")
        return
        
    stego_ai = np.array(Image.open(ai_output_path).convert('RGB'))
    
    # 3. Compute Metrics
    print("\nMETRICS COMPARISON:")
    print("-" * 50)
    
    # PSNR
    psnr_std = compute_psnr(original, stego_std)
    psnr_ai = compute_psnr(original, stego_ai)
    
    # SSIM
    ssim_std = ssim(original, stego_std, channel_axis=2)
    ssim_ai = ssim(original, stego_ai, channel_axis=2)
    
    print(f"{'Metric':<20} {'Standard LSB':>15} {'AI-Guided LSB':>15} {'Diff':>10}")
    print("-" * 65)
    print(f"{'PSNR (dB)':<20} {psnr_std:>15.2f} {psnr_ai:>15.2f} {psnr_ai-psnr_std:>+10.2f}")
    print(f"{'SSIM':<20} {ssim_std:>15.4f} {ssim_ai:>15.4f} {ssim_ai-ssim_std:>+10.4f}")
    
    print("\nobservation:")
    if psnr_ai > psnr_std:
        print("AI-Guided LSB has higher fidelity (PSNR).")
    else:
        print("Standard LSB has higher fidelity (PSNR) - this is expected if AI spreads bits more.")
        
    print("\nNote on Visual Quality:")
    print("Standard LSB modifies pixels sequentially (often visible in smooth top-left regions).")
    print("AI-Guided LSB modifies pixels in high-texture areas (harder to see).")
    print("="*60)

if __name__ == "__main__":
    main()
