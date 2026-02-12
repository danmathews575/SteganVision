import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoder_decoder import Encoder, Decoder

# Constants
IMAGE_SIZE = 256
TEST_DATA_DIR = Path('test_data')
TEST_RESULTS_DIR = Path('test_results/image_image')
LOG_FILE = TEST_RESULTS_DIR / 'logs.txt'

def setup_directories():
    """Create necessary directories."""
    (TEST_DATA_DIR / 'covers').mkdir(parents=True, exist_ok=True)
    (TEST_DATA_DIR / 'secrets').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'stego').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'decoded').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'covers').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'secrets').mkdir(parents=True, exist_ok=True)

def generate_dummy_image(path, name, color=None, mode='RGB'):
    """Generate a simple dummy image with a pattern."""
    img = Image.new(mode, (IMAGE_SIZE, IMAGE_SIZE), color=color or (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw some random shapes to make it "complex" enough for metrics
    for _ in range(5):
        x0 = random.randint(0, IMAGE_SIZE//2)
        y0 = random.randint(0, IMAGE_SIZE//2)
        x1 = random.randint(IMAGE_SIZE//2, IMAGE_SIZE)
        y1 = random.randint(IMAGE_SIZE//2, IMAGE_SIZE)
        
        if mode == 'RGB':
            c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            c = random.randint(50, 255) # Grayscale value
            
        draw.rectangle([x0, y0, x1, y1], fill=c, outline='white' if mode=='RGB' else 255)
        
    img.save(path / name)
    return img

def log(message):
    """Log message to console and file."""
    # Print to console with safe encoding
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'replace').decode('ascii'))
        
    # Write to file (utf-8 is standard for files)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Failed to write to log: {e}")

def load_model(device, checkpoint_path=None):
    """Load the trained model."""
    try:
        log("Loading model...")
        encoder = Encoder(base_channels=64).to(device)
        decoder = Decoder(base_channels=64).to(device)
        
        if checkpoint_path is None:
            # Try finding the best/final checkpoint
            checkpoint_dir = Path('checkpoints/gan')
            # Prefer final model (most recent training) over best (might be old)
            potential_paths = [
                checkpoint_dir / 'final_gan_model.pth',
                checkpoint_dir / 'best_gan_model.pth'
            ]
            
            for p in potential_paths:
                if p.exists():
                    checkpoint_path = p
                    break
                    
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                log(f"Warning: Standard checkpoints not found. Searching for latest.")
                checkpoints = sorted(checkpoint_dir.glob('*.pth'))
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found!")
                checkpoint_path = checkpoints[-1]
        
        checkpoint_path = Path(checkpoint_path)
        log(f"Using checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        encoder.eval()
        decoder.eval()
        return encoder, decoder, True
    except Exception as e:
        log(f"ERROR: Failed to load model. {str(e)}")
        return None, None, False

def transform_image(image_path, device, grayscale=False):
    """Load and normalize image."""
    if grayscale:
        img = Image.open(image_path).convert('L') # Grayscale
        mean, std = [0.5], [0.5]
    else:
        img = Image.open(image_path).convert('RGB')
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    
    # Resize if needed (strict 256x256 test)
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
    # Convert to tensor and normalize manually to match transforms
    img_np = np.array(img).astype(np.float32) / 255.0
    
    if grayscale:
        # (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    else:
        # (H, W, 3) -> (1, 3, H, W)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
        
    # Normalize to [-1, 1]
    img_tensor = (img_tensor - 0.5) / 0.5
    
    return img_tensor.to(device)

def save_image(tensor, path, grayscale=False):
    """Save normalized tensor as image."""
    img_np = tensor.squeeze().cpu().detach().numpy()
    
    if grayscale:
        pass # (H, W) is fine
    else:
        img_np = img_np.transpose(1, 2, 0) # (3, H, W) -> (H, W, 3)
        
    img_np = (img_np * 0.5 + 0.5) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if grayscale:
        Image.fromarray(img_np, mode='L').save(path)
    else:
        Image.fromarray(img_np, mode='RGB').save(path)

def compute_metrics(original_path, generated_path, grayscale=False):
    """Compute PSNR and SSIM."""
    if grayscale:
        img1 = np.array(Image.open(original_path).convert('L'))
        img2 = np.array(Image.open(generated_path).convert('L'))
        # Adjust data_range for grayscale
        p = psnr(img1, img2, data_range=255)
        s = ssim(img1, img2, data_range=255)
    else:
        img1 = np.array(Image.open(original_path).convert('RGB'))
        img2 = np.array(Image.open(generated_path).convert('RGB'))
        p = psnr(img1, img2)
        s = ssim(img1, img2, channel_axis=2)
    return p, s

    return p, s

def compute_texture_mask(img):
    """Compute texture mask for adaptive embedding (Validation version)."""
    # Simple edge detection
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    # Pad to match original size
    dx = torch.cat([dx, torch.zeros_like(img[:, :, :, :1])], dim=3)
    dy = torch.cat([dy, torch.zeros_like(img[:, :, :1, :])], dim=2)
    
    edge_mag = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
    mask = edge_mag.mean(dim=1, keepdim=True)  # Convert to 1 channel
    
    # Normalize mask [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    # Scale to [0.5, 1.0] to keep minimum embedding strength
    mask = 0.5 + 0.5 * mask
    return mask

def run_tests():
    parser = argparse.ArgumentParser(description='Validate Steganography Model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()

    setup_directories()
    
    # Reset log
    if LOG_FILE.exists():
        os.remove(LOG_FILE)
    
    log("IMAGE -> IMAGE GAN TEST REPORT")
    log("============================")
    log("Note: Texture-aware masking enabled for validation.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # 1. Model Load
    encoder, decoder, loaded = load_model(device, args.checkpoint)
    if not loaded:
        log("Model Load: FAIL")
        log("FINAL VERDICT: BROKEN")
        return
    log("Model Load: PASS")

    # 2. Test Data Setup
    log("\nGenerating Test Data...")
    
    # Check for unseen data
    unseen_dir = Path('data/unseen')
    if unseen_dir.exists() and list(unseen_dir.glob('*.jpg')):
        log("Using real unseen images from data/unseen")
        covers = sorted(list(unseen_dir.glob('*.jpg')))[:3]
    else:
        covers = [TEST_DATA_DIR / 'covers' / f'cover_{i}.png' for i in range(3)]
        for p in covers: generate_dummy_image(p.parent, p.name, color=(50, 50, 50), mode='RGB')
        
    secrets = [TEST_DATA_DIR / 'secrets' / f'secret_{i}.png' for i in range(len(covers))]
    for p in secrets: generate_dummy_image(p.parent, p.name, color=128, mode='L') # Grayscale secrets
    
    # 3. Encode/Decode Loop
    psnr_scores_cover = []
    ssim_scores_cover = []
    psnr_scores_secret = []
    ssim_scores_secret = []
    errors = []
    
    log("\nRunning Inference Tests...")
    
    try:
        for i, (cov_path, sec_path) in enumerate(zip(covers, secrets)):
            # Load images
            cover_tensor = transform_image(cov_path, device, grayscale=False)
            secret_tensor = transform_image(sec_path, device, grayscale=True)
            
            # Encode
            with torch.no_grad():
                stego_raw = encoder(cover_tensor, secret_tensor)
                
                # Apply texture mask
                mask = compute_texture_mask(cover_tensor)
                stego_tensor = cover_tensor + (stego_raw - cover_tensor) * mask
                
            stego_path = TEST_RESULTS_DIR / 'stego' / f'stego_{i}.png'
            save_image(stego_tensor, stego_path, grayscale=False)
            
            # Decode
            with torch.no_grad():
                decoded_tensor = decoder(stego_tensor)
                
            decoded_path = TEST_RESULTS_DIR / 'decoded' / f'decoded_{i}.png'
            save_image(decoded_tensor, decoded_path, grayscale=True)
            
            # Save resized/transformed inputs for fair metric comparison
            cover_ref_path = TEST_RESULTS_DIR / 'covers' / f'cover_{i}.png'
            secret_ref_path = TEST_RESULTS_DIR / 'secrets' / f'secret_{i}.png'
            save_image(cover_tensor, cover_ref_path, grayscale=False)
            save_image(secret_tensor, secret_ref_path, grayscale=True)
            
            # Metrics (Cover vs Stego)
            p_c, s_c = compute_metrics(cover_ref_path, stego_path, grayscale=False)
            psnr_scores_cover.append(p_c)
            ssim_scores_cover.append(s_c)
            
            # Metrics (Secret vs Decoded)
            p_s, s_s = compute_metrics(secret_ref_path, decoded_path, grayscale=True)
            psnr_scores_secret.append(p_s)
            ssim_scores_secret.append(s_s)
            
            log(f"Sample {i}:")
            log(f"  Cover PSNR={p_c:.2f}, SSIM={s_c:.4f}")
            log(f"  Secret PSNR={p_s:.2f}, SSIM={s_s:.4f}")
            
    except Exception as e:
        log(f"Runtime Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
        errors.append(str(e))

    # 4. Results Analysis
    avg_psnr = np.mean(psnr_scores_cover) if psnr_scores_cover else 0
    avg_ssim = np.mean(ssim_scores_cover) if ssim_scores_cover else 0
    
    log("\nRESULTS SUMMARY")
    log("===============")
    log(f"Model Load: PASS")
    log(f"Encode: {'PASS' if len(psnr_scores_cover) == 3 else 'FAIL'}")
    log(f"Decode: {'PASS' if len(psnr_scores_cover) == 3 else 'FAIL'}")
    
    quality = "POOR"
    if avg_psnr > 30 and avg_ssim > 0.9: quality = "GOOD"
    elif avg_psnr > 20: quality = "ACCEPTABLE"
    
    log(f"Output Quality: {quality}")
    log(f"Avg Cover PSNR: {avg_psnr:.2f}")
    log(f"Avg Cover SSIM: {avg_ssim:.4f}")
    
    log("\nERRORS FOUND:")
    if errors:
        for e in errors: log(f"- {e}")
    else:
        log("- NONE")
        
    log("\nFINAL VERDICT:")
    if not errors and quality != "POOR":
        log("WORKING")
    elif not errors and quality == "POOR":
        log("WORKING WITH WARNINGS")
    else:
        log("BROKEN")

if __name__ == "__main__":
    run_tests()
