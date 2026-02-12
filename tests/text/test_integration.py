#!/usr/bin/env python3
"""
Text Steganography Demo Test Script

Comprehensive demo testing with realistic secret codes, passwords, API keys,
and various code formats suitable for presentation purposes.

All credentials and codes are FAKE and for demonstration only.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Fix path and encoding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

# Import from package
try:
    from text_steganography.ai_guided_lsb.encoder import encode
    from text_steganography.ai_guided_lsb.decoder import decode
    from text_steganography.ai_guided_lsb.utils import calculate_capacity
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from text_steganography.ai_guided_lsb.encoder import encode
    from text_steganography.ai_guided_lsb.decoder import decode
    from text_steganography.ai_guided_lsb.utils import calculate_capacity


# Directories
BASE_DIR = Path('text_steganography')
TEST_DATA_DIR = BASE_DIR / 'test_data'
DEMO_RESULTS_DIR = Path('tests/outputs/text/demo')
LOG_FILE = DEMO_RESULTS_DIR / 'demo_test_log.txt'


def setup_directories():
    """Create test directories if they don't exist."""
    (TEST_DATA_DIR / 'cover').mkdir(parents=True, exist_ok=True)
    (DEMO_RESULTS_DIR / 'stego').mkdir(parents=True, exist_ok=True)
    (DEMO_RESULTS_DIR / 'decoded').mkdir(parents=True, exist_ok=True)
    (DEMO_RESULTS_DIR / 'comparisons').mkdir(parents=True, exist_ok=True)


def log(message):
    """Log to console and file."""
    print(message)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except:
        pass


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def create_comparison_image(cover_path, stego_path, output_path, test_name):
    """Create side-by-side comparison of cover and stego images."""
    try:
        cover = Image.open(cover_path)
        stego = Image.open(stego_path)
        
        # Resize if too large
        max_width = 400
        if cover.width > max_width:
            ratio = max_width / cover.width
            new_size = (max_width, int(cover.height * ratio))
            cover = cover.resize(new_size, Image.Resampling.LANCZOS)
            stego = stego.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create comparison image
        width = cover.width * 2 + 20
        height = cover.height + 60
        comparison = Image.new('RGB', (width, height), color=(255, 255, 255))
        
        # Paste images
        comparison.paste(cover, (0, 50))
        comparison.paste(stego, (cover.width + 20, 50))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((cover.width // 2 - 50, 10), "Cover Image", fill=(0, 0, 0), font=font)
        draw.text((cover.width + cover.width // 2 - 30, 10), "Stego Image", fill=(0, 0, 0), font=font)
        
        comparison.save(output_path)
        return True
    except Exception as e:
        log(f"Warning: Could not create comparison image: {e}")
        return False


def run_demo_test(name, secret_text, cover_path, description=""):
    """Run a single demo test case."""
    log(f"\n{'='*70}")
    log(f"TEST: {name}")
    log(f"{'='*70}")
    if description:
        log(f"Description: {description}")
    log(f"Secret length: {len(secret_text)} characters")
    log(f"Secret preview: {secret_text[:80]}{'...' if len(secret_text) > 80 else ''}")
    
    stego_path = DEMO_RESULTS_DIR / 'stego' / f'{name}.png'
    decoded_path = DEMO_RESULTS_DIR / 'decoded' / f'{name}.txt'
    comparison_path = DEMO_RESULTS_DIR / 'comparisons' / f'{name}_comparison.png'
    
    result = {
        'name': name,
        'description': description,
        'secret_length': len(secret_text),
        'success': False,
        'accuracy': 0.0,
        'psnr': 0.0,
        'encode_time': 0.0,
        'decode_time': 0.0,
        'error': None
    }
    
    try:
        # Encode
        start_time = time.time()
        success, message = encode(cover_path, secret_text, stego_path)
        encode_time = time.time() - start_time
        result['encode_time'] = encode_time
        
        if not success:
            log(f"❌ Encode FAILED: {message}")
            result['error'] = message
            return result
        
        log(f"✅ Encode SUCCESS ({encode_time:.3f}s)")
        
        # Calculate PSNR
        cover_img = Image.open(cover_path)
        stego_img = Image.open(stego_path)
        psnr = calculate_psnr(cover_img, stego_img)
        result['psnr'] = psnr
        log(f"📊 PSNR: {psnr:.2f} dB {'(EXCELLENT ✨)' if psnr > 50 else '(GOOD ✓)' if psnr > 40 else '(ACCEPTABLE)'}")
        
        # Decode
        start_time = time.time()
        success, decoded_text, message = decode(stego_path)
        decode_time = time.time() - start_time
        result['decode_time'] = decode_time
        
        if not success:
            log(f"❌ Decode FAILED: {message}")
            result['error'] = message
            return result
        
        log(f"✅ Decode SUCCESS ({decode_time:.3f}s)")
        
        # Save decoded text
        with open(decoded_path, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        
        # Check accuracy
        if decoded_text == secret_text:
            result['accuracy'] = 100.0
            result['success'] = True
            log(f"🎯 Accuracy: 100% EXACT MATCH ✓✓✓")
        else:
            matches = sum(1 for a, b in zip(decoded_text, secret_text) if a == b)
            accuracy = (matches / max(len(secret_text), len(decoded_text))) * 100
            result['accuracy'] = accuracy
            log(f"⚠️  Accuracy: {accuracy:.2f}% MISMATCH")
            log(f"Expected length: {len(secret_text)}, Got: {len(decoded_text)}")
            result['error'] = "Text mismatch"
        
        # Create comparison image
        create_comparison_image(cover_path, stego_path, comparison_path, name)
        
    except Exception as e:
        log(f"❌ ERROR: {str(e)}")
        log(traceback.format_exc())
        result['error'] = str(e)
    
    return result


def get_demo_test_cases():
    """Define all demo test cases with realistic secrets."""
    
    test_cases = []
    
    # 1. API Keys & Tokens
    test_cases.append((
        "api_key_openai",
        "sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghijklmnopqr",
        "OpenAI API Key"
    ))
    
    return test_cases


def generate_cover_image(name, width=512, height=512):
    """Generate a cover image for testing."""
    import random
    
    img = Image.new('RGB', (width, height), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # Add random colored rectangles
    for _ in range(40):
        x1 = random.randint(0, width - 50)
        y1 = random.randint(0, height - 50)
        x2 = x1 + random.randint(30, 100)
        y2 = y1 + random.randint(30, 100)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
    
    # Add gradient
    for i in range(0, width, 5):
        gray = int(255 * i / width)
        draw.line([(i, 0), (i, 80)], fill=(gray, gray, gray), width=5)
    
    # Add some circles
    for _ in range(20):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        r = random.randint(10, 40)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    path = TEST_DATA_DIR / 'cover' / f'{name}.png'
    img.save(path)
    return path


def main():
    """Run all demo tests."""
    setup_directories()
    
    # Clear log file
    if LOG_FILE.exists():
        os.remove(LOG_FILE)
    
    log("=" * 70)
    log("TEXT STEGANOGRAPHY - DEMO TEST SUITE")
    log("=" * 70)
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Method: AI-Guided Adaptive LSB Text Steganography")
    log(f"Purpose: Demo testing with realistic secret codes and passwords")
    log("")
    log("⚠️  NOTE: All credentials and codes are FAKE for demo purposes only!")
    log("=" * 70)
    
    # Generate cover image
    log("\n📸 Generating cover image...")
    cover_path = generate_cover_image("demo_cover", 512, 512)
    log(f"✅ Cover image created: {cover_path}")
    
    # Calculate capacity
    cover_img = Image.open(cover_path)
    capacity = calculate_capacity(np.array(cover_img).shape)
    log(f"📊 Image capacity: ~{capacity} characters")
    
    # Get test cases
    test_cases = get_demo_test_cases()
    log(f"\n🧪 Total test cases: {len(test_cases)}")
    
    # Run tests
    results = []
    for i, (name, secret, description) in enumerate(test_cases, 1):
        log(f"\n\n{'#'*70}")
        log(f"TEST {i}/{len(test_cases)}")
        log(f"{'#'*70}")
        result = run_demo_test(name, secret, cover_path, description)
        results.append(result)
    
    # Generate summary
    log("\n\n" + "=" * 70)
    log("DEMO TEST SUMMARY")
    log("=" * 70)
    
    total = len(results)
    success_count = sum(1 for r in results if r['success'])
    avg_psnr = np.mean([r['psnr'] for r in results if r['psnr'] > 0])
    avg_encode_time = np.mean([r['encode_time'] for r in results if r['encode_time'] > 0])
    avg_decode_time = np.mean([r['decode_time'] for r in results if r['decode_time'] > 0])
    
    log(f"\n📊 Overall Statistics:")
    log(f"  Total Tests: {total}")
    log(f"  Successful: {success_count} ({success_count/total*100:.1f}%)")
    log(f"  Failed: {total - success_count}")
    log(f"  Average PSNR: {avg_psnr:.2f} dB")
    log(f"  Average Encode Time: {avg_encode_time:.3f}s")
    log(f"  Average Decode Time: {avg_decode_time:.3f}s")
    
    # Detailed results table
    log("\n" + "-" * 70)
    log("DETAILED RESULTS")
    log("-" * 70)
    log(f"{'Test Name':<25} {'Status':<10} {'Accuracy':<12} {'PSNR':<10} {'Time':<10}")
    log("-" * 70)
    
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        accuracy = f"{r['accuracy']:.1f}%" if r['accuracy'] > 0 else "N/A"
        psnr = f"{r['psnr']:.1f}dB" if r['psnr'] > 0 else "N/A"
        total_time = r['encode_time'] + r['decode_time']
        time_str = f"{total_time:.3f}s" if total_time > 0 else "N/A"
        
        log(f"{r['name']:<25} {status:<10} {accuracy:<12} {psnr:<10} {time_str:<10}")
    
    log("-" * 70)
    
    # Final verdict
    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    
    if success_count == total:
        log("🎉 ✅ ALL TESTS PASSED - DEMO READY!")
    else:
        log("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    
    log("\n" + "=" * 70)
    log(f"📄 Full log saved to: {LOG_FILE}")
    log("=" * 70)
    
    return success_count == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
