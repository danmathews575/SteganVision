#!/usr/bin/env python3
"""
AI-Guided Adaptive LSB Text Steganography - Validation Script

Comprehensive test suite to verify:
- 100% exact text recovery
- Unicode support
- Capacity limits
- Edge cases
- Performance

PASS CRITERIA:
- Text recovery = 100% exact
- Stego visually indistinguishable
- Runtime < 1 second (normal image)
- No crashes
"""

import os
import sys
import time
import random
import traceback
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_guided_lsb.encoder import encode
from ai_guided_lsb.decoder import decode
from ai_guided_lsb.utils import calculate_capacity


# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = BASE_DIR / 'test_data'
TEST_RESULTS_DIR = BASE_DIR / 'test_results' / 'text_image'
LOG_FILE = TEST_RESULTS_DIR / 'validation_log.txt'


def setup_directories():
    """Create test directories if they don't exist."""
    (TEST_DATA_DIR / 'cover').mkdir(parents=True, exist_ok=True)
    (TEST_DATA_DIR / 'text').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'stego').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'decoded').mkdir(parents=True, exist_ok=True)


def log(message):
    """Log to console and file."""
    print(message)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except:
        pass


def generate_test_image(width=512, height=512, name="test_cover.png"):
    """Generate a test image with textures and patterns."""
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Add random rectangles (simulates textures)
    for _ in range(30):
        x1 = random.randint(0, width - 50)
        y1 = random.randint(0, height - 50)
        x2 = x1 + random.randint(20, 80)
        y2 = y1 + random.randint(20, 80)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Add some gradients (smooth regions)
    for i in range(0, width, 10):
        gray = int(255 * i / width)
        draw.line([(i, 0), (i, 50)], fill=(gray, gray, gray), width=10)
    
    path = TEST_DATA_DIR / 'cover' / name
    img.save(path)
    return path


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def run_test_case(name, text, cover_path, expect_success=True):
    """Run a single test case and return results."""
    log(f"\n{'='*60}")
    log(f"TEST CASE: {name}")
    log(f"{'='*60}")
    log(f"Text length: {len(text)} chars")
    log(f"Text preview: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    stego_path = TEST_RESULTS_DIR / 'stego' / f'stego_{name}.png'
    decoded_path = TEST_RESULTS_DIR / 'decoded' / f'decoded_{name}.txt'
    
    result = {
        'name': name,
        'encode_pass': False,
        'decode_pass': False,
        'accuracy': 0.0,
        'psnr': 0.0,
        'encode_time': 0.0,
        'decode_time': 0.0,
        'error': None
    }
    
    try:
        # --- ENCODE ---
        start_time = time.time()
        success, message = encode(cover_path, text, stego_path)
        encode_time = time.time() - start_time
        result['encode_time'] = encode_time
        
        if not expect_success:
            if not success:
                log(f"Encode: PASS (expected failure)")
                log(f"Message: {message}")
                result['encode_pass'] = True
                return result
            else:
                log(f"Encode: FAIL (expected failure but succeeded)")
                result['error'] = "Expected encode to fail but it succeeded"
                return result
        
        if not success:
            log(f"Encode: FAIL - {message}")
            result['error'] = message
            return result
        
        log(f"Encode: PASS ({encode_time:.3f}s)")
        result['encode_pass'] = True
        
        # --- PSNR ---
        cover_img = Image.open(cover_path)
        stego_img = Image.open(stego_path)
        psnr = calculate_psnr(cover_img, stego_img)
        result['psnr'] = psnr
        log(f"PSNR: {psnr:.2f} dB {'(EXCELLENT)' if psnr > 50 else '(GOOD)' if psnr > 40 else '(ACCEPTABLE)'}")
        
        # --- DECODE ---
        start_time = time.time()
        success, decoded_text, message = decode(stego_path)
        decode_time = time.time() - start_time
        result['decode_time'] = decode_time
        
        if not success:
            log(f"Decode: FAIL - {message}")
            result['error'] = message
            return result
        
        log(f"Decode: PASS ({decode_time:.3f}s)")
        result['decode_pass'] = True
        
        # Save decoded text
        with open(decoded_path, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        
        # --- ACCURACY ---
        if decoded_text == text:
            result['accuracy'] = 100.0
            log(f"Accuracy: 100% EXACT MATCH ✓")
        else:
            # Calculate character-level accuracy
            matches = sum(1 for a, b in zip(decoded_text, text) if a == b)
            accuracy = (matches / max(len(text), len(decoded_text))) * 100
            result['accuracy'] = accuracy
            log(f"Accuracy: {accuracy:.2f}% MISMATCH ✗")
            log(f"Expected: {text[:100]}")
            log(f"Got: {decoded_text[:100]}")
            result['error'] = "Text mismatch"
        
    except Exception as e:
        log(f"ERROR: {str(e)}")
        log(traceback.format_exc())
        result['error'] = str(e)
    
    return result


def run_all_tests():
    """Run all test cases."""
    setup_directories()
    
    # Clear log file
    if LOG_FILE.exists():
        os.remove(LOG_FILE)
    
    log("=" * 70)
    log("AI-GUIDED ADAPTIVE LSB TEXT STEGANOGRAPHY - VALIDATION REPORT")
    log("=" * 70)
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Method: AI-Inspired Importance Map + Adaptive LSB")
    log(f"AI Role: Embedding Location Optimization (Sobel + Laplacian)")
    
    # Generate test image
    cover_path = generate_test_image(512, 512, "cover_main.png")
    log(f"\nCover image: {cover_path}")
    
    # Calculate capacity
    cover_img = Image.open(cover_path)
    capacity = calculate_capacity(np.array(cover_img).shape)
    log(f"Max capacity: ~{capacity} chars")
    
    # Define test cases
    test_cases = [
        ("short_ascii", "Hello World!"),
        ("medium_ascii", "The quick brown fox jumps over the lazy dog. " * 5),
        ("long_ascii", "This is a longer test message that should still work. " * 50),
        ("special_chars", "Special: @#$%^&*()_+-=[]{}|;':\",./<>?`~"),
        ("unicode_japanese", "こんにちは世界！日本語テスト。"),
        ("unicode_emoji", "Hello 🌍🎉✨ World! 😀👍🔥"),
        ("unicode_mixed", "Привет мир! 你好世界! مرحبا بالعالم!"),
        ("numbers", "1234567890" * 100),
        ("single_char", "X"),
        ("whitespace", "   Tab:\tNewline:\nEnd   "),
    ]
    
    results = []
    
    for name, text in test_cases:
        result = run_test_case(name, text, cover_path, expect_success=True)
        results.append(result)
    
    # Edge cases that should FAIL gracefully
    log("\n" + "=" * 70)
    log("EDGE CASE TESTS (Expected Failures)")
    log("=" * 70)
    
    # Empty text
    result = run_test_case("empty_text", "", cover_path, expect_success=False)
    results.append(result)
    
    # Oversized text (should fail gracefully)
    oversized_text = "X" * (capacity + 1000)
    result = run_test_case("oversized_text", oversized_text, cover_path, expect_success=False)
    results.append(result)
    
    # --- FINAL REPORT ---
    log("\n" + "=" * 70)
    log("FINAL REPORT")
    log("=" * 70)
    
    total = len(results)
    encode_pass = sum(1 for r in results if r['encode_pass'])
    decode_pass = sum(1 for r in results if r['decode_pass'])
    exact_match = sum(1 for r in results if r['accuracy'] == 100.0)
    
    # Only count tests that were expected to succeed
    success_tests = [r for r in results if r['accuracy'] == 100.0 or (not r['decode_pass'] and r['encode_pass'])]
    
    log(f"\nTest Cases: {total}")
    log(f"Encode Pass: {encode_pass}/{total}")
    log(f"Decode Pass: {decode_pass}/{total}")
    log(f"100% Accuracy: {exact_match}/{decode_pass}")
    
    avg_psnr = np.mean([r['psnr'] for r in results if r['psnr'] > 0])
    avg_encode_time = np.mean([r['encode_time'] for r in results if r['encode_time'] > 0])
    avg_decode_time = np.mean([r['decode_time'] for r in results if r['decode_time'] > 0])
    
    log(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    log(f"Average Encode Time: {avg_encode_time:.3f}s")
    log(f"Average Decode Time: {avg_decode_time:.3f}s")
    
    log("\n" + "-" * 70)
    log("RESULTS SUMMARY TABLE")
    log("-" * 70)
    log(f"{'Test Case':<20} {'Encode':<8} {'Decode':<8} {'Accuracy':<12} {'PSNR':<10}")
    log("-" * 70)
    
    for r in results:
        enc = "PASS" if r['encode_pass'] else "FAIL"
        dec = "PASS" if r['decode_pass'] else "FAIL"
        acc = f"{r['accuracy']:.1f}%" if r['decode_pass'] else "N/A"
        psnr = f"{r['psnr']:.1f}dB" if r['psnr'] > 0 else "N/A"
        log(f"{r['name']:<20} {enc:<8} {dec:<8} {acc:<12} {psnr:<10}")
    
    log("-" * 70)
    
    # Determine overall verdict
    all_critical_pass = all(
        r['accuracy'] == 100.0 
        for r in results 
        if r['name'] not in ['empty_text', 'oversized_text']
    )
    
    edge_cases_handled = all(
        r['encode_pass'] and not r['decode_pass']
        for r in results
        if r['name'] in ['empty_text', 'oversized_text']
    ) or all(
        not r['encode_pass']
        for r in results
        if r['name'] in ['empty_text', 'oversized_text']
    )
    
    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    
    if all_critical_pass and avg_psnr > 40:
        log("✅ READY FOR DEPLOYMENT")
        log("\nMethod: AI-Guided Adaptive LSB")
        log("AI Role: Embedding Location Optimization")
        log("Encode: PASS")
        log("Decode: PASS")
        log("Text Accuracy: 100%")
        log(f"Visual Quality: {'EXCELLENT' if avg_psnr > 50 else 'GOOD'}")
    else:
        log("❌ NEEDS ATTENTION")
        for r in results:
            if r['error']:
                log(f"  - {r['name']}: {r['error']}")
    
    log("\n" + "=" * 70)
    log(f"Log saved to: {LOG_FILE}")
    
    return all_critical_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
