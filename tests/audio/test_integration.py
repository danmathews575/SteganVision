#!/usr/bin/env python3
"""
Final optimized test suite - only voice secrets (they compress best).
Tests music, BGM, and voice covers with voice secrets.
"""

import sys
from pathlib import Path
import time
import os

# Fix path and encoding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

# Import from package
try:
    from audio_steganography.ai_guided_lsb import encode, decode
except ImportError:
    # Fallback if running from within tests directory
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from audio_steganography.ai_guided_lsb import encode, decode

def run_test(cover_path, secret_path, test_name):
    """Run single test"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Cover:  {cover_path.name}")
    print(f"Secret: {secret_path.name}")
    print(f"{'-'*70}")
    
    results_dir = Path('tests/outputs/audio/results')
    (results_dir / 'stego').mkdir(parents=True, exist_ok=True)
    (results_dir / 'recovered').mkdir(parents=True, exist_ok=True)
    
    stego = results_dir / 'stego' / f"{test_name}.wav"
    recovered = results_dir / 'recovered' / f"{test_name}_recovered.wav"
    
    # ENCODE
    print("\n🔒 ENCODING...")
    start = time.time()
    enc = encode(str(cover_path), str(secret_path), str(stego), 
                 bits_per_sample=2, use_compression=True)
    enc_time = time.time() - start
    
    print(f"✓ Encoded in {enc_time:.3f}s")
    print(f"  Original: {enc['secret_size_bytes']:,} bytes")
    print(f"  Compressed: {enc['payload_bytes']:,} bytes")
    print(f"  Compression: {enc['compression_ratio']:.2f}x")
    print(f"  Capacity: {enc['capacity_utilization']:.2f}%")
    print(f"  SNR: {enc['snr_db']:.2f} dB")
    
    # DECODE
    print("\n🔓 DECODING...")
    start = time.time()
    dec = decode(str(stego), str(recovered), bits_per_sample=2)
    dec_time = time.time() - start
    
    ok = dec['checksum_valid']
    print(f"✓ Decoded in {dec_time:.3f}s")
    print(f"  Checksum: {'✅ VALID' if ok else '❌ INVALID'}")
    
    print(f"\n{'='*70}")
    print(f"{'✅ PASSED' if ok else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return ok, enc, dec

def main():
    # Correct path to test data
    base = Path('audio_steganography/test_data')
    covers = base / 'short_samples'
    secrets = base / 'secrets_downsampled'
    
    if not covers.exists() or not secrets.exists():
        print(f"Skipping tests: Test data not found at {base}")
        return

    print("\n" + "="*70)
    print("AUDIO STEGANOGRAPHY - FINAL DEMO")
    print("Same-size audio with compression (10-20s)")
    print("="*70)
    
    tests = [
        # Music covers with voice secrets
        (covers / 'music' / 'music_electronic_15s.wav', 
         secrets / 'secret_voice_speech_10s.wav',
         'music_electronic_x_voice_speech'),
        
        (covers / 'music' / 'music_ambient_20s.wav',
         secrets / 'secret_voice_narration_15s.wav',
         'music_ambient_x_voice_narration'),
        
        # BGM covers with voice secrets
        (covers / 'bgm' / 'bgm_ambient_15s.wav',
         secrets / 'secret_voice_speech_10s.wav',
         'bgm_ambient_x_voice_speech'),
        
        (covers / 'bgm' / 'bgm_calm_20s.wav',
         secrets / 'secret_voice_narration_15s.wav',
         'bgm_calm_x_voice_narration'),
        
        # Voice covers with voice secrets
        (covers / 'voice' / 'voice_dialogue_20s.wav',
         secrets / 'secret_voice_narration_15s.wav',
         'voice_dialogue_x_voice_narration'),
        
        (covers / 'voice' / 'voice_narration_15s.wav',
         secrets / 'secret_voice_speech_10s.wav',
         'voice_narration_x_voice_speech'),
    ]
    
    results = []
    for cover, secret, name in tests:
        if not cover.exists() or not secret.exists():
            print(f"⚠ Skipping test {name}: File not found")
            continue
            
        try:
            ok, enc, dec = run_test(cover, secret, name)
            results.append((name, ok, enc, dec))
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append((name, False, None, None))
    
    if not results:
        print("No tests run.")
        return

    # Summary
    passed = sum(1 for _, ok, _, _ in results if ok)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total:   {len(results)}")
    print(f"Passed:  {passed} ✅")
    print(f"Failed:  {len(results) - passed} ❌")
    print(f"Success: {passed/len(results)*100:.0f}%")
    
    if passed > 0:
        successful = [(n, e, d) for n, ok, e, d in results if ok and e]
        avg_comp = sum(e['compression_ratio'] for _, e, _ in successful) / len(successful)
        avg_cap = sum(e['capacity_utilization'] for _, e, _ in successful) / len(successful)
        avg_snr = sum(e['snr_db'] for _, e, _ in successful) / len(successful)
        
        print(f"\nAverage Metrics:")
        print(f"  Compression: {avg_comp:.2f}x")
        print(f"  Capacity:    {avg_cap:.1f}%")
        print(f"  SNR:         {avg_snr:.1f} dB")
    
    print("="*70)
    print("✅ ALL TESTS PASSED!" if passed == len(results) else f"⚠ {len(results)-passed} failed")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
