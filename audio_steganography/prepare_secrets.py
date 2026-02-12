#!/usr/bin/env python3
"""
Prepare audio secrets by downsampling to fit in same-duration covers.
Downsamples to 8kHz (phone quality) to reduce file size significantly.
"""

import wave
import numpy as np
from pathlib import Path
from scipy import signal

def downsample_audio(input_path, output_path, target_rate=8000):
    """Downsample audio to lower sample rate"""
    # Read input
    with wave.open(str(input_path), 'r') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
    
    # Resample
    num_samples = int(len(audio_data) * target_rate / sample_rate)
    resampled = signal.resample(audio_data, num_samples).astype(np.int16)
    
    # Write output
    with wave.open(str(output_path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(target_rate)
        wav.writeframes(resampled.tobytes())
    
    original_size = input_path.stat().st_size / 1024
    new_size = output_path.stat().st_size / 1024
    reduction = original_size / new_size
    
    print(f"✓ {input_path.name}")
    print(f"  {sample_rate}Hz → {target_rate}Hz")
    print(f"  {original_size:.1f}KB → {new_size:.1f}KB ({reduction:.1f}x smaller)")

def main():
    samples_dir = Path(__file__).parent / 'test_data' / 'short_samples'
    secrets_dir = Path(__file__).parent / 'test_data' / 'secrets_downsampled'
    secrets_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downsampling Audio for Same-Size Steganography")
    print("="*60)
    print()
    
    # Downsample all audio files to 8kHz
    for category in ['music', 'bgm', 'voice']:
        cat_dir = samples_dir / category
        if not cat_dir.exists():
            continue
        
        print(f"{category.upper()}:")
        for audio_file in sorted(cat_dir.glob('*.wav')):
            output_file = secrets_dir / f"secret_{audio_file.name}"
            downsample_audio(audio_file, output_file, target_rate=8000)
        print()
    
    print("="*60)
    print("✅ All secrets prepared!")
    print("="*60)

if __name__ == '__main__':
    main()
