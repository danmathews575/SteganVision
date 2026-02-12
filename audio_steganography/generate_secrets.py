#!/usr/bin/env python3
"""
Generate shorter secret audio samples for testing audio steganography.
Creates 10-second secret files that fit within cover capacity.
"""

import numpy as np
import wave
from pathlib import Path

def generate_music_sample(output_path, duration=10, sample_rate=44100):
    """Generate music-like audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Musical piece with harmonics
    bass = 0.3 * np.sin(2 * np.pi * 110 * t)
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)
    melody = 0.4 * np.sin(2 * np.pi * (440 + vibrato) * t)
    harmony = 0.2 * np.sin(2 * np.pi * 660 * t)
    beat_freq = 2
    percussion = 0.1 * np.random.randn(len(t)) * (np.sin(2 * np.pi * beat_freq * t) > 0.5)
    
    audio = bass + melody + harmony + percussion
    delay_samples = int(0.05 * sample_rate)
    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = 0.3 * audio[:-delay_samples]
    audio = audio + reverb
    audio = audio / np.max(np.abs(audio)) * 0.9
    audio_int = (audio * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated: {output_path.name} ({duration}s)")

def generate_bgm_sample(output_path, duration=10, sample_rate=44100):
    """Generate ambient BGM"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    pad1 = 0.3 * np.sin(2 * np.pi * 220 * t)
    pad2 = 0.3 * np.sin(2 * np.pi * 277.18 * t)
    pad3 = 0.3 * np.sin(2 * np.pi * 329.63 * t)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    audio = (pad1 + pad2 + pad3) * lfo
    noise = 0.05 * np.random.randn(len(t))
    audio = audio + noise
    
    window_size = 100
    audio_filtered = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
    audio_filtered = audio_filtered / np.max(np.abs(audio_filtered)) * 0.8
    audio_int = (audio_filtered * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated: {output_path.name} ({duration}s)")

def generate_voice_sample(output_path, duration=10, sample_rate=44100):
    """Generate voice-like audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    f0_base = 150
    f0_variation = 30 * np.sin(2 * np.pi * 0.5 * t)
    f0 = f0_base + f0_variation
    
    formant1 = np.sin(2 * np.pi * f0 * t)
    formant2 = 0.5 * np.sin(2 * np.pi * (f0 * 3) * t)
    formant3 = 0.3 * np.sin(2 * np.pi * (f0 * 5) * t)
    voice = formant1 + formant2 + formant3
    
    syllable_rate = 3
    amplitude_mod = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * syllable_rate * t))
    voice = voice * amplitude_mod
    noise = 0.1 * np.random.randn(len(t))
    voice = voice + noise
    pause_mask = np.sin(2 * np.pi * 0.2 * t) > -0.3
    voice = voice * pause_mask
    voice = voice / np.max(np.abs(voice)) * 0.85
    audio_int = (voice * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated: {output_path.name} ({duration}s)")

def main():
    base_dir = Path(__file__).parent / 'test_data' / 'real_world'
    
    print("="*60)
    print("Generating Shorter Secret Audio Samples (10 seconds)")
    print("="*60)
    print()
    
    # Create secrets subdirectory
    secrets_dir = base_dir / 'secrets'
    secrets_dir.mkdir(exist_ok=True)
    
    # Generate 10-second secret samples
    print("Music secrets:")
    generate_music_sample(secrets_dir / 'secret_music_10s.wav', duration=10)
    generate_music_sample(secrets_dir / 'secret_music2_10s.wav', duration=10)
    
    print("\nBGM secrets:")
    generate_bgm_sample(secrets_dir / 'secret_bgm_10s.wav', duration=10)
    
    print("\nVoice secrets:")
    generate_voice_sample(secrets_dir / 'secret_voice_10s.wav', duration=10)
    generate_voice_sample(secrets_dir / 'secret_voice2_10s.wav', duration=10)
    
    print()
    print("="*60)
    print("✅ Secret audio samples generated!")
    print("="*60)
    
    # Show file sizes
    print("\nFile sizes:")
    for f in sorted(secrets_dir.glob('*.wav')):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

if __name__ == '__main__':
    main()
