#!/usr/bin/env python3
"""
Generate 10-20 second audio samples for comprehensive testing.
Creates music, BGM, and voice samples.
"""

import numpy as np
import wave
from pathlib import Path

def generate_music_sample(output_path, duration=15, sample_rate=44100):
    """Generate music-like audio with multiple harmonics and rhythm"""
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

def generate_bgm_sample(output_path, duration=15, sample_rate=44100):
    """Generate ambient background music"""
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

def generate_voice_sample(output_path, duration=15, sample_rate=44100):
    """Generate voice-like audio with formants"""
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
    base_dir = Path(__file__).parent / 'test_data' / 'short_samples'
    
    # Create directories
    (base_dir / 'music').mkdir(parents=True, exist_ok=True)
    (base_dir / 'bgm').mkdir(parents=True, exist_ok=True)
    (base_dir / 'voice').mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating 10-20s Audio Samples for Testing")
    print("="*60)
    print()
    
    # Generate music samples (15-20s)
    print("Music samples:")
    generate_music_sample(base_dir / 'music' / 'music_electronic_15s.wav', duration=15)
    generate_music_sample(base_dir / 'music' / 'music_ambient_20s.wav', duration=20)
    generate_music_sample(base_dir / 'music' / 'music_rhythmic_18s.wav', duration=18)
    
    print("\nBGM samples:")
    generate_bgm_sample(base_dir / 'bgm' / 'bgm_ambient_15s.wav', duration=15)
    generate_bgm_sample(base_dir / 'bgm' / 'bgm_calm_20s.wav', duration=20)
    
    print("\nVoice samples:")
    generate_voice_sample(base_dir / 'voice' / 'voice_narration_15s.wav', duration=15)
    generate_voice_sample(base_dir / 'voice' / 'voice_speech_10s.wav', duration=10)
    generate_voice_sample(base_dir / 'voice' / 'voice_dialogue_20s.wav', duration=20)
    
    print()
    print("="*60)
    print("✅ All audio samples generated!")
    print("="*60)
    
    # Show file sizes
    print("\nFile sizes:")
    for category in ['music', 'bgm', 'voice']:
        cat_dir = base_dir / category
        files = sorted(cat_dir.glob('*.wav'))
        print(f"\n{category.upper()}:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            duration = f.stem.split('_')[-1].replace('s', '')
            print(f"  {f.name}: {size_kb:.1f} KB ({duration}s)")

if __name__ == '__main__':
    main()
