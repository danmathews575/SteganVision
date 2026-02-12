#!/usr/bin/env python3
"""
Generate synthetic audio samples for testing audio steganography.
Creates realistic audio with different characteristics: music, BGM, and voice.
Duration: ~1 minute each
"""

import numpy as np
import wave
import os
from pathlib import Path

def generate_music_sample(output_path, duration=60, sample_rate=44100):
    """Generate music-like audio with multiple harmonics and rhythm"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a musical piece with multiple instruments
    # Bass line (low frequency)
    bass = 0.3 * np.sin(2 * np.pi * 110 * t)  # A2 note
    
    # Melody (mid frequency with vibrato)
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)
    melody = 0.4 * np.sin(2 * np.pi * (440 + vibrato) * t)  # A4 with vibrato
    
    # Harmony (higher frequency)
    harmony = 0.2 * np.sin(2 * np.pi * 660 * t)  # E5
    
    # Percussion (rhythmic noise)
    beat_freq = 2  # 2 Hz = 120 BPM
    percussion = 0.1 * np.random.randn(len(t)) * (np.sin(2 * np.pi * beat_freq * t) > 0.5)
    
    # Combine all elements
    audio = bass + melody + harmony + percussion
    
    # Add some reverb effect (simple delay)
    delay_samples = int(0.05 * sample_rate)  # 50ms delay
    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = 0.3 * audio[:-delay_samples]
    audio = audio + reverb
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated music: {output_path.name} ({duration}s, {sample_rate}Hz)")

def generate_bgm_sample(output_path, duration=60, sample_rate=44100):
    """Generate ambient background music with slow evolution"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Ambient pad (slow-moving chords)
    pad1 = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
    pad2 = 0.3 * np.sin(2 * np.pi * 277.18 * t)  # C#4
    pad3 = 0.3 * np.sin(2 * np.pi * 329.63 * t)  # E4
    
    # Add slow LFO (Low Frequency Oscillator) for movement
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz
    
    # Combine pads with LFO
    audio = (pad1 + pad2 + pad3) * lfo
    
    # Add subtle noise for texture
    noise = 0.05 * np.random.randn(len(t))
    audio = audio + noise
    
    # Low-pass filter effect (simple moving average)
    window_size = 100
    audio_filtered = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
    
    # Normalize
    audio_filtered = audio_filtered / np.max(np.abs(audio_filtered)) * 0.8
    
    # Convert to 16-bit PCM
    audio_int = (audio_filtered * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated BGM: {output_path.name} ({duration}s, {sample_rate}Hz)")

def generate_voice_sample(output_path, duration=60, sample_rate=44100):
    """Generate voice-like audio with formants and prosody"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency (pitch) with variation (prosody)
    f0_base = 150  # Male voice ~150 Hz
    f0_variation = 30 * np.sin(2 * np.pi * 0.5 * t)  # Slow pitch variation
    f0 = f0_base + f0_variation
    
    # Generate voice with formants (resonances)
    # Formant 1 (vowel quality)
    formant1 = np.sin(2 * np.pi * f0 * t)
    
    # Formant 2
    formant2 = 0.5 * np.sin(2 * np.pi * (f0 * 3) * t)
    
    # Formant 3
    formant3 = 0.3 * np.sin(2 * np.pi * (f0 * 5) * t)
    
    # Combine formants
    voice = formant1 + formant2 + formant3
    
    # Add speech-like amplitude modulation (syllables)
    syllable_rate = 3  # 3 syllables per second
    amplitude_mod = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * syllable_rate * t))
    voice = voice * amplitude_mod
    
    # Add slight noise (breathiness)
    noise = 0.1 * np.random.randn(len(t))
    voice = voice + noise
    
    # Add pauses (silence periods)
    pause_mask = np.sin(2 * np.pi * 0.2 * t) > -0.3  # Occasional pauses
    voice = voice * pause_mask
    
    # Normalize
    voice = voice / np.max(np.abs(voice)) * 0.85
    
    # Convert to 16-bit PCM
    audio_int = (voice * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    print(f"✓ Generated voice: {output_path.name} ({duration}s, {sample_rate}Hz)")

def main():
    base_dir = Path(__file__).parent / 'test_data' / 'real_world'
    
    print("="*60)
    print("Generating Synthetic Audio Samples for Testing")
    print("="*60)
    print()
    
    # Generate music samples (different styles)
    music_dir = base_dir / 'music'
    generate_music_sample(music_dir / 'music_electronic_60s.wav', duration=60)
    generate_music_sample(music_dir / 'music_ambient_60s.wav', duration=60)
    generate_music_sample(music_dir / 'music_rhythmic_60s.wav', duration=60)
    
    print()
    
    # Generate BGM samples
    bgm_dir = base_dir / 'bgm'
    generate_bgm_sample(bgm_dir / 'bgm_ambient_60s.wav', duration=60)
    generate_bgm_sample(bgm_dir / 'bgm_calm_60s.wav', duration=60)
    
    print()
    
    # Generate voice samples
    voice_dir = base_dir / 'voice'
    generate_voice_sample(voice_dir / 'voice_narration_60s.wav', duration=60)
    generate_voice_sample(voice_dir / 'voice_speech_60s.wav', duration=60)
    
    print()
    print("="*60)
    print("✅ All audio samples generated successfully!")
    print("="*60)
    print()
    print("Generated files:")
    for category in ['music', 'bgm', 'voice']:
        cat_dir = base_dir / category
        files = list(cat_dir.glob('*.wav'))
        print(f"\n{category.upper()}:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    main()
