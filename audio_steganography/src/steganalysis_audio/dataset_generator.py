import os
import glob
import random
import torchaudio
import torch
import numpy as np
import sys
from tqdm import tqdm

# Add parent directory to path to import ai_guided_lsb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_guided_lsb.encoder import encode

def generate_dataset(source_dir, output_dir, clip_duration=1.0, num_samples=500):
    """
    Generates a dataset for steganalysis.
    
    Args:
        source_dir (str): Directory containing source .wav files.
        output_dir (str): Output directory for 'cover' and 'stego'.
        clip_duration (float): Duration of each clip in seconds.
        num_samples (int): Target number of samples (pairs).
    """
    
    cover_out = os.path.join(output_dir, 'cover')
    stego_out = os.path.join(output_dir, 'stego')
    os.makedirs(cover_out, exist_ok=True)
    os.makedirs(stego_out, exist_ok=True)
    
    # Find source files
    source_files = glob.glob(os.path.join(source_dir, '**', '*.wav'), recursive=True)
    if not source_files:
        print("No source .wav files found!")
        return
        
    print(f"Found {len(source_files)} source files.")
    
    # Create a temporary secret file
    temp_secret_path = os.path.join(output_dir, "temp_secret.txt")
    
    generated = 0
    pbar = tqdm(total=num_samples, desc="Generating Dataset")
    
    while generated < num_samples:
        # Pick random source file
        src_path = random.choice(source_files)
        
        try:
            waveform, sample_rate = torchaudio.load(src_path)
            
            # Resample to 16000Hz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Ensure mono or take first channel
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]
            
            # Calculate samples needed
            num_frames = int(clip_duration * sample_rate)
            
            # Skip if too short
            if waveform.shape[1] < num_frames:
                 continue
            
            # Random crop
            max_start = waveform.shape[1] - num_frames
            if max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = 0
                
            clip = waveform[:, start:start+num_frames]
            
            # Save Cover
            cover_filename = f"{generated:05d}.wav"
            cover_path = os.path.join(cover_out, cover_filename)
            stego_path = os.path.join(stego_out, cover_filename)
            
            torchaudio.save(cover_path, clip, sample_rate)
            
            # Generate Random Secret
            secret_len = random.randint(10, 50)
            secret_text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ', k=secret_len))
            with open(temp_secret_path, "w") as f:
                f.write(secret_text)
            
            # Encode
            try:
                # encode(cover_path, secret_path, output_path)
                encode(cover_path, temp_secret_path, stego_path, bits_per_sample=1, use_compression=False)
                
                if os.path.exists(stego_path):
                    generated += 1
                    pbar.update(1)
            except Exception as e:
                # print(f"Encoding error: {e}")
                pass
                
        except Exception as e:
            # print(f"Processing error: {e}")
            pass

    pbar.close()
    if os.path.exists(temp_secret_path):
        os.remove(temp_secret_path)
    print(f"Dataset generated at {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()
    
    generate_dataset(args.source_dir, args.output_dir, num_samples=args.num_samples)
