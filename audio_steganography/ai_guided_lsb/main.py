#!/usr/bin/env python3
"""
AI-Guided LSB Audio Steganography - Main Entry Point

A hybrid intelligent steganography system for hiding audio/data in audio.

Usage:
    # Encode secret into cover audio
    python main.py --cover cover.wav --secret secret.wav --out stego.wav
    
    # Decode secret from stego audio
    python main.py --decode --stego stego.wav --out recovered.wav
    
    # With 2-bit embedding (higher capacity, slightly lower SNR)
    python main.py --cover cover.wav --secret secret.wav --out stego.wav --bits 2

Features:
    - AI-guided embedding (psychoacoustic importance model)
    - 100% exact secret recovery (deterministic decoding)
    - Imperceptible modifications (embeds in masked regions)
    - CRC32 checksum validation
    - Fast runtime (< 1 second for typical audio)

Author: AI-Guided Steganography System
"""

import argparse
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_guided_lsb import encode, decode


def main():
    parser = argparse.ArgumentParser(
        description='AI-Guided LSB Audio Steganography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Encode: python main.py --cover music.wav --secret voice.wav --out stego.wav
  Decode: python main.py --decode --stego stego.wav --out recovered.wav
        '''
    )
    
    # Mode selection
    parser.add_argument(
        '--decode', 
        action='store_true',
        help='Decode mode (extract secret from stego audio)'
    )
    
    # Encode arguments
    parser.add_argument(
        '--cover',
        type=str,
        help='Path to cover audio file (WAV) [encode mode]'
    )
    parser.add_argument(
        '--secret',
        type=str,
        help='Path to secret file to embed [encode mode]'
    )
    
    # Decode arguments
    parser.add_argument(
        '--stego',
        type=str,
        help='Path to stego audio file (WAV) [decode mode]'
    )
    
    # Common arguments
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output path (stego.wav for encode, recovered file for decode)'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=1,
        choices=[1, 2],
        help='LSB bits per sample (default: 1)'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.decode:
            # =================================================================
            # DECODE MODE
            # =================================================================
            if not args.stego:
                parser.error("--stego is required for decode mode")
            
            print(f"\n🔓 AI-GUIDED LSB AUDIO STEGANOGRAPHY - DECODE")
            print(f"{'='*50}")
            print(f"Stego:  {args.stego}")
            print(f"Output: {args.out}")
            print(f"Bits:   {args.bits} per sample")
            print(f"{'='*50}\n")
            
            result = decode(args.stego, args.out, bits_per_sample=args.bits)
            
            elapsed = time.time() - start_time
            
            print(f"\n{result['status']}")
            print(f"Secret size:     {result['secret_size_bytes']:,} bytes")
            print(f"Checksum:        {result['checksum']} ({'VALID ✓' if result['checksum_valid'] else 'INVALID ✗'})")
            print(f"Output:          {result['output_path']}")
            print(f"Runtime:         {elapsed:.3f} seconds")
            
        else:
            # =================================================================
            # ENCODE MODE
            # =================================================================
            if not args.cover:
                parser.error("--cover is required for encode mode")
            if not args.secret:
                parser.error("--secret is required for encode mode")
            
            print(f"\n🔒 AI-GUIDED LSB AUDIO STEGANOGRAPHY - ENCODE")
            print(f"{'='*50}")
            print(f"Cover:  {args.cover}")
            print(f"Secret: {args.secret}")
            print(f"Output: {args.out}")
            print(f"Bits:   {args.bits} per sample")
            print(f"{'='*50}\n")
            
            result = encode(args.cover, args.secret, args.out, bits_per_sample=args.bits)
            
            elapsed = time.time() - start_time
            
            print(f"\n{result['status']}")
            print(f"Cover samples:   {result['cover_samples']:,}")
            print(f"Secret size:     {result['secret_size_bytes']:,} bytes")
            print(f"Capacity used:   {result['capacity_utilization']:.2f}%")
            print(f"SNR:             {result['snr_db']:.2f} dB")
            print(f"MSE:             {result['mse']:.6f}")
            print(f"Output:          {result['output_path']}")
            print(f"Runtime:         {elapsed:.3f} seconds")
            
            # Quality assessment
            if result['snr_db'] > 50:
                quality = "EXCELLENT (virtually identical)"
            elif result['snr_db'] > 40:
                quality = "VERY GOOD (imperceptible)"
            elif result['snr_db'] > 30:
                quality = "GOOD (minor artifacts possible)"
            else:
                quality = "FAIR (may have audible artifacts)"
            
            print(f"Quality:         {quality}")
    
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("✅ Operation completed successfully")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
