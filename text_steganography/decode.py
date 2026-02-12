#!/usr/bin/env python3
"""
AI-Guided Adaptive LSB Text Steganography - Decoder CLI

Extracts hidden text from stego images using the same AI-guided
pixel ordering used during encoding.

Usage:
    python decode.py --image stego.png --out decoded.txt
    
Or print to stdout:
    python decode.py --image stego.png
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_guided_lsb.decoder import decode, decode_to_file


def main():
    parser = argparse.ArgumentParser(
        description="AI-Guided Adaptive LSB Text Steganography Decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python decode.py --image stego.png --out decoded.txt
    python decode.py --image stego.png  # Prints to stdout
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to stego image containing hidden text"
    )
    
    parser.add_argument(
        "--out", "-o",
        help="Output path for decoded text (optional, prints to stdout if not specified)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    image_path = Path(args.image)
    
    try:
        if args.out:
            # Decode to file
            success, message = decode_to_file(
                image_path,
                Path(args.out)
            )
            
            if success:
                print(f"[SUCCESS] {message}")
                sys.exit(0)
            else:
                print(f"[FAILED] {message}")
                sys.exit(1)
        else:
            # Decode and print to stdout
            success, text, message = decode(image_path)
            
            if success:
                print(f"[SUCCESS] Decoded text: {text}")
                sys.exit(0)
            else:
                print(f"[FAILED] {message}")
                sys.exit(1)
                
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
