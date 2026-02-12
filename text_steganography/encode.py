#!/usr/bin/env python3
"""
AI-Guided Adaptive LSB Text Steganography - Encoder CLI

Embeds text into an image using AI-inspired importance maps
and adaptive LSB modification.

Usage:
    python encode.py --image cover.png --text secret.txt --out stego.png
    
Or with raw text string:
    python encode.py --image cover.png --message "Hello World" --out stego.png
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_guided_lsb.encoder import encode, encode_from_file


def main():
    parser = argparse.ArgumentParser(
        description="AI-Guided Adaptive LSB Text Steganography Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python encode.py --image cover.png --text secret.txt --out stego.png
    python encode.py --image cover.png --message "Secret message" --out stego.png
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to cover image (PNG recommended for lossless)"
    )
    
    # Text can be from file or direct string
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", "-t",
        help="Path to text file containing secret message"
    )
    text_group.add_argument(
        "--message", "-m",
        help="Secret message as a direct string"
    )
    
    parser.add_argument(
        "--out", "-o",
        default="stego.png",
        help="Output path for stego image (default: stego.png)"
    )
    
    parser.add_argument(
        "--save-importance-map",
        action="store_true",
        help="Save importance map visualization alongside stego image"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    image_path = Path(args.image)
    output_path = Path(args.out)
    
    try:
        if args.text:
            # Encode from text file
            success, message = encode_from_file(
                image_path,
                Path(args.text),
                output_path
            )
        else:
            # Encode from direct message string
            success, message = encode(
                image_path,
                args.message,
                output_path,
                save_importance_map=args.save_importance_map
            )
        
        if success:
            print(f"[SUCCESS] {message}")
            sys.exit(0)
        else:
            print(f"[FAILED] {message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
