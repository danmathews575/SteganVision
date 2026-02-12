"""
Inference script for embedding and extracting messages.

Usage:
    # Embed message
    python scripts/inference.py --mode embed --checkpoint model.pth --image cover.png --message "secret" --output stego.png
    
    # Extract message
    python scripts/inference.py --mode extract --checkpoint model.pth --image stego.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def embed_message(args):
    """Embed a message into a cover image."""
    # TODO: Load encoder model
    # TODO: Load cover image
    # TODO: Convert message to binary
    # TODO: Embed message
    # TODO: Save stego image
    print(f"Embedding message into {args.image}")
    print("TODO: Implement embedding logic")


def extract_message(args):
    """Extract a message from a stego image."""
    # TODO: Load decoder model
    # TODO: Load stego image
    # TODO: Extract message
    # TODO: Convert binary to text
    # TODO: Display message
    print(f"Extracting message from {args.image}")
    print("TODO: Implement extraction logic")


def main():
    parser = argparse.ArgumentParser(description="Steganography inference")
    parser.add_argument("--mode", type=str, required=True, choices=["embed", "extract"], help="Operation mode")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--message", type=str, default=None, help="Message to embed (for embed mode)")
    parser.add_argument("--output", type=str, default=None, help="Output path (for embed mode)")
    args = parser.parse_args()
    
    if args.mode == "embed":
        if args.message is None or args.output is None:
            parser.error("--message and --output are required for embed mode")
        embed_message(args)
    else:
        extract_message(args)


if __name__ == "__main__":
    main()
