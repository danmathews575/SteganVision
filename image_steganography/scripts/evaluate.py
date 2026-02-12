"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model.pth --data data/test
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate steganography model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, default="data/outputs", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    # TODO: Load model from checkpoint
    # TODO: Load test data
    # TODO: Run evaluation
    # TODO: Calculate metrics (PSNR, SSIM, message accuracy)
    # TODO: Save results and visualizations
    
    print(f"Evaluating model from: {args.checkpoint}")
    print(f"Test data: {args.data}")
    print("TODO: Implement evaluation logic")


if __name__ == "__main__":
    main()
