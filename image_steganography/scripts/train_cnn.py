"""
Training script for CNN baseline model.

Usage:
    python scripts/train_cnn.py --config configs/cnn_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Train CNN baseline steganography model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # TODO: Load configuration
    # TODO: Set random seed
    # TODO: Initialize datasets and dataloaders
    # TODO: Initialize models
    # TODO: Initialize trainer
    # TODO: Start training
    
    print(f"Training CNN baseline model with config: {args.config}")
    print("TODO: Implement training logic")


if __name__ == "__main__":
    main()
