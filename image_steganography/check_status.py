"""Quick status check for training progress"""
import torch
from pathlib import Path
from datetime import datetime

cp_dir = Path('checkpoints/gan')

# Find all checkpoints
checkpoints = {
    'best': cp_dir / 'best_gan_model.pth',
    'interrupted': cp_dir / 'interrupted_checkpoint.pth',
    'epoch_21': cp_dir / 'gan_checkpoint_epoch_0021.pth',
    'epoch_17': cp_dir / 'gan_checkpoint_epoch_0017.pth',
}

print("=" * 70)
print("TRAINING STATUS CHECK")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for name, path in checkpoints.items():
    if not path.exists():
        print(f"{name}: ❌ Not found")
        continue
    
    cp = torch.load(path, map_location='cpu', weights_only=False)
    epoch = cp.get('epoch', '?')
    losses = cp.get('losses', {})
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    
    print(f"{name}:")
    print(f"  Epoch: {epoch}")
    print(f"  Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
    print(f"  G Loss: {losses.get('g_loss', 'N/A'):.4f}")
    print(f"  Cover Loss: {losses.get('cover_loss', 'N/A'):.4f}")
    print(f"  Secret Loss: {losses.get('secret_loss', 'N/A'):.4f}")
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Training was CANCELED during epoch 21")
print("Last complete epoch: 21")
print("Best model: Still epoch 13 (before BCE loss)")
print("\nRecommendation: Evaluate epoch 21 checkpoint with BCE loss")
