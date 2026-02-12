import argparse
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .dataset import create_audio_dataloaders
from .models import AudioCNN

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloader (only need test)
    _, _, test_loader = create_audio_dataloaders(
        args.cover_dir, 
        args.stego_dir, 
        batch_size=args.batch_size,
        limit=args.limit
    )
    
    # Model
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(waveforms)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            
        # Plot CM
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cover', 'Stego'], yticklabels=['Cover', 'Stego'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover_dir', type=str, required=True)
    parser.add_argument('--stego_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    evaluate(args)
