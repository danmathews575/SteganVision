import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .dataset import create_dataloaders
from .models import SRNet

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.cover_dir, 
        args.stego_dir, 
        batch_size=args.batch_size, 
        limit=args.limit,
        crop_size=args.crop_size
    )
    
    # Model
    model = SRNet().to(device)
    
    # Optimizer & Loss
    # SRNet paper uses SGD with momentum or Adamax. We'll use Adamax or Adam for stability.
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/total, 'acc': correct/total})
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("Saved best model.")
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover_dir', type=str, required=True)
    parser.add_argument('--stego_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for debugging')
    parser.add_argument('--crop_size', type=int, default=None, help='Center crop size')
    
    args = parser.parse_args()
    train(args)
