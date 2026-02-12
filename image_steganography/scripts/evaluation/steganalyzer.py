"""
CNN-based Steganalyzer for Steganography Detection

Binary classifier to detect stego images vs cover images.
Uses ResNet-style architecture for robust feature extraction.

Training Protocol (per user requirement):
- Train ONE unified classifier on MIXED stego samples
- 50% cover, 25% CNN stego, 25% GAN stego
- Test separately on CNN vs cover and GAN vs cover
- This proves true robustness, not detector specialization
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Steganalyzer(nn.Module):
    """
    CNN-based steganalyzer for detecting stego vs cover images.
    
    Architecture:
    - SRM-style high-pass filters (optional)
    - ResNet-style backbone
    - Global average pooling
    - Binary classification head
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32, use_srm: bool = True):
        """
        Initialize steganalyzer.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base channel count for network
            use_srm: Whether to use SRM high-pass filters
        """
        super().__init__()
        self.use_srm = use_srm
        
        # Optional SRM preprocessing (Spatial Rich Model filters)
        if use_srm:
            self.srm = self._create_srm_filters()
            in_channels = 30  # SRM outputs 30 channels
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary: cover (0) vs stego (1)
        )
    
    def _create_srm_filters(self) -> nn.Conv2d:
        """Create SRM (Spatial Rich Model) high-pass filters."""
        # Standard SRM filters for steganalysis
        srm_filters = np.array([
            # 1st order
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
            # 2nd order
            [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
            [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
        ], dtype=np.float32)
        
        # Create conv layer with fixed SRM weights
        n_filters = len(srm_filters)
        srm_conv = nn.Conv2d(3, n_filters * 3, kernel_size=3, padding=1, bias=False)
        
        # Initialize with SRM filters for each input channel
        weights = np.zeros((n_filters * 3, 3, 3, 3), dtype=np.float32)
        for i, f in enumerate(srm_filters):
            for c in range(3):
                weights[i * 3 + c, c] = f
        
        srm_conv.weight.data = torch.from_numpy(weights)
        srm_conv.weight.requires_grad = False  # Freeze SRM filters
        
        return srm_conv
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer of residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Logits [B, 1] (use sigmoid for probabilities)
        """
        if self.use_srm:
            x = self.srm(x)
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of being stego (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class SteganalysisDataset(Dataset):
    """
    Dataset for steganalysis training/testing.
    
    Supports mixed training with covers, CNN stegos, and GAN stegos.
    """
    
    def __init__(
        self,
        covers: List[torch.Tensor],
        cnn_stegos: Optional[List[torch.Tensor]] = None,
        gan_stegos: Optional[List[torch.Tensor]] = None,
        mode: str = 'mixed'
    ):
        """
        Initialize dataset.
        
        Args:
            covers: List of cover image tensors
            cnn_stegos: List of CNN stego tensors (optional)
            gan_stegos: List of GAN stego tensors (optional)
            mode: 'mixed' (unified training), 'cnn', or 'gan'
        """
        self.mode = mode
        self.samples = []  # List of (image, label)
        
        # Add covers (label = 0)
        for cover in covers:
            self.samples.append((cover, 0))
        
        # Add stegos based on mode
        if mode == 'mixed':
            # Unified training: 50% cover, 25% CNN stego, 25% GAN stego
            if cnn_stegos:
                for stego in cnn_stegos:
                    self.samples.append((stego, 1))
            if gan_stegos:
                for stego in gan_stegos:
                    self.samples.append((stego, 1))
        elif mode == 'cnn' and cnn_stegos:
            for stego in cnn_stegos:
                self.samples.append((stego, 1))
        elif mode == 'gan' and gan_stegos:
            for stego in gan_stegos:
                self.samples.append((stego, 1))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


class SteganalyzerTrainer:
    """
    Trainer for steganalysis classifier with early stopping.
    """
    
    def __init__(
        self,
        model: Steganalyzer,
        device: torch.device,
        lr: float = 1e-4,
        patience: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5)
        self.patience = patience
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate and return loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels_t = labels.float().unsqueeze(1).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_t)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30
    ) -> Dict:
        """
        Train with early stopping.
        
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        print(f"\nTraining steganalyzer for up to {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return history


def evaluate_steganalyzer(
    model: Steganalyzer,
    covers: List[torch.Tensor],
    stegos: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 16
) -> Dict:
    """
    Evaluate steganalyzer on cover vs stego detection.
    
    Args:
        model: Trained steganalyzer
        covers: List of cover tensors
        stegos: List of stego tensors
        device: Torch device
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with accuracy, ROC-AUC, FPR@90TPR, etc.
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    # Covers (label 0)
    with torch.no_grad():
        for i in range(0, len(covers), batch_size):
            batch = torch.stack(covers[i:i+batch_size]).to(device)
            probs = model.predict_proba(batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend([0] * len(batch))
    
    # Stegos (label 1)
    with torch.no_grad():
        for i in range(0, len(stegos), batch_size):
            batch = torch.stack(stegos[i:i+batch_size]).to(device)
            probs = model.predict_proba(batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend([1] * len(batch))
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # FPR at 90% TPR
    idx_90_tpr = np.argmin(np.abs(tpr - 0.9))
    fpr_at_90_tpr = fpr[idx_90_tpr]
    
    # Confidence interval for ROC-AUC (bootstrap)
    n_bootstrap = 1000
    auc_samples = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(all_labels), size=len(all_labels), replace=True)
        if len(np.unique(all_labels[indices])) < 2:
            continue
        auc_sample = roc_auc_score(all_labels[indices], all_probs[indices])
        auc_samples.append(auc_sample)
    
    auc_ci_lower = np.percentile(auc_samples, 2.5)
    auc_ci_upper = np.percentile(auc_samples, 97.5)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'roc_auc_ci_lower': auc_ci_lower,
        'roc_auc_ci_upper': auc_ci_upper,
        'fpr_at_90_tpr': fpr_at_90_tpr,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


if __name__ == '__main__':
    # Test the steganalyzer
    print("Testing Steganalyzer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Steganalyzer(in_channels=3, base_channels=32, use_srm=True)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Steganalyzer parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
        probs = model.predict_proba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Probabilities: {probs.squeeze().cpu().numpy()}")
    
    print("\nSteganalyzer test complete!")
