import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random

class SteganalysisDataset(Dataset):
    def __init__(self, cover_dir, stego_dir, transform=None, limit=None):
        """
        Args:
            cover_dir (str): Path to cover images.
            stego_dir (str): Path to stego images.
            transform (callable, optional): Transform to apply to images.
            limit (int, optional): Limit the number of samples (for debugging/testing).
        """
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transform = transform
        
        # Find all images
        self.cover_paths = sorted(glob.glob(os.path.join(cover_dir, '*.*')))
        self.stego_paths = sorted(glob.glob(os.path.join(stego_dir, '*.*')))
        
        # Filter for common extensions just in case
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.cover_paths = [p for p in self.cover_paths if os.path.splitext(p)[1].lower() in valid_exts]
        self.stego_paths = [p for p in self.stego_paths if os.path.splitext(p)[1].lower() in valid_exts]

        # Ensure we have matching pairs if possible, or just load what we have
        # For steganalysis, we strictly need balanced classes usually, 
        # so we truncate to the minimum length of both to be safe, 
        # or we assume they are paired 1-to-1 if generated that way.
        # Here we will take the shorter list to ensure balance.
        min_len = min(len(self.cover_paths), len(self.stego_paths))
        if limit:
            min_len = min(min_len, limit)
            
        self.cover_paths = self.cover_paths[:min_len]
        self.stego_paths = self.stego_paths[:min_len]
        
        # Create file list: (path, label)
        # Label 0: Cover
        # Label 1: Stego
        self.files = [(p, 0) for p in self.cover_paths] + [(p, 1) for p in self.stego_paths]
        
        # Shuffle is usually done by DataLoader, but good to have random access
        # However, for splitting we might want to shuffle indices externally or use Subset.
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            image = Image.open(path).convert('L') # Convert to grayscale for SRNet usually, checking requirements
            # If the user wants RGB, we can change this. 
            # Most steganalysis (SRNet, XuNet) operates on Y channel or grayscale.
            # The prompt mentions "grayscale images; MNIST/CelebA-based" covering validation.
            # We will stick to 'L' (Grayscale) unless it fails.
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided: ToTensor
                image = transforms.ToTensor()(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor or handle error? 
            # For now, let's just re-raise or return zeros with a warning might be better but risky for training.
            raise e

def create_dataloaders(cover_dir, stego_dir, batch_size=32, train_split=0.8, val_split=0.1, num_workers=4, limit=None, crop_size=None):
    """
    Creates train, val, and test dataloaders.
    """
    
    transform_list = [transforms.ToTensor()]
    if crop_size:
        transform_list.append(transforms.CenterCrop(crop_size))
        
    transform = transforms.Compose(transform_list)

    dataset = SteganalysisDataset(cover_dir, stego_dir, transform=transform, limit=limit)
    
    # Split
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
