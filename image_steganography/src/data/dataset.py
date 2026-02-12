"""
Dataset classes for steganography training.

Handles loading diverse cover images (CelebA, COCO, etc.) and secret images (MNIST, Fashion-MNIST).
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
from pathlib import Path
from PIL import Image
import os
from .transforms import get_cover_transforms, get_secret_transforms


class SteganographyDataset(Dataset):
    """
    Dataset for image steganography training.
    
    Loads cover images from multiple sources and secret images (MNIST/Fashion-MNIST).
    Returns (cover_image, secret_image) pairs.
    
    Args:
        celeba_dir (str): Primary directory containing CelebA images (e.g., 'data/celeba')
        cover_dirs (List[str], optional): Additional directories containing cover images (COCO, DIV2K, etc.)
        mnist_root (str): Root directory for secret datasets
        secret_type (str): Type of secret to use: 'mnist' or 'fashion_mnist'
        image_size (int): Size to resize images to (default: 256)
        split (str): Dataset split - 'train' or 'test' (default: 'train')
        max_samples (int, optional): Maximum number of samples to use.
    """
    
    def __init__(
        self,
        celeba_dir: str = 'data/celeba',
        cover_dirs: Optional[List[str]] = None,
        mnist_root: str = 'data/mnist',
        secret_type: str = 'mnist',
        image_size: int = 256,
        split: str = 'train',
        max_samples: Optional[int] = None
    ):
        self.image_size = image_size
        self.split = split
        self.max_samples = max_samples
        self.secret_type = secret_type
        
        # 1. Collect Cover Images
        self.cover_paths = []
        
        # Add CelebA if it exists
        celeba_path = Path(celeba_dir)
        if celeba_path.exists():
            print(f"Loading covers from: {celeba_path}")
            self.cover_paths.extend(sorted(list(celeba_path.glob('*.jpg'))))
        
        # Add additional cover directories
        if cover_dirs:
            for d in cover_dirs:
                path = Path(d)
                if path.exists():
                    print(f"Loading covers from: {path}")
                    # Recursive search for common image formats
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        self.cover_paths.extend(sorted(list(path.rglob(ext))))
        
        if len(self.cover_paths) == 0:
            raise ValueError(
                f"No cover images found in {celeba_dir} or {cover_dirs}.\n"
                f"Please check your data paths."
            )
            
        # 2. Setup Secret Dataset
        self.mnist_root = Path(mnist_root)
        is_train = (split == 'train')
        
        if secret_type == 'mnist':
            self.secret_dataset = datasets.MNIST(
                root=str(self.mnist_root),
                train=is_train,
                download=True,
                transform=None
            )
        elif secret_type == 'fashion_mnist':
            self.secret_dataset = datasets.FashionMNIST(
                root=str(self.mnist_root),
                train=is_train,
                download=True,
                transform=None
            )
        else:
            raise ValueError(f"Unknown secret_type: {secret_type}. Use 'mnist' or 'fashion_mnist'.")
            
        # 3. Setup Transforms
        self.cover_transform = get_cover_transforms(image_size, split)
        self.secret_transform = get_secret_transforms(image_size, channels=1)
        
        # 4. Calculate Length
        # Dataset length is minimum of both datasets (or max_samples)
        full_length = min(len(self.cover_paths), len(self.secret_dataset))
        
        if max_samples is not None and max_samples > 0:
            self.length = min(max_samples, full_length)
        else:
            self.length = full_length
        
        print(f"Dataset initialized ({split}):")
        print(f"  - Cover images: {len(self.cover_paths)}")
        print(f"  - Secret images ({secret_type}): {len(self.secret_dataset)}")
        print(f"  - Using: {self.length} samples")
        
    def __len__(self) -> int:
        return self.length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of (cover, secret).
        """
        # Load Cover (Robust handling of different formats involved)
        cover_path = self.cover_paths[idx % len(self.cover_paths)]
        try:
            cover_image = Image.open(cover_path).convert('RGB')
            cover_image = self.cover_transform(cover_image)
        except Exception as e:
            print(f"Error loading cover {cover_path}: {e}")
            # Fallback to random noise or another image if corrupt
            cover_image = torch.zeros(3, self.image_size, self.image_size)
        
        # Load Secret
        secret_data, _ = self.secret_dataset[idx % len(self.secret_dataset)]
        secret_image = self.secret_transform(secret_data)
        
        return cover_image, secret_image
