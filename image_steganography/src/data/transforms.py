"""
Data transformations and augmentations for steganography.
"""

import torch
import torchvision.transforms as transforms
from typing import List, Optional


def get_cover_transforms(image_size: int = 256, split: str = 'train') -> transforms.Compose:
    """
    Get transformations for cover images.
    
    Args:
        image_size (int): Target image size
        split (str): 'train' or 'val'/'test'
        
    Returns:
        transforms.Compose: Composed transformations
    """
    if split == 'train':
        return transforms.Compose([
            # Strong augmentations for generalization
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def get_secret_transforms(image_size: int = 256, channels: int = 1) -> transforms.Compose:
    """
    Get transformations for secret images.
    
    Args:
        image_size (int): Target image size
        channels (int): Number of channels (1 for MNIST/Fashion, 3 for RGB)
        
    Returns:
        transforms.Compose: Composed transformations
    """
    mean = [0.5] * channels
    std = [0.5] * channels
    
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),  # CenterCrop in case of aspect ratio mismatch
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
