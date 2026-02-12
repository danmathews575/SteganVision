"""
DataLoader utilities and helper functions.

Optimized for Windows + CUDA environments:
- num_workers=2: Safe multiprocessing on Windows
- pin_memory: Enabled when CUDA is available for faster CPU->GPU transfer
- persistent_workers: Keeps workers alive between epochs to avoid restart overhead
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None
) -> DataLoader:
    """
    Create a DataLoader with optimized settings for Windows + CUDA.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size (default: 32)
        shuffle (bool): Whether to shuffle data (default: True)
        num_workers (int): Number of worker processes (default: 2, optimized for Windows)
        pin_memory (bool, optional): Pin memory for faster GPU transfer. 
                                     If None, automatically enabled when CUDA is available.
        drop_last (bool): Drop incomplete last batch (default: False)
        persistent_workers (bool, optional): Keep workers alive between epochs.
                                            If None, automatically enabled when num_workers > 0.
        
    Returns:
        DataLoader: Configured DataLoader
    """
    # Auto-detect optimal settings if not specified
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with optimized settings.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size (default: 32)
        num_workers (int): Number of worker processes (default: 2, optimized for Windows)
        pin_memory (bool, optional): Pin memory for faster GPU transfer.
                                     If None, automatically enabled when CUDA is available.
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    persistent_workers = (num_workers > 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for training
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all samples for validation
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader
