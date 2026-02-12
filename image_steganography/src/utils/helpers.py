"""
General utility functions.
"""

import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id: int = 0) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_id (int): GPU device ID
        
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: torch.nn.Module, save_path: str):
    """
    Save model state dict.
    
    Args:
        model: PyTorch model
        save_path: Path to save model
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_model(model: torch.nn.Module, load_path: str, device: torch.device):
    """
    Load model state dict.
    
    Args:
        model: PyTorch model
        load_path: Path to load model from
        device: Device to load model to
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model


def denormalize(tensor: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """
    Denormalize a tensor.
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean
