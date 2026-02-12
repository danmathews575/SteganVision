"""
Logging utilities for training.

Supports TensorBoard and Weights & Biases.
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path


class Logger:
    """
    Base logger class.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        raise NotImplementedError
        
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        raise NotImplementedError
        
    def log_dict(self, metrics: Dict[str, float], step: int):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step)


class TensorBoardLogger(Logger):
    """
    TensorBoard logger.
    """
    
    def __init__(self, log_dir: str):
        super(TensorBoardLogger, self).__init__(log_dir)
        # TODO: Initialize TensorBoard writer
        
    def log_scalar(self, tag: str, value: float, step: int):
        # TODO: Log to TensorBoard
        raise NotImplementedError
        
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        # TODO: Log image to TensorBoard
        raise NotImplementedError


class WandbLogger(Logger):
    """
    Weights & Biases logger.
    """
    
    def __init__(self, project: str, name: str, config: Dict[str, Any]):
        # TODO: Initialize wandb
        raise NotImplementedError
        
    def log_scalar(self, tag: str, value: float, step: int):
        # TODO: Log to wandb
        raise NotImplementedError
        
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        # TODO: Log image to wandb
        raise NotImplementedError


def get_logger(logger_type: str = "tensorboard", **kwargs) -> Logger:
    """
    Factory function to get logger.
    
    Args:
        logger_type (str): Type of logger ("tensorboard" or "wandb")
        **kwargs: Logger-specific arguments
        
    Returns:
        Logger: Logger instance
    """
    # TODO: Return appropriate logger
    raise NotImplementedError
