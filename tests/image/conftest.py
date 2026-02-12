"""
PyTest configuration file.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Fixture for PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_image():
    """Fixture for dummy image tensor."""
    return torch.randn(1, 3, 128, 128)


@pytest.fixture
def dummy_message():
    """Fixture for dummy message tensor."""
    return torch.randint(0, 2, (1, 100)).float()


@pytest.fixture
def batch_size():
    """Fixture for batch size."""
    return 4
