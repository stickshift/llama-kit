import pytest
import torch

from llama_kit.tools import torch_device

__all__ = [
    "device",
]


@pytest.fixture
def device() -> torch.device:
    return torch_device()
