"""Shared helpers. Put this file at src/utils.py and import it everywhere.

Usage:
    from utils import set_seed, get_device, get_num_workers
    set_seed(42)
    device = get_device()
"""
import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so results are as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Slower but more repeatable on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available device: NVIDIA GPU, Apple GPU (Mac), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_workers() -> int:
    """DataLoader workers. 0 on Windows (avoids hangs), 4 elsewhere."""
    return 0 if os.name == "nt" else 4


def count_parameters(model: torch.nn.Module) -> tuple:
    """Return (total, trainable) parameter counts. Useful for the report."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick setup check: everyone runs `python src/utils.py`
    set_seed()
    print("PyTorch version:", torch.__version__)
    print("Device:", get_device())
    print("DataLoader workers:", get_num_workers())