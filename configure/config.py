"""Reproducibility and device-selection helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for Python's built-in :mod:`random`, NumPy, and PyTorch
    (both CPU and CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the best available PyTorch device identifier.

    Returns:
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
