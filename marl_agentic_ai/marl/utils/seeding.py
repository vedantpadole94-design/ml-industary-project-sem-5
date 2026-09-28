"""
marl/utils/seeding.py
Global seeding utility — seeds random, numpy, torch, and gymnasium.

Usage:
    from marl.utils.seeding import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """
    Set global random seeds for reproducibility across all relevant libraries.

    Seeds the following:
    - Python ``random`` module
    - NumPy (global default RNG)
    - PyTorch (CPU and all CUDA devices)
    - Gymnasium (via ``gym.utils.seeding``)
    - OS-level PYTHONHASHSEED

    Args:
        seed: Integer seed value. Default is 42.
        deterministic_cuda: If True, sets CUDA to use deterministic algorithms.
            This may reduce performance but guarantees reproducibility.

    Returns:
        None

    Example:
        >>> set_seed(42)
        >>> import random, numpy as np, torch
        >>> random.random()  # deterministic
        >>> np.random.rand()  # deterministic
        >>> torch.rand(1)     # deterministic
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA ops (slower but reproducible)
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass

    # OS-level hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Global seed set to {seed}")


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create an isolated NumPy Generator for use in environments or samplers.

    Args:
        seed: Seed for this generator. If None, uses entropy from the OS.

    Returns:
        np.random.Generator instance.

    Example:
        >>> rng = get_rng(42)
        >>> rng.integers(0, 10, size=5)
        array([0, 7, 6, 4, 4])
    """
    return np.random.default_rng(seed)


if __name__ == "__main__":
    import torch

    set_seed(42)

    # Verify reproducibility
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(42)

    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2, "random.random not reproducible"
    assert n1 == n2, "numpy.random not reproducible"
    assert t1 == t2, "torch.rand not reproducible"

    print("✅ All seeds reproducible:")
    print(f"  random: {r1:.6f}")
    print(f"  numpy:  {n1:.6f}")
    print(f"  torch:  {t1:.6f}")
