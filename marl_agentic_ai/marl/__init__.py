"""
marl/__init__.py
Top-level package for the MARL Agentic AI framework.
"""

__version__ = "0.1.0"
__author__ = "MARL Research Team"

from marl.envs.cooperative_grid import CooperativeGridWorld
from marl.api import MARLTrainer

__all__ = [
    "CooperativeGridWorld",
    "MARLTrainer",
    "__version__",
]
