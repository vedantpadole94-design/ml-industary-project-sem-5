"""
marl/trainers/__init__.py
"""
from marl.trainers.baseline import BaselineTrainer
from marl.trainers.shared_reward import SharedRewardTrainer
from marl.trainers.ctde import CTDETrainer
from marl.trainers.comm import CommTrainer

__all__ = [
    "BaselineTrainer",
    "SharedRewardTrainer",
    "CTDETrainer",
    "CommTrainer",
]
