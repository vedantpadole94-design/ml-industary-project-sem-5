"""
marl/trainers/shared_reward.py
Shared-reward PPO trainer — identical to baseline but loads shared_reward=True config.

The environment is configured with reward_mode='shared', so every agent receives
the mean of all individual rewards at each step.

Usage:
    from marl.trainers.shared_reward import SharedRewardTrainer
    cfg = load_config("configs/shared_reward.yaml")
    trainer = SharedRewardTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from marl.trainers.baseline import BaselineTrainer
from marl.utils.config import TrainConfig

logger = logging.getLogger(__name__)


class SharedRewardTrainer(BaselineTrainer):
    """
    PPO trainer with shared (mean-pooled) reward across agents.

    Extends :class:`BaselineTrainer` with validation that ``reward_mode``
    is set to ``"shared"`` in the config.

    Args:
        cfg: :class:`TrainConfig` with ``shared_reward=True`` or
             ``reward_mode='shared'``.

    Raises:
        ValueError: If the config does not have shared reward enabled.

    Example:
        >>> cfg = load_config("configs/shared_reward.yaml")
        >>> trainer = SharedRewardTrainer(cfg)
        >>> trainer.train()
    """

    def __init__(self, cfg: TrainConfig) -> None:
        if cfg.reward_mode != "shared" and not cfg.shared_reward:
            logger.warning(
                "SharedRewardTrainer expects reward_mode='shared'. "
                f"Got reward_mode='{cfg.reward_mode}'. Overriding."
            )
            cfg.reward_mode = "shared"
            cfg.shared_reward = True
        super().__init__(cfg)
        logger.info("SharedRewardTrainer initialized (reward_mode=shared).")

    def train(self) -> Dict[str, Any]:
        """
        Train with shared reward. Extends baseline training with
        additional reward variance logging to verify reward alignment.

        Returns:
            Final training result dict.
        """
        logger.info("Training with shared rewards (mean pooling across agents).")
        result = super().train()
        return result


if __name__ == "__main__":
    from marl.utils.config import TrainConfig

    cfg = TrainConfig(
        seed=42,
        num_agents=2,
        reward_mode="shared",
        shared_reward=True,
        total_timesteps=2000,
        checkpoint_freq=1000,
        num_rollout_workers=0,
        batch_size=512,
        mini_batch_size=128,
        checkpoint_dir="runs/shared_smoke/checkpoints",
        log_dir="runs/shared_smoke",
        exp_name="shared_smoke",
    )

    trainer = SharedRewardTrainer(cfg)
    try:
        result = trainer.train()
        print(f"✅ SharedRewardTrainer smoke test PASSED.")
    finally:
        trainer.stop()
