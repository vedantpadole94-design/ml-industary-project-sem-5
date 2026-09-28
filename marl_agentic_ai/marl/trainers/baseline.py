"""
marl/trainers/baseline.py
Baseline PPO trainer using Ray RLlib with PettingZoo environment.

Supports:
- Parameter sharing (all agents share one policy)
- Independent policies (per-agent separate networks)
- Maps CooperativeGridWorld (PettingZoo Parallel) to RLlib via PettingZooEnv wrapper

Usage:
    from marl.trainers.baseline import BaselineTrainer
    from marl.utils.config import load_config
    cfg = load_config("configs/baseline.yaml")
    trainer = BaselineTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from marl.utils.config import TrainConfig, save_config
from marl.utils.logging import MARLLogger
from marl.utils.seeding import set_seed

logger = logging.getLogger(__name__)


def _make_env_creator(cfg: TrainConfig):
    """
    Return a lambda that creates a PettingZoo-wrapped RLlib env.

    Args:
        cfg: TrainConfig with environment parameters.

    Returns:
        Callable: env_config → PettingZooEnv.
    """
    def env_creator(env_config: Dict[str, Any]):
        from marl.envs.cooperative_grid import CooperativeGridWorld
        try:
            from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        except ImportError:
            from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv as ParallelPettingZooEnv

        env = CooperativeGridWorld(
            num_agents=cfg.num_agents,
            grid_size=cfg.grid_size,
            num_obstacles=cfg.num_obstacles,
            max_steps=cfg.max_steps,
            partial_observability=cfg.partial_observability,
            reward_mode=cfg.reward_mode,
        )
        return ParallelPettingZooEnv(env)

    return env_creator


class BaselineTrainer:
    """
    Baseline MARL trainer using Ray RLlib PPO.

    Supports shared-policy (parameter sharing) or independent policies.
    All agents use the same observation and action spaces.

    Args:
        cfg: :class:`TrainConfig` loaded from a YAML config file.

    Attributes:
        cfg: The training configuration.
        algo: Trained RLlib Algorithm instance (after :meth:`train`).

    Example:
        >>> cfg = load_config("configs/baseline.yaml")
        >>> trainer = BaselineTrainer(cfg)
        >>> result = trainer.train()
        >>> trainer.save_checkpoint("runs/baseline/checkpoints/final")
    """

    ENV_NAME = "cooperative_grid"

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.algo: Optional[Any] = None
        self._logger: Optional[MARLLogger] = None
        self._start_time: float = 0.0

        set_seed(cfg.seed)

    def _register_env(self) -> None:
        """Register the cooperative_grid environment with Ray Tune."""
        import ray
        from ray.tune.registry import register_env

        register_env(self.ENV_NAME, _make_env_creator(self.cfg))
        logger.info(f"Registered env '{self.ENV_NAME}' with RLlib.")

    def _build_config(self) -> Any:
        """
        Build RLlib PPOConfig from TrainConfig.

        Returns:
            Configured PPOConfig object.
        """
        from ray.rllib.algorithms.ppo import PPOConfig

        cfg = self.cfg
        n = cfg.num_agents
        agent_ids = [f"agent_{i}" for i in range(n)]

        # Build multi-agent config
        if cfg.independent:
            # One policy per agent (no parameter sharing)
            policies = {
                f"policy_{i}": (None, None, None, {})
                for i in range(n)
            }
            policy_mapping_fn = lambda agent_id, episode, **kw: (
                f"policy_{int(agent_id.split('_')[1])}"
            )
        else:
            # Shared policy for all agents (parameter sharing)
            policies = {"shared_policy": (None, None, None, {})}
            policy_mapping_fn = lambda agent_id, episode, **kw: "shared_policy"

        algo_cfg = (
            PPOConfig()
            .environment(
                env=self.ENV_NAME,
                env_config={},
            )
            .framework("torch")
            .training(
                lr=cfg.learning_rate,
                gamma=cfg.gamma,
                train_batch_size=cfg.batch_size,
                sgd_minibatch_size=cfg.mini_batch_size,
                num_sgd_iter=cfg.num_sgd_iter,
                clip_param=cfg.clip_param,
                vf_clip_param=cfg.vf_clip_param,
                entropy_coeff=cfg.entropy_coeff,
            )
            .rollouts(
                num_rollout_workers=cfg.num_rollout_workers,
                rollout_fragment_length=cfg.rollout_fragment_length,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .debugging(seed=cfg.seed)
            .resources(
                num_gpus=0,  # override with GPU if available
            )
        )

        return algo_cfg

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop for ``cfg.total_timesteps`` timesteps.

        Checkpoints are saved every ``cfg.checkpoint_freq`` timesteps.
        Config YAML and seed are saved to the checkpoint directory.

        Returns:
            Final training result dict from RLlib.

        Raises:
            RuntimeError: If Ray is not initialised or env registration fails.
        """
        import ray

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        self._register_env()

        # Save config before training
        cfg_path = Path(self.cfg.checkpoint_dir) / "config.yaml"
        save_config(self.cfg, cfg_path)

        # Set up logger
        self._logger = MARLLogger(
            log_dir=self.cfg.log_dir,
            exp_name=self.cfg.exp_name,
        )

        algo_cfg = self._build_config()
        self.algo = algo_cfg.build()

        self._start_time = time.time()
        timesteps_done = 0
        last_checkpoint_ts = 0
        result = {}

        logger.info(
            f"Starting training: {self.cfg.total_timesteps} timesteps, "
            f"checkpoint every {self.cfg.checkpoint_freq}"
        )

        while timesteps_done < self.cfg.total_timesteps:
            result = self.algo.train()
            timesteps_done = result.get("timesteps_total", timesteps_done)

            # Extract metrics
            ep_reward = result.get("episode_reward_mean", 0.0)
            ep_len = result.get("episode_len_mean", 0.0)
            ep_reward_std = result.get("episode_reward_max", 0.0) - result.get("episode_reward_min", 0.0)

            # Log to TensorBoard
            self._logger.log_train(
                step=timesteps_done,
                reward=ep_reward,
                episode_length=ep_len,
                collision_rate=result.get("custom_metrics", {}).get("collision_rate_mean", 0.0),
                reward_variance=result.get("custom_metrics", {}).get("reward_variance_mean", 0.0),
            )

            # Checkpoint
            if timesteps_done - last_checkpoint_ts >= self.cfg.checkpoint_freq:
                self.save_checkpoint(
                    os.path.join(
                        self.cfg.checkpoint_dir,
                        f"checkpoint_{timesteps_done:08d}"
                    )
                )
                last_checkpoint_ts = timesteps_done

            logger.info(
                f"ts={timesteps_done:>8d} | "
                f"reward={ep_reward:.3f} | "
                f"ep_len={ep_len:.1f}"
            )

        # Final checkpoint
        self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "checkpoint_final"))
        self._logger.close()

        wall_time = time.time() - self._start_time
        logger.info(f"Training complete in {wall_time:.1f}s")

        return result

    def save_checkpoint(self, path: str) -> str:
        """
        Save an RLlib checkpoint to *path*.

        Also writes seed.txt alongside the checkpoint.

        Args:
            path: Directory path for the checkpoint.

        Returns:
            Actual checkpoint path (RLlib may add a suffix).
        """
        if self.algo is None:
            raise RuntimeError("No trained algorithm to checkpoint.")
        Path(path).mkdir(parents=True, exist_ok=True)
        ckpt = self.algo.save(path)
        # Save seed
        with open(os.path.join(path, "seed.txt"), "w") as f:
            f.write(str(self.cfg.seed))
        logger.info(f"Checkpoint saved: {ckpt}")
        return ckpt

    def load_checkpoint(self, path: str) -> None:
        """
        Restore weights from an existing checkpoint.

        Args:
            path: Path to the checkpoint directory.
        """
        if self.algo is None:
            algo_cfg = self._build_config()
            self.algo = algo_cfg.build()
        self.algo.restore(path)
        logger.info(f"Checkpoint loaded from: {path}")

    def evaluate(self, episodes: int = 100) -> Dict[str, float]:
        """
        Run evaluation episodes and return metrics.

        Args:
            episodes: Number of evaluation episodes.

        Returns:
            Dict with keys: mean_reward, std_reward, success_rate,
            mean_ep_len, collision_rate.
        """
        if self.algo is None:
            raise RuntimeError("Must train or load checkpoint before evaluating.")

        results = self.algo.evaluate()
        return {
            "mean_reward": results.get("evaluation", {}).get("episode_reward_mean", 0.0),
            "std_reward": 0.0,
            "success_rate": results.get("evaluation", {}).get(
                "custom_metrics", {}
            ).get("success_rate_mean", 0.0),
            "mean_ep_len": results.get("evaluation", {}).get("episode_len_mean", 0.0),
            "collision_rate": 0.0,
        }

    def stop(self) -> None:
        """Shut down the RLlib algorithm and Ray workers."""
        if self.algo is not None:
            self.algo.stop()
        import ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    from marl.utils.config import TrainConfig

    cfg = TrainConfig(
        seed=42,
        num_agents=2,
        total_timesteps=2000,
        checkpoint_freq=1000,
        num_rollout_workers=0,
        batch_size=512,
        mini_batch_size=128,
        checkpoint_dir="runs/smoke_test/checkpoints",
        log_dir="runs/smoke_test",
        exp_name="smoke_test",
    )

    trainer = BaselineTrainer(cfg)
    try:
        result = trainer.train()
        print(f"✅ BaselineTrainer smoke test PASSED. Final reward: {result.get('episode_reward_mean', 'N/A')}")
    finally:
        trainer.stop()
