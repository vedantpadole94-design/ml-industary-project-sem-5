"""
marl/trainers/ctde.py
Centralized Training Decentralized Execution (CTDE) trainer.

Uses a centralized critic that observes global state (concat of all agent observations)
while each agent's policy only uses its local observation during execution.

Architecture:
- Actor:  MLP(local_obs) → policy logits
- Critic: MLP(global_state = concat(all agent obs)) → V(s)
- Global state is injected via GlobalStateWrapper and the CTDEModel custom model.

Usage:
    from marl.trainers.ctde import CTDETrainer
    cfg = load_config("configs/ctde.yaml")
    trainer = CTDETrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from marl.trainers.baseline import BaselineTrainer, _make_env_creator
from marl.utils.config import TrainConfig
from marl.utils.seeding import set_seed

logger = logging.getLogger(__name__)


def _make_ctde_env_creator(cfg: TrainConfig):
    """
    Create an env creator that wraps the base env with GlobalStateWrapper.

    Args:
        cfg: Training configuration.

    Returns:
        Callable: env_config → wrapped env.
    """
    def env_creator(env_config: Dict[str, Any]):
        from marl.envs.cooperative_grid import CooperativeGridWorld
        from marl.models.ctde_model import GlobalStateWrapper
        try:
            from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        except ImportError:
            from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv as ParallelPettingZooEnv

        base_env = CooperativeGridWorld(
            num_agents=cfg.num_agents,
            grid_size=cfg.grid_size,
            num_obstacles=cfg.num_obstacles,
            max_steps=cfg.max_steps,
            partial_observability=cfg.partial_observability,
            reward_mode=cfg.reward_mode,
        )
        wrapped = GlobalStateWrapper(base_env)
        return ParallelPettingZooEnv(wrapped)

    return env_creator


class CTDETrainer(BaselineTrainer):
    """
    CTDE PPO trainer with centralized critic.

    Inherits from :class:`BaselineTrainer` and overrides env creation
    and algorithm config to use the ``CTDEModel`` custom model.

    The global state (concatenation of all agent observations) is
    passed via the ``state`` key in observations, and the CTDEModel's
    critic uses it while the actor sees only local observations.

    Args:
        cfg: :class:`TrainConfig` with ``centralized_critic=True``.

    Example:
        >>> cfg = load_config("configs/ctde.yaml")
        >>> trainer = CTDETrainer(cfg)
        >>> trainer.train()
    """

    ENV_NAME = "cooperative_grid_ctde"

    def __init__(self, cfg: TrainConfig) -> None:
        if not cfg.centralized_critic:
            logger.warning("CTDETrainer: centralized_critic=False in config. Overriding to True.")
            cfg.centralized_critic = True
        super().__init__(cfg)

    def _register_env(self) -> None:
        """Register the CTDE-wrapped environment with Ray."""
        from ray.tune.registry import register_env

        register_env(self.ENV_NAME, _make_ctde_env_creator(self.cfg))
        logger.info(f"Registered CTDE env '{self.ENV_NAME}' with RLlib.")

    def _build_config(self) -> Any:
        """
        Build PPOConfig with CTDEModel custom model.

        The model config passes ``global_obs_dim`` so the CTDEModel
        critic knows the size of the global state vector.

        Returns:
            Configured PPOConfig.
        """
        from ray.rllib.algorithms.ppo import PPOConfig

        cfg = self.cfg
        n = cfg.num_agents
        local_obs_dim = (
            2                 # pos
            + 3 * 3 * 3       # local view (27)
            + 1               # carrying
            + 2 * (n - 1)     # others relative
        )
        global_obs_dim = local_obs_dim * n

        # Register the CTDEModel
        try:
            from ray.rllib.models import ModelCatalog
            from marl.models.ctde_model import CTDEModel
            ModelCatalog.register_custom_model("CTDEModel", CTDEModel)
        except Exception as e:
            logger.warning(f"Could not register CTDEModel: {e}. Using default model.")

        if cfg.independent:
            policies = {f"policy_{i}": (None, None, None, {}) for i in range(n)}
            policy_mapping_fn = lambda agent_id, episode, **kw: (
                f"policy_{int(agent_id.split('_')[1])}"
            )
        else:
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
                model={
                    "custom_model": "CTDEModel",
                    "custom_model_config": {
                        "global_obs_dim": global_obs_dim,
                        "global_obs_key": "state",
                    },
                },
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
        )

        return algo_cfg


if __name__ == "__main__":
    from marl.utils.config import TrainConfig

    cfg = TrainConfig(
        seed=42,
        num_agents=2,
        reward_mode="shared",
        shared_reward=True,
        centralized_critic=True,
        total_timesteps=5000,  # smoke test: just 5k
        checkpoint_freq=2500,
        num_rollout_workers=0,
        batch_size=512,
        mini_batch_size=128,
        checkpoint_dir="runs/ctde_smoke/checkpoints",
        log_dir="runs/ctde_smoke",
        exp_name="ctde_smoke",
    )

    trainer = CTDETrainer(cfg)
    try:
        result = trainer.train()
        loss = result.get("info", {}).get("learner", {}).get(
            "shared_policy", {}
        ).get("learner_stats", {}).get("total_loss", None)
        print(f"✅ CTDETrainer smoke test PASSED.")
        if loss is not None:
            assert not (loss != loss), f"Loss is NaN: {loss}"
            print(f"   Final loss: {loss:.4f}")
    finally:
        trainer.stop()
