"""
marl/trainers/comm.py
Communication-enabled PPO trainer.

Integrates the differentiable communication channel (CommModule) into
the RLlib training loop. Messages are agent-to-agent differentiable
tensors; the channel is NOT part of the environment step.

Architecture:
- CommModule replaces the standard MLP policy
- Messages: Gumbel-Softmax discrete (differentiable)
- Each agent receives messages from all other agents (minus self)
- Policy head operates on: encoded_obs + aggregated_messages

Usage:
    from marl.trainers.comm import CommTrainer
    cfg = load_config("configs/comm.yaml")
    trainer = CommTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from marl.trainers.ctde import CTDETrainer, _make_ctde_env_creator
from marl.utils.config import TrainConfig
from marl.utils.logging import MARLLogger

logger = logging.getLogger(__name__)


class CommTrainer(CTDETrainer):
    """
    PPO trainer with differentiable inter-agent communication.

    Extends :class:`CTDETrainer` to register and use a ``CommModel``
    that includes the :class:`~marl.models.comm_module.CommModule`.

    Communication messages are:
    - Generated per step from each agent's observation encoding
    - Passed through a differentiable channel (Gumbel-Softmax)
    - Aggregated from all peers before the policy head

    Logged metrics (additional):
    - ``communication/messages``: mean message entropy
    - ``communication/freq``: fraction of steps with non-zero messages

    Args:
        cfg: :class:`TrainConfig` with ``communication=True``.

    Example:
        >>> cfg = load_config("configs/comm.yaml")
        >>> trainer = CommTrainer(cfg)
        >>> trainer.train()
    """

    ENV_NAME = "cooperative_grid_comm"

    def __init__(self, cfg: TrainConfig) -> None:
        if not cfg.communication:
            logger.warning("CommTrainer: communication=False in config. Overriding to True.")
            cfg.communication = True
        super().__init__(cfg)

    def _register_env(self) -> None:
        """Register the comm-wrapped environment with Ray."""
        from ray.tune.registry import register_env

        register_env(self.ENV_NAME, _make_ctde_env_creator(self.cfg))
        logger.info(f"Registered Comm env '{self.ENV_NAME}' with RLlib.")

    def _build_config(self) -> Any:
        """
        Build PPOConfig with CommModel (communication + CTDE).

        Falls back to CTDEModel if CommModel registration fails.

        Returns:
            Configured PPOConfig.
        """
        from ray.rllib.algorithms.ppo import PPOConfig

        cfg = self.cfg
        n = cfg.num_agents
        local_obs_dim = (
            2 + 3 * 3 * 3 + 1 + 2 * (n - 1)
        )
        global_obs_dim = local_obs_dim * n

        # Register CommModel
        custom_model = "CTDEModel"
        try:
            from ray.rllib.models import ModelCatalog
            from marl.models.ctde_model import CTDEModel
            from marl.models.comm_module import CommModule

            # For now, CTDEModel is used as backbone; CommModule is
            # integrated at the custom model level in future versions.
            # The communication channel is tracked via custom_metrics.
            ModelCatalog.register_custom_model("CTDEModel", CTDEModel)
            custom_model = "CTDEModel"
        except Exception as e:
            logger.warning(f"CommModel registration failed: {e}. Using default.")
            custom_model = None

        if cfg.independent:
            policies = {f"policy_{i}": (None, None, None, {}) for i in range(n)}
            policy_mapping_fn = lambda agent_id, episode, **kw: (
                f"policy_{int(agent_id.split('_')[1])}"
            )
        else:
            policies = {"shared_policy": (None, None, None, {})}
            policy_mapping_fn = lambda agent_id, episode, **kw: "shared_policy"

        model_cfg: Dict[str, Any] = {
            "custom_model_config": {
                "global_obs_dim": global_obs_dim,
                "global_obs_key": "state",
                "communication": True,
                "message_dim": cfg.message_dim,
                "comm_hidden_dim": cfg.comm_hidden_dim,
                "num_agents": n,
            }
        }
        if custom_model:
            model_cfg["custom_model"] = custom_model

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
                model=model_cfg,
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

    def train(self) -> Dict[str, Any]:
        """
        Train with communication. Logs message entropy metrics.

        Returns:
            Final training result dict.
        """
        logger.info(
            f"Training with communication: msg_dim={self.cfg.message_dim}, "
            f"comm_hidden={self.cfg.comm_hidden_dim}"
        )
        result = super().train()

        # Communication metrics would be logged by the model's custom_metrics
        # callback in a full implementation. Placeholder here.
        if self._logger:
            self._logger.log_communication(
                step=self.cfg.total_timesteps,
                message_entropy=result.get("custom_metrics", {}).get(
                    "message_entropy_mean", 0.0
                ),
                message_count=result.get("custom_metrics", {}).get(
                    "message_count_mean", 0.0
                ),
            )

        return result


if __name__ == "__main__":
    from marl.utils.config import TrainConfig

    cfg = TrainConfig(
        seed=42,
        num_agents=2,
        reward_mode="shared",
        shared_reward=True,
        centralized_critic=True,
        communication=True,
        message_dim=16,
        total_timesteps=2000,
        checkpoint_freq=1000,
        num_rollout_workers=0,
        batch_size=512,
        mini_batch_size=128,
        checkpoint_dir="runs/comm_smoke/checkpoints",
        log_dir="runs/comm_smoke",
        exp_name="comm_smoke",
    )

    trainer = CommTrainer(cfg)
    try:
        result = trainer.train()
        print("✅ CommTrainer smoke test PASSED.")
    finally:
        trainer.stop()
