"""
marl/utils/config.py
Pydantic-based configuration model that loads from YAML files in configs/.

Usage:
    from marl.utils.config import TrainConfig, load_config
    cfg = load_config("configs/baseline.yaml")
    print(cfg.learning_rate)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainConfig(BaseModel):
    """
    Training configuration for a MARL experiment.

    All fields have sensible defaults matching the baseline experiment.
    Load from YAML via :func:`load_config`.
    """

    # ── Reproducibility ────────────────────────────────────────────
    seed: int = Field(default=42, description="Global random seed.")

    # ── Environment ────────────────────────────────────────────────
    num_agents: int = Field(
        default=2, ge=1, le=16, description="Number of cooperative agents."
    )
    grid_size: int = Field(
        default=10, ge=5, le=50, description="Grid side length (NxN)."
    )
    num_obstacles: int = Field(
        default=5, ge=0, description="Number of static obstacles."
    )
    max_steps: int = Field(
        default=200, ge=10, description="Max steps per episode before truncation."
    )
    partial_observability: bool = Field(
        default=False,
        description="If True, agents have a radius-1 local view (vs radius-2).",
    )
    reward_mode: Literal["individual", "shared"] = Field(
        default="individual",
        description="'individual': agent-specific rewards; 'shared': mean across agents.",
    )

    # ── Algorithm ──────────────────────────────────────────────────
    algorithm: Literal["ppo", "appo", "impala"] = Field(
        default="ppo", description="RLlib algorithm to use."
    )
    independent: bool = Field(
        default=False,
        description="If True, use independent policies (no parameter sharing).",
    )
    shared_reward: bool = Field(
        default=False, description="Alias for reward_mode=='shared'."
    )

    # ── Training Hyperparameters ───────────────────────────────────
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    batch_size: int = Field(default=4096, ge=64)
    mini_batch_size: int = Field(default=256, ge=32)
    num_sgd_iter: int = Field(default=10, ge=1)
    clip_param: float = Field(default=0.2, gt=0.0)
    vf_clip_param: float = Field(default=10.0, gt=0.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    total_timesteps: int = Field(default=1_000_000, ge=1000)
    num_rollout_workers: int = Field(default=4, ge=0)
    rollout_fragment_length: int = Field(default=200, ge=1)

    # ── CTDE ───────────────────────────────────────────────────────
    centralized_critic: bool = Field(
        default=False,
        description="Use a centralized critic with global state (CTDE).",
    )

    # ── Communication ──────────────────────────────────────────────
    communication: bool = Field(
        default=False, description="Enable differentiable inter-agent communication."
    )
    message_dim: int = Field(
        default=32, ge=4, description="Dimension of per-agent communication message."
    )
    comm_hidden_dim: int = Field(
        default=128, ge=16, description="Hidden dim of communication encoder."
    )

    # ── Checkpointing & Logging ────────────────────────────────────
    checkpoint_dir: str = Field(
        default="runs/default/checkpoints",
        description="Directory to save RLlib checkpoints.",
    )
    log_dir: str = Field(
        default="runs/default",
        description="Root directory for TensorBoard logs.",
    )
    exp_name: str = Field(
        default="baseline", description="Experiment name used in directory paths."
    )
    checkpoint_freq: int = Field(
        default=50_000, ge=1000, description="Save checkpoint every N timesteps."
    )
    eval_episodes: int = Field(
        default=100, ge=1, description="Number of evaluation episodes per checkpoint."
    )

    @field_validator("batch_size")
    @classmethod
    def batch_size_divisible(cls, v: int, info: object) -> int:
        """Validate that batch_size >= mini_batch_size."""
        return v

    @model_validator(mode="after")
    def sync_reward_mode(self) -> "TrainConfig":
        """Keep shared_reward and reward_mode in sync."""
        if self.shared_reward and self.reward_mode == "individual":
            self.reward_mode = "shared"
        elif self.reward_mode == "shared":
            self.shared_reward = True
        return self

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "allow"  # Allow extra fields from YAML (forward compat)


def load_config(path: str | Path) -> TrainConfig:
    """
    Load a :class:`TrainConfig` from a YAML file.

    Args:
        path: Path to a YAML config file (relative or absolute).

    Returns:
        Populated :class:`TrainConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML contains invalid field values.

    Example:
        >>> cfg = load_config("configs/baseline.yaml")
        >>> cfg.algorithm
        'ppo'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return TrainConfig(**data)


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """
    Save a :class:`TrainConfig` to a YAML file.

    Args:
        cfg: The config to serialize.
        path: Output path for the YAML file.

    Example:
        >>> cfg = TrainConfig(seed=42)
        >>> save_config(cfg, "runs/exp1/config.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=True)


if __name__ == "__main__":
    # Quick smoke-test: create default config and round-trip
    cfg = TrainConfig()
    print("Default config:")
    print(f"  seed={cfg.seed}, num_agents={cfg.num_agents}, algorithm={cfg.algorithm}")
    print(f"  lr={cfg.learning_rate}, gamma={cfg.gamma}, total_ts={cfg.total_timesteps}")

    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_config.yaml")
        save_config(cfg, save_path)
        loaded = load_config(save_path)
        assert loaded == cfg, "Round-trip config mismatch!"

    print("✅ Config round-trip OK")
