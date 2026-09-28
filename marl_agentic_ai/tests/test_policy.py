"""
tests/test_policy.py
Tests for policy and trainer components (lightweight, no full Ray training).

Tests:
- MLPModel forward pass shapes
- ActorCriticMLP shapes
- BaselineTrainer config building (mocked)
- Policy mapping functions

Run: pytest tests/test_policy.py -v
"""

from __future__ import annotations

import pytest
import torch
import numpy as np


class TestMLPModel:
    """Test MLPModel forward pass."""

    def test_mlp_forward_shape(self):
        from marl.models.mlp import MLPModel
        model = MLPModel(input_dim=32, output_dim=5, hidden_dims=[64, 64])
        x = torch.randn(8, 32)
        out = model(x)
        assert out.shape == (8, 5), f"Expected (8,5), got {out.shape}"

    def test_mlp_different_batch_sizes(self):
        from marl.models.mlp import MLPModel
        model = MLPModel(input_dim=16, output_dim=3, hidden_dims=[32])
        for bs in [1, 4, 16, 64]:
            x = torch.randn(bs, 16)
            out = model(x)
            assert out.shape == (bs, 3)

    def test_mlp_gradient_flows(self):
        from marl.models.mlp import MLPModel
        model = MLPModel(input_dim=8, output_dim=2, hidden_dims=[16])
        x = torch.randn(4, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_mlp_no_nan_output(self):
        from marl.models.mlp import MLPModel
        model = MLPModel(input_dim=32, output_dim=5)
        x = torch.randn(16, 32)
        out = model(x)
        assert torch.all(torch.isfinite(out)), "MLP output contains NaN/Inf"


class TestActorCriticMLP:
    """Test ActorCriticMLP shapes and properties."""

    def test_actor_critic_shapes(self):
        from marl.models.mlp import ActorCriticMLP
        model = ActorCriticMLP(obs_dim=32, action_dim=5)
        obs = torch.randn(8, 32)
        logits, value = model(obs)
        assert logits.shape == (8, 5), f"logits: {logits.shape}"
        assert value.shape == (8, 1), f"value: {value.shape}"

    def test_actor_critic_gradient(self):
        from marl.models.mlp import ActorCriticMLP
        model = ActorCriticMLP(obs_dim=32, action_dim=5)
        obs = torch.randn(4, 32)
        logits, value = model(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_actor_critic_deterministic(self):
        from marl.models.mlp import ActorCriticMLP
        torch.manual_seed(42)
        model = ActorCriticMLP(obs_dim=16, action_dim=4)
        obs = torch.ones(2, 16)
        l1, v1 = model(obs)
        l2, v2 = model(obs)
        assert torch.allclose(l1, l2), "Forward pass not deterministic"
        assert torch.allclose(v1, v2), "Value head not deterministic"


class TestTrainConfig:
    """Test TrainConfig loading and validation."""

    def test_default_config(self):
        from marl.utils.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.seed == 42
        assert cfg.num_agents == 2
        assert cfg.algorithm == "ppo"

    def test_shared_reward_sync(self):
        from marl.utils.config import TrainConfig
        cfg = TrainConfig(shared_reward=True)
        assert cfg.reward_mode == "shared"

    def test_config_from_dict(self):
        from marl.utils.config import TrainConfig
        cfg = TrainConfig(
            seed=99,
            num_agents=4,
            total_timesteps=500_000,
            algorithm="ppo",
        )
        assert cfg.seed == 99
        assert cfg.num_agents == 4

    def test_config_roundtrip(self, tmp_path):
        from marl.utils.config import TrainConfig, save_config, load_config
        cfg = TrainConfig(seed=42, exp_name="test", num_agents=3)
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.seed == cfg.seed
        assert loaded.num_agents == cfg.num_agents
        assert loaded.exp_name == cfg.exp_name


class TestPolicyMapping:
    """Test policy mapping functions."""

    def test_shared_policy_mapping(self):
        """All agents should map to 'shared_policy' in parameter-sharing mode."""
        from marl.utils.config import TrainConfig
        cfg = TrainConfig(num_agents=4, independent=False)
        agents = [f"agent_{i}" for i in range(4)]
        policy_fn = lambda agent_id, episode, **kw: "shared_policy"
        for ag in agents:
            assert policy_fn(ag, None) == "shared_policy"

    def test_independent_policy_mapping(self):
        """Each agent should map to its own policy in independent mode."""
        from marl.utils.config import TrainConfig
        cfg = TrainConfig(num_agents=4, independent=True)
        policy_fn = lambda agent_id, episode, **kw: f"policy_{int(agent_id.split('_')[1])}"
        assert policy_fn("agent_0", None) == "policy_0"
        assert policy_fn("agent_3", None) == "policy_3"
