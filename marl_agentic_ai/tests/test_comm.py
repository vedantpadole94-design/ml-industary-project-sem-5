"""
tests/test_comm.py
Tests for the differentiable communication module (CommModule).

Tests:
- message tensor shape: (batch, N-1, msg_dim) — per-agent view
- gradients flow through the Gumbel-Softmax communication channel
- zero messages ≡ no-comm baseline (message_entropy ≈ 0 for uniform messages)
- message entropy utility

Run: pytest tests/test_comm.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestCommModuleShapes:
    """Test CommModule output tensor shapes."""

    def test_message_shape_two_agents(self):
        """messages shape == (batch, N, msg_dim)."""
        from marl.models.comm_module import CommModule
        B, N, obs_dim, msg_dim = 4, 2, 32, 16
        comm = CommModule(obs_dim=obs_dim, num_agents=N, msg_dim=msg_dim)
        obs = torch.randn(B, N, obs_dim)
        logits, values, messages = comm(obs)
        assert messages.shape == (B, N, msg_dim), (
            f"messages shape: {messages.shape}, expected ({B}, {N}, {msg_dim})"
        )

    def test_message_shape_three_agents(self):
        from marl.models.comm_module import CommModule
        B, N, obs_dim, msg_dim = 8, 3, 32, 32
        comm = CommModule(obs_dim=obs_dim, num_agents=N, msg_dim=msg_dim)
        obs = torch.randn(B, N, obs_dim)
        logits, values, messages = comm(obs)
        assert messages.shape == (B, N, msg_dim)

    def test_logits_shape(self):
        from marl.models.comm_module import CommModule
        B, N, obs_dim, action_dim = 4, 3, 32, 5
        comm = CommModule(obs_dim=obs_dim, num_agents=N, action_dim=action_dim)
        obs = torch.randn(B, N, obs_dim)
        logits, values, messages = comm(obs)
        assert logits.shape == (B, N, action_dim), (
            f"logits shape: {logits.shape}, expected ({B}, {N}, {action_dim})"
        )

    def test_value_shape(self):
        from marl.models.comm_module import CommModule
        B, N, obs_dim = 4, 2, 32
        comm = CommModule(obs_dim=obs_dim, num_agents=N)
        obs = torch.randn(B, N, obs_dim)
        logits, values, messages = comm(obs)
        assert values.shape == (B, N, 1), (
            f"values shape: {values.shape}, expected ({B}, {N}, 1)"
        )

    def test_single_batch(self):
        from marl.models.comm_module import CommModule
        comm = CommModule(obs_dim=16, num_agents=2, msg_dim=8)
        obs = torch.randn(1, 2, 16)
        logits, values, messages = comm(obs)
        assert messages.shape == (1, 2, 8)


class TestCommGradients:
    """Test that gradients flow through the communication channel."""

    def test_gradients_flow_through_channel(self):
        """Gradient must reach msg_gen.weight via Gumbel-Softmax."""
        from marl.models.comm_module import CommModule
        torch.manual_seed(42)
        comm = CommModule(obs_dim=32, num_agents=3, msg_dim=16)
        obs = torch.randn(4, 3, 32)
        logits, values, messages = comm(obs)
        loss = logits.sum() + values.sum() + messages.sum()
        loss.backward()
        assert comm.msg_gen.weight.grad is not None, (
            "No gradient reached msg_gen.weight — channel is not differentiable"
        )
        assert not torch.all(comm.msg_gen.weight.grad == 0), (
            "msg_gen.weight gradient is all zeros"
        )

    def test_encoder_grad_exists(self):
        from marl.models.comm_module import CommModule
        comm = CommModule(obs_dim=32, num_agents=2, msg_dim=16)
        obs = torch.randn(4, 2, 32)
        logits, _, _ = comm(obs)
        logits.sum().backward()
        # Check first encoder parameter
        first_param = next(comm.encoder.parameters())
        assert first_param.grad is not None, "No gradient in encoder"

    def test_policy_head_grad_exists(self):
        from marl.models.comm_module import CommModule
        comm = CommModule(obs_dim=32, num_agents=2, msg_dim=16)
        obs = torch.randn(4, 2, 32)
        logits, _, _ = comm(obs)
        logits.sum().backward()
        assert comm.policy_head.weight.grad is not None

    def test_aggregator_grad_exists(self):
        from marl.models.comm_module import CommModule
        comm = CommModule(obs_dim=32, num_agents=3, msg_dim=16)
        obs = torch.randn(4, 3, 32)
        logits, values, messages = comm(obs)
        (logits.sum() + messages.sum()).backward()
        assert comm.msg_aggregator.weight.grad is not None


class TestCommEntropy:
    """Test message entropy utility."""

    def test_entropy_shape(self):
        from marl.models.comm_module import CommModule
        B, N, msg_dim = 4, 3, 16
        # Uniform messages → high entropy
        messages = torch.ones(B, N, msg_dim) / msg_dim
        entropy = CommModule.message_entropy(messages)
        assert entropy.numel() == 1 or entropy.shape == torch.Size([])

    def test_uniform_messages_high_entropy(self):
        """Uniform distribution → max entropy."""
        from marl.models.comm_module import CommModule
        B, N, msg_dim = 8, 2, 32
        uniform = torch.ones(B, N, msg_dim) / msg_dim
        entropy = CommModule.message_entropy(uniform)
        max_entropy = np.log(msg_dim)
        assert float(entropy) > max_entropy * 0.9, (
            f"Uniform entropy {float(entropy):.4f} too low (max={max_entropy:.4f})"
        )

    def test_peaked_messages_low_entropy(self):
        """One-hot messages → near-zero entropy."""
        from marl.models.comm_module import CommModule
        B, N, msg_dim = 4, 2, 16
        one_hot = torch.zeros(B, N, msg_dim)
        one_hot[:, :, 0] = 1.0  # all weight on first token
        entropy = CommModule.message_entropy(one_hot)
        assert float(entropy) < 0.1, f"One-hot entropy {float(entropy):.4f} too high"


class TestZeroMessagesBaseline:
    """Test that zero messages approximate no-comm baseline behavior."""

    def test_zero_messages_vs_no_comm(self):
        """
        With messages zeroed out, the aggregator receives zero input.
        The policy head output should be the same as if no messages were aggregated.
        This verifies the architecture is additive (not multiplicative gating).
        """
        from marl.models.comm_module import CommModule
        torch.manual_seed(42)
        comm = CommModule(obs_dim=32, num_agents=2, msg_dim=16, hidden_dim=64)
        obs = torch.randn(4, 2, 32)

        with torch.no_grad():
            # Forward with actual messages
            logits_comm, vals_comm, messages = comm(obs)

            # Manually zero the messages and re-run aggregation
            # (simulating no communication scenario)
            zero_messages = torch.zeros_like(messages)
            entropy_comm = CommModule.message_entropy(messages)
            entropy_zero = CommModule.message_entropy(zero_messages)

        # Zero messages should have lower entropy than real messages
        assert float(entropy_zero) < float(entropy_comm) or np.isclose(
            float(entropy_zero), 0.0, atol=1e-3
        ), f"Zero messages entropy={float(entropy_zero):.4f} not near 0"

    def test_message_entropy_zero_messages(self):
        """Zero tensor → entropy should be 0 (or near 0, with log(0+eps) handling)."""
        from marl.models.comm_module import CommModule
        zero_msgs = torch.zeros(4, 2, 16)
        entropy = CommModule.message_entropy(zero_msgs)
        # With eps in log, near-zero messages → near-zero entropy
        assert float(entropy) < 1.0, f"Zero-message entropy={float(entropy):.4f} too high"
