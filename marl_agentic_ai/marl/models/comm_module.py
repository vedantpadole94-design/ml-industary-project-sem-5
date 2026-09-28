"""
marl/models/comm_module.py

Differentiable inter-agent communication module.

Pipeline (per forward pass):
    1. Encoder    : each agent's raw observation is projected to a hidden
                    representation via a two-layer MLP.
    2. MsgGen     : a linear layer maps the hidden state to message logits;
                    Gumbel-Softmax makes the discrete channel differentiable.
    3. Channel    : each agent collects all peer messages, concatenates them,
                    and passes them through an aggregator MLP.
    4. Policy head: hidden state + aggregated message -> action logits.
    5. Value head : hidden state + aggregated message -> scalar value.

The module is fully differentiable end-to-end, including through messages,
which allows the communication protocol to be jointly learned with the policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """Two-layer MLP with ReLU activations: in_dim -> hidden_dim -> out_dim."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(),
    )


class CommModule(nn.Module):
    """
    Differentiable communication module for cooperative MARL.

    Architecture:
        Encoder -> MsgGen (Gumbel-Softmax) -> Channel (concat peer msgs)
                -> Aggregator -> Policy head + Value head

    Each agent:
        1. Encodes its own observation to ``hidden_dim`` via a 2-layer MLP.
        2. Generates a ``msg_dim``-dimensional message via a linear layer
           followed by Gumbel-Softmax (straight-through estimator).
        3. Receives messages from all ``num_agents - 1`` peers.
        4. Concatenates its hidden state with the aggregated peer messages.
        5. Produces action logits and a state-value estimate.

    Gradients flow through messages via the Gumbel-Softmax relaxation,
    allowing the communication protocol to be end-to-end differentiable.

    Args:
        obs_dim    : Dimensionality of each agent's observation.
        num_agents : Total number of agents (including self).
        msg_dim    : Dimensionality of each inter-agent message.
        hidden_dim : Width of the encoder's output representation.
        action_dim : Number of discrete actions.
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        msg_dim: int = 32,
        hidden_dim: int = 128,
        action_dim: int = 5,
    ) -> None:
        super().__init__()

        if num_agents < 2:
            raise ValueError("CommModule requires num_agents >= 2.")

        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 1. Encoder: obs_dim -> hidden_dim (two layers via helper)
        self.encoder = _make_mlp(obs_dim, hidden_dim, hidden_dim=128)

        # 2. Message generator: hidden_dim -> msg_dim (logits for Gumbel-Softmax)
        self.msg_gen = nn.Linear(hidden_dim, msg_dim)

        # 3. Message aggregator: msg_dim * (N-1) -> 64
        self.msg_aggregator = nn.Linear(msg_dim * (num_agents - 1), 64)

        # 4. Policy head: hidden_dim + 64 -> action_dim
        self.policy_head = nn.Linear(hidden_dim + 64, action_dim)

        # 5. Value head: hidden_dim + 64 -> 1
        self.value_head = nn.Linear(hidden_dim + 64, 1)

    def forward(
        self,
        obs_batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for all agents in a batch.

        Args:
            obs_batch  : Observations tensor of shape (batch, num_agents, obs_dim).
            temperature: Gumbel-Softmax temperature (lower -> more discrete).

        Returns:
            Tuple of:
                policy_logits : (batch, num_agents, action_dim)
                values        : (batch, num_agents, 1)
                messages      : (batch, num_agents, msg_dim)
        """
        batch, N, obs_d = obs_batch.shape
        assert N == self.num_agents, f"Expected {self.num_agents} agents, got {N}."
        assert obs_d == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {obs_d}."

        # 1. Encode: (B, N, obs_dim) -> (B, N, hidden_dim)
        encoded = self.encoder(obs_batch.view(-1, self.obs_dim))       # (B*N, H)
        encoded = encoded.view(batch, N, self.hidden_dim)              # (B, N, H)

        # 2. Generate messages via Gumbel-Softmax: (B, N, msg_dim)
        msg_logits = self.msg_gen(encoded)                             # (B, N, M)
        messages = F.gumbel_softmax(msg_logits, tau=temperature, hard=False)

        # 3. Channel + Aggregator + Heads (per-agent)
        all_policy_logits: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []

        for i in range(N):
            # Collect peer messages: all j != i -> (B, M*(N-1))
            peer_msgs = torch.cat(
                [messages[:, j, :] for j in range(N) if j != i],
                dim=-1,
            )
            agg = F.relu(self.msg_aggregator(peer_msgs))               # (B, 64)
            combined = torch.cat([encoded[:, i, :], agg], dim=-1)      # (B, H+64)

            all_policy_logits.append(self.policy_head(combined).unsqueeze(1))  # (B,1,A)
            all_values.append(self.value_head(combined).unsqueeze(1))          # (B,1,1)

        policy_logits = torch.cat(all_policy_logits, dim=1)            # (B, N, A)
        values        = torch.cat(all_values,        dim=1)            # (B, N, 1)

        return policy_logits, values, messages

    @staticmethod
    def message_entropy(messages: torch.Tensor) -> torch.Tensor:
        """
        Compute mean Shannon entropy of the message distributions.

        Args:
            messages: (batch, num_agents, msg_dim) -- Gumbel-Softmax output.

        Returns:
            Scalar mean entropy across all agents and batch elements.
        """
        p = messages.clamp(min=0.0)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        return entropy.mean()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    comm = CommModule(obs_dim=32, num_agents=3, msg_dim=16)
    obs = torch.randn(4, 3, 32)  # batch=4, N=3, obs_dim=32

    logits, vals, msgs = comm(obs)

    assert msgs.shape   == (4, 3, 16), f"msgs   shape mismatch: {msgs.shape}"
    assert logits.shape == (4, 3,  5), f"logits shape mismatch: {logits.shape}"
    assert vals.shape   == (4, 3,  1), f"vals   shape mismatch: {vals.shape}"

    print(f"policy_logits shape : {logits.shape}")
    print(f"values        shape : {vals.shape}")
    print(f"messages      shape : {msgs.shape}")

    # Gradient flow check
    loss = logits.sum()
    loss.backward()
    assert comm.msg_gen.weight.grad is not None, "Gradient did not flow to msg_gen.weight!"
    print(f"msg_gen.weight.grad norm : {comm.msg_gen.weight.grad.norm().item():.6f}")

    ent = CommModule.message_entropy(msgs.detach())
    print(f"Message entropy : {ent.item():.4f}")
    print("CommModule smoke-test PASSED")
