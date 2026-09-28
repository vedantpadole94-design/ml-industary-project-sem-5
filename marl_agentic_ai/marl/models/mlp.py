"""
marl/models/mlp.py
Generic configurable MLP building blocks for actor/critic networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


class MLPModel(nn.Module):
    """Generic configurable MLP. Used as building block for actor/critic networks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Build a fully-connected MLP.

        Args:
            input_dim:   Dimensionality of the input tensor.
            output_dim:  Dimensionality of the final output.
            hidden_dims: List of hidden layer widths.
            activation:  Activation function name ('relu', 'tanh', 'elu', etc.).
            dropout:     Dropout probability after each hidden activation (0.0 = off).
        """
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from: {list(_ACTIVATIONS.keys())}"
            )
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (..., input_dim) -> output (..., output_dim)."""
        return self.net(x)


class ActorCriticMLP(nn.Module):
    """Shared-body actor-critic MLP for single-agent PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ) -> None:
        """
        Shared body + actor/critic heads.

        Args:
            obs_dim:     Observation dimensionality.
            action_dim:  Number of discrete actions.
            hidden_dims: Hidden widths; last entry is trunk output width.
        """
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one element.")
        self.body = MLPModel(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation="relu",
            dropout=0.0,
        )
        self.actor_head = nn.Linear(hidden_dims[-1], action_dim)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)

        Returns:
            policy_logits: (batch, action_dim)
            value:         (batch, 1)
        """
        features = self.body(obs)
        return self.actor_head(features), self.critic_head(features)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ActorCriticMLP(obs_dim=32, action_dim=5)
    obs_batch = torch.randn(8, 32)
    logits, value = model(obs_batch)
    assert logits.shape == (8, 5), f"Expected (8,5) got {logits.shape}"
    assert value.shape == (8, 1),  f"Expected (8,1) got {value.shape}"
    print(f"logits shape : {logits.shape}")
    print(f"value  shape : {value.shape}")
    print("ActorCriticMLP smoke-test PASSED")
