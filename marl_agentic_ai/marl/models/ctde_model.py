"""
marl/models/ctde_model.py

CTDEModel: Centralized Training Decentralized Execution via RLlib TorchModelV2

The actor uses only the agent's *local* observation (decentralized execution).
The critic uses a *global* state that concatenates all agents' observations
(centralized training).  At inference time the critic is not needed, so
the architecture is fully compatible with decentralized deployment.

GlobalStateWrapper augments any PettingZoo-compatible environment to include
a 'state' key in every agent's observation dict so that the centralized critic
can consume it during training.
"""

# ---------------------------------------------------------------------------
# Optional Ray / RLlib imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.models import ModelCatalog
    from ray.rllib.utils.annotations import override
    import gymnasium as gym
    ray_available = True
except ImportError:
    ray_available = False

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from gymnasium import spaces as gym_spaces
except ImportError:
    try:
        from gym import spaces as gym_spaces  # type: ignore
    except ImportError:
        gym_spaces = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper: build a plain MLP as nn.Sequential
# ---------------------------------------------------------------------------

def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden: List[int] = [256, 256],
) -> nn.Sequential:
    """
    Build a fully-connected MLP.

    Args:
        input_dim:  Input feature dimensionality.
        output_dim: Output feature dimensionality.
        hidden:     List of hidden layer widths.

    Returns:
        nn.Sequential with Linear -> ReLU blocks followed by a final Linear.
    """
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# CTDEModel -- Ray-aware or plain fallback
# ---------------------------------------------------------------------------

if ray_available:
    _BaseModel = TorchModelV2
else:
    class _BaseModel:  # type: ignore
        """Minimal TorchModelV2 stub used when Ray is not installed."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
            pass

        def forward(self, input_dict, state, seq_lens):
            raise NotImplementedError

        def value_function(self):
            raise NotImplementedError


class CTDEModel(_BaseModel, nn.Module):
    """
    Centralized-Training / Decentralized-Execution model for RLlib.

    Actor  -- uses *local* observations only (safe for deployment).
    Critic -- uses a *global* state (all agents concatenated) during training.

    The global state is injected via ``GlobalStateWrapper`` and arrives in
    ``input_dict['obs']['state']`` during training rollouts.  When no global
    state is available (e.g. pure inference), the critic falls back to the
    local observation.

    Model config keys (``model_config['custom_model_config']``):
        ``global_obs_dim`` (int): Dimensionality of the global state fed to
            the centralized critic.  Defaults to ``local_obs_dim * 2``.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        **kwargs,
    ) -> None:
        # Initialise both parents
        _BaseModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        custom_cfg: Dict[str, Any] = model_config.get("custom_model_config", {})

        # ---- Determine local observation dimension ----------------------
        if hasattr(obs_space, "shape"):
            local_obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, "spaces") and "obs" in obs_space.spaces:
            # Dict observation space -- extract 'obs' sub-space
            local_obs_dim = int(np.prod(obs_space.spaces["obs"].shape))
        else:
            raise ValueError(
                f"Unsupported obs_space type: {type(obs_space)}. "
                "Expected Box or Dict with 'obs' key."
            )

        # ---- Determine global state dimension ---------------------------
        global_obs_dim: int = custom_cfg.get("global_obs_dim", local_obs_dim * 2)

        # ---- Networks ---------------------------------------------------
        self.actor  = make_mlp(local_obs_dim,  num_outputs)
        self.critic = make_mlp(global_obs_dim, 1)

        # Cached tensors set in forward(), consumed by value_function()
        self._value_out:    Optional[torch.Tensor] = None
        self._global_state: Optional[torch.Tensor] = None

    def forward(
        self,
        input_dict: Dict[str, Any],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Decentralized actor forward pass.

        Args:
            input_dict: RLlib sample batch dict.  Must contain
                        ``input_dict['obs_flat']`` (local observation).
                        Optionally contains ``input_dict['obs']['state']``
                        for the centralized critic.
            state:      RNN hidden state (unused for MLP; pass-through).
            seq_lens:   Sequence lengths (unused for MLP; pass-through).

        Returns:
            Tuple of (action_logits, state).
        """
        local_obs: torch.Tensor = input_dict["obs_flat"].float()

        # Capture global state when available (centralized training)
        obs_dict = input_dict.get("obs", {})
        if isinstance(obs_dict, dict) and "state" in obs_dict:
            self._global_state = obs_dict["state"].float()
        else:
            self._global_state = None

        logits = self.actor(local_obs)

        # Store local obs so value_function() has a fallback
        self._value_out = local_obs

        return logits, state

    def value_function(self) -> torch.Tensor:
        """
        Centralized (or local-fallback) value estimate.

        Returns:
            1-D tensor of shape (batch,) with per-sample values.
        """
        if self._global_state is not None:
            return self.critic(self._global_state).squeeze(1)
        assert self._value_out is not None, (
            "value_function() called before forward()."
        )
        return self.critic(self._value_out).squeeze(1)


# ---------------------------------------------------------------------------
# GlobalStateWrapper
# ---------------------------------------------------------------------------


class GlobalStateWrapper:
    """
    PettingZoo-compatible environment wrapper that augments each agent's
    observation with a shared global state vector.

    The global state is the concatenation of all agents' observations and
    is injected under the key ``'state'`` in every agent's observation dict.
    This enables the centralized critic in ``CTDEModel`` to consume full
    environment information during training without changing the agents'
    decentralized execution interface.

    Args:
        env: A PettingZoo AECEnv or ParallelEnv instance.
    """

    def __init__(self, env) -> None:
        self.env = env

        self._agents: List[str] = list(env.possible_agents)

        if self._agents and gym_spaces is not None:
            _sample_obs_space = env.observation_space(self._agents[0])
            _single_obs_dim = int(np.prod(_sample_obs_space.shape))
            self._global_state_dim = _single_obs_dim * len(self._agents)
        else:
            self._global_state_dim = 0

        # Build augmented observation spaces (Dict with 'obs' + 'state')
        self._obs_spaces: Dict[str, Any] = {}
        if gym_spaces is not None:
            for agent in self._agents:
                base_obs_space = env.observation_space(agent)
                self._obs_spaces[agent] = gym_spaces.Dict({
                    "obs": base_obs_space,
                    "state": gym_spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self._global_state_dim,),
                        dtype=np.float32,
                    ),
                })

    def _build_global_state(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all agent observations into a single state vector."""
        parts = [raw_obs[a].flatten() for a in self._agents if a in raw_obs]
        if not parts:
            return np.zeros(self._global_state_dim, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _augment_obs(
        self,
        raw_obs: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Inject the global state into every agent's observation."""
        global_state = self._build_global_state(raw_obs)
        return {
            agent: {"obs": obs, "state": global_state}
            for agent, obs in raw_obs.items()
        }

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset environment and return augmented observations."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            raw_obs, info = result
            return self._augment_obs(raw_obs), info
        return self._augment_obs(result)

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Step environment and return augmented observations."""
        raw_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._augment_obs(raw_obs), rewards, terminations, truncations, infos

    def observation_space(self, agent: str):
        """
        Return the augmented Dict observation space for ``agent``.

        Structure: Dict({'obs': <base_obs_space>, 'state': Box(N*obs_dim)}).
        """
        if agent in self._obs_spaces:
            return self._obs_spaces[agent]
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        """Delegate action space to the underlying environment."""
        return self.env.action_space(agent)

    @property
    def agents(self) -> List[str]:
        """Currently active agents (delegates to underlying env)."""
        return self.env.agents

    @property
    def possible_agents(self) -> List[str]:
        """All possible agents (delegates to underlying env)."""
        return self.env.possible_agents

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    def __getattr__(self, name: str):
        """Delegate any unknown attributes to the wrapped environment."""
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# RLlib model registration
# ---------------------------------------------------------------------------

if ray_available:
    ModelCatalog.register_custom_model("CTDEModel", CTDEModel)


# ---------------------------------------------------------------------------
# Standalone smoke-test (no Ray required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CTDEModel standalone smoke-test (no Ray required)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Minimal obs_space stub (no gymnasium needed)
    class _FakeBox:
        def __init__(self, dim: int):
            self.shape = (dim,)

    local_obs_dim  = 24
    global_obs_dim = 48
    num_outputs    = 5
    batch_size     = 8

    obs_space    = _FakeBox(local_obs_dim)
    action_space = _FakeBox(num_outputs)
    model_config = {"custom_model_config": {"global_obs_dim": global_obs_dim}}

    model = CTDEModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=model_config,
        name="ctde_test",
    )

    local_obs  = torch.randn(batch_size, local_obs_dim)
    global_obs = torch.randn(batch_size, global_obs_dim)

    input_dict = {"obs_flat": local_obs, "obs": {"state": global_obs}}

    logits, state = model.forward(input_dict, state=[], seq_lens=None)
    values        = model.value_function()

    assert logits.shape == (batch_size, num_outputs), f"logits: {logits.shape}"
    assert values.shape == (batch_size,),             f"values: {values.shape}"

    print(f"logits shape (actor)  : {logits.shape}")
    print(f"values shape (critic) : {values.shape}")

    # Gradient check
    loss = logits.sum() + values.sum()
    loss.backward()
    assert model.actor[0].weight.grad  is not None, "Actor grad missing!"
    assert model.critic[0].weight.grad is not None, "Critic grad missing!"
    print(f"actor  grad norm : {model.actor[0].weight.grad.norm().item():.6f}")
    print(f"critic grad norm : {model.critic[0].weight.grad.norm().item():.6f}")

    # Fallback: no global state
    model.zero_grad()
    input_dict_no_state = {"obs_flat": local_obs, "obs": {}}
    logits2, _ = model.forward(input_dict_no_state, state=[], seq_lens=None)
    assert logits2.shape == (batch_size, num_outputs), f"Fallback logits: {logits2.shape}"
    print("Fallback (no global state) actor forward: OK")

    print("\nCTDEModel smoke-test PASSED")
    print(f"Ray available: {ray_available}")
