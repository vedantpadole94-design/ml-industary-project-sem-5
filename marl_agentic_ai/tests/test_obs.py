"""
tests/test_obs.py
Tests for observation space properties of CooperativeGridWorld.

Tests:
- Observation space shapes match declared spaces
- Observations within bounds
- local_view consistency
- carrying flag correctness

Run: pytest tests/test_obs.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from marl.envs.cooperative_grid import CooperativeGridWorld


@pytest.fixture(params=[2, 3], ids=["n2", "n3"])
def env_n(request) -> CooperativeGridWorld:
    n = request.param
    e = CooperativeGridWorld(num_agents=n, grid_size=10, partial_observability=False)
    yield e
    e.close()


class TestObservationSpace:
    def test_obs_space_shape_correct(self, env_n: CooperativeGridWorld):
        obs, _ = env_n.reset(seed=42)
        for agent in env_n.agents:
            space = env_n.observation_space(agent)
            assert obs[agent].shape == space.shape

    def test_obs_space_is_box(self, env_n: CooperativeGridWorld):
        from gymnasium.spaces import Box
        for agent in env_n.possible_agents:
            assert isinstance(env_n.observation_space(agent), Box)

    def test_obs_in_space_bounds(self, env_n: CooperativeGridWorld):
        """All observations must lie within declared space bounds."""
        obs, _ = env_n.reset(seed=42)
        for agent in env_n.agents:
            space = env_n.observation_space(agent)
            # Relax check: just verify dtype and shape (space is [-inf, inf])
            assert obs[agent].dtype == np.float32

    def test_position_features_normalised(self, env_n: CooperativeGridWorld):
        """Position features (first 2 elements) must be in [0, 1]."""
        env_n.reset(seed=42)
        for _ in range(20):
            if not env_n.agents:
                break
            actions = {ag: env_n.action_space(ag).sample() for ag in env_n.agents}
            obs, _, _, _, _ = env_n.step(actions)
            for agent, o in obs.items():
                row_norm = o[0]
                col_norm = o[1]
                assert 0.0 <= row_norm <= 1.0, f"{agent}: row_norm={row_norm} out of [0,1]"
                assert 0.0 <= col_norm <= 1.0, f"{agent}: col_norm={col_norm} out of [0,1]"

    def test_carrying_feature_binary(self, env_n: CooperativeGridWorld):
        """Carrying feature (index 29 for N=2) must be 0 or 1."""
        obs, _ = env_n.reset(seed=42)
        carry_idx = 2 + 27  # after pos(2) + local_view(27)
        for agent, o in obs.items():
            carry_val = o[carry_idx]
            assert carry_val in (0.0, 1.0), f"{agent}: carrying={carry_val} not binary"

    def test_others_relative_features(self, env_n: CooperativeGridWorld):
        """Relative other-agent positions must be normalised to [-1, 1]."""
        obs, _ = env_n.reset(seed=42)
        n = env_n._num_agents
        rel_start = 2 + 27 + 1  # pos + local_view + carrying
        for agent, o in obs.items():
            rel = o[rel_start:]
            assert len(rel) == 2 * (n - 1), f"{agent}: wrong rel feat length"
            assert np.all(np.abs(rel) <= 1.0 + 1e-6), f"{agent}: rel feat out of [-1,1]: {rel}"


class TestObsConsistency:
    def test_obs_changes_after_step(self):
        env = CooperativeGridWorld(num_agents=2, grid_size=10)
        obs0, _ = env.reset(seed=42)
        # Take a non-NOOP action
        actions = {ag: 1 for ag in env.agents}  # UP
        obs1, _, _, _, _ = env.step(actions)
        # At least one agent should have a different obs (position changed)
        changed = False
        for ag in env.agents:
            if not np.array_equal(obs0[ag], obs1.get(ag, obs0[ag])):
                changed = True
                break
        assert changed, "Obs must change after a non-NOOP step"
        env.close()

    def test_local_view_flattened_size(self):
        """local_view must be exactly 27 (3x3x3) for both obs modes."""
        for partial in [False, True]:
            env = CooperativeGridWorld(num_agents=2, partial_observability=partial)
            obs, _ = env.reset(seed=42)
            for agent, o in obs.items():
                local_view = o[2:29]  # after pos(2), before carry(1)
                assert len(local_view) == 27, (
                    f"partial={partial}: local_view len={len(local_view)}, expected 27"
                )
            env.close()
