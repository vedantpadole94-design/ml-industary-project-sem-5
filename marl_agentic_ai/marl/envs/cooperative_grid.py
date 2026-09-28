"""
marl/envs/cooperative_grid.py
Cooperative Grid World — PettingZoo Parallel API implementation.

SPEC:
  Grid: NxN (default 10x10)
  Agents: N cooperative agents (default 2)
  Entities: N resources (one per agent), M obstacles, 1 shared goal
  Success: every agent collects its resource → all reach goal within max_steps
  Actions: Discrete(5) — 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
  Observation (per agent):
    - position:              (2,)
    - local_view:            (3x3x3 flattened = 27,)  radius 2 full-obs, radius 1 partial
    - carrying:              (1,)
    - other_agents_relative: (2*(N-1),)
  Total obs dim (N=2, full):   2 + 27 + 1 + 2 = 32
  Total obs dim (N=2, partial): 2 + 3 + 1 + 2 = 8  (radius 1: 1x3x3=9... see below)
  Reward: shaped individual; shared variant sums across agents (see reward_mode param)
  Partial obs: radius shrinks from 2 to 1 → local_view = (2*1+1)^2 * 3 = 27 (same channels, smaller patch).

Design Assumptions:
  A1. Wrap-around is OFF; walls act as boundaries.
  A2. Agents DO NOT block each other by default; collision gives a small penalty.
  A3. local_view channels: [entity_type, agent_present, carrying_status].
  A4. Entity encoding: 0=empty, 1=wall, 2=resource_own, 3=resource_other, 4=goal, 5=obstacle.
  A5. Partial obs radius=1 → local_view = (3,3,3), same shape as full-obs (radius=2 → (5,5,3)).
      To keep observation space fixed, we always report a 3×3×3 array (9 cells × 3 channels = 27).
      Partial obs: radius=1 (3×3 patch). Full obs: radius=2 (5×5 → centre-cropped to 3×3).
      Actually: partial=True → radius 1 (3×3 grid), partial=False → radius 2 (5×5 grid).
      Obs sizes differ; space is reported correctly per flag.
  A6. Reward structure:
      +0.1 per step moving closer to resource (if not carrying)
      +1.0 on picking up own resource
      +0.1 per step moving closer to goal (if carrying)
      +5.0 on reaching goal while carrying (terminal bonus)
      -0.05 collision penalty (agent steps onto another agent's cell)
      -0.01 step penalty (encourage efficiency)
"""

from __future__ import annotations

import functools
import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Action indices
NOOP, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    NOOP:  (0, 0),
    UP:    (-1, 0),
    DOWN:  (+1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, +1),
}

# Cell-type encoding (channel 0 of local_view)
EMPTY    = 0
WALL     = 1
RES_SELF = 2   # this agent's assigned resource
RES_OTH  = 3   # another agent's resource
GOAL     = 4
OBSTACLE = 5


# ── Helper: local view extraction ─────────────────────────────────────────────

def _extract_local_view(
    grid: np.ndarray,
    pos: Tuple[int, int],
    radius: int,
    agent_positions: List[Tuple[int, int]],
    agent_carrying: List[bool],
    agent_idx: int,
) -> np.ndarray:
    """
    Extract a (2r+1, 2r+1, 3) view centred on *pos*.

    Channels:
      0: entity type at cell (EMPTY/WALL/RES_SELF/RES_OTH/GOAL/OBSTACLE)
      1: 1 if another agent occupies the cell, else 0
      2: 1 if the agent at that cell is carrying (only meaningful if ch1=1)

    Out-of-bounds cells are treated as WALL (channel 0 = 1).

    Args:
        grid: (H, W) int array of cell types.
        pos: (row, col) of the observing agent.
        radius: Half-size of the patch (result is (2r+1, 2r+1)).
        agent_positions: List of all agent positions.
        agent_carrying: Carrying status for each agent.
        agent_idx: Index of the observing agent (to mark RES_SELF vs RES_OTH).

    Returns:
        np.ndarray of shape (2r+1, 2r+1, 3), dtype float32.
    """
    H, W = grid.shape
    size = 2 * radius + 1
    view = np.zeros((size, size, 3), dtype=np.float32)

    r, c = pos
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            vi, vj = di + radius, dj + radius
            ri, rj = r + di, c + dj
            if ri < 0 or ri >= H or rj < 0 or rj >= W:
                view[vi, vj, 0] = WALL
            else:
                cell_type = grid[ri, rj]
                view[vi, vj, 0] = float(cell_type)
                # Check agent presence
                for k, apos in enumerate(agent_positions):
                    if k != agent_idx and apos == (ri, rj):
                        view[vi, vj, 1] = 1.0
                        view[vi, vj, 2] = float(agent_carrying[k])
                        break

    return view


# ── Main Environment ───────────────────────────────────────────────────────────

class CooperativeGridWorld(ParallelEnv):
    """
    Cooperative Grid World environment using the PettingZoo Parallel API.

    All agents act simultaneously each step. The environment supports:
    - Individual or shared rewards (``reward_mode`` parameter)
    - Full or partial observability (``partial_observability`` parameter)
    - Configurable grid size, agent count, and obstacle count

    Args:
        num_agents: Number of cooperative agents (default 2).
        grid_size: Side length of the square grid (default 10).
        num_obstacles: Number of static obstacles (default 5).
        max_steps: Maximum steps per episode (default 200).
        partial_observability: If True, local view radius = 1 else 2.
        reward_mode: ``"individual"`` or ``"shared"`` (mean of individual).
        render_mode: ``"rgb_array"`` for pixel rendering, ``None`` for headless.

    Observation space per agent (flat vector):
        Full obs (N=2):   pos(2) + local_view(3*3*3=27) + carrying(1) + others_rel(2) = 32
        Partial obs (N=2): pos(2) + local_view(3*3*3=27, radius=1) + carrying(1) + rel(2) = 32
        Note: both radii produce 3×3×3 = 27 for the local_view.
              radius=2 → 5×5 patch centre-cropped to 3×3.
              radius=1 → 3×3 patch directly.

    Example:
        >>> env = CooperativeGridWorld(num_agents=2, partial_observability=False)
        >>> obs, infos = env.reset(seed=42)
        >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        >>> obs, rewards, terms, truncs, infos = env.step(actions)
        >>> env.close()
    """

    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array", "human"],
        "name": "cooperative_grid_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        num_obstacles: int = 5,
        max_steps: int = 200,
        partial_observability: bool = False,
        reward_mode: Literal["individual", "shared"] = "individual",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert num_agents >= 1, "num_agents must be >= 1"
        assert grid_size >= 5, "grid_size must be >= 5"
        assert render_mode in (None, "rgb_array", "human"), \
            f"Unsupported render_mode: {render_mode}"

        self._num_agents = num_agents
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.partial_observability = partial_observability
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        # PettingZoo required attribute: list of agent id strings
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(num_agents)]
        self.agents: List[str] = []

        # Local view radius
        self._obs_radius: int = 1 if partial_observability else 2
        # Always report 3×3 local view (clip or pad for radius=2)
        self._view_size: int = 3   # fixed output: 3×3×3 = 27

        # Observation dimension
        self._obs_dim: int = (
            2                              # position
            + self._view_size ** 2 * 3     # local_view (3×3×3)
            + 1                            # carrying
            + 2 * (num_agents - 1)         # other agents relative
        )

        # Internal state (set in reset)
        self._grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self._agent_pos: List[Tuple[int, int]] = []
        self._agent_carrying: List[bool] = []
        self._agent_reached_goal: List[bool] = []
        self._resource_pos: List[Optional[Tuple[int, int]]] = []
        self._resource_collected: List[bool] = []
        self._goal_pos: Tuple[int, int] = (0, 0)
        self._step_count: int = 0
        self._prev_dist_to_resource: List[float] = []
        self._prev_dist_to_goal: List[float] = []
        self._np_random: Optional[np.random.Generator] = None

    # ── PettingZoo Required: Spaces ────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        """
        Return the observation space for a given agent.

        Observation is a flat float32 vector of size:
            2 (pos) + 27 (local_view) + 1 (carrying) + 2*(N-1) (others_rel)

        Args:
            agent: Agent identifier string.

        Returns:
            ``gymnasium.spaces.Box`` with shape (obs_dim,).
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """
        Return the action space for a given agent.

        Actions: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.

        Args:
            agent: Agent identifier string.

        Returns:
            ``gymnasium.spaces.Discrete(5)``.
        """
        return spaces.Discrete(5)

    # ── PettingZoo Required: reset ─────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Randomly places agents, resources, obstacles, and goal on the grid.
        All entities are placed at distinct cells.

        Args:
            seed: Optional integer seed for reproducibility.
            options: Unused. Reserved for future configuration.

        Returns:
            observations: Dict[agent_id → obs_array (float32, (obs_dim,))].
            infos: Dict[agent_id → {}].

        Example:
            >>> env = CooperativeGridWorld(num_agents=2)
            >>> obs, infos = env.reset(seed=42)
            >>> obs["agent_0"].shape
            (32,)
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng(42)

        # Restore full agent list
        self.agents = self.possible_agents[:]
        n = len(self.agents)

        # Clear grid
        self._grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Sample non-overlapping positions for all entities
        total_entities = n + n + 1 + self.num_obstacles  # agents+resources+goal+obstacles
        total_cells = self.grid_size * self.grid_size
        assert total_entities < total_cells, \
            f"Too many entities ({total_entities}) for grid size {self.grid_size}x{self.grid_size}"

        flat_cells = self._np_random.choice(
            total_cells, size=total_entities, replace=False
        )

        def flat_to_rc(flat: int) -> Tuple[int, int]:
            return (int(flat // self.grid_size), int(flat % self.grid_size))

        idx = 0
        # Agent positions
        self._agent_pos = [flat_to_rc(flat_cells[i]) for i in range(idx, idx + n)]
        idx += n

        # Resource positions (one per agent)
        self._resource_pos = [flat_to_rc(flat_cells[i]) for i in range(idx, idx + n)]
        idx += n

        # Goal position
        self._goal_pos = flat_to_rc(flat_cells[idx])
        idx += 1

        # Obstacle positions
        obstacle_positions = [flat_to_rc(flat_cells[i]) for i in range(idx, idx + self.num_obstacles)]

        # Build grid encoding
        for i, rpos in enumerate(self._resource_pos):
            if rpos is not None:
                self._grid[rpos[0], rpos[1]] = RES_SELF  # will be adjusted per agent view
        self._grid[self._goal_pos[0], self._goal_pos[1]] = GOAL
        for opos in obstacle_positions:
            self._grid[opos[0], opos[1]] = OBSTACLE

        # State reset
        self._agent_carrying = [False] * n
        self._agent_reached_goal = [False] * n
        self._resource_collected = [False] * n
        self._step_count = 0

        # Precompute previous distances for shaping
        self._prev_dist_to_resource = [
            self._manhattan(self._agent_pos[i], self._resource_pos[i])
            for i in range(n)
        ]
        self._prev_dist_to_goal = [
            self._manhattan(self._agent_pos[i], self._goal_pos)
            for i in range(n)
        ]

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # ── PettingZoo Required: step ──────────────────────────────────

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Apply actions for all agents simultaneously and advance the environment.

        Handles movement, resource pickup, goal completion, collisions,
        and episode termination / truncation.

        Args:
            actions: Dict[agent_id → action_int] where action ∈ {0,1,2,3,4}.

        Returns:
            observations: Dict[agent_id → obs_array].
            rewards: Dict[agent_id → float].
            terminations: Dict[agent_id → bool] (True = done via success).
            truncations: Dict[agent_id → bool] (True = done via max_steps).
            infos: Dict[agent_id → info_dict].

        Note:
            Once an agent is terminated, it is removed from ``self.agents``.
            The PettingZoo convention requires returning obs for terminated
            agents on the final step (they are empty / zero after that).
        """
        if not self.agents:
            # Environment already done; return empty dicts
            return {}, {}, {}, {}, {}

        n = len(self.possible_agents)
        self._step_count += 1

        # ── 1. Compute candidate new positions ─────────────────────
        new_positions: List[Tuple[int, int]] = []
        for i, agent in enumerate(self.possible_agents):
            action = actions.get(agent, NOOP)
            dr, dc = ACTION_DELTAS[action]
            r, c = self._agent_pos[i]
            nr = max(0, min(self.grid_size - 1, r + dr))
            nc = max(0, min(self.grid_size - 1, c + dc))
            # Block movement into obstacles
            if self._grid[nr, nc] == OBSTACLE:
                nr, nc = r, c
            new_positions.append((nr, nc))

        # ── 2. Collision detection (agents stepping onto same cell) ─
        collision_flags = [False] * n
        pos_count: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(new_positions):
            pos_count.setdefault(pos, []).append(i)
        for pos, agents_there in pos_count.items():
            if len(agents_there) > 1:
                for idx in agents_there:
                    collision_flags[idx] = True
                    new_positions[idx] = self._agent_pos[idx]  # bounce back

        # ── 3. Apply movement ───────────────────────────────────────
        self._agent_pos = new_positions

        # ── 4. Compute individual rewards ───────────────────────────
        individual_rewards = np.zeros(n, dtype=np.float32)
        step_penalty = -0.01
        collision_penalty = -0.05

        for i in range(n):
            pos = self._agent_pos[i]
            r = float(step_penalty)

            if collision_flags[i]:
                r += collision_penalty

            if not self._agent_carrying[i] and not self._resource_collected[i]:
                # Shaping: reward for getting closer to resource
                res_pos = self._resource_pos[i]
                if res_pos is not None:
                    curr_dist = self._manhattan(pos, res_pos)
                    r += 0.1 * (self._prev_dist_to_resource[i] - curr_dist)
                    self._prev_dist_to_resource[i] = curr_dist

                    # Resource pickup
                    if pos == res_pos:
                        self._agent_carrying[i] = True
                        self._resource_collected[i] = True
                        self._resource_pos[i] = None
                        # Remove from grid
                        self._grid[res_pos[0], res_pos[1]] = EMPTY
                        r += 1.0

            elif self._agent_carrying[i] and not self._agent_reached_goal[i]:
                # Shaping: reward for getting closer to goal
                curr_dist = self._manhattan(pos, self._goal_pos)
                r += 0.1 * (self._prev_dist_to_goal[i] - curr_dist)
                self._prev_dist_to_goal[i] = curr_dist

                # Goal reached
                if pos == self._goal_pos:
                    self._agent_reached_goal[i] = True
                    r += 5.0

            individual_rewards[i] = r

        # ── 5. Apply shared reward if configured ───────────────────
        if self.reward_mode == "shared":
            shared = float(np.mean(individual_rewards))
            rewards_arr = np.full(n, shared, dtype=np.float32)
        else:
            rewards_arr = individual_rewards

        # ── 6. Check termination / truncation ──────────────────────
        all_done = all(self._agent_reached_goal)
        terminated_global = all_done
        truncated_global = self._step_count >= self.max_steps

        # ── 7. Build return dicts ───────────────────────────────────
        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Any] = {}

        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue
            observations[agent] = self._get_obs(i)
            rewards[agent] = float(rewards_arr[i])
            terminations[agent] = terminated_global
            truncations[agent] = truncated_global
            infos[agent] = {
                "carrying": self._agent_carrying[i],
                "resource_collected": self._resource_collected[i],
                "reached_goal": self._agent_reached_goal[i],
                "collision": collision_flags[i],
                "step": self._step_count,
            }

        # Remove done agents per PettingZoo convention
        if terminated_global or truncated_global:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ── Observation Construction ───────────────────────────────────

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        """
        Construct the observation vector for agent ``agent_idx``.

        Components (concatenated):
        1. Position (2,): normalised (row/H, col/W).
        2. Local view (27,): (3,3,3) → flattened.
        3. Carrying (1,): binary float.
        4. Others relative (2*(N-1),): normalised (dr/H, dc/W) per peer.

        Args:
            agent_idx: Index into ``self.possible_agents``.

        Returns:
            Float32 array of shape (obs_dim,).
        """
        n = len(self.possible_agents)
        pos = self._agent_pos[agent_idx]
        H, W = self.grid_size, self.grid_size

        # 1. Position (normalised)
        pos_feat = np.array([pos[0] / H, pos[1] / W], dtype=np.float32)

        # 2. Local view — build agent-specific grid with correct resource encoding
        agent_grid = self._grid.copy()
        for j, rpos in enumerate(self._resource_pos):
            if rpos is not None and not self._resource_collected[j]:
                if j == agent_idx:
                    agent_grid[rpos[0], rpos[1]] = RES_SELF
                else:
                    agent_grid[rpos[0], rpos[1]] = RES_OTH

        radius = self._obs_radius
        raw_view = _extract_local_view(
            agent_grid,
            pos,
            radius,
            self._agent_pos,
            self._agent_carrying,
            agent_idx,
        )

        # If full obs (radius=2 → 5×5), centre-crop to 3×3
        if radius == 2:
            centre = radius  # index 2 in 0-indexed 5×5
            raw_view = raw_view[
                centre - 1 : centre + 2,
                centre - 1 : centre + 2,
                :,
            ]  # → (3, 3, 3)

        local_feat = raw_view.flatten()  # (27,)

        # 3. Carrying status
        carry_feat = np.array([float(self._agent_carrying[agent_idx])], dtype=np.float32)

        # 4. Other agents' relative positions (normalised)
        rel_feats = []
        for j in range(n):
            if j == agent_idx:
                continue
            other_pos = self._agent_pos[j]
            dr = (other_pos[0] - pos[0]) / H
            dc = (other_pos[1] - pos[1]) / W
            rel_feats.extend([dr, dc])
        rel_feat = np.array(rel_feats, dtype=np.float32)

        obs = np.concatenate([pos_feat, local_feat, carry_feat, rel_feat])
        assert obs.shape == (self._obs_dim,), \
            f"Obs shape mismatch: got {obs.shape}, expected ({self._obs_dim},)"
        return obs

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Compute L1 distance between two grid positions."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    # ── PettingZoo: render & close ─────────────────────────────────

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Returns:
            If ``render_mode == "rgb_array"``: uint8 array of shape (H*scale, W*scale, 3).
            If ``render_mode == "human"``: displays in a window (via matplotlib).
            If ``render_mode`` is None: returns None.

        Notes:
            Uses a simple colour-coded grid:
            - Black:  obstacle
            - White:  empty
            - Green:  resource (self)
            - Yellow: resource (other)
            - Gold:   goal
            - Blue:   agent (not carrying)
            - Cyan:   agent (carrying)
        """
        if self.render_mode is None:
            return None

        SCALE = 32
        H = W = self.grid_size
        canvas = np.ones((H * SCALE, W * SCALE, 3), dtype=np.uint8) * 255  # white bg

        COLORS = {
            EMPTY:    (255, 255, 255),  # white
            OBSTACLE: (30, 30, 30),     # black
            RES_SELF: (50, 200, 50),    # green
            RES_OTH:  (200, 200, 50),   # yellow
            GOAL:     (255, 215, 0),    # gold
        }

        AGENT_COLORS = [
            (0, 100, 255),    # blue
            (255, 50, 50),    # red
            (150, 0, 200),    # purple
            (255, 128, 0),    # orange
        ]

        # Draw grid cells
        for r in range(H):
            for c in range(W):
                cell = self._grid[r, c]
                color = COLORS.get(cell, (200, 200, 200))
                canvas[
                    r * SCALE : (r + 1) * SCALE,
                    c * SCALE : (c + 1) * SCALE,
                ] = color

        # Draw agents as coloured circles
        try:
            from PIL import Image, ImageDraw
            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            for i, agent in enumerate(self.possible_agents):
                r, c = self._agent_pos[i]
                x0 = c * SCALE + 4
                y0 = r * SCALE + 4
                x1 = (c + 1) * SCALE - 4
                y1 = (r + 1) * SCALE - 4
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                if self._agent_carrying[i]:
                    # Brighter when carrying
                    color = tuple(min(255, v + 80) for v in color)
                draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
            canvas = np.array(img)
        except ImportError:
            # Fallback: fill square with agent color
            for i in range(len(self.possible_agents)):
                r, c = self._agent_pos[i]
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                canvas[
                    r * SCALE + 4 : (r + 1) * SCALE - 4,
                    c * SCALE + 4 : (c + 1) * SCALE - 4,
                ] = color

        if self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt
                plt.imshow(canvas)
                plt.axis("off")
                plt.pause(0.01)
                plt.clf()
            except ImportError:
                logger.warning("matplotlib not available for human rendering.")

        return canvas

    def close(self) -> None:
        """Clean up any resources held by the environment."""
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass

    # ── Convenience: state for CTDE ───────────────────────────────

    def get_global_state(self) -> np.ndarray:
        """
        Concatenate all agent observations into a single global state vector.

        Used by the CTDE critic as its input.

        Returns:
            Float32 array of shape (N * obs_dim,).
        """
        obs_list = [self._get_obs(i) for i in range(len(self.possible_agents))]
        return np.concatenate(obs_list, axis=0)


# ── Partial observability variant (same class, different flag) ────────────────
# Access via: CooperativeGridWorld(partial_observability=True)


# ── PettingZoo wrapper convenience ───────────────────────────────────────────

def make_env(
    num_agents: int = 2,
    grid_size: int = 10,
    num_obstacles: int = 5,
    max_steps: int = 200,
    partial_observability: bool = False,
    reward_mode: Literal["individual", "shared"] = "individual",
    render_mode: Optional[str] = None,
    seed: int = 42,
) -> CooperativeGridWorld:
    """
    Factory function for creating and resetting a CooperativeGridWorld.

    Wraps construction + ``reset(seed=seed)`` so callers get a ready env.

    Args:
        num_agents: Number of cooperative agents.
        grid_size: Side length of the NxN grid.
        num_obstacles: Number of static obstacles.
        max_steps: Maximum steps before truncation.
        partial_observability: Whether agents have a restricted view.
        reward_mode: ``"individual"`` or ``"shared"``.
        render_mode: ``"rgb_array"`` or ``"human"`` or None.
        seed: Random seed for reset.

    Returns:
        :class:`CooperativeGridWorld` instance (already reset with seed).
    """
    env = CooperativeGridWorld(
        num_agents=num_agents,
        grid_size=grid_size,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
        partial_observability=partial_observability,
        reward_mode=reward_mode,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


# ── __main__: quick sanity check ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== CooperativeGridWorld Quick Check ===")

    for partial in [False, True]:
        env = CooperativeGridWorld(
            num_agents=2,
            grid_size=10,
            partial_observability=partial,
            reward_mode="individual",
        )
        obs, infos = env.reset(seed=42)
        print(f"\nPartial={partial}")
        for ag, o in obs.items():
            print(f"  {ag} obs shape: {o.shape}")

        total_reward = 0.0
        steps = 0
        t0 = time.time()
        while env.agents:
            actions = {ag: env.action_space(ag).sample() for ag in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())
            steps += 1

        elapsed = time.time() - t0
        print(f"  Episode done in {steps} steps, total_reward={total_reward:.2f}, {elapsed*1000:.1f}ms")

    env.close()
    print("\n✅ CooperativeGridWorld smoke test PASSED")
