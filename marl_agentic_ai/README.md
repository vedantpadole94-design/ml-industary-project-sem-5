# MARL Agentic AI — Cooperative Grid World Research Framework

## Research Question

> **Can structured communication and centralized training improve cooperative task completion 
> in a partially-observable grid world, and does information asymmetry create a performance 
> ceiling that only inter-agent communication can overcome?**

---

## Project Overview

This framework implements a 5-experiment ablation study using a 10×10 **Cooperative Grid World** 
built on [PettingZoo](https://pettingzoo.farama.org/) with [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) 
PPO training. We systematically add:

1. **E1** — Individual reward, no communication, no centralized critic (pure baseline)
2. **E2** — Shared reward, no communication, no centralized critic
3. **E3** — Shared reward, no communication, **CTDE** (centralized critic)
4. **E4** — Shared reward, **communication**, CTDE
5. **E5** — Shared reward, communication, CTDE, **partial observability**

---

## Experiment Matrix

| ID | Algorithm | Reward   | Comm | CTDE | Partial Obs | Expected Outcome            |
|----|-----------|----------|------|------|-------------|-----------------------------|
| E1 | PPO       | Individual | ✗  | ✗    | ✗           | Baseline reference          |
| E2 | PPO       | Shared   | ✗    | ✗    | ✗           | +reward alignment           |
| E3 | PPO       | Shared   | ✗    | ✓    | ✗           | +coordination via critic     |
| E4 | PPO       | Shared   | ✓    | ✓    | ✗           | +explicit communication     |
| E5 | PPO       | Shared   | ✓    | ✓    | ✓           | Communication is *necessary* |

---

## Directory Structure

```
marl_agentic_ai/
├── marl/
│   ├── envs/cooperative_grid.py      # PettingZoo Parallel environment
│   ├── trainers/                     # baseline, shared_reward, ctde, comm
│   ├── models/                       # MLP, comm_module, ctde_model
│   ├── utils/                        # config, seeding, logging
│   └── api.py                        # High-level MARLTrainer API
├── configs/                          # YAML experiment configs
├── tests/                            # pytest suite
├── scripts/                          # CLI training/eval/viz scripts
├── notebooks/                        # Jupyter analysis notebooks
└── runs/                             # Experiment outputs (auto-generated)
```

---

## Reproduction Commands

### 1. Setup

```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev,render]"
```

### 2. Validate Environment

```bash
python scripts/validate_env.py
```

Expected output:
```
Environment reset  | PASS
Observation shape  | PASS
Action space       | PASS
Agent transitions  | PASS
Rewards            | PASS
Termination        | PASS
Deterministic seed | PASS
PettingZoo API     | PASS
```

### 3. Run Tests

```bash
pytest tests/ -v --cov=marl
```

### 4. Run Individual Experiments

```bash
# E1: Baseline
python scripts/train.py --config configs/baseline.yaml

# E2: Shared Reward
python scripts/train.py --config configs/shared_reward.yaml

# E3: CTDE
python scripts/train.py --config configs/ctde.yaml

# E4: Communication
python scripts/train.py --config configs/comm.yaml

# E5: Partial Observability
python scripts/train.py --config configs/partial_obs.yaml
```

### 5. Run All Experiments (Full Ablation)

```bash
bash run_all.sh
```

### 6. Evaluate Checkpoint

```bash
python scripts/evaluate.py --checkpoint runs/exp_1/checkpoint_final --episodes 200
```

### 7. Visualize

```bash
python scripts/visualize.py --checkpoint runs/exp_4/checkpoint_final
```

### 8. Compare Results

```bash
python scripts/compare.py --results-dir runs/
python scripts/plot_results.py --results-dir runs/
```

### 9. Using the High-Level API

```python
from marl.api import MARLTrainer

trainer = MARLTrainer(
    env="cooperative_grid",
    agents=4,
    algorithm="ppo",
    shared_reward=True,
    communication=True
)
trainer.train(timesteps=1_000_000)
results = trainer.evaluate(episodes=100)
print(results)
```

### 10. Docker

```bash
docker build -t marl-agentic-ai .
docker run --gpus all marl-agentic-ai bash run_all.sh
```

---

## Results Table (Placeholder)

| Exp | Mean Reward | Std Reward | Success Rate | Mean Ep Len | Collision Rate | Comm Freq | Wall Time |
|-----|-------------|------------|--------------|-------------|----------------|-----------|-----------|
| E1  | —           | —          | —            | —           | —              | —         | —         |
| E2  | —           | —          | —            | —           | —              | —         | —         |
| E3  | —           | —          | —            | —           | —              | —         | —         |
| E4  | —           | —          | —            | —           | —              | —         | —         |
| E5  | —           | —          | —            | —           | —              | —         | —         |

*Fill after running experiments.*

---

## Environment Details

### Cooperative Grid World (10×10)

- **Agents**: N (default 2), each with unique resource assignment
- **Entities**: N resources + M obstacles + 1 goal
- **Success**: All agents collect their resource → all reach goal within `max_steps`
- **Actions**: `Discrete(5)` — NOOP, UP, DOWN, LEFT, RIGHT
- **Observation** (per agent): `[pos(2), local_view(27), carrying(1), others_rel(2*(N-1))]`
- **Reward**: Shaped individual, or mean-shared across agents

### Communication Module

- Encoder: MLP(obs_dim → 128)
- Message: MLP(128 → 32) + Gumbel-Softmax (discrete, differentiable)
- Aggregation: concat all peer messages → MLP(64)
- Output: concat with obs → policy + value heads

---

## Key Design Decisions

1. **PettingZoo Parallel API** — all agents act simultaneously, no sequential turns
2. **Parameter Sharing** — default; independent policies optionally via `independent: bool`
3. **CTDE via GlobalStateWrapper** — concatenates all obs into `state` key for critic
4. **Gumbel-Softmax** for discrete communication — preserves gradient flow
5. **Shaped rewards** — distance-to-resource + resource pickup bonus + goal bonus − collision penalty

---

## Citation

```bibtex
@misc{marl2024cooperative,
  title  = {Cooperative Grid World: A Multi-Agent RL Research Framework},
  author = {MARL Research Team},
  year   = {2024},
  url    = {https://github.com/marl-research/marl-agentic-ai},
  note   = {Research Framework: PettingZoo + Ray RLlib + PyTorch}
}
```

---

## License

MIT License. See `LICENSE` for details.
