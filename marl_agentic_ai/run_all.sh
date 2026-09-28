#!/usr/bin/env bash
# run_all.sh — Full experiment pipeline: test → train → plot
# Usage: bash run_all.sh
set -euo pipefail

echo "=============================================="
echo "  MARL Agentic AI — Full Experiment Pipeline"
echo "=============================================="

# 1. Validate environment
echo ""
echo "[1/4] Validating environment..."
python scripts/validate_env.py
echo "Environment validation PASSED."

# 2. Run test suite
echo ""
echo "[2/4] Running test suite..."
pytest tests/ -v --tb=short -x
echo "All tests PASSED."

# 3. Run all experiments
echo ""
echo "[3/4] Running experiment suite (5 experiments)..."
python scripts/run_experiments.py
echo "All experiments COMPLETE."

# 4. Plot results
echo ""
echo "[4/4] Generating plots..."
python scripts/plot_results.py --results-dir runs/
echo "Plots saved to runs/plots/."

echo ""
echo "=============================================="
echo "  Pipeline complete! Results in runs/"
echo "  View: tensorboard --logdir runs/"
echo "=============================================="
