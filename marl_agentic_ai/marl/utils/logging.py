"""
marl/utils/logging.py
TensorBoard-backed logger with structured tags for MARL training.

All required tags:
    train/reward            — mean episode reward
    train/episode_length    — mean episode length
    train/collision_rate    — fraction of steps with agent-agent collision
    train/reward_variance   — variance of per-agent rewards
    eval/success_rate       — fraction of eval episodes fully solved
    eval/collision_rate     — eval-time collision fraction
    communication/messages  — mean per-episode message entropy + count

Usage:
    from marl.utils.logging import MARLLogger
    logger = MARLLogger(log_dir="runs/baseline", exp_name="baseline")
    logger.log_train(step=1000, reward=1.5, episode_length=80, collision_rate=0.1)
    logger.log_eval(step=50000, success_rate=0.6, collision_rate=0.05)
    logger.close()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

_log = logging.getLogger(__name__)


class MARLLogger:
    """
    Structured TensorBoard logger for MARL experiments.

    Wraps ``torch.utils.tensorboard.SummaryWriter`` with typed methods
    for each metric category defined in the project spec.

    Args:
        log_dir: Root directory for TensorBoard event files.
        exp_name: Experiment name appended to ``log_dir``.
        flush_secs: How often (in seconds) the writer auto-flushes.

    Example:
        >>> logger = MARLLogger("runs/", "baseline")
        >>> logger.log_train(step=0, reward=0.0, episode_length=200)
        >>> logger.close()
    """

    # Required tags from project spec
    REQUIRED_TAGS = {
        "train/reward",
        "train/episode_length",
        "train/collision_rate",
        "train/reward_variance",
        "eval/success_rate",
        "eval/collision_rate",
        "communication/messages",
    }

    def __init__(
        self,
        log_dir: str = "runs",
        exp_name: str = "experiment",
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir) / exp_name / "tb"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir), flush_secs=flush_secs
        )
        self._step_offset: Dict[str, int] = {}
        self._start_time = time.time()
        _log.info(f"TensorBoard logger initialized at {self.log_dir}")

    # ── Core Logging Methods ───────────────────────────────────────

    def scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log an arbitrary scalar value.

        Args:
            tag: TensorBoard tag (e.g. "train/reward").
            value: Scalar float value.
            step: Global training step.
        """
        self._writer.add_scalar(tag, value, global_step=step)

    def scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """
        Log multiple scalars under the same main tag.

        Args:
            tag: Main tag group.
            values: Dict of sub-tag → value.
            step: Global training step.
        """
        self._writer.add_scalars(tag, values, global_step=step)

    # ── Typed Logging Methods ──────────────────────────────────────

    def log_train(
        self,
        step: int,
        reward: Optional[float] = None,
        episode_length: Optional[float] = None,
        collision_rate: Optional[float] = None,
        reward_variance: Optional[float] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log training-phase metrics.

        Tags written:
        - ``train/reward``
        - ``train/episode_length``
        - ``train/collision_rate``
        - ``train/reward_variance``

        Args:
            step: Global training step / timestep.
            reward: Mean episode reward across agents and episodes.
            episode_length: Mean episode length in steps.
            collision_rate: Fraction of steps involving agent-agent collision.
            reward_variance: Variance of per-agent rewards in the batch.
            extra: Optional additional tags to log under ``train/``.
        """
        if reward is not None:
            self._writer.add_scalar("train/reward", reward, step)
        if episode_length is not None:
            self._writer.add_scalar("train/episode_length", episode_length, step)
        if collision_rate is not None:
            self._writer.add_scalar("train/collision_rate", collision_rate, step)
        if reward_variance is not None:
            self._writer.add_scalar("train/reward_variance", reward_variance, step)
        if extra:
            for k, v in extra.items():
                self._writer.add_scalar(f"train/{k}", v, step)

    def log_eval(
        self,
        step: int,
        success_rate: Optional[float] = None,
        collision_rate: Optional[float] = None,
        mean_reward: Optional[float] = None,
        std_reward: Optional[float] = None,
        mean_ep_len: Optional[float] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log evaluation-phase metrics.

        Tags written:
        - ``eval/success_rate``
        - ``eval/collision_rate``
        - ``eval/mean_reward``
        - ``eval/std_reward``
        - ``eval/mean_ep_len``

        Args:
            step: Global training step at time of evaluation.
            success_rate: Fraction of eval episodes that succeeded.
            collision_rate: Fraction of eval steps with collision.
            mean_reward: Mean reward per episode across eval episodes.
            std_reward: Std of reward per episode across eval episodes.
            mean_ep_len: Mean episode length across eval episodes.
            extra: Optional additional eval tags.
        """
        if success_rate is not None:
            self._writer.add_scalar("eval/success_rate", success_rate, step)
        if collision_rate is not None:
            self._writer.add_scalar("eval/collision_rate", collision_rate, step)
        if mean_reward is not None:
            self._writer.add_scalar("eval/mean_reward", mean_reward, step)
        if std_reward is not None:
            self._writer.add_scalar("eval/std_reward", std_reward, step)
        if mean_ep_len is not None:
            self._writer.add_scalar("eval/mean_ep_len", mean_ep_len, step)
        if extra:
            for k, v in extra.items():
                self._writer.add_scalar(f"eval/{k}", v, step)

    def log_communication(
        self,
        step: int,
        message_entropy: Optional[float] = None,
        message_count: Optional[float] = None,
        comm_freq: Optional[float] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log communication-channel metrics.

        Tags written:
        - ``communication/messages`` (alias for message_entropy)
        - ``communication/entropy``
        - ``communication/count``
        - ``communication/freq``

        Args:
            step: Global training step.
            message_entropy: Mean per-episode message entropy (bits).
            message_count: Mean number of messages sent per episode.
            comm_freq: Fraction of steps where a non-zero message is sent.
            extra: Optional additional communication tags.
        """
        # Primary required tag
        if message_entropy is not None:
            self._writer.add_scalar("communication/messages", message_entropy, step)
            self._writer.add_scalar("communication/entropy", message_entropy, step)
        if message_count is not None:
            self._writer.add_scalar("communication/count", message_count, step)
        if comm_freq is not None:
            self._writer.add_scalar("communication/freq", comm_freq, step)
        if extra:
            for k, v in extra.items():
                self._writer.add_scalar(f"communication/{k}", v, step)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log hyperparameters alongside final metrics for the HParams dashboard.

        Args:
            hparams: Dict of hyperparameter name → value.
            metrics: Dict of final metric name → value.
        """
        self._writer.add_hparams(hparams, metrics)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """
        Log a histogram of values (e.g., policy weights).

        Args:
            tag: TensorBoard tag.
            values: Array-like of values.
            step: Global step.
        """
        self._writer.add_histogram(tag, values, global_step=step)

    def wall_time(self) -> float:
        """Return elapsed wall time in seconds since logger creation."""
        return time.time() - self._start_time

    def flush(self) -> None:
        """Manually flush pending writes to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Close the underlying SummaryWriter."""
        self._writer.close()
        _log.info(f"Logger closed. Wall time: {self.wall_time():.1f}s")

    def __enter__(self) -> "MARLLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with MARLLogger(log_dir=tmpdir, exp_name="smoke_test") as logger:
            for step in range(0, 1001, 100):
                logger.log_train(
                    step=step,
                    reward=float(step) / 1000,
                    episode_length=200 - step // 10,
                    collision_rate=0.2 - step / 10000,
                    reward_variance=0.5,
                )
                logger.log_eval(
                    step=step,
                    success_rate=float(step) / 2000,
                    collision_rate=0.15,
                )
                logger.log_communication(
                    step=step,
                    message_entropy=1.5,
                    message_count=100.0,
                )

    print("✅ MARLLogger smoke test PASSED")
