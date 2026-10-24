"""Experiment execution and metrics tracking module.

This module provides reusable abstractions for simulating and evaluating
machine learning experiment trials, calculating performance metrics,
and tracking convergence behavior.
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExperimentRecord:
    """Represents the observation of a single experiment epoch or step.

    Attributes:
        epoch: The 1-indexed iteration counter.
        accuracy: Model accuracy score in the range [0.0, 1.0].
        loss: Objective loss metric, non-negative.
    """

    epoch: int
    accuracy: float
    loss: float

    def __post_init__(self) -> None:
        """Validate metric constraints upon initialization.

        Ensures data integrity across experiment runs before metrics are ingested
        into downstream analysis or visualization pipelines.
        """
        if self.epoch < 1:
            raise ValueError(f"Epoch must be >= 1, received {self.epoch}")
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError(
                f"Accuracy must be bounded between 0.0 and 1.0, received {self.accuracy}"
            )
        if self.loss < 0.0:
            raise ValueError(f"Loss cannot be negative, received {self.loss}")


class ExperimentRunner:
    """Encapsulates experiment execution, observation tracking, and statistical aggregation.

    Adheres to the Single Responsibility Principle by isolating the lifecycle of an
    experiment run and metrics summarization from presentation logic.
    """

    def __init__(self, name: str = "default_experiment", seed: Optional[int] = None) -> None:
        """Initialize an experiment runner with an optional deterministic seed.

        Args:
            name: Human-readable identifier for the experiment session.
            seed: Seed value for reproducibility across deterministic trials.
        """
        self.name: str = name
        self._history: List[ExperimentRecord] = []
        self._random: random.Random = random.Random(seed)

    @property
    def history(self) -> List[ExperimentRecord]:
        """Return a defensive copy of the recorded experiment observations."""
        return list(self._history)

    def record_step(self, epoch: int, accuracy: float, loss: float) -> ExperimentRecord:
        """Record an individual training or evaluation iteration.

        Args:
            epoch: Step index of the observation.
            accuracy: Accuracy value achieved in this step.
            loss: Loss value computed in this step.

        Returns:
            The validated ExperimentRecord added to history.
        """
        record = ExperimentRecord(epoch=epoch, accuracy=accuracy, loss=loss)
        self._history.append(record)
        return record

    def simulate_training(
        self,
        num_epochs: int = 50,
        base_accuracy: float = 0.65,
        target_accuracy: float = 0.95,
        decay_rate: float = 20.0,
    ) -> List[ExperimentRecord]:
        """Simulate a realistic training trajectory with asymptotic learning and noise.

        Why exponential decay is used:
            Real-world model optimization typically exhibits logarithmic or
            asymptotically decaying marginal gains per iteration. Modeling the step
            function with 1 - exp(-epoch / decay_rate) accurately mirrors stochastic
            gradient descent behavior without requiring heavy runtime dependencies.

        Args:
            num_epochs: Total number of iterations to simulate.
            base_accuracy: Starting accuracy prior to significant convergence.
            target_accuracy: Theoretical maximum accuracy asymptote.
            decay_rate: Learning progression timescale factor.

        Returns:
            List of generated ExperimentRecord observations.

        Raises:
            ValueError: If num_epochs < 1 or accuracy thresholds are invalid.
        """
        if num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if not (0.0 <= base_accuracy <= target_accuracy <= 1.0):
            raise ValueError("Invalid accuracy bounds: requires 0.0 <= base <= target <= 1.0")

        self.clear()
        spread = target_accuracy - base_accuracy

        for epoch in range(1, num_epochs + 1):
            progress = 1.0 - math.exp(-epoch / decay_rate)
            noise = self._random.gauss(0.0, 0.015)
            raw_acc = base_accuracy + (spread * progress) + noise
            bounded_acc = round(min(1.0, max(0.0, raw_acc)), 4)

            # Loss asymptotically decays toward ~0.05
            raw_loss = max(0.02, 1.5 * math.exp(-epoch / decay_rate) + self._random.uniform(-0.01, 0.01))
            bounded_loss = round(raw_loss, 4)

            self.record_step(epoch=epoch, accuracy=bounded_acc, loss=bounded_loss)

        return self.history

    def compute_summary(self) -> Dict[str, float]:
        """Compute aggregated statistical metrics over the collected history.

        Returns:
            Dictionary containing min/max/mean metrics.

        Raises:
            ValueError: If no observations have been recorded yet.
        """
        if not self._history:
            raise ValueError("Cannot compute summary on empty experiment history.")

        accuracies = [rec.accuracy for rec in self._history]
        losses = [rec.loss for rec in self._history]

        return {
            "total_epochs": float(len(self._history)),
            "max_accuracy": max(accuracies),
            "min_accuracy": min(accuracies),
            "mean_accuracy": round(sum(accuracies) / len(accuracies), 4),
            "final_accuracy": accuracies[-1],
            "min_loss": min(losses),
            "final_loss": losses[-1],
        }

    def clear(self) -> None:
        """Reset the internal history state for new experiment runs."""
        self._history.clear()
