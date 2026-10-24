"""Unit tests for the experiment module.

Validates data integrity constraints, simulation progression,
deterministic seed behavior, and statistical summary aggregations.
"""

import unittest
from src.experiment import ExperimentRecord, ExperimentRunner


class TestExperimentRecord(unittest.TestCase):
    """Test validation and boundary checks on ExperimentRecord."""

    def test_valid_record_creation(self) -> None:
        """Verify normal creation with valid values."""
        record = ExperimentRecord(epoch=1, accuracy=0.85, loss=0.32)
        self.assertEqual(record.epoch, 1)
        self.assertEqual(record.accuracy, 0.85)
        self.assertEqual(record.loss, 0.32)

    def test_invalid_epoch_raises_value_error(self) -> None:
        """Verify that non-positive epoch indices are rejected."""
        with self.assertRaises(ValueError):
            ExperimentRecord(epoch=0, accuracy=0.5, loss=0.1)

        with self.assertRaises(ValueError):
            ExperimentRecord(epoch=-5, accuracy=0.5, loss=0.1)

    def test_out_of_bounds_accuracy_raises_value_error(self) -> None:
        """Verify that accuracy outside [0.0, 1.0] triggers ValueError."""
        with self.assertRaises(ValueError):
            ExperimentRecord(epoch=1, accuracy=1.05, loss=0.1)

        with self.assertRaises(ValueError):
            ExperimentRecord(epoch=1, accuracy=-0.01, loss=0.1)

    def test_negative_loss_raises_value_error(self) -> None:
        """Verify that negative loss is flagged as an invalid state."""
        with self.assertRaises(ValueError):
            ExperimentRecord(epoch=1, accuracy=0.8, loss=-0.001)


class TestExperimentRunner(unittest.TestCase):
    """Test execution lifecycle, summaries, and simulation edge cases."""

    def setUp(self) -> None:
        """Set up a fresh runner instance before each test."""
        self.runner = ExperimentRunner(name="test_suite", seed=123)

    def test_initial_state_is_empty(self) -> None:
        """Ensure a newly instantiated runner has no history."""
        self.assertEqual(len(self.runner.history), 0)

    def test_record_step_appends_defensively(self) -> None:
        """Verify step recording and isolation of returned history list."""
        rec = self.runner.record_step(epoch=1, accuracy=0.70, loss=0.50)
        self.assertEqual(rec.epoch, 1)
        self.assertEqual(len(self.runner.history), 1)

        # Defensive copy check: modifying returned history list should not mutate runner
        hist = self.runner.history
        hist.clear()
        self.assertEqual(len(self.runner.history), 1)

    def test_compute_summary_on_empty_raises(self) -> None:
        """Ensure computing summary on empty runner raises ValueError."""
        with self.assertRaises(ValueError):
            self.runner.compute_summary()

    def test_simulation_generates_bounded_metrics(self) -> None:
        """Verify that simulated training metrics stay strictly within valid bounds."""
        records = self.runner.simulate_training(num_epochs=30, base_accuracy=0.6, target_accuracy=0.9)
        self.assertEqual(len(records), 30)

        for rec in records:
            self.assertTrue(0.0 <= rec.accuracy <= 1.0)
            self.assertTrue(rec.loss >= 0.0)

    def test_simulation_deterministic_seed(self) -> None:
        """Verify that runners configured with identical seeds generate identical output."""
        runner_a = ExperimentRunner(name="a", seed=999)
        runner_b = ExperimentRunner(name="b", seed=999)

        recs_a = runner_a.simulate_training(num_epochs=10)
        recs_b = runner_b.simulate_training(num_epochs=10)

        self.assertEqual(
            [(r.epoch, r.accuracy, r.loss) for r in recs_a],
            [(r.epoch, r.accuracy, r.loss) for r in recs_b],
        )

    def test_simulation_invalid_arguments_raise(self) -> None:
        """Test parameter validation on simulate_training."""
        with self.assertRaises(ValueError):
            self.runner.simulate_training(num_epochs=0)

        with self.assertRaises(ValueError):
            # Base accuracy greater than target
            self.runner.simulate_training(base_accuracy=0.9, target_accuracy=0.7)

    def test_compute_summary_calculations(self) -> None:
        """Verify statistical accuracy calculations in summary dictionary."""
        self.runner.record_step(epoch=1, accuracy=0.60, loss=0.80)
        self.runner.record_step(epoch=2, accuracy=0.80, loss=0.40)
        self.runner.record_step(epoch=3, accuracy=1.00, loss=0.10)

        summary = self.runner.compute_summary()
        self.assertEqual(summary["total_epochs"], 3.0)
        self.assertEqual(summary["max_accuracy"], 1.00)
        self.assertEqual(summary["min_accuracy"], 0.60)
        self.assertEqual(summary["mean_accuracy"], 0.80)
        self.assertEqual(summary["final_accuracy"], 1.00)
        self.assertEqual(summary["min_loss"], 0.10)
        self.assertEqual(summary["final_loss"], 0.10)

    def test_clear_resets_history(self) -> None:
        """Ensure clear method removes all recorded observations."""
        self.runner.record_step(epoch=1, accuracy=0.75, loss=0.25)
        self.assertEqual(len(self.runner.history), 1)
        self.runner.clear()
        self.assertEqual(len(self.runner.history), 0)


if __name__ == "__main__":
    unittest.main()
