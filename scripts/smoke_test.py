"""Starter entry point for AI Lab experimentation and smoke testing.

Demonstrates running a baseline trial using the modular experiment runner.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiment import ExperimentRunner


def main() -> None:
    """Execute starter verification and baseline experiment simulation."""
    print("Hello, World!")
    print("\n--- AI Lab: Initializing Starter Experiment Trial ---")

    runner = ExperimentRunner(name="starter_smoke_test", seed=42)
    history = runner.simulate_training(num_epochs=10, base_accuracy=0.60, target_accuracy=0.92)

    summary = runner.compute_summary()
    print(f"Trial completed successfully across {int(summary['total_epochs'])} epochs.")
    print(f"  • Starting Accuracy: {history[0].accuracy:.4f}")
    print(f"  • Peak Accuracy:     {summary['max_accuracy']:.4f}")
    print(f"  • Minimum Loss:      {summary['min_loss']:.4f}")
    print(f"  • Final Accuracy:    {summary['final_accuracy']:.4f}")
    print("\nEnvironment is configured and ready for experiments.")


if __name__ == "__main__":
    main()