"""Top-level command line interface for AI Lab simulations and utilities.

Allows running simulations directly via module syntax:
    python -m src vacuum --steps 5
    python -m src randomized --seed 58
    python -m src model-based
    python -m src goal-based
    python -m src smoke-test
"""

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point dispatcher for AI Lab simulations."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="AI Lab: Intelligent Agent Simulations & Experiment Runners",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Vacuum agent runner
    subparsers.add_parser("vacuum", help="Run Question 1: Simple Reflex Vacuum Cleaner Agent")

    # Randomized reflex agent runner
    subparsers.add_parser("randomized", help="Run Question 2: Randomized Reflex Agent (1D Grid)")

    # Model-based reflex agent runner
    subparsers.add_parser("model-based", help="Run Question 3: Model-Based Reflex Agent (2x2 Grid)")

    # Goal-based navigation agent runner
    subparsers.add_parser("goal-based", help="Run Question 4: Goal-Based Navigation Agent (5x5 Grid)")

    # Starter smoke test runner
    subparsers.add_parser("smoke-test", help="Run starter experiment runner smoke test")

    if argv is None:
        argv = sys.argv[1:]

    # If first arg matches a subcommand, forward remaining args to that script
    if argv and argv[0] in {"vacuum", "randomized", "model-based", "goal-based", "smoke-test"}:
        cmd = argv[0]
        remaining = argv[1:]
        if cmd == "vacuum":
            from scripts.run_vacuum_agent import main as vacuum_main
            sys.argv = [sys.argv[0]] + remaining
            vacuum_main()
        elif cmd == "randomized":
            from scripts.run_randomized_agent import main as randomized_main
            sys.argv = [sys.argv[0]] + remaining
            randomized_main()
        elif cmd == "model-based":
            from scripts.run_model_based_agent import main as model_main
            sys.argv = [sys.argv[0]] + remaining
            model_main()
        elif cmd == "goal-based":
            from scripts.run_goal_based_agent import main as goal_main
            sys.argv = [sys.argv[0]] + remaining
            goal_main()
        elif cmd == "smoke-test":
            from scripts.smoke_test import main as smoke_main
            sys.argv = [sys.argv[0]] + remaining
            smoke_main()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
