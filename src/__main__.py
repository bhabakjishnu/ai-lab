"""Top-level command line interface for AI Lab simulations and utilities.

Allows running simulations directly via module syntax:
    python -m src vacuum --steps 5
    python -m src randomized --seed 58
    python -m src model-based
    python -m src goal-based
    python -m src utility-based
    python -m src multi-agent --steps 10
    python -m src stochastic --runs 20 --seed 28
    python -m src environment-classifier --input True,False,False,False,True
    python -m src agent-evaluator --events move,move,bump,move,deliver
    python -m src state-space --table
    python -m src bfs --start A
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

    # Utility-based agent runner
    subparsers.add_parser("utility-based", help="Run Question 5: Utility-Based Agent Simulation")

    # Multi-agent simulation runner
    subparsers.add_parser("multi-agent", help="Run Question 6: Multi-Agent Environment Simulation (4x4 Grid)")

    # Stochastic environment vacuum agent runner
    subparsers.add_parser("stochastic", help="Run Question 7: Stochastic Environment Vacuum Agent Simulation")

    # Environment type classifier runner
    subparsers.add_parser(
        "environment-classifier",
        aliases=["env-classifier"],
        help="Run Question 8: Environment Type Classifier (PEAS & Knowledge Representation)",
    )

    # Agent performance measure evaluator runner
    subparsers.add_parser(
        "agent-evaluator",
        aliases=["drone-evaluator", "perf-evaluator"],
        help="Run Question 9: Agent Performance Measure Evaluator (PEAS & Rationality)",
    )

    # State-space visualizer runner
    subparsers.add_parser(
        "state-space",
        aliases=["state-visualizer", "state-space-visualizer"],
        help="Run Question 10: State-Space Visualizer for Reflex Agent (Graph & Cycles)",
    )

    # Breadth-first search runner
    subparsers.add_parser(
        "bfs",
        aliases=["breadth-first-search", "bfs-search"],
        help="Run Question 11: Breadth-First Search (BFS) on User-Defined Graph",
    )

    # Starter smoke test runner
    subparsers.add_parser("smoke-test", help="Run starter experiment runner smoke test")

    if argv is None:
        argv = sys.argv[1:]

    # If first arg matches a subcommand, forward remaining args to that script
    allowed_subcommands = {
        "vacuum",
        "randomized",
        "model-based",
        "goal-based",
        "utility-based",
        "multi-agent",
        "stochastic",
        "environment-classifier",
        "env-classifier",
        "agent-evaluator",
        "drone-evaluator",
        "perf-evaluator",
        "state-space",
        "state-visualizer",
        "state-space-visualizer",
        "bfs",
        "breadth-first-search",
        "bfs-search",
        "smoke-test",
    }
    if argv and argv[0] in allowed_subcommands:
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
        elif cmd == "utility-based":
            from scripts.run_utility_based_agent import main as utility_main
            sys.argv = [sys.argv[0]] + remaining
            utility_main()
        elif cmd == "multi-agent":
            from scripts.run_multi_agent_simulation import main as multi_agent_main
            sys.argv = [sys.argv[0]] + remaining
            multi_agent_main()
        elif cmd == "stochastic":
            from scripts.run_stochastic_vacuum_agent import main as stochastic_main
            sys.argv = [sys.argv[0]] + remaining
            stochastic_main()
        elif cmd in ("environment-classifier", "env-classifier"):
            from scripts.run_environment_classifier import main as env_main
            sys.argv = [sys.argv[0]] + remaining
            env_main()
        elif cmd in ("agent-evaluator", "drone-evaluator", "perf-evaluator"):
            from scripts.run_agent_performance_evaluator import main as agent_eval_main
            sys.argv = [sys.argv[0]] + remaining
            agent_eval_main()
        elif cmd in ("state-space", "state-visualizer", "state-space-visualizer"):
            from scripts.run_state_space_visualizer import main as state_space_main
            sys.argv = [sys.argv[0]] + remaining
            state_space_main()
        elif cmd in ("bfs", "breadth-first-search", "bfs-search"):
            from scripts.run_breadth_first_search import main as bfs_main
            sys.argv = [sys.argv[0]] + remaining
            bfs_main()
        elif cmd == "smoke-test":
            from scripts.smoke_test import main as smoke_main
            sys.argv = [sys.argv[0]] + remaining
            smoke_main()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
