"""Runner script for Agent Performance Measure Evaluator (Question 9).

Evaluates the performance score of an autonomous delivery drone based on a
sequence of simulated action events, external CSV logs, or interactive user inputs.

Usage Examples:
    # 1. Default Sample Run (Russell & Norvig Benchmark):
    python scripts/run_agent_performance_evaluator.py

    # 2. Custom inline event sequence:
    python scripts/run_agent_performance_evaluator.py --events move,move,bump,move,deliver

    # 3. Read action log from external CSV file (Optional Challenge):
    python scripts/run_agent_performance_evaluator.py --csv data/raw/drone_actions_sample.csv

    # 4. Detailed step-by-step flight audit table:
    python scripts/run_agent_performance_evaluator.py --detailed

    # 5. Run automated demonstration benchmark suite:
    python scripts/run_agent_performance_evaluator.py --demo

    # 6. Interactive step-by-step terminal prompt:
    python scripts/run_agent_performance_evaluator.py --interactive

    # 7. Display Viva Voce theory & reference study guide:
    python scripts/run_agent_performance_evaluator.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agent_performance_evaluator import (
    EVENT_BUMP,
    EVENT_DELIVER,
    EVENT_MOVE,
    DronePerformanceEvaluator,
    EvaluationResult,
    ScoringMetric,
    evaluate_drone_performance,
    get_viva_reference,
    read_events_from_csv,
)


def run_interactive_prompt(evaluator: DronePerformanceEvaluator) -> None:
    """Prompt the user interactively to evaluate simulated drone actions."""
    print("=" * 70)
    print("      INTERACTIVE DRONE PERFORMANCE MEASURE EVALUATOR")
    print("=" * 70)
    print("Enter drone events one by one or as a comma-separated list.")
    print("Valid events: 'move' (-1 pt), 'bump' (-5 pts), 'deliver' (+10 pts).")
    print("Type 'done' or press Enter on an empty line when finished.\n")

    events: List[str] = []
    step_num = 1
    while True:
        try:
            line = input(f"Event #{step_num} (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEvaluation terminated by user.")
            return

        if not line or line.lower() == "done":
            break

        # Check if line contains commas or spaces separating multiple events
        if "," in line:
            sub_events = [s.strip() for s in line.split(",") if s.strip()]
            events.extend(sub_events)
            step_num += len(sub_events)
        elif " " in line and not line.startswith("'") and not line.startswith('"'):
            sub_events = [s.strip() for s in line.split() if s.strip()]
            events.extend(sub_events)
            step_num += len(sub_events)
        else:
            events.append(line)
            step_num += 1

    if not events:
        print("No events entered. Evaluation aborted.")
        return

    print("\nProcessing action sequence...\n")
    result = evaluator.evaluate_log(events, record_trace=True)
    print(result.format_detailed_report())


def run_demonstration_suite(evaluator: DronePerformanceEvaluator) -> None:
    """Run a suite of canonical drone flight scenarios showcasing PEAS evaluation."""
    print("=" * 75)
    print("        AUTONOMOUS DRONE PERFORMANCE EVALUATION - BENCHMARK SUITE")
    print("=" * 75)

    scenarios = [
        (
            "Scenario 1: Standard Sample Benchmark",
            ["move", "move", "bump", "move", "deliver"],
            "Standard sequence featuring moves, one collision, and successful delivery.",
        ),
        (
            "Scenario 2: Flawless Express Delivery",
            ["move", "move", "move", "deliver"],
            "Ideal obstacle-free flight route maximizing net positive score.",
        ),
        (
            "Scenario 3: Severe Collision Flight (Negative Score Demonstration)",
            ["move", "bump", "bump", "move", "bump", "bump"],
            "Catastrophic flight with repeated collisions demonstrating negative score constraint.",
        ),
        (
            "Scenario 4: Multi-Delivery Mission with Weather Detours",
            ["move", "move", "deliver", "move", "bump", "move", "move", "deliver"],
            "Two distinct deliveries with an intermediate collision and routing maneuvers.",
        ),
    ]

    for title, events, explanation in scenarios:
        print(f"\n>>> {title}")
        print(f"    Context: {explanation}")
        print(f"    Input Actions : {events}")
        result = evaluator.evaluate_log(events, record_trace=True)
        print(f"    {result.format_summary()}")
        print(f"    Score Details : {result.delivery_count} deliveries (+{result.delivery_count * 10} pts), "
              f"{result.move_count} moves (-{result.move_count * 1} pts), "
              f"{result.bump_count} bumps (-{result.bump_count * 5} pts)")

    # CSV File Benchmark demonstration
    csv_sample_path = _REPO_ROOT / "data" / "raw" / "drone_actions_sample.csv"
    if csv_sample_path.is_file():
        print("\n>>> Scenario 5: External CSV Action Log (Optional Challenge)")
        print(f"    File: {csv_sample_path}")
        csv_result = evaluator.evaluate_csv(csv_sample_path)
        print(f"    {csv_result.format_summary()}")

    print("\n" + "=" * 75)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main CLI entry point for Question 9."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_agent_performance_evaluator.py",
        description="Question 9: Evaluate Autonomous Delivery Drone PEAS Performance Score",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-e",
        "--events",
        type=str,
        help="Comma-separated or space-separated action sequence (e.g., 'move,move,bump,move,deliver')",
    )
    group.add_argument(
        "--csv",
        type=str,
        help="Path to an external CSV file containing the action log (Optional Challenge)",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive demonstration benchmark suite of diverse scenarios",
    )
    group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt to input actions sequentially",
    )
    group.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory notes, PEAS analysis, and study guide",
    )

    parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="Print full tabular flight audit trace and running score breakdown",
    )
    parser.add_argument(
        "--reward-delivery",
        type=int,
        default=10,
        help="Points awarded per successful delivery (default: 10)",
    )
    parser.add_argument(
        "--cost-move",
        type=int,
        default=1,
        help="Points deducted per time step / move (default: 1)",
    )
    parser.add_argument(
        "--penalty-bump",
        type=int,
        default=5,
        help="Points deducted per obstacle collision (default: 5)",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print(get_viva_reference())
        return

    metric = ScoringMetric(
        reward_delivery=args.reward_delivery,
        cost_move=args.cost_move,
        penalty_bump=args.penalty_bump,
    )
    evaluator = DronePerformanceEvaluator(metric=metric)

    if args.demo:
        run_demonstration_suite(evaluator)
        return

    if args.interactive:
        run_interactive_prompt(evaluator)
        return

    events: List[str]
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = _REPO_ROOT / csv_path
        events = read_events_from_csv(csv_path)
    elif args.events:
        raw = args.events.strip()
        if "," in raw:
            events = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            events = [s.strip() for s in raw.split() if s.strip()]
    else:
        # Default sample from problem statement
        events = ["move", "move", "bump", "move", "deliver"]

    result = evaluator.evaluate_log(events, record_trace=True)

    if args.detailed:
        print(result.format_detailed_report())
    else:
        # Canonical format matching problem specification exactly:
        # Sample Output: Performance Score: 2 (10 - 3 moves - 5 bump)
        print(result.format_summary())


if __name__ == "__main__":
    main()
