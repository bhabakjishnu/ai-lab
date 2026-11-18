"""Command-Line Interface and Runner for 8-Puzzle Heuristic Evaluator (Question 25).

Evaluates 8-puzzle (3x3) board states using:
- h1: Misplaced Tiles heuristic (Hamming distance, blank 0 ignored)
- h2: Manhattan Distance heuristic (sum of orthogonal tile offsets, blank 0 ignored)

Usage Examples:
    # 1. Default Canonical Sample Run (Russell & Norvig Fig. 3.28):
    python scripts/run_eight_puzzle_heuristics.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_eight_puzzle_heuristics.py --format sample

    # 3. Custom Current & Goal 3x3 Matrices:
    python scripts/run_eight_puzzle_heuristics.py --current "[[7,2,4],[5,0,6],[8,3,1]]" --goal "[[0,1,2],[3,4,5],[6,7,8]]"

    # 4. Alternative Classic Goal (blank at bottom-right):
    python scripts/run_eight_puzzle_heuristics.py --current "7 2 4; 5 0 6; 8 3 1" --goal classic

    # 5. Optional Challenge: Generate Random Solvable Board(s) & Evaluate Heuristics:
    python scripts/run_eight_puzzle_heuristics.py --random --seed 42
    python scripts/run_eight_puzzle_heuristics.py --challenge --count 3 --seed 42

    # 6. Empirical Dominance Study (h2 >= h1 across N random solvable boards):
    python scripts/run_eight_puzzle_heuristics.py --dominance-study --samples 100 --seed 42

    # 7. Blank-Space Admissibility Counterexample Demo:
    python scripts/run_eight_puzzle_heuristics.py --blank-demo

    # 8. Interactive Mode:
    python scripts/run_eight_puzzle_heuristics.py --interactive

    # 9. Full Comprehensive Lab Demo:
    python scripts/run_eight_puzzle_heuristics.py --demo

    # 10. University Viva Voce Reference Guide:
    python scripts/run_eight_puzzle_heuristics.py --viva
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eight_puzzle_heuristics import (
    CLASSIC_GOAL_STATE,
    EASY_STATE,
    ONE_MOVE_STATE,
    SAMPLE_CURRENT_STATE,
    SAMPLE_GOAL_STATE,
    UNSOLVABLE_STATE,
    BoardTuple,
    EightPuzzleHeuristicEvaluator,
    HeuristicEvaluationResult,
    analyze_dominance_over_samples,
    demonstrate_blank_space_inadmissibility,
    evaluate_heuristics,
    format_evaluation_report,
    format_tile_breakdown_table,
    generate_random_solvable_board,
    get_eight_puzzle_viva_reference,
    parse_board,
    render_side_by_side_boards,
)


def display_viva_reference() -> None:
    """Print comprehensive University Viva Voce study guide to stdout."""
    viva = get_eight_puzzle_viva_reference()
    print("=" * 78)
    print("QUESTION NO. 25: 8-PUZZLE HEURISTIC EVALUATOR - VIVA VOCE STUDY GUIDE")
    print("=" * 78)
    for topic_key, entry in viva.items():
        title = topic_key.replace("_", " ").title()
        print(f"\n--- {title} ---")
        print(f"Q: {entry['question']}")
        print(f"A:\n{entry['answer']}")
        for extra_k, extra_v in entry.items():
            if extra_k not in ("question", "answer"):
                print(f"[{extra_k.replace('_', ' ').upper()}]: {extra_v}")
    print("\n" + "=" * 78)


def display_blank_space_demo() -> None:
    """Display the 1-move counterexample proving why counting blank (0) is inadmissible."""
    demo = demonstrate_blank_space_inadmissibility()
    print("=" * 78)
    print("WHY THE BLANK SPACE (0) MUST NOT BE COUNTED: ADMISSIBILITY PROOF")
    print("=" * 78)
    print(render_side_by_side_boards(demo["state"], demo["goal"]))
    print()
    print(f"True Optimal Cost h*(n) : {demo['true_optimal_cost']} move (slide tile 1 left into blank)")
    print("-" * 78)
    print(
        f"1. Valid (Blank 0 EXCLUDED): "
        f"h1 = {demo['h1_valid']} (Admissible: {demo['h1_valid_admissible']}), "
        f"h2 = {demo['h2_valid']} (Admissible: {demo['h2_valid_admissible']})"
    )
    print(
        f"2. Flawed (Blank 0 INCLUDED): "
        f"h1' = {demo['h1_with_blank']} (Admissible: {demo['h1_with_blank_admissible']}), "
        f"h2' = {demo['h2_with_blank']} (Admissible: {demo['h2_with_blank_admissible']})"
    )
    print("-" * 78)
    print(
        "Conclusion: Counting the blank space double-counts the single tile slide "
        "(h1' = h2' = 2 > h* = 1), breaking admissibility and A* optimality."
    )
    print("=" * 78)


def display_dominance_study(
    num_samples: int = 50,
    goal: BoardTuple = SAMPLE_GOAL_STATE,
    seed: Optional[int] = 42,
) -> None:
    """Run and display empirical dominance verification across random solvable boards."""
    summary = analyze_dominance_over_samples(num_samples=num_samples, goal=goal, seed=seed)
    print("=" * 78)
    print(f"EMPIRICAL DOMINANCE ANALYSIS: h2 (Manhattan) vs h1 (Misplaced) [{num_samples} Boards]")
    print("=" * 78)
    print(f"All Generated Boards Solvable? : {summary['all_solvable']}")
    print(f"h2(n) >= h1(n) on 100% States? : {summary['all_h2_dominates_h1']}")
    print(
        f"Strict Dominance (h2 > h1)     : {summary['strict_dominance_count']}/{num_samples} "
        f"({summary['strict_dominance_pct']:.1f}%)"
    )
    print(f"Mean h1 (Misplaced Tiles)      : {summary['mean_h1']:.2f} (range: {summary['min_h1']}..{summary['max_h1']})")
    print(f"Mean h2 (Manhattan Distance)   : {summary['mean_h2']:.2f} (range: {summary['min_h2']}..{summary['max_h2']})")
    print(f"Mean Dominance Gap (h2 - h1)   : +{summary['mean_dominance_gap']:.2f}")
    print("=" * 78)


def display_result(result: HeuristicEvaluationResult, output_format: str) -> None:
    """Print a ``HeuristicEvaluationResult`` in the requested format."""
    if output_format == "sample":
        print(result.format_sample_output())
    elif output_format == "table":
        print(result.format_sample_output())
        print(format_tile_breakdown_table(result))
    elif output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_evaluation_report(result))


def run_interactive_mode() -> None:
    """Prompt the user interactively for 3x3 Current and Goal boards."""
    print("=" * 78)
    print("INTERACTIVE 8-PUZZLE HEURISTIC EVALUATOR")
    print("=" * 78)
    print("Enter a 3x3 board as:")
    print("  - Python matrix : [[7,2,4],[5,0,6],[8,3,1]]")
    print("  - Semicolon rows: 7 2 4; 5 0 6; 8 3 1")
    print("  - Flat 9 numbers: 7 2 4 5 0 6 8 3 1")
    print("Press Enter without input to use default sample states.\n")

    try:
        curr_raw = input("Current State [default: [[7,2,4],[5,0,6],[8,3,1]]]: ").strip()
        goal_raw = input("Goal State    [default: [[0,1,2],[3,4,5],[6,7,8]]]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInteractive session ended.")
        return

    current = SAMPLE_CURRENT_STATE if not curr_raw else parse_board(curr_raw)
    goal = SAMPLE_GOAL_STATE if not goal_raw else parse_board(goal_raw)
    result = evaluate_heuristics(current, goal)
    print()
    print(format_evaluation_report(result))


def run_full_demo(seed: int = 42) -> None:
    """Execute a comprehensive laboratory demonstration of Question 25."""
    print("=" * 78)
    print("QUESTION NO. 25: 8-PUZZLE HEURISTIC EVALUATOR - FULL LABORATORY DEMO")
    print("=" * 78)

    # 1. Canonical Sample
    print("\n[1] CANONICAL SYLLABUS SAMPLE (Russell & Norvig Fig. 3.28)")
    sample_res = evaluate_heuristics(SAMPLE_CURRENT_STATE, SAMPLE_GOAL_STATE)
    print(format_evaluation_report(sample_res))

    # 2. Pedagogical Benchmark States
    print("\n[2] BENCHMARK STATE COMPARISON (Goal: [[0,1,2],[3,4,5],[6,7,8]])")
    evaluator = EightPuzzleHeuristicEvaluator(SAMPLE_GOAL_STATE)
    benchmarks = [
        ("Goal State (Solved)", SAMPLE_GOAL_STATE),
        ("1-Move State", ONE_MOVE_STATE),
        ("Easy State", EASY_STATE),
        ("Canonical Sample", SAMPLE_CURRENT_STATE),
        ("Unsolvable State (Odd Parity)", UNSOLVABLE_STATE),
    ]
    print(f"{'State Label':<32} | {'h1':>4} | {'h2':>4} | {'Gap':>4} | {'Inv':>4} | {'Solvable':<10}")
    print("-" * 78)
    for label, board in benchmarks:
        res = evaluator.evaluate(board)
        print(
            f"{label:<32} | {res.h1_misplaced:>4} | {res.h2_manhattan:>4} | "
            f"+{res.dominance_gap:<3} | {res.current_inversions:>4} | "
            f"{'YES' if res.is_solvable else 'NO':<10}"
        )

    # 3. Optional Challenge: Random Solvable Board
    print("\n[3] OPTIONAL CHALLENGE: RANDOM SOLVABLE BOARD GENERATION")
    rand_board = generate_random_solvable_board(goal=SAMPLE_GOAL_STATE, seed=seed)
    rand_res = evaluate_heuristics(rand_board, SAMPLE_GOAL_STATE)
    print(format_evaluation_report(rand_res))

    # 4. Blank-Space Admissibility Counterexample
    print("\n[4] BLANK-SPACE ADMISSIBILITY PROOF")
    display_blank_space_demo()

    # 5. Dominance Study
    print("\n[5] EMPIRICAL DOMINANCE VERIFICATION")
    display_dominance_study(num_samples=50, goal=SAMPLE_GOAL_STATE, seed=seed)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_eight_puzzle_heuristics.py",
        description="Question 25: 8-Puzzle Heuristic Evaluator (h1 Misplaced Tiles & h2 Manhattan Distance)",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Current 3x3 board (e.g., '[[7,2,4],[5,0,6],[8,3,1]]', '7 2 4; 5 0 6; 8 3 1', or preset name).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal 3x3 board (default: '[[0,1,2],[3,4,5],[6,7,8]]', or preset 'standard'/'classic').",
    )
    parser.add_argument(
        "--format",
        choices=["full", "sample", "table", "json"],
        default="full",
        help="Output formatting mode (default: full).",
    )
    parser.add_argument(
        "--random",
        "--challenge",
        dest="random_board",
        action="store_true",
        help="Optional Challenge: generate random solvable 8-puzzle board(s) and calculate h1 and h2.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of random solvable boards to generate when using --random/--challenge (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed for reproducible random board generation or dominance study.",
    )
    parser.add_argument(
        "--dominance-study",
        action="store_true",
        help="Empirically verify h2 >= h1 dominance across N random solvable boards.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of random solvable boards for --dominance-study (default: 50).",
    )
    parser.add_argument(
        "--blank-demo",
        action="store_true",
        help="Demonstrate why counting the blank space (0) violates admissibility.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prompt mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive laboratory demonstration covering all features and proofs.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display University Viva Voce Q&A study guide.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point for Question 25 runner script."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.viva:
        display_viva_reference()
        return

    if args.demo:
        run_full_demo(seed=42 if args.seed is None else args.seed)
        return

    if args.interactive:
        run_interactive_mode()
        return

    if args.blank_demo:
        display_blank_space_demo()
        return

    goal_board = SAMPLE_GOAL_STATE if args.goal is None else parse_board(args.goal)

    if args.dominance_study:
        display_dominance_study(
            num_samples=args.samples,
            goal=goal_board,
            seed=42 if args.seed is None else args.seed,
        )
        return

    if args.random_board:
        if args.count < 1:
            parser.error("--count must be at least 1.")
        print("=" * 78)
        print("OPTIONAL CHALLENGE: RANDOM SOLVABLE 8-PUZZLE BOARD HEURISTIC EVALUATION")
        print("=" * 78)
        for idx in range(args.count):
            board_seed = None if args.seed is None else args.seed + idx
            board = generate_random_solvable_board(goal=goal_board, seed=board_seed)
            result = evaluate_heuristics(board, goal_board)
            if args.count > 1 and args.format != "json":
                print(f"\n--- Random Solvable Board #{idx + 1} ---")
            display_result(result, args.format)
        return

    current_board = SAMPLE_CURRENT_STATE if args.current is None else parse_board(args.current)
    result = evaluate_heuristics(current_board, goal_board)
    display_result(result, args.format)


if __name__ == "__main__":
    main()
