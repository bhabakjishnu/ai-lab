"""Command-Line Interface and Runner for 8-Puzzle: Manhattan vs Misplaced Tiles (Question 31).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🔴 Expert
Practical Type: Comparison / Analysis
Core AI Concept: Heuristic Dominance & Effective Branching Factor (b*)

Problem Statement:
    Extend the 8-puzzle solver to solve the same board twice: once using h1
    (Misplaced Tiles) and once using h2 (Manhattan Distance). Track and compare
    the exact number of nodes expanded by each heuristic to prove that h2
    dominates h1.

Usage Examples:
    # 1. Default Run (Full Comparison & Dominance Verification):
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --sample

    # 3. Dynamic Computed Sample Output (Live A* solve on syllabus board):
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --sample-computed

    # 4. Comparison Table:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --table

    # 5. Theoretical Dominance Proof:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --proof

    # 6. Viva Voce Study Guide:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --viva

    # 7. Check Mathematical Solvability (Optional Challenge):
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --check-solvability

    # 8. Presets (e.g. 18-move benchmark, 2-move, 8-move):
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --preset sample-18move

    # 9. Custom Boards:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --initial "[7, 2, 4, 5, 0, 6, 8, 3, 1]" --goal "[1, 2, 3, 4, 5, 6, 7, 8, 0]"

    # 10. JSON Export:
    python scripts/run_eight_puzzle_manhattan_vs_misplaced.py --json
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

from src.eight_puzzle_manhattan_vs_misplaced import (
    BENCHMARK_PRESETS,
    CLASSIC_GOAL_STATE,
    SAMPLE_INPUT_STATE,
    SYLLABUS_SAMPLE_OUTPUT,
    EightPuzzleDominanceAnalyzer,
    HeuristicComparisonResult,
    compare_eight_puzzle_heuristics,
    explain_solvability,
    get_viva_reference,
    is_solvable,
    parse_puzzle_state,
    render_board_ascii,
    render_side_by_side,
)


def run_sample_output(quiet: bool = False, computed: bool = False) -> str:
    """Print the syllabus sample output specification for Question 31.

    Args:
        quiet: If True, suppresses console printing.
        computed: If True, computes the live search result on the starting board;
            otherwise returns the textbook illustrative sample string.

    Returns:
        Formatted sample output string.
    """
    if computed:
        res = compare_eight_puzzle_heuristics(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=CLASSIC_GOAL_STATE,
        )
        out = res.format_sample_output(use_literal_sample=False)
    else:
        out = SYLLABUS_SAMPLE_OUTPUT

    if not quiet:
        print(out)
    return out


def run_solvability_check(
    initial_raw: str,
    goal_raw: Optional[str] = None,
) -> None:
    """Run mathematical solvability verification and print inversion breakdown."""
    init_state = parse_puzzle_state(initial_raw)
    goal_state = parse_puzzle_state(goal_raw) if goal_raw else CLASSIC_GOAL_STATE

    is_solv, explanation, init_inv, goal_inv = explain_solvability(init_state, goal_state)

    print("=" * 78)
    print("  8-PUZZLE MATHEMATICAL SOLVABILITY & INVERSION PARITY REPORT")
    print("=" * 78)
    print(render_side_by_side(init_state, goal_state))
    print("-" * 78)
    print(f"Initial State Inversions : {init_inv} (parity: {'even' if init_inv % 2 == 0 else 'odd'})")
    print(f"Goal State Inversions    : {goal_inv} (parity: {'even' if goal_inv % 2 == 0 else 'odd'})")
    print(f"Solvability Status       : {'SOLVABLE' if is_solv else 'UNSOLVABLE'}")
    print(f"Mathematical Reasoning   : {explanation}")
    print("=" * 78)


def run_viva_guide() -> None:
    """Print the complete University Viva Voce study guide."""
    viva = get_viva_reference()
    meta = viva["question_metadata"]

    print("=" * 78)
    print(f"  {meta['title']}")
    print(f"  Syllabus Area: {meta['syllabus_area']} | Difficulty: {meta['difficulty']}")
    print(f"  Core AI Concept: {meta['core_concept']}")
    print("=" * 78)
    print()

    for i, item in enumerate(viva["viva_topics"], start=1):
        print(f"Topic {i}: {item['topic']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print("-" * 78)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point for Question 31."""
    parser = argparse.ArgumentParser(
        description="Question 31: 8-Puzzle Manhattan vs Misplaced Tiles Dominance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--initial",
        type=str,
        default=None,
        help="Initial board state (1D list, 2D matrix, or delimited string).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal board state (defaults to classic 1-8, 0).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(BENCHMARK_PRESETS.keys()),
        default=None,
        help="Run a standard benchmark preset.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Output the exact syllabus sample output string.",
    )
    parser.add_argument(
        "--sample-computed",
        action="store_true",
        help="Output dynamic sample output from live execution.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print full comparative metric table.",
    )
    parser.add_argument(
        "--proof",
        action="store_true",
        help="Print theoretical proof and empirical verification of dominance.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display Viva Voce concepts and examination guide.",
    )
    parser.add_argument(
        "--check-solvability",
        action="store_true",
        help="Verify inversion parity and solvability without searching.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as a JSON object.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=300_000,
        help="Maximum node expansion cutoff limit (default: 300,000).",
    )

    args = parser.parse_args(argv)

    # 1. Exact Syllabus Sample Mode
    if args.sample:
        run_sample_output(computed=False)
        return

    # 2. Computed Sample Output Mode
    if args.sample_computed:
        run_sample_output(computed=True)
        return

    # 3. Viva Voce Mode
    if args.viva:
        run_viva_guide()
        return

    # 4. Solvability Check Mode
    if args.check_solvability:
        initial_val = args.initial or str(list(SAMPLE_INPUT_STATE))
        run_solvability_check(initial_val, args.goal)
        return

    # Resolve initial and goal states
    if args.preset:
        init_state, goal_state, desc = BENCHMARK_PRESETS[args.preset]
    else:
        init_state = (
            parse_puzzle_state(args.initial)
            if args.initial
            else SAMPLE_INPUT_STATE
        )
        goal_state = (
            parse_puzzle_state(args.goal)
            if args.goal
            else CLASSIC_GOAL_STATE
        )

    # Execute comparative search
    comparison_result = compare_eight_puzzle_heuristics(
        initial_state=init_state,
        goal_state=goal_state,
        max_nodes=args.max_nodes,
    )

    # JSON output mode
    if args.json:
        print(json.dumps(comparison_result.to_dict(), indent=2))
        return

    # Proof output mode
    if args.proof:
        print(comparison_result.format_dominance_proof())
        return

    # Table output mode
    if args.table:
        print(comparison_result.format_comparison_table())
        return

    # Default verbose mode
    print("=" * 78)
    print("  QUESTION NO. 31: 8-PUZZLE MANHATTAN VS MISPLACED TILES")
    print("  Syllabus Area : Informed Search and Heuristics | Difficulty: Expert")
    print("  Core Concept  : Heuristic Dominance & Effective Branching Factor (b*)")
    print("=" * 78)
    print()
    print("BOARD CONFIGURATIONS:")
    print(render_side_by_side(comparison_result.initial_state, comparison_result.goal_state))
    print()
    print(comparison_result.format_comparison_table())
    print()
    print("SYLLABUS SAMPLE OUTPUT:")
    print(f"  Live Solved : {comparison_result.format_sample_output()}")
    print(f"  Benchmark   : {SYLLABUS_SAMPLE_OUTPUT}")
    print()
    print("CORE AI INSIGHT:")
    if comparison_result.is_dominated:
        print(
            f"  [PROVED] Manhattan Distance (h2) dominates Misplaced Tiles (h1).\n"
            f"  A* expanded {comparison_result.h2_result.nodes_expanded} nodes with h2 "
            f"vs {comparison_result.h1_result.nodes_expanded} nodes with h1.\n"
            f"  Manhattan reduced node expansions by {comparison_result.expansion_reduction_percent:.2f}% "
            f"while finding the EXACT same optimal path length ({comparison_result.path_length}).\n"
            f"  Effective branching factor b* decreased from {comparison_result.h1_result.effective_branching_factor:.4f} "
            f"to {comparison_result.h2_result.effective_branching_factor:.4f}."
        )
    else:
        print("  Warning: Dominance condition did not hold on this board.")


if __name__ == "__main__":
    main()
