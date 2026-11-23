"""Command-Line Interface and Runner for 8-Puzzle Solver using A* (Question 30).

Practical Type: Integrated Problem
Difficulty: 🔴 Expert
Syllabus Area: Informed Search and Heuristics

Problem Statement:
    Develop a complete 8-puzzle solver in Python using the A* algorithm and the
    Manhattan distance heuristic. The program must read the initial state, find
    the optimal sequence of tile movements to reach the goal, and print the
    step-by-step boards.

Task Requirements:
    - State representation as a 1D tuple for fast hashing.
    - Implement valid move generation (swapping 0).
    - Calculate Manhattan distance on the fly.
    - Reconstruct and print the sequence of moves.
    - Optional Challenge: mathematically verify if board is solvable before searching.

Usage Examples:
    # 1. Default Run (Syllabus sample with detailed step-by-step boards):
    python scripts/run_a_star_eight_puzzle.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_a_star_eight_puzzle.py --sample

    # 3. Optional Challenge: Solvability & Inversion Parity Analysis:
    python scripts/run_a_star_eight_puzzle.py --check-solvability
    python scripts/run_a_star_eight_puzzle.py --initial "1,2,3,4,5,6,8,7,0" --check-solvability

    # 4. Compare Manhattan vs. Misplaced Tiles vs. BFS Baseline:
    python scripts/run_a_star_eight_puzzle.py --compare

    # 5. Step-by-Step State Transition Table:
    python scripts/run_a_star_eight_puzzle.py --table

    # 6. University Viva Voce Reference Guide:
    python scripts/run_a_star_eight_puzzle.py --viva

    # 7. Presets:
    python scripts/run_a_star_eight_puzzle.py --preset sample-2move
    python scripts/run_a_star_eight_puzzle.py --preset classic-2move
    python scripts/run_a_star_eight_puzzle.py --preset hard-18move

    # 8. Custom Initial & Goal States:
    python scripts/run_a_star_eight_puzzle.py --initial "[1, 2, 3, 4, 0, 5, 6, 7, 8]" --goal "[1, 2, 3, 4, 5, 8, 6, 7, 0]"
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

from src.a_star_eight_puzzle import (
    BENCHMARK_PRESETS,
    CLASSIC_GOAL_STATE,
    SAMPLE_GOAL_STATE_2MOVE,
    SAMPLE_INPUT_STATE,
    EightPuzzleResult,
    a_star_eight_puzzle,
    compare_search_heuristics,
    explain_solvability,
    get_eight_puzzle_viva_reference,
    is_solvable,
    parse_puzzle_state,
    render_board_ascii,
)


def run_sample_output(quiet: bool = False) -> str:
    """Print the exact syllabus sample output for Question 30: Solved in 2 moves: [Board states...]."""
    result = a_star_eight_puzzle(
        initial_state=SAMPLE_INPUT_STATE,
        goal_state=SAMPLE_GOAL_STATE_2MOVE,
    )
    formatted = result.format_sample_output()
    if not quiet:
        print(formatted)
    return formatted


def run_solvability_check(
    initial_raw: str,
    goal_raw: Optional[str] = None,
) -> None:
    """Run mathematical solvability verification and print inversion breakdown."""
    init_state = parse_puzzle_state(initial_raw)
    goal_state = parse_puzzle_state(goal_raw) if goal_raw else CLASSIC_GOAL_STATE

    is_solv, explanation, init_inv, goal_inv = explain_solvability(init_state, goal_state)

    print("=" * 70)
    print("8-PUZZLE MATHEMATICAL SOLVABILITY VERIFICATION REPORT")
    print("=" * 70)
    print(f"Initial State: {list(init_state)}")
    print(render_board_ascii(init_state, title="Initial Board:"))
    print(f"\nGoal State   : {list(goal_state)}")
    print(render_board_ascii(goal_state, title="Goal Board:"))
    print("-" * 70)
    print(f"Initial State Inversions: {init_inv} (Parity: {'EVEN' if init_inv % 2 == 0 else 'ODD'})")
    print(f"Goal State Inversions   : {goal_inv} (Parity: {'EVEN' if goal_inv % 2 == 0 else 'ODD'})")
    print(f"Solvability Status      : {'SOLVABLE' if is_solv else 'UNSOLVABLE'}")
    print("-" * 70)
    print(f"Mathematical Proof / Reason:\n  {explanation}")
    print("=" * 70)


def print_step_table(result: EightPuzzleResult) -> None:
    """Print an aligned tabular trace of the optimal path steps."""
    if not result.success:
        print(f"Search was unsuccessful: {result.solvability_reason}")
        return

    print("=" * 78)
    print("8-PUZZLE OPTIMAL SOLUTION STEP-BY-STEP TRACE TABLE")
    print("=" * 78)
    print(f"{'Step':<5} | {'Action':<18} | {'Blank':<6} | {'g':<3} | {'h':<3} | {'f':<3} | {'Board (Row-Major)':<22}")
    print("-" * 78)

    for rec in result.step_records:
        board_str = str(list(rec.state))
        print(
            f"{rec.step_number:<5} | {rec.action:<18} | {rec.blank_move:<6} | "
            f"{rec.g_cost:<3} | {rec.h_cost:<3} | {rec.f_cost:<3} | {board_str:<22}"
        )

    print("=" * 78)
    print(f"Total Moves: {result.move_count} | Nodes Expanded: {result.nodes_expanded} | "
          f"Nodes Generated: {result.nodes_generated} | Max Frontier: {result.max_frontier_size}")
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for Question 30: 8-Puzzle Solver."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_a_star_eight_puzzle.py",
        description="Question 30: 8-Puzzle Solver using A* and Manhattan Distance Heuristic",
    )

    parser.add_argument(
        "--initial",
        "-i",
        type=str,
        default=None,
        help="Initial board state as comma-separated or bracketed string, e.g. '[1,2,3, 4,0,5, 6,7,8]'",
    )
    parser.add_argument(
        "--goal",
        "-g",
        type=str,
        default=None,
        help="Target goal board state (defaults to classic (1-8, 0))",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print exact syllabus sample output ('Solved in 2 moves: [Board states...]')",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(BENCHMARK_PRESETS.keys()),
        default=None,
        help="Select a benchmark scenario preset",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare A* (Manhattan) vs A* (Misplaced Tiles) vs BFS baseline",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display step-by-step state transition table along optimal path",
    )
    parser.add_argument(
        "--check-solvability",
        action="store_true",
        help="Mathematically check whether the board is solvable using inversion parity",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display the comprehensive University Viva Voce reference guide",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the solution result as structured JSON",
    )

    args = parser.parse_args(argv)

    # 1. Exact Syllabus Sample Output
    if args.sample:
        run_sample_output()
        return

    # 2. Viva Voce Reference Guide
    if args.viva:
        print(get_eight_puzzle_viva_reference())
        return

    # 3. Determine Initial and Goal Configurations
    if args.preset:
        init_state, goal_state, desc = BENCHMARK_PRESETS[args.preset]
        print(f"Loaded Preset: '{args.preset}' - {desc}")
    elif args.initial:
        init_state = parse_puzzle_state(args.initial)
        goal_state = parse_puzzle_state(args.goal) if args.goal else CLASSIC_GOAL_STATE
    else:
        # Default: syllabus sample scenario
        init_state = SAMPLE_INPUT_STATE
        goal_state = SAMPLE_GOAL_STATE_2MOVE if args.goal is None else parse_puzzle_state(args.goal)

    # 4. Solvability Check Mode
    if args.check_solvability:
        run_solvability_check(str(list(init_state)), str(list(goal_state)))
        return

    # 5. Comparative Evaluation Mode
    if args.compare:
        report = compare_search_heuristics(init_state, goal_state)
        print(report.format_table())
        return

    # 6. Execute A* Search
    result = a_star_eight_puzzle(
        initial_state=init_state,
        goal_state=goal_state,
        heuristic="manhattan",
    )

    # 7. Output Handling
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.table:
        print_step_table(result)
        return

    # Default formatted output
    print(result.format_detailed_solution(use_ascii=True))


if __name__ == "__main__":
    main()
