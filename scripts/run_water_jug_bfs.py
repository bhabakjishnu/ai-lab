"""Runner script for Water Jug Problem using Breadth-First Search (Question 20).

Solves the classic Water Jug Problem using Breadth-First Search (BFS) over
an implicitly generated state-space with the 6 canonical production rules.

Usage Examples:
    # 1. Default Sample Run (Capacities: 4, 3; Target: 2):
    python scripts/run_water_jug_bfs.py

    # 2. Exact Syllabus Single-Line Output Format:
    python scripts/run_water_jug_bfs.py --format single

    # 3. Custom Capacities and Target:
    python scripts/run_water_jug_bfs.py -x 5 -y 3 -z 4

    # 4. Target in Either Jug ('any'):
    python scripts/run_water_jug_bfs.py -x 4 -y 3 -z 2 --target-jug any

    # 5. Run Optional Challenge (Algebraic Reachability & GCD Rule Analysis):
    python scripts/run_water_jug_bfs.py --challenge

    # 6. Run Comprehensive Automated Benchmark Suite:
    python scripts/run_water_jug_bfs.py --demo

    # 7. Display Viva Voce Study Guide & Algebraic Proofs:
    python scripts/run_water_jug_bfs.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.water_jug_bfs import (
    COPRIME_WATER_JUG_PARAMS,
    DIE_HARD_WATER_JUG_PARAMS,
    EQUAL_CAPACITY_PARAMS,
    SAMPLE_WATER_JUG_PARAMS,
    UNSOLVABLE_WATER_JUG_PARAMS,
    WaterJugSearchResult,
    WaterJugSolver,
    check_reachability_algebraic,
    get_water_jug_viva_reference,
    solve_water_jug_bfs,
)


def print_banner(title: str) -> None:
    """Print a styled laboratory section header."""
    width = 75
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_viva_guide() -> None:
    """Print the complete Viva Voce examination reference guide."""
    viva = get_water_jug_viva_reference()
    print_banner("QUESTION NO. 20: WATER JUG PROBLEM VIVA VOCE THEORY GUIDE")

    for topic_title, content in viva.items():
        print(f"[{topic_title}]")
        print(f"Question: {content['question']}")
        print("Answer:")
        for line in content["answer"].split("\n"):
            cleaned = line.strip()
            if cleaned:
                print(f"  • {cleaned}")
        print("-" * 75)


def run_algebraic_challenge() -> None:
    """Execute Optional Challenge: Algebraic reachability and GCD rule analysis."""
    print_banner("OPTIONAL CHALLENGE: ALGEBRAIC REACHABILITY & GCD RULE ANALYSIS")
    print(
        "Theorem (Bézout's Identity & Diophantine Solvability):\n"
        "Let X and Y be jug capacities. Every fill, empty, or pour operation changes water volumes\n"
        "by integer multiples of X and Y: a*X + b*Y = Z.\n"
        "Therefore, target Z is measurable in one of the jugs IF AND ONLY IF:\n"
        "  1. Z <= max(X, Y)  [Capacity Upper Bound]\n"
        "  2. Z % gcd(X, Y) == 0  [Bézout's Divisibility Condition]\n"
    )

    test_cases = [
        ("Case 1: Canonical Sample Problem", 4, 3, 2, "1"),
        ("Case 2: Die Hard 3 Puzzle", 5, 3, 4, "1"),
        ("Case 3: Co-prime Capacities", 7, 5, 4, "1"),
        ("Case 4: Unsolvable GCD Violation (Even Jugs, Odd Target)", 6, 4, 5, "any"),
        ("Case 5: Unsolvable Capacity Exceeded", 3, 5, 7, "any"),
        ("Case 6: Target in Jug 1 Exceeds Jug 1 Capacity", 3, 7, 5, "1"),
        ("Case 7: Trivial Zero Target", 4, 3, 0, "1"),
    ]

    for title, x, y, z, tj in test_cases:
        res = check_reachability_algebraic(x, y, z, target_jug=tj)
        status = "REACHABLE" if res.is_reachable else "UNREACHABLE"
        print(f"--- {title} ---")
        print(f"  • Inputs             : Jug 1={x}L, Jug 2={y}L, Target={z}L, Target Jug={tj}")
        print(f"  • gcd({x}, {y})        : {res.gcd_value}")
        print(f"  • Bézout Identity    : ({res.bezout_x})*({x}) + ({res.bezout_y})*({y}) = {res.gcd_value}")
        print(f"  • Algebraic Outcome  : [{status}]")
        print(f"  • Mathematical Reason: {res.reason}")
        print()


def run_benchmarks() -> None:
    """Run comprehensive automated benchmarks comparing Water Jug configurations."""
    print_banner("WATER JUG BFS BENCHMARK SUITE")

    benchmarks = [
        ("Benchmark 1: Canonical Sample (Syllabus Default)", 4, 3, 2, "1"),
        ("Benchmark 2: Die Hard 3 Puzzle (5L & 3L -> 4L)", 5, 3, 4, "1"),
        ("Benchmark 3: Co-prime Capacities (7L & 5L -> 4L)", 7, 5, 4, "1"),
        ("Benchmark 4: Equal Capacities (4L & 4L -> 4L)", 4, 4, 4, "1"),
        ("Benchmark 5: Unsolvable (6L & 4L -> 5L)", 6, 4, 5, "any"),
    ]

    for title, x, y, z, tj in benchmarks:
        print(f"--- {title} ---")
        res = solve_water_jug_bfs(x, y, z, target_jug=tj)
        print(f"  • Configuration     : Jug 1={x}L, Jug 2={y}L, Target={z}L (Target Jug: {tj})")
        print(f"  • Algebraic Reach   : {'Yes' if res.algebraic_check.is_reachable else 'No'}")
        print(f"  • BFS Solvable      : {'Yes' if res.is_solvable else 'No'}")
        if res.is_solvable:
            print(f"  • Shortest Path     : {res.arrow_path_string}")
            print(f"  • Total Steps       : {res.step_count}")
            print(f"  • Nodes Explored    : {res.nodes_explored}")
            print(f"  • Nodes Generated   : {res.nodes_generated}")
            print(f"  • Execution Time    : {res.execution_time_ms:.4f} ms")
        else:
            print(f"  • Reason            : {res.algebraic_check.reason}")
        print()

    # Comparison: Target in Jug 1 vs Target in Any Jug on Sample Problem
    print("--- Benchmark 6: Target in Jug 1 vs Target in Any Jug (Capacities: 4L, 3L; Target: 2L) ---")
    res_jug1 = solve_water_jug_bfs(4, 3, 2, target_jug="1")
    res_any = solve_water_jug_bfs(4, 3, 2, target_jug="any")
    print(f"  • Target in Jug 1 (Canonical) : {res_jug1.arrow_path_string} ({res_jug1.step_count} steps)")
    print(f"  • Target in Any Jug (Shortest): {res_any.arrow_path_string} ({res_any.step_count} steps)")
    print()


def main() -> None:
    """CLI dispatcher for Water Jug Problem BFS simulation."""
    parser = argparse.ArgumentParser(
        description="Question 20: Water Jug Problem Solver using Breadth-First Search (BFS)."
    )
    parser.add_argument(
        "-x", "--capacity-x",
        type=int,
        default=4,
        help="Capacity of Jug 1 in liters (default: 4).",
    )
    parser.add_argument(
        "-y", "--capacity-y",
        type=int,
        default=3,
        help="Capacity of Jug 2 in liters (default: 3).",
    )
    parser.add_argument(
        "-z", "--target-z",
        type=int,
        default=2,
        help="Target water amount in liters (default: 2).",
    )
    parser.add_argument(
        "--target-jug",
        type=str,
        choices=["1", "2", "any"],
        default="1",
        help="Target jug constraint: '1' (in Jug 1), '2' (in Jug 2), or 'any' (either jug). Default is '1'.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["full", "single", "table", "json"],
        default="full",
        help="Output format: 'full' (verbose summary & table), 'single' (exact sample line), 'table', or 'json'.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark suite across multiple problem configurations.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge: Algebraic reachability and GCD rule analysis.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce questions and answers.",
    )

    args = parser.parse_args()

    # Mode 1: Viva Guide
    if args.viva:
        print_viva_guide()
        return

    # Mode 2: Optional Challenge Algebraic Reachability
    if args.challenge:
        run_algebraic_challenge()
        return

    # Mode 3: Demo Benchmark Suite
    if args.demo:
        run_benchmarks()
        return

    # Mode 4: Standard / Custom Execution
    result = solve_water_jug_bfs(
        capacity_x=args.capacity_x,
        capacity_y=args.capacity_y,
        target_z=args.target_z,
        target_jug=args.target_jug,
    )

    # Format output
    if args.format == "single":
        print(result.arrow_path_string)
        return

    if args.format == "table":
        print(result.format_table())
        return

    if args.format == "json":
        print(result.to_json())
        return

    # Full Format
    print_banner("QUESTION NO. 20: WATER JUG PROBLEM USING BFS")
    print("SYLLABUS SAMPLE OUTPUT:")
    print(f"  >>> {result.arrow_path_string}\n")

    print(result.format_summary())
    print("\nSTEP-BY-STEP TRANSITION TRACE TABLE:")
    print(result.format_table())
    print("\nASCII WATER LEVEL DIAGRAMS:")
    print(result.format_ascii_visual_path())


if __name__ == "__main__":
    main()
