"""Command-Line Interface and Runner for UCS vs A* Performance Comparison (Question 32).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟠 Advanced
Practical Type: Simulation
Core AI Concept: Informed vs. Uninformed Search Efficiency & Heuristic Directionality

Problem Statement:
    Write a Python program to solve a pathfinding problem on a 20x20 grid (with
    obstacles) using both Uniform Cost Search (UCS) and A* (Manhattan). Output
    the number of states evaluated by each. This practically demonstrates how
    heuristics guide search toward the goal.

Usage Examples:
    # 1. Default Run (Full Comparison Table & Summary):
    python scripts/run_ucs_vs_astar_comparison.py

    # 2. Exact Syllabus Sample Output Specification:
    python scripts/run_ucs_vs_astar_comparison.py --sample
    # Output: UCS expanded: 380 nodes. A* expanded: 45 nodes.

    # 3. Dynamic Computed Sample Output (Live solve on 20x20 grid):
    python scripts/run_ucs_vs_astar_comparison.py --sample-computed

    # 4. Comparative Metrics Table:
    python scripts/run_ucs_vs_astar_comparison.py --table

    # 5. Side-by-Side ASCII Maps of Node Expansion:
    python scripts/run_ucs_vs_astar_comparison.py --ascii

    # 6. Save Side-by-Side Matplotlib Plot (Optional Challenge):
    python scripts/run_ucs_vs_astar_comparison.py --visualize data/processed/ucs_vs_astar_comparison.png

    # 7. Theoretical Directionality & Complexity Analysis:
    python scripts/run_ucs_vs_astar_comparison.py --theory

    # 8. University Viva Voce Preparation Guide:
    python scripts/run_ucs_vs_astar_comparison.py --viva

    # 9. Structured JSON Export:
    python scripts/run_ucs_vs_astar_comparison.py --json

    # 10. Random Maze Simulation:
    python scripts/run_ucs_vs_astar_comparison.py --random-density 0.15 --seed 42
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

from src.ucs_vs_astar_comparison import (
    DEFAULT_20X20_HEIGHT,
    DEFAULT_20X20_WIDTH,
    SYLLABUS_SAMPLE_OUTPUT,
    ComparisonResult,
    Grid2D,
    SearchResult,
    compare_ucs_and_astar,
    create_canonical_20x20_grid,
    get_theoretical_analysis,
    get_viva_reference,
    visualize_ucs_vs_astar,
)


def run_sample_output(quiet: bool = False, computed: bool = False) -> str:
    """Print the syllabus sample output specification for Question 32.

    Args:
        quiet: If True, suppresses console printing.
        computed: If True, computes the live search result on the canonical grid;
            otherwise returns the literal syllabus sample string.

    Returns:
        Formatted sample output string.
    """
    if computed:
        comp = compare_ucs_and_astar()
        out = comp.format_sample_output(use_literal_sample=False)
    else:
        out = SYLLABUS_SAMPLE_OUTPUT

    if not quiet:
        print(out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for Question 32 UCS vs A* Performance Comparison.

    Args:
        argv: Command line argument tokens (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Uniform Cost Search (UCS) vs A* (Manhattan) Performance Comparison on 20x20 Grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print exact syllabus sample output: 'UCS expanded: 380 nodes. A* expanded: 45 nodes.'",
    )
    parser.add_argument(
        "--sample-computed",
        action="store_true",
        help="Run live search on canonical 20x20 grid and print computed expansion counts.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display detailed comparative performance metrics table.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Display side-by-side ASCII maps showing expanded nodes and solution path.",
    )
    parser.add_argument(
        "--visualize",
        metavar="FILEPATH",
        nargs="?",
        const="data/processed/ucs_vs_astar_comparison.png",
        help="Generate and save side-by-side Matplotlib visualization (Optional Challenge).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display interactive Matplotlib visualization window.",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Display theoretical analysis of heuristic pull, directionality, and pruning.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive University Viva Voce study guide.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full comparison results as structured JSON.",
    )
    parser.add_argument(
        "--random-density",
        type=float,
        default=None,
        help="Generate a random grid with specified obstacle density (0.0 to 0.4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible synthetic grid generation.",
    )

    args = parser.parse_args(argv)

    # 1. Exact syllabus sample output
    if args.sample:
        run_sample_output(quiet=False, computed=False)
        return 0

    # 2. Dynamic computed sample output
    if args.sample_computed:
        run_sample_output(quiet=False, computed=True)
        return 0

    # 3. Theory guide
    if args.theory:
        print(get_theoretical_analysis())
        return 0

    # 4. Viva guide
    if args.viva:
        print(get_viva_reference())
        return 0

    # Grid construction
    if args.random_density is not None:
        density = max(0.0, min(0.5, args.random_density))
        grid = Grid2D.generate_random(
            width=DEFAULT_20X20_WIDTH,
            height=DEFAULT_20X20_HEIGHT,
            obstacle_density=density,
            seed=args.seed,
        )
    else:
        grid = create_canonical_20x20_grid()

    # Run side-by-side search
    comp = compare_ucs_and_astar(grid=grid)

    # 5. JSON output
    if args.json:
        print(json.dumps(comp.to_dict(), indent=2))
        return 0

    # 6. ASCII maps
    if args.ascii:
        print(comp.format_ascii_side_by_side())
        print()

    # 7. Matplotlib visualization
    if args.visualize or args.plot:
        out_path = args.visualize
        visualize_ucs_vs_astar(
            comparison=comp,
            save_path=out_path,
            show=args.plot,
        )

    # 8. Comparison table or default summary
    if args.table or not (args.ascii or args.visualize or args.plot):
        print(comp.format_comparison_table())
        print()
        print(f"Sample Output: {comp.format_sample_output()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
