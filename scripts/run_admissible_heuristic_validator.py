"""Command-Line Interface and Runner for Admissible Heuristic Validator (Question 33).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟡 Intermediate
Practical Type: Validation
Core AI Concept: Admissibility (h(n) <= h*(n))

Problem Statement:
    Given a graph, the true minimum costs from every node to the goal, and a
    heuristic dictionary h(n), write a Python function to programmatically
    determine if the provided heuristic is Admissible.

Task Requirements:
    1. Calculate true cost h*(n) for all nodes using a reverse UCS from the goal.
    2. Compare provided h(n) against h*(n).
    3. Identify any node where h(n) > h*(n).
    4. Output True (Admissible) or False (Not Admissible with counterexample).

Usage Examples:
    # 1. Exact Syllabus Sample Output:
    python scripts/run_admissible_heuristic_validator.py --sample

    # 2. Dynamic Computed Run on Canonical Graph:
    python scripts/run_admissible_heuristic_validator.py --sample-computed

    # 3. Full Comparison Table:
    python scripts/run_admissible_heuristic_validator.py --table

    # 4. Optional Challenge (Heuristic Repair / Lowering):
    python scripts/run_admissible_heuristic_validator.py --repair

    # 5. Theoretical Analysis:
    python scripts/run_admissible_heuristic_validator.py --theory

    # 6. University Viva Voce Preparation Guide:
    python scripts/run_admissible_heuristic_validator.py --viva

    # 7. Structured JSON Export:
    python scripts/run_admissible_heuristic_validator.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.admissible_heuristic_validator import (
    CANONICAL_GOAL,
    CANONICAL_GRAPH,
    CANONICAL_HEURISTIC_ADMISSIBLE,
    CANONICAL_HEURISTIC_OVERESTIMATING,
    SYLLABUS_SAMPLE_OUTPUT,
    AdmissibilityReport,
    format_comparison_table,
    get_canonical_sample_graph,
    get_theoretical_analysis,
    get_viva_reference,
    render_ascii_graph,
    validate_admissibility,
)


def run_sample_output(quiet: bool = False, computed: bool = False) -> str:
    """Print the syllabus sample output specification for Question 33.

    Args:
        quiet: If True, suppresses console printing.
        computed: If True, computes the live validation result on the canonical graph;
                  if False, prints the exact syllabus output constant.

    Returns:
        Formatted sample output string.
    """
    if computed:
        graph, goal, heuristic = get_canonical_sample_graph()
        report = validate_admissibility(graph, goal, heuristic)
        output = report.counterexample
    else:
        output = SYLLABUS_SAMPLE_OUTPUT

    if not quiet:
        print(output)
    return output


def run_full_validation(
    graph_input: Optional[Dict[str, Any]] = None,
    goal: Optional[str] = None,
    heuristic_input: Optional[Dict[str, float]] = None,
    show_table: bool = True,
    show_repair: bool = False,
    show_ascii: bool = True,
    as_json: bool = False,
) -> AdmissibilityReport:
    """Execute complete admissibility validation workflow.

    Args:
        graph_input: Optional custom graph dictionary.
        goal: Optional custom goal node ID.
        heuristic_input: Optional custom heuristic dictionary.
        show_table: If True, prints comparison table.
        show_repair: If True, demonstrates optional challenge heuristic repair.
        show_ascii: If True, prints ASCII topological diagram.
        as_json: If True, outputs raw JSON.

    Returns:
        Completed AdmissibilityReport.
    """
    target_graph = graph_input or CANONICAL_GRAPH
    target_goal = goal or CANONICAL_GOAL
    target_heuristic = heuristic_input or CANONICAL_HEURISTIC_OVERESTIMATING

    report = validate_admissibility(target_graph, target_goal, target_heuristic)

    if as_json:
        print(report.to_json())
        return report

    print("\n" + "=" * 78)
    print(" QUESTION 33: ADMISSIBLE HEURISTIC VALIDATOR (REVERSE UCS)")
    print("=" * 78)
    print(f" Target Goal Node : '{report.goal_node}'")
    print(f" Evaluation Status : {'✅ ADMISSIBLE' if report.is_admissible else '❌ NOT ADMISSIBLE'}")
    print(f" Counterexample   : {report.counterexample}")
    print(f" Execution Time   : {report.execution_time_ms:.4f} ms")
    print("=" * 78 + "\n")

    if show_ascii and isinstance(target_graph, dict):
        print(render_ascii_graph(target_graph, report.goal_node))
        print("-" * 78)

    if show_table:
        print("\nComparative Heuristic vs. True Cost Table:")
        print(format_comparison_table(report))
        print("-" * 78)

    if show_repair or not report.is_admissible:
        print("\nOptional Challenge: Automatically Lowered Heuristic Values:")
        print("-" * 78)
        for node, rep_val in sorted(report.repaired_heuristics.items()):
            orig_val = report.heuristics.get(node, 0.0)
            if orig_val > rep_val:
                print(f"  Node {node:5s} : Clamped from {orig_val:g} -> {rep_val:g} (Admissible upper bound)")
            else:
                print(f"  Node {node:5s} : Retained at {orig_val:g} (Already admissible)")
        print("-" * 78)

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """Parse command line arguments and execute the corresponding runner action."""
    parser = argparse.ArgumentParser(
        description="Question 33: Admissible Heuristic Validator via Reverse UCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print exact syllabus sample output: 'False. Heuristic overestimates at node A (10 > 8).'",
    )
    parser.add_argument(
        "--sample-computed",
        action="store_true",
        help="Compute and print live validation output on canonical graph",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display full markdown comparison table of h(n) vs h*(n)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Show repaired admissible heuristic values (Optional Challenge)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Display ASCII topology graph representation",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Display in-depth theoretical analysis of admissibility",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display University Viva Voce preparation guide",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON representation of the validation report",
    )
    parser.add_argument(
        "--admissible-test",
        action="store_true",
        help="Run validation on canonical admissible heuristic to test True outcome",
    )
    parser.add_argument(
        "--graph-json",
        type=str,
        help="Path to JSON file or raw JSON string defining custom graph adjacency",
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="Identifier of target goal node in custom graph",
    )
    parser.add_argument(
        "--heuristic-json",
        type=str,
        help="Path to JSON file or raw JSON string defining custom heuristic dictionary",
    )

    args = parser.parse_args(argv)

    if args.sample:
        run_sample_output(computed=False)
        return

    if args.sample_computed:
        run_sample_output(computed=True)
        return

    if args.theory:
        print(get_theoretical_analysis())
        return

    if args.viva:
        print(get_viva_reference())
        return

    # Parse optional custom inputs
    custom_graph = None
    if args.graph_json:
        p = Path(args.graph_json)
        if p.exists():
            custom_graph = json.loads(p.read_text(encoding="utf-8"))
        else:
            custom_graph = json.loads(args.graph_json)

    custom_heuristic = None
    if args.heuristic_json:
        p = Path(args.heuristic_json)
        if p.exists():
            custom_heuristic = json.loads(p.read_text(encoding="utf-8"))
        else:
            custom_heuristic = json.loads(args.heuristic_json)

    if args.admissible_test:
        custom_heuristic = CANONICAL_HEURISTIC_ADMISSIBLE

    run_full_validation(
        graph_input=custom_graph,
        goal=args.goal,
        heuristic_input=custom_heuristic,
        show_table=args.table or not (args.json or args.ascii or args.repair),
        show_repair=args.repair,
        show_ascii=args.ascii or not (args.json or args.table or args.repair),
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
