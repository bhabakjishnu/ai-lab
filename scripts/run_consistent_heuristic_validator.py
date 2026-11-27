"""Command-Line Interface and Runner for Consistent Heuristic Validator (Question 34).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟠 Advanced
Practical Type: Validation
Core AI Concept: Consistency / Monotonicity in Heuristics (Triangle Inequality)

Problem Statement:
    Write a Python program to check if a given heuristic h(n) for a graph
    is Consistent (Monotonic). The program must verify the triangle inequality:
        h(n) <= c(n, a, n') + h(n')
    for every node and its successors.

Task Requirements:
    1. Iterate through every edge (u, v) with cost c.
    2. Check if h(u) <= c + h(v).
    3. Output True if consistent, else pinpoint the violating edge.

Usage Examples:
    # 1. Exact Syllabus Sample Output:
    python scripts/run_consistent_heuristic_validator.py --sample

    # 2. Canonical Consistent Heuristic Check:
    python scripts/run_consistent_heuristic_validator.py --sample-consistent

    # 3. Comprehensive Comparison Table:
    python scripts/run_consistent_heuristic_validator.py --table

    # 4. Optional Challenge (Prove Consistency => Admissibility):
    python scripts/run_consistent_heuristic_validator.py --prove

    # 5. Extended Multi-Node Graph Demonstration:
    python scripts/run_consistent_heuristic_validator.py --extended

    # 6. Theoretical Analysis:
    python scripts/run_consistent_heuristic_validator.py --theory

    # 7. University Viva Voce Preparation Guide:
    python scripts/run_consistent_heuristic_validator.py --viva

    # 8. Structured JSON Export:
    python scripts/run_consistent_heuristic_validator.py --json
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

from src.consistent_heuristic_validator import (
    CANONICAL_EXTENDED_GRAPH,
    CANONICAL_EXTENDED_HEURISTIC_CONSISTENT,
    CANONICAL_EXTENDED_HEURISTIC_INCONSISTENT_ADMISSIBLE,
    CANONICAL_SAMPLE_GRAPH,
    CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
    CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
    SYLLABUS_SAMPLE_OUTPUT,
    ConsistencyProofReport,
    ConsistencyReport,
    EdgeViolation,
    format_consistency_table,
    get_theoretical_analysis,
    get_viva_reference,
    prove_consistency_implies_admissibility,
    render_ascii_graph,
    validate_consistency,
)


def run_syllabus_sample() -> None:
    """Print the exact canonical sample output specified by the syllabus."""
    report = validate_consistency(
        CANONICAL_SAMPLE_GRAPH,
        CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
    )
    print(report.summary_message)


def run_sample_consistent() -> None:
    """Run validation on canonical sample graph with a consistent heuristic."""
    report = validate_consistency(
        CANONICAL_SAMPLE_GRAPH,
        CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
    )
    print(report.summary_message)


def run_table_view() -> None:
    """Display comprehensive verification tables for both violating and consistent scenarios."""
    print("=" * 80)
    print("SCENARIO 1: SYLLABUS SAMPLE INPUT (VIOLATION)")
    print("=" * 80)
    print(render_ascii_graph(CANONICAL_SAMPLE_GRAPH, CANONICAL_SAMPLE_HEURISTIC_VIOLATING))
    report_violating = validate_consistency(
        CANONICAL_SAMPLE_GRAPH,
        CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
    )
    print(format_consistency_table(report_violating))

    print("=" * 80)
    print("SCENARIO 2: CANONICAL CONSISTENT HEURISTIC (PASS)")
    print("=" * 80)
    print(render_ascii_graph(CANONICAL_SAMPLE_GRAPH, CANONICAL_SAMPLE_HEURISTIC_CONSISTENT))
    report_consistent = validate_consistency(
        CANONICAL_SAMPLE_GRAPH,
        CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
    )
    print(format_consistency_table(report_consistent))


def run_extended_demo() -> None:
    """Run validation on extended multi-node graph comparing consistent vs inconsistent heuristics."""
    print("=" * 80)
    print("EXTENDED MULTI-NODE GRAPH VALIDATION DEMONSTRATION")
    print("=" * 80)
    print(render_ascii_graph(CANONICAL_EXTENDED_GRAPH, CANONICAL_EXTENDED_HEURISTIC_CONSISTENT))

    print("\n--- TEST 1: Consistent Heuristic ---")
    rep1 = validate_consistency(
        CANONICAL_EXTENDED_GRAPH,
        CANONICAL_EXTENDED_HEURISTIC_CONSISTENT,
    )
    print(format_consistency_table(rep1))

    print("\n--- TEST 2: Admissible BUT Inconsistent Heuristic ---")
    print("Key Concept: Admissibility does NOT guarantee consistency!")
    rep2 = validate_consistency(
        CANONICAL_EXTENDED_GRAPH,
        CANONICAL_EXTENDED_HEURISTIC_INCONSISTENT_ADMISSIBLE,
    )
    print(format_consistency_table(rep2))


def run_programmatic_proof() -> None:
    """Execute the Optional Challenge: Programmatically prove consistency implies admissibility."""
    print("=" * 80)
    print("OPTIONAL CHALLENGE: PROGRAMMATIC PROOF (CONSISTENCY => ADMISSIBILITY)")
    print("=" * 80)
    proof_report = prove_consistency_implies_admissibility(
        CANONICAL_EXTENDED_GRAPH,
        goal_node="Goal",
        heuristic=CANONICAL_EXTENDED_HEURISTIC_CONSISTENT,
    )
    print(proof_report.explanation)
    print("\nDetailed Step-by-Step Path Induction for Reachable Nodes:")
    for node, steps in proof_report.telescoping_steps.items():
        h_val = proof_report.heuristics[node]
        h_star = proof_report.true_shortest_costs[node]
        print(f"\nNode [{node}] -> Goal (h = {h_val:g}, h* = {h_star:g}):")
        if not steps:
            print("  [Goal node or trivial base case]")
            continue
        accum_cost = 0.0
        for step in steps:
            accum_cost += step.edge_cost
            print(f"  Step {step.step_index}: {step.inequality_str} {'[OK]' if step.is_valid else '[FAIL]'}")
        print(f"  Telescoping Sum: h({node}) <= sum(edge_costs) + h(Goal) = {accum_cost:g} + 0 = {accum_cost:g}")
        print(f"  Admissibility Confirmed: h({node}) = {h_val:g} <= h*({node}) = {h_star:g}")

    print("\n" + "=" * 80)
    print("Proof Result: All consistent heuristics with h(Goal)=0 are mathematically admissible!")
    print("=" * 80)


def run_json_export() -> None:
    """Output machine-readable JSON representation of the validation report."""
    report = validate_consistency(
        CANONICAL_SAMPLE_GRAPH,
        CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
    )
    print(report.to_json())


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for the validator runner."""
    parser = argparse.ArgumentParser(
        description="Consistent Heuristic Validator CLI (Question 34)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sample",
        action="store_true",
        help="Print exact syllabus sample output for sample input.",
    )
    mode_group.add_argument(
        "--sample-consistent",
        action="store_true",
        help="Run validator on canonical sample graph with consistent heuristic.",
    )
    mode_group.add_argument(
        "--table",
        action="store_true",
        help="Print formatted ASCII verification tables.",
    )
    mode_group.add_argument(
        "--extended",
        action="store_true",
        help="Run validation on extended multi-node graph.",
    )
    mode_group.add_argument(
        "--prove",
        action="store_true",
        help="Run Optional Challenge: Programmatic proof that consistency implies admissibility.",
    )
    mode_group.add_argument(
        "--theory",
        action="store_true",
        help="Print theoretical foundations and proofs.",
    )
    mode_group.add_argument(
        "--viva",
        action="store_true",
        help="Print university viva voce examination guide.",
    )
    mode_group.add_argument(
        "--json",
        action="store_true",
        help="Output validation results as structured JSON.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample:
        run_syllabus_sample()
    elif args.sample_consistent:
        run_sample_consistent()
    elif args.table:
        run_table_view()
    elif args.extended:
        run_extended_demo()
    elif args.prove:
        run_programmatic_proof()
    elif args.theory:
        print(get_theoretical_analysis())
    elif args.viva:
        print(get_viva_reference())
    elif args.json:
        run_json_export()
    else:
        # Default behavior: run syllabus sample output
        run_syllabus_sample()


if __name__ == "__main__":
    main()
