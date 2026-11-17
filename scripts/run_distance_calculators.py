"""Command-Line Interface and Runner for Manhattan and Euclidean Distance Calculators (Question 24).

Computes and compares Manhattan (L1), Euclidean (L2), Chebyshev (L_infinity),
and Octile distance metrics across 2D coordinate pairs.

Usage Examples:
    # 1. Default Sample Run ((1, 1) and (4, 5)):
    python scripts/run_distance_calculators.py

    # 2. Exact syllabus sample output:
    python scripts/run_distance_calculators.py --format sample

    # 3. Custom Point Pair:
    python scripts/run_distance_calculators.py --p1 1 1 --p2 4 5

    # 4. Multiple Point Pairs:
    python scripts/run_distance_calculators.py --pairs "(1,1),(4,5); (0,0),(3,4); (2,3),(8,11)"

    # 5. Interactive Mode (prompts user for point pairs):
    python scripts/run_distance_calculators.py --interactive

    # 6. ASCII Grid Trajectory Visualization:
    python scripts/run_distance_calculators.py --ascii

    # 7. Metric Space Axiom Verification:
    python scripts/run_distance_calculators.py --axioms

    # 8. Movement Topology & Heuristic Admissibility Analysis:
    python scripts/run_distance_calculators.py --topology orthogonal

    # 9. Matplotlib Visualization Plot:
    python scripts/run_distance_calculators.py --plot --save-plot data/distance_metrics_plot.png

    # 10. University Viva Voce Reference Guide:
    python scripts/run_distance_calculators.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.distance_calculators import (
    Point2D,
    PointLike,
    DistanceResult,
    manhattan,
    euclidean,
    chebyshev,
    octile,
    compute_distances,
    compare_point_pairs,
    format_comparison_table,
    parse_point,
    parse_point_pairs,
    verify_metric_space_axioms,
    compare_heuristics_for_grid_topology,
    render_ascii_distance_grid,
    plot_distance_metrics,
    get_distance_viva_reference,
)

# Canonical benchmark pairs for comparison
DEFAULT_BENCHMARK_PAIRS: List[Tuple[PointLike, PointLike]] = [
    ((1, 1), (4, 5)),      # Canonical Syllabus Sample Pair: dx=3, dy=4 -> M=7, E=5.00
    ((0, 0), (3, 4)),      # Classic 3-4-5 Pythagorean right triangle
    ((0, 0), (5, 0)),      # Collinear Horizontal (Manhattan == Euclidean == Chebyshev)
    ((2, 3), (2, 9)),      # Collinear Vertical (Manhattan == Euclidean == Chebyshev)
    ((0, 0), (4, 4)),      # 45-degree Diagonal (Max M/E ratio = sqrt(2) ~ 1.414)
    ((-3, -2), (5, 6)),    # Crossing Quadrants (dx=8, dy=8 -> M=16, E=11.31)
    ((1.5, 2.5), (4.5, 6.5)), # Floating-Point Coordinates
]


def display_viva_reference() -> None:
    """Print comprehensive Viva Voce preparation guide to stdout."""
    viva_dict = get_distance_viva_reference()
    print("=" * 80)
    print("AI LAB VIVA VOCE STUDY GUIDE: INFORMED SEARCH & DISTANCE METRICS")
    print("=" * 80)
    for topic_key, content in viva_dict.items():
        title = topic_key.replace("_", " ").title()
        print(f"\n--- {title} ---")
        print(f"Q: {content['question']}")
        print(f"A:\n{content['answer']}")
        for extra_key, extra_val in content.items():
            if extra_key not in ("question", "answer"):
                print(f"[{extra_key.replace('_', ' ').upper()}]: {extra_val}")
    print("\n" + "=" * 80)


def display_axiom_verification() -> None:
    """Run and display metric space axiom verification for all metrics."""
    p1 = (1, 1)
    p2 = (4, 5)
    p3 = (2, 8)

    print("=" * 80)
    print("METRIC SPACE AXIOM VERIFICATION")
    print(f"Test Points: P1={p1}, P2={p2}, P3={p3}")
    print("=" * 80)

    for metric_name, fn in [
        ("Manhattan Distance (L1)", manhattan),
        ("Euclidean Distance (L2)", euclidean),
        ("Chebyshev Distance (L_inf)", chebyshev),
        ("Octile Distance", octile),
    ]:
        report = verify_metric_space_axioms(p1, p2, p3, fn)
        status = "PASSED" if report["all_satisfied"] else "FAILED"
        print(f"\nMetric: {metric_name} [{status}]")
        print(f"  1. Non-negativity: d(P1, P2) = {report['non_negativity']['d(p1, p2)']:.4f} >= 0 -> {report['non_negativity']['satisfied']}")
        print(f"  2. Identity:       d(P1, P1) = {report['identity_of_indiscernibles']['d(p1, p1)']:.4f} == 0 -> {report['identity_of_indiscernibles']['satisfied']}")
        print(f"  3. Symmetry:       d(P1, P2)={report['symmetry']['d(p1, p2)']:.4f}, d(P2, P1)={report['symmetry']['d(p2, p1)']:.4f} -> {report['symmetry']['satisfied']}")
        print(f"  4. Triangle Ineq:  d(P1, P3)={report['triangle_inequality']['d(p1, p3)']:.4f} <= d(P1,P2)+d(P2,P3)={report['triangle_inequality']['d(p1, p2) + d(p2, p3)']:.4f} (slack={report['triangle_inequality']['slack']:.4f}) -> {report['triangle_inequality']['satisfied']}")


def display_topology_analysis(topology: str = "orthogonal") -> None:
    """Run and display heuristic admissibility analysis under specified topology."""
    p1 = (1, 1)
    p2 = (4, 5)
    analysis = compare_heuristics_for_grid_topology(p1, p2, topology=topology)

    print("=" * 80)
    print(f"HEURISTIC ADMISSIBILITY & DOMINANCE ANALYSIS: Topology='{topology}'")
    print(f"Start={p1}, Goal={p2} | True Unobstructed Cost h* = {analysis['true_shortest_cost_without_obstacles']:.2f}")
    print("=" * 80)

    for h_name, data in analysis["evaluations"].items():
        admissible_str = "YES (Admissible)" if data["is_admissible"] else "NO (INADMISSIBLE - RISKS SUBOPTIMAL A*!)"
        print(f"  • {h_name:<10}: h(n) = {data['cost']:.2f} | Admissible: {admissible_str:<25} | Status: {data['quality']}")


def run_interactive_mode() -> List[Tuple[PointLike, PointLike]]:
    """Prompt user to interactively enter coordinate pairs."""
    print("\n--- Interactive Point Pair Distance Calculator ---")
    print("Enter coordinate pairs in format: x1 y1 x2 y2 (or press Enter without input to finish).")
    pairs: List[Tuple[PointLike, PointLike]] = []
    idx = 1
    while True:
        try:
            line = input(f"Pair {idx} [x1 y1 x2 y2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        try:
            parts = line.replace(",", " ").split()
            if len(parts) == 4:
                p1 = (float(parts[0]), float(parts[1]))
                p2 = (float(parts[2]), float(parts[3]))
                pairs.append((p1, p2))
                idx += 1
            else:
                print("  [Error] Please enter exactly 4 numbers: x1 y1 x2 y2")
        except ValueError as exc:
            print(f"  [Error] Invalid numbers: {exc}")

    if not pairs:
        print("No pairs entered. Using default sample pair: (1, 1), (4, 5)")
        pairs.append(((1, 1), (4, 5)))
    return pairs


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point dispatcher for Question 24 Distance Calculators."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_distance_calculators.py",
        description="Question 24: Manhattan and Euclidean Distance Calculators (Informed Search Heuristics)",
    )

    parser.add_argument(
        "--p1",
        nargs=2,
        type=float,
        metavar=("X1", "Y1"),
        help="Coordinates of first point (e.g., --p1 1 1)",
    )
    parser.add_argument(
        "--p2",
        nargs=2,
        type=float,
        metavar=("X2", "Y2"),
        help="Coordinates of second point (e.g., --p2 4 5)",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        help="Multiple point pairs string (e.g., '(1,1),(4,5); (0,0),(3,4)')",
    )
    parser.add_argument(
        "--format",
        choices=["sample", "fixed", "table", "json", "all"],
        default="sample",
        help="Output display format ('sample' = exact syllabus output: 'Manhattan: 7, Euclidean: 5.00')",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Floating point decimal formatting precision (default 2)",
    )
    parser.add_argument(
        "--chebyshev",
        action="store_true",
        help="Include Chebyshev (L_infinity / Chessboard) diagonal distance (Optional Challenge)",
    )
    parser.add_argument(
        "--octile",
        action="store_true",
        help="Include Octile distance (8-directional with sqrt(2) diagonal cost)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Render 2D ASCII grid showing Manhattan taxicab path and Euclidean direct line",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive multi-pair benchmark suite across canonical coordinate pairs",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive prompt to enter custom point pairs",
    )
    parser.add_argument(
        "--axioms",
        action="store_true",
        help="Verify the 4 metric space axioms (Non-negativity, Identity, Symmetry, Triangle Inequality)",
    )
    parser.add_argument(
        "--topology",
        choices=["orthogonal", "octile", "chebyshev", "continuous"],
        help="Analyze heuristic admissibility and dominance under chosen grid movement topology",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib visualization figure",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        metavar="PATH",
        help="Save generated matplotlib plot to specified file path",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theoretical and practical study guide",
    )

    args = parser.parse_args(argv)

    # 1. Viva Voce Mode
    if args.viva:
        display_viva_reference()
        return

    # 2. Metric Space Axioms Mode
    if args.axioms:
        display_axiom_verification()
        return

    # 3. Movement Topology Analysis Mode
    if args.topology:
        display_topology_analysis(args.topology)
        return

    # 4. Resolve Point Pairs
    pairs: List[Tuple[PointLike, PointLike]] = []

    if args.interactive:
        pairs = run_interactive_mode()
    elif args.p1 is not None and args.p2 is not None:
        pairs = [((args.p1[0], args.p1[1]), (args.p2[0], args.p2[1]))]
    elif args.pairs:
        pairs = parse_point_pairs(args.pairs)
    elif args.benchmark:
        pairs = DEFAULT_BENCHMARK_PAIRS
    else:
        # Default single pair: canonical sample input (1, 1), (4, 5)
        pairs = [((1.0, 1.0), (4.0, 5.0))]

    results = compare_point_pairs(pairs)

    # 5. Output Results according to requested format
    if args.format == "sample":
        # Exact syllabus sample format
        for idx, res in enumerate(results):
            if len(results) > 1:
                print(f"Pair {idx+1} ({res.p1} -> {res.p2}): {res.formatted(precision=args.precision, include_chebyshev=args.chebyshev)}")
            else:
                print(res.formatted(precision=args.precision, include_chebyshev=args.chebyshev))
    elif args.format == "fixed":
        for res in results:
            print(res.formatted_fixed(precision=args.precision))
    elif args.format == "json":
        import json
        print(json.dumps([r.as_dict() for r in results], indent=2))
    elif args.format in ("table", "all"):
        table = format_comparison_table(
            results,
            include_chebyshev=args.chebyshev or True,
            include_octile=args.octile or False,
            precision=args.precision,
        )
        print(table)

    # If benchmark mode, print summary table by default
    if args.benchmark and args.format == "sample":
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON TABLE (All Metrics)")
        print("=" * 80)
        print(format_comparison_table(results, include_chebyshev=True, include_octile=True, precision=args.precision))

    # 6. ASCII Grid Visualization
    if args.ascii:
        print("\n" + "=" * 80)
        print("2D ASCII TRAJECTORY VISUALIZATION")
        print("=" * 80)
        for res in results[:3]:  # Render up to first 3 pairs
            print(render_ascii_distance_grid(res.p1, res.p2))
            print()

    # 7. Matplotlib Plot
    if args.plot or args.save_plot:
        plot_distance_metrics(
            pairs,
            save_path=args.save_plot,
            show=args.plot and not args.save_plot,
            title=f"Distance Metrics Comparison ({len(pairs)} pairs)",
        )
        if args.save_plot:
            print(f"\n[Visualizer] Plot saved successfully to: {args.save_plot}")


if __name__ == "__main__":
    main()
