"""Command-Line Interface and Runner for Greedy Best-First Search (Question 26).

Implements Greedy Best-First Search on a graph with heuristic values h(n),
mirroring university examination standards and Russell & Norvig's AIMA specifications.

Usage Examples:
    # 1. Default Canonical Sample Run:
    python scripts/run_greedy_best_first_search.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_greedy_best_first_search.py --format sample

    # 3. Custom Graph & Heuristics via CLI:
    python scripts/run_greedy_best_first_search.py --graph "A: B; B: C; C:" --heuristics "A:5, B:2, C:0" --start A --goal C

    # 4. Optional Challenge (Suboptimality Counterexample vs. Optimal UCS):
    python scripts/run_greedy_best_first_search.py --challenge

    # 5. AIMA Romania Map Benchmark (Arad to Bucharest):
    python scripts/run_greedy_best_first_search.py --romania

    # 6. Local Minima Trap Demonstration:
    python scripts/run_greedy_best_first_search.py --local-minima

    # 7. Detailed Step-by-Step Priority Queue Table:
    python scripts/run_greedy_best_first_search.py --table

    # 8. University Viva Voce Preparation Guide:
    python scripts/run_greedy_best_first_search.py --viva

    # 9. Generate Graph Diagram (PNG):
    python scripts/run_greedy_best_first_search.py --plot gbfs_graph.png
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

from src.greedy_best_first_search import (
    CYCLIC_GRAPH,
    CYCLIC_HEURISTICS,
    DEFAULT_SAMPLE_GRAPH,
    DEFAULT_SAMPLE_HEURISTICS,
    DISCONNECTED_GRAPH,
    DISCONNECTED_HEURISTICS,
    LOCAL_MINIMUM_GRAPH,
    LOCAL_MINIMUM_HEURISTICS,
    ROMANIA_MAP_GRAPH,
    ROMANIA_SLD_HEURISTICS,
    SUBOPTIMAL_CHALLENGE_GRAPH,
    SUBOPTIMAL_CHALLENGE_HEURISTICS,
    GBFSResult,
    compare_greedy_vs_optimal,
    demonstrate_local_minimum,
    get_gbfs_viva_reference,
    greedy_best_first_search,
    parse_graph,
    parse_heuristics,
    visualize_gbfs,
)


def run_sample_output(quiet: bool = False) -> str:
    """Run the canonical syllabus sample input and return exact expected output string.

    Input:
        Graph: {'A': ['B'], 'B': ['C'], 'C': []}
        Heuristics: {'A': 5, 'B': 2, 'C': 0}
        Start: 'A', Goal: 'C'

    Expected Output:
        Path chosen based on minimum h(n): A -> B -> C
    """
    res = greedy_best_first_search(
        DEFAULT_SAMPLE_GRAPH,
        DEFAULT_SAMPLE_HEURISTICS,
        start="A",
        goal="C",
    )
    formatted = res.format_path()
    if not quiet:
        print(formatted)
    return formatted


def display_viva_reference() -> None:
    """Print the complete university Viva Voce study guide."""
    viva = get_gbfs_viva_reference()
    print("\n" + "=" * 75)
    print(f"🎓 {viva['title']}")
    print("=" * 75)
    print(f"Core Concept        : {viva['core_concept']}")
    print(f"Evaluation Function : {viva['evaluation_function']}")
    print("-" * 75)
    print("KEY VIVA QUESTIONS & DETAILED MODEL ANSWERS:")
    for idx, item in enumerate(viva["key_questions"], 1):
        print(f"\n[{idx}] Question: {item['question']}")
        print(f"    Answer  : {item['answer']}")
    print("\n" + "-" * 75)
    print("ALGORITHM COMPARISON MATRIX:")
    header = f"{'Algorithm':<32} | {'Evaluation Fn':<16} | {'Optimal?':<9} | {'Complete?'}"
    print(header)
    print("-" * len(header))
    for row in viva["comparison_matrix"]:
        print(
            f"{row['algorithm']:<32} | {row['evaluation_fn']:<16} | {row['optimal']:<9} | {row['complete']}"
        )
    print("=" * 75 + "\n")


def display_optional_challenge() -> None:
    """Execute and display the Optional Challenge (Greedy Suboptimality Counterexample)."""
    print("\n" + "=" * 75)
    print("OPTIONAL CHALLENGE: Greedy Best-First Search Suboptimality Proof")
    print("=" * 75)
    print("Problem: Construct a graph where Greedy fails to find the optimal path.")
    print("Graph Topology:")
    print("  - Start 'S' connects to 'A' (h=2, cost=10) and 'B' (h=8, cost=1)")
    print("  - 'A' connects to 'G' (cost=100) -> Total Path S->A->G Cost = 110")
    print("  - 'B' connects to 'G' (cost=1)   -> Total Path S->B->G Cost = 2")
    print("-" * 75)

    comparison = compare_greedy_vs_optimal(
        SUBOPTIMAL_CHALLENGE_GRAPH,
        SUBOPTIMAL_CHALLENGE_HEURISTICS,
        start="S",
        goal="G",
    )
    print(comparison.format_report())


def display_romania_benchmark() -> None:
    """Execute and display the classical AIMA Romania Map benchmark (Arad to Bucharest)."""
    print("\n" + "=" * 75)
    print("BENCHMARK: Russell & Norvig Romania Map (Arad to Bucharest)")
    print("=" * 75)
    print("Heuristics: Straight-Line Distance to Bucharest (h_SLD)")
    print("-" * 75)

    comparison = compare_greedy_vs_optimal(
        ROMANIA_MAP_GRAPH,
        ROMANIA_SLD_HEURISTICS,
        start="Arad",
        goal="Bucharest",
    )
    print(comparison.format_report())


def display_local_minima_demo() -> None:
    """Execute and display the local minima trap demonstration."""
    print("\n" + "=" * 75)
    print("DEMO: Getting Stuck in Local Minima and Deceptive Heuristics")
    print("=" * 75)
    demo_info = demonstrate_local_minimum()
    res: GBFSResult = demo_info["result"]
    print(res.format_summary())
    print("\nStep-by-step priority queue log:")
    print(res.format_table())
    print("\n" + demo_info["explanation"])
    print("=" * 75 + "\n")


def run_interactive_mode() -> None:
    """Prompt the user interactively to configure graph, heuristics, start, and goal."""
    print("\n" + "=" * 70)
    print("Interactive Mode: Greedy Best-First Search")
    print("=" * 70)
    print("Example Graph: A: B; B: C; C:")
    graph_input = input("Enter Graph [press Enter for default]: ").strip()
    graph = parse_graph(graph_input) if graph_input else DEFAULT_SAMPLE_GRAPH

    print("Example Heuristics: A: 5, B: 2, C: 0")
    h_input = input("Enter Heuristics [press Enter for default]: ").strip()
    heuristics = parse_heuristics(h_input) if h_input else DEFAULT_SAMPLE_HEURISTICS

    start_node = input("Enter Start Node [default 'A']: ").strip() or "A"
    goal_node = input("Enter Goal Node [default 'C']: ").strip() or "C"

    res = greedy_best_first_search(graph, heuristics, start=start_node, goal=goal_node)
    print("\n" + res.format_summary())
    print("\n" + res.format_table())


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point for Greedy Best-First Search runner."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_greedy_best_first_search.py",
        description="Greedy Best-First Search (GBFS) Implementation and Analysis (Question 26)",
    )
    parser.add_argument(
        "--format",
        choices=["sample", "full", "table", "json"],
        default="full",
        help="Output presentation format ('sample' produces exact syllabus string)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Graph specification as JSON or text (e.g. 'A: B; B: C; C:')",
    )
    parser.add_argument(
        "--heuristics",
        type=str,
        default=None,
        help="Heuristics mapping as JSON or text (e.g. 'A:5, B:2, C:0')",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="A",
        help="Start node identifier (default: 'A')",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="C",
        help="Goal node identifier (default: 'C')",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=["fifo", "alphabetical", "lifo"],
        default="fifo",
        help="Frontier tie-breaking policy when h(u) == h(v)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Shorthand for --format sample (exact syllabus output)",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge: counterexample where Greedy fails optimality",
    )
    parser.add_argument(
        "--romania",
        action="store_true",
        help="Run classic AIMA Romania Map benchmark (Arad to Bucharest)",
    )
    parser.add_argument(
        "--local-minima",
        action="store_true",
        help="Demonstrate the local minimum dead-end trap phenomenon",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce questions, answers, and matrix",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display step-by-step priority queue state trace",
    )
    parser.add_argument(
        "--plot",
        type=str,
        nargs="?",
        const="gbfs_traversal.png",
        default=None,
        help="Save graphical network visualization to specified PNG path",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prompt mode",
    )

    args = parser.parse_args(argv)

    # Route specialized flags first
    if args.sample or args.format == "sample":
        run_sample_output(quiet=False)
        return 0

    if args.viva:
        display_viva_reference()
        return 0

    if args.challenge:
        display_optional_challenge()
        return 0

    if args.romania:
        display_romania_benchmark()
        return 0

    if args.local_minima:
        display_local_minima_demo()
        return 0

    if args.interactive:
        run_interactive_mode()
        return 0

    # Resolve graph and heuristics
    if args.graph:
        graph = parse_graph(args.graph)
    else:
        graph = DEFAULT_SAMPLE_GRAPH

    if args.heuristics:
        heuristics = parse_heuristics(args.heuristics)
    else:
        heuristics = DEFAULT_SAMPLE_HEURISTICS

    res = greedy_best_first_search(
        graph,
        heuristics,
        start=args.start,
        goal=args.goal,
        tie_breaker=args.tie_breaker,
    )

    if args.format == "json":
        output_data = {
            "start": res.start_node,
            "goal": res.goal_node,
            "goal_found": res.goal_found,
            "path": res.path,
            "path_cost": res.path_cost,
            "nodes_expanded": res.nodes_expanded,
            "execution_time_ms": res.execution_time_ms,
        }
        print(json.dumps(output_data, indent=2))
    elif args.format == "table" or args.table:
        print(res.format_summary())
        print("\n" + res.format_table())
    else:
        print(res.format_summary())
        print(f"\n{res.format_path()}")

    if args.plot:
        img_path = visualize_gbfs(graph, heuristics, result=res, output_path=args.plot)
        if img_path:
            print(f"\nDiagram successfully saved to: {img_path}")
        else:
            print("\nVisualization skipped (networkx or matplotlib not available).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
