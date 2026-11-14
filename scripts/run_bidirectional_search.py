"""Runner script for Bidirectional Search Simulation (Question 19).

Demonstrates simultaneous bidirectional Breadth-First Search (BFS) on unweighted graphs,
maintaining dual queues and visited sets, alternating expansion turns, detecting frontier
collisions, and reconstructing the combined shortest path by concatenating the forward
and backward path segments at the intersection node.

Usage Examples:
    # 1. Default Sample Run (Start: 'A', Goal: 'G' -> Intersection at D. Path: A-B-D-F-G):
    python scripts/run_bidirectional_search.py

    # 2. Exact Syllabus Single-Line Output Format:
    python scripts/run_bidirectional_search.py --format single

    # 3. Custom Start and Goal Nodes:
    python scripts/run_bidirectional_search.py --start A --goal F

    # 4. Custom Graph Specification:
    python scripts/run_bidirectional_search.py --graph "A: B, C; B: A, D; C: A, D; D: B, C, G; G: D" --start A --goal G

    # 5. Run Optional Challenge (2D Grid Maze Bidirectional Search):
    python scripts/run_bidirectional_search.py --challenge

    # 6. Run Comprehensive Automated Benchmark Suite:
    python scripts/run_bidirectional_search.py --demo

    # 7. Display Viva Voce Study Guide & Complexity Proofs:
    python scripts/run_bidirectional_search.py --viva

    # 8. Generate and Save Graphical Visualization:
    python scripts/run_bidirectional_search.py --save reports/bidirectional_search.png
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bidirectional_search import (
    DEFAULT_BDS_MAZE,
    DIRECTED_REVERSIBLE_BDS_GRAPH,
    DISCONNECTED_BDS_GRAPH,
    EXPONENTIAL_TREE_BDS_GRAPH,
    LINEAR_CHAIN_BDS_GRAPH,
    SAMPLE_BDS_GRAPH,
    BidirectionalSearch,
    BidirectionalSearchResult,
    GridMazeBidirectionalResult,
    bidirectional_search,
    compare_unidirectional_vs_bidirectional,
    get_bidirectional_search_viva_reference,
    parse_graph_input,
    plot_bidirectional_search,
    plot_grid_maze_bidirectional,
    solve_grid_maze_bidirectional,
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
    viva = get_bidirectional_search_viva_reference()
    print_banner("QUESTION NO. 19: BIDIRECTIONAL SEARCH VIVA VOCE THEORY GUIDE")

    topics = [
        ("COMPLEXITY ADVANTAGE: O(b^(d/2)) vs O(b^d)", viva["complexity_advantage"]),
        ("DIFFICULTIES OF BACKWARD SEARCH IN DIRECTED GRAPHS", viva["backward_search_difficulties"]),
        ("PREDECESSOR vs SUCCESSOR GENERATION", viva["predecessor_vs_successor"]),
    ]

    for i, (title, content) in enumerate(topics, start=1):
        print(f"[{i}] {title}")
        print(f"Question: {content['question']}")
        print("Answer:")
        for line in content["answer"].split(". "):
            cleaned = line.strip()
            if cleaned:
                if not cleaned.endswith("."):
                    cleaned += "."
                print(f"  • {cleaned}")
        print("-" * 75)


def run_benchmarks() -> None:
    """Run comprehensive automated benchmarks comparing BDS across multiple topologies."""
    print_banner("BIDIRECTIONAL SEARCH (BDS) BENCHMARK SUITE")

    benchmarks = [
        ("Benchmark 1: Canonical Sample Graph (Syllabus Default)", SAMPLE_BDS_GRAPH, "A", "G", False),
        ("Benchmark 2: Linear Chain Graph (Odd vs Even Splicing)", LINEAR_CHAIN_BDS_GRAPH, "A", "F", False),
        ("Benchmark 3: Directed Reversible Graph (Predecessor Inversion)", DIRECTED_REVERSIBLE_BDS_GRAPH, "S", "G", True),
        ("Benchmark 4: Disconnected Graph (Unreachable Goal Detection)", DISCONNECTED_BDS_GRAPH, "A", "Z", False),
    ]

    for title, graph, start, goal, directed in benchmarks:
        print(f"\n--- {title} ---")
        res = bidirectional_search(graph=graph, start=start, goal=goal, is_directed=directed)
        print(f"  • Target Query      : Start '{start}' -> Goal '{goal}' (Directed={directed})")
        print(f"  • Result Output     : {res.format_sample_output()}")
        print(f"  • Status            : {res.status}")
        print(f"  • Total Expansions  : {res.total_nodes_expanded} (Fwd: {res.nodes_expanded_forward}, Bwd: {res.nodes_expanded_backward})")
        print(f"  • Execution Time    : {res.execution_time_ms:.4f} ms")

    # Benchmark 5: O(b^(d/2)) vs O(b^d) Comparison on Exponential Tree
    print("\n--- Benchmark 5: Search Space Halving: Unidirectional BFS vs Bidirectional BFS ---")
    comp = compare_unidirectional_vs_bidirectional(EXPONENTIAL_TREE_BDS_GRAPH, "root", "goal")
    print(f"  • Topology          : Complete Branching Tree (depth d=6)")
    print(f"  • Unidirectional BFS: {comp['unidirectional_expansions']} node expansions (Path Length: {comp['unidirectional_path_length']})")
    print(f"  • Bidirectional BFS : {comp['bidirectional_expansions']} node expansions (Path Length: {comp['bidirectional_path_length']})")
    print(f"  • Expansion Savings : {comp['expansion_reduction_percent']:.1f}% reduction in explored states!")
    print(f"  • Intersection Node : '{comp['intersection_node']}'")
    print(f"  • Optimal Path Match: {comp['optimal_path_match']}")


def run_grid_maze_challenge(save_path: Optional[str] = None) -> None:
    """Execute Optional Challenge: 2D Grid Maze Bidirectional Search."""
    print_banner("OPTIONAL CHALLENGE: 2D GRID MAZE BIDIRECTIONAL SEARCH")
    start = (0, 0)
    goal = (6, 6)
    print(f"Searching 7x7 Grid Maze from Start {start} to Goal {goal}...\n")

    res = solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, start=start, goal=goal)

    print(f"Search Status          : {res.status.upper()}")
    print(f"Intersection Coordinate: {res.intersection_coord}")
    print(f"Shortest Path Length   : {res.path_length} steps")
    print(f"Total Cells Expanded   : {res.nodes_expanded}")
    print(f"Forward Visited Cells  : {len(res.visited_forward)}")
    print(f"Backward Visited Cells : {len(res.visited_backward)}")
    print(f"Execution Time         : {res.execution_time_ms:.4f} ms\n")

    print("MAZE VISUALIZATION:")
    print(res.render_ascii_maze())

    if save_path:
        print(f"\nGenerating and saving maze visualization plot to '{save_path}'...")
        plot_grid_maze_bidirectional(res, save_path=save_path)
        print("Plot successfully saved.")


def main() -> None:
    """CLI dispatcher for Bidirectional Search simulation."""
    parser = argparse.ArgumentParser(
        description="Question 19: Bidirectional Search Simulation on Unweighted Graphs and Grid Mazes."
    )
    parser.add_argument("--start", type=str, default="A", help="Start node label (default: 'A').")
    parser.add_argument("--goal", type=str, default="G", help="Goal node label (default: 'G').")
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Custom graph adjacency specification (JSON or 'A: B, C; B: D').",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat graph as directed (inverts edges for backward search).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use dynamic balanced frontier expansion instead of strict alternating turns.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["full", "single"],
        default="full",
        help="Output format: 'full' (verbose summary & table) or 'single' (exact sample line).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark suite across multiple graph topologies.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge: 2D Grid Maze Bidirectional Search.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce questions and answers.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="File path to save Matplotlib visualization diagram (e.g. reports/bds.png).",
    )

    args = parser.parse_args()

    # Mode 1: Viva Guide
    if args.viva:
        print_viva_guide()
        return

    # Mode 2: Optional Challenge Grid Maze
    if args.challenge:
        run_grid_maze_challenge(save_path=args.save)
        return

    # Mode 3: Demo Benchmark Suite
    if args.demo:
        run_benchmarks()
        return

    # Mode 4: Graph Execution (Default or Custom)
    if args.graph is not None:
        graph = parse_graph_input(args.graph)
    else:
        graph = SAMPLE_BDS_GRAPH

    strategy = "balanced" if args.balanced else "turn"
    result = bidirectional_search(
        graph=graph,
        start=args.start,
        goal=args.goal,
        is_directed=args.directed,
        alternation_strategy=strategy,
    )

    # Format output
    if args.format == "single":
        print(result.format_sample_output())
        return

    print_banner("QUESTION NO. 19: BIDIRECTIONAL SEARCH SIMULATION")
    print("SYLLABUS SAMPLE OUTPUT:")
    print(f"  >>> {result.format_sample_output()}\n")

    print(result.format_summary())
    print("\nSTEP-BY-STEP ALTERNATING EXPANSION TRACE TABLE:")
    print(result.format_trace_table())

    if args.save:
        print(f"\nGenerating and saving graphical plot to '{args.save}'...")
        plot_bidirectional_search(result, save_path=args.save)
        print("Plot successfully saved.")


if __name__ == "__main__":
    main()
