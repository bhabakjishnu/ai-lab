"""Runner script for Breadth-First Search (BFS) on User-Defined Graph (Question 11).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (cyclic graphs, trees, disconnected graphs), step-by-step trace tables,
level and depth calculation (Optional Challenge), shortest-path reconstruction,
NetworkX/Matplotlib visual generation, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 11 default):
    python scripts/run_breadth_first_search.py

    # 2. Custom Graph and Start Node via CLI:
    python scripts/run_breadth_first_search.py --graph "{'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}" --start A

    # 3. Compact Syntax Graph:
    python scripts/run_breadth_first_search.py --graph "A: B, C; B: D; C:; D:" --start A

    # 4. Interactive Console Mode:
    python scripts/run_breadth_first_search.py --interactive

    # 5. Run All Standard Benchmarks (Sample, Cyclic, Binary Tree, Disconnected):
    python scripts/run_breadth_first_search.py --demo

    # 6. Run Optional Challenge Demonstration (Levels and Depth Metrics):
    python scripts/run_breadth_first_search.py --challenge

    # 7. Generate and Save Graphical Visualization:
    python scripts/run_breadth_first_search.py --save reports/bfs_traversal.png

    # 8. Display Viva Voce Study Guide & Complexity Derivations:
    python scripts/run_breadth_first_search.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.breadth_first_search import (
    BINARY_TREE_GRAPH,
    CYCLIC_BENCHMARK_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DISCONNECTED_GRAPH,
    BFSResult,
    BreadthFirstSearch,
    breadth_first_search,
    get_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_bfs,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header."""
    line = "=" * 78
    print(f"\n{line}")
    print(f" {title.upper()}")
    print(f"{line}")


def display_result(result: BFSResult, show_table: bool = True, show_tree: bool = True) -> None:
    """Display comprehensive BFS traversal findings, metrics, and diagnostics.

    Args:
        result: Completed BFSResult instance.
        show_table: Whether to display the step-by-step FIFO execution trace table.
        show_tree: Whether to print the ASCII hierarchical spanning tree.
    """
    print_banner(f"BFS TRAVERSAL EXECUTION (Start Node: '{result.start_node}')")
    print(f"Adjacency List : {result.graph}")
    print(f"Start Node     : '{result.start_node}'")
    print(f"BFS Traversal  : {result.format_traversal()}")
    print(f"Total Explored : {len(result.traversal_order)} node(s)")
    print(f"Total Levels   : {result.total_levels} level(s)")
    print(f"Maximum Depth  : {result.max_depth}")
    print(f"Execution Time : {result.execution_time_ms:.4f} ms")

    if result.unreachable_nodes:
        print(f"Unreachable    : {', '.join(result.unreachable_nodes)} (Disconnected from start)")

    print("\n[Optional Challenge: Level-by-Level Breakdown]")
    print(result.format_levels_summary())

    if show_tree:
        print("\n[BFS Spanning Tree (Shortest Hop Hierarchy)]")
        print(result.format_ascii_tree())

    if show_table:
        print("\n[Chronological Step-by-Step FIFO Frontier Trace]")
        print(result.format_step_table())


def run_benchmarks() -> None:
    """Run an automated battery of standard AI benchmark graphs."""
    print_banner("DEMO: BREADTH-FIRST SEARCH ON BENCHMARK GRAPHS")

    benchmarks = [
        ("Lab Question 11 Sample DAG", DEFAULT_SAMPLE_GRAPH, "A"),
        ("Cyclic Directed Graph (Cycle Prevention)", CYCLIC_BENCHMARK_GRAPH, "A"),
        ("Balanced Binary Tree (Complete Layer Traversal)", BINARY_TREE_GRAPH, "A"),
        ("Disconnected Graph (Reachability Detection)", DISCONNECTED_GRAPH, "A"),
    ]

    for name, g, start in benchmarks:
        print(f"\n>>> Running Benchmark: {name}")
        res = breadth_first_search(g, start)
        print(f"    Graph         : {g}")
        print(f"    Start Node    : {start}")
        print(f"    BFS Traversal : {res.format_traversal()}")
        print(f"    Total Levels  : {res.total_levels} | Max Depth: {res.max_depth}")
        if res.unreachable_nodes:
            print(f"    Unreachable   : {res.unreachable_nodes}")


def run_challenge_demonstration() -> None:
    """Execute focused demonstration of the Optional Challenge: depth and levels."""
    print_banner("OPTIONAL CHALLENGE: GRAPH LEVELS & TREE DEPTH COMPUTATION")
    print(
        "BFS explores nodes concentric ring by concentric ring. In an unweighted graph,\n"
        "the level assigned to node v equals the exact shortest path distance d(source, v).\n"
    )

    test_graphs = [
        ("Sample DAG", DEFAULT_SAMPLE_GRAPH, "A"),
        ("Deep Binary Tree", BINARY_TREE_GRAPH, "A"),
        ("Cyclic Network", CYCLIC_BENCHMARK_GRAPH, "A"),
    ]

    for name, g, start in test_graphs:
        res = breadth_first_search(g, start)
        print(f"\n--- {name} (Start: '{start}') ---")
        print(f"Traversal Sequence: {res.format_traversal()}")
        print(f"Levels Count      : {res.total_levels}")
        print(f"Max Depth         : {res.max_depth}")
        for lvl, nodes in res.nodes_by_level.items():
            print(f"  • Level {lvl} (hops = {lvl}): {nodes}")


def run_viva_notes() -> None:
    """Print authoritative viva voce study guide and theoretical derivations."""
    viva = get_viva_reference()
    print_banner(viva["title"])
    print(f"Syllabus Area: {viva['syllabus_area']}\n")
    print("Core Theoretical Topics:")
    for topic in viva["core_topics"]:
        print(f"  • {topic}")
    print("\n" + "=" * 78)
    print("VIVA VOCE QUESTIONS & COMPREHENSIVE ANSWERS")
    print("=" * 78)

    for idx, item in enumerate(viva["qna"], 1):
        print(f"\nQ{idx}: {item['question']}")
        print(f"A{idx}: {item['answer']}")


def prompt_interactive_graph() -> Tuple[Dict[str, List[str]], str]:
    """Interactively prompt user for graph definition and start node.

    Returns:
        Tuple of (normalized graph dictionary, start node string).
    """
    print_banner("INTERACTIVE BFS GRAPH INPUT")
    print("Select an input format:")
    print("  [1] Enter Python dictionary (e.g., {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []})")
    print("  [2] Enter compact adjacency list (e.g., A: B, C; B: D; C:; D:)")
    print("  [3] Enter edge list (e.g., A->B, A->C, B->D)")
    print("  [4] Use Default Sample Graph")

    choice = input("\nEnter choice [1-4] (default: 4): ").strip()
    if choice == "1":
        raw = input("Enter graph dict: ").strip()
        graph = parse_graph_input(raw)
    elif choice == "2":
        raw = input("Enter compact adjacency: ").strip()
        graph = parse_graph_input(raw)
    elif choice == "3":
        raw = input("Enter edge list: ").strip()
        graph = parse_graph_input(raw)
    else:
        print("[Using Default Sample Graph]")
        graph = DEFAULT_SAMPLE_GRAPH

    available = ", ".join(sorted(graph.keys()))
    default_start = sorted(graph.keys())[0] if graph else "A"
    start_in = input(f"Enter start node (available: [{available}], default: '{default_start}'): ").strip()
    start_node = start_in if start_in else default_start

    return graph, start_node


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the BFS runner."""
    parser = argparse.ArgumentParser(
        description="Breadth-First Search (BFS) on User-Defined Graph (Question 11)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="User-defined graph as Python dict string or compact string (e.g., \"{'A':['B','C']}\")",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="A",
        help="Start node for BFS traversal (default: 'A')",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["sample", "cyclic", "binary-tree", "disconnected"],
        default=None,
        help="Run a predefined textbook benchmark graph",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive terminal wizard to define graph and start node",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive automated demonstrations across all benchmark graphs",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Demonstrate the Optional Challenge: depth levels and layer breakdown",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display complete Viva Voce study guide and complexity derivations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save graphical visualization of BFS tree to given image path (e.g., reports/bfs.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive Matplotlib plot window",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable alphabetical sorting of tied successor nodes",
    )
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for the Breadth-First Search CLI runner.

    Args:
        args: Command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    parsed = parse_arguments(args)

    if parsed.viva:
        run_viva_notes()
        return 0

    if parsed.demo:
        run_benchmarks()
        return 0

    if parsed.challenge:
        run_challenge_demonstration()
        return 0

    # Determine graph and start node
    if parsed.interactive:
        graph, start_node = prompt_interactive_graph()
    elif parsed.benchmark == "cyclic":
        graph, start_node = CYCLIC_BENCHMARK_GRAPH, parsed.start if parsed.start in CYCLIC_BENCHMARK_GRAPH else "A"
    elif parsed.benchmark == "binary-tree":
        graph, start_node = BINARY_TREE_GRAPH, parsed.start if parsed.start in BINARY_TREE_GRAPH else "A"
    elif parsed.benchmark == "disconnected":
        graph, start_node = DISCONNECTED_GRAPH, parsed.start if parsed.start in DISCONNECTED_GRAPH else "A"
    elif parsed.graph:
        graph = parse_graph_input(parsed.graph)
        start_node = parsed.start
    else:
        graph = DEFAULT_SAMPLE_GRAPH
        start_node = parsed.start

    sort_neighbors = not parsed.no_sort
    result = breadth_first_search(graph, start_node, sort_neighbors=sort_neighbors)
    display_result(result)

    if parsed.save or parsed.show:
        visualize_bfs(result, save_path=parsed.save, show=parsed.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
