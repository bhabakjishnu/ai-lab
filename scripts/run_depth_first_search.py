"""Runner script for Depth-First Search (DFS) for Graph Traversal (Question 12).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (sample graph, cyclic graphs, binary trees, disconnected graphs),
step-by-step trace tables, cycle detection, edge classification, Optional Challenge
comparison between iterative and recursive DFS implementations, NetworkX/Matplotlib
visual generation, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 12 default: Graph {'1':['2','3'], ...}, Start: '1'):
    python scripts/run_depth_first_search.py

    # 2. Custom Graph and Start Node via CLI:
    python scripts/run_depth_first_search.py --graph "{'1': ['2', '3'], '2': ['4'], '3': [], '4': []}" --start 1

    # 3. Compact Syntax Graph:
    python scripts/run_depth_first_search.py --graph "1: 2, 3; 2: 4; 3:; 4:" --start 1

    # 4. Interactive Console Mode:
    python scripts/run_depth_first_search.py --interactive

    # 5. Run Optional Challenge Demonstration (Iterative vs. Recursive comparison):
    python scripts/run_depth_first_search.py --challenge

    # 6. Run All Standard Benchmarks (Sample, Cyclic, Binary Tree, Disconnected, DAG):
    python scripts/run_depth_first_search.py --demo

    # 7. Generate and Save Graphical Visualization:
    python scripts/run_depth_first_search.py --save reports/dfs_traversal.png

    # 8. Display Viva Voce Study Guide & Comparison Matrix:
    python scripts/run_depth_first_search.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.depth_first_search import (
    BINARY_TREE_GRAPH,
    CYCLIC_BENCHMARK_GRAPH,
    DAG_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DISCONNECTED_GRAPH,
    DFSResult,
    DepthFirstSearch,
    compare_dfs_modes,
    depth_first_search,
    dfs_iterative,
    dfs_recursive,
    get_dfs_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_dfs,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header.

    Args:
        title: Header text to display.
    """
    width = 75
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_viva_guide() -> None:
    """Print the complete Viva Voce examination reference guide."""
    viva = get_dfs_viva_reference()
    print_banner("DEPTH-FIRST SEARCH (DFS) VIVA VOCE REFERENCE & THEORY GUIDE")
    print(f"Syllabus Area: {viva['syllabus_area']}\n")

    for i, topic in enumerate(viva["core_topics"], start=1):
        print(f"[{i}] {topic['topic'].upper()}")
        print(f"Question: {topic['question']}")
        print("Answer:")
        for line in topic["answer"].split("\n"):
            print(f"  {line}")
        print("-" * 75)

    matrix = viva["comparison_matrix"]
    headers = matrix["headers"]
    print(f"\n{headers[0]:<30} | {headers[1]:<32} | {headers[2]}")
    print("=" * 95)
    for row in matrix["rows"]:
        print(f"{row[0]:<30} | {row[1]:<32} | {row[2]}")
    print("=" * 95)


def display_dfs_run(
    result: DFSResult,
    title: Optional[str] = None,
    show_table: bool = True,
    show_edges: bool = True,
) -> None:
    """Format and display detailed results of a single DFS search run.

    Args:
        result: The DFSResult object to inspect.
        title: Optional custom section header.
        show_table: Whether to print the step-by-step expansion table.
        show_edges: Whether to print the CLRS edge classification table.
    """
    if title:
        print_banner(title)

    print(f"Mode                     : {result.mode.upper()}")
    print(f"Start Node               : {result.start_node}")
    print(f"DFS Traversal            : {result.format_traversal()}")
    print(f"Neighbor Order Preserved : {result.order_preserving}")
    print(f"Nodes Explored           : {len(result.traversal_order)} of {len(result.graph)}")
    print(f"Maximum Tree Depth       : {result.max_depth}")
    print(f"Execution Wall Time      : {result.execution_time_ms:.4f} ms")

    if result.has_cycle():
        cycle_str = ", ".join([f"{u}->{v}" for u, v in result.cycles_detected])
        print(f"Cycles Detected          : YES ({cycle_str})")
    else:
        print("Cycles Detected          : NONE (Acyclic Subgraph)")

    if result.unreachable_nodes:
        print(f"Unreachable Nodes        : {', '.join(result.unreachable_nodes)}")
    else:
        print("Unreachable Nodes        : None (All graph nodes reachable)")

    if show_table and result.steps:
        print("\nStep-by-Step Execution Trace Table:")
        print(result.format_step_table())

    if show_edges and result.edge_classification:
        print("\nEdge Classification (CLRS Formulation):")
        print(result.format_edge_classification())

    print("\nDFS Spanning Tree Branch Paths from Start:")
    for node in result.traversal_order:
        path = result.path_to(node)
        if path:
            path_str = " -> ".join(path)
            print(f"  To Node {node:<4} : {path_str} (Depth {result.node_depths.get(node, 0)})")


def run_optional_challenge(graph: Dict[str, List[str]], start_node: str) -> None:
    """Execute the Optional Challenge: compare iterative and recursive implementations.

    Args:
        graph: Graph dictionary.
        start_node: Starting node label.
    """
    print_banner("OPTIONAL CHALLENGE: ITERATIVE VS. RECURSIVE DFS COMPARISON")
    comp = compare_dfs_modes(graph, start_node)

    print(f"Graph Adjacency List : {comp['graph']}")
    print(f"Starting Node        : {comp['start_node']}\n")

    print(f"{'Method / Variant':<38} | {'Traversal Order':<24} | {'Step Count':<10} | {'Time (ms)'}")
    print("-" * 88)
    print(
        f"{'1. Iterative (Standard Forward-Push)':<38} | "
        f"{comp['iterative_standard']['formatted']:<24} | "
        f"{comp['iterative_standard']['steps_count']:<10} | "
        f"{comp['iterative_standard']['time_ms']:.4f}"
    )
    print(
        f"{'2. Recursive (Standard Forward-Call)':<38} | "
        f"{comp['recursive_standard']['formatted']:<24} | "
        f"{comp['recursive_standard']['steps_count']:<10} | "
        f"{comp['recursive_standard']['time_ms']:.4f}"
    )
    print(
        f"{'3. Iterative (Order-Preserving Reverse-Push)':<38} | "
        f"{comp['iterative_order_preserving']['formatted']:<24} | "
        f"{comp['iterative_order_preserving']['steps_count']:<10} | "
        f"{comp['iterative_order_preserving']['time_ms']:.4f}"
    )
    print(
        f"{'4. Recursive (Reversed-Call)':<38} | "
        f"{comp['recursive_reversed']['formatted']:<24} | "
        f"{comp['recursive_reversed']['steps_count']:<10} | "
        f"{comp['recursive_reversed']['time_ms']:.4f}"
    )
    print("-" * 88)

    print("\nDetailed Comparison Insights:")
    print(comp["explanation"])
    print("\nTakeaway:")
    print("  - Iterative DFS with standard forward-push matches the Sample Output (1, 3, 2, 4) because")
    print("    pushing '2' then '3' places '3' on top of the LIFO stack.")
    print("  - Iterative DFS with reverse-push (reversed neighbors) matches standard recursive DFS (1, 2, 4, 3).")
    print("  - Both implementations visit the exact same set of reachable nodes in O(V + E) time.")


def run_benchmark_suite() -> None:
    """Execute all automated benchmark graphs to demonstrate various topological scenarios."""
    print_banner("DEPTH-FIRST SEARCH AUTOMATED BENCHMARK SUITE")

    benchmarks = [
        (
            "1. Default Sample Graph (Lab Question 12)",
            DEFAULT_SAMPLE_GRAPH,
            "1",
            "Unweighted graph matching lab problem statement.",
        ),
        (
            "2. Cyclic Benchmark Graph",
            CYCLIC_BENCHMARK_GRAPH,
            "1",
            "Graph with directed cycles (1->2->4->3->1) validating visited set and back-edge detection.",
        ),
        (
            "3. Binary Tree Graph",
            BINARY_TREE_GRAPH,
            "1",
            "Strict hierarchical binary tree demonstrating pure tree exploration and deep branch traversal.",
        ),
        (
            "4. Disconnected Graph",
            DISCONNECTED_GRAPH,
            "1",
            "Graph containing unreachable components (nodes 4 and 5).",
        ),
        (
            "5. Directed Acyclic Graph (DAG) with Convergent Paths",
            DAG_GRAPH,
            "1",
            "Graph with multi-parent convergence demonstrating forward and cross edges.",
        ),
    ]

    for title, graph, start, notes in benchmarks:
        print_banner(title)
        print(f"Topology Description: {notes}")
        print(f"Adjacency List      : {graph}")
        result = dfs_iterative(graph, start)
        display_dfs_run(result, show_table=True, show_edges=True)


def run_interactive_mode() -> None:
    """Run interactive command-line prompt for custom user graph entry."""
    print_banner("INTERACTIVE DEPTH-FIRST SEARCH EXPLORER")
    print("Supported Input Formats:")
    print("  - Python Dict literal : {'1': ['2', '3'], '2': ['4'], '3': [], '4': []}")
    print("  - Compact syntax      : 1: 2, 3; 2: 4; 3:; 4:")
    print("  - Edge list           : 1->2, 1->3, 2->4")
    print("  - Press Enter directly to load default sample graph.\n")

    user_graph_str = input("Enter Graph [or Enter for Default]: ").strip()
    if not user_graph_str:
        graph = DEFAULT_SAMPLE_GRAPH
        print(f"Using Default Sample Graph: {graph}")
    else:
        try:
            graph = parse_graph_input(user_graph_str)
            print(f"Successfully parsed graph: {graph}")
        except Exception as e:
            print(f"[Error] Failed to parse graph: {e}")
            return

    available_nodes = sorted(graph.keys())
    print(f"Available Nodes: {available_nodes}")
    default_start = available_nodes[0] if available_nodes else "1"
    start_str = input(f"Enter Start Node [Default: '{default_start}']: ").strip()
    if not start_str:
        start_str = default_start

    mode_choice = input("Select Mode ([i]terative, [r]ecursive, [c]hallenge comparison) [Default: i]: ").strip().lower()
    if mode_choice in ("c", "challenge", "comp"):
        run_optional_challenge(graph, start_str)
        return

    is_recursive = mode_choice.startswith("r")
    preserve_choice = input("Preserve Neighbor Ordering? (y/N) [Default: N, matching sample output]: ").strip().lower()
    order_preserving = preserve_choice == "y"

    if is_recursive:
        result = dfs_recursive(graph, start_str, order_preserving=order_preserving)
    else:
        result = dfs_iterative(graph, start_str, order_preserving=order_preserving)

    display_dfs_run(result, title=f"INTERACTIVE DFS RUN ({result.mode.upper()})")

    save_img = input("Save visualization image? (path or empty to skip): ").strip()
    if save_img:
        visualize_dfs(result, save_path=save_img)


def main() -> None:
    """CLI Entry point for Depth-First Search runner script."""
    parser = argparse.ArgumentParser(
        description="Depth-First Search (DFS) for Graph Traversal (Question 12 Runner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Graph specification as dict literal, JSON, compact '1: 2,3; 2: 4' or edge list.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1",
        help="Starting node label for DFS traversal (default: '1').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["iterative", "recursive", "both"],
        default="iterative",
        help="DFS execution mode: 'iterative' (explicit stack), 'recursive', or 'both' (default: iterative).",
    )
    parser.add_argument(
        "--order-preserving",
        action="store_true",
        help="Preserve left-to-right neighbor ordering (iterative reverses push, recursive preserves forward call).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console input prompt.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Execute Optional Challenge: compare iterative and recursive implementations side-by-side.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark suite across diverse graph topologies.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save graphical NetworkX/Matplotlib visualization (e.g. 'reports/dfs_traversal.png').",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory questions, answers, and comparison matrix.",
    )

    args = parser.parse_args()

    if args.viva:
        print_viva_guide()
        return

    if args.interactive:
        run_interactive_mode()
        return

    # Parse graph if specified, else use default sample graph
    if args.graph is not None:
        try:
            graph = parse_graph_input(args.graph)
        except Exception as err:
            print(f"[Error] Failed to parse input graph: {err}")
            sys.exit(1)
    else:
        graph = DEFAULT_SAMPLE_GRAPH

    start_node = args.start

    if args.demo:
        run_benchmark_suite()
        return

    if args.challenge or args.mode == "both":
        run_optional_challenge(graph, start_node)
        return

    # Single search run
    if args.mode == "recursive":
        result = dfs_recursive(graph, start_node, order_preserving=args.order_preserving)
    else:
        result = dfs_iterative(graph, start_node, order_preserving=args.order_preserving)

    display_dfs_run(result, title=f"DEPTH-FIRST SEARCH TRAVERSAL (QUESTION 12)")

    if args.save:
        visualize_dfs(result, save_path=args.save)


if __name__ == "__main__":
    main()
