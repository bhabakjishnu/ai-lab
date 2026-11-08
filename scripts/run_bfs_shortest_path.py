"""Runner script for BFS Shortest Path Constructor (Question 13).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (cyclic graphs, multi-path diamond graphs, disconnected graphs),
step-by-step trace tables, comparison of early vs late goal testing, comparison
of parent dictionary vs queue path tracking, optional challenge (all possible shortest paths),
NetworkX/Matplotlib graphical rendering, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 13: Start A, Goal F -> ['A', 'C', 'F']):
    python scripts/run_bfs_shortest_path.py

    # 2. Custom Start and Goal via CLI:
    python scripts/run_bfs_shortest_path.py --start A --goal F

    # 3. Custom Graph and Start/Goal via CLI:
    python scripts/run_bfs_shortest_path.py --graph "{'A': ['B', 'C'], 'B': ['D'], 'C': ['F'], 'D': [], 'F': []}" --start A --goal F

    # 4. Interactive Console Mode:
    python scripts/run_bfs_shortest_path.py --interactive

    # 5. Run All Standard Benchmarks:
    python scripts/run_bfs_shortest_path.py --demo

    # 6. Run Optional Challenge (Find All Possible Shortest Paths):
    python scripts/run_bfs_shortest_path.py --all-paths

    # 7. Compare Early Goal Testing vs Late Goal Testing:
    python scripts/run_bfs_shortest_path.py --compare-goals

    # 8. Compare Parent Dictionary vs Queue Storing Paths:
    python scripts/run_bfs_shortest_path.py --compare-modes

    # 9. Generate and Save Graphical Visualization:
    python scripts/run_bfs_shortest_path.py --save reports/bfs_shortest_path.png

    # 10. Display Viva Voce Study Guide:
    python scripts/run_bfs_shortest_path.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bfs_shortest_path import (
    CYCLIC_BENCHMARK_GRAPH,
    DEFAULT_SHORTEST_PATH_GRAPH,
    DISCONNECTED_SHORTEST_PATH_GRAPH,
    MULTI_PATH_DIAMOND_GRAPH,
    MULTI_PATH_GRID_GRAPH,
    BFSShortestPathConstructor,
    ShortestPathResult,
    bfs_shortest_path,
    compare_goal_testing_strategies,
    compare_tracking_mechanisms,
    find_all_shortest_paths,
    get_viva_reference,
    parse_graph_input,
    visualize_shortest_path,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header."""
    line = "=" * 78
    print(f"\n{line}")
    print(f" {title.upper()}")
    print(f"{line}")


def display_result(result: ShortestPathResult, show_table: bool = True) -> None:
    """Display comprehensive BFS shortest path findings, metrics, and diagnostics.

    Args:
        result: Completed ShortestPathResult instance.
        show_table: Whether to display the step-by-step FIFO execution trace table.
    """
    print_banner(f"BFS SHORTEST PATH CONSTRUCTOR (Start: '{result.start_node}' -> Goal: '{result.goal_node}')")
    print(result.format_summary())

    print("\n[Parent Dictionary Mapping (Reconstruction Structure)]")
    print(result.format_parent_map())

    if show_table:
        print("\n[Chronological Step-by-Step FIFO Frontier Trace]")
        print(result.format_step_table())


def run_benchmarks() -> None:
    """Run an automated battery of standard AI benchmark graphs."""
    print_banner("BATTERY OF STANDARD AI GRAPH BENCHMARKS")

    benchmarks = [
        (
            "Benchmark 1: Standard Lab Graph (A -> F)",
            DEFAULT_SHORTEST_PATH_GRAPH,
            "A",
            "F",
        ),
        (
            "Benchmark 2: Cyclic Graph with Loops (A -> F)",
            CYCLIC_BENCHMARK_GRAPH,
            "A",
            "F",
        ),
        (
            "Benchmark 3: Multi-Path Diamond Graph (A -> D)",
            MULTI_PATH_DIAMOND_GRAPH,
            "A",
            "D",
        ),
        (
            "Benchmark 4: Disconnected Graph Unreachable Goal (A -> F)",
            DISCONNECTED_SHORTEST_PATH_GRAPH,
            "A",
            "F",
        ),
        (
            "Benchmark 5: Identity Path (Start == Goal: A -> A)",
            DEFAULT_SHORTEST_PATH_GRAPH,
            "A",
            "A",
        ),
    ]

    for title, graph, start, goal in benchmarks:
        print(f"\n---> {title}")
        print(f"Graph Adjacency : {graph}")
        result = bfs_shortest_path(graph, start, goal)
        print(f"Shortest Path   : {result.path}")
        print(f"Path Cost       : {result.cost_edges} edge(s)")
        print(f"Nodes Expanded  : {result.nodes_expanded}")
        print(f"Status          : {'Found' if result.goal_found else 'Unreachable'}")
        all_paths = find_all_shortest_paths(graph, start, goal)
        if len(all_paths) > 1:
            print(f"All Shortest Paths ({len(all_paths)} found): {all_paths}")


def run_all_paths_demo(
    graph_input: Optional[str] = None,
    start: str = "A",
    goal: str = "F",
) -> None:
    """Demonstrate the Optional Challenge: Finding all possible shortest paths."""
    print_banner("OPTIONAL CHALLENGE: ALL POSSIBLE SHORTEST PATHS")
    if graph_input:
        target_graph = parse_graph_input(graph_input)
    else:
        target_graph = MULTI_PATH_DIAMOND_GRAPH
        if goal == "F" and "F" not in target_graph:
            goal = "D"
    print(f"Graph Definition : {target_graph}")
    print(f"Start Node       : {start}")
    print(f"Goal Node        : {goal}")

    all_paths = find_all_shortest_paths(target_graph, start, goal)
    if not all_paths:
        print("\nNo paths exist between the specified nodes.")
        return

    print(f"\nDiscovered {len(all_paths)} shortest path(s) of minimum length {len(all_paths[0]) - 1} edges:")
    for idx, p in enumerate(all_paths, start=1):
        print(f"  Path #{idx}: {' -> '.join(p)} (List: {p})")


def run_goal_comparison_demo(graph: Dict[str, List[str]], start: str, goal: str) -> None:
    """Run and format comparison between Early and Late goal testing."""
    print_banner("GOAL TESTING COMPARISON: EARLY VS LATE GOAL TESTING")
    comp = compare_goal_testing_strategies(graph, start, goal)

    print(f"Problem: Start = '{comp['start_node']}', Goal = '{comp['goal_node']}'")
    print(f"Shortest Path Found: {comp['shortest_path']} (Cost: {comp['path_cost_edges']} edges)\n")

    headers = f"{'Metric':<25} | {'Early Goal Testing (Generation)':<32} | {'Late Goal Testing (Expansion)':<30}"
    sep = "-" * len(headers)
    print(headers)
    print(sep)
    print(
        f"{'Nodes Expanded':<25} | "
        f"{comp['early_goal_test']['nodes_expanded']:<32} | "
        f"{comp['late_goal_test']['nodes_expanded']:<30}"
    )
    print(
        f"{'Nodes Generated':<25} | "
        f"{comp['early_goal_test']['nodes_generated']:<32} | "
        f"{comp['late_goal_test']['nodes_generated']:<30}"
    )
    print(
        f"{'Total Step Records':<25} | "
        f"{comp['early_goal_test']['total_steps']:<32} | "
        f"{comp['late_goal_test']['total_steps']:<30}"
    )
    print(
        f"{'Execution Time (ms)':<25} | "
        f"{comp['early_goal_test']['execution_time_ms']:<32.4f} | "
        f"{comp['late_goal_test']['execution_time_ms']:<30.4f}"
    )
    print(sep)
    print(f"\n[Efficiency Benefit]: Early goal testing saved {comp['expansions_saved']} node expansion(s)!")


def run_modes_comparison_demo(graph: Dict[str, List[str]], start: str, goal: str) -> None:
    """Run and format comparison between Parent Dict Backtracking and Queue Paths."""
    print_banner("TRACKING MECHANISM COMPARISON: PARENT DICT VS QUEUE PATHS")
    comp = compare_tracking_mechanisms(graph, start, goal)

    print(f"Problem: Start = '{comp['start']}', Goal = '{comp['goal']}'")
    print(f"Path Discovered: {comp['path']}")
    print(f"Identical Outputs: {comp['identical_paths']}\n")

    print(f"1. Parent Dictionary Backtracking (Recommended):")
    print(f"   - Auxiliary Space Complexity : {comp['parent_dict']['space_complexity']}")
    print(f"   - Nodes Expanded             : {comp['parent_dict']['nodes_expanded']}")
    print(f"   - Execution Time             : {comp['parent_dict']['execution_time_ms']:.4f} ms")

    print(f"\n2. Queue-Stored Paths:")
    print(f"   - Auxiliary Space Complexity : {comp['queue_paths']['space_complexity']}")
    print(f"   - Nodes Expanded             : {comp['queue_paths']['nodes_expanded']}")
    print(f"   - Execution Time             : {comp['queue_paths']['execution_time_ms']:.4f} ms")


def display_viva_notes() -> None:
    """Print the complete academic Viva Voce reference guide."""
    viva = get_viva_reference()
    print_banner("VIVA VOCE REFERENCE & THEORY GUIDE: BFS SHORTEST PATH CONSTRUCTOR")

    for topic_key, topic_data in viva["topics"].items():
        print(f"\n### {topic_data['question']}")
        print(f"Summary: {topic_data['summary']}")
        print("\nDetailed Answer:")
        print(topic_data["in_depth_answer"])
        print("-" * 78)


def interactive_mode() -> None:
    """Launch interactive console session prompting user for graph, start, and goal."""
    print_banner("INTERACTIVE BFS SHORTEST PATH CONSTRUCTOR")
    print("Enter your graph specification.")
    print("Examples:")
    print("  1. Dictionary literal: {'A': ['B', 'C'], 'B': ['D'], 'C': ['F'], 'D': [], 'F': []}")
    print("  2. Compact notation  : A: B, C; B: D; C: F; D:; F:")
    print("  3. Press Enter to use Default Lab Graph (A -> F)")

    raw_graph = input("\nEnter graph: ").strip()
    if not raw_graph:
        graph = DEFAULT_SHORTEST_PATH_GRAPH
        print("Using Default Graph: {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'], 'D': ['E'], 'E': ['F'], 'F': []}")
    else:
        try:
            graph = parse_graph_input(raw_graph)
            print(f"Parsed Graph: {graph}")
        except Exception as exc:
            print(f"[Error] Failed to parse graph: {exc}")
            return

    nodes = sorted(graph.keys())
    print(f"Available vertices: {nodes}")

    start = input(f"Enter Start Node [default 'A']: ").strip()
    if not start:
        start = "A" if "A" in graph else nodes[0]

    goal = input(f"Enter Goal Node [default 'F']: ").strip()
    if not goal:
        goal = "F" if "F" in graph else nodes[-1]

    try:
        result = bfs_shortest_path(graph, start, goal)
        display_result(result, show_table=True)

        all_paths = find_all_shortest_paths(graph, start, goal)
        if len(all_paths) > 1:
            print(f"\n[Optional Challenge] Discovered {len(all_paths)} tied shortest paths:")
            for p in all_paths:
                print(f"  {' -> '.join(p)}")
    except Exception as exc:
        print(f"[Error] Search execution failed: {exc}")


def main() -> None:
    """CLI Entry point for BFS Shortest Path Constructor."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_bfs_shortest_path.py",
        description="Question 13: BFS Shortest Path Constructor with Early Goal Testing & Path Reconstruction",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Graph specification (dict literal, JSON, or compact syntax).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="A",
        help="Start/source node label (default: 'A').",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="F",
        help="Target/destination goal node label (default: 'F').",
    )
    parser.add_argument(
        "--late",
        action="store_true",
        help="Use Late Goal Testing (at expansion) instead of Early Goal Testing.",
    )
    parser.add_argument(
        "--queue-paths",
        action="store_true",
        help="Use queue path tracking instead of parent dictionary backtracking.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark battery.",
    )
    parser.add_argument(
        "--all-paths",
        action="store_true",
        help="Run Optional Challenge: Find all tied shortest paths.",
    )
    parser.add_argument(
        "--compare-goals",
        action="store_true",
        help="Compare Early Goal Testing vs Late Goal Testing performance.",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare Parent Dictionary vs Queue Path tracking mechanisms.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save Matplotlib graph visualization to specified file path (.png/.svg).",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory and examination guide.",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if args.viva:
        display_viva_notes()
        return

    if args.demo:
        run_benchmarks()
        return

    if args.all-paths if hasattr(args, "all-paths") else args.all_paths:
        run_all_paths_demo(args.graph, args.start, args.goal)
        return

    target_graph = (
        parse_graph_input(args.graph)
        if args.graph
        else DEFAULT_SHORTEST_PATH_GRAPH
    )

    if args.compare_goals:
        run_goal_comparison_demo(target_graph, args.start, args.goal)
        return

    if args.compare_modes:
        run_modes_comparison_demo(target_graph, args.start, args.goal)
        return

    # Standard run
    early_goal = not args.late
    tracking = "queue_paths" if args.queue_paths else "parent_dict"

    result = bfs_shortest_path(
        target_graph,
        start=args.start,
        goal=args.goal,
        early_goal_test=early_goal,
        tracking_mode=tracking,
    )

    display_result(result, show_table=True)

    if args.save:
        visualize_shortest_path(result, save_path=args.save)


if __name__ == "__main__":
    main()
