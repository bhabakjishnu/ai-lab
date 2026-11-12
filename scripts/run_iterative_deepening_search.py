"""Runner script for Iterative Deepening Search (IDS) (Question 17).

Demonstrates Iterative Deepening Search by wrapping Depth-Limited Search (DLS) in an
incrementing depth limit loop (0, 1, 2, ...), printing nodes expanded at each depth
iteration, verifying shortest-path optimality vs BFS with O(b * d) DFS memory efficiency,
counting total node expansions across all iterations (Optional Challenge), rendering
two-panel visualizations, and displaying Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Goal at depth 2 -> Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found [Path].):
    python scripts/run_iterative_deepening_search.py

    # 2. Custom Start and Goal via CLI:
    python scripts/run_iterative_deepening_search.py --start 1 --goal 6

    # 3. Custom Graph via CLI:
    python scripts/run_iterative_deepening_search.py --graph "1: 2, 3; 2: 4, 5; 3: 6" --start 1 --goal 4

    # 4. Compare IDS vs BFS (Shortest-Path Optimality & Memory Footprint):
    python scripts/run_iterative_deepening_search.py --compare-bfs

    # 5. Optional Challenge Demonstration (Total Node Expansions Across All Iterations):
    python scripts/run_iterative_deepening_search.py --challenge

    # 6. Comprehensive Benchmark Suite (Sample, Shortcut, Cyclic, Exhausted Failure, Binary Tree):
    python scripts/run_iterative_deepening_search.py --demo

    # 7. Interactive Console Mode:
    python scripts/run_iterative_deepening_search.py --interactive

    # 8. Save Graphical Visualization:
    python scripts/run_iterative_deepening_search.py --save reports/ids_search.png

    # 9. Display Viva Voce Study Guide:
    python scripts/run_iterative_deepening_search.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.iterative_deepening_search import (
    BINARY_TREE_IDS_GRAPH,
    CYCLIC_IDS_GRAPH,
    DISCONNECTED_IDS_GRAPH,
    SAMPLE_IDS_GRAPH,
    SHORTCUT_IDS_GRAPH,
    IDSResult,
    compare_ids_with_bfs,
    get_ids_viva_reference,
    iterative_deepening_search,
    parse_graph_input,
    visualize_ids,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header."""
    width = 78
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_viva_guide() -> None:
    """Print the complete Viva Voce reference guide for Iterative Deepening Search."""
    viva = get_ids_viva_reference()
    print_banner("ITERATIVE DEEPENING SEARCH (IDS) VIVA VOCE REFERENCE & THEORY GUIDE")
    print(f"Core AI Concept: {viva['core_concept']}\n")

    print("[1] SYLLABUS VIVA TOPICS:")
    topics = viva["viva_topics"]

    print("\n  • Why IDS is Preferred in Large State Spaces:")
    large = topics["why_ids_preferred_in_large_state_spaces"]
    print(f"    - BFS Memory Bottleneck : {large['memory_bottleneck_of_bfs']}")
    print(f"    - IDS Linear Memory     : {large['ids_advantage']}")
    print(f"    - Unknown Goal Depth    : {large['unknown_solution_depth']}")

    print("\n  • Redundant Node Generation:")
    red = topics["redundant_node_generation"]
    print(f"    - Apparent Concern      : {red['apparent_concern']}")
    print("    - Mathematical Proof    :")
    for line in red["mathematical_proof"].split("\n"):
        print(f"      {line}")

    print("\n  • Time vs Space Trade-off:")
    print(f"    - {topics['time_vs_space_tradeoff']['summary']}")

    print("\n" + "-" * 78)
    print("[2] SEARCH ALGORITHMS COMPLEXITY COMPARISON MATRIX:")
    print(
        f"{'Algorithm':<9} | {'Time':<10} | {'Space':<32} | {'Complete?':<24} | {'Optimal?'}"
    )
    print("=" * 98)
    for algo, data in viva["complexity_comparison"].items():
        print(
            f"{algo:<9} | {data['time']:<10} | {data['space']:<32} | "
            f"{data['complete']:<24} | {data['optimal']}"
        )
    print("=" * 98)

    print("\n[3] HIGH-FREQUENCY VIVA VOCE QUESTIONS & ANSWERS:")
    for i, qa in enumerate(viva["key_questions"], start=1):
        print(f"\n  Q{i}: {qa['question']}")
        print(f"  A : {qa['answer']}")
    print("\n" + "=" * 78)


def display_ids_run(
    result: IDSResult,
    title: Optional[str] = None,
    show_iterations: bool = True,
    show_table: bool = True,
) -> None:
    """Format and print detailed IDS execution metrics, per-depth expansions, and path."""
    if title:
        print_banner(title)

    print(f"Start Node                  : {result.start_node}")
    print(f"Goal Node                   : {result.goal_node}")
    print(f"Final Outcome               : {result.status.upper()}")
    print(f"Sequence of Limits Tried    : {result.format_sequence()}")

    if result.is_success():
        print(f"Final Shortest Path         : {result.format_path()}")
        print(f"Goal Depth (edges)          : {result.depth}")
    elif result.is_failure():
        print("Failure Reason              : Graph exhausted without finding goal (DLS returned Failure).")
    else:
        print(f"Cutoff Reason               : Reached max_depth cap ({result.max_depth_cap}).")

    print(f"Total Node Expansions       : {result.total_nodes_expanded} (Optional Challenge)")
    print(f"Total Node Visits/Generated : {result.total_nodes_generated}")
    print(f"Redundant Prior Generations : {result.redundant_regenerations}")
    print(f"Peak Recursion Stack Depth  : {result.peak_stack_depth} nodes (O(b * d) space)")
    print(f"Execution Wall Time         : {result.execution_time_ms:.4f} ms")

    if show_iterations and result.iterations:
        print("\nNodes Expanded at Each Depth Iteration:")
        for it in result.iterations:
            expanded_display = it.expanded_nodes if it.expanded_nodes else ["(none)"]
            print(
                f"  • {it.format_outcome():<35} | Expanded ({it.nodes_expanded}): "
                f"{expanded_display} | Visited ({it.nodes_generated}): {it.visited_nodes}"
            )

    if show_table and result.iterations:
        print("\nPer-Iteration Breakdown Table:")
        print(result.format_iteration_table())


def run_bfs_comparison_demo() -> None:
    """Demonstrate that IDS finds the shortest path like BFS using DFS linear memory."""
    print_banner("IDS vs BFS COMPARISON: SHORTEST-PATH OPTIMALITY & LINEAR MEMORY")
    print("Graph: SHORTCUT_IDS_GRAPH (A -> B -> D -> E -> G [depth 4] vs A -> C -> G [depth 2])")
    print("  Standard DFS explores A -> B -> D -> E -> G (non-optimal depth 4).")
    print("  Both BFS and IDS must discover A -> C -> G (optimal depth 2).\n")

    cmp = compare_ids_with_bfs(SHORTCUT_IDS_GRAPH, start="A", goal="G")
    ids_res: IDSResult = cmp["ids_result"]

    print(f"IDS Sequence          : {ids_res.format_sequence()}")
    print(f"IDS Path (Depth {cmp['ids_depth']})    : {cmp['ids_path']}")
    print(f"BFS Path (Depth {cmp['bfs_depth']})    : {cmp['bfs_path']}")
    print(f"Shortest Path Match?  : {cmp['same_optimal_depth']}")
    print(f"IDS Peak Stack Depth  : {cmp['ids_peak_stack_depth']} nodes  [O(b * d) linear space]")
    print(
        f"BFS Stored Nodes      : {cmp['bfs_total_visited_stored']} visited, "
        f"peak frontier={cmp['bfs_peak_frontier_size']}  [O(b^d) exponential space]"
    )
    print(f"IDS Total Expansions  : {cmp['ids_total_expansions']}")
    print(f"IDS Total Generations : {cmp['ids_total_generations']}")


def run_challenge_demonstration() -> None:
    """Demonstrate Optional Challenge: counting total node expansions across all iterations."""
    print_banner("OPTIONAL CHALLENGE: TOTAL NODE EXPANSIONS & REDUNDANCY ANALYSIS")
    print("Scenario: Complete Binary Tree of Depth 3 (15 nodes), seeking leaf 'O' (depth 3).")
    print("Counts node expansions at every depth iteration 0, 1, 2, 3 and aggregates total.\n")

    res = iterative_deepening_search(BINARY_TREE_IDS_GRAPH, start="A", goal="O")
    display_ids_run(res, title=None, show_iterations=True, show_table=True)

    print(
        f"\n[Optional Challenge Result] Total node expansions across all {len(res.iterations)} "
        f"iterations = {res.total_nodes_expanded} (Total visits = {res.total_nodes_generated}, "
        f"Final iteration visits = {res.final_iteration_generated}, "
        f"Prior redundant visits = {res.redundant_regenerations})."
    )


def run_all_benchmarks() -> None:
    """Run comprehensive benchmark suite covering all Question 17 concepts."""
    print_banner("ITERATIVE DEEPENING SEARCH (IDS) COMPREHENSIVE BENCHMARK SUITE")

    print("\n[Benchmark 1: Canonical Sample Input — Goal at Depth 2]")
    r1 = iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="4")
    print(f"  Output      : {r1.format_sequence()}")
    print(f"  Total Exp.  : {r1.total_nodes_expanded}")

    print("\n[Benchmark 2: Shortest-Path Optimality vs DFS Trap]")
    r2 = iterative_deepening_search(SHORTCUT_IDS_GRAPH, start="A", goal="G")
    print(f"  Output      : {r2.format_sequence()}")
    print(f"  Optimal Path: {r2.format_path()} (Depth {r2.depth})")

    print("\n[Benchmark 3: Cyclic Graph — Goal Reachable]")
    r3 = iterative_deepening_search(CYCLIC_IDS_GRAPH, start="1", goal="6")
    print(f"  Output      : {r3.format_sequence()}")
    print(f"  Total Exp.  : {r3.total_nodes_expanded}")

    print("\n[Benchmark 4: Cyclic Graph — Unreachable Goal (Exhaustive Failure Termination)]")
    r4 = iterative_deepening_search(CYCLIC_IDS_GRAPH, start="1", goal="MISSING")
    print(f"  Output      : {r4.format_sequence()}")
    print(f"  Status      : {r4.status} after {len(r4.iterations)} iterations")

    print("\n[Benchmark 5: Disconnected Graph — Exhaustive Failure Termination]")
    r5 = iterative_deepening_search(DISCONNECTED_IDS_GRAPH, start="A", goal="Y")
    print(f"  Output      : {r5.format_sequence()}")

    run_bfs_comparison_demo()


def interactive_mode() -> None:
    """Prompt user interactively for graph, start node, goal node, and optional max depth."""
    print_banner("ITERATIVE DEEPENING SEARCH (IDS) INTERACTIVE CONSOLE")
    print("Enter graph topology in one of the following formats:")
    print("  1. Semicolon-delimited: '1: 2, 3; 2: 4, 5; 3: 6; 4:; 5:; 6:'")
    print("  2. Python dictionary:  {'1': ['2', '3'], '2': ['4', '5'], ...}")
    print("  3. Press [ENTER] to use default Sample Graph.\n")

    graph_input = input("Enter Graph [Default: Sample Graph]: ").strip()
    if not graph_input:
        graph = SAMPLE_IDS_GRAPH
        print("Using default Sample Graph.")
    else:
        try:
            graph = parse_graph_input(graph_input)
            print(f"Successfully parsed graph with {len(graph)} nodes.")
        except Exception as exc:
            print(f"Error parsing graph: {exc}. Falling back to default Sample Graph.")
            graph = SAMPLE_IDS_GRAPH

    start_node = input("Enter Start Node [Default: '1']: ").strip() or "1"
    goal_node = input("Enter Goal Node [Default: '4']: ").strip() or "4"
    cap_str = input("Enter Optional Max Depth Cap [Default: None]: ").strip()

    max_depth: Optional[int] = None
    if cap_str:
        try:
            max_depth = int(cap_str)
        except ValueError:
            print("Invalid integer for max depth; running unbounded IDS.")

    try:
        result = iterative_deepening_search(
            graph=graph,
            start=start_node,
            goal=goal_node,
            max_depth=max_depth,
        )
        display_ids_run(result, title="INTERACTIVE IDS EXECUTION RESULT")
    except Exception as exc:
        print(f"\nExecution Error: {exc}")


def main() -> None:
    """Entry point dispatcher for Iterative Deepening Search CLI."""
    parser = argparse.ArgumentParser(
        description="Iterative Deepening Search (IDS) Runner — University Lab Question 17",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1",
        help="Start node identifier (default: '1').",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="4",
        help="Goal node identifier (default: '4', at depth 2 in default sample graph).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional upper depth limit cap (default: None / unbounded until Failure or Success).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Custom graph adjacency list (JSON, Python dict, or 'u: v1, v2; ...').",
    )
    parser.add_argument(
        "--no-cycle-prevention",
        action="store_true",
        help="Disable active-branch cycle pruning inside DLS.",
    )
    parser.add_argument(
        "--compare-bfs",
        action="store_true",
        help="Compare IDS shortest-path optimality and linear memory against BFS.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge demo (total node expansions across all iterations).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full benchmark suite across sample, shortcut, cyclic, and failure scenarios.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory notes and complexity comparison.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display per-iteration tabular breakdown.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console prompting for graph, start, and goal.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save NetworkX/Matplotlib visualization (e.g. reports/ids_search.png).",
    )

    args = parser.parse_args()

    if args.viva:
        print_viva_guide()
        return

    if args.challenge:
        run_challenge_demonstration()
        return

    if args.compare_bfs:
        run_bfs_comparison_demo()
        return

    if args.demo:
        run_all_benchmarks()
        return

    if args.interactive:
        interactive_mode()
        return

    graph = parse_graph_input(args.graph) if args.graph else SAMPLE_IDS_GRAPH

    result = iterative_deepening_search(
        graph=graph,
        start=args.start,
        goal=args.goal,
        cycle_prevention=not args.no_cycle_prevention,
        max_depth=args.max_depth,
    )

    display_ids_run(
        result,
        title="ITERATIVE DEEPENING SEARCH (IDS) EXECUTION RESULT",
        show_iterations=True,
        show_table=True if args.table or not args.graph else False,
    )

    if args.save:
        saved_path = visualize_ids(result, output_path=args.save)
        print(f"\n[Success] Graphical visualization saved to: {saved_path}")


if __name__ == "__main__":
    main()
