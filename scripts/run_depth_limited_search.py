"""Runner script for Depth-Limited Search (DLS) (Question 16).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (sample graph cutoff, success cases, failure cases, cyclic graphs),
step-by-step trace tables, Optional Challenge tracking (deepest node reached before cutoff),
NetworkX/Matplotlib visual generation, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 16 default: Limit: 2, Goal is at depth 3 -> Result: Cutoff):
    python scripts/run_depth_limited_search.py

    # 2. Custom Start, Goal, and Depth Limit via CLI:
    python scripts/run_depth_limited_search.py --start 1 --goal 6 --limit 3

    # 3. Custom Graph via CLI:
    python scripts/run_depth_limited_search.py --graph "1: 2, 3; 2: 4; 4: 6" --start 1 --goal 6 --limit 2

    # 4. Interactive Console Mode:
    python scripts/run_depth_limited_search.py --interactive

    # 5. Run Optional Challenge Demonstration (deepest node before cutoff):
    python scripts/run_depth_limited_search.py --challenge

    # 6. Run All Standard Benchmarks (Cutoff, Success, Failure, Cyclic, Binary Tree):
    python scripts/run_depth_limited_search.py --demo

    # 7. Generate and Save Graphical Visualization:
    python scripts/run_depth_limited_search.py --save reports/dls_search.png

    # 8. Display Viva Voce Study Guide & Comparison Matrix:
    python scripts/run_depth_limited_search.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.depth_limited_search import (
    BINARY_TREE_DLS_GRAPH,
    CYCLIC_DLS_GRAPH,
    DISCONNECTED_DLS_GRAPH,
    LINEAR_CHAIN_DLS_GRAPH,
    SAMPLE_DLS_GRAPH,
    CUTOFF,
    FAILURE,
    SUCCESS,
    DLSResult,
    DepthLimitedSearch,
    depth_limited_search,
    dls,
    get_dls_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_dls,
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
    viva = get_dls_viva_reference()
    print_banner("DEPTH-LIMITED SEARCH (DLS) VIVA VOCE REFERENCE & THEORY GUIDE")
    print(f"Core AI Concept: {viva['core_concept']}\n")

    print("[1] SYLLABUS VIVA TOPICS:")
    topics = viva["viva_topics"]

    print("\n  • Infinite State Spaces:")
    print(f"    - Definition: {topics['infinite_state_spaces']['definition']}")
    print(f"    - Standard DFS Failure Mode: {topics['infinite_state_spaces']['dfs_failure_mode']}")
    print(f"    - DLS Resolution: {topics['infinite_state_spaces']['dls_resolution']}")

    print("\n  • Drawbacks of Standard DFS:")
    print(f"    - Incompleteness: {topics['drawback_of_standard_dfs']['incompleteness']}")
    print(f"    - Non-Optimality: {topics['drawback_of_standard_dfs']['non_optimality']}")
    print(f"    - Recursion Overflow: {topics['drawback_of_standard_dfs']['recursion_overflow']}")

    print("\n  • Cutoff vs Failure:")
    print(f"    - Cutoff Meaning: {topics['cutoff_vs_failure']['cutoff_meaning']}")
    print(f"    - Failure Meaning: {topics['cutoff_vs_failure']['failure_meaning']}")
    print("    - Foundation for IDS:")
    for line in topics["cutoff_vs_failure"]["importance_for_ids"].split("\n"):
        print(f"      {line}")

    print("\n" + "-" * 75)
    print("[2] SEARCH ALGORITHMS COMPLEXITY COMPARISON MATRIX:")
    print(f"{'Algorithm':<8} | {'Time Complexity':<16} | {'Space Complexity':<26} | {'Complete?':<32} | {'Optimal?'}")
    print("=" * 95)
    for algo, data in viva["complexity_comparison"].items():
        print(f"{algo:<8} | {data['time']:<16} | {data['space']:<26} | {data['complete']:<32} | {data['optimal']}")
    print("=" * 95)

    print("\n[3] HIGH-FREQUENCY VIVA VOCE QUESTIONS & ANSWERS:")
    for i, qa in enumerate(viva["key_questions"], start=1):
        print(f"\n  Q{i}: {qa['question']}")
        print("  A:")
        for line in qa["answer"].split("\n"):
            print(f"     {line}")
    print("\n" + "=" * 75)


def display_dls_run(
    result: DLSResult,
    title: Optional[str] = None,
    show_table: bool = True,
) -> None:
    """Format and display detailed results of a single DLS execution.

    Args:
        result: The DLSResult object to inspect.
        title: Optional custom section header.
        show_table: Whether to print the step-by-step execution trace table.
    """
    if title:
        print_banner(title)

    print(f"Start Node               : {result.start_node}")
    print(f"Goal Node                : {result.goal_node}")
    print(f"Depth Limit (l)          : {result.limit}")
    print(f"Outcome Status           : {result.status.upper()}")
    print(f"Official Practical Output: {result.format_result()}")

    if result.is_success():
        print(f"Solution Path            : {result.format_path()}")
        print(f"Solution Depth (edges)   : {result.depth}")
    elif result.is_cutoff():
        print(f"Cutoff Boundary Nodes    : {', '.join(result.cutoff_nodes)}")
        print(f"Deepest Node Reached     : {result.deepest_node} (Depth: {result.max_depth_reached}) [Optional Challenge]")
    else:
        print("Failure Reason           : Goal not present in reachable state space (exhaustively verified).")

    print(f"Nodes Generated (Visits) : {result.nodes_generated}")
    print(f"Nodes Expanded           : {result.nodes_expanded}")
    print(f"Maximum Depth Reached    : {result.max_depth_reached}")
    print(f"Deepest Nodes Reached    : {', '.join(result.deepest_nodes)}")
    print(f"Execution Wall Time      : {result.execution_time_ms:.4f} ms")

    if show_table and result.steps:
        print("\nStep-by-Step Execution Trace Table:")
        print(result.format_step_table())


def run_sample_demonstration() -> None:
    """Run the exact sample input from the university practical syllabus.

    Sample Input: Limit: 2, Goal is at depth 3.
    Sample Output: Result: Cutoff
    """
    print_banner("QUESTION 16: OFFICIAL SAMPLE INPUT DEMONSTRATION")
    print("Problem Context:")
    print("  Graph: Canonical University Practical Tree Structure")
    print("    Depth 0: Node '1'")
    print("    Depth 1: Node '2', Node '3'")
    print("    Depth 2: Node '4' (child of '2'), Node '5' (child of '3')")
    print("    Depth 3: Node '6' (child of '4') <-- GOAL TARGET")
    print("  Start Node: '1'")
    print("  Goal Node : '6' (Located at Depth 3)")
    print("  Depth Limit: 2\n")

    result = depth_limited_search(
        graph=SAMPLE_DLS_GRAPH,
        start="1",
        goal="6",
        limit=2,
    )
    display_dls_run(result, title=None, show_table=True)


def run_challenge_demonstration() -> None:
    """Demonstrate the Optional Challenge: tracking deepest node reached before cutoff."""
    print_banner("OPTIONAL CHALLENGE: TRACKING DEEPEST NODE BEFORE CUTOFF")
    print("Objective: Monitor tree descent and capture the deepest node reached prior to cutoff.")
    print("Scenario A: Deep Linear Chain (N0 -> N1 -> N2 -> N3 -> N4 -> N5), Limit=3, Goal='N5'")

    res_a = depth_limited_search(
        graph=LINEAR_CHAIN_DLS_GRAPH,
        start="N0",
        goal="N5",
        limit=3,
    )
    print(f"  • Result Status         : {res_a.status}")
    print(f"  • Max Depth Reached     : {res_a.max_depth_reached} (Limit was {res_a.limit})")
    print(f"  • Deepest Node Reached  : '{res_a.deepest_node}'")
    print(f"  • Cutoff Nodes          : {res_a.cutoff_nodes}")

    print("\nScenario B: Binary Tree with Branching Alternatives, Limit=2, Goal='J' (at Depth 3)")
    res_b = depth_limited_search(
        graph=BINARY_TREE_DLS_GRAPH,
        start="A",
        goal="J",
        limit=2,
    )
    print(f"  • Result Status         : {res_b.status}")
    print(f"  • Max Depth Reached     : {res_b.max_depth_reached} (Limit was {res_b.limit})")
    print(f"  • Deepest Nodes Reached : {res_b.deepest_nodes}")
    print(f"  • Cutoff Nodes Hit      : {res_b.cutoff_nodes}")

    print("\nExecution Step Table for Scenario B:")
    print(res_b.format_step_table())


def run_all_benchmarks() -> None:
    """Run comprehensive benchmark suite demonstrating all three DLS outcome states."""
    print_banner("DEPTH-LIMITED SEARCH (DLS) COMPREHENSIVE BENCHMARK SUITE")

    # Benchmark 1: Cutoff (Goal beyond limit)
    print("\n[Benchmark 1: Cutoff State - Goal Depth > Depth Limit]")
    r1 = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=2)
    print(f"  Input : Start='1', Goal='6' (at depth 3), Limit=2")
    print(f"  Output: {r1.format_result()} | Deepest Node: '{r1.deepest_node}' at Depth {r1.max_depth_reached}")

    # Benchmark 2: Success (Goal at or within limit)
    print("\n[Benchmark 2: Success State - Depth Limit Sufficient]")
    r2 = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=3)
    print(f"  Input : Start='1', Goal='6' (at depth 3), Limit=3")
    print(f"  Output: {r2.format_result()} | Solution: {r2.format_path()}")

    # Benchmark 3: Failure State (Goal completely absent from graph)
    print("\n[Benchmark 3: Failure State - Goal Absent / Disconnected]")
    r3 = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="NON_EXISTENT", limit=5)
    print(f"  Input : Start='1', Goal='NON_EXISTENT', Limit=5")
    print(f"  Output: {r3.format_result()} | Cutoff Occurred: {r3.cutoff_occurred}")

    # Benchmark 4: Cyclic Graph Handling
    print("\n[Benchmark 4: Cyclic Graph with Branch Pruning]")
    r4 = depth_limited_search(CYCLIC_DLS_GRAPH, start="1", goal="5", limit=3)
    print(f"  Input : Cyclic Graph, Start='1', Goal='5', Limit=3")
    print(f"  Output: {r4.format_result()} | Path: {r4.format_path()}")

    # Benchmark 5: Root is Goal (Limit=0)
    print("\n[Benchmark 5: Boundary Condition - Start Node is Goal at Limit=0]")
    r5 = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="1", limit=0)
    print(f"  Input : Start='1', Goal='1', Limit=0")
    print(f"  Output: {r5.format_result()} | Path: {r5.format_path()}")


def interactive_mode() -> None:
    """Prompt user interactively for graph topology, start node, goal node, and depth limit."""
    print_banner("DEPTH-LIMITED SEARCH (DLS) INTERACTIVE CONSOLE")
    print("Enter graph topology in one of the following formats:")
    print("  1. Semicolon-delimited: '1: 2, 3; 2: 4; 3: 5; 4: 6; 5:; 6:'")
    print("  2. Python dictionary:  {'1': ['2', '3'], '2': ['4'], ...}")
    print("  3. Press [ENTER] to use default Sample Graph.\n")

    graph_input = input("Enter Graph [Default: Sample Graph]: ").strip()
    if not graph_input:
        graph = SAMPLE_DLS_GRAPH
        print("Using default Sample Graph.")
    else:
        try:
            graph = parse_graph_input(graph_input)
            print(f"Successfully parsed graph with {len(graph)} nodes.")
        except Exception as exc:
            print(f"Error parsing graph: {exc}. Falling back to default Sample Graph.")
            graph = SAMPLE_DLS_GRAPH

    start_node = input("Enter Start Node [Default: '1']: ").strip() or "1"
    goal_node = input("Enter Goal Node [Default: '6']: ").strip() or "6"
    limit_str = input("Enter Depth Limit (non-negative integer) [Default: 2]: ").strip() or "2"

    try:
        limit = int(limit_str)
    except ValueError:
        print("Invalid limit integer. Defaulting to 2.")
        limit = 2

    try:
        result = depth_limited_search(
            graph=graph,
            start=start_node,
            goal=goal_node,
            limit=limit,
        )
        display_dls_run(result, title="INTERACTIVE DLS EXECUTION RESULT", show_table=True)
    except Exception as exc:
        print(f"\nExecution Error: {exc}")


def main() -> None:
    """Entry point dispatcher for DLS CLI execution."""
    parser = argparse.ArgumentParser(
        description="Depth-Limited Search (DLS) Runner — University Lab Question 16",
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
        default="6",
        help="Goal node identifier (default: '6').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum depth limit integer (default: 2).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Custom graph adjacency list (JSON, Python dict, or 'u: v1, v2; ...').",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run standard syllabus sample problem (Limit: 2, Goal depth 3 -> Result: Cutoff).",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge demo (tracking deepest node reached before cutoff).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full benchmark demonstration across Cutoff, Success, Failure, and Cyclic scenarios.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory notes, questions, and comparison matrix.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print full step-by-step execution trace table.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console prompting for graph, start, goal, and limit.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save NetworkX/Matplotlib graphical visualization (e.g. reports/dls.png).",
    )

    args = parser.parse_args()

    if args.viva:
        print_viva_guide()
        return

    if args.challenge:
        run_challenge_demonstration()
        return

    if args.demo:
        run_all_benchmarks()
        return

    if args.interactive:
        interactive_mode()
        return

    # Parse graph if specified, else use sample graph
    if args.graph:
        graph = parse_graph_input(args.graph)
    else:
        graph = SAMPLE_DLS_GRAPH

    # Execute search
    result = depth_limited_search(
        graph=graph,
        start=args.start,
        goal=args.goal,
        limit=args.limit,
    )

    title = "DEPTH-LIMITED SEARCH (DLS) EXECUTION RESULT"
    display_dls_run(result, title=title, show_table=args.table or (not args.graph and args.limit == 2))

    if args.save:
        saved_path = visualize_dls(result, output_path=args.save)
        print(f"\n[Success] Graphical visualization saved to: {saved_path}")


if __name__ == "__main__":
    main()
