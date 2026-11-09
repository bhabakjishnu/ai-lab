"""Runner script for Uniform Cost Search (UCS) Implementation (Question 14).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (sample graph, diamond graph, cyclic graphs, Romania map, disconnected graphs),
step-by-step trace tables, priority queue inspection, Optional Challenge demonstration
(detecting negative edges and tracing negative cycles with Bellman-Ford), NetworkX/Matplotlib
visual generation, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 14 default: Graph 'A->B:1, B->C:2, A->C:4', Start: 'A', Goal: 'C'):
    python scripts/run_uniform_cost_search.py

    # 2. Custom Graph, Start, and Goal via CLI:
    python scripts/run_uniform_cost_search.py --graph "A->B:1, B->C:2, A->C:4" --start A --goal C

    # 3. Interactive Console Mode:
    python scripts/run_uniform_cost_search.py --interactive

    # 4. Run Optional Challenge Demonstration (Negative edges & negative cycles):
    python scripts/run_uniform_cost_search.py --challenge

    # 5. Run All Standard Benchmarks (Sample, Diamond, Romania Map, Cyclic, Disconnected):
    python scripts/run_uniform_cost_search.py --demo

    # 6. Generate and Save Graphical Visualization:
    python scripts/run_uniform_cost_search.py --save reports/ucs_optimal_path.png

    # 7. Display Viva Voce Study Guide & Comparison Matrix:
    python scripts/run_uniform_cost_search.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.uniform_cost_search import (
    CYCLIC_WEIGHTED_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DIAMOND_BENCHMARK_GRAPH,
    DISCONNECTED_WEIGHTED_GRAPH,
    NEGATIVE_CYCLE_GRAPH,
    NEGATIVE_EDGE_DAG,
    ROMANIA_MAP_SUBGRAPH,
    UCSResult,
    UniformCostSearch,
    detect_negative_cycle,
    get_ucs_viva_reference,
    normalize_weighted_graph,
    parse_weighted_graph,
    uniform_cost_search,
    validate_strictly_positive_weights,
    visualize_ucs,
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
    viva = get_ucs_viva_reference()
    print_banner("UNIFORM COST SEARCH (UCS) VIVA VOCE REFERENCE & THEORY GUIDE")
    print(f"Syllabus Area: {viva['syllabus_area']}")
    print(f"Algorithm:     {viva['algorithm']}\n")

    for i, topic in enumerate(viva["core_topics"], start=1):
        print(f"[{i}] {topic['topic'].upper()}")
        print(f"Question: {topic['question']}")
        print("Answer:")
        for line in topic["answer"].split("\n"):
            print(f"  {line}")
        print("-" * 75)

    matrix = viva["comparison_matrix"]
    headers = matrix["headers"]
    print(f"\n{headers[0]:<18} | {headers[1]:<32} | {headers[2]:<10} | {headers[3]:<12} | {headers[4]:<22} | {headers[5]}")
    print("=" * 115)
    for row in matrix["rows"]:
        print(f"{row[0]:<18} | {row[1]:<32} | {row[2]:<10} | {row[3]:<12} | {row[4]:<22} | {row[5]}")
    print("=" * 115)


def display_ucs_run(
    result: UCSResult,
    show_table: bool = True,
    show_graph: bool = True,
) -> None:
    """Display comprehensive results of a UCS search execution.

    Args:
        result: The UCSResult object to inspect and format.
        show_table: Whether to print the execution step table.
        show_graph: Whether to print the input graph adjacency structure.
    """
    print_banner(f"UNIFORM COST SEARCH: {result.start_node} -> {result.goal_node}")

    if show_graph:
        print("[Graph Adjacency Structure with Edge Costs]")
        for u in sorted(result.graph.keys()):
            edges = [f"{v} (w={w})" for v, w in result.graph[u]]
            edges_str = ", ".join(edges) if edges else "(no outgoing edges)"
            print(f"  {u} -> [{edges_str}]")
        print()

    # Academic & University Expected Output Format
    print("[Primary Result]")
    print(f"  {result.format_path()}")
    print()

    if result.warning_message:
        print(f"[Notice / Warning]: {result.warning_message}\n")

    print("[Search Statistics]")
    print(f"  - Start Node:            {result.start_node}")
    print(f"  - Destination Goal:      {result.goal_node}")
    print(f"  - Solution Path Found:   {'YES' if result.goal_found else 'NO (Unreachable)'}")
    print(f"  - Minimum Total Cost:    {result.total_cost if result.total_cost != float('inf') else 'Infinity'}")
    print(f"  - Optimal Path Hops:     {len(result.path) - 1 if result.path else 'N/A'}")
    print(f"  - Sequence of Path:      {' -> '.join(result.path) if result.path else 'None'}")
    print(f"  - Nodes Expanded:        {', '.join(result.nodes_expanded) if result.nodes_expanded else 'None'}")
    print(f"  - Total Node Expansions: {len(result.nodes_expanded)}")
    print(f"  - Total Search Steps:    {len(result.steps)}")
    print(f"  - Wall-Clock Run Time:   {result.execution_time_ms:.4f} ms")
    print()

    if show_table and result.steps:
        print("[Step-by-Step Priority Queue Trace Table]")
        print(result.format_step_table())
        print()


def run_challenge_demo() -> None:
    """Run Optional Challenge demonstration: handling negative edge weights and cycles."""
    print_banner("OPTIONAL CHALLENGE: NEGATIVE EDGE WEIGHTS & NEGATIVE CYCLES IN UCS")
    print("Theoretical Context:")
    print("  Uniform Cost Search fundamentally assumes non-decreasing path costs (edge weights >= eps > 0).")
    print("  When negative edge weights or negative cycles exist:")
    print("  1. The greedy assumption fails: a closed/explored node might receive a cheaper path later.")
    print("  2. In a negative cycle, looping continuously reduces path cost toward -infinity.\n")

    print("[Challenge Part 1: Directed Acyclic Graph with Negative Edge (No Cycle)]")
    print("Graph: S->A:4, S->B:2, A->C:-1, B->C:2, C->G:3. Start: S, Goal: G")
    res_neg_dag = uniform_cost_search(NEGATIVE_EDGE_DAG, start="S", goal="G")
    display_ucs_run(res_neg_dag, show_table=True)

    print("\n[Challenge Part 2: Reachable Negative Weight Cycle Detection via Bellman-Ford]")
    print("Graph: A->B:1, B->C:2, C->D:-5, D->B:1, D->Goal:3")
    print("Notice the cycle B -> C -> D -> B: 2 + (-5) + 1 = -2 (Net Negative Cycle!).")
    has_cycle, cycle_nodes, _ = detect_negative_cycle(NEGATIVE_CYCLE_GRAPH, start_node="A")
    print(f"Cycle Detection Status:  {'DETECTED' if has_cycle else 'NONE'}")
    print(f"Negative Cycle Path:     {' -> '.join(cycle_nodes or [])}")

    res_cycle = uniform_cost_search(NEGATIVE_CYCLE_GRAPH, start="A", goal="Goal")
    print(f"UCS Result with Cycle:   {res_cycle.format_path()}")
    print(f"Warning / Diagnostics:   {res_cycle.warning_message}")


def run_all_benchmarks() -> None:
    """Run comprehensive automated benchmarks covering key graph topologies."""
    print_banner("AUTOMATED BENCHMARKS: UNIFORM COST SEARCH TOPOLOGIES")

    benchmarks = [
        (
            "Benchmark 1: Sample Input Graph (Question 14)",
            DEFAULT_SAMPLE_GRAPH,
            "A",
            "C",
            "Sample graph: A->B:1, B->C:2, A->C:4. Direct path costs 4, multi-hop via B costs 3.",
        ),
        (
            "Benchmark 2: Diamond Graph (Direct Expensive vs Cheap Multi-Hop)",
            DIAMOND_BENCHMARK_GRAPH,
            "S",
            "G",
            "Demonstrates UCS prioritizing lower cumulative cost (S->A->C->G = 4) over fewer hops (S->B->G = 7).",
        ),
        (
            "Benchmark 3: AIMA Romania Map Subgraph (Arad to Bucharest)",
            ROMANIA_MAP_SUBGRAPH,
            "Arad",
            "Bucharest",
            "Classical AI navigation benchmark finding the optimal 418 km driving route across Romania.",
        ),
        (
            "Benchmark 4: Cyclic Weighted Graph (Cycle Avoidance)",
            CYCLIC_WEIGHTED_GRAPH,
            "A",
            "D",
            "Ensures search handles positive directed cycles cleanly without getting stuck in infinite loops.",
        ),
        (
            "Benchmark 5: Disconnected Weighted Graph (Unreachable Destination)",
            DISCONNECTED_WEIGHTED_GRAPH,
            "A",
            "E",
            "Tests graceful termination when the goal node exists in an isolated component.",
        ),
    ]

    for title, graph, start, goal, desc in benchmarks:
        print(f"\n>>> {title}")
        print(f"Description: {desc}")
        result = uniform_cost_search(graph, start=start, goal=goal)
        display_ucs_run(result, show_table=True)


def prompt_user_interactive() -> Tuple[Dict[str, List[Tuple[str, Union[int, float]]]], str, str]:
    """Prompt the user interactively in the console to enter a weighted graph, start, and goal.

    Returns:
        3-tuple (normalized_graph, start_node, goal_node).
    """
    print_banner("INTERACTIVE UNIFORM COST SEARCH (UCS) CONSOLE")
    print("Enter your weighted graph representation.")
    print("Supported Formats:")
    print("  1. Arrow format (Sample):  A->B:1, B->C:2, A->C:4")
    print("  2. Adjacency list format:  A: B:1, C:4; B: C:2; C:")
    print("  3. JSON dict format:       {\"A\": [[\"B\", 1], [\"C\", 4]], \"B\": [[\"C\", 2]], \"C\": []}")
    print()

    default_graph_str = "A->B:1, B->C:2, A->C:4"
    graph_input = input(f"Graph specification [press Enter for '{default_graph_str}']: ").strip()
    if not graph_input:
        graph_input = default_graph_str

    parsed_graph = parse_weighted_graph(graph_input)
    available_nodes = sorted(parsed_graph.keys())
    print(f"\nDiscovered Graph Nodes ({len(available_nodes)}): {', '.join(available_nodes)}")

    default_start = "A" if "A" in available_nodes else available_nodes[0]
    start_input = input(f"Start node [press Enter for '{default_start}']: ").strip()
    start_node = start_input if start_input else default_start

    default_goal = "C" if "C" in available_nodes else available_nodes[-1]
    goal_input = input(f"Goal destination node [press Enter for '{default_goal}']: ").strip()
    goal_node = goal_input if goal_input else default_goal

    return parsed_graph, start_node, goal_node


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main CLI entry point for Uniform Cost Search."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_uniform_cost_search.py",
        description="Uniform Cost Search (UCS) on Weighted Directed Graphs (Question 14)",
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="A->B:1, B->C:2, A->C:4",
        help="Weighted graph specification (arrow format, JSON string, or semicolon adjacency list).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="A",
        help="Origin start node identifier (default: 'A').",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="C",
        help="Target goal node identifier (default: 'C').",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=["lexicographical", "fifo", "hops"],
        default="lexicographical",
        help="Tie-breaking rule when cumulative path costs match: 'lexicographical' (default), 'fifo', or 'hops'.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console prompting for graph, start, and goal.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated benchmarks across 5 standard graph topologies.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge demonstration (negative edges and negative cycle detection).",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display Viva Voce theory questions, model answers, and algorithm comparison matrix.",
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILEPATH",
        help="Generate and save graphical NetworkX/Matplotlib diagram to specified PNG file path.",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print_viva_guide()
        return

    if args.challenge:
        run_challenge_demo()
        return

    if args.demo:
        run_all_benchmarks()
        return

    if args.interactive:
        graph, start_node, goal_node = prompt_user_interactive()
    else:
        graph = parse_weighted_graph(args.graph)
        start_node = args.start
        goal_node = args.goal

    result = uniform_cost_search(
        graph=graph,
        start=start_node,
        goal=goal_node,
        tie_breaker=args.tie_breaker,
    )
    display_ucs_run(result, show_table=True)

    if args.save:
        saved_file = visualize_ucs(result, output_path=args.save)
        print(f"[Success] Visual graph saved to: {saved_file}")


if __name__ == "__main__":
    main()
