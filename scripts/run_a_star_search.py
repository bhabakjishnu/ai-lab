"""Command-Line Interface and Runner for A* Search Algorithm on Graphs (Question 28).

Implements A* Search on a weighted graph with heuristic estimates h(n),
evaluating nodes via f(n) = g(n) + h(n), mirroring university examination standards
and Russell & Norvig's AIMA specifications.

Usage Examples:
    # 1. Default Canonical Sample Run:
    python scripts/run_a_star_search.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_a_star_search.py --sample

    # 3. Optional Challenge (Print frontier contents at each step):
    python scripts/run_a_star_search.py --challenge
    python scripts/run_a_star_search.py --frontier

    # 4. Detailed Step-by-Step Priority Queue Trace Table:
    python scripts/run_a_star_search.py --table

    # 5. AIMA Romania Map Benchmark (Arad to Bucharest: 418 km):
    python scripts/run_a_star_search.py --romania

    # 6. Reopening Closed Node Demonstration (Admissible but Inconsistent Heuristic):
    python scripts/run_a_star_search.py --reopen-demo

    # 7. Algorithm Comparison (A* vs. UCS vs. Greedy Best-First):
    python scripts/run_a_star_search.py --compare

    # 8. Verify Heuristic Admissibility & Consistency (Monotonicity):
    python scripts/run_a_star_search.py --check-heuristics

    # 9. University Viva Voce Preparation Guide:
    python scripts/run_a_star_search.py --viva

    # 10. Custom Graph & Heuristics via CLI:
    python scripts/run_a_star_search.py --graph "S: A=3, B=1, C=4; A: G=8; B: A=1; C: G=5; G:" --heuristics "S:7, A:6, B:7, C:4, G:0" --start S --goal G
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

from src.a_star_search import (
    CYCLIC_GRAPH,
    CYCLIC_HEURISTICS,
    DEFAULT_SAMPLE_GRAPH,
    DEFAULT_SAMPLE_HEURISTICS,
    DIAMOND_GRAPH,
    DIAMOND_HEURISTICS,
    DISCONNECTED_GRAPH,
    DISCONNECTED_HEURISTICS,
    REOPENING_DEMO_GRAPH,
    REOPENING_INCONSISTENT_HEURISTICS,
    ROMANIA_MAP_GRAPH,
    ROMANIA_SLD_HEURISTICS,
    AStarResult,
    AStarSearch,
    a_star_search,
    check_admissibility,
    check_consistency,
    compare_astar_vs_ucs_and_greedy,
    get_a_star_viva_reference,
    parse_graph,
    parse_heuristics,
    visualize_a_star,
)


def run_sample_output(quiet: bool = False) -> str:
    """Run the canonical syllabus sample input and return exact expected output string.

    Input:
        Graph: {'S': [('A', 3), ('B', 1), ('C', 4)], 'A': [('G', 8)], 'B': [('A', 1)], 'C': [('G', 5)], 'G': []}
        Heuristics: {'S': 7, 'A': 6, 'B': 7, 'C': 4, 'G': 0}
        Start: 'S', Goal: 'G'

    Expected Output:
        Optimal Path: S -> C -> G, Total Cost: 9
    """
    res = a_star_search(
        DEFAULT_SAMPLE_GRAPH,
        DEFAULT_SAMPLE_HEURISTICS,
        start="S",
        goal="G",
    )
    formatted = res.format_path()
    if not quiet:
        print(formatted)
    return formatted


def display_frontier_challenge(res: Optional[AStarResult] = None) -> None:
    """Execute and display the Optional Challenge (Frontier contents at each step)."""
    if res is None:
        res = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
    print("\n" + "=" * 78)
    print("OPTIONAL CHALLENGE: Print Frontier (Priority Queue) Contents at Each Step")
    print("=" * 78)
    print(res.format_frontier_at_steps())
    print("-" * 78)
    print(f"Outcome: {res.format_path()}")
    print(f"Nodes Expanded: {' -> '.join(res.nodes_expanded)}")
    print(f"Re-evaluations with cheaper g_cost: {res.re_evaluations_count}")
    print("=" * 78 + "\n")


def display_step_table(res: Optional[AStarResult] = None) -> None:
    """Print the detailed ASCII step trace table for A* search."""
    if res is None:
        res = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
    print("\n" + "=" * 105)
    print("A* SEARCH: STEP-BY-STEP EXECUTION TRACE TABLE")
    print(f"Start: {res.start_node} | Goal: {res.goal_node} | {res.format_path()}")
    print("=" * 105)
    print(res.format_step_table())
    print("=" * 105 + "\n")


def display_romania_benchmark() -> None:
    """Execute and display the classical AIMA Romania Map benchmark (Arad to Bucharest)."""
    print("\n" + "=" * 78)
    print("BENCHMARK: Russell & Norvig Romania Map (Arad to Bucharest)")
    print("=" * 78)
    print("Start: Arad, Goal: Bucharest")
    print("Heuristics: Straight-Line Distance to Bucharest (h_SLD)")
    print("-" * 78)
    res = a_star_search(ROMANIA_MAP_GRAPH, ROMANIA_SLD_HEURISTICS, start="Arad", goal="Bucharest")
    print(res.format_path())
    print(f"Nodes Expanded in Order ({len(res.nodes_expanded)} nodes):")
    print(" -> ".join(res.nodes_expanded))
    print(f"Execution Time: {res.execution_time_ms:.3f} ms")
    print("=" * 78 + "\n")


def display_reopen_demo() -> None:
    """Execute and display the reopening demonstration with an inconsistent heuristic."""
    print("\n" + "=" * 78)
    print("DEMO: Reopening Closed Nodes (Admissible but Inconsistent Heuristic)")
    print("=" * 78)
    print("Topology: S -> A (cost 5), S -> B (cost 1), B -> A (cost 2), A -> G (cost 4)")
    print("Heuristics: S: 5, A: 0, B: 6, G: 0")
    print("  - Notice: f(A via S) = 5 + 0 = 5 < f(B via S) = 1 + 6 = 7.")
    print("  - Therefore, A is expanded FIRST and enters the closed set with g(A)=5.")
    print("  - Later, B is expanded, discovering path S -> B -> A with g(A)=3 < 5!")
    print("-" * 78)

    # 1. With Reopening Enabled:
    res_reopen = a_star_search(
        REOPENING_DEMO_GRAPH,
        REOPENING_INCONSISTENT_HEURISTICS,
        start="S",
        goal="G",
        allow_reopen=True,
    )
    print("[1] With Reopening Enabled (allow_reopen=True):")
    print(f"    Outcome: {res_reopen.format_path()} (OPTIMAL!)")
    print(f"    Closed Nodes Reopened: {res_reopen.nodes_reopened_count}")
    print(f"    Nodes Expanded: {' -> '.join(res_reopen.nodes_expanded)}")

    # 2. With Reopening Disabled:
    res_no_reopen = a_star_search(
        REOPENING_DEMO_GRAPH,
        REOPENING_INCONSISTENT_HEURISTICS,
        start="S",
        goal="G",
        allow_reopen=False,
    )
    print("\n[2] With Reopening Disabled (allow_reopen=False):")
    print(f"    Outcome: {res_no_reopen.format_path()} (SUBOPTIMAL / Missed cheaper route!)")
    print(f"    Nodes Expanded: {' -> '.join(res_no_reopen.nodes_expanded)}")
    print("=" * 78 + "\n")


def display_algorithm_comparison() -> None:
    """Execute and display comparison between A*, Uniform Cost Search, and Greedy Best-First."""
    print("\n" + "=" * 85)
    print("ALGORITHM COMPARISON: A* Search vs. Uniform Cost Search vs. Greedy Best-First Search")
    print("=" * 85)
    comp = compare_astar_vs_ucs_and_greedy(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS, start="S", goal="G")

    header = f"{'Algorithm':<28} | {'Evaluation Fn':<16} | {'Path Found':<20} | {'Cost':<5} | {'Expanded'}"
    print(header)
    print("-" * len(header))
    for key, name in [
        ("a_star", "A* Search"),
        ("uniform_cost_search", "Uniform Cost Search (UCS)"),
        ("greedy_best_first_search", "Greedy Best-First (GBFS)"),
    ]:
        data = comp[key]
        path_str = "->".join(data["path"]) if data["path"] else "None"
        exp_count = data["expansion_count"]
        print(f"{name:<28} | {data['evaluation']:<16} | {path_str:<20} | {data['total_cost']:<5} | {exp_count} nodes")
    print("=" * 85 + "\n")


def display_heuristic_check(
    graph: Optional[dict] = None,
    heuristics: Optional[dict] = None,
    goal: str = "G",
) -> None:
    """Check and display admissibility and consistency of heuristics."""
    g = graph if graph is not None else DEFAULT_SAMPLE_GRAPH
    h = heuristics if heuristics is not None else DEFAULT_SAMPLE_HEURISTICS

    print("\n" + "=" * 78)
    print("HEURISTIC VERIFICATION REPORT")
    print(f"Target Goal: '{goal}'")
    print("=" * 78)

    adm = check_admissibility(g, h, goal=goal)
    cons = check_consistency(g, h)

    print(f"1. Admissibility  : {'YES (h(n) <= h*(n) for all nodes)' if adm['is_admissible'] else 'NO (Violations detected!)'}")
    if not adm["is_admissible"]:
        for v in adm["violations"]:
            print(f"   - Node '{v['node']}': h={v['h']} > h*={v['h_star']} (Overestimates by {v['overestimation']})")

    print(f"2. Consistency (Monotonicity) : {'YES (Triangle inequality satisfied on all edges)' if cons['is_consistent'] else 'NO (Violations detected!)'}")
    if not cons["is_consistent"]:
        for v in cons["violations"]:
            print(f"   - Edge {v['edge']}: h(u)={v['h(u)']} > c(u,v)={v['c(u,v)']} + h(v)={v['h(v)']} (Deficit: {v['deficit']})")
    print("=" * 78 + "\n")


def display_viva_reference() -> None:
    """Print the complete university Viva Voce study guide."""
    viva = get_a_star_viva_reference()
    print("\n" + "=" * 78)
    print(f"🎓 {viva['title']}")
    print(f"📚 Syllabus Area: {viva['syllabus_topic']}")
    print("=" * 78)
    print("CORE EVALUATION FUNCTION:")
    print(f"  {viva['core_formula']['f_n']}")
    for comp, desc in viva["core_formula"]["components"].items():
        print(f"  • {comp:<5} : {desc}")
    print("-" * 78)
    print("KEY VIVA QUESTIONS & IN-DEPTH ANSWERS:")
    for idx, qa in enumerate(viva["questions_and_answers"], 1):
        print(f"\n[Q{idx}] {qa['question']}")
        print(f"Answer:\n{qa['answer']}")
    print("=" * 78 + "\n")


def run_interactive_mode() -> None:
    """Prompt the user interactively to configure graph, heuristics, start, and goal."""
    print("\n" + "=" * 75)
    print("Interactive Mode: A* Search Algorithm on Weighted Graph")
    print("=" * 75)
    print("Example Graph: S: A=3, B=1, C=4; A: G=8; B: A=1; C: G=5; G:")
    graph_input = input("Enter Graph [press Enter for default]: ").strip()
    graph = parse_graph(graph_input) if graph_input else DEFAULT_SAMPLE_GRAPH

    print("Example Heuristics: S: 7, A: 6, B: 7, C: 4, G: 0")
    h_input = input("Enter Heuristics [press Enter for default]: ").strip()
    heuristics = parse_heuristics(h_input) if h_input else DEFAULT_SAMPLE_HEURISTICS

    start_node = input("Enter Start Node [default 'S']: ").strip() or "S"
    goal_node = input("Enter Goal Node [default 'G']: ").strip() or "G"

    res = a_star_search(graph, heuristics, start=start_node, goal=goal_node)
    print("\n" + "=" * 75)
    print("SEARCH RESULT:")
    print(res.format_path())
    print("=" * 75)
    print("\nStep Table:")
    print(res.format_step_table())
    print("\nFrontier at each step:")
    print(res.format_frontier_at_steps())


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point for A* Search runner."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_a_star_search.py",
        description="A* Search Algorithm on a Weighted Graph (Question 28)",
    )
    parser.add_argument(
        "--format",
        choices=["sample", "canonical", "detailed", "json"],
        default="sample",
        help="Output presentation format (default: sample)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Output exact canonical sample output: 'Optimal Path: S -> C -> G, Total Cost: 9'",
    )
    parser.add_argument(
        "--challenge",
        "--frontier",
        action="store_true",
        dest="frontier_challenge",
        help="Optional Challenge: Print frontier (priority queue) contents at each step",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display detailed ASCII execution trace table of steps",
    )
    parser.add_argument(
        "--romania",
        action="store_true",
        help="Run classical AIMA Romania Map benchmark (Arad to Bucharest: 418 km)",
    )
    parser.add_argument(
        "--reopen-demo",
        action="store_true",
        help="Run reopening demonstration (admissible but inconsistent heuristic)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare A* vs Uniform Cost Search vs Greedy Best-First Search",
    )
    parser.add_argument(
        "--check-heuristics",
        action="store_true",
        help="Verify admissibility and consistency (monotonicity) of heuristics",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display complete University Viva Voce examination study guide",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive console input mode",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Custom graph definition (JSON, arrow notation, or semicolon notation)",
    )
    parser.add_argument(
        "--heuristics",
        type=str,
        default=None,
        help="Custom heuristic estimates (JSON or comma key-value pairs)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="S",
        help="Starting node identifier (default: 'S')",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="G",
        help="Goal node identifier (default: 'G')",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=["fifo", "alphabetical", "lifo"],
        default="fifo",
        help="Priority queue tie-breaker policy (default: fifo)",
    )
    parser.add_argument(
        "--no-reopen",
        action="store_true",
        help="Disable reopening of closed nodes upon discovering cheaper g_cost",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="FILE",
        help="Render and save graph visualization to specified image file (e.g. astar.png)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON summary",
    )

    args = parser.parse_args(argv)

    if args.interactive:
        run_interactive_mode()
        return 0

    if args.viva:
        display_viva_reference()
        return 0

    if args.reopen_demo:
        display_reopen_demo()
        return 0

    if args.romania:
        display_romania_benchmark()
        return 0

    if args.compare:
        display_algorithm_comparison()
        return 0

    if args.check_heuristics:
        display_heuristic_check(
            graph=parse_graph(args.graph) if args.graph else DEFAULT_SAMPLE_GRAPH,
            heuristics=parse_heuristics(args.heuristics) if args.heuristics else DEFAULT_SAMPLE_HEURISTICS,
            goal=args.goal,
        )
        return 0

    # Parse graph and heuristics if specified
    graph = parse_graph(args.graph) if args.graph else DEFAULT_SAMPLE_GRAPH
    heuristics = parse_heuristics(args.heuristics) if args.heuristics else DEFAULT_SAMPLE_HEURISTICS

    res = a_star_search(
        graph=graph,
        heuristics=heuristics,
        start=args.start,
        goal=args.goal,
        tie_breaking=args.tie_breaker,
        allow_reopen=not args.no_reopen,
    )

    if args.plot:
        saved = visualize_a_star(res, output_path=args.plot)
        if saved:
            print(f"Graph visualization saved to: {saved}")
        else:
            print("Visualization libraries (networkx/matplotlib) not available; skipping plot.")

    if args.json or args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.frontier_challenge:
        display_frontier_challenge(res)
        return 0

    if args.table or args.format == "detailed":
        display_step_table(res)
        return 0

    # Default / Sample run:
    print(res.format_path())
    return 0


if __name__ == "__main__":
    sys.exit(main())
