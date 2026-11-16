"""Runner script for Multi-Goal BFS Maze Solver (Question 23: Maze with Multiple Exits).

Syllabus Area: Uninformed Search Techniques
Difficulty: 🟡 Intermediate
Practical Type: Algorithm Modification
Problem Statement: Modify a standard Breadth-First Search algorithm to solve a maze
that has multiple possible exit points (goals). The algorithm must stop and return
the path to the first exit it encounters, which is guaranteed to be the closest exit.

Usage Examples:
    # 1. Default execution on sample maze (Exits: (5,5), (1,8); Start: (0,0)):
    python scripts/run_multi_goal_bfs_maze.py

    # 2. Exact sample output only:
    python scripts/run_multi_goal_bfs_maze.py --sample

    # 3. Optional Challenge: Return paths to ALL exits sorted by distance:
    python scripts/run_multi_goal_bfs_maze.py --all-exits

    # 4. Viva Voce Topic 1: Benchmark set vs list membership efficiency:
    python scripts/run_multi_goal_bfs_maze.py --benchmark-set

    # 5. Viva Voce Topic 2: Compare BFS vs UCS on multiple goals:
    python scripts/run_multi_goal_bfs_maze.py --compare-ucs

    # 6. Display ASCII maze solution:
    python scripts/run_multi_goal_bfs_maze.py --ascii

    # 7. Step-by-step terminal search animation:
    python scripts/run_multi_goal_bfs_maze.py --animate --delay 0.05

    # 8. University Viva Voce examination guide:
    python scripts/run_multi_goal_bfs_maze.py --viva

    # 9. Machine-readable JSON output:
    python scripts/run_multi_goal_bfs_maze.py --json

    # 10. Comprehensive laboratory demonstration:
    python scripts/run_multi_goal_bfs_maze.py --demo
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.multi_goal_bfs_maze import (
    DETOUR_MULTI_EXIT_MAZE,
    PARTIALLY_BLOCKED_MULTI_EXIT_MAZE,
    SAMPLE_EXITS_Q23,
    SAMPLE_MULTI_EXIT_MAZE_6X10,
    SAMPLE_START_Q23,
    Coordinate,
    MultiGoalBFSResult,
    MultiGoalBFSSolver,
    SingleExitPathResult,
    benchmark_set_vs_list_lookup,
    compare_multi_goal_bfs_vs_ucs,
    find_all_exits_sorted,
    find_closest_exit,
    get_multi_goal_bfs_viva_reference,
    parse_ascii_maze,
    solve_multi_goal_bfs,
    visualize_multi_goal_maze,
)


def print_banner(title: str) -> None:
    """Print a styled section header."""
    print()
    print("=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def run_sample_only() -> None:
    """Print the exact sample output as specified in Question 23 prompt."""
    exit_coord, steps = find_closest_exit(
        SAMPLE_MULTI_EXIT_MAZE_6X10,
        SAMPLE_START_Q23,
        SAMPLE_EXITS_Q23,
    )
    # Sample Output format: Closest exit found at (1,8) in 9 steps.
    print(f"Closest exit found at ({exit_coord[0]},{exit_coord[1]}) in {steps} steps.")


def run_default(
    show_ascii: bool = True,
    find_all: bool = False,
    plot: bool = False,
) -> None:
    """Run default Multi-Goal BFS execution on the canonical Question 23 benchmark."""
    print_banner("Question 23: Maze with Multiple Exits (Multi-Goal BFS)")
    print(f"Start Coordinate : {SAMPLE_START_Q23}")
    print(f"Target Exits     : {SAMPLE_EXITS_Q23}")
    print(f"Maze Dimensions  : {len(SAMPLE_MULTI_EXIT_MAZE_6X10)}x{len(SAMPLE_MULTI_EXIT_MAZE_6X10[0])}")

    result = solve_multi_goal_bfs(
        SAMPLE_MULTI_EXIT_MAZE_6X10,
        SAMPLE_START_Q23,
        SAMPLE_EXITS_Q23,
        find_all=find_all,
        record_trace=True,
    )

    print()
    print("--- SAMPLE OUTPUT REQUIREMENT ---")
    if result.closest_exit:
        cr, cc = result.closest_exit
        print(f"Closest exit found at ({cr},{cc}) in {result.path_length} steps.")
    else:
        print("No reachable exit found.")

    print()
    print("--- EXECUTION DETAILS ---")
    print(f"Status           : {'SUCCESS' if result.found else 'FAILED'}")
    print(f"Shortest Path    : {' -> '.join(str(c) for c in result.path)}")
    print(f"Action Moves     : {result.actions_str}")
    print(f"Nodes Expanded   : {result.nodes_expanded}")
    print(f"Nodes Visited    : {result.nodes_visited}")
    print(f"Execution Time   : {result.execution_time_ms:.4f} ms")

    if find_all and result.all_exits_sorted:
        print()
        print("--- OPTIONAL CHALLENGE: ALL EXITS SORTED BY DISTANCE ---")
        for rank, exit_res in enumerate(result.all_exits_sorted, 1):
            status = f"{exit_res.path_length} steps" if exit_res.found else "UNREACHABLE"
            marker = " [CLOSEST]" if exit_res.exit_coord == result.closest_exit else ""
            print(f"  {rank}. Exit {exit_res.exit_coord}: {status}{marker}")
            if exit_res.found:
                print(f"     Path : {' -> '.join(str(c) for c in exit_res.path)}")
                print(f"     Moves: {exit_res.actions_str}")

    if show_ascii:
        print()
        print("--- ASCII MAZE SOLUTION ---")
        print("Glyphs: S = Start, E* = Closest Exit, E = Other Exit, * = Path, # = Wall, . = Open")
        print(result.render_ascii_solution(show_all_paths=find_all))

    if plot:
        img_path = visualize_multi_goal_maze(result)
        if img_path:
            print(f"\n[Visualization Saved] {img_path}")


def run_benchmark_set() -> None:
    """Run Viva Topic 1 empirical benchmark: Set vs List membership testing."""
    print_banner("Viva Voce Topic 1: Testing for Multiple Goals Efficiently")
    print("Evaluating average lookup performance: Set Hash Table O(1) vs List Scan O(K)...")

    bench = benchmark_set_vs_list_lookup(num_goals=1000, num_queries=50000)
    print(f"\nBenchmark Parameters:")
    print(f"  Candidate Goals Collection Size : {bench['num_goals']} exits")
    print(f"  Membership Queries Evaluated    : {bench['num_queries']:,} queries")
    print(f"\nTiming Results:")
    print(f"  List Lookup Time (O(k))         : {bench['list_time_ms']:.2f} ms")
    print(f"  Set Lookup Time (O(1))          : {bench['set_time_ms']:.2f} ms")
    print(f"  Speedup Factor                  : {bench['speedup_factor']:.1f}x FASTER")
    print(f"\nComplexity Comparison:")
    print(f"  List Membership Complexity      : {bench['list_complexity']}")
    print(f"  Set Membership Complexity       : {bench['set_complexity']}")
    print(f"\nTheoretical Takeaway:")
    print(f"  {bench['explanation']}")


def run_compare_ucs() -> None:
    """Run Viva Topic 2: Comparative analysis of BFS vs UCS on multiple goals."""
    print_banner("Viva Voce Topic 2: Does UCS Still Work with Multiple Goals?")
    print("Evaluating scenario with non-uniform step costs (Swamp=10 vs Road=1)...")

    comparison = compare_multi_goal_bfs_vs_ucs()
    print(f"\nScenario Setup:")
    print(f"  Start Node          : {comparison['start']}")
    print(f"  Candidate Exits     : {comparison['exits']}")
    print(f"    - Exit A (0,2)    : 2 steps away through Deep Swamp (cost 10 each, total cost = 20)")
    print(f"    - Exit B (1,3)    : 4 steps away through Paved Road (cost 1 each, total cost = 4)")
    print(f"\nResults Comparison:")
    print(f"  Multi-Goal BFS Result:")
    print(f"    - Chosen Exit     : {comparison['bfs_chosen_exit']}")
    print(f"    - Metric          : {comparison['bfs_metric']}")
    print(f"    - Steps (Hops)    : {comparison['bfs_steps']} steps")
    print(f"    - Total Cost      : {comparison['bfs_cost']}")
    print(f"  Multi-Goal UCS Result:")
    print(f"    - Chosen Exit     : {comparison['ucs_chosen_exit']}")
    print(f"    - Metric          : {comparison['ucs_metric']}")
    print(f"    - Steps (Hops)    : {comparison['ucs_steps']} steps")
    print(f"    - Total Cost      : {comparison['ucs_cost']}")
    print(f"\nKey Takeaways:")
    print(f"  UCS Compatible with Multiple Goals? : {'YES' if comparison['ucs_compatible'] else 'NO'}")
    print(f"  Viva Summary: {comparison['viva_takeaway']}")


def run_viva() -> None:
    """Display comprehensive University Viva Voce reference questions and answers."""
    viva_data = get_multi_goal_bfs_viva_reference()
    print_banner(f"University Viva Voce Guide: {viva_data['title']}")
    print(f"Syllabus Area : {viva_data['syllabus_area']}")
    print(f"Difficulty    : {viva_data['difficulty']}")
    print(f"Core Concept  : {viva_data['core_concept']}")
    print()

    for idx, q_item in enumerate(viva_data["questions"], 1):
        print(f"Q{idx}. {q_item['question']}")
        print(f"Ans: {q_item['answer']}")
        print("-" * 70)


def run_terminal_animation(delay: float = 0.05) -> None:
    """Animate the search expansion step-by-step in the terminal."""
    print_banner("Interactive Terminal Search Animation")
    solver = MultiGoalBFSSolver()
    gen = solver.step_generator(
        SAMPLE_MULTI_EXIT_MAZE_6X10,
        SAMPLE_START_Q23,
        SAMPLE_EXITS_Q23,
    )

    step_count = 0
    try:
        for record in gen:
            step_count += 1
            exit_flag = " [EXIT FOUND!]" if record.is_exit else ""
            print(
                f"\rStep {record.step_number:3d} | Popped: {record.current_coord} (dist: {record.distance}) "
                f"| Queue size: {len(record.frontier_snapshot):2d} | Visited: {record.visited_count:2d}{exit_flag}",
                end="",
                flush=True,
            )
            time.sleep(delay)
    except StopIteration:
        pass
    print(f"\nSearch concluded in {step_count} expansion steps.")


def run_demo() -> None:
    """Run comprehensive demonstration covering all features, edge cases, and Viva topics."""
    print_banner("Comprehensive Multi-Goal BFS Demonstration")

    # 1. Exact sample
    print("[1] Exact Sample Input & Output:")
    run_sample_only()

    # 2. Wall Detour Maze
    print("\n[2] Detour Demonstration (Physical Distance vs True Path Length):")
    res_detour = solve_multi_goal_bfs(
        DETOUR_MULTI_EXIT_MAZE,
        start=(0, 0),
        exits=[(2, 0), (0, 4)],
        find_all=True,
    )
    print(f"Start: (0,0) | Candidate Exits: Exit A (2,0) [Manhattan=2], Exit B (0,4) [Manhattan=4]")
    print(f"Closest Exit Chosen: {res_detour.closest_exit} in {res_detour.path_length} steps")
    print(f"Explanation: Exit A requires navigating around walls ({res_detour.all_exits_sorted[-1].path_length} steps), "
          f"while Exit B is open ({res_detour.path_length} steps).")

    # 3. Unreachable Exits
    print("\n[3] Handling Unreachable Exits:")
    res_unreach = solve_multi_goal_bfs(
        PARTIALLY_BLOCKED_MULTI_EXIT_MAZE,
        start=(0, 0),
        exits=[(0, 2), (2, 0)],
        find_all=True,
    )
    print(f"Start: (0,0) | Exits: [(0,2) [Sealed in Wall], (2,0) [Open]]")
    print(f"Closest Exit Found : {res_unreach.closest_exit} ({res_unreach.path_length} steps)")
    print(f"Unreachable Exits  : {res_unreach.unreachable_exits}")

    # 4. Viva Topic 1 Benchmark
    print("\n[4] Empirical Set vs List Benchmark:")
    run_benchmark_set()

    # 5. Viva Topic 2 Comparison with UCS
    print("\n[5] Comparison of Multi-Goal BFS vs Multi-Goal UCS:")
    run_compare_ucs()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main CLI entry point for Question 23 Multi-Goal BFS."""
    parser = argparse.ArgumentParser(
        description="Question 23: Maze with Multiple Exits (Multi-Goal BFS Solver)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print only the exact sample output format: 'Closest exit found at (1,8) in 9 steps.'",
    )
    parser.add_argument(
        "--all-exits",
        "--challenge",
        action="store_true",
        help="Execute Optional Challenge: Return paths to ALL exits sorted by distance.",
    )
    parser.add_argument(
        "--benchmark-set",
        action="store_true",
        help="Run Viva Voce Topic 1: Benchmark set vs list membership testing.",
    )
    parser.add_argument(
        "--compare-ucs",
        action="store_true",
        help="Run Viva Voce Topic 2: Compare Multi-Goal BFS vs Multi-Goal UCS on variable costs.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display University Viva Voce examination questions, answers, and concepts.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Display ASCII rendering of the maze and discovered solution path.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate step-by-step search progression in the terminal.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay in seconds between animation frames (default: 0.05).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a publication-quality Matplotlib visualization PNG.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON-serialized search result.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a comprehensive demonstration covering all features and Viva topics.",
    )

    args = parser.parse_args(argv)

    if args.sample:
        run_sample_only()
    elif args.benchmark_set:
        run_benchmark_set()
    elif args.compare_ucs:
        run_compare_ucs()
    elif args.viva:
        run_viva()
    elif args.animate:
        run_terminal_animation(delay=args.delay)
    elif args.demo:
        run_demo()
    elif args.json:
        res = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            find_all=args.all_exits,
        )
        print(json.dumps(res.to_dict(), indent=2))
    else:
        run_default(show_ascii=True, find_all=args.all_exits, plot=args.plot)


if __name__ == "__main__":
    main()
