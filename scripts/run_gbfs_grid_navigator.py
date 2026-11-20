"""Command-Line Interface and Runner for Greedy Best-First Search Grid Navigator (Question 27).

Practical Type: Problem Solving
Difficulty: 🟠 Advanced
Syllabus Area: Informed Search and Heuristics

Demonstrates an agent navigating a 10x10 maze using Greedy Best-First Search with the
Manhattan distance heuristic. Compares Greedy BFS against Breadth-First Search (BFS) to
empirically demonstrate that Greedy search explores fewer nodes than BFS but might take
a suboptimal path.

Usage Examples:
    # 1. Default Run (Full output with comparison and path coordinates):
    python scripts/run_gbfs_grid_navigator.py

    # 2. Exact Syllabus Sample Output:
    python scripts/run_gbfs_grid_navigator.py --sample

    # 3. Optional Challenge (Output grid with path marked by '*'):
    python scripts/run_gbfs_grid_navigator.py --grid

    # 4. Comparative Benchmark (Greedy vs. BFS vs. A*):
    python scripts/run_gbfs_grid_navigator.py --compare

    # 5. University Viva Voce Preparation Guide:
    python scripts/run_gbfs_grid_navigator.py --viva

    # 6. Step-by-Step Priority Queue Trace Table:
    python scripts/run_gbfs_grid_navigator.py --trace

    # 7. Multi-Benchmark Evaluation (Default, Concave Trap, Open, Unsolvable):
    python scripts/run_gbfs_grid_navigator.py --benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gbfs_grid_navigator import (
    CONCAVE_TRAP_10X10_MAZE,
    DEFAULT_10X10_MAZE,
    DEFAULT_GOAL_COORD,
    DEFAULT_START_COORD,
    OPEN_10X10_MAZE,
    UNSOLVABLE_10X10_MAZE,
    GBFSGridComparisonResult,
    GBFSGridNavigator,
    GridSearchResult,
    compare_gbfs_vs_bfs,
    get_gbfs_grid_viva_reference,
    greedy_best_first_search_grid,
    render_grid_with_path,
    render_side_by_side_grids,
)


def parse_coord(coord_str: str) -> Tuple[int, int]:
    """Parse a coordinate string in 'r,c' or '(r,c)' format."""
    clean = coord_str.strip().strip("()[]")
    parts = clean.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{coord_str}'. Expected 'row,col' (e.g. 0,0).")
    return int(parts[0].strip()), int(parts[1].strip())


def run_sample_output() -> None:
    """Print the exact syllabus sample output for Question 27."""
    res = greedy_best_first_search_grid(
        DEFAULT_10X10_MAZE,
        DEFAULT_START_COORD,
        DEFAULT_GOAL_COORD,
    )
    # Compare with BFS to confirm suboptimality flag
    bfs_res = compare_gbfs_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
    res.is_suboptimal = bfs_res.is_gbfs_suboptimal
    print(res.format_sample_output())


def run_grid_visualization(maze: Optional[List[List[int]]] = None) -> None:
    """Execute Optional Challenge: Render 10x10 maze with path marked by '*'."""
    target_maze = maze or DEFAULT_10X10_MAZE
    res = greedy_best_first_search_grid(target_maze, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
    print("=" * 60)
    print("OPTIONAL CHALLENGE: 10x10 Maze Path Visualization ('*')")
    print("=" * 60)
    print(f"Algorithm      : {res.algorithm}")
    print(f"Start / Goal   : {res.start} -> {res.goal}")
    print(f"Nodes Expanded : {res.nodes_expanded}")
    print(f"Path Length    : {res.path_length} steps ({len(res.path)} coordinates)")
    print("-" * 60)
    print(render_grid_with_path(target_maze, res.path, start=res.start, goal=res.goal))
    print("=" * 60)
    print("Legend: 'S'=Start, 'G'=Goal, '*' = Solution Path, '#' = Wall, '.' = Open Cell")


def run_comparative_analysis(maze: Optional[List[List[int]]] = None) -> None:
    """Display comprehensive comparative analysis between GBFS and BFS."""
    target_maze = maze or DEFAULT_10X10_MAZE
    comp = compare_gbfs_vs_bfs(target_maze, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
    print(comp.format_summary_table())
    print("\nVisual Side-by-Side Path Comparison:")
    print(render_side_by_side_grids(target_maze, comp.gbfs_result, comp.bfs_result))


def run_step_trace(maze: Optional[List[List[int]]] = None) -> None:
    """Display detailed step-by-step expansion trace."""
    target_maze = maze or DEFAULT_10X10_MAZE
    res = greedy_best_first_search_grid(
        target_maze,
        DEFAULT_START_COORD,
        DEFAULT_GOAL_COORD,
        record_steps=True,
    )
    print("=" * 80)
    print("Greedy Best-First Search: Step-by-Step Priority Queue Trace")
    print("=" * 80)
    print(f"{'Step':<5} | {'Node':<8} | {'h(n)':<6} | {'Action':<14} | {'Queue':<6} | {'Evaluated / Enqueued'}")
    print("-" * 80)
    for step in res.steps:
        eval_str = f"eval={len(step.neighbors_evaluated)}, push={len(step.neighbors_enqueued)}"
        print(
            f"{step.step_number:<5} | {str(step.current_coord):<8} | "
            f"{step.heuristic_val:<6.1f} | {step.action:<14} | {step.frontier_size:<6} | {eval_str}"
        )
    print("=" * 80)
    print(f"Search Completed in {len(res.steps)} steps. Total Nodes Expanded: {res.nodes_expanded}")


def run_viva_guide() -> None:
    """Display complete university viva voce guide."""
    viva = get_gbfs_grid_viva_reference()
    print("=" * 78)
    print(viva["title"].upper())
    print("=" * 78)
    print("Core Viva Topics Covered:")
    for topic in viva["core_topics"]:
        print(f"  • {topic}")
    print("-" * 78)
    print("Frequently Asked Viva Questions & Model Answers:")
    for idx, item in enumerate(viva["qa_pairs"], start=1):
        print(f"\n[Q{idx}] {item['question']}")
        print(f"[Ans] {item['answer']}")
    print("=" * 78)


def run_multi_benchmarks() -> None:
    """Evaluate across multiple benchmark mazes."""
    benchmarks = [
        ("Default 10x10 Benchmark (Suboptimal Demo)", DEFAULT_10X10_MAZE, (0, 0), (9, 9)),
        ("Concave Trap (U-Shaped Obstacle)", CONCAVE_TRAP_10X10_MAZE, (0, 0), (9, 9)),
        ("Open Grid (Zero Obstacles)", OPEN_10X10_MAZE, (0, 0), (9, 9)),
        ("Unsolvable Grid (Severed Barrier)", UNSOLVABLE_10X10_MAZE, (0, 0), (9, 9)),
    ]

    print("=" * 82)
    print("GREEDY BEST-FIRST SEARCH: MULTI-BENCHMARK EVALUATION")
    print("=" * 82)
    print(f"{'Benchmark Scenario':<36} | {'GBFS Exp':<8} | {'GBFS Len':<8} | {'BFS Exp':<8} | {'BFS Len':<8} | {'Suboptimal?'}")
    print("-" * 82)

    for name, maze, start, goal in benchmarks:
        comp = compare_gbfs_vs_bfs(maze, start, goal)
        g_exp = comp.gbfs_result.nodes_expanded
        g_len = comp.gbfs_result.path_length if comp.gbfs_result.found else "N/A"
        b_exp = comp.bfs_result.nodes_expanded
        b_len = comp.bfs_result.path_length if comp.bfs_result.found else "N/A"
        subopt_flag = "YES" if comp.is_gbfs_suboptimal else "NO"
        print(f"{name:<36} | {str(g_exp):<8} | {str(g_len):<8} | {str(b_exp):<8} | {str(b_len):<8} | {subopt_flag}")
    print("=" * 82)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for Question 27."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_gbfs_grid_navigator.py",
        description="Question 27: Greedy Best-First Search Grid Navigator & Heuristic Analysis",
    )
    parser.add_argument(
        "--format",
        choices=["sample", "full", "grid", "json"],
        default="full",
        help="Output presentation format ('sample' produces exact syllabus string)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Shorthand for --format sample (exact syllabus output)",
    )
    parser.add_argument(
        "--grid",
        "--render",
        dest="grid",
        action="store_true",
        help="Optional Challenge: Render the 10x10 maze with path marked by '*'",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comprehensive comparative analysis: Greedy BFS vs. BFS vs. A*",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display university viva voce theory, questions, and model answers",
    )
    parser.add_argument(
        "--trace",
        "--table",
        dest="trace",
        action="store_true",
        help="Display step-by-step priority queue state trace table",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run evaluation across all predefined benchmark mazes",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="0,0",
        help="Starting coordinate 'row,col' (default: 0,0)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="9,9",
        help="Goal coordinate 'row,col' (default: 9,9)",
    )
    parser.add_argument(
        "--maze",
        type=str,
        default=None,
        help="JSON string or file path containing 2D maze matrix",
    )
    parser.add_argument(
        "--heuristic",
        choices=["manhattan", "euclidean", "chebyshev"],
        default="manhattan",
        help="Dynamic distance heuristic calculation metric (default: manhattan)",
    )

    args = parser.parse_args(argv)

    # Route specialized flags first
    if args.sample or args.format == "sample":
        run_sample_output()
        return 0

    if args.viva:
        run_viva_guide()
        return 0

    if args.benchmarks:
        run_multi_benchmarks()
        return 0

    # Parse custom maze if provided
    maze = DEFAULT_10X10_MAZE
    if args.maze:
        maze_path = Path(args.maze)
        if maze_path.is_file():
            with open(maze_path, "r", encoding="utf-8") as f:
                maze = json.load(f)
        else:
            maze = json.loads(args.maze)

    start_coord = parse_coord(args.start)
    goal_coord = parse_coord(args.goal)

    if args.trace:
        run_step_trace(maze)
        return 0

    if args.compare:
        run_comparative_analysis(maze)
        return 0

    if args.grid or args.format == "grid":
        run_grid_visualization(maze)
        return 0

    # Default / Full Execution
    comp = compare_gbfs_vs_bfs(maze, start_coord, goal_coord)
    gbfs_res = comp.gbfs_result

    if args.format == "json":
        print(json.dumps(gbfs_res.to_dict(), indent=2))
        return 0

    print("=" * 76)
    print("Greedy Best-First Search (GBFS) Grid Navigator (Question 27)")
    print("=" * 76)
    print(f"Maze Dimensions    : {len(maze)} x {len(maze[0])}")
    print(f"Start Coordinate   : {start_coord}")
    print(f"Goal Coordinate    : {goal_coord}")
    print(f"Heuristic Metric   : {args.heuristic.capitalize()} Distance (Dynamic)")
    print("-" * 76)
    print(f"Sample Output      : {gbfs_res.format_sample_output()}")
    print(f"Path Length        : {gbfs_res.path_length} steps")
    print(f"Nodes Expanded     : {gbfs_res.nodes_expanded}")
    print(f"Peak Frontier      : {gbfs_res.peak_frontier} nodes")
    print(f"Execution Time     : {gbfs_res.execution_time_ms:.3f} ms")
    print("-" * 76)
    print(f"Path Coordinates   :\n{gbfs_res.path}")
    print("-" * 76)
    print("Optional Challenge - Path Marked Grid:")
    print(render_grid_with_path(maze, gbfs_res.path, start=start_coord, goal=goal_coord))
    print("-" * 76)
    print(f"BFS Optimal Length : {comp.bfs_result.path_length} steps (expanded {comp.bfs_result.nodes_expanded} nodes)")
    print(
        f"Empirical Verdict  : Greedy explored {comp.expansion_difference} fewer nodes "
        f"({comp.expansion_reduction_pct:.1f}% reduction), but took a suboptimal path "
        f"(+{comp.path_length_difference} steps longer than BFS optimal)."
    )
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
