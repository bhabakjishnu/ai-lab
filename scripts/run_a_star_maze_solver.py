"""Command-Line Interface and Runner for A* Pathfinding in 2D Maze (Question 29).

Practical Type: Simulation
Difficulty: 🟠 Advanced
Syllabus Area: Informed Search and Heuristics

Problem Statement:
    Implement A* pathfinding for a 2D grid containing obstacles. The agent can move
    Up, Down, Left, and Right. Step cost is 1. The program must use Manhattan distance
    for h(n) and plot the explored nodes and final path in the terminal.

Task Requirements:
    - 2D grid parsing.
    - f(n) = g(n) + manhattan(current, goal).
    - Track expanded nodes.
    - Render the maze with 'X' for obstacles, '.' for open, '*' for path, and '#' for explored.

Usage Examples:
    # 1. Default Run (Formatted overview and ASCII visualization):
    python scripts/run_a_star_maze_solver.py

    # 2. Exact Syllabus Sample Output (Pure ASCII Grid):
    python scripts/run_a_star_maze_solver.py --sample

    # 3. Optional Challenge: Animate node expansion in terminal:
    python scripts/run_a_star_maze_solver.py --animate
    python scripts/run_a_star_maze_solver.py --animate --delay 0.05

    # 4. Compare A* vs. BFS (Explored set size and optimality):
    python scripts/run_a_star_maze_solver.py --compare

    # 5. Step-by-Step Priority Queue Trace Table:
    python scripts/run_a_star_maze_solver.py --table

    # 6. University Viva Voce Preparation Guide:
    python scripts/run_a_star_maze_solver.py --viva

    # 7. Run Multiple Benchmark Mazes (5x5, 10x10, Concave Trap, Serpentine):
    python scripts/run_a_star_maze_solver.py --benchmarks

    # 8. Custom Grid via Command Line:
    python scripts/run_a_star_maze_solver.py --grid ". . X\n. . .\n. X ." --start 0,0 --goal 2,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a_star_maze_solver import (
    CONCAVE_TRAP_MAZE,
    DEFAULT_5X5_GOAL,
    DEFAULT_5X5_MAZE,
    DEFAULT_5X5_START,
    DEFAULT_10X10_GOAL,
    DEFAULT_10X10_MAZE,
    DEFAULT_10X10_START,
    OPEN_5X5_MAZE,
    SERPENTINE_5X5_MAZE,
    VIVA_VOCE_GUIDE,
    AStarMazeComparisonResult,
    AStarMazeResult,
    a_star_maze_solver,
    animate_a_star_search,
    compare_astar_vs_bfs,
    parse_grid,
    render_side_by_side_comparison,
    render_solved_maze,
)


def parse_coord(coord_str: str) -> Tuple[int, int]:
    """Parse coordinate string formatted as 'r,c' or '(r,c)'."""
    clean = coord_str.strip().strip("()[]")
    parts = clean.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{coord_str}'. Expected 'r,c' (e.g. '0,0').")
    return int(parts[0].strip()), int(parts[1].strip())


def run_sample_output(quiet: bool = False) -> str:
    """Print the exact syllabus sample output for Question 29: ASCII Grid output."""
    res = a_star_maze_solver(
        maze=DEFAULT_5X5_MAZE,
        start=DEFAULT_5X5_START,
        goal=DEFAULT_5X5_GOAL,
        tie_breaker="fifo",
    )
    rendered = render_solved_maze(
        maze=DEFAULT_5X5_MAZE,
        path=res.path,
        explored=res.explored_set,
        start=DEFAULT_5X5_START,
        goal=DEFAULT_5X5_GOAL,
        preserve_endpoints=False,
        show_coordinates=False,
    )
    if not quiet:
        print(rendered)
    return rendered


def run_grid_visualization(
    maze: Optional[List[List[int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    show_coordinates: bool = True,
    preserve_endpoints: bool = False,
) -> None:
    """Render the solved maze with detailed pathfinding metrics."""
    target_maze = maze or DEFAULT_5X5_MAZE
    s = start or DEFAULT_5X5_START
    g = goal or DEFAULT_5X5_GOAL

    res = a_star_maze_solver(
        maze=target_maze,
        start=s,
        goal=g,
        record_steps=True,
    )

    print("=" * 64)
    print("A* PATHFINDING IN 2D MAZE: ASCII VISUALIZATION")
    print("=" * 64)
    print(f"Algorithm       : {res.algorithm} (Evaluation: f(n) = g(n) + h(n))")
    print(f"Heuristic       : Manhattan Distance (L1 norm)")
    print(f"Maze Dimensions : {len(target_maze)} rows x {len(target_maze[0])} columns")
    print(f"Start / Goal    : {res.start} -> {res.goal}")
    print(f"Solution Found  : {'YES' if res.found else 'NO'}")
    print(f"Optimal Steps   : {res.path_length} transitions ({len(res.path)} coordinates)")
    print(f"Nodes Expanded  : {res.nodes_expanded} nodes")
    print(f"Explored Set    : {len(res.explored_set)} total visited coordinates")
    print(f"Peak Frontier   : {res.peak_frontier} entries in min-heap")
    print(f"Execution Time  : {res.execution_time_ms:.3f} ms")
    print("-" * 64)
    print("Legend: [X] Obstacle  [.] Open Walkable  [*] Solution Path  [#] Explored")
    print("-" * 64)

    rendered = render_solved_maze(
        maze=target_maze,
        path=res.path,
        explored=res.explored_set,
        start=s,
        goal=g,
        preserve_endpoints=preserve_endpoints,
        show_coordinates=show_coordinates,
    )
    print(rendered)
    print("=" * 64)
    path_str = " -> ".join(str(pt) for pt in res.path)
    print(f"Optimal Path Trajectory:\n{path_str}")


def run_comparative_analysis(
    maze: Optional[List[List[int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
) -> None:
    """Compare A* against BFS and Greedy Best-First Search on the maze."""
    target_maze = maze or DEFAULT_5X5_MAZE
    s = start or DEFAULT_5X5_START
    g = goal or DEFAULT_5X5_GOAL

    comp = compare_astar_vs_bfs(target_maze, start=s, goal=g, include_gbfs=True)
    astar = comp.astar_result
    bfs = comp.bfs_result
    gbfs = comp.gbfs_result

    print("=" * 72)
    print("SEARCH ALGORITHM COMPARISON: A* vs. BFS vs. GREEDY BEST-FIRST")
    print("=" * 72)
    print(f"{'Metric':<28} | {'A* Pathfinding':<16} | {'BFS (Blind)':<16} | {'Greedy BFS'}")
    print("-" * 72)
    print(f"{'Evaluation Strategy':<28} | {'f = g + h':<16} | {'f = depth':<16} | {'f = h'}")
    print(f"{'Heuristic Function':<28} | {'Manhattan':<16} | {'None':<16} | {'Manhattan'}")
    print(f"{'Nodes Expanded':<28} | {f'{astar.nodes_expanded} nodes':<16} | {f'{bfs.nodes_expanded} nodes':<16} | {f'{gbfs.nodes_expanded if gbfs else -1} nodes'}")
    print(f"{'Explored Set Size':<28} | {f'{len(astar.explored_set)} cells':<16} | {f'{len(bfs.explored_set)} cells':<16} | {f'{len(gbfs.explored_set) if gbfs else -1} cells'}")
    print(f"{'Solution Path Length':<28} | {f'{astar.path_length} steps':<16} | {f'{bfs.path_length} steps':<16} | {f'{gbfs.path_length if gbfs else -1} steps'}")
    print(f"{'Path Optimality':<28} | {'OPTIMAL':<16} | {'OPTIMAL':<16} | {'SUBOPTIMAL' if (gbfs and gbfs.path_length > astar.path_length) else 'OPTIMAL'}")
    print(f"{'Execution Time':<28} | {f'{astar.execution_time_ms:.3f} ms':<16} | {f'{bfs.execution_time_ms:.3f} ms':<16} | {f'{gbfs.execution_time_ms if gbfs else 0:.3f} ms'}")
    print("-" * 72)
    print(f"Nodes Explored Reduction vs BFS : {comp.expansion_difference_bfs} fewer nodes ({comp.expansion_reduction_pct_bfs:.1f}% reduction)")
    print(f"Path Length Difference vs BFS   : {comp.path_length_difference_bfs} steps (Strictly Optimal)")
    print("=" * 72)
    print("\nVisual Side-by-Side Comparison (A* vs. BFS):")
    print(render_side_by_side_comparison(comp))


def run_step_trace(
    maze: Optional[List[List[int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
) -> None:
    """Display detailed step-by-step priority queue trace table."""
    target_maze = maze or DEFAULT_5X5_MAZE
    s = start or DEFAULT_5X5_START
    g = goal or DEFAULT_5X5_GOAL

    res = a_star_maze_solver(
        target_maze,
        start=s,
        goal=g,
        record_steps=True,
    )

    print("=" * 86)
    print("A* PATHFINDING: STEP-BY-STEP PRIORITY QUEUE EXECUTION TRACE TABLE")
    print("=" * 86)
    print(
        f"{'Step':<5} | {'Node (r,c)':<10} | {'g(n)':<5} | {'h(n)':<5} | {'f(n)':<5} | "
        f"{'Action':<14} | {'Queue':<6} | {'Evaluated / Enqueued'}"
    )
    print("-" * 86)
    for st in res.steps:
        eval_str = f"eval={len(st.neighbors_evaluated)}, push={len(st.neighbors_enqueued)}"
        print(
            f"{st.step_number:<5} | {str(st.current_coord):<10} | "
            f"{st.g_cost:<5} | {st.h_cost:<5} | {st.f_cost:<5} | "
            f"{st.action:<14} | {st.frontier_size:<6} | {eval_str}"
        )
    print("=" * 86)
    print(f"Search Completed in {len(res.steps)} steps. Total Nodes Expanded: {res.nodes_expanded}")
    print(f"Optimal Path: {' -> '.join(str(pt) for pt in res.path)}")


def run_benchmarks() -> None:
    """Execute A* pathfinding on multiple benchmark maze topologies."""
    benchmarks = [
        ("Canonical 5x5 Grid (Syllabus Default)", DEFAULT_5X5_MAZE, DEFAULT_5X5_START, DEFAULT_5X5_GOAL),
        ("Benchmark 10x10 Maze", DEFAULT_10X10_MAZE, DEFAULT_10X10_START, DEFAULT_10X10_GOAL),
        ("Concave Trap 10x10 Maze (U-Obstacle)", CONCAVE_TRAP_MAZE, (0, 0), (9, 9)),
        ("Serpentine 5x5 Corridor Maze", SERPENTINE_5X5_MAZE, (0, 0), (4, 4)),
        ("Open 5x5 Obstacle-Free Grid", OPEN_5X5_MAZE, (0, 0), (4, 4)),
    ]

    print("=" * 80)
    print("A* 2D MAZE PATHFINDING: MULTI-BENCHMARK SUITE")
    print("=" * 80)

    for name, grid, start, goal in benchmarks:
        astar_res = a_star_maze_solver(grid, start=start, goal=goal)
        bfs_res = compare_astar_vs_bfs(grid, start=start, goal=goal)
        print(f"\n[BENCHMARK] {name}")
        print(f"Dimensions : {len(grid)}x{len(grid[0])} | Start: {start} | Goal: {goal}")
        print(
            f"A* Result  : Length = {astar_res.path_length} steps, "
            f"Expanded = {astar_res.nodes_expanded} nodes, Explored = {len(astar_res.explored_set)} cells"
        )
        print(
            f"Comparison : BFS Expanded = {bfs_res.bfs_result.nodes_expanded} nodes "
            f"({bfs_res.expansion_reduction_pct_bfs:.1f}% reduction by A*)"
        )
        print("Visual Grid Output:")
        print(render_solved_maze(grid, astar_res.path, astar_res.explored_set))
        print("-" * 80)


def run_viva_guide() -> None:
    """Display comprehensive Viva Voce preparation guide."""
    print("=" * 80)
    print("QUESTION NO. 29: A* PATHFINDING IN 2D MAZE - VIVA VOCE REFERENCE GUIDE")
    print("=" * 80)
    print("Syllabus Area : Informed Search and Heuristics")
    print("Difficulty    : 🟠 Advanced | Importance: Very High")
    print("-" * 80)

    for topic_title, content in VIVA_VOCE_GUIDE.items():
        print(f"\n### TOPIC: {topic_title.upper()}")
        print(f"Q: {content['question']}")
        print(f"A:\n{content['answer']}")
        print("-" * 80)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI dispatcher for A* 2D Maze Pathfinding."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_a_star_maze_solver.py",
        description="A* Pathfinding in 2D Maze Simulation (Question 29).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print exact syllabus sample output (ASCII grid output on 5x5 maze).",
    )
    parser.add_argument(
        "--grid",
        nargs="?",
        const="",
        default=None,
        help="Render ASCII visualization of solved maze. Optionally supply custom grid string.",
    )
    parser.add_argument(
        "--animate",
        "--challenge",
        action="store_true",
        help="Optional Challenge: Live terminal animation of node expansion and path reconstruction.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.08,
        help="Animation delay in seconds between frames (default: 0.08s).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare A* vs BFS and GBFS (explored set size and solution optimality).",
    )
    parser.add_argument(
        "--table",
        "--trace",
        action="store_true",
        help="Display step-by-step priority queue trace table.",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run benchmark suite across multiple maze topologies.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display University Viva Voce theory guide.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Starting coordinate 'row,col' (default: 0,0).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal coordinate 'row,col' (default: bottom-right).",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=["fifo", "lifo", "goal_directed"],
        default="fifo",
        help="Tie-breaking policy when f(n) values are identical (default: fifo).",
    )
    parser.add_argument(
        "--show-coords",
        action="store_true",
        help="Display numerical row and column headers on rendered grid.",
    )

    args = parser.parse_args(argv)

    # 1. Exact sample output mode
    if args.sample:
        run_sample_output()
        return

    # Parse custom coordinates if supplied
    custom_start = parse_coord(args.start) if args.start else None
    custom_goal = parse_coord(args.goal) if args.goal else None

    # Parse custom grid if supplied
    parsed_maze = None
    if args.grid and args.grid.strip():
        parsed_maze, detected_s, detected_g = parse_grid(
            args.grid,
            default_start=custom_start,
            default_goal=custom_goal,
        )
        if custom_start is None:
            custom_start = detected_s
        if custom_goal is None:
            custom_goal = detected_g

    # 2. Terminal Animation Mode (Optional Challenge)
    if args.animate:
        target_grid = parsed_maze or DEFAULT_5X5_MAZE
        target_s = custom_start or DEFAULT_5X5_START
        target_g = custom_goal or DEFAULT_5X5_GOAL
        animate_a_star_search(
            maze=target_grid,
            start=target_s,
            goal=target_g,
            delay=args.delay,
        )
        return

    # 3. Comparative Analysis Mode
    if args.compare:
        run_comparative_analysis(
            maze=parsed_maze,
            start=custom_start,
            goal=custom_goal,
        )
        return

    # 4. Step Trace Table Mode
    if args.table:
        run_step_trace(
            maze=parsed_maze,
            start=custom_start,
            goal=custom_goal,
        )
        return

    # 5. Benchmarks Suite Mode
    if args.benchmarks:
        run_benchmarks()
        return

    # 6. Viva Guide Mode
    if args.viva:
        run_viva_guide()
        return

    # 7. Default Mode: Formatted Grid Visualization & Overview
    run_grid_visualization(
        maze=parsed_maze,
        start=custom_start,
        goal=custom_goal,
        show_coordinates=args.show_coords,
    )


if __name__ == "__main__":
    main()
