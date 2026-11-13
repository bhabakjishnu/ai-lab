"""Runner script for BFS vs DFS Performance Analyzer (Question 18).

Demonstrates and analyzes Breadth-First Search (BFS) versus Depth-First Search (DFS)
on a 10x10 maze, recording exact nodes expanded, path length, memory consumption,
and execution time, while maintaining identical neighbor ordering (Up, Down, Left, Right).
Highlights BFS shortest-path optimality versus DFS arbitrary/suboptimal pathfinding,
and explores viva scenarios where DFS expands fewer nodes.

Usage Examples:
    # 1. Default Sample Run (10x10 Maze):
    python scripts/run_bfs_dfs_performance_analyzer.py

    # 2. Display syllabus sample output format only:
    python scripts/run_bfs_dfs_performance_analyzer.py --format single

    # 3. Run DFS-Favored 10x10 Maze (DFS 14 expansions vs BFS 45 expansions):
    python scripts/run_bfs_dfs_performance_analyzer.py --maze dfs-favored

    # 4. Run Classic Corner-to-Corner 10x10 Maze (Start (0,0) -> Goal (9,9)):
    python scripts/run_bfs_dfs_performance_analyzer.py --maze corner

    # 5. Run Comprehensive Benchmark Suite across all predefined mazes:
    python scripts/run_bfs_dfs_performance_analyzer.py --demo

    # 6. Optional Challenge: Generate & Save Matplotlib Expansion Heatmaps:
    python scripts/run_bfs_dfs_performance_analyzer.py --save reports/bfs_vs_dfs_performance.png

    # 7. Display Viva Voce Theory Guide & Authoritative Answers:
    python scripts/run_bfs_dfs_performance_analyzer.py --viva

    # 8. Custom Start and Goal Coordinates:
    python scripts/run_bfs_dfs_performance_analyzer.py --start 0,0 --goal 9,9
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for direct script invocation
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bfs_dfs_performance_analyzer import (
    CORNER_10X10_MAZE,
    CORNER_GOAL_COORD,
    DEFAULT_10X10_MAZE,
    DEFAULT_DIRECTIONS,
    DEFAULT_GOAL_COORD,
    DEFAULT_START_COORD,
    DFS_FAVORED_10X10_MAZE,
    DFS_FAVORED_GOAL_COORD,
    OPEN_10X10_MAZE,
    SAMPLE_OUTPUT_TARGET_STRING,
    UNSOLVABLE_10X10_MAZE,
    ComparisonResult,
    compare_maze_solvers,
    get_viva_reference,
    parse_grid,
    plot_bfs_vs_dfs_expansion,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header."""
    width = 78
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def parse_coordinate(coord_str: str) -> Tuple[int, int]:
    """Parse a coordinate string in 'r,c' format into an integer tuple (r, c)."""
    parts = coord_str.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{coord_str}'. Expected 'row,col' (e.g. '0,0').")
    return int(parts[0].strip()), int(parts[1].strip())


def print_viva_guide() -> None:
    """Print the complete Viva Voce reference guide for BFS vs DFS Performance Analysis."""
    viva = get_viva_reference()
    print_banner("BFS vs DFS PERFORMANCE ANALYZER VIVA VOCE THEORY GUIDE")
    print(f"Syllabus Area: {viva['syllabus_area']}")
    print(f"Core Concept : {viva['core_concept']}\n")

    topics = viva["viva_topics"]

    print("[1] ALGORITHM OPTIMALITY:")
    opt = topics["algorithm_optimality"]
    print(f"  Question: {opt['question']}")
    print("  • BFS Optimality Proof:")
    print(f"    {opt['bfs_optimality_proof']}")
    print("  • DFS Non-Optimality / Arbitrary Pathfinding:")
    print(f"    {opt['dfs_non_optimality']}\n")

    print("[2] SCENARIOS WHERE DFS OUTPERFORMS BFS:")
    scen = topics["scenarios_where_dfs_outperforms_bfs"]
    print(f"  Question: {scen['question']}")
    for s in scen["scenarios"]:
        print(f"  • {s['name']}:")
        print(f"    {s['explanation']}")
    print()

    print("[3] BRANCHING FACTOR IMPACT:")
    bf = topics["branching_factor_impact"]
    print(f"  Question: {bf['question']}")
    print(f"  • Definition : {bf['definition']}")
    print(f"  • BFS Impact : {bf['impact_on_bfs']}")
    print(f"  • DFS Impact : {bf['impact_on_dfs']}\n")

    print("[4] COMPLEXITY COMPARISON MATRIX:")
    print(f"{'Algorithm':<10} | {'Time':<12} | {'Space':<22} | {'Complete?':<28} | {'Optimal?'}")
    print("-" * 88)
    for algo, data in viva["complexity_matrix"].items():
        print(f"{algo:<10} | {data['time']:<12} | {data['space']:<22} | {data['complete']:<28} | {data['optimal']}")
    print()


def run_benchmark_suite(save_dir: Optional[Path] = None) -> None:
    """Execute comparative benchmarks across all canonical 10x10 maze configurations."""
    print_banner("BFS vs DFS 10x10 MAZE BENCHMARK SUITE")

    benchmarks = [
        (
            "Default 10x10 Benchmark Maze (BFS Optimal 12 vs DFS Suboptimal 22)",
            DEFAULT_10X10_MAZE,
            DEFAULT_START_COORD,
            DEFAULT_GOAL_COORD,
            "default_10x10.png",
        ),
        (
            "DFS-Favored 10x10 Maze (DFS Low-Expansions: 14 vs BFS: 45)",
            DFS_FAVORED_10X10_MAZE,
            (0, 0),
            DFS_FAVORED_GOAL_COORD,
            "dfs_favored_10x10.png",
        ),
        (
            "Corner-to-Corner 10x10 Maze (Start (0,0) -> Goal (9,9))",
            CORNER_10X10_MAZE,
            (0, 0),
            CORNER_GOAL_COORD,
            "corner_10x10.png",
        ),
        (
            "Open 10x10 Grid (Zero Obstacles / Manhattan Optimal)",
            OPEN_10X10_MAZE,
            (0, 0),
            (9, 9),
            "open_10x10.png",
        ),
        (
            "Unsolvable 10x10 Maze (Continuous Wall Barrier)",
            UNSOLVABLE_10X10_MAZE,
            (0, 0),
            (9, 9),
            "unsolvable_10x10.png",
        ),
    ]

    for idx, (desc, grid, start, goal, plot_name) in enumerate(benchmarks, 1):
        print(f"\n>>> Benchmark {idx}: {desc}")
        comp = compare_maze_solvers(grid, start=start, goal=goal)
        print(f"Sample Output Format:")
        print(f"  {comp.sample_output_formatted}")
        print(comp.format_summary_table())

        if save_dir:
            plot_path = save_dir / plot_name
            plot_bfs_vs_dfs_expansion(comp, save_path=plot_path, title_suffix=desc)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for BFS vs DFS Performance Analyzer."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_bfs_dfs_performance_analyzer.py",
        description="Question 18: BFS vs DFS Performance Analyzer on 10x10 Mazes",
    )
    parser.add_argument(
        "--maze",
        choices=["default", "dfs-favored", "corner", "open", "unsolvable"],
        default="default",
        help="Select predefined hardcoded 10x10 maze (default: 'default')",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start coordinate in 'row,col' format (e.g. '0,0')",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal coordinate in 'row,col' format (e.g. '5,7' or '9,9')",
    )
    parser.add_argument(
        "--format",
        choices=["single", "table", "full", "json"],
        default="full",
        help="Output display format (default: 'full')",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="File path to save Matplotlib dynamic expansion comparison plot",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Execute Optional Challenge: plot expansion process and save visualization",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark suite across all predefined 10x10 mazes",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display Viva Voce examination guide and authoritative theory notes",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print_viva_guide()
        return

    if args.demo:
        save_dir = Path("reports") if (args.save or args.challenge) else None
        run_benchmark_suite(save_dir=save_dir)
        return

    # Select Maze Matrix and Default Coordinates
    if args.maze == "default":
        grid = DEFAULT_10X10_MAZE
        default_start = DEFAULT_START_COORD
        default_goal = DEFAULT_GOAL_COORD
        maze_label = "Default 10x10 Benchmark Maze (BFS Optimal vs DFS Arbitrary)"
    elif args.maze == "dfs-favored":
        grid = DFS_FAVORED_10X10_MAZE
        default_start = (0, 0)
        default_goal = DFS_FAVORED_GOAL_COORD
        maze_label = "DFS-Favored 10x10 Maze (DFS 14 Expansions vs BFS 45 Expansions)"
    elif args.maze == "corner":
        grid = CORNER_10X10_MAZE
        default_start = (0, 0)
        default_goal = CORNER_GOAL_COORD
        maze_label = "Classic Corner-to-Corner 10x10 Maze (Start (0,0) -> Goal (9,9))"
    elif args.maze == "open":
        grid = OPEN_10X10_MAZE
        default_start = (0, 0)
        default_goal = (9, 9)
        maze_label = "Open 10x10 Grid (Zero Obstacles)"
    else:  # unsolvable
        grid = UNSOLVABLE_10X10_MAZE
        default_start = (0, 0)
        default_goal = (9, 9)
        maze_label = "Unsolvable 10x10 Maze (Wall Barrier)"

    start_coord = parse_coordinate(args.start) if args.start else default_start
    goal_coord = parse_coordinate(args.goal) if args.goal else default_goal

    # Execute BFS vs DFS Comparison
    comp = compare_maze_solvers(grid, start=start_coord, goal=goal_coord)

    # Format Output
    if args.format == "single":
        print(comp.sample_output_formatted)
        return

    if args.format == "json":
        print(json.dumps(comp.to_dict(), indent=2))
        return

    # Table or Full output
    print_banner(f"QUESTION NO. 18: BFS vs DFS PERFORMANCE ANALYZER\n  Maze: {maze_label}")
    print(f"Start Coordinate: {comp.bfs.start} | Goal Coordinate: {comp.bfs.goal}")
    print(f"Neighbor Order  : Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, +1)\n")

    print("[1] SYLLABUS SAMPLE OUTPUT MATCH:")
    print(f"  Target  : {SAMPLE_OUTPUT_TARGET_STRING}")
    print(f"  Live Run: {comp.sample_output_formatted}\n")

    print("[2] COMPARATIVE METRICS SUMMARY TABLE:")
    print(comp.format_summary_table())
    print()

    if args.format == "full":
        print("[3] ASCII MAZE PATH VISUALIZATION:")
        print("\n--- BFS Path (Optimal Shortest Path: *): ---")
        print(comp.bfs.render_ascii_path())
        print(f"BFS Moves: {comp.bfs.actions_str}")

        print("\n--- DFS Path (Arbitrary Path: *): ---")
        print(comp.dfs.render_ascii_path())
        print(f"DFS Moves: {comp.dfs.actions_str}")
        print()

    # Optional Challenge Plot
    save_target = args.save
    if args.challenge and not save_target:
        save_target = "reports/bfs_vs_dfs_performance.png"

    if save_target:
        plot_bfs_vs_dfs_expansion(comp, save_path=save_target, title_suffix=maze_label)


if __name__ == "__main__":
    main()
