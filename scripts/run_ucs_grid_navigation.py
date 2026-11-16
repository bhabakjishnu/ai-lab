"""Runner script for Uniform Cost Search (UCS) Grid Navigation with Cost Modifiers (Question 22).

Practical Type: Simulation
Problem Statement: Implement UCS on a 2D grid representing a topological map. Different
characters represent different movement costs: '.' (Clear) = 1, 'M' (Mountain) = 5,
'W' (Water) = 10. Find the cheapest path from the start to the goal.

Usage Examples:
    # 1. Default execution on sample grid:
    python scripts/run_ucs_grid_navigation.py

    # 2. Output only the sample format:
    python scripts/run_ucs_grid_navigation.py --sample

    # 3. Optional Challenge: Allow diagonal moves with Euclidean multiplier:
    python scripts/run_ucs_grid_navigation.py --diagonal

    # 4. Comparative analysis against BFS (Cost vs Hop Count):
    python scripts/run_ucs_grid_navigation.py --compare-bfs

    # 5. Display ASCII grid visualization:
    python scripts/run_ucs_grid_navigation.py --ascii

    # 6. Step-by-step terminal animation of search expansion:
    python scripts/run_ucs_grid_navigation.py --animate --delay 0.1

    # 7. University Viva Voce examination theory guide:
    python scripts/run_ucs_grid_navigation.py --viva

    # 8. Machine-readable JSON output:
    python scripts/run_ucs_grid_navigation.py --json

    # 9. Comprehensive laboratory demonstration:
    python scripts/run_ucs_grid_navigation.py --demo
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ucs_grid_navigation import (
    DEFAULT_TERRAIN_COSTS,
    Coordinate,
    TopologicalGrid,
    UCSGridResult,
    UCSGridSolver,
    compare_ucs_vs_bfs,
    format_path_coords,
    get_ucs_grid_viva_reference,
    parse_grid_to_cost_dict,
    render_grid_with_path,
    solve_ucs_grid,
)

# Canonical benchmark grids
SAMPLE_GRID: List[str] = [
    "...",
    ".M.",
    "..G",
]

LAKE_DETOUR_GRID: List[str] = [
    "S.W.G",
    "..W..",
    ".....",
]

MOUNTAIN_PASS_GRID: List[str] = [
    "S..M.....",
    "...MM.MM.",
    "...M...M.",
    "W..M.M.M.",
    "W....M..G",
]

COMPLEX_TERRAIN_GRID: List[str] = [
    "S...MM...W",
    ".M..MM...W",
    ".MM......W",
    "...WWW...W",
    ".M.W.W...W",
    ".M.W.W...W",
    "...W.....W",
    "MM.W.MMM..",
    "...W......",
    "WWWW....MG",
]


def print_banner(title: str) -> None:
    """Print a styled section header."""
    width = 76
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_viva_guide() -> None:
    """Print university viva voce examination questions and answers."""
    viva = get_ucs_grid_viva_reference()
    print_banner("QUESTION NO. 22: UCS GRID NAVIGATION VIVA VOCE THEORY GUIDE")
    print(f"Topic         : {viva['topic']}")
    print(f"Syllabus Area : {viva['syllabus_area']}")
    print(f"Difficulty    : {viva['difficulty']}\n")

    for i, q in enumerate(viva["questions"], 1):
        print(f"--- Question {i} ---")
        print(f"Q: {q['question']}")
        print(f"Key Concepts: {', '.join(q['key_concepts'])}\n")
        print("Authoritative Answer:")
        print(q["answer"])
        print()


def run_animation(
    grid_rows: List[str],
    start: Optional[Coordinate] = None,
    goal: Optional[Coordinate] = None,
    delay: float = 0.1,
    allow_diagonal: bool = False,
) -> None:
    """Run interactive or step-by-step text animation of UCS exploration wavefront."""
    print_banner("UCS WAVEFRONT EXPANSION ANIMATION")
    topo = TopologicalGrid(grid_rows)
    s_coord = start or topo.detected_start or (0, 0)
    g_coord = goal or topo.detected_goal or (topo.height - 1, topo.width - 1)

    solver = UCSGridSolver(allow_diagonal=allow_diagonal)
    gen = solver.step_by_step_generator(topo, start=s_coord, goal=g_coord)

    final_result: Optional[UCSGridResult] = None
    step_num = 0

    try:
        while True:
            step_data = next(gen)
            step_num = step_data["step"]
            curr = step_data["current_node"]
            cost = step_data["current_cost"]
            path = step_data["partial_path"]
            f_size = step_data["frontier_size"]
            exp_count = step_data["explored_count"]

            # Clear screen or print step header
            print(f"\n--- Step {step_num} | Expanding: {curr} | Cumulative Cost g(n) = {cost} | Frontier: {f_size} | Explored: {exp_count} ---")
            ascii_view = topo.render_ascii(path=path, start=s_coord, goal=g_coord)
            print(ascii_view)

            if delay > 0:
                time.sleep(delay)
    except StopIteration as e:
        final_result = e.value

    if final_result and final_result.success:
        print("\n" + "=" * 60)
        print("   GOAL REACHED! OPTIMAL PATH FOUND!")
        print("=" * 60)
        print(f"Total Steps (Nodes Expanded): {final_result.nodes_expanded}")
        print(f"Final Optimal Cost           : {final_result.cost}")
        print(f"Path Coordinates             : {final_result.path_arrow_string}")
        print("\nFinal Solution Map:")
        print(topo.render_ascii(path=final_result.path, explored=final_result.explored_set, start=s_coord, goal=g_coord))


def run_demo() -> None:
    """Execute a comprehensive laboratory demonstration covering all scenarios."""
    print_banner("QUESTION NO. 22: UCS GRID NAVIGATION COMPREHENSIVE LABORATORY DEMO")

    # 1. Standard Sample Input & Output
    print("[1] STANDARD SAMPLE EXECUTION")
    print(f"Grid Input: {SAMPLE_GRID}")
    res_sample = solve_ucs_grid(SAMPLE_GRID, start=(0, 0), goal=(2, 2))
    print(f"Sample Output:")
    print(f"  {res_sample.formatted_output}")
    print(f"  Coordinate list : {res_sample.path}")
    print(f"  Total Cost      : {res_sample.cost}")
    print("\nSample Grid ASCII Rendering:")
    print(render_grid_with_path(SAMPLE_GRID, res_sample.path, res_sample.explored_set))

    # 2. Coordinate Cost Dictionary Parsing
    print("\n[2] PARSING CHARACTER GRID INTO COORDINATE COST DICTIONARY")
    cost_dict = parse_grid_to_cost_dict(SAMPLE_GRID)
    print("Parsed Coordinate Cost Map (Row, Col) -> Cost:")
    for (r, c), cost in sorted(cost_dict.items()):
        char = SAMPLE_GRID[r][c]
        print(f"  Cell ({r}, {c}) ['{char}'] -> Cost {cost}")

    # 3. Optional Challenge: Diagonal Moves with Euclidean Multiplier
    print("\n[3] OPTIONAL CHALLENGE: DIAGONAL MOVES WITH EUCLIDEAN MULTIPLIER")
    res_diag = solve_ucs_grid(SAMPLE_GRID, start=(0, 0), goal=(2, 2), allow_diagonal=True)
    print(f"Orthogonal 4-Directional Cost : {res_sample.cost}")
    print(f"Diagonal 8-Directional Cost   : {res_diag.cost:.3f}")
    print(f"Diagonal Path: {res_diag.path_arrow_string}")
    print("Diagonal Grid ASCII Rendering:")
    print(render_grid_with_path(SAMPLE_GRID, res_diag.path, res_diag.explored_set))

    # 4. Water Detour Scenario & BFS Comparison
    print("\n[4] COMPARATIVE ANALYSIS: UCS (Cost-Optimal) vs BFS (Hop-Optimal)")
    print("Grid with Water Barrier 'W' (Cost 10):")
    topo_lake = TopologicalGrid(LAKE_DETOUR_GRID)
    print(topo_lake.render_ascii())

    comp = compare_ucs_vs_bfs(LAKE_DETOUR_GRID, start=(0, 0), goal=(0, 4))
    print(f"\nBFS Path (Shortest Hops) : {format_path_coords(comp['bfs']['path'])}")
    print(f"  Hop Count : {comp['bfs']['hops']} steps")
    print(f"  Total Cost: {comp['bfs']['cost']} (Stepped through Water!)")
    print(f"\nUCS Path (Cheapest Cost) : {format_path_coords(comp['ucs']['path'])}")
    print(f"  Hop Count : {comp['ucs']['hops']} steps (Longer Detour)")
    print(f"  Total Cost: {comp['ucs']['cost']} (Circumvented Water via Clear Ground!)")
    print(f"Cost Savings using UCS   : {comp['cost_savings']} cost units")
    print(f"Did UCS find cheaper path? {comp['is_ucs_cheaper']}")

    # 5. Large Complex Topological Map
    print("\n[5] COMPLEX 10x10 TOPOLOGICAL MAP NAVIGATION")
    res_complex = solve_ucs_grid(COMPLEX_TERRAIN_GRID, start=(0, 0), goal=(9, 9))
    print(res_complex.summary())
    print("\nComplex Map Solution Trajectory:")
    print(render_grid_with_path(COMPLEX_TERRAIN_GRID, res_complex.path, res_complex.explored_set))

    print("\n[6] LABORATORY SUMMARY")
    print("All tasks for Question 22 (Parsing, heapq UCS, 4-Way restriction, Output format,")
    print("and Optional Diagonal Challenge) completed successfully.")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI dispatcher for Question 22."""
    parser = argparse.ArgumentParser(
        description="Question 22: Uniform Cost Search (UCS) on 2D Grid with Cost Modifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on standard sample input ['...', '.M.', '..G'] and print sample output",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        help="Custom grid rows, e.g. --grid '...' '.M.' '..G'",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start coordinate as 'r,c', e.g. --start 0,0",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal coordinate as 'r,c', e.g. --goal 2,2",
    )
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="Enable diagonal moves with Euclidean multiplier (Optional Challenge)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=math.sqrt(2),
        help="Cost multiplier for diagonal moves (default sqrt(2) ≈ 1.414)",
    )
    parser.add_argument(
        "--costs",
        type=str,
        default=None,
        help="Custom terrain costs as JSON string, e.g. '{\".\": 1, \"M\": 8, \"W\": 15}'",
    )
    parser.add_argument(
        "--compare-bfs",
        action="store_true",
        help="Compare UCS (Cost-Optimal) vs BFS (Hop-Optimal) performance",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Render ASCII map of terrain, wavefront, and path",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate step-by-step search progression in terminal",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay in seconds between animation frames (default 0.05)",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display University Viva Voce examination theory reference guide",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output search result as JSON",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive laboratory demonstration covering all scenarios",
    )

    args = parser.parse_args(argv)

    # 1. Viva Voce Guide
    if args.viva:
        print_viva_guide()
        return

    # 2. Comprehensive Demo
    if args.demo:
        run_demo()
        return

    # Resolve grid rows
    grid_rows = args.grid if args.grid else SAMPLE_GRID

    # Resolve start and goal coordinates
    start_coord: Optional[Coordinate] = None
    if args.start:
        parts = [int(p.strip()) for p in args.start.split(",")]
        start_coord = (parts[0], parts[1])

    goal_coord: Optional[Coordinate] = None
    if args.goal:
        parts = [int(p.strip()) for p in args.goal.split(",")]
        goal_coord = (parts[0], parts[1])

    # Resolve custom terrain costs
    terrain_costs = None
    if args.costs:
        terrain_costs = json.loads(args.costs)

    # 3. BFS Comparison Mode
    if args.compare_bfs:
        print_banner("UCS VS BFS COMPARATIVE PERFORMANCE ANALYSIS")
        comp = compare_ucs_vs_bfs(grid_rows, start=start_coord, goal=goal_coord)
        print(f"Grid Dimension: {len(grid_rows)}x{len(grid_rows[0])}")
        print(f"BFS (Hop-Optimal) Path : {format_path_coords(comp['bfs']['path'])}")
        print(f"  Hops: {comp['bfs']['hops']}, Total Cost: {comp['bfs']['cost']}")
        print(f"UCS (Cost-Optimal) Path: {format_path_coords(comp['ucs']['path'])}")
        print(f"  Hops: {comp['ucs']['hops']}, Total Cost: {comp['ucs']['cost']}")
        print(f"Cost Difference        : {comp['cost_savings']} units")
        print(f"Did UCS find a cheaper path? {comp['is_ucs_cheaper']}")
        return

    # 4. Animation Mode
    if args.animate:
        run_animation(
            grid_rows=grid_rows,
            start=start_coord,
            goal=goal_coord,
            delay=args.delay,
            allow_diagonal=args.diagonal,
        )
        return

    # 5. Standard Search
    result = solve_ucs_grid(
        grid=grid_rows,
        start=start_coord,
        goal=goal_coord,
        allow_diagonal=args.diagonal,
        diagonal_multiplier=args.multiplier,
        terrain_costs=terrain_costs,
    )

    # 6. JSON output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    # 7. Sample output format (or default execution)
    if args.sample or (not args.ascii and not args.grid):
        print(result.formatted_output)
        return

    # 8. Detailed Summary output
    print(result.summary())
    if args.ascii:
        print("\nASCII Map Visualization:")
        print(render_grid_with_path(grid_rows, result.path, result.explored_set, start=result.start, goal=result.goal))


if __name__ == "__main__":
    main()
