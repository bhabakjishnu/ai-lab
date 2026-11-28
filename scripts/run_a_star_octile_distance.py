"""Command-Line Interface and Runner for A* with Octile Distance (Question 35).

Practical Type: Implementation
Difficulty: 🟠 Advanced
Syllabus Area: Informed Search and Heuristics

Problem Statement:
    Modify the standard grid-based A* algorithm to allow 8-directional movement
    (diagonal moves cost 1.414, straight moves cost 1). The program must use the
    Octile distance heuristic instead of Manhattan to remain admissible.

Task Requirements:
    1. Add diagonal successors.
    2. Implement Octile distance.
    3. Run A* and output the path cost.
    4. Input: 2D Maze.
    5. Expected Output: Shortest path utilizing diagonal shortcuts.
    6. Constraints / Conditions: Cannot move diagonally through two corner walls.
    7. Sample Input: Start (0,0), Goal (2,2), open grid.
    8. Sample Output: Cost: 2.828 (Two diagonal moves).

Usage Examples:
    # 1. Exact Syllabus Sample Output:
    python scripts/run_a_star_octile_distance.py --sample

    # 2. Detailed Sample Output with ASCII Grid:
    python scripts/run_a_star_octile_distance.py

    # 3. Step-by-Step Priority Queue Trace Table:
    python scripts/run_a_star_octile_distance.py --table

    # 4. Multi-Heuristic Comparative Benchmark (Octile vs Euclidean vs Manhattan vs UCS):
    python scripts/run_a_star_octile_distance.py --compare

    # 5. Run Across Canonical Benchmarks (Open 3x3, Corner Pinch, 5x5 Obstacle, Concave Trap):
    python scripts/run_a_star_octile_distance.py --benchmarks

    # 6. Terminal Animation of Search Expansion:
    python scripts/run_a_star_octile_distance.py --animate --delay 0.05

    # 7. Corner Wall Pinch Demonstration:
    python scripts/run_a_star_octile_distance.py --corner-demo

    # 8. University Viva Voce Preparation Guide:
    python scripts/run_a_star_octile_distance.py --viva

    # 9. Theoretical Foundations Analysis:
    python scripts/run_a_star_octile_distance.py --theory

    # 10. Structured JSON Export:
    python scripts/run_a_star_octile_distance.py --json

    # 11. Custom Grid via Command Line:
    python scripts/run_a_star_octile_distance.py --grid ". . .\n. X .\n. . ." --start 0,0 --goal 2,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a_star_octile_distance import (
    CONCAVE_TRAP_GOAL,
    CONCAVE_TRAP_MAZE,
    CONCAVE_TRAP_START,
    CORNER_PINCH_GOAL,
    CORNER_PINCH_MAZE,
    CORNER_PINCH_START,
    DEFAULT_5X5_GOAL,
    DEFAULT_5X5_MAZE,
    DEFAULT_5X5_START,
    DEFAULT_DIAGONAL_COST,
    DEFAULT_STRAIGHT_COST,
    OPEN_3X3_GOAL,
    OPEN_3X3_MAZE,
    OPEN_3X3_START,
    SINGLE_CORNER_GOAL,
    SINGLE_CORNER_MAZE,
    SINGLE_CORNER_START,
    VIVA_VOCE_GUIDE,
    AStarOctileComparisonResult,
    AStarOctileResult,
    CornerConstraintMode,
    a_star_octile_search,
    animate_octile_search,
    compare_octile_vs_heuristics,
    euclidean_distance,
    manhattan_distance,
    octile_distance,
    parse_grid,
    render_octile_maze,
    zero_heuristic,
)


def parse_coord(coord_str: str) -> Tuple[int, int]:
    """Parse coordinate string formatted as 'r,c' or '(r,c)'."""
    clean = coord_str.strip().strip("()[]")
    parts = clean.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{coord_str}'. Expected 'r,c' (e.g. '0,0').")
    return int(parts[0].strip()), int(parts[1].strip())


def run_sample_output(quiet: bool = False) -> str:
    """Print the exact syllabus sample output for Question 35.

    Sample Input: Start (0,0), Goal (2,2), open grid
    Sample Output: Cost: 2.828 (Two diagonal moves)
    """
    res = a_star_octile_search(
        maze=OPEN_3X3_MAZE,
        start=OPEN_3X3_START,
        goal=OPEN_3X3_GOAL,
    )
    output = f"Cost: {res.format_cost(3)} ({res.move_breakdown_summary()})"
    if not quiet:
        print(output)
    return output


def print_step_table(result: AStarOctileResult) -> None:
    """Print a formatted step-by-step priority queue trace table."""
    headers = ["Step", "Node", "Action", "g(n)", "h(n)", "f(n)", "Frontier", "Closed", "Notes"]
    rows = []
    for step in result.steps:
        coord_str = f"({step.current_coord[0]}, {step.current_coord[1]})"
        rows.append(
            [
                str(step.step_number),
                coord_str,
                step.action,
                f"{step.g_cost:.3f}",
                f"{step.h_cost:.3f}",
                f"{step.f_cost:.3f}",
                str(step.frontier_size),
                str(step.closed_size),
                step.notes,
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print("\n" + sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        print(line)
    print(sep + "\n")


def print_theory() -> None:
    """Print in-depth theoretical analysis of 8-way movement and Octile heuristic."""
    theory_text = """
================================================================================
A* PATHFINDING WITH OCTILE DISTANCE (8-DIRECTIONAL MOVEMENT) - THEORY
================================================================================

1. THE NEED FOR 8-DIRECTIONAL HEURISTICS:
   In spatial navigation, agents frequently have 8 degrees of freedom:
   - 4 Cardinal / Orthogonal moves: Up, Down, Left, Right (cost D1 = 1.0)
   - 4 Diagonal moves: Up-Left, Up-Right, Down-Left, Down-Right (cost D2 = 1.414)
   Diagonal moves allow taking shortcuts across grid cells, reducing overall path length.

2. WHY MANHATTAN DISTANCE IS INADMISSIBLE FOR 8-WAY MOVEMENT:
   - Manhattan distance assumes only 4-directional moves:
         h_Manhattan = D1 * (dx + dy)
   - Between (0, 0) and (2, 2), dx = 2, dy = 2:
         h_Manhattan = 1.0 * (2 + 2) = 4.0
         True cost h* = 2 * 1.414 = 2.828 (two diagonal moves)
   - Because h_Manhattan( (0, 0) ) = 4.0 > h*( (0, 0) ) = 2.828, Manhattan OVERESTIMATES!
   - Under an inadmissible heuristic, A* search loses its guarantee of returning the
     shortest/cheapest path and can terminate with suboptimal routes.

3. THE OCTILE DISTANCE FORMULATION:
   - In an obstacle-free 8-connected grid, the optimal path covers:
     * min(dx, dy) diagonal moves
     * |dx - dy| = max(dx, dy) - min(dx, dy) straight moves
   - Exact remaining cost:
         h_Octile(n) = D1 * |dx - dy| + D2 * min(dx, dy)
                     = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)
   - With D1 = 1.0 and D2 = 1.414:
         h_Octile(n) = max(dx, dy) + 0.414 * min(dx, dy)
   - Admissibility: Since obstacles can only lengthen the path, 0 <= h_Octile(n) <= h*(n).
   - Consistency: Satisfies the triangle inequality h(u) <= c(u, v) + h(v) for all 8 moves.

4. EUCLIDEAN VS. OCTILE HEURISTIC:
   - Euclidean distance: h_Euclidean = sqrt(dx^2 + dy^2).
   - While Euclidean is admissible on an 8-way grid, it strictly underestimates the discrete
     grid transitions: for dx = 1, dy = 2, Euclidean = sqrt(5) ~ 2.236, while the true
     discrete cost is 1.414 + 1.0 = 2.414.
   - Octile strictly dominates Euclidean: h_Euclidean(n) <= h_Octile(n) <= h*(n).
   - A* with Octile expands fewer or equal nodes than A* with Euclidean distance.

5. CORNER WALL SQUEEZE CONSTRAINT:
   - When moving diagonally from (r, c) to (r + dr, c + dc), the agent passes between
     intermediate orthogonal cells (r + dr, c) and (r, c + dc).
   - If BOTH adjacent orthogonal cells are obstacles, passing through would mean squeezing
     between two touching corners ("Cannot move diagonally through two corner walls").
   - This constraint is mandatory in physical robotics to account for non-zero vehicle body width.
================================================================================
"""
    print(theory_text)


def print_viva() -> None:
    """Print the university viva voce preparation guide."""
    print("\n" + "=" * 80)
    print("UNIVERSITY VIVA VOCE EXAMINATION GUIDE: A* OCTILE DISTANCE")
    print("=" * 80)
    for q_id, entry in VIVA_VOCE_GUIDE.items():
        print(f"\nQ: {entry['question']}")
        print(f"A: {entry['answer']}\n" + "-" * 80)


def print_corner_demo() -> None:
    """Demonstrate the corner wall pinch constraint vs. open diagonal moves."""
    print("\n" + "=" * 80)
    print("CORNER WALL PINCH CONSTRAINT DEMONSTRATION")
    print("=" * 80)
    print("\nCondition: 'Cannot move diagonally through two corner walls.'")

    # 1. Corner Pinch Maze
    print("\n[Scenario 1: Corner Wall Pinch - BLOCKED]")
    print("Grid (0, 1) and (1, 0) are obstacles:")
    r1 = a_star_octile_search(CORNER_PINCH_MAZE, CORNER_PINCH_START, CORNER_PINCH_GOAL)
    print(render_octile_maze(CORNER_PINCH_MAZE, r1.path, r1.explored_set, CORNER_PINCH_START, CORNER_PINCH_GOAL))
    print(f"Path Found: {r1.found}")
    print(
        "Explanation: From (0,0), diagonal move to (1,1) is BLOCKED because both (0,1) and (1,0) are walls."
    )

    # 2. Single Corner Maze
    print("\n[Scenario 2: Single Corner Obstacle - PERMITTED]")
    print("Grid (0, 1) is obstacle, (1, 0) is open:")
    r2 = a_star_octile_search(SINGLE_CORNER_MAZE, SINGLE_CORNER_START, SINGLE_CORNER_GOAL)
    print(render_octile_maze(SINGLE_CORNER_MAZE, r2.path, r2.explored_set, SINGLE_CORNER_START, SINGLE_CORNER_GOAL))
    print(f"Path Found: {r2.found} | Cost: {r2.format_cost(3)} ({r2.move_breakdown_summary()})")
    print(
        "Explanation: Only one corner wall is present. The agent glides past without squeezing through two corner walls."
    )
    print("=" * 80 + "\n")


def run_benchmarks() -> None:
    """Run search across all canonical benchmark mazes."""
    benchmarks = [
        ("Syllabus Sample Input (3x3 Open Grid)", OPEN_3X3_MAZE, OPEN_3X3_START, OPEN_3X3_GOAL),
        ("Corner Pinch Barrier (3x3 Grid)", CORNER_PINCH_MAZE, CORNER_PINCH_START, CORNER_PINCH_GOAL),
        ("Single Corner Obstacle (3x3 Grid)", SINGLE_CORNER_MAZE, SINGLE_CORNER_START, SINGLE_CORNER_GOAL),
        ("Canonical 5x5 Maze with Vertical Wall", DEFAULT_5X5_MAZE, DEFAULT_5X5_START, DEFAULT_5X5_GOAL),
        ("U-Shaped Concave Trap Maze (8x6 Grid)", CONCAVE_TRAP_MAZE, CONCAVE_TRAP_START, CONCAVE_TRAP_GOAL),
    ]

    print("\n" + "=" * 80)
    print("A* OCTILE DISTANCE: CANONICAL BENCHMARK SUITE")
    print("=" * 80)

    for title, maze, start, goal in benchmarks:
        print(f"\n--- Benchmark: {title} ---")
        res = a_star_octile_search(maze, start, goal)
        print(res.summary())
        print("\nMaze Solution Rendering:")
        print(render_octile_maze(maze, res.path, res.explored_set, start, goal, use_directional_glyphs=True))
        print("-" * 80)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface entry point for A* Octile Search."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_a_star_octile_distance.py",
        description="A* Pathfinding with Octile Distance Heuristic (8-Directional Diagonal Movement)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print the exact syllabus sample output: Cost: 2.828 (Two diagonal moves)",
    )
    parser.add_argument(
        "--sample-detailed",
        action="store_true",
        help="Print sample run with full ASCII grid visualization and move details",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Custom 2D grid string (e.g. '. . .\\n. X .\\n. . .')",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start coordinate formatted as 'r,c' (e.g. '0,0')",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal coordinate formatted as 'r,c' (e.g. '2,2')",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print step-by-step priority queue expansion trace table",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Octile vs Euclidean vs Manhattan vs Zero (UCS) heuristics",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run A* Octile search across all canonical benchmark mazes",
    )
    parser.add_argument(
        "--corner-demo",
        action="store_true",
        help="Demonstrate the corner wall pinch constraint",
    )
    parser.add_argument(
        "--corner-mode",
        choices=["no_double_corner_walls", "strict_no_corner_cutting", "allow_all"],
        default="no_double_corner_walls",
        help="Corner wall constraint mode (default: no_double_corner_walls)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate node expansion and path reconstruction in terminal",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Animation delay per frame in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Display comprehensive theoretical analysis of 8-way heuristics",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display university viva voce examination questions and answers",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output search results formatted as JSON",
    )

    args = parser.parse_args(argv)

    # 1. Exact Syllabus Sample Output
    if args.sample:
        run_sample_output()
        return

    # 2. Theory
    if args.theory:
        print_theory()
        return

    # 3. Viva
    if args.viva:
        print_viva()
        return

    # 4. Corner Demo
    if args.corner_demo:
        print_corner_demo()
        return

    # 5. Benchmarks
    if args.benchmarks:
        run_benchmarks()
        return

    # Determine Grid, Start, Goal
    corner_mode = CornerConstraintMode(args.corner_mode)
    if args.grid is not None:
        maze, parsed_start, parsed_goal = parse_grid(args.grid)
        start_coord = parse_coord(args.start) if args.start else parsed_start
        goal_coord = parse_coord(args.goal) if args.goal else parsed_goal
    else:
        maze = OPEN_3X3_MAZE
        start_coord = parse_coord(args.start) if args.start else OPEN_3X3_START
        goal_coord = parse_coord(args.goal) if args.goal else OPEN_3X3_GOAL

    # 6. Multi-Heuristic Comparison
    if args.compare:
        comp = compare_octile_vs_heuristics(maze, start_coord, goal_coord, corner_mode=corner_mode)
        print("\n" + "=" * 80)
        print("8-DIRECTIONAL GRID SEARCH: MULTI-HEURISTIC COMPARATIVE ANALYSIS")
        print("=" * 80)
        print(comp.to_table())
        print("=" * 80 + "\n")
        return

    # 7. Standard or Detailed Search Run
    record_steps = args.table or args.animate or args.json
    res = a_star_octile_search(
        maze=maze,
        start=start_coord,
        goal=goal_coord,
        corner_mode=corner_mode,
        record_steps=record_steps,
    )

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return

    if args.animate:
        animate_octile_search(maze, res, delay=args.delay)
        return

    if args.table:
        print("\n=== STEP-BY-STEP PRIORITY QUEUE EXPANSION TRACE ===")
        print_step_table(res)

    # Default formatted output
    print("\n" + "=" * 80)
    print("A* WITH OCTILE DISTANCE (8-DIRECTIONAL DIAGONAL MOVEMENT)")
    print("=" * 80)
    print(f"Sample Output Format: Cost: {res.format_cost(3)} ({res.move_breakdown_summary()})")
    print("-" * 80)
    print(res.summary())
    print("-" * 80)
    print("ASCII Maze Visualization:")
    print(render_octile_maze(maze, res.path, res.explored_set, start_coord, goal_coord, use_directional_glyphs=True))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
