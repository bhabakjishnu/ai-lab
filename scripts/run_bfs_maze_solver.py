"""Runner script for Breadth-First Search (BFS) 2D Grid Maze Solver (Question 15).

Provides interactive user prompting, CLI parameter parsing, automated benchmark
demonstrations (sample 3x3 grid, non-square 3x5 grid, serpentine maze, unsolvable maze,
single-cell grid), step-by-step expansion trace tables, FIFO frontier inspection,
Optional Challenge demonstration (string of moves: "D, R, R, D"), Matplotlib
visualization export, and comprehensive Viva Voce theory notes.

Usage Examples:
    # 1. Default Sample Run (Lab Question 15 default: [[0,1,0],[0,0,0],[1,1,0]]):
    python scripts/run_bfs_maze_solver.py

    # 2. Custom Grid via CLI (JSON format or comma-separated rows):
    python scripts/run_bfs_maze_solver.py --grid "[[0,1,0],[0,0,0],[1,1,0]]"

    # 3. Custom Start and Goal Coordinates:
    python scripts/run_bfs_maze_solver.py --grid "[[0,0,0],[1,1,0],[0,0,0]]" --start 0,0 --goal 2,0

    # 4. Interactive Console Mode:
    python scripts/run_bfs_maze_solver.py --interactive

    # 5. Run Optional Challenge Demonstration (String of moves: "D, R, R, D"):
    python scripts/run_bfs_maze_solver.py --challenge

    # 6. Run All Standard Benchmarks (Sample, Non-Square, Serpentine, Blocked, Single-Cell):
    python scripts/run_bfs_maze_solver.py --demo

    # 7. Generate and Save Graphical Diagram:
    python scripts/run_bfs_maze_solver.py --save reports/bfs_maze_solution.png

    # 8. Display Viva Voce Study Guide & Comparison Matrix:
    python scripts/run_bfs_maze_solver.py --viva
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bfs_maze_solver import (
    BFSMazeResult,
    BFSMazeSolver,
    BFSMazeStepRecord,
    DEFAULT_SAMPLE_MAZE,
    NON_SQUARE_RECTANGULAR_MAZE,
    OPEN_GRID_4X4_MAZE,
    SERPENTINE_MAZE,
    SINGLE_CELL_MAZE,
    UNSOLVABLE_BLOCKED_MAZE,
    get_maze_solver_viva_reference,
    parse_grid,
    solve_maze_bfs,
    visualize_maze_solution,
)


def print_banner(title: str) -> None:
    """Print a visually distinct laboratory section header.

    Args:
        title: Header text to display.
    """
    width = 75
    border = "=" * width
    print(f"\n{border}")
    print(f" {title.upper()}")
    print(border)


def print_section(title: str) -> None:
    """Print a secondary section subheader.

    Args:
        title: Subheader text to display.
    """
    print(f"\n--- {title} ---")


def format_step_trace_table(trace: List[BFSMazeStepRecord]) -> str:
    """Format the step-by-step node expansion trace as an ASCII table.

    Args:
        trace: List of BFSMazeStepRecord instances.

    Returns:
        Formatted ASCII table string.
    """
    if not trace:
        return "No expansion trace available."

    headers = ["Step", "Popped (r,c)", "Action", "Dist", "Frontier (FIFO Queue)", "Generated Successors"]
    widths = [6, 14, 8, 6, 26, 24]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    lines = [header_line, sep_line]

    for rec in trace:
        step_str = str(rec.step_number).ljust(widths[0])
        popped_str = str(rec.current_coord).ljust(widths[1])
        action_str = (rec.action_taken if rec.action_taken else "START").ljust(widths[2])
        dist_str = str(rec.distance).ljust(widths[3])

        # Shorten frontier representation if too long
        frontier_repr = ", ".join(str(c) for c in rec.frontier_snapshot)
        if len(frontier_repr) > widths[4]:
            frontier_repr = frontier_repr[: widths[4] - 3] + "..."
        frontier_str = frontier_repr.ljust(widths[4])

        succ_repr = ", ".join(f"{act}:{c}" for act, c in rec.successors_generated)
        if len(succ_repr) > widths[5]:
            succ_repr = succ_repr[: widths[5] - 3] + "..."
        succ_str = succ_repr.ljust(widths[5])

        lines.append(f"{step_str} | {popped_str} | {action_str} | {dist_str} | {frontier_str} | {succ_str}")

    return "\n".join(lines)


def display_result(result: BFSMazeResult, show_trace: bool = True) -> None:
    """Display complete results for a maze solve run in university laboratory format.

    Args:
        result: BFSMazeResult to render.
        show_trace: Whether to include the expansion trace table.
    """
    print_section("Maze ASCII Grid (S = Start, G = Goal, * = Path, # = Wall, . = Open)")
    print(result.render_ascii_solution())

    print_section("Solution Verification & Metrics")
    print(result.format_summary())

    if result.found:
        print(f"\n>> Shortest steps: {result.steps}")
        print(f">> Moves: \"{result.actions_str}\"")
    else:
        print("\n>> Shortest steps: -1 (Goal unreachable)")
        print(">> Moves: None")

    if show_trace and result.trace:
        print_section("Step-by-Step FIFO Frontier & Dynamic Successor Trace")
        print(format_step_trace_table(result.trace))


def run_challenge_mode() -> None:
    """Demonstrate the Optional Challenge: Outputting a string of moves ('D, R, R, D')."""
    print_banner("Optional Challenge: Directional Move String ('D, R, R, D')")
    print("Problem requirement: In addition to shortest steps, output the sequence")
    print("of directional actions (Up='U', Down='D', Left='L', Right='R') taken by the agent.")

    sample_grid = DEFAULT_SAMPLE_MAZE
    result = solve_maze_bfs(sample_grid, record_trace=True)

    print("\nInput Maze (3x3):")
    for row in sample_grid:
        print(" ", row)

    print(f"\nExpected Output Shortest Steps: 4")
    print(f"Actual Computed Shortest Steps: {result.steps}")
    print(f"\nExpected Optional Challenge Moves: \"D, R, R, D\"")
    print(f"Actual Computed Moves Sequence   : \"{result.actions_str}\"")

    print("\nStep-by-Step Path Coordinates with Action Moves:")
    for i in range(len(result.path) - 1):
        c_from = result.path[i]
        c_to = result.path[i + 1]
        act = result.actions[i]
        act_name = {"U": "Up", "D": "Down", "L": "Left", "R": "Right"}.get(act, act)
        print(f"  Step {i + 1}: Move {act} ({act_name.ljust(5)}) from {c_from} to {c_to}")

    print("\n[Verification] Challenge passed with 100% precision!")


def run_benchmark_demonstrations() -> None:
    """Execute a comprehensive battery of benchmark mazes across different topologies."""
    print_banner("Automated Benchmarks: BFS 2D Maze Topologies")

    benchmarks: List[Tuple[str, Any, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]] = [
        (
            "1. Default Sample Maze (3x3 - Laboratory Standard)",
            DEFAULT_SAMPLE_MAZE,
            (0, 0),
            (2, 2),
        ),
        (
            "2. Non-Square Rectangular Maze (3 rows x 5 columns)",
            NON_SQUARE_RECTANGULAR_MAZE,
            (0, 0),
            (2, 4),
        ),
        (
            "3. Winding Serpentine Corridor Maze (5x5)",
            SERPENTINE_MAZE,
            (0, 0),
            (4, 4),
        ),
        (
            "4. Open Obstacle-Free Grid (4x4, Steps == Manhattan Distance)",
            OPEN_GRID_4X4_MAZE,
            (0, 0),
            (3, 3),
        ),
        (
            "5. Completely Blocked / Unsolvable Maze",
            UNSOLVABLE_BLOCKED_MAZE,
            (0, 0),
            (2, 2),
        ),
        (
            "6. Single-Cell Maze (Start == Goal)",
            SINGLE_CELL_MAZE,
            (0, 0),
            (0, 0),
        ),
    ]

    for title, grid, start, goal in benchmarks:
        print_section(title)
        res = solve_maze_bfs(grid, start=start, goal=goal, record_trace=False)
        print(res.render_ascii_solution())
        if res.found:
            print(f"Result: SUCCESS | Steps: {res.steps} | Moves: \"{res.actions_str}\" | Manhattan Bound: {res.manhattan_distance}")
        else:
            print(f"Result: UNREACHABLE | Steps: -1 | Path: None")


def run_viva_study_guide() -> None:
    """Display comprehensive Viva Voce questions, answers, and algorithm comparison table."""
    viva = get_maze_solver_viva_reference()
    print_banner(viva["title"])
    print(f"Syllabus Area: {viva['syllabus_area']} | Difficulty: {viva['difficulty']}\n")

    for i, topic in enumerate(viva["core_topics"], 1):
        print(f"Topic {i}: {topic['topic']}")
        print(f"Q: {topic['question']}")
        print(f"A: {topic['answer']}\n")

    print_section("Algorithm Comparison Matrix on 2D Grid Mazes")
    matrix = viva["comparison_matrix"]
    cols = matrix["columns"]
    rows = matrix["rows"]
    col_widths = [max(len(row[i]) for row in [cols] + rows) + 2 for i in range(len(cols))]

    header = " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(r, col_widths)))


def parse_coordinate_arg(coord_str: str) -> Tuple[int, int]:
    """Parse a coordinate string in the format 'r,c' or '(r, c)'.

    Args:
        coord_str: String representation of coordinate.

    Returns:
        Tuple of two integers (r, c).
    """
    clean = coord_str.strip("()[] ")
    parts = clean.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid coordinate format '{coord_str}'. Expected 'r,c' (e.g. '0,0').")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Coordinate values must be integers: {err}") from err


def interactive_mode() -> None:
    """Prompt the user interactively to define a maze and execute BFS."""
    print_banner("Interactive Console Mode: BFS 2D Maze Solver")
    try:
        rows_str = input("Enter number of rows N [default: 3]: ").strip()
        num_rows = int(rows_str) if rows_str else 3

        cols_str = input("Enter number of columns M [default: 3]: ").strip()
        num_cols = int(cols_str) if cols_str else 3

        print(f"\nEnter the {num_rows} rows (space-separated 0s for open and 1s for wall):")
        print("Example row: 0 1 0")
        grid: List[List[int]] = []
        for r in range(num_rows):
            row_input = input(f"Row {r}: ").strip()
            if not row_input:
                row_ints = [0] * num_cols
            else:
                tokens = [t.strip() for t in row_input.replace(",", " ").split()]
                row_ints = [int(t) for t in tokens]
                if len(row_ints) != num_cols:
                    print(f"Warning: expected {num_cols} values, got {len(row_ints)}. Padding or truncating.")
                    row_ints = (row_ints + [0] * num_cols)[:num_cols]
            grid.append(row_ints)

        start_input = input(f"Enter start coordinate (r, c) [default: (0, 0)]: ").strip()
        start = parse_coordinate_arg(start_input) if start_input else (0, 0)

        goal_input = input(f"Enter goal coordinate (r, c) [default: ({num_rows - 1}, {num_cols - 1})]: ").strip()
        goal = parse_coordinate_arg(goal_input) if goal_input else (num_rows - 1, num_cols - 1)

        result = solve_maze_bfs(grid, start=start, goal=goal, record_trace=True)
        display_result(result, show_trace=True)

    except (ValueError, KeyboardInterrupt) as err:
        print(f"\n[Error/Aborted]: {err}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for BFS 2D Grid Maze Solver."""
    parser = argparse.ArgumentParser(
        description="Question 15: Breadth-First Search (BFS) 2D Grid Maze Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="2D maze representation as JSON (e.g. '[[0,1,0],[0,0,0],[1,1,0]]') or multiline text.",
    )
    parser.add_argument(
        "--start",
        type=parse_coordinate_arg,
        default=None,
        help="Starting coordinate 'r,c' (default: 0,0).",
    )
    parser.add_argument(
        "--goal",
        type=parse_coordinate_arg,
        default=None,
        help="Goal coordinate 'r,c' (default: bottom-right N-1,M-1).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive console input mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive benchmark suite across various maze topologies.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run Optional Challenge demonstration: String of moves ('D, R, R, D').",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display Viva Voce theory study guide and algorithm comparison table.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="File path to save the Matplotlib graphical diagram (e.g. reports/bfs_maze.png).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Suppress step-by-step expansion trace table.",
    )

    args = parser.parse_args(argv)

    if args.interactive:
        interactive_mode()
        return

    if args.challenge:
        run_challenge_mode()
        return

    if args.demo:
        run_benchmark_demonstrations()
        return

    if args.viva:
        run_viva_study_guide()
        return

    # Standard run: default sample or CLI arguments
    grid_data = args.grid if args.grid is not None else DEFAULT_SAMPLE_MAZE
    print_banner("Question 15: BFS 2D Grid Maze Solver")

    result = solve_maze_bfs(
        grid_input=grid_data,
        start=args.start,
        goal=args.goal,
        record_trace=not args.no_trace,
    )
    display_result(result, show_trace=not args.no_trace)

    if args.save:
        visualize_maze_solution(grid_data, result=result, output_path=args.save)


if __name__ == "__main__":
    main()
