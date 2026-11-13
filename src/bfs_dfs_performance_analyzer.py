"""BFS vs DFS Performance Analyzer (Question 18).

This module implements and compares Breadth-First Search (BFS) and Depth-First Search (DFS)
on 2D grid mazes, analyzing empirical performance metrics:
- Exact number of nodes expanded before reaching the goal.
- Solution path length and optimality (shortest path vs. arbitrary path).
- Memory consumption (FIFO queue frontier size vs. LIFO stack depth).
- Wall-clock execution time (milliseconds).
- Step-by-step exploration dynamics and search space visualization.

Theoretical Background & AI Concepts:
- Uninformed (Blind) Search on Implicit Graphs:
    In grid maze navigation, vertices (cells) and transitions (orthogonal moves) are generated
    dynamically on demand using a successor function: S(s) -> { (action, s') }.
    Both BFS and DFS operate blindly without heuristic distance estimates to the goal,
    relying entirely on graph topology and traversal strategy.
- Frontier Data Structures & Search Bias:
    - BFS uses a First-In, First-Out (FIFO) queue (collections.deque). It explores the state
      space systematically in concentric waves of increasing distance from the start node
      (shallowest unexpanded nodes first).
    - DFS uses a Last-In, First-Out (LIFO) stack (explicit Python list or recursion). It plunges
      deeply down a single branch before backtracking upon hitting dead-ends or visited states.
- Neighbor Ordering Constraint:
    Both algorithms MUST evaluate orthogonal neighbors in the exact same priority order:
    Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, +1).
    To ensure that the iterative DFS stack pops and explores neighbors in the strict forward order
    (Up, then Down, then Left, then Right) matching standard recursive DFS, candidate successors
    are pushed onto the LIFO stack in reverse order (`reversed(successors)`).
- Optimality vs. Arbitrary Pathfinding:
    - BFS Optimality: Because all transitions have uniform unit cost (step cost = 1), BFS is
      theoretically guaranteed to find the optimal (shallowest / minimum-step) path.
    - DFS Arbitrariness: DFS is NOT optimal. It returns the first path that stumbles upon the goal,
      which frequently produces long, winding, or circuitous detours.
- Empirical Trade-offs & Viva Voce Scenarios:
    - Scenario 1 (BFS Optimality): In mazes with multiple branching paths, BFS explores wider
      frontiers (expanding more nodes) but guarantees finding the shortest path (e.g., 12 steps).
      DFS may explore an arbitrary deep branch, returning a suboptimal path (e.g., 22 steps).
    - Scenario 2 (DFS Low-Expansion Scenario): When the goal happens to reside deep along the
      very first branch explored by the neighbor ordering, DFS reaches it with significantly
      fewer expansions (e.g., 14 expansions vs BFS's 45 expansions), avoiding BFS's exhaustive
      level-by-level exploration.
    - Memory Consumption: BFS requires O(b^d) memory (exponential in goal depth), whereas DFS
      requires only O(b * m) linear memory (proportional to maximum search depth).

Complexity:
- Time Complexity:
    - BFS: O(V + E) = O(N * M) on an N x M grid with 4-connectivity.
    - DFS: O(V + E) = O(N * M) on an N x M grid with 4-connectivity.
- Space Complexity:
    - BFS: O(N * M) auxiliary space for FIFO queue and visited set.
    - DFS: O(N * M) auxiliary space for visited set, with active stack bounded by maximum path depth <= N * M.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Cardinal Actions & Directional Vectors (row_delta, col_delta)
ACTION_UP: str = "U"
ACTION_DOWN: str = "D"
ACTION_LEFT: str = "L"
ACTION_RIGHT: str = "R"

# Orthogonal movement offsets in standard priority order: Up, Down, Left, Right
DEFAULT_DIRECTIONS: Tuple[Tuple[str, int, int], ...] = (
    (ACTION_UP, -1, 0),
    (ACTION_DOWN, 1, 0),
    (ACTION_LEFT, 0, -1),
    (ACTION_RIGHT, 0, 1),
)

# Alternative move ordering: Down, Right, Up, Left (favors forward downward progression)
DOWN_RIGHT_DIRECTIONS: Tuple[Tuple[str, int, int], ...] = (
    (ACTION_DOWN, 1, 0),
    (ACTION_RIGHT, 0, 1),
    (ACTION_UP, -1, 0),
    (ACTION_LEFT, 0, -1),
)

# Grid Cell Definitions
CELL_OPEN: int = 0
CELL_WALL: int = 1

# ASCII Visualization Glyphs
GLYPH_START: str = "S"
GLYPH_GOAL: str = "G"
GLYPH_PATH: str = "*"
GLYPH_WALL: str = "#"
GLYPH_OPEN: str = "."

# Sample Output target specification string from syllabus / lab manual
SAMPLE_OUTPUT_TARGET_STRING: str = (
    "BFS Nodes Expanded: 45, Path Length: 12 | DFS Nodes Expanded: 14, Path Length: 22"
)

# ============================================================================
# Predefined Hardcoded 10x10 Mazes for Laboratory Testing & Benchmarks
# ============================================================================

# Default 10x10 Benchmark Maze:
# Demonstrates BFS finding optimal path (12 steps, 45 nodes expanded) vs.
# DFS taking a circuitous, winding path (22 steps, 40 nodes expanded) to Goal (5, 7).
DEFAULT_10X10_MAZE: List[List[int]] = [
    [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]
DEFAULT_START_COORD: Tuple[int, int] = (0, 0)
DEFAULT_GOAL_COORD: Tuple[int, int] = (5, 7)

# DFS-Favored 10x10 Maze:
# Demonstrates viva scenario where DFS plunges directly toward the goal (9, 3) in
# only 14 expansions (matching sample output count 14), whereas BFS must exhaustively
# expand 45 nodes across all branches to reach depth 12.
DFS_FAVORED_10X10_MAZE: List[List[int]] = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
]
DFS_FAVORED_GOAL_COORD: Tuple[int, int] = (9, 3)

# Classic Corner-to-Corner 10x10 Maze: Start (0, 0) to Bottom-Right Goal (9, 9)
CORNER_10X10_MAZE: List[List[int]] = [
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]
CORNER_GOAL_COORD: Tuple[int, int] = (9, 9)

# Open 10x10 Grid (Zero Obstacles): BFS path length equals Manhattan distance
OPEN_10X10_MAZE: List[List[int]] = [[0 for _ in range(10)] for _ in range(10)]

# Unsolvable 10x10 Maze: Completely severed by a continuous wall barrier
UNSOLVABLE_10X10_MAZE: List[List[int]] = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Solid wall barrier across row 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
]


# ============================================================================
# Dataclasses for Metric Tracking & Step Tracing
# ============================================================================


@dataclass(frozen=True)
class MazeStepRecord:
    """Immutable trace record capturing an individual node expansion during search.

    Attributes:
        step_number: 1-based chronological expansion index.
        current_coord: Coordinate tuple (row, col) popped and expanded.
        action_taken: Move action that transitioned into this cell ('U', 'D', 'L', 'R', or None).
        distance: Distance in steps from start to current_coord.
        frontier_size: Number of candidate coordinates currently waiting in frontier.
        visited_count: Total number of coordinates marked visited up to this point.
    """

    step_number: int
    current_coord: Tuple[int, int]
    action_taken: Optional[str]
    distance: int
    frontier_size: int
    visited_count: int


@dataclass
class SearchAlgorithmResult:
    """Comprehensive metric and trajectory result for a single search algorithm execution.

    Attributes:
        algorithm: Name of algorithm ('BFS' or 'DFS').
        grid: 2D list representing the maze grid.
        start: Starting coordinate tuple (row, col).
        goal: Destination coordinate tuple (row, col).
        found: True if a valid path exists; False otherwise.
        path_length: Number of transitions/steps from start to goal (-1 if unreachable).
        path: Chronological list of coordinate tuples from start to goal (inclusive).
        actions: List of directional action letters taken along the path.
        actions_str: Comma-separated string of move actions (e.g., "D, D, R, R").
        nodes_expanded: Exact count of nodes popped and expanded from frontier.
        nodes_visited: Total unique coordinates discovered and marked visited.
        peak_frontier_size: Maximum simultaneous memory footprint in the queue/stack.
        execution_time_ms: Wall-clock execution time in milliseconds.
        expansion_order: Mapping of coordinate -> chronological expansion index (1-based).
        trace: Step-by-step expansion records for visualization and debugging.
    """

    algorithm: str
    grid: List[List[int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    found: bool
    path_length: int
    path: List[Tuple[int, int]]
    actions: List[str]
    actions_str: str
    nodes_expanded: int
    nodes_visited: int
    peak_frontier_size: int
    execution_time_ms: float
    expansion_order: Dict[Tuple[int, int], int] = field(default_factory=dict)
    trace: List[MazeStepRecord] = field(default_factory=list)

    def render_ascii_path(self) -> str:
        """Render the maze grid highlighting start (S), goal (G), path (*), and walls (#)."""
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0
        path_set = set(self.path)

        lines: List[str] = []
        for r in range(rows):
            row_chars: List[str] = []
            for c in range(cols):
                coord = (r, c)
                if coord == self.start:
                    row_chars.append(GLYPH_START)
                elif coord == self.goal:
                    row_chars.append(GLYPH_GOAL)
                elif coord in path_set:
                    row_chars.append(GLYPH_PATH)
                elif self.grid[r][c] == CELL_WALL:
                    row_chars.append(GLYPH_WALL)
                else:
                    row_chars.append(GLYPH_OPEN)
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "found": self.found,
            "path_length": self.path_length,
            "path": [list(c) for c in self.path],
            "actions": self.actions,
            "actions_str": self.actions_str,
            "nodes_expanded": self.nodes_expanded,
            "nodes_visited": self.nodes_visited,
            "peak_frontier_size": self.peak_frontier_size,
            "execution_time_ms": round(self.execution_time_ms, 4),
            "start": list(self.start),
            "goal": list(self.goal),
            "dimensions": [len(self.grid), len(self.grid[0]) if self.grid else 0],
        }


@dataclass
class ComparisonResult:
    """Encapsulates the side-by-side comparative analysis of BFS vs DFS.

    Attributes:
        bfs: Complete result for BFS execution.
        dfs: Complete result for DFS execution.
        manhattan_distance: Theoretical minimum L1 distance from start to goal.
        expanded_difference: (BFS nodes expanded) - (DFS nodes expanded).
        expanded_ratio: Ratio of BFS expansions to DFS expansions.
        path_length_difference: (DFS path length) - (BFS path length) [suboptimality penalty].
        optimality_ratio: Ratio of DFS path length to optimal BFS path length.
        is_bfs_strictly_shorter: True if BFS path is strictly shorter than DFS path.
        sample_output_formatted: Single-line string matching the laboratory sample output specification.
    """

    bfs: SearchAlgorithmResult
    dfs: SearchAlgorithmResult
    manhattan_distance: int
    expanded_difference: int
    expanded_ratio: float
    path_length_difference: int
    optimality_ratio: float
    is_bfs_strictly_shorter: bool
    sample_output_formatted: str

    def format_summary_table(self) -> str:
        """Generate an ASCII comparison table of performance metrics."""
        b = self.bfs
        d = self.dfs

        bfs_status = "Optimal (Shortest)" if b.found else "No Path"
        dfs_status = (
            "Optimal" if (d.found and d.path_length == b.path_length)
            else "Suboptimal (Arbitrary)" if d.found
            else "No Path"
        )

        lines = [
            f"+----------------------------------+--------------------+--------------------+",
            f"| Metric                           | BFS (Breadth-First)| DFS (Depth-First)  |",
            f"+----------------------------------+--------------------+--------------------+",
            f"| Solution Found                   | {str(b.found):<18} | {str(d.found):<18} |",
            f"| Path Length (Steps)              | {b.path_length:<18} | {d.path_length:<18} |",
            f"| Optimality Status                | {bfs_status:<18} | {dfs_status:<18} |",
            f"| Nodes Expanded (Popped)          | {b.nodes_expanded:<18} | {d.nodes_expanded:<18} |",
            f"| Nodes Visited (Discovered)       | {b.nodes_visited:<18} | {d.nodes_visited:<18} |",
            f"| Peak Frontier Size (Memory)      | {b.peak_frontier_size:<18} | {d.peak_frontier_size:<18} |",
            f"| Execution Time (ms)              | {b.execution_time_ms:<18.4f} | {d.execution_time_ms:<18.4f} |",
            f"| Frontier Data Structure          | deque (FIFO)       | list stack (LIFO)  |",
            f"| Neighbor Ordering                | Up, Down, Left, R  | Up, Down, Left, R  |",
            f"+----------------------------------+--------------------+--------------------+",
            f"| Manhattan Distance Lower Bound   | {self.manhattan_distance} steps               |                    |",
            f"| Path Length Difference (DFS-BFS) | {self.path_length_difference} steps (Suboptimality gap)            |",
            f"| Expansion Difference (BFS-DFS)   | {self.expanded_difference} nodes (Exploration overhead)           |",
            f"+----------------------------------+--------------------+--------------------+",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparative results into a structured dictionary."""
        return {
            "sample_output": self.sample_output_formatted,
            "manhattan_distance": self.manhattan_distance,
            "expanded_difference": self.expanded_difference,
            "expanded_ratio": round(self.expanded_ratio, 2) if self.expanded_ratio != float("inf") else "inf",
            "path_length_difference": self.path_length_difference,
            "optimality_ratio": round(self.optimality_ratio, 2) if self.optimality_ratio != float("inf") else "inf",
            "is_bfs_strictly_shorter": self.is_bfs_strictly_shorter,
            "bfs": self.bfs.to_dict(),
            "dfs": self.dfs.to_dict(),
        }


# ============================================================================
# Core Grid Utilities & Successor Generator
# ============================================================================


def calculate_manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    """Compute the L1 Manhattan distance between two grid coordinates.

    Formula:
        D = |r1 - r2| + |c1 - c2|
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def is_valid_coordinate(r: int, c: int, num_rows: int, num_cols: int) -> bool:
    """Verify whether (r, c) lies strictly within grid boundary constraints."""
    return 0 <= r < num_rows and 0 <= c < num_cols


def is_open_cell(r: int, c: int, grid: Sequence[Sequence[int]]) -> bool:
    """Verify whether (r, c) is inside bounds and represents an open traversable cell (0)."""
    if not grid or len(grid) == 0:
        return False
    if not is_valid_coordinate(r, c, len(grid), len(grid[0])):
        return False
    return grid[r][c] == CELL_OPEN


def get_successors(
    coord: Tuple[int, int],
    grid: Sequence[Sequence[int]],
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Dynamically generate valid orthogonal successor states for a grid coordinate.

    Respects the exact neighbor ordering constraint (e.g. Up, Down, Left, Right).
    Only transitions passing both boundary validation and wall collision checks are emitted.

    Args:
        coord: Current coordinate tuple (row, col).
        grid: 2D grid matrix.
        directions: Sequence of (action_name, dr, dc) tuples. Defaults to DEFAULT_DIRECTIONS.

    Returns:
        List of (action_name, (next_row, next_col)) pairs.
    """
    if directions is None:
        directions = DEFAULT_DIRECTIONS

    num_rows = len(grid)
    if num_rows == 0:
        return []
    num_cols = len(grid[0])

    r, c = coord
    valid_successors: List[Tuple[str, Tuple[int, int]]] = []

    for action, dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if is_valid_coordinate(nr, nc, num_rows, num_cols) and grid[nr][nc] == CELL_OPEN:
            valid_successors.append((action, (nr, nc)))

    return valid_successors


def parse_grid(raw_input: Any) -> List[List[int]]:
    """Parse and normalize various grid input formats into a standard 2D List[List[int]].

    Supports:
    - 2D Python list of ints: `[[0, 1, 0], [0, 0, 0]]`
    - JSON string representation
    - Multiline space- or comma-delimited strings
    - ASCII glyphs: '.' or '0' for open, '#' or '1' for wall

    Args:
        raw_input: Raw grid input representation.

    Returns:
        Validated 2D list of integers (0 and 1).

    Raises:
        ValueError: If grid is empty, non-rectangular, or contains illegal cell tokens.
        TypeError: If input cannot be converted to a grid.
    """
    if raw_input is None:
        raise ValueError("Grid input cannot be None.")

    if isinstance(raw_input, str):
        trimmed = raw_input.strip()
        if not trimmed:
            raise ValueError("Grid input string is empty.")

        # Try JSON parsing
        if (trimmed.startswith("[") and trimmed.endswith("]")) or (
            trimmed.startswith("(") and trimmed.endswith(")")
        ):
            try:
                parsed = json.loads(trimmed)
                return parse_grid(parsed)
            except Exception:
                pass

        lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
        if not lines:
            raise ValueError("No valid rows found in grid string.")

        grid_matrix: List[List[int]] = []
        for line_idx, line in enumerate(lines):
            clean = line.strip("[](),;")
            tokens = [t.strip() for t in clean.replace(",", " ").split() if t.strip()]
            row_ints: List[int] = []
            for t in tokens:
                if t in ("0", ".", "O", "o"):
                    row_ints.append(CELL_OPEN)
                elif t in ("1", "#", "W", "w", "X", "x"):
                    row_ints.append(CELL_WALL)
                else:
                    try:
                        val = int(t)
                        if val not in (CELL_OPEN, CELL_WALL):
                            raise ValueError(f"Invalid cell value {val} at row {line_idx}.")
                        row_ints.append(val)
                    except ValueError as err:
                        raise ValueError(f"Unrecognized cell token '{t}' at row {line_idx}: {err}") from err
            grid_matrix.append(row_ints)
        return validate_grid(grid_matrix)

    if isinstance(raw_input, (list, tuple)):
        if len(raw_input) == 0:
            raise ValueError("Grid cannot be an empty list.")
        grid_matrix = []
        for r_idx, row in enumerate(raw_input):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"Row {r_idx} must be a sequence (list or tuple).")
            row_ints = []
            for c_idx, cell in enumerate(row):
                if isinstance(cell, str):
                    c_clean = cell.strip()
                    if c_clean in ("0", ".", "O", "o"):
                        row_ints.append(CELL_OPEN)
                    elif c_clean in ("1", "#", "W", "w", "X", "x"):
                        row_ints.append(CELL_WALL)
                    else:
                        row_ints.append(int(c_clean))
                elif isinstance(cell, (int, float)):
                    row_ints.append(int(cell))
                else:
                    raise TypeError(f"Invalid cell type at ({r_idx}, {c_idx}): {type(cell).__name__}")
            grid_matrix.append(row_ints)
        return validate_grid(grid_matrix)

    raise TypeError(f"Unsupported grid input type: {type(raw_input).__name__}.")


def validate_grid(grid: List[List[int]]) -> List[List[int]]:
    """Validate that a grid is non-empty, rectangular, and contains only 0 and 1."""
    num_rows = len(grid)
    if num_rows == 0:
        raise ValueError("Grid must contain at least one row.")
    num_cols = len(grid[0])
    if num_cols == 0:
        raise ValueError("Grid rows cannot be empty.")

    for r_idx, row in enumerate(grid):
        if len(row) != num_cols:
            raise ValueError(
                f"Non-rectangular grid: Row 0 has {num_cols} columns, but Row {r_idx} has {len(row)} columns."
            )
        for c_idx, cell in enumerate(row):
            if cell not in (CELL_OPEN, CELL_WALL):
                raise ValueError(f"Cell at ({r_idx}, {c_idx}) has invalid value {cell}. Must be 0 or 1.")
    return grid


# ============================================================================
# Breadth-First Search (BFS) Solver with Exact Node Counter
# ============================================================================


def solve_maze_bfs(
    grid_input: Any,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
    record_trace: bool = True,
) -> SearchAlgorithmResult:
    """Solve 2D grid maze using Breadth-First Search (BFS), tracking exact node expansions.

    BFS Implementation Characteristics:
    - Frontier: FIFO queue (`collections.deque`), shallowest unexpanded states first.
    - Visited marking: Marked immediately upon ENQUEUEING (generation time) to prevent
      redundant duplicate states in the queue.
    - Optimality: Guaranteed shortest path (minimum transitions) for uniform step costs.
    - Neighbor order: Evaluated in the exact priority order passed (e.g. Up, Down, Left, Right).

    Args:
        grid_input: 2D list of integers, JSON string, or multiline text.
        start: Starting coordinate tuple (row, col). Defaults to (0, 0).
        goal: Destination coordinate tuple (row, col). Defaults to (N-1, M-1).
        directions: Neighbor priority order. Defaults to DEFAULT_DIRECTIONS.
        record_trace: Whether to capture step records and expansion order.

    Returns:
        SearchAlgorithmResult containing path length, path, nodes expanded, and timing.
    """
    start_time = time.perf_counter()
    grid = parse_grid(grid_input)
    num_rows = len(grid)
    num_cols = len(grid[0])

    resolved_start = (0, 0) if start is None else (int(start[0]), int(start[1]))
    resolved_goal = (num_rows - 1, num_cols - 1) if goal is None else (int(goal[0]), int(goal[1]))

    if not is_valid_coordinate(resolved_start[0], resolved_start[1], num_rows, num_cols):
        raise ValueError(f"Start coordinate {resolved_start} is out of grid bounds.")
    if not is_valid_coordinate(resolved_goal[0], resolved_goal[1], num_rows, num_cols):
        raise ValueError(f"Goal coordinate {resolved_goal} is out of grid bounds.")

    # Collision at start or goal
    if grid[resolved_start[0]][resolved_start[1]] == CELL_WALL or grid[resolved_goal[0]][resolved_goal[1]] == CELL_WALL:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchAlgorithmResult(
            algorithm="BFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            path_length=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=0,
            nodes_visited=0,
            peak_frontier_size=0,
            execution_time_ms=exec_ms,
            expansion_order={},
            trace=[],
        )

    # Base case: Start is Goal
    if resolved_start == resolved_goal:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchAlgorithmResult(
            algorithm="BFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=True,
            path_length=0,
            path=[resolved_start],
            actions=[],
            actions_str="",
            nodes_expanded=1,
            nodes_visited=1,
            peak_frontier_size=1,
            execution_time_ms=exec_ms,
            expansion_order={resolved_start: 1},
            trace=[
                MazeStepRecord(
                    step_number=1,
                    current_coord=resolved_start,
                    action_taken=None,
                    distance=0,
                    frontier_size=0,
                    visited_count=1,
                )
            ] if record_trace else [],
        )

    # FIFO Frontier Queue and Visited Set
    queue: deque[Tuple[int, int]] = deque([resolved_start])
    visited: Set[Tuple[int, int]] = {resolved_start}
    parent: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[str]]] = {
        resolved_start: (None, None)
    }
    distance: Dict[Tuple[int, int], int] = {resolved_start: 0}

    nodes_expanded = 0
    peak_frontier = 1
    expansion_order: Dict[Tuple[int, int], int] = {}
    trace_records: List[MazeStepRecord] = []
    goal_reached = False

    while queue:
        peak_frontier = max(peak_frontier, len(queue))
        curr = queue.popleft()
        nodes_expanded += 1
        curr_dist = distance[curr]
        _, curr_action = parent[curr]
        expansion_order[curr] = nodes_expanded

        if record_trace:
            trace_records.append(
                MazeStepRecord(
                    step_number=nodes_expanded,
                    current_coord=curr,
                    action_taken=curr_action,
                    distance=curr_dist,
                    frontier_size=len(queue),
                    visited_count=len(visited),
                )
            )

        # Early termination upon expanding the goal
        if curr == resolved_goal:
            goal_reached = True
            break

        # Generate orthogonal successors in strict priority order
        successors = get_successors(curr, grid, directions=directions)
        for action, next_coord in successors:
            if next_coord not in visited:
                visited.add(next_coord)
                parent[next_coord] = (curr, action)
                distance[next_coord] = curr_dist + 1
                queue.append(next_coord)

    exec_ms = (time.perf_counter() - start_time) * 1000.0

    if not goal_reached:
        return SearchAlgorithmResult(
            algorithm="BFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            path_length=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=nodes_expanded,
            nodes_visited=len(visited),
            peak_frontier_size=peak_frontier,
            execution_time_ms=exec_ms,
            expansion_order=expansion_order,
            trace=trace_records,
        )

    # Backtrack path from goal to start
    path_coords: List[Tuple[int, int]] = []
    path_actions: List[str] = []
    trace_node = resolved_goal
    while trace_node is not None:
        path_coords.append(trace_node)
        prev_node, act = parent[trace_node]
        if act is not None:
            path_actions.append(act)
        trace_node = prev_node  # type: ignore

    path_coords.reverse()
    path_actions.reverse()

    return SearchAlgorithmResult(
        algorithm="BFS",
        grid=grid,
        start=resolved_start,
        goal=resolved_goal,
        found=True,
        path_length=len(path_coords) - 1,
        path=path_coords,
        actions=path_actions,
        actions_str=", ".join(path_actions),
        nodes_expanded=nodes_expanded,
        nodes_visited=len(visited),
        peak_frontier_size=peak_frontier,
        execution_time_ms=exec_ms,
        expansion_order=expansion_order,
        trace=trace_records,
    )


# ============================================================================
# Depth-First Search (DFS) Solver with Exact Node Counter
# ============================================================================


def solve_maze_dfs(
    grid_input: Any,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
    record_trace: bool = True,
) -> SearchAlgorithmResult:
    """Solve 2D grid maze using Depth-First Search (DFS), tracking exact node expansions.

    DFS Implementation Characteristics:
    - Frontier: LIFO stack (`list.pop()`), deepest unexpanded states first.
    - Neighbor ordering preservation: To ensure that the top of the LIFO stack pops
      neighbors in the exact forward order (Up, then Down, then Left, then Right),
      successors are pushed onto the stack in reverse order (`reversed(successors)`).
      This mathematically produces the exact same traversal order as standard recursive DFS!
    - Non-Optimality: Returns the first path reaching the goal, highlighting arbitrary pathfinding.
    - Visited marking: Visited coordinates are marked upon push to prevent duplicate stack entries.

    Args:
        grid_input: 2D list of integers, JSON string, or multiline text.
        start: Starting coordinate tuple (row, col). Defaults to (0, 0).
        goal: Destination coordinate tuple (row, col). Defaults to (N-1, M-1).
        directions: Neighbor priority order. Defaults to DEFAULT_DIRECTIONS.
        record_trace: Whether to capture step records and expansion order.

    Returns:
        SearchAlgorithmResult containing path length, path, nodes expanded, and timing.
    """
    start_time = time.perf_counter()
    grid = parse_grid(grid_input)
    num_rows = len(grid)
    num_cols = len(grid[0])

    resolved_start = (0, 0) if start is None else (int(start[0]), int(start[1]))
    resolved_goal = (num_rows - 1, num_cols - 1) if goal is None else (int(goal[0]), int(goal[1]))

    if not is_valid_coordinate(resolved_start[0], resolved_start[1], num_rows, num_cols):
        raise ValueError(f"Start coordinate {resolved_start} is out of grid bounds.")
    if not is_valid_coordinate(resolved_goal[0], resolved_goal[1], num_rows, num_cols):
        raise ValueError(f"Goal coordinate {resolved_goal} is out of grid bounds.")

    # Collision at start or goal
    if grid[resolved_start[0]][resolved_start[1]] == CELL_WALL or grid[resolved_goal[0]][resolved_goal[1]] == CELL_WALL:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchAlgorithmResult(
            algorithm="DFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            path_length=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=0,
            nodes_visited=0,
            peak_frontier_size=0,
            execution_time_ms=exec_ms,
            expansion_order={},
            trace=[],
        )

    # Base case: Start is Goal
    if resolved_start == resolved_goal:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchAlgorithmResult(
            algorithm="DFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=True,
            path_length=0,
            path=[resolved_start],
            actions=[],
            actions_str="",
            nodes_expanded=1,
            nodes_visited=1,
            peak_frontier_size=1,
            execution_time_ms=exec_ms,
            expansion_order={resolved_start: 1},
            trace=[
                MazeStepRecord(
                    step_number=1,
                    current_coord=resolved_start,
                    action_taken=None,
                    distance=0,
                    frontier_size=0,
                    visited_count=1,
                )
            ] if record_trace else [],
        )

    # LIFO Frontier Stack and Visited Set
    stack: List[Tuple[int, int]] = [resolved_start]
    visited: Set[Tuple[int, int]] = {resolved_start}
    parent: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[str]]] = {
        resolved_start: (None, None)
    }
    distance: Dict[Tuple[int, int], int] = {resolved_start: 0}

    nodes_expanded = 0
    peak_frontier = 1
    expansion_order: Dict[Tuple[int, int], int] = {}
    trace_records: List[MazeStepRecord] = []
    goal_reached = False

    while stack:
        peak_frontier = max(peak_frontier, len(stack))
        curr = stack.pop()
        nodes_expanded += 1
        curr_dist = distance[curr]
        _, curr_action = parent[curr]
        expansion_order[curr] = nodes_expanded

        if record_trace:
            trace_records.append(
                MazeStepRecord(
                    step_number=nodes_expanded,
                    current_coord=curr,
                    action_taken=curr_action,
                    distance=curr_dist,
                    frontier_size=len(stack),
                    visited_count=len(visited),
                )
            )

        # Early termination upon expanding the goal
        if curr == resolved_goal:
            goal_reached = True
            break

        # Generate orthogonal successors in strict priority order
        successors = get_successors(curr, grid, directions=directions)

        # Push successors in REVERSED order onto the LIFO stack
        # This guarantees that the top of the stack pops Up first, then Down, Left, Right!
        for action, next_coord in reversed(successors):
            if next_coord not in visited:
                visited.add(next_coord)
                parent[next_coord] = (curr, action)
                distance[next_coord] = curr_dist + 1
                stack.append(next_coord)

    exec_ms = (time.perf_counter() - start_time) * 1000.0

    if not goal_reached:
        return SearchAlgorithmResult(
            algorithm="DFS",
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            path_length=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=nodes_expanded,
            nodes_visited=len(visited),
            peak_frontier_size=peak_frontier,
            execution_time_ms=exec_ms,
            expansion_order=expansion_order,
            trace=trace_records,
        )

    # Backtrack path from goal to start
    path_coords: List[Tuple[int, int]] = []
    path_actions: List[str] = []
    trace_node = resolved_goal
    while trace_node is not None:
        path_coords.append(trace_node)
        prev_node, act = parent[trace_node]
        if act is not None:
            path_actions.append(act)
        trace_node = prev_node  # type: ignore

    path_coords.reverse()
    path_actions.reverse()

    return SearchAlgorithmResult(
        algorithm="DFS",
        grid=grid,
        start=resolved_start,
        goal=resolved_goal,
        found=True,
        path_length=len(path_coords) - 1,
        path=path_coords,
        actions=path_actions,
        actions_str=", ".join(path_actions),
        nodes_expanded=nodes_expanded,
        nodes_visited=len(visited),
        peak_frontier_size=peak_frontier,
        execution_time_ms=exec_ms,
        expansion_order=expansion_order,
        trace=trace_records,
    )


# ============================================================================
# Comparative Analyzer: Encapsulated Benchmark Engine
# ============================================================================


def compare_maze_solvers(
    grid_input: Any = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
    record_trace: bool = True,
) -> ComparisonResult:
    """Run both BFS and DFS under identical conditions and compile a comparative summary.

    Args:
        grid_input: 2D list or string representing the maze. Defaults to DEFAULT_10X10_MAZE.
        start: Starting coordinate tuple. Defaults to DEFAULT_START_COORD (0, 0).
        goal: Destination coordinate tuple. Defaults to DEFAULT_GOAL_COORD (5, 7).
        directions: Move ordering. Defaults to DEFAULT_DIRECTIONS.
        record_trace: Whether to capture step-by-step traces.

    Returns:
        ComparisonResult containing both results and comparative metrics.
    """
    if grid_input is None:
        grid_input = DEFAULT_10X10_MAZE
    if start is None:
        start = DEFAULT_START_COORD
    if goal is None:
        goal = DEFAULT_GOAL_COORD

    bfs_res = solve_maze_bfs(
        grid_input, start=start, goal=goal, directions=directions, record_trace=record_trace
    )
    dfs_res = solve_maze_dfs(
        grid_input, start=start, goal=goal, directions=directions, record_trace=record_trace
    )

    manhattan = calculate_manhattan_distance(bfs_res.start, bfs_res.goal)
    exp_diff = bfs_res.nodes_expanded - dfs_res.nodes_expanded
    exp_ratio = (
        bfs_res.nodes_expanded / dfs_res.nodes_expanded
        if dfs_res.nodes_expanded > 0
        else float("inf")
    )

    if bfs_res.found and dfs_res.found:
        len_diff = dfs_res.path_length - bfs_res.path_length
        opt_ratio = (
            dfs_res.path_length / bfs_res.path_length
            if bfs_res.path_length > 0
            else 1.0
        )
        bfs_shorter = bfs_res.path_length < dfs_res.path_length
    else:
        len_diff = 0
        opt_ratio = 1.0
        bfs_shorter = False

    sample_str = (
        f"BFS Nodes Expanded: {bfs_res.nodes_expanded}, Path Length: {bfs_res.path_length} | "
        f"DFS Nodes Expanded: {dfs_res.nodes_expanded}, Path Length: {dfs_res.path_length}"
    )

    return ComparisonResult(
        bfs=bfs_res,
        dfs=dfs_res,
        manhattan_distance=manhattan,
        expanded_difference=exp_diff,
        expanded_ratio=exp_ratio,
        path_length_difference=len_diff,
        optimality_ratio=opt_ratio,
        is_bfs_strictly_shorter=bfs_shorter,
        sample_output_formatted=sample_str,
    )


# ============================================================================
# Matplotlib Visualizer (Optional Challenge)
# ============================================================================


def plot_bfs_vs_dfs_expansion(
    comparison: ComparisonResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title_suffix: str = "",
) -> Any:
    """Plot the expansion process and solution paths dynamically using matplotlib.

    Visualizes:
    - Left Subplot: BFS expansion order heatmap + BFS shortest optimal path overlay.
    - Center Subplot: DFS expansion order heatmap + DFS arbitrary winding path overlay.
    - Right Subplot: Performance Metrics Bar Chart comparing nodes expanded, path length,
      and peak frontier memory.

    Args:
        comparison: ComparisonResult object returned by `compare_maze_solvers`.
        save_path: Optional file path to save the generated figure (e.g., 'reports/maze.png').
        show: Whether to invoke plt.show() (disabled by default for non-blocking runs).
        title_suffix: Optional extra text for the figure title.

    Returns:
        Matplotlib Figure object, or None if matplotlib is not installed.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Warning] Matplotlib or NumPy is not installed. Skipping plot generation.")
        return None

    bfs_res = comparison.bfs
    dfs_res = comparison.dfs
    grid = bfs_res.grid
    num_rows = len(grid)
    num_cols = len(grid[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    title = "BFS vs. DFS Maze Performance Analyzer: Optimality vs. Arbitrary Pathfinding"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # 1. Helper to render maze heatmaps
    def render_search_panel(ax: Any, result: SearchAlgorithmResult, title: str, path_color: str) -> None:
        # Base matrix: walls are negative (-1)
        matrix = np.zeros((num_rows, num_cols), dtype=float)
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] == CELL_WALL:
                    matrix[r, c] = -1.0
                elif (r, c) in result.expansion_order:
                    matrix[r, c] = float(result.expansion_order[(r, c)])
                else:
                    matrix[r, c] = 0.0

        # Mask walls
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps["viridis"].copy()
        elif hasattr(plt.cm, "get_cmap"):
            cmap = plt.cm.get_cmap("viridis").copy()
        else:
            cmap = plt.get_cmap("viridis").copy()

        if hasattr(cmap, "with_extremes"):
            cmap = cmap.with_extremes(under="black")
        else:
            cmap.set_under("black")

        vmin = 1.0 if result.nodes_expanded > 0 else 0.0
        vmax = float(max(result.nodes_expanded, 1))

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Node Expansion Step", rotation=270, labelpad=12)

        # Plot path
        if result.found and len(result.path) > 1:
            pr = [p[0] for p in result.path]
            pc = [p[1] for p in result.path]
            ax.plot(pc, pr, color=path_color, linewidth=3.5, linestyle="-", label=f"Path ({len(result.path)-1} steps)", zorder=4)

        # Plot Start and Goal
        sr, sc = result.start
        gr, gc = result.goal
        ax.plot(sc, sr, marker="o", markersize=11, color="lime", markeredgecolor="darkgreen", markeredgewidth=2, label="Start", zorder=5)
        ax.plot(gc, gr, marker="*", markersize=16, color="red", markeredgecolor="black", markeredgewidth=1.5, label="Goal", zorder=5)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    # Left: BFS Panel
    render_search_panel(
        axes[0],
        bfs_res,
        f"BFS (Optimal Shortest Path)\nExpanded: {bfs_res.nodes_expanded} | Path: {bfs_res.path_length} steps",
        path_color="gold",
    )

    # Center: DFS Panel
    render_search_panel(
        axes[1],
        dfs_res,
        f"DFS (Arbitrary Pathfinding)\nExpanded: {dfs_res.nodes_expanded} | Path: {dfs_res.path_length} steps",
        path_color="cyan",
    )

    # Right: Comparative Bar Chart
    ax_bar = axes[2]
    categories = ["Nodes Expanded", "Path Length", "Peak Memory (Cells)"]
    bfs_values = [bfs_res.nodes_expanded, max(bfs_res.path_length, 0), bfs_res.peak_frontier_size]
    dfs_values = [dfs_res.nodes_expanded, max(dfs_res.path_length, 0), dfs_res.peak_frontier_size]

    x = np.arange(len(categories))
    bar_width = 0.35

    rects1 = ax_bar.bar(x - bar_width / 2, bfs_values, bar_width, label="BFS", color="#2ca02c")
    rects2 = ax_bar.bar(x + bar_width / 2, dfs_values, bar_width, label="DFS", color="#1f77b4")

    ax_bar.set_ylabel("Count / Cells")
    ax_bar.set_title("Empirical Metric Comparison", fontsize=12, fontweight="bold", pad=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
    ax_bar.legend(loc="upper right", framealpha=0.85)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with values
    for rect in rects1:
        height = rect.get_height()
        ax_bar.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects2:
        height = rect.get_height()
        ax_bar.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[Success] Saved comparative expansion visualization to: {out_path.resolve()}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Viva Voce Theory Guide & Reference
# ============================================================================


def get_viva_reference() -> Dict[str, Any]:
    """Return the structured Viva Voce examination reference and theoretical answers.

    Topics Covered:
    1. Algorithm Optimality
    2. Scenarios Where DFS Outperforms BFS
    3. Branching Factor Impact
    """
    return {
        "title": "BFS vs DFS Performance Analyzer Viva Voce Guide",
        "syllabus_area": "Uninformed Search Techniques",
        "core_concept": "Algorithm Comparison, Time/Space Efficiency, and Optimality Guarantees",
        "viva_topics": {
            "algorithm_optimality": {
                "question": "Why is BFS guaranteed to be optimal for uniform-cost grid mazes, whereas DFS is not?",
                "bfs_optimality_proof": (
                    "In a grid maze where all orthogonal transitions have uniform unit cost (c = 1), "
                    "the path cost g(n) is strictly equivalent to the node depth d. BFS uses a First-In, "
                    "First-Out (FIFO) queue, guaranteeing that all nodes at depth d are expanded before "
                    "any node at depth d + 1. Therefore, the first time the goal state is selected for "
                    "expansion, it is mathematically guaranteed to be via a path of minimal length."
                ),
                "dfs_non_optimality": (
                    "DFS uses a Last-In, First-Out (LIFO) stack, venturing deeply down a single path until "
                    "a dead-end or cycle is encountered. It has no depth-level synchronization. If an arbitrary "
                    "deep path reaches the goal first (e.g. winding through 22 steps), DFS immediately returns "
                    "that path, completely oblivious to a much shorter 12-step path lying unexpanded in an "
                    "adjacent shallow branch."
                ),
            },
            "scenarios_where_dfs_outperforms_bfs": {
                "question": "In what practical scenarios does DFS significantly outperform BFS?",
                "scenarios": [
                    {
                        "name": "Goal Resides Deep Along First Explored Subtree",
                        "explanation": (
                            "If the goal happens to reside deep along the very branch explored first by the "
                            "neighbor ordering (e.g. Down, Right), DFS plunges directly to the goal in O(d) "
                            "expansions without exploring any other sibling branches. BFS, in contrast, must "
                            "exhaustively expand all branches level-by-level across the entire state space, "
                            "generating O(b^d) nodes (e.g. DFS takes 14 expansions vs BFS's 45 expansions)."
                        ),
                    },
                    {
                        "name": "Strict Memory-Constrained Environments",
                        "explanation": (
                            "In large or infinite state spaces, BFS suffers an exponential memory explosion "
                            "O(b^d), quickly crashing due to Out-Of-Memory (OOM) errors. DFS requires only "
                            "O(b * m) linear memory (storing only active branch ancestors and unexplored siblings), "
                            "making DFS executable on embedded systems and constrained hardware."
                        ),
                    },
                    {
                        "name": "Dense Solution Spaces (Finding Any Path Rapidly)",
                        "explanation": (
                            "When a problem contains many valid goal states or paths (e.g., game tree search, "
                            "n-queens, or satisfiability) and any valid solution is acceptable, DFS frequently "
                            "stumbles upon a solution far quicker than BFS's exhaustive spherical wave exploration."
                        ),
                    },
                ],
            },
            "branching_factor_impact": {
                "question": "What is the branching factor 'b', and how does it impact BFS vs DFS differently?",
                "definition": (
                    "The branching factor 'b' is the average number of successor states generated from each node. "
                    "On a 4-connected grid without walls, b = 4; with obstacles, 1 <= b <= 4."
                ),
                "impact_on_bfs": (
                    "BFS memory and time scale exponentially with branching factor: Memory = O(b^d). "
                    "If b increases from 2 to 4, the number of nodes at depth 12 explodes from ~4,096 to "
                    "~16,777,216 nodes! BFS becomes computationally intractable very quickly."
                ),
                "impact_on_dfs": (
                    "DFS time complexity is O(b^m), but its SPACE complexity remains linear: O(b * m), where "
                    "m is maximum search depth. Increasing branching factor b only scales DFS memory linearly, "
                    "retaining feasibility even when b is large."
                ),
            },
        },
        "complexity_matrix": {
            "BFS": {
                "time": "O(b^d)",
                "space": "O(b^d) [Exponential]",
                "complete": "Yes (if b is finite)",
                "optimal": "Yes (for uniform step costs)",
            },
            "DFS": {
                "time": "O(b^m)",
                "space": "O(b * m) [Linear]",
                "complete": "No (in infinite spaces; Yes in finite graphs with visited set)",
                "optimal": "No (Arbitrary pathfinding)",
            },
        },
    }
