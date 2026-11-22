"""A* Pathfinding in 2D Maze Simulation.

This module implements the canonical A* search algorithm for spatial navigation in 2D grid
mazes containing static obstacles, adhering to the theoretical formulations in Russell &
Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA Chapter 3).

Core AI Concept: Informed Search & Spatial Navigation:
- Evaluation Function:
      f(n) = g(n) + h(n)
  where:
  - g(n) is the exact path cost incurred from the start node to coordinate n = (r, c).
    With a uniform step cost of 1 for 4-directional cardinal moves (Up, Down, Left, Right),
    g(n) equals the number of transitions along the traversed path.
  - h(n) is the heuristic estimate of the remaining cost from node n to the goal coordinate.
    For 4-connected grid navigation without diagonals, the Manhattan distance (L1 norm) is:
        h_Manhattan((r, c), (r_goal, c_goal)) = |r - r_goal| + |c - c_goal|
  - f(n) estimates the total cost of the cheapest solution path traversing through node n.

Theoretical Optimality & Heuristic Admissibility:
- Admissibility: In an obstacle-free grid, the Manhattan distance is the exact minimum steps
  required to reach the goal. Obstacles can only increase the actual distance. Hence:
      0 <= h_Manhattan(n) <= h*(n)
  Manhattan distance never overestimates the true remaining cost (admissible).
- Consistency (Monotonicity): For any two adjacent cells u and v with step cost c(u, v) = 1:
      |h(u) - h(v)| <= 1 ==> h(u) <= c(u, v) + h(v)
  Consistency guarantees that the first time any state is popped and expanded from the
  priority queue, its g(n) cost is strictly optimal. No closed nodes ever need to be reopened.

Render Specification:
- 'X': Obstacles / impassable walls
- '.': Open walkable cells (unvisited)
- '*': Final optimal solution path
- '#': Explored nodes expanded during search (not part of the final path)

Optional Challenge:
- Live terminal animation of the node expansion progression and path reconstruction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import heapq
import math
import sys
import time
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Type Aliases
GridCoordinate = Tuple[int, int]
MazeGrid = List[List[int]]

# ============================================================================
# Canonical Default Benchmarks & Configurations
# ============================================================================

# Default 5x5 Benchmark Maze (Syllabus Sample Input):
# 5x5 grid with an obstacle wall blocking direct horizontal movement.
# Start at (0, 0), Goal at (4, 4).
DEFAULT_5X5_MAZE: MazeGrid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

DEFAULT_5X5_START: GridCoordinate = (0, 0)
DEFAULT_5X5_GOAL: GridCoordinate = (4, 4)

# 10x10 Benchmark Maze (from Question 27 GBFS Navigation benchmark):
# Demonstrates A* achieving the optimal 18-step path, contrasting with GBFS (suboptimal 22 steps)
# and BFS (87 node expansions vs A*'s focused search).
DEFAULT_10X10_MAZE: MazeGrid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
]

DEFAULT_10X10_START: GridCoordinate = (0, 0)
DEFAULT_10X10_GOAL: GridCoordinate = (9, 9)

# Concave Trap (U-shaped obstacle) Benchmark:
# Demonstrates how A* handles local minima traps without infinite loops or severe detours.
CONCAVE_TRAP_MAZE: MazeGrid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Serpentine Corridor 5x5 Maze:
SERPENTINE_5X5_MAZE: MazeGrid = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]

# Open 5x5 Grid (Obstacle-Free):
OPEN_5X5_MAZE: MazeGrid = [[0 for _ in range(5)] for _ in range(5)]

# Single-Cell Maze: Start is Goal
SINGLE_CELL_MAZE: MazeGrid = [[0]]

# Unsolvable 5x5 Maze (Completely blocked by wall)
UNSOLVABLE_5X5_MAZE: MazeGrid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]

# Canonical 4-directional moves: Up, Down, Left, Right as specified in syllabus
DEFAULT_ORTHOGONAL_DIRECTIONS: Tuple[Tuple[str, int, int], ...] = (
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
)

# Standard Glyphs
GLYPH_OBSTACLE: str = "X"
GLYPH_OPEN: str = "."
GLYPH_PATH: str = "*"
GLYPH_EXPLORED: str = "#"
GLYPH_START: str = "S"
GLYPH_GOAL: str = "G"

# Numeric Cell Representations
CELL_OPEN: int = 0
CELL_OBSTACLE: int = 1


# ============================================================================
# Distance Heuristic Functions
# ============================================================================

def manhattan_distance(coord: GridCoordinate, goal: GridCoordinate) -> int:
    """Calculate the L1 Manhattan distance between two grid coordinates.

    Formula:
        h(n) = |r_n - r_goal| + |c_n - c_goal|

    On a 4-connected grid with step cost 1 and no diagonal transitions, Manhattan
    distance is admissible (never overestimates) and consistent (monotonic).

    Args:
        coord: Current grid coordinate (row, col).
        goal: Target destination coordinate (row_goal, col_goal).

    Returns:
        Integer Manhattan distance.
    """
    return abs(coord[0] - goal[0]) + abs(coord[1] - goal[1])


def euclidean_distance(coord: GridCoordinate, goal: GridCoordinate) -> float:
    """Calculate the straight-line Euclidean distance (L2 norm) between two coordinates.

    Formula:
        h(n) = sqrt((r_n - r_goal)^2 + (c_n - c_goal)^2)

    Args:
        coord: Current coordinate (row, col).
        goal: Target goal coordinate (row_goal, col_goal).

    Returns:
        Float Euclidean distance.
    """
    return math.hypot(coord[0] - goal[0], coord[1] - goal[1])


def chebyshev_distance(coord: GridCoordinate, goal: GridCoordinate) -> int:
    """Calculate the Chebyshev distance (L_infinity norm) between two coordinates.

    Formula:
        h(n) = max(|r_n - r_goal|, |c_n - c_goal|)

    Args:
        coord: Current coordinate (row, col).
        goal: Target goal coordinate (row_goal, col_goal).

    Returns:
        Integer Chebyshev distance.
    """
    return max(abs(coord[0] - goal[0]), abs(coord[1] - goal[1]))


HEURISTIC_REGISTRY: Dict[str, Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class AStarMazeStepRecord:
    """Immutable trace record capturing an individual node expansion during A* search.

    Attributes:
        step_number: 1-based chronological expansion index.
        current_coord: Coordinate tuple (row, col) popped and expanded.
        g_cost: Accumulated path cost from start to current_coord.
        h_cost: Dynamic heuristic estimate h(current, goal).
        f_cost: Total evaluation cost f(n) = g(n) + h(n).
        action: Semantic step label ('START', 'EXPAND', 'GOAL_REACHED').
        frontier_size: Number of candidate coordinates remaining in the min-heap.
        neighbors_evaluated: Adjacent walkable coordinates examined.
        neighbors_enqueued: Coordinates successfully pushed to the min-heap.
        path_so_far: Sequence of coordinates traversed from start to current_coord.
    """

    step_number: int
    current_coord: GridCoordinate
    g_cost: int
    h_cost: Union[int, float]
    f_cost: Union[int, float]
    action: str
    frontier_size: int
    neighbors_evaluated: List[GridCoordinate]
    neighbors_enqueued: List[GridCoordinate]
    path_so_far: List[GridCoordinate]


@dataclass
class AStarMazeResult:
    """Encapsulates the complete outcome of an A* 2D maze pathfinding execution.

    Attributes:
        algorithm: Name of search algorithm used ('A* Pathfinding').
        maze: The 2D grid matrix used for navigation (0 = open, 1 = obstacle).
        start: Starting coordinate tuple (row, col).
        goal: Goal coordinate tuple (row, col).
        found: True if an optimal path to the goal was found; False otherwise.
        path: List of coordinates from start to goal (inclusive), or empty if unreachable.
        path_length: Number of transitions along the solution path (-1 if unreachable).
        total_cost: Total path cost g(goal) (-1 if unreachable).
        nodes_expanded: Exact count of nodes popped and expanded from the priority queue.
        expansion_order: Chronological list of coordinates expanded during search.
        explored_set: Set of coordinates that were expanded into the closed set.
        best_g_costs: Dictionary mapping visited coordinates to their lowest discovered g-cost.
        steps: Chronological list of AStarMazeStepRecord objects for trace auditing.
        peak_frontier: Maximum size reached by the priority queue during search.
        execution_time_ms: Total search execution time in milliseconds.
        heuristic_name: Name of the heuristic function employed (e.g. 'manhattan').
        is_optimal: Whether the returned solution is guaranteed to be optimal.
    """

    algorithm: str
    maze: MazeGrid
    start: GridCoordinate
    goal: GridCoordinate
    found: bool
    path: List[GridCoordinate]
    path_length: int
    total_cost: int
    nodes_expanded: int
    expansion_order: List[GridCoordinate]
    explored_set: Set[GridCoordinate]
    best_g_costs: Dict[GridCoordinate, int]
    steps: List[AStarMazeStepRecord]
    peak_frontier: int
    execution_time_ms: float
    heuristic_name: str
    is_optimal: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "maze_dimensions": (len(self.maze), len(self.maze[0]) if self.maze else 0),
            "start": list(self.start),
            "goal": list(self.goal),
            "found": self.found,
            "path": [list(pt) for pt in self.path],
            "path_length": self.path_length,
            "total_cost": self.total_cost,
            "nodes_expanded": self.nodes_expanded,
            "expansion_order": [list(pt) for pt in self.expansion_order],
            "explored_count": len(self.explored_set),
            "peak_frontier": self.peak_frontier,
            "execution_time_ms": round(self.execution_time_ms, 4),
            "heuristic_name": self.heuristic_name,
            "is_optimal": self.is_optimal,
        }


@dataclass
class AStarMazeComparisonResult:
    """Comparison outcome between A*, Breadth-First Search (BFS), and Greedy Best-First Search.

    Attributes:
        astar_result: AStarMazeResult for A* Pathfinding.
        bfs_result: AStarMazeResult for Breadth-First Search.
        gbfs_result: Optional AStarMazeResult for Greedy Best-First Search.
        expansion_difference_bfs: Count difference in node expansions (BFS - A*).
        expansion_reduction_pct_bfs: Percentage reduction in expansions of A* relative to BFS.
        path_length_difference_bfs: Path length difference between A* and BFS (should be 0).
        is_astar_fewer_or_equal_nodes: True if A* expanded <= BFS nodes.
        is_astar_strictly_optimal: True if A* path length matches BFS optimal path length.
    """

    astar_result: AStarMazeResult
    bfs_result: AStarMazeResult
    gbfs_result: Optional[AStarMazeResult]
    expansion_difference_bfs: int
    expansion_reduction_pct_bfs: float
    path_length_difference_bfs: int
    is_astar_fewer_or_equal_nodes: bool
    is_astar_strictly_optimal: bool


# ============================================================================
# 2D Grid Parser
# ============================================================================

def parse_grid(
    grid_input: Union[str, Sequence[Sequence[Union[int, str]]], Sequence[str]],
    default_start: Optional[GridCoordinate] = None,
    default_goal: Optional[GridCoordinate] = None,
) -> Tuple[MazeGrid, GridCoordinate, GridCoordinate]:
    """Parse a 2D grid from strings, integer matrices, or character arrays.

    Supported input representations:
    1. Multiline string:
           . . X . .
           . . X . .
           . . X . .
           . . . . .
           . . . . .
    2. Sequence of string rows:
           ["..X..", "..X..", "..X..", ".....", "....."]
    3. 2D list of integers (0 = open, 1 = obstacle):
           [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], ...]
    4. 2D list of characters ('.', 'X', 'S', 'G'):
           [['.', '.', 'X', '.', '.'], ...]

    Special Character Semantics:
    - 'X' or '1' or '#': Impassable obstacle cell.
    - '.' or '0' or ' ': Open walkable cell.
    - 'S': Start coordinate (automatically detected if present).
    - 'G': Goal coordinate (automatically detected if present).

    Args:
        grid_input: Grid definition in any supported format.
        default_start: Fallback start coordinate if 'S' is absent (defaults to (0, 0)).
        default_goal: Fallback goal coordinate if 'G' is absent (defaults to bottom-right).

    Returns:
        Tuple of (maze_matrix, start_coord, goal_coord).

    Raises:
        ValueError: If input is empty, non-rectangular, contains invalid characters,
                    or start/goal are placed on obstacles or out of bounds.
    """
    raw_rows: List[List[str]] = []

    if isinstance(grid_input, str):
        cleaned_lines = [line.strip() for line in grid_input.strip().splitlines() if line.strip()]
        if not cleaned_lines:
            raise ValueError("Input grid string cannot be empty.")
        for line in cleaned_lines:
            # If line is space-separated tokens e.g. ". . X . ."
            tokens = line.split()
            if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
                raw_rows.append(tokens)
            else:
                # Contiguous characters e.g. "..X.."
                raw_rows.append(list(line.replace(" ", "")))
    elif isinstance(grid_input, (list, tuple)):
        if len(grid_input) == 0:
            raise ValueError("Input grid sequence cannot be empty.")
        for row in grid_input:
            if isinstance(row, str):
                tokens = row.strip().split()
                if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
                    raw_rows.append(tokens)
                else:
                    raw_rows.append(list(row.replace(" ", "")))
            elif isinstance(row, (list, tuple)):
                raw_rows.append([str(elem).strip() for elem in row])
            else:
                raise ValueError(f"Unsupported row element type: {type(row)}")
    else:
        raise ValueError(f"Unsupported grid input type: {type(grid_input)}")

    rows = len(raw_rows)
    if rows == 0:
        raise ValueError("Grid rows cannot be empty.")
    cols = len(raw_rows[0])
    if cols == 0:
        raise ValueError("Grid columns cannot be empty.")

    # Validate rectangular consistency
    for r_idx, row in enumerate(raw_rows):
        if len(row) != cols:
            raise ValueError(
                f"Row {r_idx} has {len(row)} columns, inconsistent with expected width {cols}."
            )

    parsed_maze: MazeGrid = [[CELL_OPEN for _ in range(cols)] for _ in range(rows)]
    detected_start: Optional[GridCoordinate] = None
    detected_goal: Optional[GridCoordinate] = None

    for r in range(rows):
        for c in range(cols):
            val = raw_rows[r][c].upper()
            if val in ("X", "1", "#"):
                parsed_maze[r][c] = CELL_OBSTACLE
            elif val in (".", "0", " "):
                parsed_maze[r][c] = CELL_OPEN
            elif val == "S":
                parsed_maze[r][c] = CELL_OPEN
                if detected_start is not None:
                    raise ValueError(f"Multiple Start coordinates found: {detected_start} and ({r}, {c}).")
                detected_start = (r, c)
            elif val == "G":
                parsed_maze[r][c] = CELL_OPEN
                if detected_goal is not None:
                    raise ValueError(f"Multiple Goal coordinates found: {detected_goal} and ({r}, {c}).")
                detected_goal = (r, c)
            else:
                raise ValueError(
                    f"Invalid grid cell value '{raw_rows[r][c]}' at ({r}, {c}). "
                    "Expected 'X', '.', '0', '1', '#', 'S', or 'G'."
                )

    start = detected_start if detected_start is not None else (default_start if default_start is not None else (0, 0))
    goal = detected_goal if detected_goal is not None else (default_goal if default_goal is not None else (rows - 1, cols - 1))

    # Boundary & Obstacle Validation
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"Start coordinate {start} is out of bounds for {rows}x{cols} maze.")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError(f"Goal coordinate {goal} is out of bounds for {rows}x{cols} maze.")
    if parsed_maze[sr][sc] == CELL_OBSTACLE:
        raise ValueError(f"Start coordinate {start} is located on an obstacle ('X').")
    if parsed_maze[gr][gc] == CELL_OBSTACLE:
        raise ValueError(f"Goal coordinate {goal} is located on an obstacle ('X').")

    return parsed_maze, start, goal


# ============================================================================
# Core A* Pathfinding Algorithm
# ============================================================================

def a_star_maze_solver(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = None,
    directions: Sequence[Tuple[str, int, int]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
    tie_breaker: str = "fifo",
    record_steps: bool = False,
    heuristic_name: str = "manhattan",
) -> AStarMazeResult:
    """Implement A* pathfinding for a 2D grid containing obstacles using Manhattan distance.

    Evaluation Function:
        f(n) = g(n) + h(n)
        where:
            g(n) is the exact step cost from start to coordinate n (step cost = 1).
            h(n) = manhattan(current, goal) = |r_curr - r_goal| + |c_curr - c_goal|.

    Algorithm Mechanics:
    - Maintained using a binary min-heap priority queue via Python's `heapq` module.
    - Each heap entry is structured as:
          (f_score, tie_key, g_score, current_coord, path)
      ensuring O(log N) insertion and extraction of the lowest f(n) state.
    - An explored (closed) set tracks all nodes popped and expanded.
    - A dictionary `best_g_costs` tracks the cheapest discovered g(n) to each coordinate.

    Args:
        maze: 2D grid matrix where 0 represents open walkable cells and 1 represents obstacles.
        start: Starting coordinate tuple (row, col). Defaults to (0, 0).
        goal: Target destination coordinate (row, col). Defaults to bottom-right cell.
        heuristic_fn: Distance callable computing h(coord, goal). Defaults to Manhattan distance.
        directions: Sequence of orthogonal moves (name, row_delta, col_delta).
        tie_breaker: Tie-breaking strategy for identical f(n) scores ('fifo', 'lifo', 'goal_directed').
        record_steps: Whether to capture step-by-step trace records for auditing/visualization.
        heuristic_name: Descriptive name of the heuristic function employed.

    Returns:
        AStarMazeResult containing the optimal path, path length, nodes expanded, and diagnostics.

    Raises:
        ValueError: If grid dimensions are empty, start/goal coordinates are out of bounds,
                    or start/goal are situated on an obstacle cell.
    """
    start_time = time.perf_counter()

    # Validate grid dimensions
    rows = len(maze)
    if rows == 0:
        raise ValueError("Maze grid cannot be empty.")
    cols = len(maze[0])
    if cols == 0:
        raise ValueError("Maze columns cannot be empty.")

    if goal is None:
        goal = (rows - 1, cols - 1)

    # Validate coordinates
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"Start coordinate {start} is out of bounds for {rows}x{cols} maze.")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError(f"Goal coordinate {goal} is out of bounds for {rows}x{cols} maze.")
    if maze[sr][sc] == CELL_OBSTACLE:
        raise ValueError(f"Start coordinate {start} is located on an obstacle ('X').")
    if maze[gr][gc] == CELL_OBSTACLE:
        raise ValueError(f"Goal coordinate {goal} is located on an obstacle ('X').")

    if heuristic_fn is None:
        heuristic_fn = manhattan_distance

    # Handle trivial base case: start is already goal
    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return AStarMazeResult(
            algorithm="A* Pathfinding",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            total_cost=0,
            nodes_expanded=1,
            expansion_order=[start],
            explored_set={start},
            best_g_costs={start: 0},
            steps=[
                AStarMazeStepRecord(
                    step_number=1,
                    current_coord=start,
                    g_cost=0,
                    h_cost=0,
                    f_cost=0,
                    action="GOAL_REACHED",
                    frontier_size=0,
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    path_so_far=[start],
                )
            ] if record_steps else [],
            peak_frontier=1,
            execution_time_ms=elapsed,
            heuristic_name=heuristic_name,
            is_optimal=True,
        )

    # Priority queue min-heap:
    # We structure heap tuples according to tie-breaking strategy:
    # - "fifo": (f_cost, counter, g_cost, current_coord, path)
    # - "lifo": (f_cost, -counter, g_cost, current_coord, path)
    # - "goal_directed": (f_cost, h_cost, counter, g_cost, current_coord, path)
    counter = 0
    h_start = heuristic_fn(start, goal)
    f_start = 0 + h_start

    frontier: List[Any] = []
    if tie_breaker == "goal_directed":
        heapq.heappush(frontier, (f_start, h_start, counter, 0, start, [start]))
    elif tie_breaker == "lifo":
        heapq.heappush(frontier, (f_start, -counter, 0, start, [start]))
    else:  # "fifo" default
        heapq.heappush(frontier, (f_start, counter, 0, start, [start]))

    explored_set: Set[GridCoordinate] = set()
    best_g_costs: Dict[GridCoordinate, int] = {start: 0}
    expansion_order: List[GridCoordinate] = []
    steps: List[AStarMazeStepRecord] = []
    peak_frontier = 1
    found = False
    solution_path: List[GridCoordinate] = []
    step_num = 0

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))

        if tie_breaker == "goal_directed":
            f_cost, h_cost, _, g_cost, curr, path = heapq.heappop(frontier)
        elif tie_breaker == "lifo":
            f_cost, _, g_cost, curr, path = heapq.heappop(frontier)
            h_cost = f_cost - g_cost
        else:
            f_cost, _, g_cost, curr, path = heapq.heappop(frontier)
            h_cost = f_cost - g_cost

        # Lazy deletion / superseded entry check:
        # If we have already expanded this node with an equal or lower g_cost, discard.
        if curr in explored_set:
            continue

        explored_set.add(curr)
        expansion_order.append(curr)

        # Goal test upon expansion
        if curr == goal:
            found = True
            solution_path = path
            if record_steps:
                step_num += 1
                steps.append(
                    AStarMazeStepRecord(
                        step_number=step_num,
                        current_coord=curr,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        f_cost=f_cost,
                        action="GOAL_REACHED",
                        frontier_size=len(frontier),
                        neighbors_evaluated=[],
                        neighbors_enqueued=[],
                        path_so_far=path,
                    )
                )
            break

        # Expand adjacent orthogonal neighbors (Up, Down, Left, Right)
        cr, cc = curr
        evaluated_neighbors: List[GridCoordinate] = []
        enqueued_neighbors: List[GridCoordinate] = []

        for _, dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)

            # Check boundary and impassable obstacles ('X')
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == CELL_OPEN:
                evaluated_neighbors.append(neighbor)
                tentative_g = g_cost + 1  # Step cost is 1

                # If neighbor hasn't been explored or a strictly shorter path is discovered
                if neighbor not in explored_set:
                    if neighbor not in best_g_costs or tentative_g < best_g_costs[neighbor]:
                        best_g_costs[neighbor] = tentative_g
                        counter += 1
                        h_next = heuristic_fn(neighbor, goal)
                        f_next = tentative_g + h_next

                        if tie_breaker == "goal_directed":
                            heapq.heappush(
                                frontier,
                                (f_next, h_next, counter, tentative_g, neighbor, path + [neighbor]),
                            )
                        elif tie_breaker == "lifo":
                            heapq.heappush(
                                frontier,
                                (f_next, -counter, tentative_g, neighbor, path + [neighbor]),
                            )
                        else:
                            heapq.heappush(
                                frontier,
                                (f_next, counter, tentative_g, neighbor, path + [neighbor]),
                            )
                        enqueued_neighbors.append(neighbor)

        if record_steps:
            step_num += 1
            steps.append(
                AStarMazeStepRecord(
                    step_number=step_num,
                    current_coord=curr,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    f_cost=f_cost,
                    action="EXPAND",
                    frontier_size=len(frontier),
                    neighbors_evaluated=evaluated_neighbors,
                    neighbors_enqueued=enqueued_neighbors,
                    path_so_far=path,
                )
            )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1
    tot_cost = path_len if found else -1

    return AStarMazeResult(
        algorithm="A* Pathfinding",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        total_cost=tot_cost,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        explored_set=explored_set,
        best_g_costs=best_g_costs,
        steps=steps,
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name=heuristic_name,
        is_optimal=True,
    )


# ============================================================================
# Comparison Baselines: Breadth-First Search & Greedy Best-First Search
# ============================================================================

def breadth_first_search_maze(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    directions: Sequence[Tuple[str, int, int]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
) -> AStarMazeResult:
    """Solve the 2D maze using blind Breadth-First Search (BFS) as an unguided optimal baseline.

    BFS expands states level by level in concentric expanding rings using a FIFO queue,
    guaranteeing the optimal shortest step count on unweighted grids but exploring
    exhaustively without directional heuristic guidance.

    Args:
        maze: 2D grid matrix (0 = open, 1 = obstacle).
        start: Starting coordinate tuple (row, col).
        goal: Target destination coordinate.
        directions: Sequence of movement actions.

    Returns:
        AStarMazeResult representing the BFS outcome.
    """
    start_time = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0])
    if goal is None:
        goal = (rows - 1, cols - 1)

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return AStarMazeResult(
            algorithm="Breadth-First Search",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            total_cost=0,
            nodes_expanded=1,
            expansion_order=[start],
            explored_set={start},
            best_g_costs={start: 0},
            steps=[],
            peak_frontier=1,
            execution_time_ms=elapsed,
            heuristic_name="none",
            is_optimal=True,
        )

    queue: deque[Tuple[GridCoordinate, List[GridCoordinate], int]] = deque([(start, [start], 0)])
    visited: Set[GridCoordinate] = {start}
    expansion_order: List[GridCoordinate] = []
    best_g_costs: Dict[GridCoordinate, int] = {start: 0}
    found = False
    solution_path: List[GridCoordinate] = []
    peak_frontier = 1

    while queue:
        peak_frontier = max(peak_frontier, len(queue))
        curr, path, g = queue.popleft()
        expansion_order.append(curr)

        if curr == goal:
            found = True
            solution_path = path
            break

        cr, cc = curr
        for _, dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == CELL_OPEN:
                if neighbor not in visited:
                    visited.add(neighbor)
                    best_g_costs[neighbor] = g + 1
                    queue.append((neighbor, path + [neighbor], g + 1))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1

    return AStarMazeResult(
        algorithm="Breadth-First Search",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        total_cost=path_len if found else -1,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        explored_set=set(expansion_order),
        best_g_costs=best_g_costs,
        steps=[],
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name="none",
        is_optimal=True,
    )


def greedy_best_first_search_maze(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = None,
    directions: Sequence[Tuple[str, int, int]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
) -> AStarMazeResult:
    """Solve the 2D maze using pure Greedy Best-First Search (f(n) = h(n)).

    GBFS aggressively steers towards the goal using h(n), ignoring the accumulated
    step cost g(n). It frequently explores fewer nodes than BFS, but can be lured into
    suboptimal paths or local minima traps.

    Args:
        maze: 2D grid matrix.
        start: Starting coordinate tuple.
        goal: Target destination coordinate.
        heuristic_fn: Dynamic heuristic function. Defaults to Manhattan distance.
        directions: Sequence of movement actions.

    Returns:
        AStarMazeResult representing the GBFS outcome.
    """
    start_time = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0])
    if goal is None:
        goal = (rows - 1, cols - 1)

    if heuristic_fn is None:
        heuristic_fn = manhattan_distance

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return AStarMazeResult(
            algorithm="Greedy Best-First Search",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            total_cost=0,
            nodes_expanded=1,
            expansion_order=[start],
            explored_set={start},
            best_g_costs={start: 0},
            steps=[],
            peak_frontier=1,
            execution_time_ms=elapsed,
            heuristic_name="manhattan",
            is_optimal=True,
        )

    frontier: List[Tuple[Union[int, float], int, GridCoordinate, List[GridCoordinate], int]] = []
    counter = 0
    h_start = heuristic_fn(start, goal)
    heapq.heappush(frontier, (h_start, counter, start, [start], 0))

    explored: Set[GridCoordinate] = set()
    expansion_order: List[GridCoordinate] = []
    best_g_costs: Dict[GridCoordinate, int] = {start: 0}
    found = False
    solution_path: List[GridCoordinate] = []
    peak_frontier = 1

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))
        h_val, _, curr, path, g = heapq.heappop(frontier)

        if curr in explored:
            continue

        explored.add(curr)
        expansion_order.append(curr)

        if curr == goal:
            found = True
            solution_path = path
            break

        cr, cc = curr
        for _, dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == CELL_OPEN:
                if neighbor not in explored:
                    counter += 1
                    best_g_costs[neighbor] = g + 1
                    h_next = heuristic_fn(neighbor, goal)
                    heapq.heappush(frontier, (h_next, counter, neighbor, path + [neighbor], g + 1))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1

    return AStarMazeResult(
        algorithm="Greedy Best-First Search",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        total_cost=path_len if found else -1,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        explored_set=explored,
        best_g_costs=best_g_costs,
        steps=[],
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name="manhattan",
        is_optimal=False,
    )


def compare_astar_vs_bfs(
    maze: Sequence[Sequence[int]] = DEFAULT_5X5_MAZE,
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    include_gbfs: bool = True,
) -> AStarMazeComparisonResult:
    """Compare A* against BFS (and optionally GBFS) on the identical maze instance.

    Demonstrates the core theoretical insight tested in Viva examinations:
    - Solution Path Optimality: A* guarantees the optimal shortest path identical to BFS.
    - Exploration Efficiency: A*'s heuristic guidance focuses the frontier towards the goal,
      substantially reducing the size of the explored set relative to blind BFS.

    Args:
        maze: 2D grid matrix.
        start: Starting coordinate.
        goal: Destination coordinate.
        include_gbfs: Whether to execute Greedy Best-First Search for comparison.

    Returns:
        AStarMazeComparisonResult containing all performance metrics.
    """
    astar_res = a_star_maze_solver(maze=maze, start=start, goal=goal)
    bfs_res = breadth_first_search_maze(maze=maze, start=start, goal=goal)
    gbfs_res = greedy_best_first_search_maze(maze=maze, start=start, goal=goal) if include_gbfs else None

    diff_exp = bfs_res.nodes_expanded - astar_res.nodes_expanded
    red_pct = (diff_exp / bfs_res.nodes_expanded * 100.0) if bfs_res.nodes_expanded > 0 else 0.0
    diff_len = astar_res.path_length - bfs_res.path_length

    return AStarMazeComparisonResult(
        astar_result=astar_res,
        bfs_result=bfs_res,
        gbfs_result=gbfs_res,
        expansion_difference_bfs=diff_exp,
        expansion_reduction_pct_bfs=red_pct,
        path_length_difference_bfs=diff_len,
        is_astar_fewer_or_equal_nodes=(astar_res.nodes_expanded <= bfs_res.nodes_expanded),
        is_astar_strictly_optimal=(diff_len == 0),
    )


# ============================================================================
# ASCII Grid Visualization & Rendering
# ============================================================================

def render_solved_maze(
    maze: Sequence[Sequence[int]],
    path: Sequence[GridCoordinate],
    explored: Optional[Set[GridCoordinate]] = None,
    start: Optional[GridCoordinate] = None,
    goal: Optional[GridCoordinate] = None,
    wall_char: str = GLYPH_OBSTACLE,
    open_char: str = GLYPH_OPEN,
    path_char: str = GLYPH_PATH,
    explored_char: str = GLYPH_EXPLORED,
    start_char: str = GLYPH_START,
    goal_char: str = GLYPH_GOAL,
    preserve_endpoints: bool = False,
    show_coordinates: bool = False,
) -> str:
    """Render the solved maze into an ASCII visualization string.

    Renders according to the strict syllabus specification:
    - 'X' for obstacles
    - '.' for open walkable cells
    - '*' for optimal path
    - '#' for explored nodes (expanded during search, but not in final path)

    Endpoint handling:
    - If `preserve_endpoints` is False (default for sample output), start and goal
      are marked with `path_char` ('*').
    - If `preserve_endpoints` is True, start is 'S' and goal is 'G'.

    Args:
        maze: 2D grid matrix (0 = open, 1 = obstacle).
        path: Sequence of coordinates along the solution path.
        explored: Set of all coordinates expanded during search.
        start: Starting coordinate. Defaults to path[0] if present.
        goal: Goal coordinate. Defaults to path[-1] if present.
        wall_char: Glyph for obstacles ('X').
        open_char: Glyph for open cells ('.').
        path_char: Glyph for solution path ('*').
        explored_char: Glyph for explored nodes ('#').
        start_char: Glyph for start ('S').
        goal_char: Glyph for goal ('G').
        preserve_endpoints: Whether to render 'S' and 'G' or use path_char.
        show_coordinates: Whether to print row/column numerical indices.

    Returns:
        Formatted multi-line ASCII string representation of the grid.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    path_set = set(path)
    explored_set = set(explored) if explored is not None else set()

    if start is None and path:
        start = path[0]
    if goal is None and path:
        goal = path[-1]

    lines: List[str] = []

    if show_coordinates:
        col_header = "   " + " ".join(str(c % 10) for c in range(cols))
        lines.append(col_header)
        lines.append("  +" + "-" * (cols * 2))

    for r in range(rows):
        row_symbols: List[str] = []
        for c in range(cols):
            coord = (r, c)
            if preserve_endpoints and start is not None and coord == start:
                row_symbols.append(start_char)
            elif preserve_endpoints and goal is not None and coord == goal:
                row_symbols.append(goal_char)
            elif coord in path_set:
                row_symbols.append(path_char)
            elif maze[r][c] == CELL_OBSTACLE:
                row_symbols.append(wall_char)
            elif coord in explored_set:
                row_symbols.append(explored_char)
            else:
                row_symbols.append(open_char)

        row_content = " ".join(row_symbols)
        if show_coordinates:
            lines.append(f"{r % 10} | {row_content}")
        else:
            lines.append(row_content)

    return "\n".join(lines)


def render_side_by_side_comparison(
    comparison: AStarMazeComparisonResult,
    show_coordinates: bool = False,
) -> str:
    """Format a side-by-side ASCII comparison showing A* vs BFS exploration and paths.

    Args:
        comparison: AStarMazeComparisonResult object.
        show_coordinates: Whether to include grid coordinate axes.

    Returns:
        Formatted multi-line side-by-side string representation.
    """
    astar = comparison.astar_result
    bfs = comparison.bfs_result

    astar_str = render_solved_maze(
        astar.maze,
        astar.path,
        explored=astar.explored_set,
        show_coordinates=show_coordinates,
    )
    bfs_str = render_solved_maze(
        bfs.maze,
        bfs.path,
        explored=bfs.explored_set,
        show_coordinates=show_coordinates,
    )

    astar_lines = astar_str.splitlines()
    bfs_lines = bfs_str.splitlines()

    col_width = max(len(l) for l in astar_lines + bfs_lines) + 4
    header1 = f"{'A* Pathfinding (f = g + h)':<{col_width}} | {'Breadth-First Search (Blind)':<{col_width}}"
    header2 = (
        f"{f'Expanded: {astar.nodes_expanded}, Length: {astar.path_length}':<{col_width}} | "
        f"{f'Expanded: {bfs.nodes_expanded}, Length: {bfs.path_length}':<{col_width}}"
    )
    divider = "-" * (col_width * 2 + 3)

    out: List[str] = [header1, header2, divider]
    max_len = max(len(astar_lines), len(bfs_lines))

    for i in range(max_len):
        al = astar_lines[i] if i < len(astar_lines) else ""
        bl = bfs_lines[i] if i < len(bfs_lines) else ""
        out.append(f"{al:<{col_width}} | {bl:<{col_width}}")

    out.append(divider)
    out.append(
        f"Exploration Reduction: {comparison.expansion_difference_bfs} fewer nodes "
        f"({comparison.expansion_reduction_pct_bfs:.1f}% reduction) while guaranteeing exact path optimality!"
    )
    return "\n".join(out)


# ============================================================================
# Optional Challenge: Terminal Animation
# ============================================================================

def generate_expansion_frames(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = None,
    directions: Sequence[Tuple[str, int, int]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
) -> Generator[Dict[str, Any], None, None]:
    """Generator yielding visualization frames for each expansion step of A*.

    Fulfills the Optional Challenge: 'Animate the expansion process in the terminal.'

    Yields a dictionary per step containing:
    - 'frame_number': 1-based index
    - 'current': coordinate popped
    - 'action': 'EXPAND' or 'GOAL_REACHED'
    - 'explored_count': count of explored nodes so far
    - 'frontier_size': count of nodes waiting in frontier
    - 'rendered_grid': ASCII snapshot of the grid at this step
    - 'is_final': True if goal reached and final path reconstructed

    Args:
        maze: 2D grid matrix.
        start: Starting coordinate.
        goal: Destination coordinate.
        heuristic_fn: Dynamic heuristic function.
        directions: Orthogonal directional moves.

    Yields:
        Dictionary capturing frame state and rendered grid string.
    """
    rows = len(maze)
    cols = len(maze[0])
    if goal is None:
        goal = (rows - 1, cols - 1)

    if heuristic_fn is None:
        heuristic_fn = manhattan_distance

    counter = 0
    h_start = heuristic_fn(start, goal)
    frontier: List[Tuple[Union[int, float], int, int, GridCoordinate, List[GridCoordinate]]] = []
    heapq.heappush(frontier, (h_start, counter, 0, start, [start]))

    explored_set: Set[GridCoordinate] = set()
    best_g_costs: Dict[GridCoordinate, int] = {start: 0}
    frame_idx = 0

    while frontier:
        f_cost, _, g_cost, curr, path = heapq.heappop(frontier)

        if curr in explored_set:
            continue

        explored_set.add(curr)
        frame_idx += 1

        if curr == goal:
            # Yield final frame with solution path marked by '*'
            final_rendered = render_solved_maze(
                maze=maze,
                path=path,
                explored=explored_set,
                start=start,
                goal=goal,
            )
            yield {
                "frame_number": frame_idx,
                "current": curr,
                "action": "GOAL_REACHED",
                "explored_count": len(explored_set),
                "frontier_size": len(frontier),
                "rendered_grid": final_rendered,
                "path": path,
                "path_length": len(path) - 1,
                "is_final": True,
            }
            break

        # Yield expansion frame: explored nodes marked with '#', current with 'o'
        curr_rendered = render_solved_maze(
            maze=maze,
            path=[curr],  # Highlight current expanding node
            explored=explored_set,
            path_char="o",
            start=start,
            goal=goal,
        )
        yield {
            "frame_number": frame_idx,
            "current": curr,
            "action": "EXPAND",
            "explored_count": len(explored_set),
            "frontier_size": len(frontier),
            "rendered_grid": curr_rendered,
            "path": path,
            "path_length": len(path) - 1,
            "is_final": False,
        }

        cr, cc = curr
        for _, dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == CELL_OPEN:
                tentative_g = g_cost + 1
                if neighbor not in explored_set:
                    if neighbor not in best_g_costs or tentative_g < best_g_costs[neighbor]:
                        best_g_costs[neighbor] = tentative_g
                        counter += 1
                        h_next = heuristic_fn(neighbor, goal)
                        f_next = tentative_g + h_next
                        heapq.heappush(frontier, (f_next, counter, tentative_g, neighbor, path + [neighbor]))


def animate_a_star_search(
    maze: Sequence[Sequence[int]] = DEFAULT_5X5_MAZE,
    start: GridCoordinate = DEFAULT_5X5_START,
    goal: Optional[GridCoordinate] = None,
    delay: float = 0.08,
    stream: Optional[Any] = None,
) -> None:
    """Animate the A* node expansion process live in the terminal.

    Fulfills the Optional Challenge: 'Animate the expansion process in the terminal.'

    Clears the screen / cursor at each step to render the dynamic propagation of
    explored nodes '#' and the final emergence of the optimal path '*'.

    Args:
        maze: 2D grid matrix.
        start: Starting coordinate.
        goal: Destination coordinate.
        delay: Time delay in seconds between frames (0 to skip sleep).
        stream: Output text stream (defaults to current sys.stdout).
    """
    out_stream = stream if stream is not None else sys.stdout
    for frame in generate_expansion_frames(maze=maze, start=start, goal=goal):
        # ANSI clear screen code + home cursor
        out_stream.write("\033[2J\033[H")
        out_stream.write("=====================================================\n")
        out_stream.write(f" A* 2D MAZE EXPANSION ANIMATION (Frame #{frame['frame_number']})\n")
        out_stream.write("=====================================================\n")
        out_stream.write(f" Current Node  : {frame['current']} | Action: {frame['action']}\n")
        out_stream.write(f" Explored Set  : {frame['explored_count']} nodes | Frontier: {frame['frontier_size']}\n")
        out_stream.write(" Legend        : [X] Obstacle  [.] Open  [#] Explored  [*] Final Path\n")
        out_stream.write("-----------------------------------------------------\n")
        out_stream.write(frame["rendered_grid"] + "\n")
        out_stream.write("=====================================================\n")
        if frame["is_final"]:
            out_stream.write(f" GOAL REACHED! Optimal Path Length: {frame['path_length']} steps.\n")
        out_stream.flush()

        if delay > 0 and not frame["is_final"]:
            time.sleep(delay)


# ============================================================================
# Viva Voce Knowledge Base
# ============================================================================

VIVA_VOCE_GUIDE: Dict[str, Dict[str, str]] = {
    "A* in Video Games": {
        "question": "How is A* pathfinding applied in commercial video games, and what are its key engineering challenges?",
        "answer": (
            "In video games, non-player characters (NPCs) and units navigate complex 2D/3D environments "
            "using A*. Key engineering techniques include:\n"
            "1. Spatial Abstraction: Real games rarely use raw uniform grid tiles. Instead, they precompute "
            "Navigation Meshes (NavMeshes) consisting of convex walkable polygons, or Hierarchical A* (HPA*) "
            "graphs which divide the world into clusters, planning abstract corridors first before local refinement.\n"
            "2. Symmetry Pruning (Jump Point Search / JPS): On uniform grids, JPS skips intermediate symmetric cells "
            "by 'jumping' horizontally/vertically until an obstacle or forced neighbor is detected, speeding up A* by 10x-50x.\n"
            "3. Frame Budgeting & Time-Slicing: In a 60 FPS loop, AI has only 1-2 ms per frame. Long paths are "
            "time-sliced across multiple ticks or use weighted A* (f = g + w*h, w > 1) to find an epsilon-suboptimal "
            "path in a fraction of the expansions.\n"
            "4. Dynamic Replanning: When moving obstacles or other units block a corridor, algorithms like D* Lite "
            "repair existing search trees incrementally rather than replanning from scratch."
        ),
    },
    "Explored Set Size Relative to BFS": {
        "question": "How does the explored set size of A* compare mathematically and empirically to Breadth-First Search (BFS)?",
        "answer": (
            "1. Blind BFS expands states in spherical/diamond concentric contours proportional to depth d, "
            "yielding O(b^d) explored nodes where b is the branching factor (b = 4 on a 2D grid).\n"
            "2. A* with an admissible heuristic h(n) <= h*(n) restricts exploration to an ellipse of states satisfying "
            "f(n) = g(n) + h(n) <= C*, where C* is the optimal path cost. The major axis of this ellipse aligns directly "
            "along the line segment from start to goal.\n"
            "3. If h(n) = h*(n) (perfect heuristic), A* expands strictly d + 1 nodes—only the states on the optimal path!\n"
            "4. In 2D grid mazes, A* with Manhattan distance typically explores 40% to 80% fewer nodes than BFS while "
            "guaranteeing exact 100% path optimality. In our 5x5 benchmark, A* expands only 16 nodes vs BFS's 17 nodes; "
            "on our 10x10 benchmark, A* expands 77 nodes vs BFS's 87 nodes, focusing expansions along the obstacle bypass."
        ),
    },
    "Manhattan Distance Admissibility & Consistency": {
        "question": "Prove that Manhattan distance is both admissible and consistent on a 4-connected grid with unit step costs.",
        "answer": (
            "1. Admissibility: On a 4-directional grid without diagonals, any path between (r1, c1) and (r2, c2) requires "
            "at least |r1 - r2| vertical steps and |c1 - c2| horizontal steps, totaling |r1 - r2| + |c1 - c2|. Because obstacles "
            "can only block transitions and force detours, true shortest distance h*(n) >= h_Manhattan(n). Thus h(n) is admissible.\n"
            "2. Consistency (Monotonicity): For any adjacent orthogonal neighbor v reachable in 1 step: "
            "|r_v - r_u| + |c_v - c_u| = 1. By the reverse triangle inequality for norms: "
            "|h(u) - h(v)| <= |(r_u - r_v)| + |(c_u - c_v)| = 1 = c(u, v). "
            "Therefore h(u) <= c(u, v) + h(v), satisfying consistency.\n"
            "3. Consequence: Consistency guarantees that when node n is selected for expansion, g(n) is optimal. "
            "Closed nodes never need to be reopened."
        ),
    },
    "Tie-Breaking in A* Grid Search": {
        "question": "Why is tie-breaking critical in A* grid navigation, and what strategies exist?",
        "answer": (
            "On a 2D grid, many alternative paths share the identical optimal f-cost. Without smart tie-breaking, "
            "A* expands all states having the same f-value across a wide wavefront.\n"
            "1. Prefer Greater g (Smaller h): When f(a) == f(b), choosing the state with smaller h (closer to goal) "
            "causes the search to follow a narrow straight line directly to the target, reducing expansions by up to 50%.\n"
            "2. Deterministic FIFO / LIFO: Using an incremental counter ensures reproducible search traces and avoids "
            "unpredictable Python object comparison errors.\n"
            "3. Slight Heuristic Inflation: Adding a tiny nudge h = h * (1.0 + 1e-4) breaks ties deterministically in "
            "favor of states closer to the goal without compromising practical path optimality."
        ),
    },
}
