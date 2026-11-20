"""Greedy Best-First Search (GBFS) 2D Grid Navigator.

This module implements Greedy Best-First Search (GBFS) for navigating 2D grid mazes,
demonstrating the behavior and limitations of pure heuristic search as detailed in
Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA Chapter 3).

Core AI Concept: Informed Search & Pure Heuristic Evaluation:
- Evaluation Function:
      f(n) = h(n)
  where h(n) is the heuristic estimate of the distance from coordinate n = (r, c)
  to the target goal state (r_goal, c_goal).
- Pure Heuristic Search: Unlike Uniform Cost Search (f(n) = g(n)) and A* Search
  (f(n) = g(n) + h(n)), Greedy Best-First Search completely ignores the accumulated
  step cost g(n) incurred so far.
- Greediness & Suboptimality:
  At each step, GBFS expands the node that appears to be closest to the goal.
  While this aggressive steering often explores significantly fewer nodes than blind
  search algorithms such as Breadth-First Search (BFS), it can easily be misled by
  obstacles into choosing suboptimal (longer) paths or walking into dead-end traps
  (local minima).

Dynamic Heuristic Calculation:
- For 4-directional grid navigation, the Manhattan distance (L1 norm) is the canonical
  natural heuristic:
      h_Manhattan((r, c), (r_goal, c_goal)) = |r - r_goal| + |c - c_goal|
  It is calculated dynamically at runtime using the destination goal coordinates.
- Additional supported distance metrics:
  - Euclidean distance (L2 norm): sqrt((r - r_goal)^2 + (c - c_goal)^2)
  - Chebyshev distance (L_inf norm): max(|r - r_goal|, |c - c_goal|)

Priority Queue & Frontier Mechanics:
- The frontier is maintained as a binary min-heap using Python's standard `heapq` module.
- Each heap entry is structured as:
      (h_value, tie_breaker_key, current_coord, path)
  This guarantees O(log N) insertion and extraction of the minimum-heuristic cell.
- Deterministic FIFO tie-breaking is provided using an incremental insertion counter.

Complexity:
- Time Complexity: O(N log N) in standard grid topologies where N = Rows * Cols.
  In the worst-case infinite state space, O(b^m).
- Space Complexity: O(N) auxiliary space to maintain the frontier heap and visited set.

Optional Challenge:
- Renders the 2D grid maze with the discovered path marked by '*' characters,
  distinguishing open cells ('.'), walls ('#'), Start ('S'), and Goal ('G').
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Type Aliases
GridCoordinate = Tuple[int, int]
MazeGrid = List[List[int]]

# ============================================================================
# Canonical Default Benchmarks
# ============================================================================

# Default 10x10 Benchmark Maze:
# Demonstrates Greedy Best-First Search expanding only 25 nodes and taking a
# suboptimal 22-step path, whereas BFS explores 87 nodes to find the optimal 18-step path.
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

DEFAULT_START_COORD: GridCoordinate = (0, 0)
DEFAULT_GOAL_COORD: GridCoordinate = (9, 9)

# Concave Trap (U-shaped obstacle) Benchmark:
# Demonstrates Greedy Search entering a local minimum trap where h(n) continuously decreases
# towards the back of the cul-de-sac, forcing extensive backtracking to escape.
CONCAVE_TRAP_10X10_MAZE: MazeGrid = [
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

# Open 10x10 Grid (Zero Obstacles):
OPEN_10X10_MAZE: MazeGrid = [[0 for _ in range(10)] for _ in range(10)]

# Unsolvable 10x10 Maze: Completely severed by a solid wall barrier
UNSOLVABLE_10X10_MAZE: MazeGrid = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
]

# Movement Directions: (Action Name, row_delta, col_delta, symbol)
# Canonical order: Down, Right, Up, Left
DEFAULT_ORTHOGONAL_DIRECTIONS: Tuple[Tuple[str, int, int, str], ...] = (
    ("Down", 1, 0, "v"),
    ("Right", 0, 1, ">"),
    ("Up", -1, 0, "^"),
    ("Left", 0, -1, "<"),
)


# ============================================================================
# Dynamic Heuristic Functions
# ============================================================================

def manhattan_distance(coord: GridCoordinate, goal: GridCoordinate) -> int:
    """Calculate the L1 Manhattan distance between two grid coordinates.

    Formula:
        h(n) = |r_n - r_goal| + |c_n - c_goal|

    On a 4-connected grid without diagonal transitions, Manhattan distance is the
    exact shortest distance in an obstacle-free environment and represents an
    admissible lower bound when obstacles are present.

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
class GridStepRecord:
    """Immutable record capturing a single step in the grid search progression.

    Attributes:
        step_number: 1-based chronological index of the expansion.
        current_coord: Coordinate tuple popped from the priority queue.
        heuristic_val: The h(n) value computed dynamically for the popped coordinate.
        action: Label of the step ('EXPAND', 'GOAL_REACHED', 'SKIP_EXPLORED').
        frontier_size: Number of candidate coordinates remaining in the frontier.
        neighbors_evaluated: Coordinates of adjacent valid cells examined.
        neighbors_enqueued: Coordinates of non-explored cells pushed to the heap.
        path_so_far: Sequence of coordinates traversed to reach current_coord.
    """

    step_number: int
    current_coord: GridCoordinate
    heuristic_val: float
    action: str
    frontier_size: int
    neighbors_evaluated: List[GridCoordinate]
    neighbors_enqueued: List[GridCoordinate]
    path_so_far: List[GridCoordinate]


@dataclass
class GridSearchResult:
    """Encapsulates the complete outcome of a grid navigation search.

    Attributes:
        algorithm: Name of search algorithm used (e.g. 'Greedy Best-First Search', 'BFS').
        maze: The 2D grid matrix used for navigation.
        start: Starting coordinate tuple (row, col).
        goal: Goal coordinate tuple (row, col).
        found: True if a path to the goal was found; False otherwise.
        path: List of coordinates from start to goal (inclusive), or empty if not found.
        path_length: Number of steps/transitions in the path (-1 if unreachable).
        nodes_expanded: Exact count of nodes popped and expanded from the frontier.
        expansion_order: Chronological list of coordinates expanded during search.
        steps: Chronological list of GridStepRecord objects for trace auditing.
        peak_frontier: Maximum size reached by the frontier during search.
        execution_time_ms: Total search execution time in milliseconds.
        heuristic_name: Name of the heuristic function employed.
        is_suboptimal: Optional flag marking path suboptimality when compared to ground truth.
    """

    algorithm: str
    maze: MazeGrid
    start: GridCoordinate
    goal: GridCoordinate
    found: bool
    path: List[GridCoordinate]
    path_length: int
    nodes_expanded: int
    expansion_order: List[GridCoordinate]
    steps: List[GridStepRecord] = field(default_factory=list)
    peak_frontier: int = 0
    execution_time_ms: float = 0.0
    heuristic_name: str = "manhattan"
    is_suboptimal: Optional[bool] = None

    def format_sample_output(self) -> str:
        """Format the output matching the university practical syllabus standard.

        Example:
            'Expanded: 25 nodes, Path Length: 22 (Suboptimal)'
        """
        if not self.found:
            return f"Expanded: {self.nodes_expanded} nodes, Path: Not Found (Unreachable)"
        subopt_suffix = " (Suboptimal)" if self.is_suboptimal else ""
        return f"Expanded: {self.nodes_expanded} nodes, Path Length: {self.path_length}{subopt_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result into a serializable dictionary representation."""
        return {
            "algorithm": self.algorithm,
            "start": list(self.start),
            "goal": list(self.goal),
            "found": self.found,
            "path_length": self.path_length,
            "path_coordinates": [list(c) for c in self.path],
            "nodes_expanded": self.nodes_expanded,
            "expansion_order": [list(c) for c in self.expansion_order],
            "peak_frontier": self.peak_frontier,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "heuristic_name": self.heuristic_name,
            "is_suboptimal": self.is_suboptimal,
        }


@dataclass
class GBFSGridComparisonResult:
    """Comparative analysis result between Greedy Best-First Search and BFS / A*.

    Attributes:
        gbfs_result: GridSearchResult for Greedy Best-First Search.
        bfs_result: GridSearchResult for Breadth-First Search (baseline shortest path).
        a_star_result: Optional GridSearchResult for A* Search.
        expansion_difference: BFS nodes expanded minus GBFS nodes expanded.
        expansion_reduction_pct: Percentage reduction in expansions achieved by GBFS.
        path_length_difference: GBFS path length minus BFS optimal path length.
        suboptimality_ratio: Ratio of GBFS path length to BFS path length.
        is_gbfs_strictly_fewer_nodes: True if GBFS expanded fewer nodes than BFS.
        is_gbfs_suboptimal: True if GBFS produced a strictly longer path than BFS.
    """

    gbfs_result: GridSearchResult
    bfs_result: GridSearchResult
    a_star_result: Optional[GridSearchResult] = None
    expansion_difference: int = 0
    expansion_reduction_pct: float = 0.0
    path_length_difference: int = 0
    suboptimality_ratio: float = 1.0
    is_gbfs_strictly_fewer_nodes: bool = False
    is_gbfs_suboptimal: bool = False

    def format_summary_table(self) -> str:
        """Format an educational comparison table for terminal display and lab records."""
        lines = [
            "=" * 78,
            "Greedy Best-First Search vs. Breadth-First Search (10x10 Maze Navigation)",
            "=" * 78,
            f"{'Metric':<28} | {'Greedy Best-First Search':<22} | {'Breadth-First Search (BFS)':<22}",
            "-" * 78,
            f"{'Evaluation Strategy':<28} | {'f(n) = h(n) [Pure Heuristic]':<22} | {'f(n) = depth [Uninformed]':<22}",
            f"{'Heuristic Used':<28} | {self.gbfs_result.heuristic_name.capitalize():<22} | {'None (Uniform Step Cost)':<22}",
            f"{'Nodes Expanded':<28} | {self.gbfs_result.nodes_expanded:<22} | {self.bfs_result.nodes_expanded:<22}",
            f"{'Solution Path Length':<28} | {self.gbfs_result.path_length:<22} | {self.bfs_result.path_length:<22}",
            f"{'Path Optimality':<28} | {'Suboptimal' if self.is_gbfs_suboptimal else 'Optimal':<22} | {'Guaranteed Shortest':<22}",
            f"{'Peak Frontier Size':<28} | {self.gbfs_result.peak_frontier:<22} | {self.bfs_result.peak_frontier:<22}",
            "-" * 78,
            f"Fewer Nodes Explored?       : {'YES' if self.is_gbfs_strictly_fewer_nodes else 'NO'} "
            f"({self.expansion_difference} fewer expansions, {self.expansion_reduction_pct:.1f}% reduction)",
            f"Suboptimal Path Chosen?     : {'YES' if self.is_gbfs_suboptimal else 'NO'} "
            f"(+{self.path_length_difference} extra steps, ratio: {self.suboptimality_ratio:.2f}x optimal)",
            "=" * 78,
        ]
        return "\n".join(lines)


# ============================================================================
# Core Search Algorithms
# ============================================================================

def greedy_best_first_search_grid(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_START_COORD,
    goal: GridCoordinate = DEFAULT_GOAL_COORD,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = None,
    directions: Sequence[Tuple[str, int, int, str]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
    tie_breaker: str = "fifo",
    record_steps: bool = False,
    heuristic_name: str = "manhattan",
) -> GridSearchResult:
    """Navigate a 2D maze using Greedy Best-First Search with dynamic heuristic estimates.

    The algorithm evaluates nodes purely by their heuristic value h(n) to the goal coordinate:
        f(n) = h(n)
    using a binary min-heap priority queue via Python's `heapq` module.

    Args:
        maze: 2D matrix where 0 represents open walkable cells and 1 represents obstacles.
        start: Starting coordinate tuple (row, col).
        goal: Target goal coordinate tuple (row, col).
        heuristic_fn: Callable computing distance between (coord, goal). Defaults to Manhattan.
        directions: Sequence of directional moves (name, dr, dc, symbol).
        tie_breaker: Strategy for breaking ties when h(a) == h(b). Options: 'fifo', 'lifo'.
        record_steps: Whether to record step-by-step trace records.
        heuristic_name: Descriptive name of the heuristic used.

    Returns:
        GridSearchResult containing path, length, expansion count, and diagnostics.

    Raises:
        ValueError: If start or goal coordinates are out of bounds or on an obstacle.
    """
    start_time = time.perf_counter()

    # Grid dimension validation
    rows = len(maze)
    if rows == 0:
        raise ValueError("Maze grid cannot be empty.")
    cols = len(maze[0])
    if cols == 0:
        raise ValueError("Maze columns cannot be empty.")

    # Coordinate boundary & obstacle validation
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"Start coordinate {start} is out of bounds for {rows}x{cols} maze.")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError(f"Goal coordinate {goal} is out of bounds for {rows}x{cols} maze.")
    if maze[sr][sc] == 1:
        raise ValueError(f"Start coordinate {start} is located on an impassable obstacle (1).")
    if maze[gr][gc] == 1:
        raise ValueError(f"Goal coordinate {goal} is located on an impassable obstacle (1).")

    if heuristic_fn is None:
        heuristic_fn = manhattan_distance

    # Handle base case: start coordinate is already the goal
    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GridSearchResult(
            algorithm="Greedy Best-First Search",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            nodes_expanded=1,
            expansion_order=[start],
            steps=[
                GridStepRecord(
                    step_number=1,
                    current_coord=start,
                    heuristic_val=0.0,
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
            is_suboptimal=False,
        )

    # Priority Queue Heap Elements: (h(n), tie_breaker_counter, current_coord, path)
    frontier: List[Tuple[Union[int, float], int, GridCoordinate, List[GridCoordinate]]] = []
    counter = 0

    def get_tie_key(c: int) -> int:
        return -c if tie_breaker == "lifo" else c

    h_start = heuristic_fn(start, goal)
    heapq.heappush(frontier, (h_start, get_tie_key(counter), start, [start]))

    explored: Set[GridCoordinate] = set()
    expansion_order: List[GridCoordinate] = []
    steps: List[GridStepRecord] = []
    peak_frontier = 1
    found = False
    solution_path: List[GridCoordinate] = []
    step_num = 0

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))
        h_val, _, current, path = heapq.heappop(frontier)

        if current in explored:
            continue

        explored.add(current)
        expansion_order.append(current)

        # Goal test upon expansion
        if current == goal:
            found = True
            solution_path = path
            if record_steps:
                step_num += 1
                steps.append(
                    GridStepRecord(
                        step_number=step_num,
                        current_coord=current,
                        heuristic_val=float(h_val),
                        action="GOAL_REACHED",
                        frontier_size=len(frontier),
                        neighbors_evaluated=[],
                        neighbors_enqueued=[],
                        path_so_far=path,
                    )
                )
            break

        # Expand orthogonal neighbors
        cr, cc = current
        evaluated_neighbors: List[GridCoordinate] = []
        enqueued_neighbors: List[GridCoordinate] = []

        for _, dr, dc, _ in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                evaluated_neighbors.append(neighbor)
                if neighbor not in explored:
                    counter += 1
                    h_next = heuristic_fn(neighbor, goal)
                    heapq.heappush(
                        frontier,
                        (h_next, get_tie_key(counter), neighbor, path + [neighbor]),
                    )
                    enqueued_neighbors.append(neighbor)

        if record_steps:
            step_num += 1
            steps.append(
                GridStepRecord(
                    step_number=step_num,
                    current_coord=current,
                    heuristic_val=float(h_val),
                    action="EXPAND",
                    frontier_size=len(frontier),
                    neighbors_evaluated=evaluated_neighbors,
                    neighbors_enqueued=enqueued_neighbors,
                    path_so_far=path,
                )
            )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1

    return GridSearchResult(
        algorithm="Greedy Best-First Search",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        steps=steps,
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name=heuristic_name,
    )


def breadth_first_search_grid(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_START_COORD,
    goal: GridCoordinate = DEFAULT_GOAL_COORD,
    directions: Sequence[Tuple[str, int, int, str]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
    record_steps: bool = False,
) -> GridSearchResult:
    """Navigate a 2D maze using Breadth-First Search (BFS) to find the optimal shortest path.

    BFS explores concentric rings outwards from the start using a FIFO queue (`collections.deque`),
    guaranteeing the optimal (minimum step) path on unweighted grids.

    Args:
        maze: 2D grid matrix (0 = open, 1 = obstacle).
        start: Starting coordinate tuple (row, col).
        goal: Target goal coordinate tuple (row, col).
        directions: Sequence of directional moves (name, dr, dc, symbol).
        record_steps: Whether to record step-by-step history.

    Returns:
        GridSearchResult containing optimal path, length, and nodes expanded.
    """
    start_time = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0])

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GridSearchResult(
            algorithm="Breadth-First Search",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            nodes_expanded=1,
            expansion_order=[start],
            peak_frontier=1,
            execution_time_ms=elapsed,
            heuristic_name="none",
            is_suboptimal=False,
        )

    queue: deque[Tuple[GridCoordinate, List[GridCoordinate]]] = deque([(start, [start])])
    visited: Set[GridCoordinate] = {start}
    expansion_order: List[GridCoordinate] = []
    found = False
    solution_path: List[GridCoordinate] = []
    peak_frontier = 1

    while queue:
        peak_frontier = max(peak_frontier, len(queue))
        curr, path = queue.popleft()
        expansion_order.append(curr)

        if curr == goal:
            found = True
            solution_path = path
            break

        cr, cc = curr
        for _, dr, dc, _ in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1

    return GridSearchResult(
        algorithm="Breadth-First Search",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name="none",
        is_suboptimal=False,
    )


def a_star_search_grid(
    maze: Sequence[Sequence[int]],
    start: GridCoordinate = DEFAULT_START_COORD,
    goal: GridCoordinate = DEFAULT_GOAL_COORD,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], Union[int, float]]] = None,
    directions: Sequence[Tuple[str, int, int, str]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
) -> GridSearchResult:
    """Navigate a 2D maze using A* Search (f(n) = g(n) + h(n)) as an optimal informed baseline.

    Args:
        maze: 2D grid matrix (0 = open, 1 = obstacle).
        start: Starting coordinate tuple (row, col).
        goal: Target goal coordinate tuple (row, col).
        heuristic_fn: Dynamic heuristic calculator. Defaults to Manhattan distance.
        directions: Sequence of directional moves.

    Returns:
        GridSearchResult containing optimal path and expansion count.
    """
    start_time = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0])

    if heuristic_fn is None:
        heuristic_fn = manhattan_distance

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GridSearchResult(
            algorithm="A* Search",
            maze=[list(row) for row in maze],
            start=start,
            goal=goal,
            found=True,
            path=[start],
            path_length=0,
            nodes_expanded=1,
            expansion_order=[start],
            peak_frontier=1,
            execution_time_ms=elapsed,
            heuristic_name="manhattan",
            is_suboptimal=False,
        )

    # (f_cost, counter, g_cost, coord, path)
    frontier: List[Tuple[Union[int, float], int, int, GridCoordinate, List[GridCoordinate]]] = []
    counter = 0
    h_start = heuristic_fn(start, goal)
    heapq.heappush(frontier, (h_start, counter, 0, start, [start]))

    explored: Dict[GridCoordinate, int] = {}
    expansion_order: List[GridCoordinate] = []
    peak_frontier = 1
    found = False
    solution_path: List[GridCoordinate] = []

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))
        f_cost, _, g_cost, curr, path = heapq.heappop(frontier)

        if curr in explored and explored[curr] <= g_cost:
            continue

        explored[curr] = g_cost
        expansion_order.append(curr)

        if curr == goal:
            found = True
            solution_path = path
            break

        cr, cc = curr
        for _, dr, dc, _ in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                ng = g_cost + 1
                if neighbor not in explored or ng < explored[neighbor]:
                    counter += 1
                    nf = ng + heuristic_fn(neighbor, goal)
                    heapq.heappush(frontier, (nf, counter, ng, neighbor, path + [neighbor]))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    path_len = len(solution_path) - 1 if found else -1

    return GridSearchResult(
        algorithm="A* Search",
        maze=[list(row) for row in maze],
        start=start,
        goal=goal,
        found=found,
        path=solution_path,
        path_length=path_len,
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        peak_frontier=peak_frontier,
        execution_time_ms=elapsed,
        heuristic_name="manhattan",
        is_suboptimal=False,
    )


# ============================================================================
# Comparison and Analysis Utilities
# ============================================================================

def compare_gbfs_vs_bfs(
    maze: Sequence[Sequence[int]] = DEFAULT_10X10_MAZE,
    start: GridCoordinate = DEFAULT_START_COORD,
    goal: GridCoordinate = DEFAULT_GOAL_COORD,
    directions: Sequence[Tuple[str, int, int, str]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
    include_astar: bool = True,
) -> GBFSGridComparisonResult:
    """Compare Greedy Best-First Search against BFS (and optionally A*) on the same maze.

    Demonstrates the fundamental theoretical trade-off:
    - Greedy Best-First Search explores fewer nodes due to heuristic guidance.
    - Greedy Best-First Search might return a suboptimal (longer) path.
    - BFS explores more nodes exhaustively but guarantees the optimal shortest path.

    Args:
        maze: 2D maze matrix.
        start: Start coordinate.
        goal: Goal coordinate.
        directions: Direction tuple.
        include_astar: Whether to also run A* search for comparison.

    Returns:
        GBFSGridComparisonResult containing both results and comparative metrics.
    """
    gbfs_res = greedy_best_first_search_grid(
        maze=maze,
        start=start,
        goal=goal,
        directions=directions,
        record_steps=True,
    )
    bfs_res = breadth_first_search_grid(
        maze=maze,
        start=start,
        goal=goal,
        directions=directions,
        record_steps=True,
    )
    astar_res = a_star_search_grid(
        maze=maze,
        start=start,
        goal=goal,
        directions=directions,
    ) if include_astar else None

    # Determine suboptimality
    is_subopt = gbfs_res.path_length > bfs_res.path_length if (gbfs_res.found and bfs_res.found) else False
    gbfs_res.is_suboptimal = is_subopt

    exp_diff = bfs_res.nodes_expanded - gbfs_res.nodes_expanded
    exp_red_pct = (exp_diff / bfs_res.nodes_expanded * 100.0) if bfs_res.nodes_expanded > 0 else 0.0
    len_diff = gbfs_res.path_length - bfs_res.path_length if (gbfs_res.found and bfs_res.found) else 0
    subopt_ratio = (gbfs_res.path_length / bfs_res.path_length) if (bfs_res.path_length > 0 and gbfs_res.found) else 1.0

    return GBFSGridComparisonResult(
        gbfs_result=gbfs_res,
        bfs_result=bfs_res,
        a_star_result=astar_res,
        expansion_difference=exp_diff,
        expansion_reduction_pct=exp_red_pct,
        path_length_difference=len_diff,
        suboptimality_ratio=subopt_ratio,
        is_gbfs_strictly_fewer_nodes=(gbfs_res.nodes_expanded < bfs_res.nodes_expanded),
        is_gbfs_suboptimal=is_subopt,
    )


# ============================================================================
# Optional Challenge: Grid Path Rendering
# ============================================================================

def render_grid_with_path(
    maze: Sequence[Sequence[int]],
    path: Sequence[GridCoordinate],
    start: Optional[GridCoordinate] = None,
    goal: Optional[GridCoordinate] = None,
    path_char: str = "*",
    wall_char: str = "#",
    open_char: str = ".",
    preserve_endpoints: bool = True,
    show_coordinates: bool = True,
) -> str:
    """Render the 2D grid maze to a string with the agent's path marked by '*'.

    This fulfills the Optional Challenge: 'Output the grid with the path marked by '*''.

    Legend:
    - 'S': Start coordinate
    - 'G': Goal coordinate
    - '*': Path steps traversed by the agent
    - '#': Obstacle / impassable wall
    - '.': Open walkable cell

    Args:
        maze: 2D matrix representing the maze.
        path: Ordered sequence of coordinates along the solution path.
        start: Start coordinate (defaults to path[0] if path is non-empty).
        goal: Goal coordinate (defaults to path[-1] if path is non-empty).
        path_char: Character symbol for path cells. Default is '*'.
        wall_char: Character symbol for obstacles. Default is '#'.
        open_char: Character symbol for open cells. Default is '.'.
        preserve_endpoints: If True, renders 'S' and 'G' at endpoints; if False, uses path_char.
        show_coordinates: If True, prints row and column indices as a header/axis.

    Returns:
        Formatted multi-line ASCII string representing the solved grid maze.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    path_set = set(path)

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
                row_symbols.append("S")
            elif preserve_endpoints and goal is not None and coord == goal:
                row_symbols.append("G")
            elif coord in path_set:
                row_symbols.append(path_char)
            elif maze[r][c] == 1:
                row_symbols.append(wall_char)
            else:
                row_symbols.append(open_char)

        row_content = " ".join(row_symbols)
        if show_coordinates:
            lines.append(f"{r % 10} | {row_content}")
        else:
            lines.append(row_content)

    return "\n".join(lines)


def render_side_by_side_grids(
    maze: Sequence[Sequence[int]],
    gbfs_result: GridSearchResult,
    bfs_result: GridSearchResult,
) -> str:
    """Format a side-by-side ASCII comparison showing GBFS vs BFS navigation paths.

    Args:
        maze: The 2D grid maze.
        gbfs_result: GridSearchResult for GBFS.
        bfs_result: GridSearchResult for BFS.

    Returns:
        Multi-line side-by-side string representation.
    """
    gbfs_grid_str = render_grid_with_path(maze, gbfs_result.path, path_char="*", show_coordinates=False)
    bfs_grid_str = render_grid_with_path(maze, bfs_result.path, path_char="o", show_coordinates=False)

    gbfs_lines = gbfs_grid_str.split("\n")
    bfs_lines = bfs_grid_str.split("\n")

    out_lines: List[str] = [
        f"{'Greedy Best-First Search (Path: *)':<32} | {'Breadth-First Search (Path: o)':<32}",
        f"{f'Expanded: {gbfs_result.nodes_expanded}, Len: {gbfs_result.path_length}':<32} | "
        f"{f'Expanded: {bfs_result.nodes_expanded}, Len: {bfs_result.path_length}':<32}",
        "-" * 68,
    ]

    for g_line, b_line in zip(gbfs_lines, bfs_lines):
        out_lines.append(f"{g_line:<32} | {b_line:<32}")

    return "\n".join(out_lines)


# ============================================================================
# Object-Oriented Interface: GBFSGridNavigator
# ============================================================================

class GBFSGridNavigator:
    """Object-Oriented Agent for 2D Grid Maze Navigation using Informed Search.

    Encapsulates maze geometry, dynamic heuristic calculation, search execution,
    comparative benchmarks, and ASCII/graphical visualization.
    """

    def __init__(
        self,
        maze: Optional[Sequence[Sequence[int]]] = None,
        start: GridCoordinate = DEFAULT_START_COORD,
        goal: GridCoordinate = DEFAULT_GOAL_COORD,
        heuristic: str = "manhattan",
        directions: Sequence[Tuple[str, int, int, str]] = DEFAULT_ORTHOGONAL_DIRECTIONS,
        tie_breaker: str = "fifo",
    ) -> None:
        """Initialize the grid navigation agent.

        Args:
            maze: 2D binary grid (0 = open, 1 = obstacle). Defaults to DEFAULT_10X10_MAZE.
            start: Start coordinate tuple. Defaults to (0, 0).
            goal: Goal coordinate tuple. Defaults to (9, 9).
            heuristic: Distance heuristic ('manhattan', 'euclidean', 'chebyshev').
            directions: Movement directions. Defaults to orthogonal (Down, Right, Up, Left).
            tie_breaker: Tie-breaking policy ('fifo', 'lifo').
        """
        raw_maze = maze if maze is not None else DEFAULT_10X10_MAZE
        self.maze: MazeGrid = [list(row) for row in raw_maze]
        self.start = start
        self.goal = goal
        self.heuristic_name = heuristic.lower()
        if self.heuristic_name not in HEURISTIC_REGISTRY:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. Available: {list(HEURISTIC_REGISTRY.keys())}"
            )
        self.heuristic_fn = HEURISTIC_REGISTRY[self.heuristic_name]
        self.directions = directions
        self.tie_breaker = tie_breaker
        self._last_gbfs_result: Optional[GridSearchResult] = None
        self._last_comparison: Optional[GBFSGridComparisonResult] = None

    @property
    def rows(self) -> int:
        """Return the number of rows in the maze."""
        return len(self.maze)

    @property
    def cols(self) -> int:
        """Return the number of columns in the maze."""
        return len(self.maze[0]) if self.maze else 0

    def compute_heuristic(self, coord: GridCoordinate) -> Union[int, float]:
        """Compute the dynamic heuristic value from coordinate to the current goal."""
        return self.heuristic_fn(coord, self.goal)

    def solve_greedy(self, record_steps: bool = False) -> GridSearchResult:
        """Execute Greedy Best-First Search on the configured maze."""
        result = greedy_best_first_search_grid(
            maze=self.maze,
            start=self.start,
            goal=self.goal,
            heuristic_fn=self.heuristic_fn,
            directions=self.directions,
            tie_breaker=self.tie_breaker,
            record_steps=record_steps,
            heuristic_name=self.heuristic_name,
        )
        self._last_gbfs_result = result
        return result

    def solve_bfs(self, record_steps: bool = False) -> GridSearchResult:
        """Execute Breadth-First Search on the configured maze."""
        return breadth_first_search_grid(
            maze=self.maze,
            start=self.start,
            goal=self.goal,
            directions=self.directions,
            record_steps=record_steps,
        )

    def solve_a_star(self) -> GridSearchResult:
        """Execute A* Search on the configured maze."""
        return a_star_search_grid(
            maze=self.maze,
            start=self.start,
            goal=self.goal,
            heuristic_fn=self.heuristic_fn,
            directions=self.directions,
        )

    def compare_with_bfs(self) -> GBFSGridComparisonResult:
        """Run both Greedy Best-First Search and BFS, analyzing suboptimality and efficiency."""
        comp = compare_gbfs_vs_bfs(
            maze=self.maze,
            start=self.start,
            goal=self.goal,
            directions=self.directions,
            include_astar=True,
        )
        self._last_comparison = comp
        self._last_gbfs_result = comp.gbfs_result
        return comp

    def render_path_grid(self, path: Optional[Sequence[GridCoordinate]] = None) -> str:
        """Render the maze grid with the agent's path marked by '*' (Optional Challenge)."""
        if path is None:
            if self._last_gbfs_result is None:
                self.solve_greedy()
            path = self._last_gbfs_result.path if self._last_gbfs_result else []
        return render_grid_with_path(self.maze, path, start=self.start, goal=self.goal)


# ============================================================================
# Viva Voce Knowledge Guide
# ============================================================================

def get_gbfs_grid_viva_reference() -> Dict[str, Any]:
    """Retrieve university viva voce theory, questions, and model answers for Question 27."""
    return {
        "title": "Question 27 Viva Voce Guide: Greedy Best-First Search Grid Navigator",
        "core_topics": [
            "Greedy Best-First Search vs A*",
            "Influence of Obstacles on Greedy Search",
            "Heuristic Evaluation Function: f(n) = h(n)",
            "Dynamic Manhattan Distance Calculation",
            "Why Greedy Explores Fewer Nodes than BFS",
            "Why Greedy Can Return Suboptimal Paths",
            "Local Minima & Concave Dead-End Traps",
        ],
        "qa_pairs": [
            {
                "question": "What is the primary difference between Greedy Best-First Search and A* Search?",
                "answer": (
                    "Greedy Best-First Search evaluates nodes using purely the heuristic function f(n) = h(n), "
                    "completely ignoring the accumulated path cost g(n) incurred from the start node. "
                    "In contrast, A* Search balances past cost and future estimate: f(n) = g(n) + h(n). "
                    "Because Greedy ignores g(n), it is not optimal and can take arbitrarily long or winding paths. "
                    "A* is mathematically guaranteed to find the optimal (minimal cost) path when h(n) is admissible."
                ),
            },
            {
                "question": "How do obstacles influence the behavior and performance of Greedy Best-First Search?",
                "answer": (
                    "Obstacles create barriers between the agent and the goal. When facing convex obstacles, Greedy "
                    "can often slide around edges. However, when encountering concave (U-shaped) obstacles or cul-de-sacs, "
                    "Greedy search aggressively moves deeper into the trap because cells near the back wall have lower h(n) "
                    "values (physically closer to the goal). Once trapped at a local minimum where all successors have "
                    "higher h(n), Greedy is forced to backtrack, expanding many unnecessary nodes before escaping."
                ),
            },
            {
                "question": "Why does Greedy Search often explore fewer nodes than Breadth-First Search (BFS)?",
                "answer": (
                    "BFS is an uninformed (blind) search that expands nodes exhaustively in concentric circular "
                    "ripples of increasing depth in every direction, exploring nodes that lead away from the goal. "
                    "Greedy Search is informed: its priority queue steers exploration like a beam directly towards the "
                    "goal, expanding only nodes that appear closest to the destination, often finding a path with "
                    "drastically fewer node expansions (e.g. 25 expansions vs 87 in our 10x10 benchmark)."
                ),
            },
            {
                "question": "Why is Manhattan distance the natural heuristic for 4-directional grid navigation?",
                "answer": (
                    "On a grid where movement is restricted to 4 orthogonal cardinal directions (Up, Down, Left, Right), "
                    "the L1 Manhattan distance |r1 - r2| + |c1 - c2| represents the exact number of steps required to "
                    "reach the goal in an obstacle-free grid. When obstacles are present, it is admissible (never overestimates) "
                    "and consistent (monotonic), strictly dominating Euclidean distance on orthogonal grids."
                ),
            },
            {
                "question": "What is a local minimum in informed search, and how does Greedy handle it?",
                "answer": (
                    "A local minimum is a search state that has a lower heuristic value than all its immediate neighbors, "
                    "but is not the goal state. When Greedy search reaches a local minimum (e.g. a dead end facing the goal), "
                    "it has exhausted the lowest-h branch and must pop the next best alternative from its priority queue frontier, "
                    "effectively backtracking out of the dead end."
                ),
            },
        ],
    }
