"""A* Pathfinding with Octile Distance Heuristic (8-Directional Diagonal Movement).

This module implements Question 35: A* Search on a 2D grid supporting 8-directional
movement (orthogonal moves cost 1.0, diagonal moves cost 1.414). To remain admissible
and consistent under 8-way movement, the canonical Octile distance heuristic is utilized
in place of Manhattan distance.

Problem Statement:
    Modify the standard grid-based A* algorithm to allow 8-directional movement
    (diagonal moves cost 1.414, straight moves cost 1). The program must use the
    Octile distance heuristic instead of Manhattan to remain admissible.

Task Requirements:
    1. Add diagonal successors (8-way movement: Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right).
    2. Implement the Octile distance heuristic.
    3. Run A* search and output the path cost (Sample Output: Cost: 2.828 for Start (0,0) to Goal (2,2)).
    4. Enforce corner constraint: Cannot move diagonally through two corner walls.
    5. Provide terminal visualization of explored nodes and final path with directional glyphs.

Theoretical Foundations:
    1. 8-Directional Grid Movement & Step Costs:
       - Orthogonal (Straight) Transitions: (dr, dc) in {(-1, 0), (1, 0), (0, -1), (0, 1)} with step cost D1 = 1.0.
       - Diagonal Transitions: (dr, dc) in {(-1, -1), (-1, 1), (1, -1), (1, 1)} with step cost D2 = 1.414.

    2. Admissibility & The Octile Distance Formulation:
       - On an obstacle-free 8-connected grid, between points (r1, c1) and (r2, c2) with dx = |r1 - r2|
         and dy = |c1 - c2|, the optimal path performs min(dx, dy) diagonal moves and |dx - dy| straight moves.
       - The exact true remaining cost h*(n) in free space is:
             h_Octile(n) = D1 * |dx - dy| + D2 * min(dx, dy)
                         = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)
       - With D1 = 1.0 and D2 = 1.414:
             h_Octile(n) = max(dx, dy) + 0.414 * min(dx, dy)
       - Admissibility: Since obstacles can only increase the true shortest path length:
             0 <= h_Octile(n) <= h*(n)
         Octile distance is strictly admissible and never overestimates the true remaining cost.
       - Consistency: Satisfies the triangle inequality h(u) <= c(u, v) + h(v) across all 8 transitions.

    3. Why Manhattan is Inadmissible for 8-Directional Search:
       - Manhattan distance: h_Manhattan(n) = D1 * (dx + dy).
       - For (0, 0) to (2, 2), dx = 2, dy = 2.
         h_Manhattan = 2 + 2 = 4.0.
         True 8-way cost h* = 2 * 1.414 = 2.828.
       - Here, h_Manhattan( (0, 0) ) = 4.0 > h*( (0, 0) ) = 2.828 (Overestimates by 41.4%!).
       - When an overestimating heuristic is used, A* loses its mathematical guarantee of optimality.

    4. Euclidean vs. Octile Heuristic on Discrete Grids:
       - Euclidean distance: h_Euclidean(n) = sqrt(dx^2 + dy^2).
       - While Euclidean is admissible on an 8-way grid, it strictly underestimates the cost of grid paths:
         For (0, 0) to (1, 2): dx = 1, dy = 2.
         Euclidean = sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236.
         Octile = 1.0 * (2 - 1) + 1.414 * 1 = 2.414 = h*.
       - Octile dominates Euclidean (h_Octile(n) >= h_Euclidean(n) everywhere), causing A* with Octile
         to expand fewer or equal nodes than A* with Euclidean distance.

    5. Corner Wall Squeeze Constraint:
       - When attempting a diagonal transition from (r, c) to (r + dr, c + dc), the agent crosses the corner
         junction between orthogonal neighbors (r + dr, c) and (r, c + dc).
       - If BOTH adjacent cells are obstacles/walls, the diagonal transition is strictly prohibited
         ("Cannot move diagonally through two corner walls").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import heapq
import math
import sys
import time
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Type Aliases
GridCoordinate = Tuple[int, int]
MazeGrid = List[List[int]]

# ============================================================================
# Canonical Costs & Directions
# ============================================================================

DEFAULT_STRAIGHT_COST: float = 1.0
DEFAULT_DIAGONAL_COST: float = 1.414
EXACT_SQRT2_DIAGONAL_COST: float = math.sqrt(2.0)

# 8-Directional Movements: (Action Name, dr, dc, is_diagonal)
DIRECTIONS_8: Tuple[Tuple[str, int, int, bool], ...] = (
    ("Up", -1, 0, False),
    ("Down", 1, 0, False),
    ("Left", 0, -1, False),
    ("Right", 0, 1, False),
    ("Up-Left", -1, -1, True),
    ("Up-Right", -1, 1, True),
    ("Down-Left", 1, -1, True),
    ("Down-Right", 1, 1, True),
)

# Directional Arrow Glyphs for Enhanced Visualizer
DIRECTION_GLYPHS: Dict[str, str] = {
    "Up": "↑",
    "Down": "↓",
    "Left": "←",
    "Right": "→",
    "Up-Left": "↖",
    "Up-Right": "↗",
    "Down-Left": "↙",
    "Down-Right": "↘",
    "Start": "S",
    "Goal": "G",
}

# Standard Maze Visualizer Glyphs
GLYPH_OBSTACLE: str = "X"
GLYPH_OPEN: str = "."
GLYPH_PATH: str = "*"
GLYPH_EXPLORED: str = "#"
GLYPH_START: str = "S"
GLYPH_GOAL: str = "G"

# Numeric Cell Constants
CELL_OPEN: int = 0
CELL_OBSTACLE: int = 1


# ============================================================================
# Corner Wall Constraint Modes
# ============================================================================

class CornerConstraintMode(str, Enum):
    """Modes governing diagonal movement past obstacle corners."""

    NO_DOUBLE_CORNER_WALLS = "no_double_corner_walls"
    """Syllabus constraint: Cannot move diagonally through two corner walls.
    Disallows diagonal transition only if BOTH adjacent orthogonal cells are obstacles."""

    STRICT_NO_CORNER_CUTTING = "strict_no_corner_cutting"
    """Strict robotics constraint: Cannot clip any corner.
    Disallows diagonal transition if EITHER adjacent orthogonal cell is an obstacle."""

    ALLOW_ALL = "allow_all"
    """Free diagonal transitions: Only requires destination cell to be open."""


# ============================================================================
# Canonical Default Benchmarks
# ============================================================================

# 1. Syllabus Sample Input: 3x3 Open Grid (Start: (0, 0), Goal: (2, 2))
OPEN_3X3_MAZE: MazeGrid = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
OPEN_3X3_START: GridCoordinate = (0, 0)
OPEN_3X3_GOAL: GridCoordinate = (2, 2)

# 2. Corner Wall Pinch Maze (Demonstrates "Cannot move diagonally through two corner walls"):
# (0, 1) and (1, 0) are obstacles, so diagonal move (0, 0) -> (1, 1) is strictly blocked.
CORNER_PINCH_MAZE: MazeGrid = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]
CORNER_PINCH_START: GridCoordinate = (0, 0)
CORNER_PINCH_GOAL: GridCoordinate = (2, 2)

# 3. Single Corner Maze (Demonstrates valid glide past a single corner obstacle):
# Only (0, 1) is an obstacle; (1, 0) is open, so diagonal move (0, 0) -> (1, 1) is allowed.
SINGLE_CORNER_MAZE: MazeGrid = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
]
SINGLE_CORNER_START: GridCoordinate = (0, 0)
SINGLE_CORNER_GOAL: GridCoordinate = (2, 2)

# 4. Canonical 5x5 Maze (from Question 29):
# 5x5 grid with vertical obstacle wall. With 8-way movement, diagonal turns allow faster traversal.
DEFAULT_5X5_MAZE: MazeGrid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
DEFAULT_5X5_START: GridCoordinate = (0, 0)
DEFAULT_5X5_GOAL: GridCoordinate = (4, 4)

# 5. U-Shaped Concave Trap Maze:
CONCAVE_TRAP_MAZE: MazeGrid = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
CONCAVE_TRAP_START: GridCoordinate = (0, 0)
CONCAVE_TRAP_GOAL: GridCoordinate = (2, 3)

# 6. Single-cell maze (Start is Goal)
SINGLE_CELL_MAZE: MazeGrid = [[0]]


# ============================================================================
# Distance Heuristic Functions
# ============================================================================

def octile_distance(
    coord: GridCoordinate,
    goal: GridCoordinate,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
    diagonal_cost: float = DEFAULT_DIAGONAL_COST,
) -> float:
    """Calculate the Octile distance heuristic for 8-connected grid navigation.

    Formula:
        dx = |r - r_goal|, dy = |c - c_goal|
        h(n) = D1 * |dx - dy| + D2 * min(dx, dy)
             = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)

    where:
        D1 = straight_cost (step cost for orthogonal moves, default 1.0)
        D2 = diagonal_cost (step cost for diagonal moves, default 1.414)

    In an obstacle-free 8-connected grid, the optimal path makes min(dx, dy)
    diagonal transitions and |dx - dy| orthogonal transitions. Thus, Octile distance
    equals the exact minimum remaining cost h*(n), guaranteeing admissibility:
        h_Octile(n) <= h*(n)
    and consistency (satisfying the triangle inequality):
        h_Octile(u) <= c(u, v) + h_Octile(v)

    Args:
        coord: Current grid coordinate (row, col).
        goal: Target destination coordinate (row_goal, col_goal).
        straight_cost: Orthogonal step cost (default 1.0).
        diagonal_cost: Diagonal step cost (default 1.414).

    Returns:
        Computed Octile distance as a float.
    """
    dx = abs(coord[0] - goal[0])
    dy = abs(coord[1] - goal[1])
    min_d = min(dx, dy)
    max_d = max(dx, dy)
    return straight_cost * (max_d - min_d) + diagonal_cost * min_d


def manhattan_distance(
    coord: GridCoordinate,
    goal: GridCoordinate,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
) -> float:
    """Calculate the L1 Manhattan distance between two grid coordinates.

    Formula:
        h(n) = D1 * (|r - r_goal| + |c - c_goal|)

    NOTE: For 8-directional movement, Manhattan distance is INADMISSIBLE because
    it assumes 4-directional cardinal movement. For diagonal moves, it overestimates
    by up to (2 - 1.414) / 1.414 ~ 41.4%.

    Args:
        coord: Current grid coordinate (row, col).
        goal: Target destination coordinate (row_goal, col_goal).
        straight_cost: Orthogonal step cost (default 1.0).

    Returns:
        Float Manhattan distance.
    """
    return straight_cost * (abs(coord[0] - goal[0]) + abs(coord[1] - goal[1]))


def euclidean_distance(
    coord: GridCoordinate,
    goal: GridCoordinate,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
) -> float:
    """Calculate the straight-line Euclidean distance (L2 norm) between two coordinates.

    Formula:
        h(n) = D1 * sqrt((r - r_goal)^2 + (c - c_goal)^2)

    NOTE: Euclidean distance is admissible for 8-way movement, but it strictly
    underestimates the discrete grid path cost compared to Octile distance.
    Octile dominates Euclidean: h_Euclidean(n) <= h_Octile(n) <= h*(n).

    Args:
        coord: Current coordinate (row, col).
        goal: Target goal coordinate (row_goal, col_goal).
        straight_cost: Orthogonal step cost (default 1.0).

    Returns:
        Float Euclidean distance.
    """
    return straight_cost * math.hypot(coord[0] - goal[0], coord[1] - goal[1])


def chebyshev_distance(
    coord: GridCoordinate,
    goal: GridCoordinate,
    step_cost: float = DEFAULT_STRAIGHT_COST,
) -> float:
    """Calculate the Chebyshev distance (L_infinity norm) between two coordinates.

    Formula:
        h(n) = step_cost * max(|r - r_goal|, |c - c_goal|)

    NOTE: Chebyshev distance assumes diagonal moves have the same cost as straight moves (1.0).
    When diagonal moves cost 1.414, Chebyshev significantly underestimates the true cost.

    Args:
        coord: Current coordinate (row, col).
        goal: Target goal coordinate (row_goal, col_goal).
        step_cost: Uniform step cost (default 1.0).

    Returns:
        Float Chebyshev distance.
    """
    return step_cost * max(abs(coord[0] - goal[0]), abs(coord[1] - goal[1]))


def zero_heuristic(
    coord: GridCoordinate,
    goal: GridCoordinate,
) -> float:
    """Zero heuristic function h(n) = 0 (Uniform Cost Search / Dijkstra).

    Args:
        coord: Current coordinate (unused).
        goal: Goal coordinate (unused).

    Returns:
        0.0 for all nodes.
    """
    return 0.0


# ============================================================================
# Successor Generation with Corner Wall Constraint
# ============================================================================

def is_valid_coordinate(coord: GridCoordinate, rows: int, cols: int) -> bool:
    """Check if coordinate is within grid bounds."""
    return 0 <= coord[0] < rows and 0 <= coord[1] < cols


def get_8way_successors(
    maze: MazeGrid,
    current: GridCoordinate,
    corner_mode: CornerConstraintMode = CornerConstraintMode.NO_DOUBLE_CORNER_WALLS,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
    diagonal_cost: float = DEFAULT_DIAGONAL_COST,
) -> List[Tuple[GridCoordinate, str, float]]:
    """Generate valid 8-directional successors respecting obstacles and corner wall constraints.

    Corner Wall Squeeze Constraint:
        When moving diagonally from (r, c) to (r + dr, c + dc), the adjacent orthogonal
        cells are (r + dr, c) and (r, c + dc).
        - In NO_DOUBLE_CORNER_WALLS (default):
          Blocked ONLY if BOTH (r + dr, c) and (r, c + dc) are obstacles.
          Matches syllabus condition: "Cannot move diagonally through two corner walls."
        - In STRICT_NO_CORNER_CUTTING:
          Blocked if EITHER (r + dr, c) or (r, c + dc) is an obstacle.
        - In ALLOW_ALL:
          Permitted as long as (r + dr, c + dc) is open.

    Args:
        maze: 2D grid matrix where 0 = open, 1 = obstacle.
        current: Current coordinate (row, col).
        corner_mode: CornerConstraintMode enum controlling corner-cutting policy.
        straight_cost: Step cost for orthogonal transitions (default 1.0).
        diagonal_cost: Step cost for diagonal transitions (default 1.414).

    Returns:
        List of tuples: (neighbor_coord, action_name, transition_cost).
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    r, c = current
    successors: List[Tuple[GridCoordinate, str, float]] = []

    for action_name, dr, dc, is_diagonal in DIRECTIONS_8:
        nr, nc = r + dr, c + dc
        neighbor = (nr, nc)

        # 1. Boundary check
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue

        # 2. Destination obstacle check
        if maze[nr][nc] == CELL_OBSTACLE:
            continue

        # 3. Diagonal corner wall constraints
        cost = diagonal_cost if is_diagonal else straight_cost
        if is_diagonal:
            # The two adjacent orthogonal cells sharing the corner
            wall1 = maze[nr][c] == CELL_OBSTACLE
            wall2 = maze[r][nc] == CELL_OBSTACLE

            if corner_mode == CornerConstraintMode.NO_DOUBLE_CORNER_WALLS:
                if wall1 and wall2:
                    # Cannot squeeze diagonally through two touching corner walls
                    continue
            elif corner_mode == CornerConstraintMode.STRICT_NO_CORNER_CUTTING:
                if wall1 or wall2:
                    # Strict avoidance: cannot cut past any obstacle corner
                    continue
            # If ALLOW_ALL, no corner check needed

        successors.append((neighbor, action_name, cost))

    return successors


# ============================================================================
# Step Record and Result Dataclasses
# ============================================================================

@dataclass
class AStarOctileStepRecord:
    """Record of an individual node expansion during A* search.

    Attributes:
        step_number: 1-indexed sequential expansion counter.
        current_coord: Coordinate of the expanded cell (r, c).
        action: Transition action that reached this node ("Start" for initial).
        g_cost: Accumulated path cost from start to current node.
        h_cost: Heuristic cost estimate from current node to goal.
        f_cost: Total evaluation score f(n) = g(n) + h(n).
        frontier_size: Number of candidate nodes remaining in priority queue.
        closed_size: Cumulative count of unique nodes expanded so far.
        notes: Contextual observation or milestone description.
    """

    step_number: int
    current_coord: GridCoordinate
    action: str
    g_cost: float
    h_cost: float
    f_cost: float
    frontier_size: int
    closed_size: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert step record to JSON-serializable dictionary."""
        return {
            "step_number": self.step_number,
            "current_coord": list(self.current_coord),
            "action": self.action,
            "g_cost": round(self.g_cost, 4),
            "h_cost": round(self.h_cost, 4),
            "f_cost": round(self.f_cost, 4),
            "frontier_size": self.frontier_size,
            "closed_size": self.closed_size,
            "notes": self.notes,
        }


@dataclass
class AStarOctileResult:
    """Encapsulates the complete result of an A* Octile pathfinding search.

    Attributes:
        algorithm: Algorithm name descriptor.
        maze: 2D grid matrix navigated.
        start: Initial origin coordinate (r_start, c_start).
        goal: Destination coordinate (r_goal, c_goal).
        found: True if a feasible obstacle-free path was found, False otherwise.
        path: List of coordinates from start to goal inclusive.
        actions: List of tuples (action_name, step_cost) for each transition.
        path_length: Number of transitions in solution path (len(path) - 1).
        total_cost: Exact accumulated path cost (sum of step costs).
        straight_moves_count: Number of orthogonal transitions along path.
        diagonal_moves_count: Number of diagonal transitions along path.
        nodes_expanded: Total number of nodes popped and expanded from priority queue.
        expansion_order: Ordered list of coordinates as they were expanded.
        explored_set: Set of unique coordinates expanded during search.
        best_g_costs: Dictionary mapping reached coordinates to their minimum discovered g(n).
        steps: Detailed step records (if recording enabled).
        peak_frontier: Maximum size reached by priority queue during search.
        execution_time_ms: Search execution time in milliseconds.
        heuristic_name: Name of heuristic function utilized.
        corner_mode: CornerConstraintMode applied during search.
        is_optimal: Theoretical optimality flag.
    """

    algorithm: str
    maze: MazeGrid
    start: GridCoordinate
    goal: GridCoordinate
    found: bool
    path: List[GridCoordinate]
    actions: List[Tuple[str, float]]
    path_length: int
    total_cost: float
    straight_moves_count: int
    diagonal_moves_count: int
    nodes_expanded: int
    expansion_order: List[GridCoordinate]
    explored_set: Set[GridCoordinate]
    best_g_costs: Dict[GridCoordinate, float]
    steps: List[AStarOctileStepRecord]
    peak_frontier: int
    execution_time_ms: float
    heuristic_name: str
    corner_mode: str
    is_optimal: bool = True

    def format_cost(self, decimals: int = 3) -> str:
        """Format total cost with fixed decimal places."""
        return f"{self.total_cost:.{decimals}f}"

    def move_breakdown_summary(self) -> str:
        """Return a human-readable summary of straight vs. diagonal moves."""
        parts = []
        if self.diagonal_moves_count > 0:
            word = "diagonal move" if self.diagonal_moves_count == 1 else "diagonal moves"
            # Number to word for small counts (e.g. Two diagonal moves)
            num_words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
            count_str = num_words.get(self.diagonal_moves_count, str(self.diagonal_moves_count))
            parts.append(f"{count_str} {word}")
        if self.straight_moves_count > 0:
            word = "straight move" if self.straight_moves_count == 1 else "straight moves"
            num_words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
            count_str = num_words.get(self.straight_moves_count, str(self.straight_moves_count))
            parts.append(f"{count_str} {word}")
        return " and ".join(parts) if parts else "Zero moves (Start is Goal)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "maze_dimensions": (len(self.maze), len(self.maze[0]) if self.maze else 0),
            "start": list(self.start),
            "goal": list(self.goal),
            "found": self.found,
            "path": [list(pt) for pt in self.path],
            "actions": [{"action": act, "cost": round(c, 4)} for act, c in self.actions],
            "path_length": self.path_length,
            "total_cost": round(self.total_cost, 4),
            "straight_moves_count": self.straight_moves_count,
            "diagonal_moves_count": self.diagonal_moves_count,
            "nodes_expanded": self.nodes_expanded,
            "expansion_order": [list(pt) for pt in self.expansion_order],
            "explored_count": len(self.explored_set),
            "peak_frontier": self.peak_frontier,
            "execution_time_ms": round(self.execution_time_ms, 4),
            "heuristic_name": self.heuristic_name,
            "corner_mode": self.corner_mode,
            "is_optimal": self.is_optimal,
        }

    def summary(self) -> str:
        """Produce a formatted textual summary of the search outcome."""
        status = "PATH FOUND" if self.found else "NO PATH FOUND"
        lines = [
            f"=== A* Octile Pathfinding Result: {status} ===",
            f"Start Coordinate      : {self.start}",
            f"Goal Coordinate       : {self.goal}",
            f"Total Cost            : {self.format_cost(3)} ({self.move_breakdown_summary()})",
            f"Path Length (Moves)   : {self.path_length}",
            f"Straight Moves        : {self.straight_moves_count} (Cost: {self.straight_moves_count * DEFAULT_STRAIGHT_COST:.3f})",
            f"Diagonal Moves        : {self.diagonal_moves_count} (Cost: {self.diagonal_moves_count * DEFAULT_DIAGONAL_COST:.3f})",
            f"Nodes Expanded        : {self.nodes_expanded}",
            f"Explored Set Size     : {len(self.explored_set)}",
            f"Peak Frontier Size    : {self.peak_frontier}",
            f"Heuristic             : {self.heuristic_name}",
            f"Corner Wall Mode      : {self.corner_mode}",
            f"Execution Time        : {self.execution_time_ms:.3f} ms",
        ]
        if self.found:
            path_str = " -> ".join(f"({r},{c})" for r, c in self.path)
            lines.append(f"Optimal Path Trajectory: {path_str}")
        return "\n".join(lines)


# ============================================================================
# Core A* Search Algorithm
# ============================================================================

def a_star_octile_search(
    maze: MazeGrid,
    start: GridCoordinate,
    goal: GridCoordinate,
    heuristic_fn: Optional[Callable[[GridCoordinate, GridCoordinate], float]] = None,
    heuristic_name: str = "Octile",
    corner_mode: CornerConstraintMode = CornerConstraintMode.NO_DOUBLE_CORNER_WALLS,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
    diagonal_cost: float = DEFAULT_DIAGONAL_COST,
    tie_breaker: str = "min_h",
    record_steps: bool = False,
) -> AStarOctileResult:
    """Execute A* Search with 8-directional movement and Octile heuristic on a 2D maze.

    Algorithm Mechanics:
        1. Initialize open priority queue with start node:
           f(start) = g(start) + h(start) = 0 + h(start).
        2. Pop node n with minimum f(n). On ties, resolve using `tie_breaker`:
           - 'min_h': pick candidate closest to goal (highest g).
           - 'fifo': pick candidate inserted earlier.
           - 'lifo': pick candidate inserted later.
        3. If n == goal, terminate and reconstruct shortest path.
        4. For each 8-directional neighbor n' generated by get_8way_successors():
           - Evaluate tentative g(n') = g(n) + step_cost.
           - If tentative g(n') < best_g(n'), record parent transition, update best_g,
             and push to open priority queue with f(n') = tentative g(n') + h(n').
        5. Return structured AStarOctileResult containing solution path, step records,
           total cost, and search metrics.

    Args:
        maze: 2D grid matrix where 0 = open, 1 = obstacle.
        start: Starting grid coordinate (row, col).
        goal: Target destination coordinate (row, col).
        heuristic_fn: Optional custom heuristic function h(coord, goal). Defaults to octile_distance.
        heuristic_name: Descriptive name for heuristic (default 'Octile').
        corner_mode: CornerConstraintMode for diagonal corner wall behavior.
        straight_cost: Step cost for orthogonal transitions (default 1.0).
        diagonal_cost: Step cost for diagonal transitions (default 1.414).
        tie_breaker: Strategy for breaking f-cost ties ('min_h', 'fifo', 'lifo').
        record_steps: If True, records step-by-step expansion records for visualization.

    Returns:
        AStarOctileResult instance detailing path, cost, and expansion telemetry.

    Raises:
        ValueError: If start or goal coordinates are out of bounds or on obstacles.
    """
    start_time = time.perf_counter()

    if not maze or not maze[0]:
        raise ValueError("Maze grid cannot be empty.")

    rows = len(maze)
    cols = len(maze[0])

    if not is_valid_coordinate(start, rows, cols):
        raise ValueError(f"Start coordinate {start} is out of bounds for {rows}x{cols} grid.")
    if not is_valid_coordinate(goal, rows, cols):
        raise ValueError(f"Goal coordinate {goal} is out of bounds for {rows}x{cols} grid.")
    if maze[start[0]][start[1]] == CELL_OBSTACLE:
        raise ValueError(f"Start coordinate {start} is located on an impassable obstacle wall.")
    if maze[goal[0]][goal[1]] == CELL_OBSTACLE:
        raise ValueError(f"Goal coordinate {goal} is located on an impassable obstacle wall.")

    # Select Heuristic
    if heuristic_fn is None:
        h_func: Callable[[GridCoordinate, GridCoordinate], float] = lambda c, g: octile_distance(
            c, g, straight_cost=straight_cost, diagonal_cost=diagonal_cost
        )
    else:
        h_func = heuristic_fn

    # Priority Queue Elements:
    # Priority Tuple: (f_cost, secondary_key, counter, current_coord)
    # Floating point comparison precision tolerance:
    EPSILON = 1e-9

    counter = 0
    initial_h = h_func(start, goal)
    initial_f = initial_h

    # secondary_key depends on tie_breaker strategy:
    # min_h: lower h is preferred (closer to goal)
    # fifo: counter (lower counter popped first)
    # lifo: -counter (higher counter popped first)
    def make_entry(f_val: float, h_val: float, count_val: int, coord: GridCoordinate) -> Tuple[float, float, int, GridCoordinate]:
        if tie_breaker == "min_h":
            return (round(f_val, 7), round(h_val, 7), count_val, coord)
        elif tie_breaker == "lifo":
            return (round(f_val, 7), -count_val, count_val, coord)
        else:  # fifo
            return (round(f_val, 7), count_val, count_val, coord)

    open_heap: List[Tuple[float, float, int, GridCoordinate]] = []
    heapq.heappush(open_heap, make_entry(initial_f, initial_h, counter, start))

    # Search State Tracking
    best_g_costs: Dict[GridCoordinate, float] = {start: 0.0}
    # came_from maps node -> (parent_coord, action_name, transition_cost)
    came_from: Dict[GridCoordinate, Tuple[Optional[GridCoordinate], Optional[str], float]] = {
        start: (None, "Start", 0.0)
    }
    explored_set: Set[GridCoordinate] = set()
    expansion_order: List[GridCoordinate] = []
    steps: List[AStarOctileStepRecord] = []
    peak_frontier = 1

    goal_reached = False
    step_num = 0

    while open_heap:
        peak_frontier = max(peak_frontier, len(open_heap))
        f_curr, sec_key, _, current = heapq.heappop(open_heap)
        current_g = best_g_costs[current]

        # Skip if already closed (since nodes with worse g might remain in heap)
        if current in explored_set:
            continue

        explored_set.add(current)
        expansion_order.append(current)
        step_num += 1

        curr_action = came_from[current][1] or "Start"
        curr_h = h_func(current, goal)

        if record_steps:
            note = "GOAL REACHED!" if current == goal else f"Expanded ({current[0]}, {current[1]}) via {curr_action}"
            steps.append(
                AStarOctileStepRecord(
                    step_number=step_num,
                    current_coord=current,
                    action=curr_action,
                    g_cost=current_g,
                    h_cost=curr_h,
                    f_cost=current_g + curr_h,
                    frontier_size=len(open_heap),
                    closed_size=len(explored_set),
                    notes=note,
                )
            )

        # Check Goal Condition
        if current == goal:
            goal_reached = True
            break

        # Generate 8-Way Successors
        successors = get_8way_successors(
            maze=maze,
            current=current,
            corner_mode=corner_mode,
            straight_cost=straight_cost,
            diagonal_cost=diagonal_cost,
        )

        for neighbor, action_name, step_cost in successors:
            tentative_g = current_g + step_cost

            # Check if this neighbor is unvisited or offers a strictly cheaper path
            if neighbor not in best_g_costs or tentative_g < (best_g_costs[neighbor] - EPSILON):
                best_g_costs[neighbor] = tentative_g
                came_from[neighbor] = (current, action_name, step_cost)
                neighbor_h = h_func(neighbor, goal)
                neighbor_f = tentative_g + neighbor_h
                counter += 1
                heapq.heappush(open_heap, make_entry(neighbor_f, neighbor_h, counter, neighbor))

    # Reconstruct Shortest Path
    path: List[GridCoordinate] = []
    actions: List[Tuple[str, float]] = []
    straight_moves = 0
    diagonal_moves = 0
    total_cost = 0.0

    if goal_reached:
        curr: Optional[GridCoordinate] = goal
        while curr is not None:
            path.append(curr)
            parent, act, move_cost = came_from[curr]
            if act != "Start" and act is not None:
                actions.append((act, move_cost))
                total_cost += move_cost
                if move_cost >= (diagonal_cost - EPSILON):
                    diagonal_moves += 1
                else:
                    straight_moves += 1
            curr = parent
        path.reverse()
        actions.reverse()
    elif start == goal:
        # Edge case: start is goal
        goal_reached = True
        path = [start]
        total_cost = 0.0

    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    return AStarOctileResult(
        algorithm="A* with Octile Distance (8-Directional)",
        maze=maze,
        start=start,
        goal=goal,
        found=goal_reached,
        path=path,
        actions=actions,
        path_length=len(actions),
        total_cost=total_cost,
        straight_moves_count=straight_moves,
        diagonal_moves_count=diagonal_moves,
        nodes_expanded=len(explored_set),
        expansion_order=expansion_order,
        explored_set=explored_set,
        best_g_costs=best_g_costs,
        steps=steps,
        peak_frontier=peak_frontier,
        execution_time_ms=execution_time_ms,
        heuristic_name=heuristic_name,
        corner_mode=corner_mode.value,
        is_optimal=True,
    )


# ============================================================================
# 2D Grid Parser
# ============================================================================

def parse_grid(
    grid_input: Union[str, Sequence[Sequence[Union[int, str]]], Sequence[str]],
    default_start: Optional[GridCoordinate] = None,
    default_goal: Optional[GridCoordinate] = None,
) -> Tuple[MazeGrid, GridCoordinate, GridCoordinate]:
    """Parse a 2D maze grid from multiline strings, matrices, or character arrays.

    Supported inputs:
    - Multiline string:
          . . .
          . X .
          . . .
      with 'S' for start, 'G' for goal, 'X' or '1' for obstacle, '.' or '0' for open.
    - 2D int matrix: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    - Sequence of strings: [". . .", ". X .", ". . ."]

    Args:
        grid_input: String, list of strings, or 2D matrix.
        default_start: Fallback start coordinate if not embedded in grid (defaults to (0, 0)).
        default_goal: Fallback goal coordinate if not embedded in grid (defaults to bottom-right).

    Returns:
        Tuple: (maze_matrix, start_coord, goal_coord).

    Raises:
        ValueError: If grid is empty or non-rectangular.
    """
    start_found: Optional[GridCoordinate] = None
    goal_found: Optional[GridCoordinate] = None
    matrix: MazeGrid = []

    if isinstance(grid_input, str):
        lines = [line.strip() for line in grid_input.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Input grid string cannot be empty.")
        for r, line in enumerate(lines):
            # Tokens separated by whitespace or individual characters
            tokens = line.split() if " " in line else list(line)
            row_vals: List[int] = []
            for c, token in enumerate(tokens):
                t_upper = token.upper()
                if t_upper == "S":
                    start_found = (r, c)
                    row_vals.append(CELL_OPEN)
                elif t_upper == "G":
                    goal_found = (r, c)
                    row_vals.append(CELL_OPEN)
                elif t_upper in ("X", "1", "#", "W"):
                    row_vals.append(CELL_OBSTACLE)
                elif t_upper in (".", "0", " ", "_"):
                    row_vals.append(CELL_OPEN)
                else:
                    raise ValueError(f"Unrecognized cell token '{token}' at position ({r}, {c}).")
            matrix.append(row_vals)
    elif isinstance(grid_input, (list, tuple)):
        if not grid_input:
            raise ValueError("Grid sequence cannot be empty.")
        for r, row in enumerate(grid_input):
            if isinstance(row, str):
                tokens = row.split() if " " in row else list(row)
                row_vals = []
                for c, token in enumerate(tokens):
                    t_upper = token.upper()
                    if t_upper == "S":
                        start_found = (r, c)
                        row_vals.append(CELL_OPEN)
                    elif t_upper == "G":
                        goal_found = (r, c)
                        row_vals.append(CELL_OPEN)
                    elif t_upper in ("X", "1", "#", "W"):
                        row_vals.append(CELL_OBSTACLE)
                    else:
                        row_vals.append(CELL_OPEN)
                matrix.append(row_vals)
            elif isinstance(row, (list, tuple)):
                row_vals = []
                for c, val in enumerate(row):
                    if isinstance(val, int):
                        row_vals.append(CELL_OBSTACLE if val == 1 else CELL_OPEN)
                    elif isinstance(val, str):
                        v_upper = val.upper()
                        if v_upper == "S":
                            start_found = (r, c)
                            row_vals.append(CELL_OPEN)
                        elif v_upper == "G":
                            goal_found = (r, c)
                            row_vals.append(CELL_OPEN)
                        elif v_upper in ("X", "1", "#", "W"):
                            row_vals.append(CELL_OBSTACLE)
                        else:
                            row_vals.append(CELL_OPEN)
                    else:
                        row_vals.append(CELL_OPEN)
                matrix.append(row_vals)
    else:
        raise TypeError(f"Unsupported grid input type: {type(grid_input)}")

    if not matrix:
        raise ValueError("Parsed grid matrix has zero rows.")
    cols = len(matrix[0])
    for r, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError(f"Non-rectangular grid: row {r} has length {len(row)}, expected {cols}.")

    # Resolve Start Coordinate
    final_start = start_found if start_found is not None else (default_start if default_start is not None else (0, 0))
    # Resolve Goal Coordinate
    final_goal = (
        goal_found
        if goal_found is not None
        else (default_goal if default_goal is not None else (len(matrix) - 1, cols - 1))
    )

    return matrix, final_start, final_goal


# ============================================================================
# Terminal Maze Visualizers
# ============================================================================

def render_octile_maze(
    maze: MazeGrid,
    path: Optional[Sequence[GridCoordinate]] = None,
    explored_set: Optional[Set[GridCoordinate]] = None,
    start: Optional[GridCoordinate] = None,
    goal: Optional[GridCoordinate] = None,
    use_directional_glyphs: bool = True,
    actions: Optional[Sequence[Tuple[str, float]]] = None,
) -> str:
    """Render the 2D maze in ASCII format showing obstacles, explored nodes, and path.

    Render Specifications:
        - 'S': Start coordinate
        - 'G': Goal coordinate
        - 'X': Obstacle walls
        - '.': Open walkable cells (unvisited)
        - '#': Explored nodes expanded during search
        - '*' or Directional Arrows (↑, ↓, ←, →, ↖, ↗, ↘, ↙): Final optimal path cells

    Args:
        maze: 2D grid matrix.
        path: Ordered list of path coordinates.
        explored_set: Set of explored coordinates.
        start: Start coordinate (defaults to path[0]).
        goal: Goal coordinate (defaults to path[-1]).
        use_directional_glyphs: If True, uses directional arrows for path steps.
        actions: Optional sequence of actions corresponding to path steps.

    Returns:
        Formatted ASCII grid string with coordinate indices.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    path_set = set(path) if path else set()
    path_list = list(path) if path else []
    explored = explored_set if explored_set is not None else set()

    s_coord = start if start is not None else (path_list[0] if path_list else (0, 0))
    g_coord = goal if goal is not None else (path_list[-1] if path_list else (rows - 1, cols - 1))

    # Map path coordinates to directional glyphs
    glyph_map: Dict[GridCoordinate, str] = {}
    if use_directional_glyphs and len(path_list) >= 2:
        for idx in range(len(path_list) - 1):
            curr_pt = path_list[idx]
            next_pt = path_list[idx + 1]
            dr = next_pt[0] - curr_pt[0]
            dc = next_pt[1] - curr_pt[1]
            # Match direction
            matched = "*"
            for act_name, adr, adc, _ in DIRECTIONS_8:
                if adr == dr and adc == dc:
                    matched = DIRECTION_GLYPHS.get(act_name, "*")
                    break
            glyph_map[curr_pt] = matched

    # Header with column indices
    col_header = "   " + " ".join(f"{c % 10}" for c in range(cols))
    divider = "   " + "-" * (cols * 2 - 1)
    lines = [col_header, divider]

    for r in range(rows):
        row_chars = []
        for c in range(cols):
            pt = (r, c)
            if pt == s_coord:
                row_chars.append("S")
            elif pt == g_coord:
                row_chars.append("G")
            elif pt in path_set:
                if use_directional_glyphs and pt in glyph_map:
                    row_chars.append(glyph_map[pt])
                else:
                    row_chars.append(GLYPH_PATH)
            elif pt in explored:
                row_chars.append(GLYPH_EXPLORED)
            elif maze[r][c] == CELL_OBSTACLE:
                row_chars.append(GLYPH_OBSTACLE)
            else:
                row_chars.append(GLYPH_OPEN)
        lines.append(f"{r:2d}| " + " ".join(row_chars))

    return "\n".join(lines)


def generate_octile_expansion_frames(
    maze: MazeGrid,
    result: AStarOctileResult,
) -> Generator[str, None, None]:
    """Generate progressive step-by-step ASCII frames showing search expansion and path.

    Yields:
        ASCII string for each expansion step.
    """
    explored_so_far: Set[GridCoordinate] = set()
    for step in result.steps:
        explored_so_far.add(step.current_coord)
        frame = [
            f"Step {step.step_number}/{len(result.steps)}: Expanding {step.current_coord} via {step.action}",
            f"Cost: g={step.g_cost:.3f}, h={step.h_cost:.3f}, f={step.f_cost:.3f} | Frontier: {step.frontier_size} | Closed: {step.closed_size}",
            render_octile_maze(
                maze=maze,
                path=None,
                explored_set=explored_so_far,
                start=result.start,
                goal=result.goal,
                use_directional_glyphs=False,
            ),
        ]
        yield "\n".join(frame)

    # Final frame showing solution path
    final_frame = [
        f"=== Search Completed: Goal Reached! ===",
        f"Total Cost: {result.format_cost(3)} ({result.move_breakdown_summary()}) | Steps: {result.path_length}",
        render_octile_maze(
            maze=maze,
            path=result.path,
            explored_set=result.explored_set,
            start=result.start,
            goal=result.goal,
            use_directional_glyphs=True,
            actions=result.actions,
        ),
    ]
    yield "\n".join(final_frame)


def animate_octile_search(
    maze: MazeGrid,
    result: AStarOctileResult,
    delay: float = 0.05,
    clear_screen: bool = True,
) -> None:
    """Animate the A* Octile search expansion and path reconstruction in the terminal."""
    for frame in generate_octile_expansion_frames(maze, result):
        if clear_screen:
            sys.stdout.write("\033[H\033[J")
        sys.stdout.write(frame + "\n\n")
        sys.stdout.flush()
        time.sleep(delay)


# ============================================================================
# Multi-Heuristic Comparative Analysis
# ============================================================================

@dataclass
class HeuristicComparisonEntry:
    """Record of search performance under a specific heuristic."""

    heuristic_name: str
    is_admissible: bool
    is_consistent: bool
    found: bool
    path_length: int
    total_cost: float
    nodes_expanded: int
    peak_frontier: int
    execution_time_ms: float
    optimality_status: str


@dataclass
class AStarOctileComparisonResult:
    """Encapsulates side-by-side comparison across heuristics for 8-way movement."""

    maze: MazeGrid
    start: GridCoordinate
    goal: GridCoordinate
    entries: List[HeuristicComparisonEntry]
    octile_result: AStarOctileResult
    manhattan_result: AStarOctileResult
    euclidean_result: AStarOctileResult
    zero_result: AStarOctileResult

    def to_table(self) -> str:
        """Format the comparison as an ASCII markdown table."""
        headers = [
            "Heuristic",
            "Admissible?",
            "Consistent?",
            "Path Moves",
            "Total Cost",
            "Expansions",
            "Peak Frontier",
            "Optimality",
        ]
        rows = []
        for e in self.entries:
            adm_str = "Yes" if e.is_admissible else "NO (Overestimates!)"
            con_str = "Yes" if e.is_consistent else "No"
            rows.append(
                [
                    e.heuristic_name,
                    adm_str,
                    con_str,
                    str(e.path_length),
                    f"{e.total_cost:.3f}",
                    str(e.nodes_expanded),
                    str(e.peak_frontier),
                    e.optimality_status,
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        hdr_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

        out_lines = [sep_line, hdr_line, sep_line]
        for row in rows:
            line = "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
            out_lines.append(line)
        out_lines.append(sep_line)
        return "\n".join(out_lines)


def compare_octile_vs_heuristics(
    maze: MazeGrid,
    start: GridCoordinate,
    goal: GridCoordinate,
    corner_mode: CornerConstraintMode = CornerConstraintMode.NO_DOUBLE_CORNER_WALLS,
    straight_cost: float = DEFAULT_STRAIGHT_COST,
    diagonal_cost: float = DEFAULT_DIAGONAL_COST,
) -> AStarOctileComparisonResult:
    """Compare A* with Octile distance against Manhattan, Euclidean, and Zero (UCS) heuristics.

    Demonstrates:
    - Octile Heuristic: Admissible, consistent, strictly optimal, minimal expansions.
    - Manhattan Heuristic: Inadmissible for 8-way movement (overestimates diagonal steps).
    - Euclidean Heuristic: Admissible, but dominated by Octile (expands >= nodes).
    - Zero Heuristic (UCS): Blind baseline expanding broadest frontier.

    Args:
        maze: 2D grid matrix.
        start: Start coordinate.
        goal: Goal coordinate.
        corner_mode: CornerConstraintMode.
        straight_cost: Orthogonal cost.
        diagonal_cost: Diagonal cost.

    Returns:
        AStarOctileComparisonResult containing detailed comparison metrics.
    """
    # 1. Octile (Canonical)
    res_octile = a_star_octile_search(
        maze=maze,
        start=start,
        goal=goal,
        heuristic_fn=lambda c, g: octile_distance(c, g, straight_cost, diagonal_cost),
        heuristic_name="Octile Distance",
        corner_mode=corner_mode,
        straight_cost=straight_cost,
        diagonal_cost=diagonal_cost,
    )

    # 2. Manhattan (Inadmissible for 8-way movement)
    res_manhattan = a_star_octile_search(
        maze=maze,
        start=start,
        goal=goal,
        heuristic_fn=lambda c, g: manhattan_distance(c, g, straight_cost),
        heuristic_name="Manhattan Distance",
        corner_mode=corner_mode,
        straight_cost=straight_cost,
        diagonal_cost=diagonal_cost,
    )

    # 3. Euclidean (Admissible, but weaker than Octile)
    res_euclidean = a_star_octile_search(
        maze=maze,
        start=start,
        goal=goal,
        heuristic_fn=lambda c, g: euclidean_distance(c, g, straight_cost),
        heuristic_name="Euclidean Distance",
        corner_mode=corner_mode,
        straight_cost=straight_cost,
        diagonal_cost=diagonal_cost,
    )

    # 4. Zero Heuristic (Uniform Cost Search)
    res_zero = a_star_octile_search(
        maze=maze,
        start=start,
        goal=goal,
        heuristic_fn=zero_heuristic,
        heuristic_name="Zero (UCS / Dijkstra)",
        corner_mode=corner_mode,
        straight_cost=straight_cost,
        diagonal_cost=diagonal_cost,
    )

    optimal_cost = res_octile.total_cost
    EPS = 1e-6

    def check_opt(cost: float) -> str:
        if abs(cost - optimal_cost) < EPS:
            return "Optimal"
        elif cost > optimal_cost:
            return f"Suboptimal (+{cost - optimal_cost:.3f})"
        else:
            return "Cheaper"

    entries = [
        HeuristicComparisonEntry(
            heuristic_name="Octile (Canonical)",
            is_admissible=True,
            is_consistent=True,
            found=res_octile.found,
            path_length=res_octile.path_length,
            total_cost=res_octile.total_cost,
            nodes_expanded=res_octile.nodes_expanded,
            peak_frontier=res_octile.peak_frontier,
            execution_time_ms=res_octile.execution_time_ms,
            optimality_status=check_opt(res_octile.total_cost),
        ),
        HeuristicComparisonEntry(
            heuristic_name="Euclidean (Weaker)",
            is_admissible=True,
            is_consistent=True,
            found=res_euclidean.found,
            path_length=res_euclidean.path_length,
            total_cost=res_euclidean.total_cost,
            nodes_expanded=res_euclidean.nodes_expanded,
            peak_frontier=res_euclidean.peak_frontier,
            execution_time_ms=res_euclidean.execution_time_ms,
            optimality_status=check_opt(res_euclidean.total_cost),
        ),
        HeuristicComparisonEntry(
            heuristic_name="Manhattan (INADMISSIBLE)",
            is_admissible=False,
            is_consistent=False,
            found=res_manhattan.found,
            path_length=res_manhattan.path_length,
            total_cost=res_manhattan.total_cost,
            nodes_expanded=res_manhattan.nodes_expanded,
            peak_frontier=res_manhattan.peak_frontier,
            execution_time_ms=res_manhattan.execution_time_ms,
            optimality_status=check_opt(res_manhattan.total_cost),
        ),
        HeuristicComparisonEntry(
            heuristic_name="Zero / UCS (Uninformed)",
            is_admissible=True,
            is_consistent=True,
            found=res_zero.found,
            path_length=res_zero.path_length,
            total_cost=res_zero.total_cost,
            nodes_expanded=res_zero.nodes_expanded,
            peak_frontier=res_zero.peak_frontier,
            execution_time_ms=res_zero.execution_time_ms,
            optimality_status=check_opt(res_zero.total_cost),
        ),
    ]

    return AStarOctileComparisonResult(
        maze=maze,
        start=start,
        goal=goal,
        entries=entries,
        octile_result=res_octile,
        manhattan_result=res_manhattan,
        euclidean_result=res_euclidean,
        zero_result=res_zero,
    )


# ============================================================================
# University Viva Voce Examination Guide
# ============================================================================

VIVA_VOCE_GUIDE: Dict[str, Dict[str, str]] = {
    "Q1_Manhattan_Inadmissible": {
        "question": "Why is Manhattan distance inadmissible for 8-way movement?",
        "answer": (
            "An admissible heuristic must never overestimate the true remaining cost to the goal (h(n) <= h*(n)). "
            "Manhattan distance is formulated for 4-connected grids where only orthogonal moves (Up, Down, Left, Right) "
            "are allowed: h_Manhattan = D1 * (dx + dy). "
            "However, with 8-directional movement, diagonal transitions allow the agent to traverse both a horizontal "
            "and vertical unit simultaneously at cost D2 = 1.414. "
            "For example, moving from (0, 0) to (2, 2) has dx = 2, dy = 2. "
            "The true 8-way cost h* is 2 * 1.414 = 2.828 (two diagonal moves). "
            "Manhattan estimates h = 2 + 2 = 4.0. "
            "Since h_Manhattan( (0, 0) ) = 4.0 > h*( (0, 0) ) = 2.828, Manhattan overestimates the true cost by ~41.4%. "
            "Using an inadmissible heuristic breaks A*'s mathematical proof of optimality, potentially returning suboptimal paths."
        ),
    },
    "Q2_Euclidean_vs_Octile": {
        "question": "What is the difference between Euclidean and Octile distance in discrete grid pathfinding?",
        "answer": (
            "Both Euclidean and Octile distances are admissible for 8-connected grid navigation, but they differ in heuristic dominance: "
            "1. Euclidean distance measures continuous straight-line distance: h_Euclidean = sqrt(dx^2 + dy^2). "
            "   Because grid navigation is constrained to discrete steps (multiples of 1.0 and 1.414), Euclidean strictly underestimates "
            "   the achievable path cost. For dx = 1, dy = 2: Euclidean = sqrt(5) ~ 2.236, while the true discrete cost is 1.414 + 1.0 = 2.414. "
            "2. Octile distance accounts directly for the 8-directional geometry: h_Octile = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy). "
            "   In obstacle-free space, Octile distance equals the exact true shortest path cost (h_Octile = h*). "
            "   Therefore, Octile strictly dominates Euclidean (h_Octile(n) >= h_Euclidean(n) everywhere). "
            "   In A* search, a more informed, dominant heuristic prunes more nodes, causing A* with Octile to expand fewer or equal nodes "
            "   than A* with Euclidean distance without sacrificing optimality."
        ),
    },
    "Q3_Corner_Wall_Constraint": {
        "question": "What is the corner wall cutting constraint and why is it enforced?",
        "answer": (
            "The corner wall constraint ('Cannot move diagonally through two corner walls') prevents an agent from squeezing "
            "diagonally through the microscopic junction formed by two diagonally adjacent obstacles. "
            "If an agent is at (r, c) and attempts to move to (r + 1, c + 1), the two orthogonal intermediate cells are (r + 1, c) "
            "and (r, c + 1). If both of these cells are solid walls, the diagonal line of travel cuts directly through the touching "
            "corners of the two obstacle blocks. "
            "In physical robotics and real-world navigation, an agent has non-zero physical clearance and would collide with the corner corners. "
            "Hence, diagonal movement between two corner walls is strictly prohibited."
        ),
    },
    "Q4_Time_and_Space_Complexity": {
        "question": "What are the time and space complexities of A* with Octile distance?",
        "answer": (
            "Let N = V = R * C be the total number of cells in the 2D grid: "
            "- Time Complexity: O(N log N). In the worst case, each cell is inserted into and extracted from the priority queue. "
            "  Priority queue operations (push/pop) take O(log N) time. Each cell has at most 8 neighbors (O(1) transition evaluations). "
            "  Thus, total time is bounded by O(N log N). "
            "- Space Complexity: O(N). The open priority queue and closed explored set store at most N grid coordinates, along with "
            "  the came_from parent pointers and g-cost lookup table."
        ),
    },
    "Q5_Octile_Formula_Derivation": {
        "question": "How is the Octile distance formula derived?",
        "answer": (
            "Given two coordinates with coordinate deltas dx = |r1 - r2| and dy = |c1 - c2|: "
            "Without loss of generality, assume dx >= dy. "
            "To minimize path cost, the agent should take as many diagonal steps as possible because one diagonal step covers 1 unit in row "
            "and 1 unit in column for cost D2 = 1.414, which is cheaper than two orthogonal steps (cost 2.0). "
            "The maximum number of diagonal steps that reduce both coordinates simultaneously is min(dx, dy) = dy. "
            "After dy diagonal steps, the remaining distance along the longer dimension is (dx - dy). "
            "These remaining steps must be covered by pure orthogonal moves at cost D1 = 1.0 each. "
            "Summing these two components gives: "
            "h_Octile = D1 * (dx - dy) + D2 * dy = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)."
        ),
    },
}
