"""Uniform Cost Search (UCS) on 2D Topological Grid with Cost Modifiers.

This module implements Uniform Cost Search (UCS) on a 2D topological map grid where
different terrain characters represent distinct movement costs (e.g. Clear '.' = 1,
Mountain 'M' = 5, Water 'W' = 10, Obstacle '#' = impassable).

Theoretical Background & AI Concepts:
- Problem Formulation:
    A mobile agent navigates an H x W 2D grid from a start coordinate (r_start, c_start)
    to a goal coordinate (r_goal, c_goal). Unlike uniform grid navigation (where all
    steps have cost 1), topological maps assign variable traversal costs to different
    terrain types. The cost to enter an adjacent cell is determined by the destination
    cell's terrain cost modifier.
- Uniform Cost Search (UCS) Foundation:
    UCS is an uninformed (blind) search strategy that explores the state space in order
    of cumulative path cost g(n) from the initial state, equivalent to Dijkstra's
    algorithm on an implicitly generated state graph. UCS does not require a heuristic
    function h(n) (effectively A* with h(n) = 0).
- Priority Queue Frontier using heapq:
    The frontier (open list) is maintained as a binary min-heap using Python's `heapq`
    module. Each element in the heap is a tuple `(cumulative_cost, insertion_counter, coordinate)`:
    - `cumulative_cost`: The total path cost g(n) from start to node n.
    - `insertion_counter`: A monotonic integer counter providing deterministic FIFO
      tie-breaking for nodes with identical cumulative path costs, preventing comparison
      errors when comparing coordinate tuples.
    - `coordinate`: The (row, col) grid location.
- Goal Test Placement (Expansion vs. Generation):
    In UCS with non-uniform edge weights, the goal test MUST be performed upon *expansion*
    (when the node is popped from the priority queue), NOT upon *generation* (when the node
    is first discovered). The first discovered path to the goal may not be the cheapest;
    a cheaper detour might still reside in the frontier awaiting expansion.
- Explored Set and Lazy Deletion:
    Standard Python `heapq` does not implement an O(log N) `decrease_key` operation.
    When a cheaper path to a previously discovered cell is found, the cheaper entry is pushed
    into the heap. When an entry is popped, if its recorded cost exceeds the known optimal
    cost (`cost > best_cost[node]`), it represents a superseded entry and is discarded in O(1) time.
- Completeness and Optimality:
    - Completeness: UCS is complete if every step cost is bounded away from zero by a positive
      constant epsilon (c >= eps > 0) and the branching factor b is finite. This prevents
      infinite loops along zero-cost or infinitesimally small cost paths (Zeno's paradox).
    - Optimality: Because nodes are expanded in strictly non-decreasing order of cumulative
      cost g(n), the first goal state popped from the priority queue is guaranteed to be
      an optimal (minimum-cost) path.
- 4-Directional vs. Diagonal Movement (Optional Challenge):
    - Default: Movement is restricted to 4 orthogonal cardinal directions: Up (-1, 0),
      Down (1, 0), Left (0, -1), Right (0, 1).
    - Optional Challenge: Diagonal movements (Up-Right, Down-Right, Down-Left, Up-Left)
      are enabled with a calculated Euclidean multiplier cost (step_cost = cell_cost * sqrt(2)).
      Optional corner-cutting prevention ensures agents cannot squeeze diagonally past
      touching obstacles.

Complexity:
- Time Complexity: O(E log V) where V = H * W (total grid cells) and E <= 4 * V (or 8 * V with diagonals).
  Each cell is expanded at most once (O(V log V)) and each edge relaxation pushes to the
  min-heap at most once (O(E log V)).
- Space Complexity: O(V) auxiliary memory for the frontier priority queue, best_cost table,
  and predecessor pointers for path reconstruction.
"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

# Coordinate type alias: (row, column)
Coordinate = Tuple[int, int]

# Canonical Default Terrain Costs
DEFAULT_TERRAIN_COSTS: Dict[str, int] = {
    ".": 1,   # Clear ground / Flat land
    "M": 5,   # Mountain / Rough terrain
    "W": 10,  # Water / Swamp / River
    "S": 1,   # Start cell terrain
    "G": 1,   # Goal cell terrain
}

# Impassable obstacle characters
DEFAULT_OBSTACLES: Set[str] = {"#", "X"}

# Cardinal 4-directional movements: (dr, dc, name, arrow_symbol)
ORTHOGONAL_DIRECTIONS: List[Tuple[int, int, str, str]] = [
    (-1, 0, "Up", "^"),
    (1, 0, "Down", "v"),
    (0, -1, "Left", "<"),
    (0, 1, "Right", ">"),
]

# 8-directional movements (including diagonals): (dr, dc, name, arrow_symbol, is_diagonal)
ALL_8_DIRECTIONS: List[Tuple[int, int, str, str, bool]] = [
    (-1, 0, "Up", "^", False),
    (1, 0, "Down", "v", False),
    (0, -1, "Left", "<", False),
    (0, 1, "Right", ">", False),
    (-1, -1, "Up-Left", "\\", True),
    (-1, 1, "Up-Right", "/", True),
    (1, -1, "Down-Left", "/", True),
    (1, 1, "Down-Right", "\\", True),
]


class TerrainType(Enum):
    """Enumeration of standard topological map terrain types."""

    CLEAR = (".", 1, "Clear Terrain")
    MOUNTAIN = ("M", 5, "Mountain Terrain")
    WATER = ("W", 10, "Water / Swamp")
    OBSTACLE = ("#", float("inf"), "Impassable Obstacle")
    START = ("S", 1, "Start Location")
    GOAL = ("G", 1, "Goal Location")

    def __init__(self, char: str, default_cost: Union[int, float], description: str) -> None:
        self.char = char
        self.default_cost = default_cost
        self.description = description

    @classmethod
    def from_char(cls, char: str) -> "TerrainType":
        """Resolve a terrain enum member from a grid character."""
        for member in cls:
            if member.char == char:
                return member
        if char in DEFAULT_OBSTACLES:
            return cls.OBSTACLE
        return cls.CLEAR


@dataclass(frozen=True)
class MoveAction:
    """Represents a movement action from one cell to an adjacent cell.

    Attributes:
        delta: Coordinate displacement (dr, dc).
        name: Human-readable action name ('Up', 'Down', 'Left', 'Right', etc.).
        arrow: Symbolic representation ('^', 'v', '<', '>', etc.).
        is_diagonal: Whether the action represents a diagonal movement.
    """

    delta: Tuple[int, int]
    name: str
    arrow: str
    is_diagonal: bool = False


@dataclass
class UCSGridResult:
    """Encapsulates the complete result of a Uniform Cost Search execution on a 2D grid.

    Attributes:
        success: Whether an optimal path to the goal was found.
        start: Starting coordinate (row, col).
        goal: Target coordinate (row, col).
        path: List of coordinates from start to goal inclusive.
        cost: Minimum cumulative cost of the optimal path (integer or float).
        nodes_expanded: Number of nodes popped from the priority queue and expanded.
        nodes_generated: Total number of nodes inserted into the frontier.
        max_frontier_size: Maximum simultaneous size of the priority queue.
        explored_set: Set of all unique coordinates expanded during search.
        execution_time_seconds: Total algorithm runtime in seconds.
        allow_diagonal: Whether diagonal movements were permitted during search.
        diagonal_multiplier: The cost scaling factor applied to diagonal moves.
        grid_shape: Tuple of (height, width) of the search map.
    """

    success: bool
    start: Coordinate
    goal: Coordinate
    path: List[Coordinate] = field(default_factory=list)
    cost: Union[int, float] = float("inf")
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    explored_set: Set[Coordinate] = field(default_factory=set)
    execution_time_seconds: float = 0.0
    allow_diagonal: bool = False
    diagonal_multiplier: float = math.sqrt(2)
    grid_shape: Tuple[int, int] = (0, 0)

    @property
    def path_arrow_string(self) -> str:
        """Format the coordinate path as an arrow-delimited string: (r,c)->(r,c)->..."""
        if not self.path:
            return "No path found"
        return "->".join(f"({r},{c})" for r, c in self.path)

    @property
    def formatted_output(self) -> str:
        """Format the output matching the sample format: Cost: X, Path: ..."""
        if not self.success:
            return "Cost: inf, Path: None"
        # If the cost is a whole number (e.g. 4.0), format as clean integer
        cost_str = str(int(self.cost)) if isinstance(self.cost, (int, float)) and self.cost == int(self.cost) else f"{self.cost:.2f}"
        return f"Cost: {cost_str}, Path: {self.path_arrow_string}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result metrics and path to a dictionary for logging and JSON export."""
        return {
            "success": self.success,
            "start": list(self.start),
            "goal": list(self.goal),
            "path": [list(coord) for coord in self.path],
            "cost": int(self.cost) if self.cost == int(self.cost) else round(float(self.cost), 4),
            "path_length": len(self.path),
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "max_frontier_size": self.max_frontier_size,
            "explored_count": len(self.explored_set),
            "execution_time_seconds": round(self.execution_time_seconds, 6),
            "allow_diagonal": self.allow_diagonal,
            "diagonal_multiplier": round(self.diagonal_multiplier, 4),
            "grid_shape": list(self.grid_shape),
            "formatted_output": self.formatted_output,
        }

    def summary(self) -> str:
        """Generate a structured multi-line text summary of search performance."""
        lines = [
            "--------------------------------------------------",
            "         UNIFORM COST SEARCH GRID RESULT          ",
            "--------------------------------------------------",
            f"Status                 : {'SUCCESS (Optimal Path Found)' if self.success else 'FAILED (No Path Exists)'}",
            f"Start Coordinate       : {self.start}",
            f"Goal Coordinate        : {self.goal}",
            f"Path Total Cost        : {int(self.cost) if self.cost == int(self.cost) else f'{self.cost:.4f}'}",
            f"Path Hop Count         : {max(0, len(self.path) - 1)} steps ({len(self.path)} coordinates)",
            f"Diagonal Moves Enabled : {self.allow_diagonal} (Multiplier: {self.diagonal_multiplier:.3f})",
            f"Nodes Expanded (Popped): {self.nodes_expanded}",
            f"Nodes Generated (Push) : {self.nodes_generated}",
            f"Max Frontier Size      : {self.max_frontier_size}",
            f"Explored Set Size      : {len(self.explored_set)} coordinates",
            f"Execution Time         : {self.execution_time_seconds * 1000:.3f} ms",
            f"Path Representation    : {self.path_arrow_string}",
            "--------------------------------------------------",
        ]
        return "\n".join(lines)


class TopologicalGrid:
    """Represents a 2D topological map with variable terrain movement costs.

    Provides coordinate parsing, bounds checking, cost queries, and ASCII visualization.
    """

    def __init__(
        self,
        raw_grid: Union[Sequence[str], str],
        terrain_costs: Optional[Dict[str, int]] = None,
        obstacle_chars: Optional[Set[str]] = None,
        default_cost: int = 1,
    ) -> None:
        """Initialize the topological grid from raw text or a list of character strings.

        Args:
            raw_grid: A multi-line string or a sequence of row strings (e.g. ['...', '.M.', '..G']).
            terrain_costs: Optional custom mapping of characters to positive integer/float costs.
            obstacle_chars: Optional set of characters treated as impassable walls.
            default_cost: Fallback cost for unmapped non-obstacle characters (default 1).
        """
        # Normalize input to list of strings
        if isinstance(raw_grid, str):
            lines = [line.strip() for line in raw_grid.strip().splitlines() if line.strip()]
        else:
            lines = [str(row).strip() for row in raw_grid]

        if not lines:
            raise ValueError("TopologicalGrid cannot be empty.")

        self.height: int = len(lines)
        self.width: int = max(len(row) for row in lines)
        self.raw_grid: List[str] = [row.ljust(self.width, ".") for row in lines]

        # Merge terrain costs with defaults
        self.terrain_costs: Dict[str, Union[int, float]] = dict(DEFAULT_TERRAIN_COSTS)
        if terrain_costs:
            self.terrain_costs.update(terrain_costs)

        # Merge obstacle characters
        self.obstacle_chars: Set[str] = set(DEFAULT_OBSTACLES)
        if obstacle_chars:
            self.obstacle_chars.update(obstacle_chars)

        self.default_cost: int = default_cost

        # Parse coordinate cost dictionary
        self.cost_dict: Dict[Coordinate, Union[int, float]] = self._build_cost_dict()

        # Automatically locate 'S' and 'G' in grid if present
        self.detected_start: Optional[Coordinate] = self._find_first_char("S")
        self.detected_goal: Optional[Coordinate] = self._find_first_char("G")

    def _build_cost_dict(self) -> Dict[Coordinate, Union[int, float]]:
        """Parse the 2D grid into a coordinate cost dictionary: (r, c) -> cost.

        Fulfills Task Requirement 1: 'Parse the character grid into a coordinate cost dictionary'.
        """
        cost_map: Dict[Coordinate, Union[int, float]] = {}
        for r in range(self.height):
            for c in range(self.width):
                char = self.raw_grid[r][c]
                if char in self.obstacle_chars:
                    cost_map[(r, c)] = float("inf")
                elif char in self.terrain_costs:
                    cost_map[(r, c)] = self.terrain_costs[char]
                else:
                    cost_map[(r, c)] = self.default_cost
        return cost_map

    def _find_first_char(self, target_char: str) -> Optional[Coordinate]:
        """Locate the first occurrence of a character within the grid."""
        for r in range(self.height):
            for c in range(self.width):
                if self.raw_grid[r][c].upper() == target_char.upper():
                    return (r, c)
        return None

    def in_bounds(self, r: int, c: int) -> bool:
        """Verify whether coordinate (r, c) falls within grid boundaries."""
        return 0 <= r < self.height and 0 <= c < self.width

    def is_passable(self, r: int, c: int) -> bool:
        """Check if cell (r, c) is inside bounds and not an obstacle."""
        if not self.in_bounds(r, c):
            return False
        return not math.isinf(self.cost_dict.get((r, c), float("inf")))

    def get_cell_cost(self, r: int, c: int) -> Union[int, float]:
        """Return the movement cost required to step into coordinate (r, c)."""
        if not self.in_bounds(r, c):
            return float("inf")
        return self.cost_dict.get((r, c), self.default_cost)

    def get_cell_char(self, r: int, c: int) -> str:
        """Return the original character at coordinate (r, c)."""
        if not self.in_bounds(r, c):
            return "#"
        return self.raw_grid[r][c]

    def get_neighbors(
        self,
        r: int,
        c: int,
        allow_diagonal: bool = False,
        diagonal_multiplier: float = math.sqrt(2),
        prevent_corner_cutting: bool = True,
    ) -> List[Tuple[Coordinate, Union[int, float], MoveAction]]:
        """Generate valid adjacent neighbor coordinates, step costs, and actions.

        Args:
            r: Current row.
            c: Current column.
            allow_diagonal: Whether 8-directional movement is enabled.
            diagonal_multiplier: Multiplier cost applied to diagonal moves.
            prevent_corner_cutting: If True, prevents diagonal squeezing between two obstacles.

        Returns:
            List of tuples: (neighbor_coord, step_cost, move_action).
        """
        neighbors: List[Tuple[Coordinate, Union[int, float], MoveAction]] = []

        if not allow_diagonal:
            # 4 orthogonal directions in order: Up, Down, Left, Right
            for dr, dc, name, arrow in ORTHOGONAL_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if self.is_passable(nr, nc):
                    step_cost = self.get_cell_cost(nr, nc)
                    action = MoveAction(delta=(dr, dc), name=name, arrow=arrow, is_diagonal=False)
                    neighbors.append(((nr, nc), step_cost, action))
        else:
            # 8 directions
            for dr, dc, name, arrow, is_diag in ALL_8_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not self.is_passable(nr, nc):
                    continue

                if is_diag:
                    # Optional challenge: corner cutting prevention
                    if prevent_corner_cutting:
                        # Cannot cut corner if both adjacent orthogonal cells are impassable
                        ortho1_pass = self.is_passable(r + dr, c)
                        ortho2_pass = self.is_passable(r, c + dc)
                        if not ortho1_pass and not ortho2_pass:
                            continue

                    base_cost = self.get_cell_cost(nr, nc)
                    step_cost = base_cost * diagonal_multiplier
                else:
                    step_cost = self.get_cell_cost(nr, nc)

                action = MoveAction(delta=(dr, dc), name=name, arrow=arrow, is_diagonal=is_diag)
                neighbors.append(((nr, nc), step_cost, action))

        return neighbors

    def render_ascii(
        self,
        path: Optional[Sequence[Coordinate]] = None,
        explored: Optional[Set[Coordinate]] = None,
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
    ) -> str:
        """Render an ASCII visualization of the grid with optional path, wavefront, and markers.

        Args:
            path: Sequence of coordinates on the solution path.
            explored: Set of expanded coordinates.
            start: Start coordinate (defaults to detected_start or (0,0)).
            goal: Goal coordinate (defaults to detected_goal or (H-1, W-1)).

        Returns:
            Formatted multi-line ASCII string.
        """
        path_set = set(path) if path else set()
        explored_set = set(explored) if explored else set()

        start_pt = start if start is not None else (self.detected_start or (0, 0))
        goal_pt = goal if goal is not None else (self.detected_goal or (self.height - 1, self.width - 1))

        # Build path direction arrows if path is provided
        path_arrow_map: Dict[Coordinate, str] = {}
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                curr = path[i]
                nxt = path[i + 1]
                dr = nxt[0] - curr[0]
                dc = nxt[1] - curr[1]
                # Determine arrow
                arrow = "*"
                for ddr, ddc, _, s_arrow, *extra in (ALL_8_DIRECTIONS if len(extra := [1]) else ORTHOGONAL_DIRECTIONS):
                    if (ddr, ddc) == (dr, dc):
                        arrow = s_arrow
                        break
                path_arrow_map[curr] = arrow

        col_header = "    " + " ".join(f"{c % 10}" for c in range(self.width))
        divider = "   +" + "--" * self.width + "+"
        lines = [col_header, divider]

        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                coord = (r, c)
                orig_char = self.raw_grid[r][c]

                if coord == start_pt:
                    row_chars.append("S")
                elif coord == goal_pt:
                    row_chars.append("G")
                elif coord in path_arrow_map:
                    row_chars.append(path_arrow_map[coord])
                elif coord in path_set:
                    row_chars.append("o")
                elif coord in explored_set:
                    row_chars.append("·" if orig_char == "." else orig_char.lower())
                else:
                    row_chars.append(orig_char)

            line_str = f"{r:2d} | " + " ".join(row_chars) + " |"
            lines.append(line_str)

        lines.append(divider)
        return "\n".join(lines)


def parse_grid_to_cost_dict(
    grid: Union[Sequence[str], str],
    terrain_costs: Optional[Dict[str, int]] = None,
    obstacle_chars: Optional[Set[str]] = None,
    default_cost: int = 1,
) -> Dict[Coordinate, Union[int, float]]:
    """Parse a character grid into a coordinate cost dictionary.

    Fulfills Task Requirement 1:
        'Parse the character grid into a coordinate cost dictionary.'

    Args:
        grid: 2D representation as a list of strings or newline-separated string.
        terrain_costs: Optional dictionary mapping terrain character -> cost.
        obstacle_chars: Optional set of characters treated as impassable walls.
        default_cost: Fallback cost for unmapped passables.

    Returns:
        Dictionary mapping coordinate tuple (row, col) -> numerical cost.
    """
    topological_grid = TopologicalGrid(
        raw_grid=grid,
        terrain_costs=terrain_costs,
        obstacle_chars=obstacle_chars,
        default_cost=default_cost,
    )
    return topological_grid.cost_dict


class UCSGridSolver:
    """Solves 2D grid pathfinding using Uniform Cost Search (UCS) with heapq.

    Supports:
    - Min-heap priority queue via `heapq`
    - Non-uniform step costs (Variable Terrain Modifiers)
    - Orthogonal 4-directional movement (Up, Down, Left, Right)
    - Optional Challenge: 8-directional movement with Euclidean diagonal multiplier
    - Step-by-step state space generation for visual traces and educational analysis
    """

    def __init__(
        self,
        allow_diagonal: bool = False,
        diagonal_multiplier: float = math.sqrt(2),
        prevent_corner_cutting: bool = True,
        movement_order: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Initialize the Uniform Cost Search Grid Solver.

        Args:
            allow_diagonal: Whether diagonal moves are allowed (False = 4-directional).
            diagonal_multiplier: Multiplier factor for diagonal steps (default sqrt(2)).
            prevent_corner_cutting: Whether to forbid diagonal movement past touching walls.
            movement_order: Optional custom list of directional deltas to prioritize.
        """
        self.allow_diagonal: bool = allow_diagonal
        self.diagonal_multiplier: float = diagonal_multiplier
        self.prevent_corner_cutting: bool = prevent_corner_cutting
        self.movement_order: Optional[List[Tuple[int, int]]] = movement_order

    def search(
        self,
        grid: Union[TopologicalGrid, Sequence[str], str],
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
        terrain_costs: Optional[Dict[str, int]] = None,
        obstacle_chars: Optional[Set[str]] = None,
    ) -> UCSGridResult:
        """Execute Uniform Cost Search on the provided grid from start to goal.

        Args:
            grid: TopologicalGrid instance or raw grid (list of strings / multiline string).
            start: Start coordinate (row, col). If omitted, inferred from 'S' or defaults to (0, 0).
            goal: Goal coordinate (row, col). If omitted, inferred from 'G' or defaults to bottom-right.
            terrain_costs: Optional custom terrain cost dictionary.
            obstacle_chars: Optional set of impassable obstacle characters.

        Returns:
            UCSGridResult containing optimal path, cost, and search metrics.
        """
        start_time = time.perf_counter()

        # Wrap in TopologicalGrid if not already
        if not isinstance(grid, TopologicalGrid):
            topo_grid = TopologicalGrid(
                raw_grid=grid,
                terrain_costs=terrain_costs,
                obstacle_chars=obstacle_chars,
            )
        else:
            topo_grid = grid

        # Resolve start coordinate
        if start is not None:
            resolved_start = (int(start[0]), int(start[1]))
        elif topo_grid.detected_start is not None:
            resolved_start = topo_grid.detected_start
        else:
            resolved_start = (0, 0)

        # Resolve goal coordinate
        if goal is not None:
            resolved_goal = (int(goal[0]), int(goal[1]))
        elif topo_grid.detected_goal is not None:
            resolved_goal = topo_grid.detected_goal
        else:
            resolved_goal = (topo_grid.height - 1, topo_grid.width - 1)

        # Input boundary and passability checks
        if not topo_grid.in_bounds(*resolved_start):
            raise ValueError(f"Start coordinate {resolved_start} is outside grid boundaries ({topo_grid.height}x{topo_grid.width}).")
        if not topo_grid.in_bounds(*resolved_goal):
            raise ValueError(f"Goal coordinate {resolved_goal} is outside grid boundaries ({topo_grid.height}x{topo_grid.width}).")
        if not topo_grid.is_passable(*resolved_start):
            return UCSGridResult(
                success=False,
                start=resolved_start,
                goal=resolved_goal,
                path=[],
                cost=float("inf"),
                grid_shape=(topo_grid.height, topo_grid.width),
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
            )
        if not topo_grid.is_passable(*resolved_goal):
            return UCSGridResult(
                success=False,
                start=resolved_start,
                goal=resolved_goal,
                path=[],
                cost=float("inf"),
                grid_shape=(topo_grid.height, topo_grid.width),
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
            )

        # Trivial edge case: Start == Goal
        if resolved_start == resolved_goal:
            elapsed = time.perf_counter() - start_time
            return UCSGridResult(
                success=True,
                start=resolved_start,
                goal=resolved_goal,
                path=[resolved_start],
                cost=0,
                nodes_expanded=1,
                nodes_generated=1,
                max_frontier_size=1,
                explored_set={resolved_start},
                execution_time_seconds=elapsed,
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
                grid_shape=(topo_grid.height, topo_grid.width),
            )

        # Priority Queue Frontier: elements are (cumulative_cost, counter, coordinate)
        frontier: List[Tuple[Union[int, float], int, Coordinate]] = []
        counter = 0

        # Push initial start state: g(start) = 0
        heapq.heappush(frontier, (0, counter, resolved_start))
        counter += 1

        # Best cost dictionary: tracks minimum known cost to reach coordinate
        best_cost: Dict[Coordinate, Union[int, float]] = {resolved_start: 0}

        # Parent pointer dictionary for optimal path reconstruction
        parent: Dict[Coordinate, Coordinate] = {}

        # Explored set: tracks expanded coordinates
        explored_set: Set[Coordinate] = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        found_goal = False

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))

            # Pop node with minimum cumulative path cost g(n)
            curr_cost, _, curr_coord = heapq.heappop(frontier)

            # Lazy deletion check: if cost is strictly greater than known best, skip
            if curr_cost > best_cost.get(curr_coord, float("inf")):
                continue

            # Check if coordinate already closed
            if curr_coord in explored_set:
                continue

            # Mark coordinate as explored upon expansion
            explored_set.add(curr_coord)
            nodes_expanded += 1

            # Goal Test: MUST be performed upon expansion (popped from frontier)
            if curr_coord == resolved_goal:
                found_goal = True
                break

            # Expand neighbors
            neighbors = topo_grid.get_neighbors(
                r=curr_coord[0],
                c=curr_coord[1],
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
                prevent_corner_cutting=self.prevent_corner_cutting,
            )

            for neighbor_coord, step_cost, _ in neighbors:
                if neighbor_coord in explored_set:
                    continue

                new_cost = curr_cost + step_cost

                # Relaxation: check if this is a strictly cheaper path to neighbor
                if new_cost < best_cost.get(neighbor_coord, float("inf")):
                    best_cost[neighbor_coord] = new_cost
                    parent[neighbor_coord] = curr_coord
                    heapq.heappush(frontier, (new_cost, counter, neighbor_coord))
                    counter += 1
                    nodes_generated += 1

        elapsed = time.perf_counter() - start_time

        if not found_goal:
            return UCSGridResult(
                success=False,
                start=resolved_start,
                goal=resolved_goal,
                path=[],
                cost=float("inf"),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                explored_set=explored_set,
                execution_time_seconds=elapsed,
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
                grid_shape=(topo_grid.height, topo_grid.width),
            )

        # Reconstruct path from goal back to start using parent pointers
        path: List[Coordinate] = []
        curr: Optional[Coordinate] = resolved_goal
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        path.reverse()

        final_cost = best_cost[resolved_goal]
        # Normalize integer costs to clean int if exact whole number
        if isinstance(final_cost, (int, float)) and final_cost == int(final_cost):
            final_cost = int(final_cost)

        return UCSGridResult(
            success=True,
            start=resolved_start,
            goal=resolved_goal,
            path=path,
            cost=final_cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            explored_set=explored_set,
            execution_time_seconds=elapsed,
            allow_diagonal=self.allow_diagonal,
            diagonal_multiplier=self.diagonal_multiplier,
            grid_shape=(topo_grid.height, topo_grid.width),
        )

    def step_by_step_generator(
        self,
        grid: Union[TopologicalGrid, Sequence[str], str],
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
    ) -> Generator[Dict[str, Any], None, UCSGridResult]:
        """Yield search state at each node expansion for animations and educational inspection.

        Yields:
            Dict containing step number, current node popped, current cost, frontier contents,
            explored set, and current reconstructed path.
        """
        # Ensure TopologicalGrid
        topo_grid = grid if isinstance(grid, TopologicalGrid) else TopologicalGrid(grid)
        resolved_start = start if start is not None else (topo_grid.detected_start or (0, 0))
        resolved_goal = goal if goal is not None else (topo_grid.detected_goal or (topo_grid.height - 1, topo_grid.width - 1))

        frontier: List[Tuple[Union[int, float], int, Coordinate]] = []
        counter = 0
        heapq.heappush(frontier, (0, counter, resolved_start))
        counter += 1

        best_cost: Dict[Coordinate, Union[int, float]] = {resolved_start: 0}
        parent: Dict[Coordinate, Coordinate] = {}
        explored_set: Set[Coordinate] = set()

        step = 0
        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            curr_cost, _, curr_coord = heapq.heappop(frontier)

            if curr_cost > best_cost.get(curr_coord, float("inf")):
                continue
            if curr_coord in explored_set:
                continue

            explored_set.add(curr_coord)
            nodes_expanded += 1
            step += 1

            # Partial path to current node
            partial_path: List[Coordinate] = []
            c_node: Optional[Coordinate] = curr_coord
            while c_node is not None:
                partial_path.append(c_node)
                c_node = parent.get(c_node)
            partial_path.reverse()

            yield {
                "step": step,
                "current_node": curr_coord,
                "current_cost": curr_cost,
                "partial_path": partial_path,
                "explored_count": len(explored_set),
                "frontier_size": len(frontier),
                "is_goal": (curr_coord == resolved_goal),
            }

            if curr_coord == resolved_goal:
                final_cost = int(curr_cost) if curr_cost == int(curr_cost) else curr_cost
                return UCSGridResult(
                    success=True,
                    start=resolved_start,
                    goal=resolved_goal,
                    path=partial_path,
                    cost=final_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    explored_set=explored_set,
                    allow_diagonal=self.allow_diagonal,
                    diagonal_multiplier=self.diagonal_multiplier,
                    grid_shape=(topo_grid.height, topo_grid.width),
                )

            neighbors = topo_grid.get_neighbors(
                r=curr_coord[0],
                c=curr_coord[1],
                allow_diagonal=self.allow_diagonal,
                diagonal_multiplier=self.diagonal_multiplier,
                prevent_corner_cutting=self.prevent_corner_cutting,
            )

            for neighbor_coord, step_cost, _ in neighbors:
                if neighbor_coord in explored_set:
                    continue
                new_cost = curr_cost + step_cost
                if new_cost < best_cost.get(neighbor_coord, float("inf")):
                    best_cost[neighbor_coord] = new_cost
                    parent[neighbor_coord] = curr_coord
                    heapq.heappush(frontier, (new_cost, counter, neighbor_coord))
                    counter += 1
                    nodes_generated += 1

        return UCSGridResult(
            success=False,
            start=resolved_start,
            goal=resolved_goal,
            path=[],
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            explored_set=explored_set,
            allow_diagonal=self.allow_diagonal,
            diagonal_multiplier=self.diagonal_multiplier,
            grid_shape=(topo_grid.height, topo_grid.width),
        )


def solve_ucs_grid(
    grid: Union[Sequence[str], str],
    start: Optional[Coordinate] = None,
    goal: Optional[Coordinate] = None,
    allow_diagonal: bool = False,
    diagonal_multiplier: float = math.sqrt(2),
    terrain_costs: Optional[Dict[str, int]] = None,
    obstacle_chars: Optional[Set[str]] = None,
) -> UCSGridResult:
    """Convenience functional interface to execute Uniform Cost Search on a 2D grid.

    Args:
        grid: List of strings or multi-line string representing the grid.
        start: (row, col) start coordinate (or None to infer from 'S' / (0,0)).
        goal: (row, col) goal coordinate (or None to infer from 'G' / bottom-right).
        allow_diagonal: Whether diagonal moves are allowed.
        diagonal_multiplier: Cost multiplier for diagonal moves (default sqrt(2)).
        terrain_costs: Optional custom character cost dictionary.
        obstacle_chars: Optional custom set of obstacle characters.

    Returns:
        UCSGridResult with path, cost, and search metrics.
    """
    solver = UCSGridSolver(
        allow_diagonal=allow_diagonal,
        diagonal_multiplier=diagonal_multiplier,
    )
    return solver.search(
        grid=grid,
        start=start,
        goal=goal,
        terrain_costs=terrain_costs,
        obstacle_chars=obstacle_chars,
    )


def format_path_coords(path: Sequence[Coordinate]) -> str:
    """Format a sequence of coordinates into the canonical arrow representation: (r,c)->(r,c)->..."""
    return "->".join(f"({r},{c})" for r, c in path)


def render_grid_with_path(
    grid: Union[Sequence[str], str],
    path: Sequence[Coordinate],
    explored: Optional[Set[Coordinate]] = None,
    start: Optional[Coordinate] = None,
    goal: Optional[Coordinate] = None,
) -> str:
    """Render the grid with the solution path and optional explored wavefront in ASCII."""
    topo_grid = TopologicalGrid(grid)
    return topo_grid.render_ascii(path=path, explored=explored, start=start, goal=goal)


# Comparative Analysis Helper: UCS vs BFS
def compare_ucs_vs_bfs(
    grid: Union[Sequence[str], str],
    start: Optional[Coordinate] = None,
    goal: Optional[Coordinate] = None,
) -> Dict[str, Any]:
    """Demonstrate why Breadth-First Search (BFS) is suboptimal when step costs vary.

    Runs both standard BFS (minimizing hops) and UCS (minimizing cost),
    comparing their resulting hop counts and total cumulative terrain costs.
    """
    from collections import deque

    topo_grid = TopologicalGrid(grid)
    resolved_start = start if start is not None else (topo_grid.detected_start or (0, 0))
    resolved_goal = goal if goal is not None else (topo_grid.detected_goal or (topo_grid.height - 1, topo_grid.width - 1))

    # 1. Run UCS (Cost-Optimal)
    ucs_res = solve_ucs_grid(grid, start=resolved_start, goal=resolved_goal, allow_diagonal=False)

    # 2. Run standard BFS (Hop-Optimal)
    queue = deque([resolved_start])
    bfs_parent: Dict[Coordinate, Coordinate] = {}
    bfs_visited = {resolved_start}
    bfs_found = False

    while queue:
        curr = queue.popleft()
        if curr == resolved_goal:
            bfs_found = True
            break
        for nxt, _, _ in topo_grid.get_neighbors(curr[0], curr[1], allow_diagonal=False):
            if nxt not in bfs_visited:
                bfs_visited.add(nxt)
                bfs_parent[nxt] = curr
                queue.append(nxt)

    bfs_path: List[Coordinate] = []
    bfs_cost = float("inf")
    if bfs_found:
        curr = resolved_goal
        while curr != resolved_start:
            bfs_path.append(curr)
            curr = bfs_parent[curr]
        bfs_path.append(resolved_start)
        bfs_path.reverse()

        # Calculate actual cost of BFS path
        bfs_cost = sum(topo_grid.get_cell_cost(r, c) for r, c in bfs_path[1:])

    return {
        "ucs": {
            "path": ucs_res.path,
            "cost": ucs_res.cost,
            "hops": len(ucs_res.path) - 1 if ucs_res.path else 0,
            "expanded": ucs_res.nodes_expanded,
        },
        "bfs": {
            "path": bfs_path,
            "cost": int(bfs_cost) if bfs_cost == int(bfs_cost) else bfs_cost,
            "hops": len(bfs_path) - 1 if bfs_path else 0,
            "expanded": len(bfs_visited),
        },
        "cost_savings": (bfs_cost - ucs_res.cost) if bfs_found and ucs_res.success else 0,
        "is_ucs_cheaper": ucs_res.cost < bfs_cost if bfs_found and ucs_res.success else False,
    }


# Comprehensive Viva Voce Reference Guide
def get_ucs_grid_viva_reference() -> Dict[str, Any]:
    """Return structured university examination Viva Voce questions, theory, and authoritative answers."""
    return {
        "topic": "Uniform Cost Search on 2D Grids with Non-Uniform Costs",
        "syllabus_area": "Uninformed Search Techniques",
        "difficulty": "Advanced",
        "questions": [
            {
                "question": "Explain the Priority Queue popping mechanism in UCS and why the goal test must occur at pop rather than push.",
                "key_concepts": ["Min-Heap Property", "Goal Test Placement", "Optimality Proof", "Lazy Deletion"],
                "answer": (
                    "In Uniform Cost Search (UCS), the frontier is maintained as a min-heap ordered by cumulative path cost g(n). "
                    "When heapq.heappop() extracts the minimum element, it guarantees that no unexplored path currently in the "
                    "frontier has a lower path cost than the popped node. "
                    "\n\nCrucially, the goal test MUST be performed when a node is selected for EXPANSION (popped from the priority queue), "
                    "NOT when it is GENERATED (pushed to the queue). If the goal test were applied at generation (like in standard BFS), "
                    "the algorithm would terminate upon discovering the FIRST path to the goal. In graphs with non-uniform edge weights, "
                    "the first path found may consist of fewer hops but carry a vastly higher cumulative cost (e.g. crossing an expensive "
                    "mountain or water body in 2 steps instead of 4 clear ground steps). Testing at expansion guarantees that all paths with "
                    "cost strictly less than the goal's cost have already been fully explored, mathematically proving path cost optimality."
                ),
            },
            {
                "question": "How does Uniform Cost Search deal with non-uniform step costs, and why does Breadth-First Search (BFS) fail here?",
                "key_concepts": ["Non-Uniform Step Costs", "Wavefront Propagation", "Hop-Count vs Path-Cost"],
                "answer": (
                    "Breadth-First Search assumes uniform unit edge weights (c = 1 for all transitions). It expands states in order of hop-depth d, "
                    "meaning it guarantees finding the path with the minimum number of transitions, but NOT the minimum total cost. "
                    "\n\nUCS generalizes BFS by expanding states along contours of equal cumulative cost g(n). It treats the map as a continuous "
                    "cost gradient: a step onto clear ground '.' costs 1, a mountain 'M' costs 5, and water 'W' costs 10. "
                    "UCS will happily take a 4-step detour around a mountain through clear ground (cost 1+1+1+1 = 4) rather than a 1-step direct "
                    "climb into the mountain (cost 5). In terms of search geometry, while BFS expands concentric circles in grid space, UCS expands "
                    "deformed cost contours that bend and flow around high-cost terrain features."
                ),
            },
            {
                "question": "Discuss the Completeness of Uniform Cost Search with variable edge weights.",
                "key_concepts": ["Completeness", "Epsilon Lower Bound", "Infinite Paths", "Zeno's Paradox"],
                "answer": (
                    "UCS is complete if and only if two formal mathematical conditions are satisfied: "
                    "\n1. The branching factor b is finite. "
                    "\n2. Every step cost is bounded below by a strictly positive constant epsilon: c(s, a, s') >= eps > 0. "
                    "\n\nIf step costs were permitted to be zero (or approach zero arbitrarily without a positive lower bound), the search could "
                    "enter an infinite zero-cost loop or an infinite sequence of fractional steps whose infinite sum converges to a finite value less "
                    "than the goal's cost (Zeno's paradox, e.g. steps of 1/2, 1/4, 1/8...). Under the positive lower bound condition (eps > 0), "
                    "the cumulative cost along any infinite path diverges to infinity (lim g(n) = inf). Consequently, the number of states with "
                    "g(n) <= C* (where C* is the optimal goal cost) is strictly finite, guaranteeing that UCS will terminate and find the optimal goal."
                ),
            },
            {
                "question": "Explain the mechanics of the Optional Challenge: allowing diagonal moves with a calculated multiplier cost.",
                "key_concepts": ["Euclidean Metric", "Octile Distance", "Corner Cutting", "Calculated Multiplier"],
                "answer": (
                    "When diagonal movements are enabled, an agent transitions to 8 neighboring cells instead of 4. "
                    "Moving diagonally across a unit grid square covers a geometric distance of sqrt(1^2 + 1^2) = sqrt(2) approx 1.4142. "
                    "To maintain physical metric realism, the step cost for entering a cell diagonally is scaled by this multiplier: "
                    "step_cost = cell_cost * sqrt(2). "
                    "\n\nFurthermore, realistic pathfinding must account for 'corner cutting'. If an agent attempts to move diagonally from "
                    "(r, c) to (r+1, c+1), but both orthogonal adjacent cells (r+1, c) and (r, c+1) contain impassable obstacles (walls), "
                    "physical geometry prevents the agent from squeezing through the infinitely narrow zero-width tangent point. "
                    "The solver includes corner-cutting prevention rules to enforce valid spatial navigation."
                ),
            },
            {
                "question": "What is the Time and Space Complexity of Uniform Cost Search on an H x W grid?",
                "key_concepts": ["Time Complexity O(E log V)", "Space Complexity O(V)", "Binary Min-Heap"],
                "answer": (
                    "Let V = H * W be the total number of cells in the grid, and E be the total number of valid directed edges between adjacent cells. "
                    "In 4-directional grid navigation, each cell has at most 4 neighbors, so E <= 4 * V. With diagonals enabled, E <= 8 * V. "
                    "\n\n- Time Complexity: O(E log V). Each unique grid cell is expanded at most once (popped from min-heap in O(log V)), "
                    "and each edge relaxation inserts into the min-heap at most once (O(log V)). Hence the overall time complexity is O(E log V). "
                    "\n- Space Complexity: O(V). The frontier priority queue, the best_cost hash table, the parent pointer dictionary, "
                    "and the explored set each store at most O(V) coordinates in the worst case."
                ),
            },
        ],
    }
