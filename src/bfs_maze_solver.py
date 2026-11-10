"""Breadth-First Search (BFS) 2D Grid Maze Solver.

This module implements Breadth-First Search (BFS) to navigate and solve 2D grid mazes,
finding the optimal (shortest) path from an initial state (top-left by default) to a
goal state (bottom-right by default) on an implicit graph.

Theoretical Background & AI Concepts:
- Implicit vs. Explicit Graphs:
    In an explicit graph, all vertices and edges are pre-allocated in memory (e.g., as
    adjacency lists or adjacency matrices). In an implicit graph (such as a 2D grid maze,
    the 8-puzzle, or chess), states and transitions are generated dynamically on demand
    using a successor function: S(s) -> { (action, s') }. This drastically conserves
    memory and enables exploration of vast or infinite state spaces.
- Dynamic Successor Function & Action Generation:
    The agent can transition orthogonally in four cardinal directions:
    Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, +1).
    Valid actions are dynamically generated through two essential checks:
    1. Boundary Validation: Ensures row index 0 <= r < N and column index 0 <= c < M.
    2. Obstacle / Wall Collision: Ensures the candidate cell is open (grid[r][c] == 0)
       and not a blocking wall (grid[r][c] == 1).
- FIFO Frontier Queue (collections.deque):
    BFS explores the state space level-by-level (shallowest unexpanded nodes first).
    Python's `collections.deque` provides O(1) amortized enqueue (`append`) and dequeue
    (`popleft`). Using a standard Python list with `pop(0)` would degrade time complexity
    to O(K) per dequeue due to contiguous memory shifts, incurring an undesirable O(V^2)
    overall runtime penalty.
- Visited / Explored Set:
    A hash set (`set`) of coordinate tuples `(r, c)` tracks all visited states.
    To prevent duplicate states in the frontier and avoid combinatorial explosion, cells
    are marked visited immediately upon ENQUEUEING (generation time), rather than upon
    expansion time.
- Optimality & Completeness:
    - Completeness: BFS is guaranteed to find a solution if one exists on any finite grid.
    - Optimality: Because all transitions have uniform unit cost (step cost = 1), BFS
      guarantees finding the shortest path (minimum number of steps/actions).
- Manhattan Distance Relation:
    On a 4-connected grid without diagonal movement, the Manhattan distance (L1 norm)
    between two coordinates (r1, c1) and (r2, c2) is:
        D_Manhattan = |r1 - r2| + |c1 - c2|
    In an obstacle-free grid, the BFS shortest path length is strictly equal to the
    Manhattan distance. When walls are present, the Manhattan distance serves as an
    admissible (never overestimating) lower bound for the true path length, forming the
    theoretical bedrock for informed search algorithms such as A*.
- Path & Action Sequence Reconstruction (Optional Challenge):
    By recording parent pointers `parent[child] = (parent_coord, action)` during search,
    the algorithm backtracks from the goal coordinate back to the start coordinate to
    reconstruct both the ordered list of coordinates and the action sequence
    (e.g., "D, R, R, D").

Complexity:
- Time Complexity: O(N * M), where N is the number of rows and M is the number of columns.
  In a 4-connected grid, every cell is visited at most once, and each cell has at most 4
  orthogonal neighbors (edges <= 4 * N * M).
- Space Complexity: O(N * M) auxiliary memory for the FIFO queue and visited set.
"""

from collections import deque
from dataclasses import dataclass, field
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

# Alternative move ordering: Down, Right, Up, Left (often favors forward downward progression)
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

# Benchmark and sample mazes for laboratory testing, demonstrations, and viva examinations
# Default Sample Maze from Question 15: 3x3 grid, shortest steps = 4, moves = "D, R, R, D"
DEFAULT_SAMPLE_MAZE: List[List[int]] = [
    [0, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
]

# Non-Square Rectangular Maze: 3 rows by 5 columns
NON_SQUARE_RECTANGULAR_MAZE: List[List[int]] = [
    [0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
]

# Unsolvable / Completely Blocked Maze
UNSOLVABLE_BLOCKED_MAZE: List[List[int]] = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
]

# Single-Cell Maze: Start is Goal
SINGLE_CELL_MAZE: List[List[int]] = [
    [0],
]

# Open Obstacle-Free Grid: BFS steps strictly equal Manhattan distance
OPEN_GRID_4X4_MAZE: List[List[int]] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

# Winding Serpentine Corridor Maze (5x5)
SERPENTINE_MAZE: List[List[int]] = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]


@dataclass(frozen=True)
class BFSMazeStepRecord:
    """Immutable trace record capturing an individual node expansion during BFS.

    Attributes:
        step_number: 1-based chronological expansion index.
        current_coord: Coordinate tuple (row, col) popped and expanded.
        action_taken: Action that led to this cell from its parent ('U', 'D', 'L', 'R', or None for start).
        distance: Shortest distance (number of steps) from start to current_coord.
        frontier_snapshot: Snapshot of coordinate tuples currently waiting in the FIFO frontier.
        visited_count: Total number of coordinates marked visited up to this point.
        successors_generated: List of (action, coordinate) pairs generated and enqueued from this state.
    """

    step_number: int
    current_coord: Tuple[int, int]
    action_taken: Optional[str]
    distance: int
    frontier_snapshot: List[Tuple[int, int]]
    visited_count: int
    successors_generated: List[Tuple[str, Tuple[int, int]]]


@dataclass
class BFSMazeResult:
    """Comprehensive result object returned by the BFS 2D Maze Solver.

    Attributes:
        grid: 2D list representing the maze grid (0 = open, 1 = wall).
        start: Starting coordinate tuple (row, col).
        goal: Goal coordinate tuple (row, col).
        found: True if a valid path exists from start to goal; False otherwise.
        steps: Shortest path length in steps (transitions). Returns -1 if no path exists.
        path: Ordered list of coordinate tuples from start to goal (inclusive). Empty if no path.
        actions: Ordered list of action characters (e.g. ['D', 'R', 'R', 'D']). Empty if no path.
        actions_str: Comma-separated string of moves (e.g. "D, R, R, D"). Empty if no path.
        nodes_expanded: Total number of nodes popped from the FIFO frontier.
        nodes_visited: Total number of unique nodes discovered and marked visited.
        manhattan_distance: Theoretical minimum lower bound distance from start to goal.
        execution_time_ms: Wall-clock execution time in milliseconds.
        trace: Optional step-by-step expansion trace for pedagogical debugging.
    """

    grid: List[List[int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    found: bool
    steps: int
    path: List[Tuple[int, int]]
    actions: List[str]
    actions_str: str
    nodes_expanded: int
    nodes_visited: int
    manhattan_distance: int
    execution_time_ms: float
    trace: List[BFSMazeStepRecord] = field(default_factory=list)

    def render_ascii_solution(self) -> str:
        """Render the maze grid as an ASCII string highlighting start, goal, and the solution path.

        Glyphs:
            'S' : Start cell
            'G' : Goal cell
            '*' : Solution path cell
            '#' : Wall obstacle (cell == 1)
            '.' : Open unvisited/unselected cell (cell == 0)

        Returns:
            Multiline string representing the annotated maze grid.
        """
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

    def format_summary(self) -> str:
        """Format a human-readable summary of the BFS maze solution."""
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0
        lines = [
            f"Maze Dimensions   : {rows} x {cols} ({'Square' if rows == cols else 'Non-Square Rectangular'})",
            f"Start Coordinate  : {self.start}",
            f"Goal Coordinate   : {self.goal}",
            f"Solution Found    : {'Yes' if self.found else 'No (Unreachable)'}",
            f"Shortest Steps    : {self.steps if self.found else -1}",
            f"Path Coordinates  : {' -> '.join(str(c) for c in self.path) if self.found else 'None'}",
            f"Action Moves      : {self.actions_str if self.found else 'None'}",
            f"Manhattan Lower   : {self.manhattan_distance} steps (Optimality bound)",
            f"Nodes Expanded    : {self.nodes_expanded}",
            f"Nodes Visited     : {self.nodes_visited} (out of {rows * cols} total cells)",
            f"Execution Time    : {self.execution_time_ms:.4f} ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a serializable dictionary."""
        return {
            "found": self.found,
            "steps": self.steps,
            "path": self.path,
            "actions": self.actions,
            "actions_str": self.actions_str,
            "start": self.start,
            "goal": self.goal,
            "dimensions": (len(self.grid), len(self.grid[0]) if self.grid else 0),
            "manhattan_distance": self.manhattan_distance,
            "nodes_expanded": self.nodes_expanded,
            "nodes_visited": self.nodes_visited,
            "execution_time_ms": self.execution_time_ms,
        }


def calculate_manhattan_distance(start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Calculate the L1 Manhattan distance between two grid coordinates.

    Formula:
        D_Manhattan = |r_start - r_goal| + |c_start - c_goal|

    On a 4-connected grid without diagonal movement, the Manhattan distance
    represents the strict theoretical minimum number of steps required to reach
    the goal, acting as an admissible heuristic lower bound.

    Args:
        start: Starting coordinate tuple (r1, c1).
        goal: Destination coordinate tuple (r2, c2).

    Returns:
        Integer representing the Manhattan distance.
    """
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def is_valid_coordinate(r: int, c: int, num_rows: int, num_cols: int) -> bool:
    """Check whether a given row and column index lies strictly within grid bounds.

    Args:
        r: Row index to test.
        c: Column index to test.
        num_rows: Total number of rows N.
        num_cols: Total number of columns M.

    Returns:
        True if 0 <= r < num_rows and 0 <= c < num_cols; False otherwise.
    """
    return 0 <= r < num_rows and 0 <= c < num_cols


def is_open_cell(r: int, c: int, grid: Sequence[Sequence[int]]) -> bool:
    """Check whether a coordinate is within bounds and is an open traversable path (0).

    Args:
        r: Row index.
        c: Column index.
        grid: 2D grid representation.

    Returns:
        True if the cell is within bounds and equals CELL_OPEN (0); False if out of bounds or a wall (1).
    """
    num_rows = len(grid)
    if num_rows == 0:
        return False
    num_cols = len(grid[0])
    if not is_valid_coordinate(r, c, num_rows, num_cols):
        return False
    return grid[r][c] == CELL_OPEN


def get_successors(
    coord: Tuple[int, int],
    grid: Sequence[Sequence[int]],
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Dynamically generate all valid orthogonal successors for an implicit grid graph state.

    Dynamic Successor Generation Process:
    1. For each candidate direction (e.g. Up, Down, Left, Right):
       - Compute next coordinate: next_r = r + dr, next_c = c + dc.
       - Boundary Check: Verify 0 <= next_r < N and 0 <= next_c < M.
       - Wall / Obstacle Check: Verify grid[next_r][next_c] == 0 (open path).
    2. If both checks pass, the (action, (next_r, next_c)) pair is appended to valid successors.

    Args:
        coord: Current coordinate tuple (r, c).
        grid: 2D sequence representing the maze.
        directions: Optional custom sequence of (action, dr, dc) tuples. Defaults to DEFAULT_DIRECTIONS.

    Returns:
        List of valid (action_name, next_coord) pairs.
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
        next_r = r + dr
        next_c = c + dc

        # Dynamic boundary and obstacle check
        if is_valid_coordinate(next_r, next_c, num_rows, num_cols):
            if grid[next_r][next_c] == CELL_OPEN:
                valid_successors.append((action, (next_r, next_c)))

    return valid_successors


def parse_grid(raw_input: Any) -> List[List[int]]:
    """Parse and normalize various grid input representations into a standard List[List[int]].

    Supports:
    - 2D Python list of ints: `[[0, 1, 0], [0, 0, 0], [1, 1, 0]]`
    - 2D Python list of strings: `[['0', '1', '0'], ...]`
    - JSON string: `'[[0, 1, 0], [0, 0, 0], [1, 1, 0]]'`
    - Multiline text with spaces/commas:
        `0 1 0`
        `0 0 0`
        `1 1 0`
    - Multiline text with ASCII wall characters ('#' for 1, '.' or '0' for 0):
        `. # .`
        `. . .`
        `# # .`

    Args:
        raw_input: Raw input data representing the maze grid.

    Returns:
        Normalized 2D list of integers (0 and 1).

    Raises:
        ValueError: If grid is empty, non-rectangular, or contains invalid cell tokens.
        TypeError: If input cannot be converted to a 2D grid.
    """
    if raw_input is None:
        raise ValueError("Grid input cannot be None.")

    # 1. Handle JSON string or raw text string
    if isinstance(raw_input, str):
        trimmed = raw_input.strip()
        if not trimmed:
            raise ValueError("Grid input string is empty.")

        # Try JSON parsing first (e.g. "[[0,1,0],[0,0,0],[1,1,0]]")
        if (trimmed.startswith("[") and trimmed.endswith("]")) or (trimmed.startswith("(") and trimmed.endswith(")")):
            try:
                parsed_json = json.loads(trimmed)
                return parse_grid(parsed_json)
            except Exception:
                # If JSON fails, proceed to text-line parsing
                pass

        # Split into lines
        lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
        if not lines:
            raise ValueError("No valid rows found in grid string.")

        grid_matrix: List[List[int]] = []
        for line_idx, line in enumerate(lines):
            # Strip brackets if line is e.g. "[0, 1, 0],"
            clean_line = line.strip("[](),;")
            # Split by comma or whitespace
            tokens = [t.strip() for t in clean_line.replace(",", " ").split() if t.strip()]
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
                            raise ValueError(f"Invalid cell value {val} at row {line_idx}. Must be 0 or 1.")
                        row_ints.append(val)
                    except ValueError as err:
                        raise ValueError(f"Unrecognized cell token '{t}' at row {line_idx}: {err}") from err
            grid_matrix.append(row_ints)

        return validate_grid(grid_matrix)

    # 2. Handle Sequence of Sequences (lists, tuples, numpy arrays)
    if isinstance(raw_input, (list, tuple)):
        if len(raw_input) == 0:
            raise ValueError("Grid cannot be an empty list.")

        grid_matrix = []
        for r_idx, row in enumerate(raw_input):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"Row {r_idx} must be a list or tuple, got {type(row).__name__}.")
            row_ints = []
            for c_idx, cell in enumerate(row):
                if isinstance(cell, str):
                    c_clean = cell.strip()
                    if c_clean in ("0", ".", "O", "o"):
                        row_ints.append(CELL_OPEN)
                    elif c_clean in ("1", "#", "W", "w", "X", "x"):
                        row_ints.append(CELL_WALL)
                    else:
                        try:
                            val = int(c_clean)
                            if val not in (CELL_OPEN, CELL_WALL):
                                raise ValueError(f"Invalid cell value {val} at ({r_idx}, {c_idx}). Must be 0 or 1.")
                            row_ints.append(val)
                        except ValueError as err:
                            raise ValueError(f"Invalid cell token '{cell}' at ({r_idx}, {c_idx}): {err}") from err
                elif isinstance(cell, (int, float)):
                    val = int(cell)
                    if val not in (CELL_OPEN, CELL_WALL):
                        raise ValueError(f"Invalid cell value {val} at ({r_idx}, {c_idx}). Must be 0 or 1.")
                    row_ints.append(val)
                elif isinstance(cell, bool):
                    row_ints.append(CELL_WALL if cell else CELL_OPEN)
                else:
                    raise TypeError(f"Invalid cell type {type(cell).__name__} at ({r_idx}, {c_idx}).")
            grid_matrix.append(row_ints)

        return validate_grid(grid_matrix)

    raise TypeError(f"Unsupported grid input type: {type(raw_input).__name__}. Expected 2D list or string.")


def validate_grid(grid: List[List[int]]) -> List[List[int]]:
    """Validate that a grid is non-empty, rectangular, and contains valid cell integers.

    Args:
        grid: 2D list of integers.

    Returns:
        The validated grid.

    Raises:
        ValueError: If grid is empty, rows are empty, or rows have mismatched column lengths.
    """
    num_rows = len(grid)
    if num_rows == 0:
        raise ValueError("Grid must contain at least one row.")

    num_cols = len(grid[0])
    if num_cols == 0:
        raise ValueError("Grid rows cannot be empty.")

    for r_idx, row in enumerate(grid):
        if len(row) != num_cols:
            raise ValueError(
                f"Non-rectangular grid detected: Row 0 has {num_cols} columns, but Row {r_idx} has {len(row)} columns."
            )
        for c_idx, cell in enumerate(row):
            if cell not in (CELL_OPEN, CELL_WALL):
                raise ValueError(f"Cell at ({r_idx}, {c_idx}) has invalid value {cell}. Must be 0 or 1.")

    return grid


def solve_maze_bfs(
    grid_input: Any,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    record_trace: bool = True,
    directions: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> BFSMazeResult:
    """Solve a 2D grid maze using Breadth-First Search (BFS) to find the shortest path.

    Algorithm Workflow:
    1. Parse and validate the 2D grid matrix of size N x M (supports non-square grids).
    2. Define start coordinate (defaults to (0, 0)) and goal coordinate (defaults to (N-1, M-1)).
    3. Verify that start and goal lie within bounds and are not wall obstacles.
       - If start or goal is blocked by a wall, return found=False, steps=-1 immediately.
    4. If start == goal, return immediately with 0 steps and path=[start].
    5. Initialize:
       - FIFO queue: `queue = deque([start])`
       - Visited set: `visited = {start}` (enqueue-time marking prevents duplicate queue entries)
       - Parent dictionary: `parent[start] = (None, None)` mapping child -> (parent_coord, action)
       - Distance dictionary: `distance[start] = 0`
    6. While queue is not empty:
       - Dequeue current coordinate: `curr = queue.popleft()`
       - If curr == goal:
           Backtrack via parent pointers to reconstruct shortest path coordinates and action string.
           Terminate search.
       - Generate valid orthogonal successors dynamically using `get_successors(curr, grid)`:
           For each (action, next_coord):
               If next_coord not in visited:
                   Mark next_coord in visited
                   Record parent[next_coord] = (curr, action)
                   Record distance[next_coord] = distance[curr] + 1
                   Enqueue next_coord into queue
    7. If queue becomes empty without reaching goal, return found=False, steps=-1 (unreachable).

    Args:
        grid_input: 2D list of ints, JSON string, or multiline text representing the maze.
        start: Starting coordinate tuple (row, col). Defaults to (0, 0).
        goal: Destination coordinate tuple (row, col). Defaults to (N-1, M-1).
        record_trace: Whether to capture step-by-step expansion records for visualization/debugging.
        directions: Optional custom directional priority order. Defaults to DEFAULT_DIRECTIONS.

    Returns:
        BFSMazeResult containing shortest steps, path coordinates, moves string, and metrics.
    """
    start_time = time.perf_counter()

    # Parse and validate grid
    grid = parse_grid(grid_input)
    num_rows = len(grid)
    num_cols = len(grid[0])

    # Resolve default start and goal
    resolved_start = (0, 0) if start is None else (int(start[0]), int(start[1]))
    resolved_goal = (num_rows - 1, num_cols - 1) if goal is None else (int(goal[0]), int(goal[1]))

    # Validate coordinate boundaries
    if not is_valid_coordinate(resolved_start[0], resolved_start[1], num_rows, num_cols):
        raise ValueError(
            f"Start coordinate {resolved_start} is out of grid bounds (0 <= r < {num_rows}, 0 <= c < {num_cols})."
        )
    if not is_valid_coordinate(resolved_goal[0], resolved_goal[1], num_rows, num_cols):
        raise ValueError(
            f"Goal coordinate {resolved_goal} is out of grid bounds (0 <= r < {num_rows}, 0 <= c < {num_cols})."
        )

    manhattan = calculate_manhattan_distance(resolved_start, resolved_goal)

    # Check for wall collisions at start or goal
    if grid[resolved_start[0]][resolved_start[1]] == CELL_WALL or grid[resolved_goal[0]][resolved_goal[1]] == CELL_WALL:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return BFSMazeResult(
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            steps=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=0,
            nodes_visited=0,
            manhattan_distance=manhattan,
            execution_time_ms=exec_ms,
            trace=[],
        )

    # Base Case: Start is Goal
    if resolved_start == resolved_goal:
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return BFSMazeResult(
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=True,
            steps=0,
            path=[resolved_start],
            actions=[],
            actions_str="",
            nodes_expanded=1,
            nodes_visited=1,
            manhattan_distance=0,
            execution_time_ms=exec_ms,
            trace=[],
        )

    # FIFO Frontier Queue and Visited Set
    queue: deque[Tuple[int, int]] = deque([resolved_start])
    visited: Set[Tuple[int, int]] = {resolved_start}
    parent: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[str]]] = {
        resolved_start: (None, None)
    }
    distance: Dict[Tuple[int, int], int] = {resolved_start: 0}

    trace_records: List[BFSMazeStepRecord] = []
    nodes_expanded = 0
    goal_reached = False

    while queue:
        curr = queue.popleft()
        nodes_expanded += 1
        curr_dist = distance[curr]
        _, curr_action = parent[curr]

        # Early termination upon expanding the goal
        if curr == resolved_goal:
            goal_reached = True
            if record_trace:
                trace_records.append(
                    BFSMazeStepRecord(
                        step_number=nodes_expanded,
                        current_coord=curr,
                        action_taken=curr_action,
                        distance=curr_dist,
                        frontier_snapshot=list(queue),
                        visited_count=len(visited),
                        successors_generated=[],
                    )
                )
            break

        # Dynamic successor generation with boundary and wall checks
        successors = get_successors(curr, grid, directions=directions)
        enqueued_successors: List[Tuple[str, Tuple[int, int]]] = []

        for action, next_coord in successors:
            if next_coord not in visited:
                visited.add(next_coord)
                parent[next_coord] = (curr, action)
                distance[next_coord] = curr_dist + 1
                queue.append(next_coord)
                enqueued_successors.append((action, next_coord))

        if record_trace:
            trace_records.append(
                BFSMazeStepRecord(
                    step_number=nodes_expanded,
                    current_coord=curr,
                    action_taken=curr_action,
                    distance=curr_dist,
                    frontier_snapshot=list(queue),
                    visited_count=len(visited),
                    successors_generated=enqueued_successors,
                )
            )

    exec_ms = (time.perf_counter() - start_time) * 1000.0

    if not goal_reached:
        return BFSMazeResult(
            grid=grid,
            start=resolved_start,
            goal=resolved_goal,
            found=False,
            steps=-1,
            path=[],
            actions=[],
            actions_str="",
            nodes_expanded=nodes_expanded,
            nodes_visited=len(visited),
            manhattan_distance=manhattan,
            execution_time_ms=exec_ms,
            trace=trace_records,
        )

    # Reconstruct optimal path and action sequence via parent backtracking
    reconstructed_path: List[Tuple[int, int]] = []
    reconstructed_actions: List[str] = []
    curr_backtrack: Optional[Tuple[int, int]] = resolved_goal

    while curr_backtrack is not None:
        reconstructed_path.append(curr_backtrack)
        prev_node, action = parent[curr_backtrack]
        if action is not None:
            reconstructed_actions.append(action)
        curr_backtrack = prev_node

    # Reverse to restore chronological start-to-goal order
    reconstructed_path.reverse()
    reconstructed_actions.reverse()
    actions_string = ", ".join(reconstructed_actions)
    shortest_steps = len(reconstructed_path) - 1

    return BFSMazeResult(
        grid=grid,
        start=resolved_start,
        goal=resolved_goal,
        found=True,
        steps=shortest_steps,
        path=reconstructed_path,
        actions=reconstructed_actions,
        actions_str=actions_string,
        nodes_expanded=nodes_expanded,
        nodes_visited=len(visited),
        manhattan_distance=manhattan,
        execution_time_ms=exec_ms,
        trace=trace_records,
    )


class BFSMazeSolver:
    """Object-Oriented BFS Maze Solver encapsulating grid state, search execution, and visualization."""

    def __init__(
        self,
        grid: Any,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        directions: Optional[Sequence[Tuple[str, int, int]]] = None,
    ) -> None:
        """Initialize the maze solver with a parsed grid and endpoints.

        Args:
            grid: 2D list of ints, JSON string, or multiline text.
            start: Optional start coordinate (r, c). Defaults to (0, 0).
            goal: Optional goal coordinate (r, c). Defaults to (N-1, M-1).
            directions: Optional custom directional priority order.
        """
        self.grid = parse_grid(grid)
        self.num_rows = len(self.grid)
        self.num_cols = len(self.grid[0]) if self.num_rows > 0 else 0
        self.start = (0, 0) if start is None else (int(start[0]), int(start[1]))
        self.goal = (self.num_rows - 1, self.num_cols - 1) if goal is None else (int(goal[0]), int(goal[1]))
        self.directions = directions
        self._last_result: Optional[BFSMazeResult] = None

    def solve(self, record_trace: bool = True) -> BFSMazeResult:
        """Execute BFS maze solve and cache the result.

        Args:
            record_trace: Whether to capture step-by-step expansion records.

        Returns:
            BFSMazeResult object.
        """
        self._last_result = solve_maze_bfs(
            grid_input=self.grid,
            start=self.start,
            goal=self.goal,
            record_trace=record_trace,
            directions=self.directions,
        )
        return self._last_result

    def render_ascii(self) -> str:
        """Render the maze grid with solution path as ASCII art."""
        if self._last_result is None:
            self.solve(record_trace=False)
        assert self._last_result is not None
        return self._last_result.render_ascii_solution()

    def get_dimensions(self) -> Tuple[int, int]:
        """Return the grid dimensions (num_rows, num_cols)."""
        return (self.num_rows, self.num_cols)


def visualize_maze_solution(
    grid_input: Any,
    result: Optional[BFSMazeResult] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Path:
    """Generate and save a visual diagram of the maze and its shortest path using Matplotlib.

    Features:
    - Walls drawn in dark slate / black with crisp borders.
    - Open paths drawn in clean off-white / light gray.
    - Start cell marked in vivid green with 'S' label.
    - Goal cell marked in vivid red with 'G' label.
    - Solution path traced with connected blue line and directional arrows.
    - Legend showing Start, Goal, Walls, and Path steps.

    Args:
        grid_input: 2D list of ints representing the maze grid.
        result: Optional pre-computed BFSMazeResult. If None, solve_maze_bfs will be invoked.
        output_path: Target path to save the diagram (e.g. .png). Defaults to reports/bfs_maze_solution.png.
        title: Optional custom figure title.
        show: Whether to display the plot interactively.

    Returns:
        Path to the saved image file.
    """
    import matplotlib

    # Ensure non-interactive backend for headless environments
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    grid = parse_grid(grid_input)
    num_rows = len(grid)
    num_cols = len(grid[0])

    if result is None:
        result = solve_maze_bfs(grid, record_trace=False)

    if output_path is None:
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "bfs_maze_solution.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, num_cols * 1.2), max(6, num_rows * 1.2)))

    # Draw maze grid cells
    for r in range(num_rows):
        for c in range(num_cols):
            # In matplotlib, x corresponds to column index and y corresponds to (num_rows - 1 - r)
            x = c
            y = num_rows - 1 - r

            if grid[r][c] == CELL_WALL:
                facecolor = "#2D3748"  # Dark slate for walls
                edgecolor = "#1A202C"
            else:
                facecolor = "#F7FAFC"  # Light off-white for open corridors
                edgecolor = "#E2E8F0"

            rect = patches.Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
            ax.add_patch(rect)

    # Highlight Start and Goal
    start_r, start_c = result.start
    goal_r, goal_c = result.goal

    start_x, start_y = start_c, num_rows - 1 - start_r
    goal_x, goal_y = goal_c, num_rows - 1 - goal_r

    start_rect = patches.Rectangle(
        (start_x, start_y), 1, 1, facecolor="#48BB78", edgecolor="#276749", linewidth=2.5, alpha=0.9
    )
    ax.add_patch(start_rect)
    ax.text(
        start_x + 0.5,
        start_y + 0.5,
        "S\n(Start)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    goal_rect = patches.Rectangle(
        (goal_x, goal_y), 1, 1, facecolor="#F56565", edgecolor="#9B2C2C", linewidth=2.5, alpha=0.9
    )
    ax.add_patch(goal_rect)
    ax.text(
        goal_x + 0.5,
        goal_y + 0.5,
        "G\n(Goal)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    # Draw Solution Path with line and markers
    if result.found and len(result.path) > 1:
        path_xs = [c + 0.5 for _, c in result.path]
        path_ys = [num_rows - 1 - r + 0.5 for r, _ in result.path]

        # Draw path line
        ax.plot(path_xs, path_ys, color="#3182CE", linewidth=3.5, linestyle="-", zorder=4, label="Shortest BFS Path")
        # Draw intermediate waypoints
        ax.scatter(path_xs[1:-1], path_ys[1:-1], color="#3182CE", s=80, zorder=5)

        # Draw small directional arrows along the path
        for i in range(len(path_xs) - 1):
            dx = path_xs[i + 1] - path_xs[i]
            dy = path_ys[i + 1] - path_ys[i]
            ax.annotate(
                "",
                xy=(path_xs[i + 1], path_ys[i + 1]),
                xytext=(path_xs[i], path_ys[i]),
                arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=2.5, mutation_scale=15),
                zorder=6,
            )

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(num_cols + 1))
    ax.set_yticks(range(num_rows + 1))
    ax.grid(True, which="both", color="#CBD5E0", linestyle=":", linewidth=0.8)

    # Custom Title & Subtitle
    if title is None:
        title = (
            f"BFS 2D Maze Solver - Shortest Steps: {result.steps} | Moves: \"{result.actions_str}\""
            if result.found
            else "BFS 2D Maze Solver - No Path Found (Goal Unreachable)"
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Column Index (M)", fontsize=10)
    ax.set_ylabel("Row Index (N)", fontsize=10)

    # Invert Y tick labels so (0,0) is visually top-left
    ax.set_yticklabels([str(num_rows - 1 - y) for y in range(num_rows + 1)])

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"[Success] Saved BFS 2D Maze visualization to: {output_path.resolve()}")
    return output_path.resolve()


def get_maze_solver_viva_reference() -> Dict[str, Any]:
    """Retrieve comprehensive Viva Voce questions, answers, and comparison matrix for Question 15.

    Returns:
        Structured dictionary covering implicit vs explicit graphs, successor functions,
        Manhattan distance relations, and complexity characteristics.
    """
    return {
        "title": "Question 15: BFS 2D Maze Solver - Viva Voce & Theory Guide",
        "syllabus_area": "Uninformed Search Techniques",
        "difficulty": "Intermediate",
        "core_topics": [
            {
                "topic": "Implicit vs. Explicit Graphs",
                "question": "What is the key difference between an implicit graph and an explicit graph?",
                "answer": (
                    "In an explicit graph, all vertices, edges, and connection topologies are pre-allocated "
                    "in memory (e.g., using adjacency lists or matrices). In contrast, in an implicit graph "
                    "(like a 2D grid maze or chess board), states and edges are not pre-stored; they are generated "
                    "dynamically on demand using an initial state and a successor function S(s) -> {actions, next_states}. "
                    "This saves massive memory and enables searches over exponentially large or infinite state spaces."
                ),
            },
            {
                "topic": "Valid Action Generation",
                "question": "How are valid actions generated dynamically for a 2D grid maze?",
                "answer": (
                    "For any coordinate (r, c), the successor function evaluates candidate orthogonal displacements: "
                    "Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, +1). An action is valid if and only if: "
                    "(1) Boundary Check: 0 <= next_r < N and 0 <= next_c < M (preventing IndexError). "
                    "(2) Obstacle / Wall Check: grid[next_r][next_c] == 0 (open corridor, avoiding walls of value 1). "
                    "Only surviving actions generate child states to be enqueued."
                ),
            },
            {
                "topic": "Manhattan Distance Relation",
                "question": "How does Manhattan distance relate to the BFS shortest path length on a grid?",
                "answer": (
                    "On a 4-connected grid without diagonal movement, the Manhattan distance "
                    "L1 = |r_goal - r_start| + |c_goal - c_start| represents the absolute theoretical minimum "
                    "number of orthogonal moves required to reach the goal. "
                    "If there are zero obstacles, the BFS shortest path length is strictly equal to the Manhattan distance. "
                    "When walls are present, detours are required, making the BFS path length >= Manhattan distance. "
                    "Therefore, Manhattan distance serves as an admissible (never overestimating) heuristic for informed search (A*)."
                ),
            },
            {
                "topic": "FIFO Queue & collections.deque",
                "question": "Why is collections.deque used instead of a standard Python list for the BFS frontier?",
                "answer": (
                    "collections.deque is implemented as a doubly-linked list of blocks in C, providing O(1) amortized "
                    "time complexity for both append() and popleft(). A standard Python list is a contiguous dynamic array, "
                    "where pop(0) shifts all remaining N-1 elements to the left, taking O(N) time. "
                    "Using list.pop(0) degrades BFS overall time complexity from O(V) to O(V^2)."
                ),
            },
            {
                "topic": "Enqueue-Time vs Expansion-Time Visited Marking",
                "question": "Why must nodes in BFS be marked visited upon ENQUEUEING rather than EXPANSION?",
                "answer": (
                    "In unweighted BFS, marking a node visited as soon as it is enqueued (generation time) prevents "
                    "duplicate copies of the same node from being inserted into the queue by different neighbors on "
                    "the same level. If visited marking was deferred to expansion (popping), the queue would suffer "
                    "redundant insertions, exploding memory and operations exponentially."
                ),
            },
            {
                "topic": "Time and Space Complexity",
                "question": "What are the time and space complexities of BFS on an N x M grid maze?",
                "answer": (
                    "Time Complexity: O(N * M). Each of the V = N * M cells is entered into the queue at most once. "
                    "Each cell has at most 4 orthogonal edges, so total edges E <= 4 * N * M. BFS runs in O(V + E) = O(N * M). "
                    "Space Complexity: O(N * M). The visited set and the FIFO queue store at most O(N * M) coordinate tuples."
                ),
            },
        ],
        "comparison_matrix": {
            "columns": ["Algorithm", "Frontier Data Structure", "Optimality on Maze", "Time Complexity", "Space Complexity"],
            "rows": [
                ["BFS (Breadth-First)", "FIFO Queue (deque)", "Yes (Guaranteed minimum steps)", "O(N * M)", "O(N * M)"],
                ["DFS (Depth-First)", "LIFO Stack / Recursion", "No (Finds arbitrary long paths)", "O(N * M)", "O(N * M)"],
                ["UCS (Uniform Cost)", "Priority Queue (heapq)", "Yes (Minimizes cumulative cost)", "O(NM log(NM))", "O(N * M)"],
                ["A* Search", "Priority Queue (heapq)", "Yes (Optimal with Manhattan heuristic)", "O(NM log(NM))", "O(N * M)"],
            ],
        },
    }
