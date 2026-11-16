"""Breadth-First Search (BFS) for Mazes with Multiple Exits.

This module implements an algorithm modification of Breadth-First Search (BFS)
tailored for multi-goal search spaces (mazes with multiple candidate exits).
The algorithm terminates as soon as it discovers any exit, guaranteeing that
the returned exit is the closest (minimum step distance) from the start coordinate.

Theoretical Background & AI Concepts:
- Multi-Goal Search Spaces:
    In classical single-goal pathfinding, the search terminates when current == goal.
    In multi-goal search spaces, the goal state G is a set of target states: G = {g_1, g_2, ..., g_k}.
    The search problem is defined as: find a path from s_start to any g in G that minimizes path length.
- Optimality Guarantee of Multi-Goal BFS:
    Because BFS expands states in strictly non-decreasing order of path length (level-order traversal,
    d = 0, 1, 2, ...), the first goal state encountered in the frontier/expansion is guaranteed
    to have the minimal path length d* = min_{g in G} dist(s_start, g).
- Efficient Set Membership Testing:
    Testing membership in a list of k exits (`coord in exits_list`) takes O(k) linear time.
    Over V expanded nodes, this results in O(k * V) overhead.
    Converting the exits collection into a hash set (`coord in exits_set`) reduces membership testing
    to O(1) average time, preserving the optimal O(V + E) or O(b^d) BFS complexity.
- Goal Test Placement (Expansion vs. Generation):
    In unit-cost BFS, testing goals on generation (when discovering child nodes) is an admissible
    optimization that saves up to a factor of b node expansions. Testing on expansion (when popping
    the node from the FIFO queue) adheres to classical graph search formulations. Both strategies
    are supported and verified in this module.
- Comparison with Uniform Cost Search (UCS) on Multiple Goals:
    UCS also supports multi-goal search spaces. While BFS finds the exit with the fewest steps (hops),
    UCS expands states in order of cumulative cost g(n) and is guaranteed to find the exit with the
    minimal cumulative path cost under non-uniform edge weights, provided step costs >= epsilon > 0.
- Optional Challenge:
    The solver supports continuing exploration until ALL reachable exits are discovered,
    returning a complete list of paths to all exits sorted by ascending distance.

Complexity:
- Time Complexity: O(b^d) where b is the effective branching factor (b <= 4 on orthogonal grids)
  and d is the depth (shortest path distance) of the closest exit.
- Space Complexity: O(b^d) auxiliary memory for the FIFO queue and visited set.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# Coordinate type alias: (row, column)
Coordinate = Tuple[int, int]

# Grid Cell Definitions
CELL_OPEN: int = 0
CELL_WALL: int = 1

# Cardinal Actions & Directional Vectors (row_delta, col_delta, action_name, glyph)
ACTION_UP: str = "U"
ACTION_DOWN: str = "D"
ACTION_LEFT: str = "L"
ACTION_RIGHT: str = "R"

DEFAULT_DIRECTIONS: Tuple[Tuple[str, int, int], ...] = (
    (ACTION_UP, -1, 0),
    (ACTION_DOWN, 1, 0),
    (ACTION_LEFT, 0, -1),
    (ACTION_RIGHT, 0, 1),
)

# ASCII Visualization Glyphs
GLYPH_START: str = "S"
GLYPH_CLOSEST_EXIT: str = "E*"
GLYPH_OTHER_EXIT: str = "E"
GLYPH_PATH: str = "*"
GLYPH_WALL: str = "#"
GLYPH_OPEN: str = "."

# Canonical Benchmark Mazes

# Question 23 Benchmark: 6 rows x 10 columns
# Start at (0, 0), Exits at [(5, 5), (1, 8)]
# Path to (1, 8) is 9 steps: (0,0)->(0,1)->...->(0,8)->(1,8)
# Path to (5, 5) is 10 steps: (0,0)->(1,0)->...->(5,0)->...->(5,5)
# Closest exit found at (1,8) in 9 steps.
SAMPLE_MULTI_EXIT_MAZE_6X10: List[List[int]] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

SAMPLE_START_Q23: Coordinate = (0, 0)
SAMPLE_EXITS_Q23: List[Coordinate] = [(5, 5), (1, 8)]

# Wall Detour Maze: Exit A (2, 0) is physically closer by Manhattan distance (2),
# but blocked by walls, requiring a detour of 8 steps; Exit B (0, 4) is 4 steps away.
DETOUR_MULTI_EXIT_MAZE: List[List[int]] = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
]

# Maze with Unreachable Exits: Exit (0, 2) is sealed in walls; Exit (2, 2) is open.
PARTIALLY_BLOCKED_MULTI_EXIT_MAZE: List[List[int]] = [
    [0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]


@dataclass(frozen=True)
class MultiGoalStepRecord:
    """Immutable trace record capturing an individual node expansion during Multi-Goal BFS.

    Attributes:
        step_number: 1-based chronological expansion index.
        current_coord: Coordinate tuple (row, col) popped and expanded.
        action_taken: Action that led to this cell from its parent ('U', 'D', 'L', 'R', or None for start).
        distance: Shortest distance (number of steps) from start to current_coord.
        frontier_snapshot: Snapshot of coordinate tuples currently waiting in the FIFO frontier.
        visited_count: Total number of coordinates marked visited up to this point.
        successors_generated: List of (action, coordinate) pairs generated and enqueued from this state.
        is_exit: Whether current_coord is one of the exit coordinates.
    """

    step_number: int
    current_coord: Coordinate
    action_taken: Optional[str]
    distance: int
    frontier_snapshot: List[Coordinate]
    visited_count: int
    successors_generated: List[Tuple[str, Coordinate]]
    is_exit: bool


@dataclass
class SingleExitPathResult:
    """Structured path result for an individual exit found in the multi-exit maze.

    Attributes:
        exit_coord: Coordinate tuple (row, col) of the exit.
        found: True if this exit was reachable from start; False otherwise.
        path_length: Shortest number of steps from start to this exit (-1 if unreachable).
        path: Ordered list of coordinates from start to this exit (inclusive). Empty if unreachable.
        actions: Ordered list of action characters (e.g. ['R', 'D', 'R']). Empty if unreachable.
        actions_str: Comma-separated string of moves. Empty if unreachable.
        manhattan_distance: L1 Manhattan distance |r1 - r2| + |c1 - c2| from start.
    """

    exit_coord: Coordinate
    found: bool
    path_length: int
    path: List[Coordinate]
    actions: List[str]
    actions_str: str
    manhattan_distance: int


@dataclass
class MultiGoalBFSResult:
    """Comprehensive result object returned by the Multi-Goal BFS Maze Solver.

    Attributes:
        maze: 2D list representing the maze grid (0 = open, 1 = wall).
        start: Starting coordinate tuple (row, col).
        exits: List of candidate exit coordinates provided to the search.
        found: True if at least one exit was reached; False otherwise.
        closest_exit: Coordinate tuple (row, col) of the closest exit found (None if unreachable).
        path_length: Shortest path length in steps to the closest exit (-1 if unreachable).
        path: Ordered list of coordinate tuples from start to the closest exit (inclusive).
        actions: Ordered list of action characters to closest exit.
        actions_str: Comma-separated string of moves to closest exit.
        nodes_expanded: Total number of nodes popped from the FIFO frontier.
        nodes_visited: Total number of unique coordinates discovered and marked visited.
        execution_time_ms: Wall-clock execution time in milliseconds.
        all_exits_sorted: List of SingleExitPathResult for all exits, sorted ascending by distance.
        unreachable_exits: List of candidate exit coordinates that could not be reached.
        trace: Optional step-by-step expansion trace for pedagogical debugging.
    """

    maze: List[List[int]]
    start: Coordinate
    exits: List[Coordinate]
    found: bool
    closest_exit: Optional[Coordinate]
    path_length: int
    path: List[Coordinate]
    actions: List[str]
    actions_str: str
    nodes_expanded: int
    nodes_visited: int
    execution_time_ms: float
    all_exits_sorted: List[SingleExitPathResult] = field(default_factory=list)
    unreachable_exits: List[Coordinate] = field(default_factory=list)
    trace: List[MultiGoalStepRecord] = field(default_factory=list)

    def render_ascii_solution(self, show_all_paths: bool = False) -> str:
        """Render the maze grid as an ASCII string highlighting start, exits, and paths.

        Glyphs:
            'S'  : Start cell
            'E*' : Closest exit discovered
            'E'  : Other exit coordinates
            '*'  : Solution path cells
            '#'  : Impassable wall
            '.'  : Open unselected cell

        Args:
            show_all_paths: If True, draws paths to all reachable exits; otherwise only closest exit.

        Returns:
            Multiline string representing the annotated maze grid.
        """
        rows = len(self.maze)
        cols = len(self.maze[0]) if rows > 0 else 0
        path_cells: Set[Coordinate] = set(self.path)

        if show_all_paths:
            for exit_res in self.all_exits_sorted:
                if exit_res.found:
                    path_cells.update(exit_res.path)

        exit_set = set(self.exits)
        lines: List[str] = []

        for r in range(rows):
            row_tokens: List[str] = []
            for c in range(cols):
                coord = (r, c)
                if coord == self.start:
                    row_tokens.append(GLYPH_START)
                elif coord == self.closest_exit:
                    row_tokens.append(GLYPH_CLOSEST_EXIT)
                elif coord in exit_set:
                    row_tokens.append(GLYPH_OTHER_EXIT)
                elif coord in path_cells:
                    row_tokens.append(GLYPH_PATH)
                elif self.maze[r][c] == CELL_WALL:
                    row_tokens.append(GLYPH_WALL)
                else:
                    row_tokens.append(GLYPH_OPEN)
            lines.append(" ".join(f"{tok:>2}" for tok in row_tokens))
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Format a concise human-readable summary of the Multi-Goal BFS search."""
        rows = len(self.maze)
        cols = len(self.maze[0]) if rows > 0 else 0
        lines = [
            "=" * 60,
            "MULTI-GOAL BREADTH-FIRST SEARCH (BFS) MAZE SOLVER",
            "=" * 60,
            f"Maze Dimensions    : {rows} x {cols} ({rows * cols} cells)",
            f"Start Coordinate   : {self.start}",
            f"Target Exits       : {self.exits}",
            f"Closest Exit Found : {self.closest_exit if self.found else 'None (Unreachable)'}",
            f"Shortest Path Len  : {self.path_length if self.found else -1} steps",
            f"Path Sequence      : {' -> '.join(str(c) for c in self.path) if self.found else 'None'}",
            f"Action Moves       : {self.actions_str if self.found else 'None'}",
            f"Nodes Expanded     : {self.nodes_expanded}",
            f"Nodes Visited      : {self.nodes_visited} (out of {rows * cols} total cells)",
            f"Execution Time     : {self.execution_time_ms:.4f} ms",
        ]
        if self.all_exits_sorted:
            lines.append("-" * 60)
            lines.append("ALL CANDIDATE EXITS (SORTED BY DISTANCE):")
            for idx, exit_res in enumerate(self.all_exits_sorted, 1):
                status = f"{exit_res.path_length} steps" if exit_res.found else "UNREACHABLE"
                is_closest = " [CLOSEST]" if exit_res.exit_coord == self.closest_exit else ""
                lines.append(
                    f"  {idx}. Exit {exit_res.exit_coord}: {status}{is_closest} | Moves: {exit_res.actions_str or 'N/A'}"
                )
        if self.unreachable_exits:
            lines.append(f"Unreachable Exits  : {self.unreachable_exits}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a JSON-serializable dictionary."""
        return {
            "start": list(self.start),
            "exits": [list(e) for e in self.exits],
            "found": self.found,
            "closest_exit": list(self.closest_exit) if self.closest_exit else None,
            "path_length": self.path_length,
            "path": [list(p) for p in self.path],
            "actions": self.actions,
            "actions_str": self.actions_str,
            "nodes_expanded": self.nodes_expanded,
            "nodes_visited": self.nodes_visited,
            "execution_time_ms": self.execution_time_ms,
            "all_exits_sorted": [
                {
                    "exit": list(e.exit_coord),
                    "found": e.found,
                    "path_length": e.path_length,
                    "path": [list(p) for p in e.path],
                    "actions": e.actions,
                    "actions_str": e.actions_str,
                    "manhattan_distance": e.manhattan_distance,
                }
                for e in self.all_exits_sorted
            ],
            "unreachable_exits": [list(e) for e in self.unreachable_exits],
        }


class MultiGoalBFSSolver:
    """Configurable solver class for Breadth-First Search on mazes with multiple exits.

    Features:
    - O(1) set membership testing for goal identification.
    - Automatic path and action sequence reconstruction via predecessor pointers.
    - Optional full exploration mode to rank all candidate exits by distance.
    - Configurable goal test placement (on expansion vs on generation).
    - Step-by-step generator for visual animations and pedagogical inspection.
    """

    def __init__(
        self,
        directions: Sequence[Tuple[str, int, int]] = DEFAULT_DIRECTIONS,
        goal_test_on_expansion: bool = True,
    ) -> None:
        """Initialize the Multi-Goal BFS Solver.

        Args:
            directions: Priority sequence of movement vectors (action, delta_row, delta_col).
            goal_test_on_expansion: If True, checks goal upon popping node (textbook standard).
                                    If False, checks goal upon discovering neighbor (early exit).
        """
        self.directions = tuple(directions)
        self.goal_test_on_expansion = goal_test_on_expansion

    @staticmethod
    def _validate_inputs(
        maze: Sequence[Sequence[int]],
        start: Coordinate,
        exits: Iterable[Coordinate],
    ) -> Tuple[List[List[int]], Coordinate, List[Coordinate], Set[Coordinate]]:
        """Validate maze dimensions, start coordinate, and candidate exits.

        Args:
            maze: 2D rectangular grid.
            start: Start coordinate tuple.
            exits: Iterable of exit coordinate tuples.

        Returns:
            Tuple of (validated_maze, start, exits_list, exits_set).

        Raises:
            ValueError: If maze is empty, non-rectangular, or start is out-of-bounds/in wall.
        """
        if not maze or not maze[0]:
            raise ValueError("Maze grid cannot be empty.")

        rows = len(maze)
        cols = len(maze[0])
        for r_idx, row in enumerate(maze):
            if len(row) != cols:
                raise ValueError(
                    f"Inconsistent row length at row {r_idx}: expected {cols}, got {len(row)}."
                )

        grid = [list(row) for row in maze]
        r_start, c_start = start
        if not (0 <= r_start < rows and 0 <= c_start < cols):
            raise ValueError(
                f"Start coordinate {start} is out of bounds for maze of size {rows}x{cols}."
            )

        if grid[r_start][c_start] == CELL_WALL:
            raise ValueError(f"Start coordinate {start} is placed inside an impassable wall.")

        exits_list = list(exits)
        if not exits_list:
            raise ValueError("Candidate exits list cannot be empty.")

        # Filter and validate exits within boundaries
        valid_exits_set: Set[Coordinate] = set()
        for exit_coord in exits_list:
            er, ec = exit_coord
            if 0 <= er < rows and 0 <= ec < cols and grid[er][ec] != CELL_WALL:
                valid_exits_set.add(exit_coord)

        return grid, start, exits_list, valid_exits_set

    @staticmethod
    def _reconstruct_path(
        target: Coordinate,
        parent_map: Dict[Coordinate, Tuple[Optional[Coordinate], Optional[str]]],
    ) -> Tuple[List[Coordinate], List[str], str]:
        """Backtrack from target coordinate to start using predecessor map.

        Args:
            target: The destination coordinate reached.
            parent_map: Dictionary mapping child_coord -> (parent_coord, action_taken).

        Returns:
            Tuple of (ordered_coordinates, ordered_actions, actions_string).
        """
        curr = target
        path: List[Coordinate] = []
        actions: List[str] = []

        while curr is not None:
            path.append(curr)
            prev_node, act = parent_map[curr]
            if act is not None:
                actions.append(act)
            curr = prev_node  # type: ignore[assignment]

        path.reverse()
        actions.reverse()
        actions_str = ", ".join(actions)
        return path, actions, actions_str

    def solve(
        self,
        maze: Sequence[Sequence[int]],
        start: Coordinate,
        exits: Iterable[Coordinate],
        find_all: bool = False,
        record_trace: bool = False,
    ) -> MultiGoalBFSResult:
        """Execute Multi-Goal BFS to find the closest exit (or all exits).

        Args:
            maze: 2D grid where 0 = open and 1 = wall.
            start: Starting coordinate tuple (row, col).
            exits: Collection of candidate exit coordinates.
            find_all: If True, continues search to find paths to all reachable exits.
                      If False, terminates immediately upon finding the first (closest) exit.
            record_trace: If True, captures detailed step-by-step expansion records.

        Returns:
            MultiGoalBFSResult detailing the optimal exit, path length, and trajectory.
        """
        start_time = time.perf_counter()
        grid, start_coord, exits_list, valid_exits_set = self._validate_inputs(maze, start, exits)
        rows, cols = len(grid), len(grid[0])

        # Fast-path: Start coordinate is already one of the exits
        if start_coord in valid_exits_set:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            single_res = SingleExitPathResult(
                exit_coord=start_coord,
                found=True,
                path_length=0,
                path=[start_coord],
                actions=[],
                actions_str="",
                manhattan_distance=0,
            )
            all_sorted = [single_res]
            if find_all:
                # Still check other exits if find_all is requested
                other_res = self._find_remaining_from_start(grid, start_coord, exits_list, {start_coord})
                all_sorted.extend(other_res)
                all_sorted.sort(key=lambda item: (item.path_length if item.found else 999999, item.exit_coord))

            unreachable = [e for e in exits_list if e != start_coord and e not in valid_exits_set]
            return MultiGoalBFSResult(
                maze=grid,
                start=start_coord,
                exits=exits_list,
                found=True,
                closest_exit=start_coord,
                path_length=0,
                path=[start_coord],
                actions=[],
                actions_str="",
                nodes_expanded=1,
                nodes_visited=1,
                execution_time_ms=elapsed_ms,
                all_exits_sorted=all_sorted,
                unreachable_exits=unreachable,
                trace=[],
            )

        # FIFO frontier queue: stores (current_coordinate, current_depth)
        frontier: deque[Tuple[Coordinate, int]] = deque([(start_coord, 0)])

        # Visited tracking and predecessor pointer map
        visited: Set[Coordinate] = {start_coord}
        parent_map: Dict[Coordinate, Tuple[Optional[Coordinate], Optional[str]]] = {
            start_coord: (None, None)
        }

        # Multi-goal tracking structures
        found_exits_map: Dict[Coordinate, SingleExitPathResult] = {}
        nodes_expanded: int = 0
        trace: List[MultiGoalStepRecord] = []
        closest_exit_found: Optional[Coordinate] = None
        closest_path: List[Coordinate] = []
        closest_actions: List[str] = []
        closest_actions_str: str = ""
        closest_distance: int = -1

        while frontier:
            current, dist = frontier.popleft()
            nodes_expanded += 1

            # Goal Test on Expansion
            if self.goal_test_on_expansion and current in valid_exits_set:
                path, actions, actions_str = self._reconstruct_path(current, parent_map)
                m_dist = abs(current[0] - start_coord[0]) + abs(current[1] - start_coord[1])
                exit_result = SingleExitPathResult(
                    exit_coord=current,
                    found=True,
                    path_length=dist,
                    path=path,
                    actions=actions,
                    actions_str=actions_str,
                    manhattan_distance=m_dist,
                )
                if current not in found_exits_map:
                    found_exits_map[current] = exit_result

                if closest_exit_found is None:
                    closest_exit_found = current
                    closest_path = path
                    closest_actions = actions
                    closest_actions_str = actions_str
                    closest_distance = dist

                    if not find_all:
                        # Standard termination: stop at first (closest) exit
                        break

                # If all valid exits have been discovered, stop early even in find_all mode
                if len(found_exits_map) == len(valid_exits_set):
                    break

            successors_generated: List[Tuple[str, Coordinate]] = []

            # Successor Generation across cardinal directions
            for action, dr, dc in self.directions:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor: Coordinate = (nr, nc)

                # Boundary and collision checks
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] != CELL_WALL and neighbor not in visited:
                        visited.add(neighbor)
                        parent_map[neighbor] = (current, action)
                        frontier.append((neighbor, dist + 1))
                        successors_generated.append((action, neighbor))

                        # Goal Test on Generation (optional optimization)
                        if not self.goal_test_on_expansion and neighbor in valid_exits_set:
                            path, actions, actions_str = self._reconstruct_path(neighbor, parent_map)
                            m_dist = abs(neighbor[0] - start_coord[0]) + abs(neighbor[1] - start_coord[1])
                            exit_result = SingleExitPathResult(
                                exit_coord=neighbor,
                                found=True,
                                path_length=dist + 1,
                                path=path,
                                actions=actions,
                                actions_str=actions_str,
                                manhattan_distance=m_dist,
                            )
                            if neighbor not in found_exits_map:
                                found_exits_map[neighbor] = exit_result

                            if closest_exit_found is None:
                                closest_exit_found = neighbor
                                closest_path = path
                                closest_actions = actions
                                closest_actions_str = actions_str
                                closest_distance = dist + 1

                                if not find_all:
                                    break

            if record_trace:
                is_exit = current in valid_exits_set
                _, act = parent_map[current]
                snapshot = [coord for coord, _ in frontier]
                trace.append(
                    MultiGoalStepRecord(
                        step_number=nodes_expanded,
                        current_coord=current,
                        action_taken=act,
                        distance=dist,
                        frontier_snapshot=snapshot,
                        visited_count=len(visited),
                        successors_generated=successors_generated,
                        is_exit=is_exit,
                    )
                )

            if not find_all and closest_exit_found is not None:
                break

            if find_all and len(found_exits_map) == len(valid_exits_set):
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Compile all exit results (sorted by distance)
        all_sorted: List[SingleExitPathResult] = []
        unreachable: List[Coordinate] = []

        for e in exits_list:
            if e in found_exits_map:
                all_sorted.append(found_exits_map[e])
            else:
                m_dist = abs(e[0] - start_coord[0]) + abs(e[1] - start_coord[1])
                unreachable.append(e)
                all_sorted.append(
                    SingleExitPathResult(
                        exit_coord=e,
                        found=False,
                        path_length=-1,
                        path=[],
                        actions=[],
                        actions_str="",
                        manhattan_distance=m_dist,
                    )
                )

        # Sort reachable exits by path_length ascending, then coordinate tuple; unreachable at end
        all_sorted.sort(
            key=lambda item: (0 if item.found else 1, item.path_length if item.found else 999999, item.exit_coord)
        )

        found_any = closest_exit_found is not None
        return MultiGoalBFSResult(
            maze=grid,
            start=start_coord,
            exits=exits_list,
            found=found_any,
            closest_exit=closest_exit_found,
            path_length=closest_distance if found_any else -1,
            path=closest_path,
            actions=closest_actions,
            actions_str=closest_actions_str,
            nodes_expanded=nodes_expanded,
            nodes_visited=len(visited),
            execution_time_ms=elapsed_ms,
            all_exits_sorted=all_sorted,
            unreachable_exits=unreachable,
            trace=trace,
        )

    def _find_remaining_from_start(
        self,
        grid: List[List[int]],
        start: Coordinate,
        exits_list: List[Coordinate],
        already_found: Set[Coordinate],
    ) -> List[SingleExitPathResult]:
        """Auxiliary search to discover remaining exits when start is itself an exit."""
        sub_solver = MultiGoalBFSSolver(directions=self.directions, goal_test_on_expansion=self.goal_test_on_expansion)
        remaining_targets = [e for e in exits_list if e not in already_found]
        if not remaining_targets:
            return []
        res = sub_solver.solve(grid, start, remaining_targets, find_all=True)
        return [r for r in res.all_exits_sorted if r.exit_coord in remaining_targets]

    def step_generator(
        self,
        maze: Sequence[Sequence[int]],
        start: Coordinate,
        exits: Iterable[Coordinate],
    ) -> Generator[MultiGoalStepRecord, None, MultiGoalBFSResult]:
        """Generator yielding step records step-by-step during search execution.

        Useful for live console animation, Jupyter notebook widgets, or pedagogical tracing.

        Yields:
            MultiGoalStepRecord for each node expanded.

        Returns:
            MultiGoalBFSResult when the search concludes.
        """
        grid, start_coord, exits_list, valid_exits_set = self._validate_inputs(maze, start, exits)
        rows, cols = len(grid), len(grid[0])
        start_time = time.perf_counter()

        frontier: deque[Tuple[Coordinate, int]] = deque([(start_coord, 0)])
        visited: Set[Coordinate] = {start_coord}
        parent_map: Dict[Coordinate, Tuple[Optional[Coordinate], Optional[str]]] = {
            start_coord: (None, None)
        }

        nodes_expanded = 0
        trace: List[MultiGoalStepRecord] = []
        closest_exit: Optional[Coordinate] = None
        closest_path: List[Coordinate] = []
        closest_actions: List[str] = []
        closest_actions_str: str = ""
        closest_dist: int = -1

        while frontier:
            current, dist = frontier.popleft()
            nodes_expanded += 1
            is_exit = current in valid_exits_set

            if self.goal_test_on_expansion and is_exit:
                closest_exit = current
                closest_path, closest_actions, closest_actions_str = self._reconstruct_path(
                    current, parent_map
                )
                closest_dist = dist
                record = MultiGoalStepRecord(
                    step_number=nodes_expanded,
                    current_coord=current,
                    action_taken=parent_map[current][1],
                    distance=dist,
                    frontier_snapshot=[c for c, _ in frontier],
                    visited_count=len(visited),
                    successors_generated=[],
                    is_exit=True,
                )
                trace.append(record)
                yield record
                break

            successors_generated: List[Tuple[str, Coordinate]] = []
            for action, dr, dc in self.directions:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] != CELL_WALL and neighbor not in visited:
                        visited.add(neighbor)
                        parent_map[neighbor] = (current, action)
                        frontier.append((neighbor, dist + 1))
                        successors_generated.append((action, neighbor))

            record = MultiGoalStepRecord(
                step_number=nodes_expanded,
                current_coord=current,
                action_taken=parent_map[current][1],
                distance=dist,
                frontier_snapshot=[c for c, _ in frontier],
                visited_count=len(visited),
                successors_generated=successors_generated,
                is_exit=is_exit,
            )
            trace.append(record)
            yield record

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return MultiGoalBFSResult(
            maze=grid,
            start=start_coord,
            exits=exits_list,
            found=closest_exit is not None,
            closest_exit=closest_exit,
            path_length=closest_dist,
            path=closest_path,
            actions=closest_actions,
            actions_str=closest_actions_str,
            nodes_expanded=nodes_expanded,
            nodes_visited=len(visited),
            execution_time_ms=elapsed_ms,
            all_exits_sorted=[],
            unreachable_exits=[],
            trace=trace,
        )


# ============================================================================
# High-Level Functional Interfaces
# ============================================================================


def solve_multi_goal_bfs(
    maze: Sequence[Sequence[int]],
    start: Coordinate,
    exits: Iterable[Coordinate],
    find_all: bool = False,
    record_trace: bool = False,
    goal_test_on_expansion: bool = True,
) -> MultiGoalBFSResult:
    """Solve a maze with multiple exit goals using Breadth-First Search (BFS).

    Args:
        maze: 2D rectangular grid (0 = open, 1 = wall).
        start: Coordinate tuple (row, col) for start position.
        exits: Iterable of exit coordinates.
        find_all: If True, evaluates paths to all candidate exits.
        record_trace: If True, records chronological expansion trace.
        goal_test_on_expansion: If True, tests goal at pop time; False at neighbor generation.

    Returns:
        MultiGoalBFSResult containing optimal exit and path details.
    """
    solver = MultiGoalBFSSolver(goal_test_on_expansion=goal_test_on_expansion)
    return solver.solve(maze, start, exits, find_all=find_all, record_trace=record_trace)


def find_closest_exit(
    maze: Sequence[Sequence[int]],
    start: Coordinate,
    exits: Iterable[Coordinate],
) -> Tuple[Optional[Coordinate], int]:
    """Convenience helper returning just the coordinate of the optimal exit and path length.

    Adheres directly to Task Requirement 3:
    "Return the coordinate of the found exit and the path length."

    Args:
        maze: 2D rectangular grid.
        start: Starting coordinate tuple.
        exits: Iterable of candidate exit coordinates.

    Returns:
        Tuple of (closest_exit_coordinate, path_length_in_steps).
        Returns (None, -1) if no exit is reachable.
    """
    result = solve_multi_goal_bfs(maze, start, exits, find_all=False)
    return result.closest_exit, result.path_length


def find_all_exits_sorted(
    maze: Sequence[Sequence[int]],
    start: Coordinate,
    exits: Iterable[Coordinate],
) -> List[SingleExitPathResult]:
    """Execute Optional Challenge: Return paths to ALL exits sorted by distance.

    Args:
        maze: 2D rectangular grid.
        start: Starting coordinate tuple.
        exits: Iterable of candidate exit coordinates.

    Returns:
        List of SingleExitPathResult sorted in ascending order of path length.
    """
    result = solve_multi_goal_bfs(maze, start, exits, find_all=True)
    return result.all_exits_sorted


def parse_ascii_maze(
    ascii_grid: str,
) -> Tuple[List[List[int]], Coordinate, List[Coordinate]]:
    """Parse a multiline ASCII maze string into binary grid, start, and exits.

    Characters:
        'S' : Start cell (open)
        'E' : Candidate exit cell (open)
        '#' : Impassable wall (1)
        '.' or ' ' : Open passable cell (0)

    Args:
        ascii_grid: Multiline string containing characters.

    Returns:
        Tuple of (binary_maze_grid, start_coordinate, list_of_exit_coordinates).

    Raises:
        ValueError: If start 'S' is missing or no exits 'E' are found.
    """
    raw_lines = [line.strip() for line in ascii_grid.strip().splitlines() if line.strip()]
    rows = len(raw_lines)
    cols = len(raw_lines[0].split()) if " " in raw_lines[0] else len(raw_lines[0])

    maze: List[List[int]] = []
    start: Optional[Coordinate] = None
    exits: List[Coordinate] = []

    for r, line in enumerate(raw_lines):
        tokens = line.split() if " " in line else list(line)
        row_vals: List[int] = []
        for c, ch in enumerate(tokens):
            ch_upper = ch.upper()
            if ch_upper == "S":
                start = (r, c)
                row_vals.append(CELL_OPEN)
            elif ch_upper in ("E", "G"):
                exits.append((r, c))
                row_vals.append(CELL_OPEN)
            elif ch_upper in ("#", "1", "W", "X"):
                row_vals.append(CELL_WALL)
            else:
                row_vals.append(CELL_OPEN)
        maze.append(row_vals)

    if start is None:
        raise ValueError("ASCII maze must contain a start cell denoted by 'S'.")
    if not exits:
        raise ValueError("ASCII maze must contain at least one exit cell denoted by 'E' or 'G'.")

    return maze, start, exits


# ============================================================================
# Pedagogical & University Viva Reference Functions
# ============================================================================


def benchmark_set_vs_list_lookup(
    num_goals: int = 1000,
    num_queries: int = 50000,
) -> Dict[str, Any]:
    """Empirically compare set membership testing vs list membership testing.

    Demonstrates Viva Topic 1: 'Testing for multiple goals efficiently'.
    Shows why converting goals to a hash set is critical for algorithmic scalability.

    Args:
        num_goals: Number of target goal coordinates in collection.
        num_queries: Number of coordinate lookup queries to perform.

    Returns:
        Dictionary with lookup times, speedup ratio, and asymptotic explanations.
    """
    import random

    goals_list = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(num_goals)]
    goals_set = set(goals_list)

    queries = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(num_queries)]

    # Measure List linear search O(K)
    t0 = time.perf_counter()
    list_hits = sum(1 for q in queries if q in goals_list)
    t_list = (time.perf_counter() - t0) * 1000.0

    # Measure Set hash lookup O(1)
    t1 = time.perf_counter()
    set_hits = sum(1 for q in queries if q in goals_set)
    t_set = (time.perf_counter() - t1) * 1000.0

    speedup = t_list / max(t_set, 1e-6)

    return {
        "num_goals": num_goals,
        "num_queries": num_queries,
        "list_time_ms": round(t_list, 2),
        "set_time_ms": round(t_set, 2),
        "speedup_factor": round(speedup, 1),
        "list_complexity": "O(k) per check => O(k * V) total",
        "set_complexity": "O(1) per check => O(V + k) total",
        "explanation": (
            "Python's set uses an open-addressing hash table with average O(1) membership testing. "
            f"For {num_goals} goals, hash set is approximately {speedup:.1f}x faster than list scanning."
        ),
    }


def compare_multi_goal_bfs_vs_ucs() -> Dict[str, Any]:
    """Demonstrate Viva Topic 2: 'Does UCS still work with multiple goals?'.

    Constructs a scenario with non-uniform step costs:
    - Path North through Swamp to Exit A (0, 2): 2 steps, but cost = 10 + 10 = 20.
    - Path South through Clear Road to Exit B (1, 3): 4 steps, but cost = 1 + 1 + 1 + 1 = 4.

    Shows that:
    - Multi-Goal BFS optimizes step count (hops), choosing Exit A (2 steps, cost 20).
    - Multi-Goal UCS optimizes cumulative path cost, choosing Exit B (4 steps, cost 4).
    - Both algorithms work cleanly with multiple goals, but optimize different metrics.

    Returns:
        Dictionary comparing BFS vs UCS decisions, path lengths, and costs.
    """
    import heapq

    # 2x4 grid with start at (0, 0)
    # Row 0: (0,0)=0, (0,1)=10 (Swamp), (0,2)=10 (Exit A, Swamp), (0,3)=Wall
    # Row 1: (1,0)=1 (Road), (1,1)=1 (Road), (1,2)=1 (Road), (1,3)=1 (Exit B, Road)
    start = (0, 0)
    exit_a = (0, 2)
    exit_b = (1, 3)
    exits = [exit_a, exit_b]
    exit_set = set(exits)

    costs: Dict[Coordinate, int] = {
        (0, 0): 0,
        (0, 1): 10,
        (0, 2): 10,  # Exit A
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
        (1, 3): 1,   # Exit B
    }

    # Multi-Goal BFS (unit step cost = 1)
    queue: deque[Tuple[Coordinate, int]] = deque([(start, 0)])
    visited_bfs = {start}
    bfs_exit = None
    bfs_steps = -1
    while queue:
        curr, steps = queue.popleft()
        if curr in exit_set:
            bfs_exit = curr
            bfs_steps = steps
            break
        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            if neighbor in costs and neighbor not in visited_bfs:
                visited_bfs.add(neighbor)
                queue.append((neighbor, steps + 1))

    # Multi-Goal UCS (variable step cost)
    pq: List[Tuple[int, int, Coordinate, int]] = [(0, 0, start, 0)]  # (cost, counter, coord, steps)
    best_cost: Dict[Coordinate, int] = {start: 0}
    ucs_exit = None
    ucs_cost = -1
    ucs_steps = -1
    counter = 0

    while pq:
        c_cost, _, curr, steps = heapq.heappop(pq)
        if curr in exit_set:
            ucs_exit = curr
            ucs_cost = c_cost
            ucs_steps = steps
            break
        if c_cost > best_cost.get(curr, 999999):
            continue

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            if neighbor in costs:
                new_cost = c_cost + costs[neighbor]
                if new_cost < best_cost.get(neighbor, 999999):
                    best_cost[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, neighbor, steps + 1))

    return {
        "start": start,
        "exits": exits,
        "bfs_chosen_exit": bfs_exit,
        "bfs_metric": "Minimum step count (hops)",
        "bfs_steps": bfs_steps,
        "bfs_cost": 20,
        "ucs_chosen_exit": ucs_exit,
        "ucs_metric": "Minimum cumulative path cost g(n)",
        "ucs_steps": ucs_steps,
        "ucs_cost": ucs_cost,
        "ucs_compatible": True,
        "viva_takeaway": (
            "Yes! UCS works seamlessly with multiple goals. While BFS minimizes step hops (choosing Exit A in 2 steps), "
            "UCS expands nodes in non-decreasing order of path cost g(n) and guarantees finding "
            "the exit with the lowest cumulative cost (choosing Exit B with cost 4 despite taking 4 steps). "
            "Crucially, goal testing in UCS MUST be performed upon expansion (heap pop), NOT generation."
        ),
    }


def get_multi_goal_bfs_viva_reference() -> Dict[str, Any]:
    """Retrieve comprehensive Viva Voce questions, answers, and core theoretical concepts.

    Covers:
    - Multi-goal search spaces.
    - Set membership vs list scanning complexity.
    - Does UCS still work with multiple goals?
    - Goal test timing: Expansion vs Generation in BFS.
    - Sorting paths to all exits (Optional Challenge).
    - Time and space complexity.
    """
    return {
        "title": "Question 23: Maze with Multiple Exits (Multi-Goal BFS)",
        "syllabus_area": "Uninformed Search Techniques",
        "difficulty": "Intermediate",
        "core_concept": "Multi-goal search spaces and set membership testing",
        "questions": [
            {
                "question": "How do you modify standard Breadth-First Search to handle multiple goals (exits)?",
                "answer": (
                    "In standard BFS, the goal test is an equality check `current == goal`. "
                    "For multiple goals, the goal state is represented as a collection G = {g_1, g_2, ..., g_k}. "
                    "The goal test is modified to set membership testing `current in goal_set`. "
                    "By using Python's `set`, membership is verified in O(1) time. "
                    "Because BFS expands states in strictly non-decreasing order of path length (level by level), "
                    "the very first goal state reached is guaranteed to be the closest exit (minimal steps)."
                ),
            },
            {
                "question": "Why is testing for multiple goals using a set more efficient than using a list?",
                "answer": (
                    "Testing membership in a Python list (`coord in exits_list`) performs an O(k) linear scan, "
                    "where k is the number of exits. Over a search graph with V nodes, this degrades runtime to O(k * V). "
                    "In contrast, a Python `set` uses open-addressing hash table buckets with O(1) average lookup. "
                    "This reduces total goal-testing overhead to O(V + k) time, preserving the optimal O(b^d) BFS complexity."
                ),
            },
            {
                "question": "Does Uniform Cost Search (UCS) still work with multiple goals?",
                "answer": (
                    "Yes, absolutely. UCS handles multiple goals by maintaining a goal set `goal_set` and testing "
                    "`if current_node in goal_set` upon popping from the min-heap priority queue. "
                    "Because nodes are popped in non-decreasing order of cumulative cost g(n), the first goal popped "
                    "is guaranteed to be the optimal (cheapest) exit, even if another exit was fewer physical steps away. "
                    "Crucially, in UCS the goal test MUST be performed upon expansion (pop), NOT generation."
                ),
            },
            {
                "question": "Can the goal test in BFS be performed upon generation instead of expansion?",
                "answer": (
                    "Yes. In unit-cost BFS, because all edges have weight 1, any goal discovered at depth d+1 "
                    "is strictly at distance d+1. Since all states at depth <= d have already been evaluated, "
                    "no shallower goal can exist. Therefore, checking upon generation is safe, complete, and optimal, "
                    "saving up to a factor of b expansions. Both strategies are supported in this module."
                ),
            },
            {
                "question": "How do you solve the Optional Challenge: Return paths to ALL exits sorted by distance?",
                "answer": (
                    "Instead of halting upon the first goal, the BFS loop continues until either the FIFO queue "
                    "becomes empty or all valid exits in the goal set have been discovered. For each discovered exit, "
                    "its path length and reconstructed path are recorded. Finally, the list of discovered exits is "
                    "sorted in ascending order of path length (with deterministic coordinate tie-breaking)."
                ),
            },
            {
                "question": "What are the Time and Space Complexities of Multi-Goal BFS?",
                "answer": (
                    "Time Complexity is O(b^d) where b is the branching factor (b <= 4 on an orthogonal grid) and "
                    "d is the depth of the CLOSEST goal (or d_max if finding all exits). "
                    "Space Complexity is O(b^d) for storing nodes in the FIFO queue and visited set. "
                    "On an explicit R x C grid, time and space are upper-bounded by O(R * C)."
                ),
            },
        ],
    }


def visualize_multi_goal_maze(
    result: MultiGoalBFSResult,
    save_path: Optional[Union[str, Path]] = None,
    show_all_exits: bool = True,
) -> Optional[Path]:
    """Generate and save a publication-quality Matplotlib visualization of the Multi-Goal BFS search.

    Renders:
    - Obstacle grid walls in charcoal dark grey.
    - Start position with bright blue marker.
    - Closest exit highlighted with prominent green star.
    - Other candidate exits marked with orange/red markers.
    - Optimal path drawn with bright cyan line and step indices.
    - Optional alternative paths to other exits with dashed orange lines.

    Args:
        result: MultiGoalBFSResult from solve_multi_goal_bfs.
        save_path: Optional file path to save PNG. If None, saves to a temporary path.
        show_all_exits: If True, draws paths to all reachable exits.

    Returns:
        Path to the saved image file, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    rows = len(result.maze)
    cols = len(result.maze[0]) if rows > 0 else 0

    grid_matrix = np.array(result.maze)

    fig, ax = plt.subplots(figsize=(max(6, cols * 0.8), max(5, rows * 0.8)))

    # Draw grid cells: walls as dark gray, open cells as light cream
    cmap = plt.cm.colors.ListedColormap(["#F8F9FA", "#212529"])
    ax.imshow(grid_matrix, cmap=cmap, origin="upper")

    # Minor grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#CED4DA", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Major ticks
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))

    # Draw alternative paths if enabled
    if show_all_exits and result.all_exits_sorted:
        for exit_res in result.all_exits_sorted:
            if exit_res.found and exit_res.exit_coord != result.closest_exit:
                pr = [p[0] for p in exit_res.path]
                pc = [p[1] for p in exit_res.path]
                ax.plot(pc, pr, color="#FD7E14", linestyle="--", linewidth=2.0, alpha=0.6, zorder=3)

    # Draw optimal path to closest exit
    if result.found and result.path:
        pr = [p[0] for p in result.path]
        pc = [p[1] for p in result.path]
        ax.plot(pc, pr, color="#0D6EFD", linestyle="-", linewidth=3.5, label="Optimal Path", zorder=4)
        ax.scatter(pc, pr, color="#0D6EFD", s=50, zorder=5)

    # Draw Start cell
    sr, sc = result.start
    ax.scatter(sc, sr, color="#0D6EFD", s=250, marker="o", edgecolors="black", linewidth=2, label="Start (0,0)", zorder=6)
    ax.text(sc, sr, "S", color="white", weight="bold", fontsize=11, ha="center", va="center", zorder=7)

    # Draw candidate exits
    for exit_coord in result.exits:
        er, ec = exit_coord
        if exit_coord == result.closest_exit:
            ax.scatter(
                ec, er, color="#198754", s=350, marker="*", edgecolors="black", linewidth=2,
                label=f"Closest Exit {exit_coord} ({result.path_length} steps)", zorder=8
            )
            ax.text(ec, er + 0.35, f"Exit* ({result.path_length}s)", color="#198754", weight="bold", fontsize=9, ha="center")
        else:
            # Find status
            status = "Unreachable"
            for r in result.all_exits_sorted:
                if r.exit_coord == exit_coord and r.found:
                    status = f"{r.path_length}s"
                    break
            ax.scatter(
                ec, er, color="#DC3545", s=200, marker="X", edgecolors="black", linewidth=1.5,
                label=f"Exit {exit_coord} ({status})", zorder=6
            )
            ax.text(ec, er + 0.35, f"Exit ({status})", color="#DC3545", weight="bold", fontsize=8, ha="center")

    title_str = f"Multi-Goal BFS Maze: Closest Exit at {result.closest_exit} in {result.path_length} steps"
    ax.set_title(title_str, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Column Index", fontsize=10)
    ax.set_ylabel("Row Index", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=True, fontsize=9)

    plt.tight_layout()

    if save_path is None:
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        target_path = tmp_dir / "multi_goal_bfs_maze.png"
    else:
        target_path = Path(save_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target_path
