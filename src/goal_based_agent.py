"""Goal-Based Navigation Agent for 5x5 Grid Matrix.

This module implements a Goal-Based Agent tasked with finding a charging
station (or goal location) in a 5x5 grid matrix (and arbitrary N x M grids).
The agent utilizes an internal representation of its goal coordinates and a
distance-reducing heuristic (Manhattan distance) to select valid orthogonal
actions (Up, Down, Left, Right) that strictly minimize the distance to the goal.

Core AI Concepts:
- Goal-Based Agent: An agent that incorporates explicit goal descriptions to
  evaluate actions based on whether they achieve or progress toward the goal state.
  Unlike reflex agents that map condition -> action reactively, a goal-based agent
  combines environmental percepts with internal goal criteria.
- Distance Metrics & Heuristics:
  - Manhattan Distance (L1 Norm / Taxicab metric): d_M = |x1 - x2| + |y1 - y2|
    Represents the exact minimal path length in a 4-connected orthogonal grid.
  - Euclidean Distance (L2 Norm): d_E = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    Represents the straight-line distance ("as the crow flies").
- Greedy Coordinate Reduction: A search technique that deterministically selects
  successor states that strictly reduce the heuristic distance metric.
- Successor Function with Diagonal Movement Prevention (Optional Challenge):
  The successor function strictly generates 4-connected orthogonal transitions
  (von Neumann neighborhood) and explicitly rejects / prevents diagonal moves
  (|dx| > 0 and |dy| > 0).

Complexity:
- Time Complexity: O(D) where D = |x1 - x2| + |y1 - y2| is the Manhattan distance.
- Space Complexity: O(D) auxiliary space for path storage.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Cardinal Action Constants
ACTION_UP: str = "Up"
ACTION_DOWN: str = "Down"
ACTION_LEFT: str = "Left"
ACTION_RIGHT: str = "Right"

VALID_ACTIONS: Tuple[str, ...] = (
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)

# Orthogonal coordinate displacements: (delta_row / delta_x, delta_col / delta_y)
ORTHOGONAL_DELTAS: Dict[str, Tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

DELTA_TO_ACTION: Dict[Tuple[int, int], str] = {
    (-1, 0): ACTION_UP,
    (1, 0): ACTION_DOWN,
    (0, -1): ACTION_LEFT,
    (0, 1): ACTION_RIGHT,
}

# Diagonal displacements (prohibited in 4-connected grid worlds)
DIAGONAL_DELTAS: Dict[str, Tuple[int, int]] = {
    "Up-Left": (-1, -1),
    "Up-Right": (-1, 1),
    "Down-Left": (1, -1),
    "Down-Right": (1, 1),
}


def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    """Calculate the Manhattan (L1 / Taxicab) distance between two grid coordinates.

    The Manhattan distance corresponds to the minimum number of orthogonal
    moves required to travel between two points on an obstacle-free grid:
        d_M((x1, y1), (x2, y2)) = |x1 - x2| + |y1 - y2|

    Args:
        coord1: First coordinate tuple (x1, y1).
        coord2: Second coordinate tuple (x2, y2).

    Returns:
        Integer Manhattan distance between coord1 and coord2.

    Raises:
        TypeError: If coordinates are not 2-element tuples or ints.
    """
    _validate_coord_tuple(coord1, "coord1")
    _validate_coord_tuple(coord2, "coord2")
    return int(math.fabs(coord1[0] - coord2[0]) + math.fabs(coord1[1] - coord2[1]))


def euclidean_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
    """Calculate the Euclidean (L2 / Straight-line) distance between two grid coordinates.

    The Euclidean distance is given by:
        d_E((x1, y1), (x2, y2)) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

    On a 4-connected grid, Euclidean distance serves as an admissible heuristic
    for search algorithms because d_E <= d_M always holds.

    Args:
        coord1: First coordinate tuple (x1, y1).
        coord2: Second coordinate tuple (x2, y2).

    Returns:
        Float Euclidean distance between coord1 and coord2.
    """
    _validate_coord_tuple(coord1, "coord1")
    _validate_coord_tuple(coord2, "coord2")
    return float(math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1]))


def is_diagonal_movement(delta: Tuple[int, int]) -> bool:
    """Check whether a coordinate displacement vector represents diagonal movement.

    In 2D grid navigation, a diagonal move modifies both coordinate axes simultaneously
    (i.e., |dx| > 0 and |dy| > 0).

    Args:
        delta: Displacement vector (delta_x, delta_y).

    Returns:
        True if the movement is diagonal, False if orthogonal or stationary.
    """
    return delta[0] != 0 and delta[1] != 0


def validate_orthogonal_transition(
    current: Tuple[int, int],
    successor: Tuple[int, int],
) -> None:
    """Enforce strict 4-connected orthogonal movement, rejecting diagonal transitions.

    Optional Challenge Implementation:
    Explicitly checks the coordinate difference between current and successor.
    If both dx != 0 and dy != 0, a ValueError is raised to prevent diagonal cutting.

    Args:
        current: Starting coordinate tuple (x, y).
        successor: Proposed next coordinate tuple (nx, ny).

    Raises:
        ValueError: If the transition represents diagonal movement or step size > 1.
    """
    dx = abs(successor[0] - current[0])
    dy = abs(successor[1] - current[1])

    if dx != 0 and dy != 0:
        raise ValueError(
            f"Diagonal movement prohibited: transition from {current} to {successor} "
            f"has dx={dx} and dy={dy}. Only 4-connected orthogonal actions (Up, Down, "
            f"Left, Right) are permitted."
        )

    if dx + dy > 1:
        raise ValueError(
            f"Invalid step size: transition from {current} to {successor} spans "
            f"distance {dx + dy}, exceeding single-step limit of 1."
        )


def _validate_coord_tuple(coord: Tuple[int, int], arg_name: str) -> None:
    """Validate that an argument is a 2-tuple of integers."""
    if not isinstance(coord, tuple) or len(coord) != 2:
        raise TypeError(f"{arg_name} must be a 2-tuple of integers, got {coord!r}.")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in coord):
        raise TypeError(f"All components of {arg_name} must be integers, got {coord!r}.")


def validate_grid_boundary(
    coord: Tuple[int, int],
    grid_dimensions: Tuple[int, int] = (5, 5),
    coord_name: str = "Coordinate",
) -> None:
    """Validate that a coordinate lies strictly within grid matrix bounds.

    Args:
        coord: Coordinate tuple (x, y) / (row, col).
        grid_dimensions: Grid matrix bounds (rows, cols). Defaults to (5, 5).
        coord_name: Descriptive name for clear error messaging.

    Raises:
        ValueError: If coord is outside [0, rows) x [0, cols).
    """
    _validate_coord_tuple(coord, coord_name)
    rows, cols = grid_dimensions
    r, c = coord
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(
            f"{coord_name} {coord} is outside valid grid matrix dimensions "
            f"[0, {rows}) x [0, {cols})."
        )


def get_successors(
    current_pos: Tuple[int, int],
    grid_dimensions: Tuple[int, int] = (5, 5),
    allow_diagonal: bool = False,
    action_priority: Sequence[str] = (ACTION_DOWN, ACTION_UP, ACTION_RIGHT, ACTION_LEFT),
) -> List[Tuple[str, Tuple[int, int]]]:
    """Generate valid successor actions and coordinates from the current position.

    Enforces grid boundaries and explicitly filters out diagonal movements when
    allow_diagonal is False (default behavior).

    Optional Challenge Note:
    Diagonal movements (e.g., Up-Left, Up-Right, Down-Left, Down-Right) are
    explicitly detected and blocked to uphold 4-connected grid constraints.

    Args:
        current_pos: Current agent coordinate (x, y).
        grid_dimensions: Size of the matrix (rows, cols). Defaults to (5, 5).
        allow_diagonal: Whether diagonal moves are permitted. Defaults to False.
        action_priority: Ordered sequence of actions to evaluate.

    Returns:
        List of valid (action_name, next_coordinate) tuples.
    """
    validate_grid_boundary(current_pos, grid_dimensions, "Current position")
    rows, cols = grid_dimensions
    successors: List[Tuple[str, Tuple[int, int]]] = []

    for action in action_priority:
        if action not in ORTHOGONAL_DELTAS:
            continue
        dr, dc = ORTHOGONAL_DELTAS[action]

        # Explicit diagonal verification guard
        if not allow_diagonal and is_diagonal_movement((dr, dc)):
            continue

        next_r = current_pos[0] + dr
        next_c = current_pos[1] + dc

        if 0 <= next_r < rows and 0 <= next_c < cols:
            next_pos = (next_r, next_c)
            # Enforce orthogonal validation
            if not allow_diagonal:
                validate_orthogonal_transition(current_pos, next_pos)
            successors.append((action, next_pos))

    return successors


def generate_random_coordinate(
    grid_dimensions: Tuple[int, int] = (5, 5),
    exclude: Optional[Sequence[Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> Tuple[int, int]:
    """Generate a pseudo-random coordinate inside the grid boundaries.

    Args:
        grid_dimensions: Grid bounds as (rows, cols). Defaults to (5, 5).
        exclude: Optional sequence of coordinates that cannot be selected.
        seed: Optional integer seed for reproducibility.

    Returns:
        Random coordinate tuple (r, c).

    Raises:
        ValueError: If all grid positions are excluded.
    """
    rows, cols = grid_dimensions
    exclude_set = set(exclude or [])
    all_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in exclude_set]

    if not all_positions:
        raise ValueError("Cannot generate random coordinate: all grid positions are excluded.")

    rng = random.Random(seed)
    return rng.choice(all_positions)


@dataclass(frozen=True)
class NavigationStepRecord:
    """Immutable record of an individual navigation step taken by the agent.

    Attributes:
        step_number: Discrete time step index (1-based).
        position: Agent coordinate before taking the action.
        action: Orthogonal action chosen (Up, Down, Left, Right).
        next_position: Coordinate reached after action.
        distance_before: Manhattan distance to goal prior to move.
        distance_after: Manhattan distance to goal after move.
        path_so_far: Immutable tuple of coordinates visited up to this step.
    """

    step_number: int
    position: Tuple[int, int]
    action: str
    next_position: Tuple[int, int]
    distance_before: int
    distance_after: int
    path_so_far: Tuple[Tuple[int, int], ...]

    @property
    def strictly_reduces_distance(self) -> bool:
        """Check whether this step strictly decreased the Manhattan distance."""
        return self.distance_after < self.distance_before

    def format_trace(self) -> str:
        """Format the step into a descriptive trace log line."""
        return (
            f"Step {self.step_number:2d}: Pos {self.position} -> Action: {self.action:<5} "
            f"-> Next: {self.next_position} | Dist: {self.distance_before} -> {self.distance_after} "
            f"({'Decreased' if self.strictly_reduces_distance else 'Failed'})"
        )


@dataclass
class GoalSimulationResult:
    """Comprehensive summary of a goal-based agent navigation run.

    Attributes:
        start_pos: Initial coordinates of the agent.
        goal_pos: Target charging station coordinates.
        grid_dimensions: Bounds of the grid matrix (rows, cols).
        path: List of coordinate tuples representing the complete path taken.
        actions_taken: List of cardinal action strings executed.
        initial_distance: Manhattan distance at start position.
        total_steps: Number of movement actions taken.
        goal_reached: True if final coordinate equals goal_pos.
        history: Chronological sequence of NavigationStepRecord objects.
    """

    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    grid_dimensions: Tuple[int, int]
    path: List[Tuple[int, int]]
    actions_taken: List[str]
    initial_distance: int
    total_steps: int
    goal_reached: bool
    history: List[NavigationStepRecord] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of the navigation simulation.

        Returns:
            Formatted multiline string summary.
        """
        status_line = "SUCCESS (Goal Reached)" if self.goal_reached else "FAILED (Goal Not Reached)"
        return (
            f"Navigation Status:        {status_line}\n"
            f"Start Coordinate:         {self.start_pos}\n"
            f"Goal Coordinate:          {self.goal_pos}\n"
            f"Grid Matrix Dimensions:   {self.grid_dimensions[0]}x{self.grid_dimensions[1]}\n"
            f"Initial Manhattan Dist:   {self.initial_distance}\n"
            f"Total Moves Executed:     {self.total_steps}\n"
            f"Total Path Coordinates:   {len(self.path)}\n"
            f"Path:                     {self.path}\n"
            f"Actions Taken:            {self.actions_taken}"
        )

    def render_grid(self) -> str:
        """Render a 2D text matrix visualizing the grid, agent, goal, and path.

        Legend:
            'A': Start position
            'G': Goal charging station
            '*': Intermediate path coordinate
            '.': Unvisited empty cell
            'S': Start and Goal coincident (if start == goal)

        Returns:
            String rendering of the 5x5 matrix.
        """
        rows, cols = self.grid_dimensions
        grid = [[" . " for _ in range(cols)] for _ in range(rows)]

        # Mark intermediate path
        for r, c in self.path:
            grid[r][c] = " * "

        # Mark Start
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        if self.start_pos == self.goal_pos:
            grid[sr][sc] = " S "
        else:
            grid[sr][sc] = " A "
            grid[gr][gc] = " G "

        lines = ["    " + "  ".join(f"C{c}" for c in range(cols))]
        lines.append("   +" + "---+" * cols)
        for r in range(rows):
            row_str = f"R{r} |" + "|".join(grid[r]) + "|"
            lines.append(row_str)
            lines.append("   +" + "---+" * cols)

        return "\n".join(lines)


class GoalBasedNavigationAgent:
    """Goal-Based Intelligent Agent using greedy Manhattan distance reduction.

    A Goal-Based Agent uses an explicit representation of its goal to guide its
    decisions. Rather than relying on simple condition-action reflex pairs, this
    agent evaluates available successor actions by estimating the distance to
    the goal from each candidate state, selecting actions that strictly reduce
    the Manhattan distance.

    Attributes:
        goal_pos: Immutable coordinate tuple (gx, gy) of the target charging station.
        grid_dimensions: Grid bounds (rows, cols). Defaults to (5, 5).
        allow_diagonal: Flag enforcing strict orthogonal movement.
        action_priority: Preference order for breaking heuristic ties.
    """

    def __init__(
        self,
        goal_pos: Tuple[int, int],
        grid_dimensions: Tuple[int, int] = (5, 5),
        allow_diagonal: bool = False,
        action_priority: Sequence[str] = (ACTION_DOWN, ACTION_UP, ACTION_RIGHT, ACTION_LEFT),
    ) -> None:
        """Initialize the Goal-Based Navigation Agent.

        Args:
            goal_pos: Goal coordinate tuple (gx, gy).
            grid_dimensions: Matrix size (rows, cols). Defaults to (5, 5).
            allow_diagonal: Whether diagonal moves are permitted. Defaults to False.
            action_priority: Tie-breaking priority order. Defaults to Down, Up, Right, Left.

        Raises:
            ValueError: If goal_pos is outside grid_dimensions.
        """
        validate_grid_boundary(goal_pos, grid_dimensions, "Goal coordinate")
        self.goal_pos: Tuple[int, int] = goal_pos
        self.grid_dimensions: Tuple[int, int] = grid_dimensions
        self.allow_diagonal: bool = allow_diagonal
        self.action_priority: Tuple[str, ...] = tuple(action_priority)

    def calculate_distance(self, current_pos: Tuple[int, int]) -> int:
        """Calculate Manhattan distance from current position to the agent's goal.

        Args:
            current_pos: Position tuple (x, y).

        Returns:
            Manhattan distance integer.
        """
        return manhattan_distance(current_pos, self.goal_pos)

    def select_action(
        self,
        current_pos: Tuple[int, int],
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """Select an action that strictly reduces the Manhattan distance to the goal.

        Greedy Coordinate Reduction:
        1. Evaluate all valid 4-connected successors from current_pos.
        2. Compute Manhattan distance from each candidate successor to goal_pos.
        3. Filter successors strictly where next_dist < current_dist.
        4. Select the best successor according to action_priority tie-breaking.

        Args:
            current_pos: Current position (x, y).

        Returns:
            Tuple of (selected_action, next_position). If already at the goal,
            returns (None, None).

        Raises:
            RuntimeError: If not at goal but no distance-reducing action exists.
        """
        if current_pos == self.goal_pos:
            return None, None

        current_dist = self.calculate_distance(current_pos)
        successors = get_successors(
            current_pos=current_pos,
            grid_dimensions=self.grid_dimensions,
            allow_diagonal=self.allow_diagonal,
            action_priority=self.action_priority,
        )

        # Filter strictly distance-reducing actions
        reducing_candidates: List[Tuple[str, Tuple[int, int], int]] = []
        for action, next_coord in successors:
            dist = manhattan_distance(next_coord, self.goal_pos)
            if dist < current_dist:
                reducing_candidates.append((action, next_coord, dist))

        if not reducing_candidates:
            raise RuntimeError(
                f"Agent trapped: No distance-reducing action available from {current_pos} "
                f"toward goal {self.goal_pos}."
            )

        # Sort by distance (smallest first), preserving tie-break from action_priority
        reducing_candidates.sort(key=lambda item: item[2])
        best_action, best_coord, _ = reducing_candidates[0]
        return best_action, best_coord

    def navigate(
        self,
        start_pos: Tuple[int, int],
        max_steps: int = 100,
    ) -> GoalSimulationResult:
        """Autonomously navigate from start_pos to the charging station goal.

        Tracks the complete path and verifies strict distance reduction at each step.

        Args:
            start_pos: Starting coordinate tuple (x1, y1).
            max_steps: Maximum step safety limit. Defaults to 100.

        Returns:
            GoalSimulationResult containing path, actions, history, and status.

        Raises:
            ValueError: If start_pos is outside grid dimensions.
        """
        validate_grid_boundary(start_pos, self.grid_dimensions, "Start position")

        path: List[Tuple[int, int]] = [start_pos]
        actions_taken: List[str] = []
        history: List[NavigationStepRecord] = []

        current = start_pos
        step_number = 0
        initial_dist = self.calculate_distance(start_pos)

        while current != self.goal_pos and step_number < max_steps:
            step_number += 1
            dist_before = self.calculate_distance(current)
            action, next_pos = self.select_action(current)

            if action is None or next_pos is None:
                break

            dist_after = self.calculate_distance(next_pos)

            # Record step
            record = NavigationStepRecord(
                step_number=step_number,
                position=current,
                action=action,
                next_position=next_pos,
                distance_before=dist_before,
                distance_after=dist_after,
                path_so_far=tuple(path + [next_pos]),
            )
            history.append(record)

            # Move agent
            current = next_pos
            path.append(current)
            actions_taken.append(action)

        goal_reached = (current == self.goal_pos)

        return GoalSimulationResult(
            start_pos=start_pos,
            goal_pos=self.goal_pos,
            grid_dimensions=self.grid_dimensions,
            path=path,
            actions_taken=actions_taken,
            initial_distance=initial_dist,
            total_steps=step_number,
            goal_reached=goal_reached,
            history=history,
        )


class GridEnvironment:
    """5x5 Grid Matrix Environment with a charging station goal.

    Attributes:
        dimensions: Grid bounds (rows, cols). Defaults to (5, 5).
        goal_pos: Charging station coordinates.
        agent_pos: Current agent coordinates.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int] = (5, 5),
        goal_pos: Optional[Tuple[int, int]] = None,
        agent_pos: Tuple[int, int] = (0, 0),
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the 5x5 grid environment.

        Args:
            dimensions: Matrix dimensions (rows, cols). Defaults to (5, 5).
            goal_pos: Optional goal coordinates. If None, placed randomly.
            agent_pos: Initial agent position. Defaults to (0, 0).
            seed: Optional seed for reproducible random placement.
        """
        validate_grid_boundary(agent_pos, dimensions, "Agent position")
        self.dimensions: Tuple[int, int] = dimensions
        self.agent_pos: Tuple[int, int] = agent_pos

        if goal_pos is None:
            self.goal_pos: Tuple[int, int] = generate_random_coordinate(
                grid_dimensions=dimensions,
                exclude=[agent_pos],
                seed=seed,
            )
        else:
            validate_grid_boundary(goal_pos, dimensions, "Goal position")
            self.goal_pos = goal_pos

    @property
    def is_goal_reached(self) -> bool:
        """Check if the agent is currently located at the charging station."""
        return self.agent_pos == self.goal_pos

    def apply_action(self, action: str) -> Tuple[int, int]:
        """Apply an action to move the agent within the environment.

        Args:
            action: Orthogonal action name ('Up', 'Down', 'Left', 'Right').

        Returns:
            New agent coordinate tuple.

        Raises:
            ValueError: If action is invalid or causes collision with grid boundaries.
        """
        if action not in ORTHOGONAL_DELTAS:
            raise ValueError(f"Unknown action '{action}'. Permitted: {VALID_ACTIONS}.")

        dr, dc = ORTHOGONAL_DELTAS[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        rows, cols = self.dimensions
        if not (0 <= nr < rows and 0 <= nc < cols):
            raise ValueError(
                f"Action '{action}' causes boundary violation from {self.agent_pos}."
            )

        self.agent_pos = (nr, nc)
        return self.agent_pos


def run_goal_based_simulation(
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None,
    grid_dimensions: Tuple[int, int] = (5, 5),
    random_goal: bool = False,
    seed: Optional[int] = None,
    allow_diagonal: bool = False,
) -> GoalSimulationResult:
    """High-level simulation runner for Goal-Based Navigation Agent.

    Demonstrates the complete end-to-end task:
    1. Sets up the 5x5 grid matrix with a defined or random charging station goal.
    2. Initializes the GoalBasedNavigationAgent.
    3. Executes greedy coordinate reduction minimizing Manhattan distance.
    4. Produces the complete path and performance metrics.

    Args:
        start_pos: Starting coordinate tuple (x1, y1). Defaults to (0, 0).
        goal_pos: Target charging station (x2, y2). Defaults to (3, 4) if None and not random.
        grid_dimensions: Grid matrix bounds (rows, cols). Defaults to (5, 5).
        random_goal: Whether to place the goal randomly. Defaults to False.
        seed: Random seed for reproducibility.
        allow_diagonal: Flag enforcing strict orthogonal movement. Defaults to False.

    Returns:
        GoalSimulationResult object with path, actions, history, and status.
    """
    if random_goal:
        target_goal = generate_random_coordinate(
            grid_dimensions=grid_dimensions,
            exclude=[start_pos],
            seed=seed,
        )
    elif goal_pos is None:
        target_goal = (3, 4)  # Default sample goal
    else:
        target_goal = goal_pos

    agent = GoalBasedNavigationAgent(
        goal_pos=target_goal,
        grid_dimensions=grid_dimensions,
        allow_diagonal=allow_diagonal,
    )

    return agent.navigate(start_pos=start_pos)
