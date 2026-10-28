"""Model-Based Reflex Agent for 2x2 Grid Environment.

This module implements a Model-Based Reflex Agent capable of cleaning a 2x2
grid environment (and arbitrary N x M grid layouts), addressing the fundamental
limitation of simple reflex agents: lack of internal state and inability to know
when a task is complete.

Core AI Concepts:
- Model-Based Reflex Agent: Maintains an internal state representation reflecting
  both visited locations and belief of cleanliness across the entire environment.
- Internal State Representation:
  - `visited`: Set of visited room coordinates (r, c).
  - `model`: Dictionary mapping coordinates to known cleanliness statuses.
  - `current_pos`: Agent's internal belief of its position.
- Transition Model: Encapsulates how actions transform the world state:
  - "Clean" transforms current room status to "Clean" in the internal model.
  - Directional moves ("Up", "Down", "Left", "Right") update coordinate positions.
- Sensor Model: Encapsulates how physical environment states map to percepts,
  including an optional sensor failure probability where dirty rooms are perceived
  as clean.
- Goal-Directed Termination: The agent stops execution autonomously when its
  internal model confirms all accessible rooms have been visited and are clean.

Complexity:
- Time Complexity: O(N) where N is total rooms (4 for 2x2 grid).
- Space Complexity: O(N) auxiliary space to store visited set and internal model.
"""

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Domain Cleanliness Constants
STATUS_CLEAN: str = "Clean"
STATUS_DIRTY: str = "Dirty"
VALID_STATUSES: Tuple[str, ...] = (STATUS_CLEAN, STATUS_DIRTY)

# Action Constants
ACTION_CLEAN: str = "Clean"
ACTION_SUCK: str = "Suck"  # Alias for Clean in vacuum literature
ACTION_UP: str = "Up"
ACTION_DOWN: str = "Down"
ACTION_LEFT: str = "Left"
ACTION_RIGHT: str = "Right"
ACTION_SHUTDOWN: str = "Shutdown"
ACTION_NOOP: str = "NoOp"

VALID_ACTIONS: Tuple[str, ...] = (
    ACTION_CLEAN,
    ACTION_SUCK,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SHUTDOWN,
    ACTION_NOOP,
)

# Movement vectors: (delta_row, delta_col)
MOVE_DELTAS: Dict[str, Tuple[int, int]] = {
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

TERMINATION_MESSAGE: str = "Agent visited all rooms. Cleaning complete. Shutting down."


def normalize_status(raw_status: str) -> str:
    """Normalize and validate a room status string.

    Args:
        raw_status: Input status string (e.g., 'Clean', 'dirty', 'DIRTY').

    Returns:
        Canonical status ('Clean' or 'Dirty').

    Raises:
        ValueError: If raw_status is not a recognized cleanliness status.
    """
    clean_val = str(raw_status).strip().capitalize()
    if clean_val not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{raw_status}'. Expected one of {VALID_STATUSES}."
        )
    return clean_val


@dataclass(frozen=True)
class ModelAgentStepRecord:
    """Immutable log of a single agent-environment interaction step.

    Attributes:
        step_number: Discrete time step index (1-based).
        position: Agent coordinate (row, col) at perception time.
        percept_status: Cleanliness perceived by the agent's sensor.
        action: Action chosen by the agent.
        internal_visited: Defensive copy of visited coordinates set after step.
        internal_model: Defensive copy of agent's world model dictionary after step.
        env_state_after: Defensive copy of actual 2D grid environment after action.
        sensor_failed: Whether the sensor suffered a failure on this step.
        termination_message: Message produced if agent shuts down on this step.
    """

    step_number: int
    position: Tuple[int, int]
    percept_status: str
    action: str
    internal_visited: Set[Tuple[int, int]]
    internal_model: Dict[Tuple[int, int], str]
    env_state_after: List[List[str]]
    sensor_failed: bool = False
    termination_message: Optional[str] = None

    def format_trace(self) -> str:
        """Format the step into a human-readable log line.

        Returns:
            Formatted trace string.
        """
        sensor_tag = " [SENSOR FAILURE: Dirty perceived as Clean]" if self.sensor_failed else ""
        visited_str = "{" + ", ".join(f"({r},{c})" for r, c in sorted(self.internal_visited)) + "}"
        term_tag = f" -> {self.termination_message}" if self.termination_message else ""
        return (
            f"Step {self.step_number:2d}: Pos {self.position} | "
            f"Percept: {self.percept_status:<5}{sensor_tag} | "
            f"Action: {self.action:<8} | Visited: {visited_str}{term_tag}"
        )


@dataclass
class ModelSimulationResult:
    """Summary of a completed model-based agent simulation run.

    Attributes:
        initial_grid: Copy of initial 2D environment state.
        final_grid: Copy of final 2D environment state.
        start_pos: Initial coordinates of the agent.
        final_pos: Final coordinates of the agent.
        total_steps: Number of interaction steps executed.
        total_cleans: Number of cleaning operations performed.
        total_moves: Number of navigation moves executed.
        visited_rooms: Set of all coordinates visited by the agent.
        is_all_clean_actual: Whether the actual environment is 100% clean.
        is_all_clean_model: Whether the agent's internal model believes all is clean.
        stopped_by_model: True if agent terminated autonomously via internal model.
        termination_message: Final message reported by the agent.
        history: Sequence of step records.
    """

    initial_grid: List[List[str]]
    final_grid: List[List[str]]
    start_pos: Tuple[int, int]
    final_pos: Tuple[int, int]
    total_steps: int
    total_cleans: int
    total_moves: int
    visited_rooms: Set[Tuple[int, int]]
    is_all_clean_actual: bool
    is_all_clean_model: bool
    stopped_by_model: bool
    termination_message: str
    history: List[ModelAgentStepRecord] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary of the simulation run.

        Returns:
            Formatted multiline string.
        """
        status_line = (
            "SUCCESS (Cleanliness Confirmed by Internal Model)"
            if self.stopped_by_model and self.is_all_clean_model
            else "HALTED (Max Steps Exceeded or Model Incomplete)"
        )
        return (
            f"Simulation Status:       {status_line}\n"
            f"Termination Message:     {self.termination_message}\n"
            f"Total Steps Taken:       {self.total_steps}\n"
            f"Total Clean Operations:  {self.total_cleans}\n"
            f"Total Moves Executed:    {self.total_moves}\n"
            f"Rooms Visited:           {len(self.visited_rooms)} / "
            f"{len(self.initial_grid) * len(self.initial_grid[0])} {sorted(self.visited_rooms)}\n"
            f"Actual Environment Clean:{self.is_all_clean_actual}\n"
            f"Internal Model Clean:    {self.is_all_clean_model}\n"
            f"Initial Grid:            {self.initial_grid}\n"
            f"Final Grid:              {self.final_grid}"
        )


class ModelBasedAgent:
    """Model-Based Reflex Agent with internal state tracking and transition model.

    Unlike simple reflex agents which respond only to current percepts without
    memory, a model-based agent maintains an internal world model and visited
    state set. It uses this internal representation to track progress, navigate
    unexplored or dirty rooms, and stop autonomously when the entire grid is clean.

    Attributes:
        grid_dimensions: Tuple (rows, cols) representing environment bounds.
        visited: Set of (row, col) coordinates visited by the agent.
        model: Internal dictionary mapping (row, col) -> status ('Clean' / 'Dirty').
        current_pos: Agent's internal tracking of its current coordinates.
        is_shutdown: Boolean flag indicating if agent reached termination.
    """

    def __init__(
        self,
        grid_dimensions: Tuple[int, int] = (2, 2),
        initial_pos: Tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize the Model-Based Reflex Agent.

        Args:
            grid_dimensions: Grid size as (rows, cols). Defaults to (2, 2).
            initial_pos: Starting position (row, col). Defaults to (0, 0).

        Raises:
            ValueError: If dimensions are <= 0 or initial_pos is out of bounds.
        """
        rows, cols = grid_dimensions
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Grid dimensions must be positive integers, got {grid_dimensions}.")

        r, c = initial_pos
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"Initial position {initial_pos} is outside grid dimensions {grid_dimensions}."
            )

        self.grid_dimensions: Tuple[int, int] = (rows, cols)
        self.visited: Set[Tuple[int, int]] = set()
        self.model: Dict[Tuple[int, int], str] = {}
        self.current_pos: Tuple[int, int] = initial_pos
        self.is_shutdown: bool = False
        self._last_action: Optional[str] = None

    @property
    def total_rooms(self) -> int:
        """Return total number of accessible rooms in the grid."""
        return self.grid_dimensions[0] * self.grid_dimensions[1]

    def all_coordinates(self) -> List[Tuple[int, int]]:
        """Return all valid (row, col) coordinates within grid dimensions."""
        rows, cols = self.grid_dimensions
        return [(r, c) for r in range(rows) for c in range(cols)]

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check whether a coordinate is within grid bounds.

        Args:
            pos: Coordinate tuple (row, col).

        Returns:
            True if position is within bounds, False otherwise.
        """
        r, c = pos
        return 0 <= r < self.grid_dimensions[0] and 0 <= c < self.grid_dimensions[1]

    def update_state(
        self,
        percept: Tuple[Tuple[int, int], str],
        action_executed: Optional[str] = None,
    ) -> None:
        """Update the agent's internal state using percept and transition model.

        The internal state update incorporates:
        1. Sensor Model: Map incoming percept (pos, status) into the internal
           world model `self.model[pos] = status` and mark `pos` in `self.visited`.
        2. Transition Model: If the agent previously performed a "Clean" action,
           update the internal model to reflect that the room is now "Clean".
           If the agent performed a movement action, update `self.current_pos`.

        Args:
            percept: Tuple of (position, status) received from sensors.
            action_executed: Optional action executed in the previous step.
        """
        pos, status = percept
        status_norm = normalize_status(status)

        # Update position belief from percept and register in visited set
        self.current_pos = pos
        self.visited.add(pos)

        # Transition model integration
        if action_executed in (ACTION_CLEAN, ACTION_SUCK):
            self.model[pos] = STATUS_CLEAN
        else:
            self.model[pos] = status_norm

    def transition_model(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Predict next position based on transition model physics.

        Args:
            pos: Current position (row, col).
            action: Movement action ('Up', 'Down', 'Left', 'Right').

        Returns:
            Next position tuple (row, col), bounded by environment walls.
        """
        if action not in MOVE_DELTAS:
            return pos

        dr, dc = MOVE_DELTAS[action]
        candidate = (pos[0] + dr, pos[1] + dc)
        return candidate if self.is_in_bounds(candidate) else pos

    def is_all_clean_in_model(self) -> bool:
        """Determine whether the internal model confirms all rooms are clean.

        Condition for complete cleanliness according to internal model:
        1. All accessible grid coordinates have been visited (len(visited) == total_rooms).
        2. Every accessible room in the internal model is recorded as 'Clean'.

        Returns:
            True if the internal model confirms the entire grid is clean, False otherwise.
        """
        if len(self.visited) < self.total_rooms:
            return False

        for coord in self.all_coordinates():
            if self.model.get(coord) != STATUS_CLEAN:
                return False

        return True

    def get_unvisited_rooms(self) -> Set[Tuple[int, int]]:
        """Return the set of rooms that have not yet been visited.

        Returns:
            Set of unvisited coordinate tuples.
        """
        return set(self.all_coordinates()) - self.visited

    def get_known_dirty_rooms(self) -> Set[Tuple[int, int]]:
        """Return rooms believed to be Dirty based on internal model.

        Returns:
            Set of dirty coordinate tuples.
        """
        return {coord for coord, status in self.model.items() if status == STATUS_DIRTY}

    def plan_next_move(self) -> str:
        """Determine the next navigation action towards an unvisited or dirty room.

        Uses Breadth-First Search (BFS) pathfinding on the internal grid map
        to find the shortest path from `self.current_pos` to the closest
        target room (prioritizing unvisited rooms, then known dirty rooms).

        Returns:
            Next directional movement action ('Up', 'Down', 'Left', 'Right')
            or 'Shutdown' if no valid targets remain.
        """
        # Prioritize unvisited rooms first, then known dirty rooms
        unvisited = self.get_unvisited_rooms()
        targets = unvisited if unvisited else self.get_known_dirty_rooms()

        if not targets:
            return ACTION_SHUTDOWN

        # BFS to find shortest path to nearest target
        start = self.current_pos
        queue: deque[Tuple[Tuple[int, int], List[str]]] = deque([(start, [])])
        visited_in_search: Set[Tuple[int, int]] = {start}

        while queue:
            curr, path = queue.popleft()

            if curr in targets and path:
                return path[0]

            for action, (dr, dc) in MOVE_DELTAS.items():
                nbr = (curr[0] + dr, curr[1] + dc)
                if self.is_in_bounds(nbr) and nbr not in visited_in_search:
                    visited_in_search.add(nbr)
                    queue.append((nbr, path + [action]))

        # Fallback (should not occur in connected grid)
        return ACTION_NOOP

    def decide_action(self, percept: Tuple[Tuple[int, int], str]) -> str:
        """Select action based on current percept and internal world model.

        Algorithmic Decision Policy:
        1. Update internal state with incoming percept.
        2. If current room is perceived/believed 'Dirty' -> Return 'Clean'.
        3. If current room is 'Clean':
           a. Check internal model: If ALL accessible rooms are visited and clean ->
              Set `is_shutdown = True` and return 'Shutdown'.
           b. Otherwise, plan next move towards the closest unvisited/dirty room.

        Args:
            percept: Tuple of (position, status).

        Returns:
            Chosen action ('Clean', 'Up', 'Down', 'Left', 'Right', or 'Shutdown').
        """
        if self.is_shutdown:
            return ACTION_NOOP

        # 1. Update internal state with incoming percept
        self.update_state(percept, action_executed=self._last_action)

        pos, _ = percept

        # 2. Reflex condition: clean dirty room immediately
        if self.model.get(pos) == STATUS_DIRTY:
            action = ACTION_CLEAN
            self._last_action = action
            return action

        # 3. Model-based condition: check if whole grid is clean
        if self.is_all_clean_in_model():
            self.is_shutdown = True
            action = ACTION_SHUTDOWN
            self._last_action = action
            return action

        # 4. Goal-directed navigation towards unvisited or dirty rooms
        action = self.plan_next_move()
        self._last_action = action
        return action


class TwoByTwoGridEnvironment:
    """Simulated 2x2 (or arbitrary N x M) grid environment for intelligent agents.

    Maintains true ground-truth physical environment state in a 2D list array,
    simulates agent actuators with boundary collision physics, and generates
    percepts with optional stochastic sensor failure.

    Attributes:
        grid: 2D list of cleanliness status strings.
        agent_pos: Current true coordinate (row, col) of the agent.
        sensor_failure_prob: Probability [0.0, 1.0] that a Dirty room is
                             perceived falsely as Clean.
        step_count: Total actions executed.
        clean_count: Total clean operations executed.
        move_count: Total directional moves executed.
        last_sensor_failed: Whether the sensor suffered failure on latest percept.
    """

    def __init__(
        self,
        initial_grid: Optional[Sequence[Sequence[str]]] = None,
        start_pos: Tuple[int, int] = (0, 0),
        sensor_failure_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the 2x2 grid environment.

        Args:
            initial_grid: 2D sequence of status strings ('Clean' / 'Dirty').
                          Defaults to standard sample [[Dirty, Clean], [Dirty, Dirty]].
            start_pos: Starting position (row, col). Defaults to (0, 0).
            sensor_failure_prob: Probability that dirty room is perceived clean.
            seed: Optional random seed for reproducible stochastic sensor noise.

        Raises:
            ValueError: If initial_grid is malformed, has irregular rows,
                        or contains invalid statuses, or if start_pos is invalid.
        """
        if initial_grid is None:
            # Standard Question 3 sample input
            initial_grid = [
                [STATUS_DIRTY, STATUS_CLEAN],
                [STATUS_DIRTY, STATUS_DIRTY],
            ]

        if not initial_grid or not initial_grid[0]:
            raise ValueError("Grid cannot be empty. At least 1x1 cells required.")

        rows = len(initial_grid)
        cols = len(initial_grid[0])

        normalized_grid: List[List[str]] = []
        for r_idx, row in enumerate(initial_grid):
            if len(row) != cols:
                raise ValueError(
                    f"Inconsistent row length at row {r_idx}. Expected {cols}, got {len(row)}."
                )
            normalized_row = [normalize_status(cell) for cell in row]
            normalized_grid.append(normalized_row)

        r, c = start_pos
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"Start position {start_pos} is outside grid dimensions ({rows}, {cols})."
            )

        if not (0.0 <= sensor_failure_prob <= 1.0):
            raise ValueError(
                f"Sensor failure probability must be in range [0.0, 1.0], got {sensor_failure_prob}."
            )

        self._grid: List[List[str]] = normalized_grid
        self._rows: int = rows
        self._cols: int = cols
        self._agent_pos: Tuple[int, int] = start_pos
        self.sensor_failure_prob: float = sensor_failure_prob
        self._rng: random.Random = random.Random(seed)

        self.step_count: int = 0
        self.clean_count: int = 0
        self.move_count: int = 0
        self.last_sensor_failed: bool = False

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Return dimensions (rows, cols) of the grid."""
        return (self._rows, self._cols)

    @property
    def grid(self) -> List[List[str]]:
        """Return a defensive deep copy of the current 2D grid state."""
        return [list(row) for row in self._grid]

    @property
    def agent_pos(self) -> Tuple[int, int]:
        """Return current agent coordinates (row, col)."""
        return self._agent_pos

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self._rows and 0 <= c < self._cols

    def get_percept(self) -> Tuple[Tuple[int, int], str]:
        """Generate sensor percept for current agent position.

        If sensor_failure_prob > 0 and the current room is actually 'Dirty',
        with probability `sensor_failure_prob` the sensor will report 'Clean'.

        Returns:
            Tuple of (agent_position, perceived_status).
        """
        r, c = self._agent_pos
        actual_status = self._grid[r][c]

        if actual_status == STATUS_DIRTY and self.sensor_failure_prob > 0.0:
            if self._rng.random() < self.sensor_failure_prob:
                self.last_sensor_failed = True
                return (self._agent_pos, STATUS_CLEAN)

        self.last_sensor_failed = False
        return (self._agent_pos, actual_status)

    def execute_action(self, action: str) -> None:
        """Execute agent action and update physical environment state.

        Args:
            action: Action string to execute.

        Raises:
            ValueError: If action is unknown.
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Expected one of {VALID_ACTIONS}.")

        self.step_count += 1

        if action in (ACTION_CLEAN, ACTION_SUCK):
            r, c = self._agent_pos
            self._grid[r][c] = STATUS_CLEAN
            self.clean_count += 1
        elif action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            next_pos = (self._agent_pos[0] + dr, self._agent_pos[1] + dc)
            if self.is_in_bounds(next_pos):
                self._agent_pos = next_pos
            self.move_count += 1
        elif action in (ACTION_SHUTDOWN, ACTION_NOOP):
            # No-op action
            pass

    def is_all_clean(self) -> bool:
        """Check if all rooms in the actual physical environment are clean."""
        return all(cell == STATUS_CLEAN for row in self._grid for cell in row)

    def render(self) -> str:
        """Render an ASCII representation of the grid with agent position.

        Returns:
            ASCII grid visualization string.
        """
        lines = []
        for r in range(self._rows):
            row_tokens = []
            for c in range(self._cols):
                status_short = "D" if self._grid[r][c] == STATUS_DIRTY else "C"
                agent_mark = "*" if (r, c) == self._agent_pos else " "
                row_tokens.append(f"[{agent_mark}{status_short}]")
            lines.append(" ".join(row_tokens))
        return "\n".join(lines)


def run_model_based_simulation(
    env: TwoByTwoGridEnvironment,
    agent: ModelBasedAgent,
    max_steps: int = 50,
) -> ModelSimulationResult:
    """Execute simulation loop connecting ModelBasedAgent to TwoByTwoGridEnvironment.

    Simulation Lifecycle per Step:
    1. Environment generates percept: `percept = env.get_percept()`
    2. Agent decides action (updating internal state): `action = agent.decide_action(percept)`
    3. If action is 'Shutdown':
       Agent logs termination message, records final step, and simulation halts.
    4. Otherwise:
       Environment executes action: `env.execute_action(action)`
    5. Loop continues until shutdown or max_steps limit is reached.

    Args:
        env: TwoByTwoGridEnvironment instance.
        agent: ModelBasedAgent instance.
        max_steps: Maximum safety step limit. Defaults to 50.

    Returns:
        ModelSimulationResult object with comprehensive metrics and history.
    """
    initial_grid = env.grid
    start_pos = env.agent_pos
    history: List[ModelAgentStepRecord] = []
    step_number = 0
    stopped_by_model = False
    term_message = ""

    while step_number < max_steps:
        step_number += 1

        # 1. Sense environment
        percept = env.get_percept()
        pos, perceived_status = percept
        sensor_failed = env.last_sensor_failed

        # 2. Decide action
        action = agent.decide_action(percept)

        # 3. Handle shutdown
        if action == ACTION_SHUTDOWN:
            stopped_by_model = True
            term_message = TERMINATION_MESSAGE
            record = ModelAgentStepRecord(
                step_number=step_number,
                position=pos,
                percept_status=perceived_status,
                action=action,
                internal_visited=set(agent.visited),
                internal_model=dict(agent.model),
                env_state_after=env.grid,
                sensor_failed=sensor_failed,
                termination_message=term_message,
            )
            history.append(record)
            break

        # 4. Act on environment
        env.execute_action(action)

        record = ModelAgentStepRecord(
            step_number=step_number,
            position=pos,
            percept_status=perceived_status,
            action=action,
            internal_visited=set(agent.visited),
            internal_model=dict(agent.model),
            env_state_after=env.grid,
            sensor_failed=sensor_failed,
            termination_message=None,
        )
        history.append(record)

    if not stopped_by_model:
        term_message = f"HALTED: Reached maximum step limit of {max_steps}."

    return ModelSimulationResult(
        initial_grid=initial_grid,
        final_grid=env.grid,
        start_pos=start_pos,
        final_pos=env.agent_pos,
        total_steps=step_number,
        total_cleans=env.clean_count,
        total_moves=env.move_count,
        visited_rooms=set(agent.visited),
        is_all_clean_actual=env.is_all_clean(),
        is_all_clean_model=agent.is_all_clean_in_model(),
        stopped_by_model=stopped_by_model,
        termination_message=term_message,
        history=history,
    )
