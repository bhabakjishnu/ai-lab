"""Simple Reflex Vacuum Cleaner Agent Simulation.

This module implements a classic Simple Reflex Agent operating within a
two-square vacuum environment (Location A and Location B), modeled after
Russell and Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA).

Core Concepts:
- Simple Reflex Agent: Selects actions based solely on the current percept
  (current location and cleanliness status) without retaining memory of past percepts.
- Condition-Action Rules: Deterministic if-elif mapping:
    - If status is Dirty -> Suck
    - Else if location is A -> Right
    - Else if location is B -> Left
- Environment Representation: Python dictionary tracking state per location.

Complexity:
- Time Complexity: O(1) per step.
- Space Complexity: O(1) auxiliary memory.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Literal, Optional, Tuple

# Standardized domain constants
CLEAN = "Clean"
DIRTY = "Dirty"

ACTION_SUCK = "Suck"
ACTION_RIGHT = "Right"
ACTION_LEFT = "Left"
ACTION_UP = "Up"
ACTION_DOWN = "Down"
ACTION_NOOP = "NoOp"

LOCATION_A = "A"
LOCATION_B = "B"

ValidLocation = Literal["A", "B"]
ValidStatus = Literal["Clean", "Dirty"]
ValidAction = Literal["Suck", "Right", "Left", "NoOp"]


@dataclass(frozen=True)
class StepRecord:
    """Immutable snapshot of an agent-environment interaction step.

    Attributes:
        step_number: Discrete time step index (1-based).
        location: Location where the agent was situated during perception.
        status: Cleanliness status perceived at that location.
        action: Action chosen by the simple reflex agent.
        environment_state_after: State of the environment after action execution.
    """

    step_number: int
    location: str
    status: str
    action: str
    environment_state_after: Dict[str, str]

    def format_trace(self) -> str:
        """Produce the standard lab format output for this step.

        Returns:
            Human-readable formatted step trace string.
        """
        return f"Step {self.step_number}: Location {self.location} is {self.status} -> Action: {self.action}."


class SimpleReflexVacuumAgent:
    """Simple Reflex Agent for the two-square vacuum environment.

    A reflex agent possesses no internal memory or state history. It maps
    current percept (location, status) directly to an action using hardwired
    condition-action rules.
    """

    @staticmethod
    def reflex_rule(location: str, status: str) -> str:
        """Evaluate condition-action reflex rule based strictly on current percept.

        Rules:
        1. If current location status is Dirty -> Suck.
        2. Else if location is 'A' -> Right.
        3. Else if location is 'B' -> Left.

        Args:
            location: The current square identifier ('A' or 'B').
            status: The cleanliness status of the current square ('Clean' or 'Dirty').

        Returns:
            Chosen action ('Suck', 'Right', or 'Left').

        Raises:
            ValueError: If location or status is unrecognized.
        """
        normalized_location = location.strip().upper()
        normalized_status = status.strip().capitalize()

        if normalized_status not in (CLEAN, DIRTY):
            raise ValueError(
                f"Invalid environment status '{status}'. Expected '{CLEAN}' or '{DIRTY}'."
            )

        if normalized_location not in (LOCATION_A, LOCATION_B):
            raise ValueError(
                f"Invalid location '{location}'. Expected '{LOCATION_A}' or '{LOCATION_B}'."
            )

        # Condition-Action Reflex Mapping (Pure function, zero memory)
        if normalized_status == DIRTY:
            return ACTION_SUCK
        elif normalized_location == LOCATION_A:
            return ACTION_RIGHT
        elif normalized_location == LOCATION_B:
            return ACTION_LEFT

        return ACTION_NOOP


class TwoSquareEnvironment:
    """Simulation environment consisting of two squares: Location A and Location B.

    The environment holds the true state in a Python dictionary and updates
    state and agent position according to physics/actuator rules.
    """

    def __init__(
        self,
        initial_location: str = LOCATION_A,
        state: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the two-square vacuum environment.

        Args:
            initial_location: Starting square ('A' or 'B').
            state: Dictionary specifying state of each square.
                   Defaults to {'A': 'Dirty', 'B': 'Dirty'}.

        Raises:
            ValueError: If initial location or states are invalid.
        """
        norm_loc = initial_location.strip().upper()
        if norm_loc not in (LOCATION_A, LOCATION_B):
            raise ValueError(
                f"Invalid starting location '{initial_location}'. Must be '{LOCATION_A}' or '{LOCATION_B}'."
            )
        self._agent_location: str = norm_loc

        # Initialize dictionary representation
        if state is None:
            self._state: Dict[str, str] = {LOCATION_A: DIRTY, LOCATION_B: DIRTY}
        else:
            self._state = {}
            for loc in (LOCATION_A, LOCATION_B):
                if loc not in state:
                    raise ValueError(f"Missing required state key '{loc}' in state dictionary.")
                status_val = str(state[loc]).strip().capitalize()
                if status_val not in (CLEAN, DIRTY):
                    raise ValueError(
                        f"Invalid status '{state[loc]}' for location '{loc}'. Expected '{CLEAN}' or '{DIRTY}'."
                    )
                self._state[loc] = status_val

    @property
    def agent_location(self) -> str:
        """Return the current location of the agent."""
        return self._agent_location

    @property
    def state(self) -> Dict[str, str]:
        """Return a copy of the current environment state dictionary."""
        return dict(self._state)

    def perceive(self) -> Tuple[str, str]:
        """Provide current percept to the agent.

        Returns:
            A tuple of (current_location, current_location_status).
        """
        return self._agent_location, self._state[self._agent_location]

    def execute_action(self, action: str) -> None:
        """Transition environment state based on agent's action.

        Args:
            action: Actuator command ('Suck', 'Right', 'Left', or 'NoOp').

        Raises:
            ValueError: If an unrecognized action is passed.
        """
        if action == ACTION_SUCK:
            self._state[self._agent_location] = CLEAN
        elif action == ACTION_RIGHT:
            self._agent_location = LOCATION_B
        elif action == ACTION_LEFT:
            self._agent_location = LOCATION_A
        elif action == ACTION_NOOP:
            pass
        else:
            raise ValueError(f"Unrecognized action '{action}'.")

    def step(self, step_number: int, agent: Optional[SimpleReflexVacuumAgent] = None) -> StepRecord:
        """Execute a single discrete time step simulation.

        1. Percept is captured from the environment.
        2. Reflex agent determines action using condition-action rules.
        3. Environment state transitions based on the action.
        4. Step record is captured and returned.

        Args:
            step_number: Discrete time step index.
            agent: Agent instance (uses SimpleReflexVacuumAgent if None).

        Returns:
            StepRecord containing location, percept, action, and resulting state.
        """
        location, status = self.perceive()
        action = (
            agent.reflex_rule(location, status)
            if agent is not None
            else SimpleReflexVacuumAgent.reflex_rule(location, status)
        )
        self.execute_action(action)

        return StepRecord(
            step_number=step_number,
            location=location,
            status=status,
            action=action,
            environment_state_after=self.state,
        )

    def simulate(
        self,
        steps: int = 5,
        agent: Optional[SimpleReflexVacuumAgent] = None,
    ) -> List[StepRecord]:
        """Run the simulation over a discrete sequence of steps.

        Args:
            steps: Number of discrete time steps to execute (default: 5).
            agent: Agent instance.

        Returns:
            List of StepRecord objects for the entire run.

        Raises:
            ValueError: If steps < 1.
        """
        if steps < 1:
            raise ValueError(f"Steps must be at least 1, received {steps}.")

        history: List[StepRecord] = []
        for i in range(1, steps + 1):
            record = self.step(step_number=i, agent=agent)
            history.append(record)
        return history

    def is_all_clean(self) -> bool:
        """Check if all squares in the environment are clean.

        Returns:
            True if all locations are Clean, False otherwise.
        """
        return all(status == CLEAN for status in self._state.values())


# =====================================================================
# Optional Challenge: 3x3 Grid Vacuum Environment Extension
# =====================================================================

GridCoordinate = Tuple[int, int]


class GridVacuumEnvironment:
    """A 3x3 grid environment with randomized or explicit dirt placement.

    Implements the optional challenge to demonstrate agent-environment
    interaction scaling from 1D two-square to 2D grid coordinates.
    """

    GRID_SIZE = 3

    def __init__(
        self,
        agent_pos: GridCoordinate = (0, 0),
        dirt_probability: float = 0.4,
        seed: Optional[int] = None,
        custom_grid: Optional[Dict[GridCoordinate, str]] = None,
    ) -> None:
        """Initialize the 3x3 grid environment.

        Args:
            agent_pos: Initial (row, col) position of the agent (0-indexed).
            dirt_probability: Probability of a square being dirty when randomized.
            seed: Optional random seed for reproducible dirt placement.
            custom_grid: Optional explicit mapping of (row, col) to Clean/Dirty.
        """
        self._validate_coordinate(agent_pos)
        self.agent_pos = agent_pos

        if custom_grid is not None:
            self.grid: Dict[GridCoordinate, str] = {}
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    coord = (r, c)
                    val = custom_grid.get(coord, CLEAN)
                    self.grid[coord] = CLEAN if val == CLEAN else DIRTY
        else:
            rng = random.Random(seed)
            self.grid = {
                (r, c): (DIRTY if rng.random() < dirt_probability else CLEAN)
                for r in range(self.GRID_SIZE)
                for c in range(self.GRID_SIZE)
            }

    @classmethod
    def _validate_coordinate(cls, coord: GridCoordinate) -> None:
        """Ensure coordinate is within 3x3 grid bounds."""
        r, c = coord
        if not (0 <= r < cls.GRID_SIZE and 0 <= c < cls.GRID_SIZE):
            raise ValueError(f"Coordinate {coord} is out of 3x3 bounds.")

    def perceive(self) -> Tuple[GridCoordinate, str]:
        """Return current agent position and cleanliness status."""
        return self.agent_pos, self.grid[self.agent_pos]

    def execute_action(self, action: str) -> None:
        """Apply action on the grid environment."""
        r, c = self.agent_pos
        if action == ACTION_SUCK:
            self.grid[self.agent_pos] = CLEAN
        elif action == ACTION_RIGHT and c + 1 < self.GRID_SIZE:
            self.agent_pos = (r, c + 1)
        elif action == ACTION_LEFT and c - 1 >= 0:
            self.agent_pos = (r, c - 1)
        elif action == ACTION_DOWN and r + 1 < self.GRID_SIZE:
            self.agent_pos = (r + 1, c)
        elif action == ACTION_UP and r - 1 >= 0:
            self.agent_pos = (r - 1, c)
        # Any move into a wall acts as a NoOp in physical contact.

    def step(self, step_number: int) -> Tuple[int, GridCoordinate, str, str]:
        """Execute one step with simple reflex rules for 2D grid sweep.

        Reflex logic for 2D:
        - If dirty -> Suck
        - Else move following a fixed reflex sweep pattern (Right on even rows, Left on odd, Down at ends).
        """
        pos, status = self.perceive()
        r, c = pos

        if status == DIRTY:
            action = ACTION_SUCK
        else:
            # Deterministic reflex sweep pattern
            if r % 2 == 0:
                if c < self.GRID_SIZE - 1:
                    action = ACTION_RIGHT
                elif r < self.GRID_SIZE - 1:
                    action = ACTION_DOWN
                else:
                    action = ACTION_LEFT
            else:
                if c > 0:
                    action = ACTION_LEFT
                elif r < self.GRID_SIZE - 1:
                    action = ACTION_DOWN
                else:
                    action = ACTION_RIGHT

        self.execute_action(action)
        return step_number, pos, status, action

    def simulate(self, steps: int = 10) -> List[Tuple[int, GridCoordinate, str, str]]:
        """Run grid simulation for specified steps."""
        return [self.step(i) for i in range(1, steps + 1)]
