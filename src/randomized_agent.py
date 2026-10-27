"""Randomized Reflex Agent for 1D Grid Environment.

This module models a Randomized Reflex Agent operating in a one-dimensional grid
of rooms (default: 5 rooms), addressing partially observable environments and
preventing infinite loops in boundary rooms.

Core Concepts:
- Randomized Reflex Agent: When situated in a clean room, selects navigation
  actions (Left / Right) stochastically via Python's `random` module, breaking
  deterministic oscillation and boundary deadlocks.
- Condition-Action Rules:
    - If status is Dirty (1) -> Action: Clean
    - If status is Clean (0) -> Action: Random choice between Left and Right
- Partial Observability: The agent does NOT have access to the global grid array
  state, only receiving the local cleanliness percept of its current room.
- Termination: Simulation runs until all rooms are clean, with an optional
  safety step ceiling to prevent infinite loops.

Complexity:
- Time Complexity: O(N) on average for grid of size N (random walk with absorbing boundaries).
- Space Complexity: O(N) to store room states in a list array.
"""

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence

# Domain constants
ROOM_CLEAN: int = 0
ROOM_DIRTY: int = 1

ACTION_CLEAN: str = "Clean"
ACTION_LEFT: str = "Left"
ACTION_RIGHT: str = "Right"
ACTION_NOOP: str = "NoOp"

VALID_ACTIONS = (ACTION_CLEAN, ACTION_LEFT, ACTION_RIGHT, ACTION_NOOP)


@dataclass(frozen=True)
class AgentStepRecord:
    """Immutable record of an agent-environment interaction step.

    Attributes:
        step_number: Discrete time step index (1-based).
        room_index: Current 0-based room index where the agent is situated.
        percept: Perceived cleanliness status of current room (0=Clean, 1=Dirty).
        action: Action chosen by the agent ('Clean', 'Left', or 'Right').
        grid_state_after: Snapshot of the grid list array after executing the action.
        is_boundary_bump: True if movement was blocked by a wall boundary.
    """

    step_number: int
    room_index: int
    percept: int
    action: str
    grid_state_after: List[int]
    is_boundary_bump: bool = False

    def format_trace(self) -> str:
        """Produce formatted log trace for laboratory output.

        Returns:
            Human-readable formatted step trace string.
        """
        status_label = "Dirty" if self.percept == ROOM_DIRTY else "Clean"
        bump_tag = " (Boundary Bump)" if self.is_boundary_bump else ""
        return (
            f"Step {self.step_number:2d}: Room [{self.room_index}] is {status_label} ({self.percept}) "
            f"-> Action: {self.action:<5}{bump_tag} | Grid: {self.grid_state_after}"
        )


@dataclass
class SimulationResult:
    """Summary of a completed simulation run.

    Attributes:
        initial_grid: Copy of initial grid state.
        final_grid: Copy of final grid state.
        start_index: Starting room index of agent.
        final_index: Room index where agent terminated.
        total_steps: Total interaction steps executed.
        total_moves: Total navigation moves (Left/Right) attempted.
        total_cleans: Total cleaning actions performed.
        is_all_clean: Whether all rooms in the grid are clean (0).
        stopped_by_limit: Whether execution halted due to maximum step limit.
        history: Complete sequence of interaction step records.
    """

    initial_grid: List[int]
    final_grid: List[int]
    start_index: int
    final_index: int
    total_steps: int
    total_moves: int
    total_cleans: int
    is_all_clean: bool
    stopped_by_limit: bool
    history: List[AgentStepRecord]

    def summary(self) -> str:
        """Generate human-readable summary of simulation results.

        Returns:
            Formatted multiline summary string.
        """
        status = "SUCCESS" if self.is_all_clean else "HALTED (Max Steps Exceeded)"
        return (
            f"Simulation Status: {status}\n"
            f"All rooms clean in {self.total_steps} steps.\n"
            f"  • Total Moves Executed:    {self.total_moves}\n"
            f"  • Total Clean Operations:  {self.total_cleans}\n"
            f"  • Initial State:           {self.initial_grid} (Start Index: {self.start_index})\n"
            f"  • Final State:             {self.final_grid} (Final Index: {self.final_index})"
        )


class RandomizedReflexAgent:
    """Randomized Reflex Agent for 1D room environments.

    A reflex agent with stochastic action selection. The agent has no access
    to the global environment array and retains zero historical memory.
    It maps current local percept to action:
    - Dirty (1) -> Clean
    - Clean (0) -> Random choice: Left or Right
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the randomized reflex agent.

        Args:
            seed: Optional random seed for reproducible stochastic choices.
        """
        self._rng: random.Random = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Reset or configure the random seed for reproducible execution.

        Args:
            seed: Integer seed or None for non-deterministic execution.
        """
        self._rng.seed(seed)

    def decide_action(self, percept: int) -> str:
        """Select action strictly based on current local room percept.

        Rule 1: If room is Dirty (1), return 'Clean'.
        Rule 2: If room is Clean (0), choose randomly between 'Left' and 'Right'.

        Args:
            percept: Cleanliness status of the current room (0 or 1).

        Returns:
            Chosen action ('Clean', 'Left', or 'Right').

        Raises:
            ValueError: If percept is not 0 or 1.
        """
        if percept == ROOM_DIRTY:
            return ACTION_CLEAN
        elif percept == ROOM_CLEAN:
            return self._rng.choice([ACTION_LEFT, ACTION_RIGHT])
        else:
            raise ValueError(
                f"Invalid room percept '{percept}'. Expected {ROOM_CLEAN} (Clean) or {ROOM_DIRTY} (Dirty)."
            )


class OneDimensionalGridEnvironment:
    """One-dimensional grid simulation environment of discrete rooms.

    Encapsulates the global state array of rooms and simulates agent physics,
    actuator transitions, and boundary constraints. The agent perceives only its
    immediate room via `get_percept()`.
    """

    def __init__(
        self,
        initial_state: Sequence[int] = (1, 0, 0, 1, 1),
        start_index: int = 2,
    ) -> None:
        """Initialize the 1D grid environment.

        Args:
            initial_state: Sequence representing initial room states (1=Dirty, 0=Clean).
            start_index: Initial position index of agent (0-indexed).

        Raises:
            ValueError: If initial_state is empty or contains non-binary values,
                        or if start_index is out of grid boundaries.
        """
        if not initial_state:
            raise ValueError("Grid state cannot be empty. At least one room is required.")

        for idx, val in enumerate(initial_state):
            if val not in (ROOM_CLEAN, ROOM_DIRTY):
                raise ValueError(
                    f"Invalid room state at index {idx}: {val}. Rooms must be {ROOM_CLEAN} (Clean) or {ROOM_DIRTY} (Dirty)."
                )

        if not (0 <= start_index < len(initial_state)):
            raise ValueError(
                f"Starting index {start_index} is out of bounds for grid of size {len(initial_state)}. "
                f"Must be between 0 and {len(initial_state) - 1}."
            )

        self._grid: List[int] = list(initial_state)
        self._agent_index: int = start_index
        self._step_count: int = 0
        self._move_count: int = 0
        self._clean_count: int = 0
        self._history: List[AgentStepRecord] = []

    @property
    def grid(self) -> List[int]:
        """Return a defensive copy of the current grid state."""
        return list(self._grid)

    @property
    def agent_index(self) -> int:
        """Return current agent room index."""
        return self._agent_index

    @property
    def total_rooms(self) -> int:
        """Return total number of rooms in the grid."""
        return len(self._grid)

    @property
    def step_count(self) -> int:
        """Return total interaction steps executed so far."""
        return self._step_count

    @property
    def move_count(self) -> int:
        """Return total navigation moves executed."""
        return self._move_count

    @property
    def clean_count(self) -> int:
        """Return total cleaning actions executed."""
        return self._clean_count

    @property
    def history(self) -> List[AgentStepRecord]:
        """Return history of all step records."""
        return list(self._history)

    def is_all_clean(self) -> bool:
        """Determine whether all rooms in the environment are clean.

        Returns:
            True if all rooms equal 0 (Clean), False otherwise.
        """
        return all(room == ROOM_CLEAN for room in self._grid)

    def get_percept(self) -> int:
        """Expose local percept to the agent.

        Enforces partial observability constraint: agent only observes the
        cleanliness of its current room, not the global state array.

        Returns:
            0 (Clean) or 1 (Dirty) for the agent's current room.
        """
        return self._grid[self._agent_index]

    def execute_step(self, agent: RandomizedReflexAgent) -> AgentStepRecord:
        """Execute a single discrete interaction step with the agent.

        Workflow:
        1. Environment queries local percept for current location.
        2. Agent decides action based strictly on local percept.
        3. Environment applies physics (updates room state or updates position).
        4. Step record is generated and appended to history.

        Args:
            agent: The randomized reflex agent instance.

        Returns:
            AgentStepRecord detailing the step outcome.
        """
        percept = self.get_percept()
        current_room = self._agent_index
        action = agent.decide_action(percept)

        self._step_count += 1
        is_boundary_bump = False

        if action == ACTION_CLEAN:
            self._grid[self._agent_index] = ROOM_CLEAN
            self._clean_count += 1
        elif action == ACTION_LEFT:
            self._move_count += 1
            if self._agent_index == 0:
                is_boundary_bump = True
            else:
                self._agent_index -= 1
        elif action == ACTION_RIGHT:
            self._move_count += 1
            if self._agent_index == len(self._grid) - 1:
                is_boundary_bump = True
            else:
                self._agent_index += 1
        else:
            raise ValueError(f"Unrecognized action '{action}'.")

        record = AgentStepRecord(
            step_number=self._step_count,
            room_index=current_room,
            percept=percept,
            action=action,
            grid_state_after=self.grid,
            is_boundary_bump=is_boundary_bump,
        )
        self._history.append(record)
        return record

    def simulate(
        self,
        agent: RandomizedReflexAgent,
        max_steps: Optional[int] = 1000,
    ) -> SimulationResult:
        """Run the simulation loop until all rooms are clean or max_steps reached.

        Args:
            agent: The randomized reflex agent.
            max_steps: Optional maximum step threshold to prevent infinite loops.
                       Defaults to 1000. Set to None for unbounded execution.

        Returns:
            SimulationResult summarizing steps, moves, cleans, and final state.
        """
        initial_grid_snapshot = self.grid
        start_idx = self._agent_index
        stopped_by_limit = False

        while not self.is_all_clean():
            if max_steps is not None and self._step_count >= max_steps:
                stopped_by_limit = True
                break
            self.execute_step(agent)

        return SimulationResult(
            initial_grid=initial_grid_snapshot,
            final_grid=self.grid,
            start_index=start_idx,
            final_index=self._agent_index,
            total_steps=self._step_count,
            total_moves=self._move_count,
            total_cleans=self._clean_count,
            is_all_clean=self.is_all_clean(),
            stopped_by_limit=stopped_by_limit,
            history=self.history,
        )


def run_randomized_reflex_simulation(
    grid: Sequence[int] = (1, 0, 0, 1, 1),
    start_index: int = 2,
    seed: Optional[int] = None,
    max_steps: Optional[int] = 1000,
) -> SimulationResult:
    """Convenience function to run a 1D randomized reflex simulation in one call.

    Args:
        grid: Sequence of binary room cleanliness states (1=Dirty, 0=Clean).
        start_index: Starting room index of agent (0 to len(grid)-1).
        seed: Random seed for deterministic reproducibility.
        max_steps: Maximum step limit to prevent infinite loops.

    Returns:
        SimulationResult instance with complete telemetry.
    """
    env = OneDimensionalGridEnvironment(initial_state=grid, start_index=start_index)
    agent = RandomizedReflexAgent(seed=seed)
    return env.simulate(agent=agent, max_steps=max_steps)
