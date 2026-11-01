"""Multi-Agent Environment Simulation for 4x4 Grid Matrix (Question 6).

This module implements a turn-based multi-agent simulation in a shared 2D grid
environment. Two autonomous agents operate concurrently on the grid:
  1. Agent A ('Builder'): Operates under a constructive percept-action rule to
     place physical blocks (setting cell value to 1).
  2. Agent B ('Destroyer'): Operates under a deconstructive percept-action rule to
     remove physical blocks (setting cell value to 0).

Core AI Concepts:
- Multi-Agent Systems (MAS): Systems where multiple autonomous entities interact,
  sharing an environment, resources, and communication/perception channels.
- Competitive vs. Cooperative Environments:
  * Competitive (Zero-Sum): Agent A's utility is directly opposed to Agent B's
    utility (Builder maximizes block count; Destroyer minimizes block count).
  * Shared State Mutation: Agents do not maintain private environments; they
    hold shared references to a single environment object that mutates over time.
- Environment Taxonomies (Russell & Norvig):
  * Sequential vs. Episodic: Actions taken now dictate future percepts and choices.
  * Dynamic vs. Static: The world changes between an agent's individual turns due
    to the autonomous actions of the other agent.
  * Discrete vs. Continuous: Grid locations and turn iterations are discrete.
- Percept-Action Function:
  * f_Builder: Percept -> {PlaceBlock, Move, NoOp}
  * f_Destroyer: Percept -> {RemoveBlock, Move, NoOp}
- Collision Avoidance (Optional Challenge):
  Agents perceive the adversary's coordinate and proactively filter occupied
  cells from candidate movement transitions, preventing physical collisions.

Complexity:
- Time Complexity: O(S) where S is the number of simulation steps.
- Space Complexity: O(R * C) where R, C are grid dimensions (O(16) = O(1) for 4x4).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Grid Cell States
CELL_EMPTY: int = 0
CELL_BLOCK: int = 1
VALID_CELL_VALUES: Tuple[int, ...] = (CELL_EMPTY, CELL_BLOCK)

# Agent Roles
ROLE_BUILDER: str = "Builder"
ROLE_DESTROYER: str = "Destroyer"
VALID_ROLES: Tuple[str, ...] = (ROLE_BUILDER, ROLE_DESTROYER)

# Action Identifiers
ACTION_BUILD: str = "PlaceBlock"
ACTION_DESTROY: str = "RemoveBlock"
ACTION_MOVE_UP: str = "MoveUp"
ACTION_MOVE_DOWN: str = "MoveDown"
ACTION_MOVE_LEFT: str = "MoveLeft"
ACTION_MOVE_RIGHT: str = "MoveRight"
ACTION_NOOP: str = "NoOp"

VALID_ACTIONS: Tuple[str, ...] = (
    ACTION_BUILD,
    ACTION_DESTROY,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_NOOP,
)

# Orthogonal Movement Vectors (delta_row, delta_col)
ORTHOGONAL_DELTAS: Dict[str, Tuple[int, int]] = {
    ACTION_MOVE_UP: (-1, 0),
    ACTION_MOVE_DOWN: (1, 0),
    ACTION_MOVE_LEFT: (0, -1),
    ACTION_MOVE_RIGHT: (0, 1),
}

DELTA_TO_ACTION: Dict[Tuple[int, int], str] = {
    (-1, 0): ACTION_MOVE_UP,
    (1, 0): ACTION_MOVE_DOWN,
    (0, -1): ACTION_MOVE_LEFT,
    (0, 1): ACTION_MOVE_RIGHT,
}

# Execution Modes
TURN_MODE_DISCRETE: str = "discrete"
TURN_MODE_COMPOUND: str = "compound"
VALID_TURN_MODES: Tuple[str, ...] = (TURN_MODE_DISCRETE, TURN_MODE_COMPOUND)

# Movement Strategies
STRATEGY_RANDOM: str = "random"
STRATEGY_GREEDY: str = "greedy"
VALID_STRATEGIES: Tuple[str, ...] = (STRATEGY_RANDOM, STRATEGY_GREEDY)


def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    """Compute the L1 Manhattan distance between two 2D coordinates.

    Args:
        coord1: First coordinate (row, col).
        coord2: Second coordinate (row, col).

    Returns:
        Integer Manhattan distance |r1 - r2| + |c1 - c2|.
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


@dataclass(frozen=True)
class MultiAgentPercept:
    """Percept received by an agent at the beginning of its turn.

    Attributes:
        agent_name: Identifier of the perceiving agent.
        position: Current 2D grid coordinates (row, col).
        cell_value: Physical state of the current cell (0 = Empty, 1 = Block).
        available_moves: Orthogonal adjacent coordinates within grid boundaries.
        other_agent_position: Optional coordinates of the peer agent (for collision avoidance).
        grid_view: Optional defensive copy of the full grid matrix (full observability).
    """

    agent_name: str
    position: Tuple[int, int]
    cell_value: int
    available_moves: Tuple[Tuple[int, int], ...]
    other_agent_position: Optional[Tuple[int, int]] = None
    grid_view: Optional[Tuple[Tuple[int, ...], ...]] = None


@dataclass(frozen=True)
class AgentTurnRecord:
    """Log record documenting an agent's individual turn execution.

    Attributes:
        step_number: Discrete simulation step number (1-based).
        turn_index: Turn order within step (1 for Agent A, 2 for Agent B).
        agent_name: Name of the agent taking the turn.
        agent_role: Role of the agent ('Builder' or 'Destroyer').
        pos_before: Coordinates prior to turn action.
        cell_before: Cell value (0 or 1) prior to turn action.
        action: Primary action performed ('PlaceBlock', 'RemoveBlock', 'MoveUp', etc.).
        move_action: Optional secondary move action executed during compound steps.
        pos_after: Coordinates after turn completion.
        cell_after: Cell value at agent's position after turn completion.
        collision_avoided: True if a candidate move was pruned to prevent collision.
        grid_snapshot: 2D list snapshot of the entire grid after this turn.
    """

    step_number: int
    turn_index: int
    agent_name: str
    agent_role: str
    pos_before: Tuple[int, int]
    cell_before: int
    action: str
    move_action: Optional[str]
    pos_after: Tuple[int, int]
    cell_after: int
    collision_avoided: bool
    grid_snapshot: List[List[int]]

    def format_trace(self) -> str:
        """Format the turn record into a concise, readable log line.

        Returns:
            Formatted trace string.
        """
        action_desc = self.action
        if self.move_action:
            action_desc = f"{self.action} -> {self.move_action}"
        avoid_tag = " [Collision Avoided]" if self.collision_avoided else ""
        return (
            f"Step {self.step_number:2d}.{self.turn_index}: {self.agent_name:<10} "
            f"({self.agent_role:<9}) | Pos {self.pos_before} -> {self.pos_after} | "
            f"Cell {self.cell_before} -> {self.cell_after} | "
            f"Action: {action_desc:<22}{avoid_tag}"
        )


@dataclass
class MultiAgentSimulationResult:
    """Container holding the complete outcome of a multi-agent simulation run.

    Attributes:
        initial_grid: Numpy array copy of the grid at step 0.
        final_grid: Numpy array copy of the grid at step S.
        agent_a_name: Name of Agent A.
        agent_a_role: Role of Agent A.
        agent_a_start: Initial coordinate of Agent A.
        agent_a_final: Final coordinate of Agent A.
        agent_b_name: Name of Agent B.
        agent_b_role: Role of Agent B.
        agent_b_start: Initial coordinate of Agent B.
        agent_b_final: Final coordinate of Agent B.
        total_steps: Number of rounds completed.
        blocks_placed: Total count of blocks placed by Builder.
        blocks_destroyed: Total count of blocks destroyed by Destroyer.
        final_block_count: Total blocks remaining in grid at termination.
        collision_avoidance_active: Whether collision avoidance was enabled.
        collisions_avoided_count: Number of times agents pruned candidate moves to avoid collision.
        collisions_occurred: Number of times agents occupied the same square (0 when avoidance active).
        step_mode: Execution mode ('compound' or 'discrete').
        history: Chronological list of turn records.
    """

    initial_grid: np.ndarray
    final_grid: np.ndarray
    agent_a_name: str
    agent_a_role: str
    agent_a_start: Tuple[int, int]
    agent_a_final: Tuple[int, int]
    agent_b_name: str
    agent_b_role: str
    agent_b_start: Tuple[int, int]
    agent_b_final: Tuple[int, int]
    total_steps: int
    blocks_placed: int
    blocks_destroyed: int
    final_block_count: int
    collision_avoidance_active: bool
    collisions_avoided_count: int
    collisions_occurred: int
    step_mode: str
    history: List[AgentTurnRecord] = field(default_factory=list)

    def display_matrix(self) -> str:
        """Return the final grid as a standard NumPy matrix string.

        Conforms strictly to the sample output format:
        Matrix displaying 1s (blocks) and 0s (empty space).

        Returns:
            String representation of the NumPy 2D array.
        """
        return str(self.final_grid)

    def display_board(self, show_agents: bool = True) -> str:
        """Render an ASCII visual representation of the grid board.

        Args:
            show_agents: Whether to overlay agent position tags ([A], [B], [AB]).

        Returns:
            Multiline ASCII grid representation.
        """
        rows, cols = self.final_grid.shape
        lines: List[str] = []
        col_header = "     " + "   ".join(f"C{c}" for c in range(cols))
        lines.append(col_header)
        lines.append("   +" + "---+" * cols)

        for r in range(rows):
            row_str = f"R{r} |"
            for c in range(cols):
                cell_val = int(self.final_grid[r, c])
                is_a = show_agents and self.agent_a_final == (r, c)
                is_b = show_agents and self.agent_b_final == (r, c)

                if is_a and is_b:
                    cell_rep = "AB!"
                elif is_a:
                    cell_rep = " A " if cell_val == 0 else " A*"
                elif is_b:
                    cell_rep = " B " if cell_val == 0 else " B*"
                elif cell_val == CELL_BLOCK:
                    cell_rep = " 1 "
                else:
                    cell_rep = " 0 "
                row_str += cell_rep + "|"
            lines.append(row_str)
            lines.append("   +" + "---+" * cols)

        legend = (
            "Legend: [ 0 ] Empty | [ 1 ] Block | [ A ] Agent A (Builder) | "
            "[ B ] Agent B (Destroyer) | [*] Agent on Block"
        )
        lines.append(legend)
        return "\n".join(lines)

    def summary(self) -> str:
        """Generate a structured summary report of the simulation.

        Returns:
            Multi-line formatted summary.
        """
        net_blocks = self.blocks_placed - self.blocks_destroyed
        efficiency = (
            f"Builder placed {self.blocks_placed} blocks | "
            f"Destroyer removed {self.blocks_destroyed} blocks | Net: {net_blocks:+d}"
        )
        return (
            f"======================================================================\n"
            f"               MULTI-AGENT ENVIRONMENT SIMULATION SUMMARY             \n"
            f"======================================================================\n"
            f"Execution Mode:             {self.step_mode.capitalize()} Turns\n"
            f"Total Rounds (Steps):       {self.total_steps}\n"
            f"Agent A:                    {self.agent_a_name} ({self.agent_a_role}) | "
            f"Start: {self.agent_a_start} -> Final: {self.agent_a_final}\n"
            f"Agent B:                    {self.agent_b_name} ({self.agent_b_role}) | "
            f"Start: {self.agent_b_start} -> Final: {self.agent_b_final}\n"
            f"Performance Metrics:        {efficiency}\n"
            f"Final Block Count in Grid:  {self.final_block_count} / {self.final_grid.size}\n"
            f"Collision Avoidance:        {'Enabled' if self.collision_avoidance_active else 'Disabled'}\n"
            f"Collisions Avoided:         {self.collisions_avoided_count}\n"
            f"Collisions Occurred:        {self.collisions_occurred}\n"
            f"----------------------------------------------------------------------\n"
            f"Final Grid State Matrix (NumPy Array):\n"
            f"{self.display_matrix()}\n"
            f"======================================================================"
        )


class BaseAgent(ABC):
    """Abstract base class for autonomous agents in a multi-agent grid world.

    Encapsulates core agent properties: name, role, internal coordinates,
    and pseudo-random number generator for reproducible action selection.
    """

    def __init__(
        self,
        name: str,
        role: str,
        initial_position: Tuple[int, int],
        seed: Optional[int] = None,
        strategy: str = STRATEGY_RANDOM,
    ) -> None:
        """Initialize base agent attributes.

        Args:
            name: Human-readable agent identifier.
            role: Functional role ('Builder' or 'Destroyer').
            initial_position: Starting grid coordinates (row, col).
            seed: Optional integer seed for reproducible decision-making.
            strategy: Movement selection strategy ('random' or 'greedy').

        Raises:
            ValueError: If role or strategy is invalid, or coordinates are malformed.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of {VALID_ROLES}.")
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Expected one of {VALID_STRATEGIES}."
            )
        self._validate_coord(initial_position, "initial_position")

        self.name: str = name
        self.role: str = role
        self.position: Tuple[int, int] = initial_position
        self.strategy: str = strategy
        self.rng: random.Random = random.Random(seed)
        self.actions_performed: int = 0
        self.moves_executed: int = 0

    @staticmethod
    def _validate_coord(coord: Tuple[int, int], name: str) -> None:
        """Verify that a coordinate is a 2-tuple of non-negative integers."""
        if not (
            isinstance(coord, (tuple, list))
            and len(coord) == 2
            and isinstance(coord[0], int)
            and isinstance(coord[1], int)
        ):
            raise TypeError(
                f"{name} must be a 2-element tuple of integers, got: {coord!r}"
            )

    def set_position(self, new_pos: Tuple[int, int]) -> None:
        """Update internal agent position after an environmental transition.

        Args:
            new_pos: Target (row, col) coordinates.
        """
        self._validate_coord(new_pos, "new_pos")
        self.position = new_pos
        self.moves_executed += 1

    def filter_safe_moves(
        self,
        available_moves: Sequence[Tuple[int, int]],
        other_agent_position: Optional[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], bool]:
        """Prune transitions that would cause a physical collision (Optional Challenge).

        Args:
            available_moves: Sequence of candidate orthogonal grid moves.
            other_agent_position: Coordinate currently occupied by the peer agent.

        Returns:
            Tuple of (safe_moves_list, collision_avoided_flag).
        """
        if other_agent_position is None:
            return list(available_moves), False

        safe_moves = [m for m in available_moves if m != other_agent_position]
        collision_avoided = len(safe_moves) < len(available_moves)
        return safe_moves, collision_avoided

    def choose_move(
        self,
        candidate_moves: Sequence[Tuple[int, int]],
        grid_view: Optional[Tuple[Tuple[int, ...], ...]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Select next coordinate from candidate moves according to agent strategy.

        Args:
            candidate_moves: Sequence of admissible target coordinates.
            grid_view: Optional full grid snapshot for greedy heuristic evaluation.

        Returns:
            Selected target coordinate, or None if no candidate moves exist.
        """
        if not candidate_moves:
            return None

        if self.strategy == STRATEGY_GREEDY and grid_view is not None:
            target_val = CELL_EMPTY if self.role == ROLE_BUILDER else CELL_BLOCK
            # Find all target cells matching agent's functional objective
            target_cells: List[Tuple[int, int]] = []
            for r_idx, row in enumerate(grid_view):
                for c_idx, val in enumerate(row):
                    if val == target_val and (r_idx, c_idx) != self.position:
                        target_cells.append((r_idx, c_idx))

            if target_cells:
                # Rank candidate moves by minimal Manhattan distance to closest target cell
                def score_move(move: Tuple[int, int]) -> int:
                    return min(manhattan_distance(move, t) for t in target_cells)

                sorted_moves = sorted(candidate_moves, key=score_move)
                best_score = score_move(sorted_moves[0])
                top_tier = [m for m in sorted_moves if score_move(m) == best_score]
                return self.rng.choice(top_tier)

        # Default or fallback: random choice from candidates
        return self.rng.choice(list(candidate_moves))

    @abstractmethod
    def decide_discrete_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[Tuple[int, int]], bool]:
        """Decide next action in discrete turn mode.

        An agent either executes its primary state modification (Build/Destroy)
        or moves if the current cell does not require modification.

        Args:
            percept: Environmental percept at start of turn.
            avoid_collisions: Whether collision avoidance filtering is enabled.

        Returns:
            Tuple of (action_name, target_move_pos, collision_avoided_flag).
        """

    @abstractmethod
    def decide_compound_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[str], Optional[Tuple[int, int]], bool]:
        """Decide both cell action and subsequent move in compound turn mode.

        Args:
            percept: Environmental percept at start of turn.
            avoid_collisions: Whether collision avoidance filtering is enabled.

        Returns:
            Tuple of (cell_action, move_action_name, target_pos, collision_avoided_flag).
        """


class BuilderAgent(BaseAgent):
    """Agent A: The 'Builder' Agent.

    Percept-Action Rule:
    - If current cell is EMPTY (0): Place a block (set cell to 1).
    - If current cell is BLOCK (1): Move to an adjacent cell to find empty space.
    """

    def __init__(
        self,
        name: str = "Agent_A",
        initial_position: Tuple[int, int] = (0, 0),
        seed: Optional[int] = None,
        strategy: str = STRATEGY_RANDOM,
    ) -> None:
        """Initialize Builder agent."""
        super().__init__(
            name=name,
            role=ROLE_BUILDER,
            initial_position=initial_position,
            seed=seed,
            strategy=strategy,
        )

    def decide_discrete_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[Tuple[int, int]], bool]:
        """Evaluate percept under Builder percept-action rule (Discrete Mode).

        If cell is 0: Action = PlaceBlock.
        If cell is 1: Action = Move to adjacent neighbor.
        """
        # Rule 1: Place block if cell is empty
        if percept.cell_value == CELL_EMPTY:
            self.actions_performed += 1
            return ACTION_BUILD, None, False

        # Rule 2: Cell already contains a block; select move
        safe_moves, avoided = self.filter_safe_moves(
            percept.available_moves,
            percept.other_agent_position if avoid_collisions else None,
        )
        chosen_move = self.choose_move(safe_moves, percept.grid_view)
        if chosen_move is None:
            return ACTION_NOOP, None, avoided

        delta = (
            chosen_move[0] - percept.position[0],
            chosen_move[1] - percept.position[1],
        )
        move_action = DELTA_TO_ACTION.get(delta, ACTION_NOOP)
        return move_action, chosen_move, avoided

    def decide_compound_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[str], Optional[Tuple[int, int]], bool]:
        """Evaluate percept under Builder percept-action rule (Compound Mode).

        Step 1: If cell is 0, PlaceBlock; otherwise NoOp (already built).
        Step 2: Transition to adjacent neighbor.
        """
        # Step 1: Physical modification
        if percept.cell_value == CELL_EMPTY:
            cell_action = ACTION_BUILD
            self.actions_performed += 1
        else:
            cell_action = ACTION_NOOP

        # Step 2: Spatial movement
        safe_moves, avoided = self.filter_safe_moves(
            percept.available_moves,
            percept.other_agent_position if avoid_collisions else None,
        )
        chosen_move = self.choose_move(safe_moves, percept.grid_view)
        if chosen_move is None:
            return cell_action, None, None, avoided

        delta = (
            chosen_move[0] - percept.position[0],
            chosen_move[1] - percept.position[1],
        )
        move_action = DELTA_TO_ACTION.get(delta, ACTION_NOOP)
        return cell_action, move_action, chosen_move, avoided


class DestroyerAgent(BaseAgent):
    """Agent B: The 'Destroyer' Agent.

    Percept-Action Rule:
    - If current cell is BLOCK (1): Remove the block (set cell to 0).
    - If current cell is EMPTY (0): Move to an adjacent cell to seek blocks.
    """

    def __init__(
        self,
        name: str = "Agent_B",
        initial_position: Tuple[int, int] = (3, 3),
        seed: Optional[int] = None,
        strategy: str = STRATEGY_RANDOM,
    ) -> None:
        """Initialize Destroyer agent."""
        super().__init__(
            name=name,
            role=ROLE_DESTROYER,
            initial_position=initial_position,
            seed=seed,
            strategy=strategy,
        )

    def decide_discrete_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[Tuple[int, int]], bool]:
        """Evaluate percept under Destroyer percept-action rule (Discrete Mode).

        If cell is 1: Action = RemoveBlock.
        If cell is 0: Action = Move to adjacent neighbor.
        """
        # Rule 1: Remove block if cell contains one
        if percept.cell_value == CELL_BLOCK:
            self.actions_performed += 1
            return ACTION_DESTROY, None, False

        # Rule 2: Cell is empty; select move
        safe_moves, avoided = self.filter_safe_moves(
            percept.available_moves,
            percept.other_agent_position if avoid_collisions else None,
        )
        chosen_move = self.choose_move(safe_moves, percept.grid_view)
        if chosen_move is None:
            return ACTION_NOOP, None, avoided

        delta = (
            chosen_move[0] - percept.position[0],
            chosen_move[1] - percept.position[1],
        )
        move_action = DELTA_TO_ACTION.get(delta, ACTION_NOOP)
        return move_action, chosen_move, avoided

    def decide_compound_action(
        self, percept: MultiAgentPercept, avoid_collisions: bool = False
    ) -> Tuple[str, Optional[str], Optional[Tuple[int, int]], bool]:
        """Evaluate percept under Destroyer percept-action rule (Compound Mode).

        Step 1: If cell is 1, RemoveBlock; otherwise NoOp (already clean).
        Step 2: Transition to adjacent neighbor.
        """
        # Step 1: Physical modification
        if percept.cell_value == CELL_BLOCK:
            cell_action = ACTION_DESTROY
            self.actions_performed += 1
        else:
            cell_action = ACTION_NOOP

        # Step 2: Spatial movement
        safe_moves, avoided = self.filter_safe_moves(
            percept.available_moves,
            percept.other_agent_position if avoid_collisions else None,
        )
        chosen_move = self.choose_move(safe_moves, percept.grid_view)
        if chosen_move is None:
            return cell_action, None, None, avoided

        delta = (
            chosen_move[0] - percept.position[0],
            chosen_move[1] - percept.position[1],
        )
        move_action = DELTA_TO_ACTION.get(delta, ACTION_NOOP)
        return cell_action, move_action, chosen_move, avoided


class MultiAgentGridEnvironment:
    """Shared 2D Grid Environment object managed by both agents.

    Maintains the 2D grid matrix (NumPy array), validates boundaries,
    executes agent turns sequentially, tracks collisions, and logs history.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        agent_a: Optional[BaseAgent] = None,
        agent_b: Optional[BaseAgent] = None,
        initial_grid: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the shared environment.

        Args:
            rows: Number of grid rows (default: 4).
            cols: Number of grid columns (default: 4).
            agent_a: Reference to Agent A (Builder).
            agent_b: Reference to Agent B (Destroyer).
            initial_grid: Optional predefined NumPy 2D array of initial states.

        Raises:
            ValueError: If grid dimensions are non-positive or agents occupy invalid cells.
        """
        if rows < 1 or cols < 1:
            raise ValueError(f"Grid dimensions must be positive integers, got ({rows}, {cols}).")

        self.rows: int = rows
        self.cols: int = cols

        if initial_grid is not None:
            arr = np.array(initial_grid, dtype=int)
            if arr.shape != (rows, cols):
                raise ValueError(
                    f"initial_grid shape {arr.shape} does not match ({rows}, {cols})."
                )
            if not np.all(np.isin(arr, VALID_CELL_VALUES)):
                raise ValueError("Grid cells must only contain 0 (Empty) or 1 (Block).")
            self._grid: np.ndarray = arr.copy()
        else:
            self._grid = np.zeros((rows, cols), dtype=int)

        self.agent_a: BaseAgent = (
            agent_a if agent_a is not None else BuilderAgent(initial_position=(0, 0))
        )
        self.agent_b: BaseAgent = (
            agent_b
            if agent_b is not None
            else DestroyerAgent(initial_position=(rows - 1, cols - 1))
        )

        # Validate starting positions within grid boundaries
        self.validate_coordinate(self.agent_a.position, "Agent A starting position")
        self.validate_coordinate(self.agent_b.position, "Agent B starting position")

        self.step_count: int = 0
        self.blocks_placed_total: int = 0
        self.blocks_destroyed_total: int = 0
        self.collisions_avoided_total: int = 0
        self.collisions_occurred_total: int = 0
        self.history: List[AgentTurnRecord] = []

    def validate_coordinate(self, coord: Tuple[int, int], label: str = "Coordinate") -> None:
        """Ensure coordinate falls strictly within environment boundaries.

        Args:
            coord: (row, col) tuple.
            label: Descriptive label for error messages.

        Raises:
            IndexError: If coordinate falls outside grid bounds.
        """
        r, c = coord
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(
                f"{label} ({r}, {c}) is outside grid boundaries (0..{self.rows - 1}, 0..{self.cols - 1})."
            )

    @property
    def grid(self) -> np.ndarray:
        """Return a defensive copy of the current grid matrix."""
        return self._grid.copy()

    def get_cell(self, coord: Tuple[int, int]) -> int:
        """Get cell value at given coordinates."""
        self.validate_coordinate(coord)
        return int(self._grid[coord[0], coord[1]])

    def set_cell(self, coord: Tuple[int, int], value: int) -> None:
        """Set cell value at given coordinates."""
        self.validate_coordinate(coord)
        if value not in VALID_CELL_VALUES:
            raise ValueError(f"Invalid cell value {value}. Must be 0 or 1.")
        self._grid[coord[0], coord[1]] = value

    def get_orthogonal_neighbors(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid in-bounds orthogonal neighbors (von Neumann neighborhood).

        Args:
            coord: Source (row, col) coordinate.

        Returns:
            List of valid adjacent (row, col) coordinates.
        """
        neighbors: List[Tuple[int, int]] = []
        r, c = coord
        for dr, dc in ORTHOGONAL_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def generate_percept(
        self, agent: BaseAgent, include_peer: bool = False
    ) -> MultiAgentPercept:
        """Construct environmental percept for a specific agent.

        Args:
            agent: Agent receiving the percept.
            include_peer: Whether to reveal peer agent's coordinate (collision avoidance).

        Returns:
            Immutable MultiAgentPercept instance.
        """
        peer_agent = self.agent_b if agent is self.agent_a else self.agent_a
        peer_pos = peer_agent.position if include_peer else None
        available_moves = tuple(self.get_orthogonal_neighbors(agent.position))
        grid_tuple = tuple(tuple(int(v) for v in row) for row in self._grid)

        return MultiAgentPercept(
            agent_name=agent.name,
            position=agent.position,
            cell_value=self.get_cell(agent.position),
            available_moves=available_moves,
            other_agent_position=peer_pos,
            grid_view=grid_tuple,
        )

    def execute_agent_turn(
        self,
        agent: BaseAgent,
        turn_index: int,
        step_number: int,
        avoid_collisions: bool = False,
        step_mode: str = TURN_MODE_COMPOUND,
    ) -> AgentTurnRecord:
        """Execute a single agent's turn on the shared environment.

        Args:
            agent: The active agent.
            turn_index: Turn order (1 or 2).
            step_number: Current simulation round.
            avoid_collisions: Whether collision avoidance is active.
            step_mode: 'compound' (act + move) or 'discrete' (act or move).

        Returns:
            Recorded AgentTurnRecord.
        """
        pos_before = agent.position
        cell_before = self.get_cell(pos_before)
        percept = self.generate_percept(agent, include_peer=avoid_collisions)

        if step_mode == TURN_MODE_DISCRETE:
            action, target_pos, avoided = agent.decide_discrete_action(
                percept, avoid_collisions=avoid_collisions
            )
            move_action = None

            if action == ACTION_BUILD:
                self.set_cell(pos_before, CELL_BLOCK)
                self.blocks_placed_total += 1
            elif action == ACTION_DESTROY:
                self.set_cell(pos_before, CELL_EMPTY)
                self.blocks_destroyed_total += 1
            elif target_pos is not None:
                self.validate_coordinate(target_pos)
                agent.set_position(target_pos)
        else:
            # Compound mode: execute physical cell action THEN spatial move
            cell_action, move_action, target_pos, avoided = agent.decide_compound_action(
                percept, avoid_collisions=avoid_collisions
            )
            action = cell_action

            if cell_action == ACTION_BUILD:
                self.set_cell(pos_before, CELL_BLOCK)
                self.blocks_placed_total += 1
            elif cell_action == ACTION_DESTROY:
                self.set_cell(pos_before, CELL_EMPTY)
                self.blocks_destroyed_total += 1

            if target_pos is not None:
                self.validate_coordinate(target_pos)
                agent.set_position(target_pos)

        if avoided:
            self.collisions_avoided_total += 1

        # Check for collision
        if self.agent_a.position == self.agent_b.position:
            self.collisions_occurred_total += 1

        pos_after = agent.position
        cell_after = self.get_cell(pos_after)
        grid_snapshot = self._grid.tolist()

        record = AgentTurnRecord(
            step_number=step_number,
            turn_index=turn_index,
            agent_name=agent.name,
            agent_role=agent.role,
            pos_before=pos_before,
            cell_before=cell_before,
            action=action,
            move_action=move_action,
            pos_after=pos_after,
            cell_after=cell_after,
            collision_avoided=avoided,
            grid_snapshot=grid_snapshot,
        )
        self.history.append(record)
        return record

    def step(
        self,
        step_number: int,
        avoid_collisions: bool = False,
        step_mode: str = TURN_MODE_COMPOUND,
    ) -> Tuple[AgentTurnRecord, AgentTurnRecord]:
        """Execute one full simulation round: Agent A takes turn, then Agent B.

        Args:
            step_number: Discrete round index (1-based).
            avoid_collisions: Whether collision avoidance is enabled.
            step_mode: Turn execution mode ('compound' or 'discrete').

        Returns:
            Tuple of (record_agent_a, record_agent_b).
        """
        self.step_count += 1
        rec_a = self.execute_agent_turn(
            self.agent_a,
            turn_index=1,
            step_number=step_number,
            avoid_collisions=avoid_collisions,
            step_mode=step_mode,
        )
        rec_b = self.execute_agent_turn(
            self.agent_b,
            turn_index=2,
            step_number=step_number,
            avoid_collisions=avoid_collisions,
            step_mode=step_mode,
        )
        return rec_a, rec_b


def run_multi_agent_simulation(
    steps: int = 10,
    grid_size: Tuple[int, int] = (4, 4),
    start_corner_a: Tuple[int, int] = (0, 0),
    start_corner_b: Optional[Tuple[int, int]] = None,
    avoid_collisions: bool = False,
    movement_strategy: str = STRATEGY_RANDOM,
    step_mode: str = TURN_MODE_COMPOUND,
    seed: Optional[int] = 42,
    initial_grid: Optional[np.ndarray] = None,
) -> MultiAgentSimulationResult:
    """Execute complete Multi-Agent Environment Simulation over S steps.

    Args:
        steps: Total number of sequential interaction rounds (default: 10).
        grid_size: Dimensions of the grid (rows, cols) (default: (4, 4)).
        start_corner_a: Starting coordinates of Agent A (Builder) (default: (0, 0)).
        start_corner_b: Starting coordinates of Agent B (Destroyer) (default: opposite corner).
        avoid_collisions: Optional challenge: enable collision awareness and avoidance.
        movement_strategy: Movement heuristic ('random' or 'greedy').
        step_mode: Turn execution mode ('compound' or 'discrete').
        seed: Deterministic random seed for reproducible trajectories.
        initial_grid: Optional initial 2D matrix (defaults to empty 4x4 matrix of 0s).

    Returns:
        MultiAgentSimulationResult documenting the complete interaction history and final state.

    Raises:
        ValueError: If steps < 1 or parameters are invalid.
    """
    if steps < 1:
        raise ValueError(f"Simulation steps must be >= 1, got {steps}.")
    if step_mode not in VALID_TURN_MODES:
        raise ValueError(
            f"Invalid step_mode '{step_mode}'. Expected one of {VALID_TURN_MODES}."
        )

    rows, cols = grid_size
    if start_corner_b is None:
        start_corner_b = (rows - 1, cols - 1)

    # Derive agent-specific seeds for independent, reproducible RNG streams
    seed_a = seed if seed is not None else None
    seed_b = (seed + 1000) if seed is not None else None

    agent_a = BuilderAgent(
        name="Agent_A",
        initial_position=start_corner_a,
        seed=seed_a,
        strategy=movement_strategy,
    )
    agent_b = DestroyerAgent(
        name="Agent_B",
        initial_position=start_corner_b,
        seed=seed_b,
        strategy=movement_strategy,
    )

    env = MultiAgentGridEnvironment(
        rows=rows,
        cols=cols,
        agent_a=agent_a,
        agent_b=agent_b,
        initial_grid=initial_grid,
    )

    init_grid_snapshot = env.grid.copy()

    for s in range(1, steps + 1):
        env.step(
            step_number=s,
            avoid_collisions=avoid_collisions,
            step_mode=step_mode,
        )

    final_grid_matrix = env.grid.copy()
    final_blocks = int(np.sum(final_grid_matrix == CELL_BLOCK))

    return MultiAgentSimulationResult(
        initial_grid=init_grid_snapshot,
        final_grid=final_grid_matrix,
        agent_a_name=agent_a.name,
        agent_a_role=agent_a.role,
        agent_a_start=start_corner_a,
        agent_a_final=agent_a.position,
        agent_b_name=agent_b.name,
        agent_b_role=agent_b.role,
        agent_b_start=start_corner_b,
        agent_b_final=agent_b.position,
        total_steps=steps,
        blocks_placed=env.blocks_placed_total,
        blocks_destroyed=env.blocks_destroyed_total,
        final_block_count=final_blocks,
        collision_avoidance_active=avoid_collisions,
        collisions_avoided_count=env.collisions_avoided_total,
        collisions_occurred=env.collisions_occurred_total,
        step_mode=step_mode,
        history=env.history,
    )
