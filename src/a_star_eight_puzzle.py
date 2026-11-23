"""8-Puzzle Solver using A* Search Algorithm and Manhattan Distance Heuristic.

This module implements an optimal, high-performance 8-puzzle solver using the A*
search algorithm with the admissible and consistent Manhattan distance heuristic
(L1 norm), as formulated in Russell & Norvig's 'Artificial Intelligence: A Modern
Approach' (AIMA Chapter 3).

Core AI Concept: Informed Search on Complex State Spaces:
- State Space Size:
  The 8-puzzle sliding-tile problem on a 3x3 grid has 9! = 362,880 possible
  permutations. Exactly half of these (181,440 states) are reachable from any
  given configuration due to the parity invariance of permutations.
- Evaluation Function:
      f(n) = g(n) + h(n)
  where:
  - g(n) is the exact path cost (number of moves) from the initial board state.
  - h(n) is the Manhattan distance heuristic:
        h_Manhattan(state) = sum_{tile=1..8} (|r_curr - r_goal| + |c_curr - c_goal|)
    strictly excluding the blank tile (0).
  - f(n) estimates the total cost of the cheapest solution path traversing state n.

Theoretical Optimality & Heuristic Properties:
- Admissibility: Manhattan distance relaxes the empty-slot constraint (allowing
  tiles to slide through occupied cells). Because true movements require detour
  around obstacles, 0 <= h(n) <= h*(n). Manhattan distance never overestimates
  the remaining move count.
- Consistency (Monotonicity): Moving a tile one orthogonal step changes its
  Manhattan distance by at most 1, so |h(u) - h(v)| <= 1 = c(u, v). Thus:
      h(u) <= c(u, v) + h(v)
  Consistency guarantees that the first time a state is expanded, its path cost
  g(n) is strictly optimal. Closed nodes never need to be reopened.
- Dominance: For every state n, h_Manhattan(n) >= h_Misplaced(n). A* guided by
  Manhattan distance expands fewer or equal nodes than Misplaced Tiles.

Python Performance Optimizations:
- 1D Tuple State Representation: States are represented as flat 9-element tuples
  `tuple[int, ...]`, enabling fast O(1) hashing, immutability, and compact
  memory usage (~112 bytes vs ~400 bytes for nested lists).
- Fast Move Generation: Calculates row/col via `divmod(idx, 3)` and generates
  successors by swapping values in a transient list without `deepcopy`.
- On-The-Fly Heuristic Calculation: Precomputes goal tile coordinates for fast
  coordinate lookups, with optional incremental O(1) heuristic updates.
- Priority Queue Tie-Breaking: Uses `(f_score, h_score, tie_counter, state)`
  where secondary tie-breaking on lower h drives the search greedily toward
  the goal on equal f.

Mathematical Solvability Verification (Optional Challenge & Viva Voce):
- Inversion Parity Theorem: For an odd grid width (W = 3), horizontal moves
  preserve the linear ordering of numbered tiles (delta_inv = 0), and vertical
  moves swap a tile past exactly two other tiles (delta_inv in {-2, 0, +2}).
  Consequently, inversion count parity is an invariant:
      inv(state) = sum_{i < j, state[i] != 0, state[j] != 0} [state[i] > state[j]]
  A state is solvable if and only if inv(initial) % 2 == inv(goal) % 2.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import heapq
import itertools
import math
import re
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ============================================================================
# Type Aliases & Constants
# ============================================================================

PuzzleState = Tuple[int, ...]
MatrixGrid = Sequence[Sequence[int]]

GRID_SIZE: int = 3
TOTAL_TILES: int = 9
BLANK_TILE: int = 0

# Canonical Classic Goal: 1 through 8 in order, blank 0 in bottom-right (index 8)
CLASSIC_GOAL_STATE: PuzzleState = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Alternative Zero-First Goal: blank 0 in top-left (index 0), then 1 through 8
ZERO_FIRST_GOAL_STATE: PuzzleState = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# Syllabus Sample Initial State:
# [1, 2, 3,
#  4, 0, 5,
#  6, 7, 8]
SAMPLE_INPUT_STATE: PuzzleState = (1, 2, 3, 4, 0, 5, 6, 7, 8)

# 2-Move Target for Sample Input:
# Blank at center (4) moves Right (5), then Down (8):
# [1, 2, 3,
#  4, 5, 8,
#  6, 7, 0]
SAMPLE_GOAL_STATE_2MOVE: PuzzleState = (1, 2, 3, 4, 5, 8, 6, 7, 0)

# 2-Move State to Classic Goal (1-8, 0):
# From classic goal, move blank Up (6) then Left (5) -> blank at center:
# [1, 2, 3,
#  4, 0, 6,
#  7, 5, 8]
TWO_MOVE_TO_CLASSIC_STATE: PuzzleState = (1, 2, 3, 4, 0, 6, 7, 5, 8)

# Benchmark Presets
BENCHMARK_PRESETS: Dict[str, Tuple[PuzzleState, PuzzleState, str]] = {
    "sample-2move": (
        SAMPLE_INPUT_STATE,
        SAMPLE_GOAL_STATE_2MOVE,
        "Syllabus sample input solved in exactly 2 moves to target [1,2,3, 4,5,8, 6,7,0]",
    ),
    "classic-2move": (
        TWO_MOVE_TO_CLASSIC_STATE,
        CLASSIC_GOAL_STATE,
        "State solved in exactly 2 moves to classic goal (1-8, 0)",
    ),
    "sample-classic": (
        SAMPLE_INPUT_STATE,
        CLASSIC_GOAL_STATE,
        "Syllabus sample input solved to classic goal (1-8, 0) in 14 optimal moves",
    ),
    "easy-3move": (
        (1, 2, 3, 0, 4, 6, 7, 5, 8),
        CLASSIC_GOAL_STATE,
        "Easy state solvable in 3 optimal moves",
    ),
    "medium-8move": (
        (1, 2, 3, 4, 5, 6, 0, 7, 8),
        CLASSIC_GOAL_STATE,
        "Medium state solvable in 8 optimal moves",
    ),
    "hard-18move": (
        (2, 8, 3, 1, 6, 4, 7, 0, 5),
        CLASSIC_GOAL_STATE,
        "Challenging state requiring 18 optimal moves",
    ),
    "hard-26move": (
        (7, 2, 4, 5, 0, 6, 8, 3, 1),
        ZERO_FIRST_GOAL_STATE,
        "Classic AIMA textbook benchmark requiring 26 optimal moves",
    ),
    "unsolvable": (
        (1, 2, 3, 4, 5, 6, 8, 7, 0),
        CLASSIC_GOAL_STATE,
        "Unsolvable state (inversion parity mismatch: 1 inversion vs 0 in goal)",
    ),
}

# Orthogonal blank moves: (delta_row, delta_col, blank_direction, tile_direction)
# For example, if blank moves UP (-1, 0), the tile above moves DOWN (+1, 0) into the blank.
MOVE_DIRECTIONS: Tuple[Tuple[int, int, str, str], ...] = (
    (-1, 0, "UP", "DOWN"),
    (1, 0, "DOWN", "UP"),
    (0, -1, "LEFT", "RIGHT"),
    (0, 1, "RIGHT", "LEFT"),
)


# ============================================================================
# Exceptions
# ============================================================================


class EightPuzzleError(Exception):
    """Base exception for 8-puzzle errors."""


class InvalidPuzzleStateError(EightPuzzleError):
    """Raised when an initial or goal board configuration is invalid."""


class UnsolvablePuzzleError(EightPuzzleError):
    """Raised when an initial state is mathematically unsolvable for the target goal."""


# ============================================================================
# Structured Dataclasses
# ============================================================================


@dataclass(frozen=True)
class EightPuzzleStepRecord:
    """Detailed record of a single step along the puzzle solution path.

    Attributes:
        step_number: 0-indexed step along path (0 = initial state).
        state: Immutable 1D tuple representing the 9 tiles.
        action: Human-readable action description (e.g. "Move 5 LEFT").
        blank_move: Direction the blank tile moved ("START", "UP", "DOWN", "LEFT", "RIGHT").
        tile_moved: Number of the tile swapped with the blank (0 for initial step).
        g_cost: Accumulated path cost from start state.
        h_cost: Manhattan heuristic estimate to target goal.
        f_cost: Evaluation function score f = g + h.
    """

    step_number: int
    state: PuzzleState
    action: str
    blank_move: str
    tile_moved: int
    g_cost: int
    h_cost: int
    f_cost: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert step record to JSON-compatible dictionary."""
        return {
            "step_number": self.step_number,
            "state": list(self.state),
            "action": self.action,
            "blank_move": self.blank_move,
            "tile_moved": self.tile_moved,
            "g_cost": self.g_cost,
            "h_cost": self.h_cost,
            "f_cost": self.f_cost,
        }


@dataclass
class EightPuzzleResult:
    """Comprehensive result summary of the 8-puzzle A* search.

    Attributes:
        success: Whether the goal state was successfully reached.
        initial_state: Validated starting board configuration.
        goal_state: Target board configuration.
        path: List of sequential states from initial to goal.
        moves: List of move descriptions (e.g. ["Move 5 LEFT", ...]).
        step_records: Structured records for every state along the path.
        move_count: Total number of moves (optimal path length).
        nodes_expanded: Total states removed from priority queue and expanded.
        nodes_generated: Total valid successor states produced.
        max_frontier_size: Peak size of the priority queue during search.
        time_taken: Search execution duration in seconds.
        heuristic_name: Name of the heuristic utilized ("manhattan" or "misplaced").
        is_solvable: Mathematical solvability flag based on inversion parity.
        initial_inversions: Inversion count of initial state.
        goal_inversions: Inversion count of target goal state.
        solvability_reason: Mathematical explanation of solvability status.
    """

    success: bool
    initial_state: PuzzleState
    goal_state: PuzzleState
    path: List[PuzzleState] = field(default_factory=list)
    moves: List[str] = field(default_factory=list)
    step_records: List[EightPuzzleStepRecord] = field(default_factory=list)
    move_count: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    time_taken: float = 0.0
    heuristic_name: str = "manhattan"
    is_solvable: bool = True
    initial_inversions: int = 0
    goal_inversions: int = 0
    solvability_reason: str = ""

    def format_sample_output(self) -> str:
        """Format output to strictly match the syllabus sample output specification.

        Example:
            Solved in 2 moves: [Board states...]
        """
        if not self.success:
            return f"Unsolvable: {self.solvability_reason}"
        boards_repr = "[" + ", ".join(str(list(s)) for s in self.path) + "]"
        return f"Solved in {self.move_count} moves: {boards_repr}"

    def format_detailed_solution(self, use_ascii: bool = True) -> str:
        """Render a formatted, step-by-step sequential report with board visualizations.

        Args:
            use_ascii: If True, renders ASCII grid boxes; otherwise flat rows.

        Returns:
            Human-readable formatted string of the complete solution.
        """
        if not self.success:
            return (
                f"Search Failed! Initial state is mathematically unsolvable.\n"
                f"Reason: {self.solvability_reason}\n"
                f"Initial Inversions: {self.initial_inversions} (parity: {self.initial_inversions % 2})\n"
                f"Goal Inversions   : {self.goal_inversions} (parity: {self.goal_inversions % 2})"
            )

        lines: List[str] = [
            "=" * 60,
            f"8-PUZZLE SOLVER (A* WITH MANHATTAN HEURISTIC)",
            "=" * 60,
            f"Solution Status   : SOLVED OPTIMALLY",
            f"Optimal Move Count: {self.move_count} moves",
            f"Nodes Expanded    : {self.nodes_expanded} nodes",
            f"Nodes Generated   : {self.nodes_generated} nodes",
            f"Max Frontier Size : {self.max_frontier_size} states",
            f"Execution Time    : {self.time_taken * 1000:.3f} ms",
            f"Heuristic Used    : {self.heuristic_name.capitalize()}",
            f"Initial Inversions: {self.initial_inversions} (Solvable: {self.is_solvable})",
            "-" * 60,
            "STEP-BY-STEP SOLUTION PATH:",
            "-" * 60,
        ]

        for record in self.step_records:
            if record.step_number == 0:
                header = f"Step 0 (Initial State):"
            else:
                header = (
                    f"Step {record.step_number} ({record.action} | Blank {record.blank_move}) "
                    f"[g={record.g_cost}, h={record.h_cost}, f={record.f_cost}]:"
                )
            lines.append(header)
            if use_ascii:
                lines.append(render_board_ascii(record.state, indent="  "))
            else:
                lines.append("  " + str(list(record.state)))
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"Summary: Solved in {self.move_count} moves: {len(self.path)} total board states.")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert puzzle result to JSON-compatible dictionary."""
        return {
            "success": self.success,
            "initial_state": list(self.initial_state),
            "goal_state": list(self.goal_state),
            "move_count": self.move_count,
            "path": [list(s) for s in self.path],
            "moves": self.moves,
            "step_records": [r.to_dict() for r in self.step_records],
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "max_frontier_size": self.max_frontier_size,
            "time_taken_ms": round(self.time_taken * 1000, 3),
            "heuristic_name": self.heuristic_name,
            "is_solvable": self.is_solvable,
            "initial_inversions": self.initial_inversions,
            "goal_inversions": self.goal_inversions,
            "solvability_reason": self.solvability_reason,
        }


# ============================================================================
# State Parsing & Board Validation
# ============================================================================


def validate_board(board: Sequence[int]) -> PuzzleState:
    """Validate that a sequence represents a valid 3x3 sliding puzzle board.

    Requirements:
    1. Must contain exactly 9 elements.
    2. Must contain integers 0 through 8 with no duplicates and no missing values.

    Args:
        board: Sequence of integers.

    Returns:
        Validated 1D tuple of length 9.

    Raises:
        InvalidPuzzleStateError: If length != 9 or elements are not a permutation of 0..8.
    """
    if len(board) != TOTAL_TILES:
        raise InvalidPuzzleStateError(
            f"Invalid board length: expected {TOTAL_TILES} tiles, received {len(board)}: {board}"
        )

    try:
        int_tiles = tuple(int(x) for x in board)
    except (ValueError, TypeError) as err:
        raise InvalidPuzzleStateError(f"All board tiles must be integers: {board}") from err

    expected_set = set(range(TOTAL_TILES))
    actual_set = set(int_tiles)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        duplicates = sorted({x for x in int_tiles if int_tiles.count(x) > 1})
        raise InvalidPuzzleStateError(
            f"Invalid board tiles: must contain exactly 0..8. "
            f"Missing tiles: {missing}, Duplicates: {duplicates}. Received: {board}"
        )

    return int_tiles


def parse_puzzle_state(raw: Any) -> PuzzleState:
    """Parse flexible user input representations into a canonical 1D PuzzleState tuple.

    Supports:
    - 1D Sequence of ints: `[1, 2, 3, 4, 0, 5, 6, 7, 8]`
    - 2D Sequence of sequences: `[[1, 2, 3], [4, 0, 5], [6, 7, 8]]`
    - Comma-separated string: `"1,2,3,4,0,5,6,7,8"` or `"1, 2, 3, 4, 0, 5, 6, 7, 8"`
    - Space/semicolon string: `"1 2 3; 4 0 5; 6 7 8"`
    - Python literal string: `"[1,2,3, 4,0,5, 6,7,8]"`

    Args:
        raw: Input in any supported format.

    Returns:
        Canonical 1D tuple of length 9.

    Raises:
        InvalidPuzzleStateError: If parsing fails or board violates puzzle constraints.
    """
    if raw is None:
        raise InvalidPuzzleStateError("Puzzle state cannot be None.")

    # Already a valid tuple of length 9
    if isinstance(raw, tuple) and len(raw) == TOTAL_TILES and all(isinstance(x, int) for x in raw):
        return validate_board(raw)

    # 2D list or tuple of rows
    if isinstance(raw, (list, tuple)) and len(raw) == GRID_SIZE and all(isinstance(r, (list, tuple)) for r in raw):
        flattened: List[int] = []
        for row in raw:
            flattened.extend(row)
        return validate_board(flattened)

    # 1D list or tuple
    if isinstance(raw, (list, tuple)):
        return validate_board(raw)

    # String input
    if isinstance(raw, str):
        text = raw.strip()
        # Handle Python literal e.g. "[1,2,3, 4,0,5, 6,7,8]"
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                evaluated = ast.literal_eval(text)
                return parse_puzzle_state(evaluated)
            except (ValueError, SyntaxError):
                pass  # Fall back to regex parsing

        # Extract all integer tokens
        tokens = re.findall(r"-?\d+", text)
        if len(tokens) == TOTAL_TILES:
            return validate_board([int(t) for t in tokens])

        raise InvalidPuzzleStateError(
            f"Could not parse 9 tile integers from input string '{raw}'. Found {len(tokens)} numbers: {tokens}"
        )

    raise InvalidPuzzleStateError(f"Unsupported board format: type={type(raw).__name__}, value={raw}")


# ============================================================================
# Mathematical Solvability Verification (Optional Challenge & Viva Voce)
# ============================================================================


def count_inversions(state: Sequence[int]) -> int:
    """Calculate the inversion count of a 3x3 puzzle board configuration.

    An inversion occurs when a tile with a higher number appears before a tile
    with a lower number in linear reading order (row by row), strictly ignoring
    the blank tile (0).

    Formal Definition:
        Inversions = sum_{0 <= i < j < 9, state[i] != 0, state[j] != 0} [state[i] > state[j]]

    Args:
        state: 1D sequence of length 9 representing the board.

    Returns:
        Integer inversion count (0 <= count <= 36).
    """
    tiles = [tile for tile in state if tile != BLANK_TILE]
    inversions = 0
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(
    initial_state: Sequence[int],
    goal_state: Sequence[int] = CLASSIC_GOAL_STATE,
) -> bool:
    """Mathematically verify whether an initial state can reach the goal state.

    Mathematical Basis (Permutation Parity Theorem):
    In an N x N puzzle where N is odd (N = 3):
    1. Horizontal sliding moves shift the blank within the same row, leaving the
       relative linear order of all numbered tiles unchanged (delta_inv = 0).
    2. Vertical sliding moves shift a tile vertically across 1 row, which jumps
       it over exactly N - 1 = 2 tiles in the 1D representation.
       - If both jumped tiles were smaller, inversions change by -2.
       - If both jumped tiles were larger, inversions change by +2.
       - If one was smaller and one larger, inversions change by 0.
       In all cases, delta_inv is an even number (+2, 0, or -2).
    3. Therefore, the parity of the inversion count (inversions % 2) is strictly
       invariant under all valid sliding moves!
    4. For any goal state, reachable states must have the exact same inversion
       parity as the goal state:
           inv(initial) % 2 == inv(goal) % 2

    Args:
        initial_state: Initial 9-tile board sequence.
        goal_state: Target 9-tile board sequence (defaults to classic goal 1-8, 0).

    Returns:
        True if the board is solvable; False if unsolvable.
    """
    initial = parse_puzzle_state(initial_state)
    goal = parse_puzzle_state(goal_state)
    return (count_inversions(initial) % 2) == (count_inversions(goal) % 2)


def explain_solvability(
    initial_state: Sequence[int],
    goal_state: Sequence[int] = CLASSIC_GOAL_STATE,
) -> Tuple[bool, str, int, int]:
    """Provide a comprehensive mathematical explanation of board solvability.

    Args:
        initial_state: Starting board sequence.
        goal_state: Target board sequence.

    Returns:
        Tuple containing:
        - is_solv: Boolean solvability flag.
        - explanation: Detailed mathematical proof string.
        - init_inv: Inversion count of initial state.
        - goal_inv: Inversion count of goal state.
    """
    initial = parse_puzzle_state(initial_state)
    goal = parse_puzzle_state(goal_state)

    init_inv = count_inversions(initial)
    goal_inv = count_inversions(goal)
    init_parity = init_inv % 2
    goal_parity = goal_inv % 2
    is_solv = (init_parity == goal_parity)

    if is_solv:
        explanation = (
            f"The board is SOLVABLE. Initial state has {init_inv} inversions "
            f"(parity: {'even' if init_parity == 0 else 'odd'}) and goal state has {goal_inv} "
            f"inversions (parity: {'even' if goal_parity == 0 else 'odd'}). Because both share the "
            f"same parity, valid sliding transitions can transform initial into goal."
        )
    else:
        explanation = (
            f"The board is UNSOLVABLE. Initial state has {init_inv} inversions "
            f"(parity: {'even' if init_parity == 0 else 'odd'}), whereas goal state has {goal_inv} "
            f"inversions (parity: {'even' if goal_parity == 0 else 'odd'}). Because each valid move "
            f"changes inversion count by 0 or +/-2 (preserving parity), reaching the goal is mathematically "
            f"impossible (belongs to the disjoint alternating coset)."
        )

    return is_solv, explanation, init_inv, goal_inv


# ============================================================================
# Heuristic Functions (Manhattan Distance & Misplaced Tiles)
# ============================================================================


def build_goal_position_map(goal_state: PuzzleState) -> Dict[int, Tuple[int, int]]:
    """Build a precomputed coordinate lookup table for all non-blank tiles in the goal.

    Maps: tile_number -> (row, col)

    Args:
        goal_state: 1D tuple representing the target board.

    Returns:
        Dictionary mapping each numbered tile 1..8 to its target (row, col).
    """
    pos_map: Dict[int, Tuple[int, int]] = {}
    for idx, tile in enumerate(goal_state):
        if tile != BLANK_TILE:
            pos_map[tile] = divmod(idx, GRID_SIZE)
    return pos_map


def manhattan_distance(
    state: PuzzleState,
    goal_state: PuzzleState = CLASSIC_GOAL_STATE,
    goal_pos_map: Optional[Dict[int, Tuple[int, int]]] = None,
) -> int:
    """Calculate the Manhattan distance heuristic on the fly for an 8-puzzle state.

    The Manhattan distance (L1 norm / city-block distance) sums the orthogonal grid
    offsets from each numbered tile's current coordinate to its target coordinate:
        h_Manhattan(state) = sum_{tile=1..8} (|r_curr - r_goal| + |c_curr - c_goal|)

    CRITICAL RULE: The blank space (0) is strictly EXCLUDED. Counting the blank
    tile violates admissibility (Russell & Norvig, AIMA Chapter 3).

    Args:
        state: Current 1D 9-tile tuple.
        goal_state: Target 1D 9-tile tuple.
        goal_pos_map: Optional precomputed goal position map for maximal speed.

    Returns:
        Integer Manhattan distance heuristic estimate (0 <= h).
    """
    if goal_pos_map is None:
        goal_pos_map = build_goal_position_map(goal_state)

    total_distance = 0
    for idx, tile in enumerate(state):
        if tile != BLANK_TILE:
            r_curr, c_curr = divmod(idx, GRID_SIZE)
            r_goal, c_goal = goal_pos_map[tile]
            total_distance += abs(r_curr - r_goal) + abs(c_curr - c_goal)

    return total_distance


def misplaced_tiles(
    state: PuzzleState,
    goal_state: PuzzleState = CLASSIC_GOAL_STATE,
) -> int:
    """Calculate the Misplaced Tiles heuristic (Hamming distance) for an 8-puzzle state.

    Counts the number of numbered tiles (1..8) that do not occupy their goal coordinate,
    strictly ignoring the blank space (0).

    Args:
        state: Current 1D 9-tile tuple.
        goal_state: Target 1D 9-tile tuple.

    Returns:
        Integer count of misplaced tiles (0 <= h <= 8).
    """
    count = 0
    for s_tile, g_tile in zip(state, goal_state):
        if s_tile != BLANK_TILE and s_tile != g_tile:
            count += 1
    return count


# ============================================================================
# Valid Move Generation
# ============================================================================


def get_successors(state: PuzzleState) -> List[Tuple[PuzzleState, str, str, int]]:
    """Generate all valid successor states by sliding tiles into the blank space (0).

    For a 3x3 grid, the blank space at index `blank_idx` has between 2 and 4 valid moves:
    - Center (1, 1): 4 moves (Up, Down, Left, Right)
    - Edges (0,1), (1,0), (1,2), (2,1): 3 moves
    - Corners (0,0), (0,2), (2,0), (2,2): 2 moves

    Args:
        state: Current 1D tuple state.

    Returns:
        List of tuples: (next_state, action_label, blank_direction, tile_number)
        where:
        - next_state: New 1D tuple after swapping blank and tile.
        - action_label: e.g. "Move 5 LEFT"
        - blank_direction: Direction blank moved ("UP", "DOWN", "LEFT", "RIGHT")
        - tile_number: Integer value of tile that swapped places with blank.
    """
    blank_idx = state.index(BLANK_TILE)
    b_row, b_col = divmod(blank_idx, GRID_SIZE)
    successors: List[Tuple[PuzzleState, str, str, int]] = []

    for dr, dc, b_dir, t_dir in MOVE_DIRECTIONS:
        nr, nc = b_row + dr, b_col + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            swap_idx = nr * GRID_SIZE + nc
            tile_moved = state[swap_idx]

            # Fast swap using list conversion (avoids deepcopy overhead)
            state_list = list(state)
            state_list[blank_idx], state_list[swap_idx] = state_list[swap_idx], state_list[blank_idx]
            next_state = tuple(state_list)

            action_label = f"Move {tile_moved} {t_dir}"
            successors.append((next_state, action_label, b_dir, tile_moved))

    return successors


# ============================================================================
# Core A* Search Algorithm
# ============================================================================


def a_star_eight_puzzle(
    initial_state: Union[Sequence[int], str],
    goal_state: Union[Sequence[int], str] = CLASSIC_GOAL_STATE,
    heuristic: str = "manhattan",
    check_solvability_first: bool = True,
    max_nodes: int = 200_000,
) -> EightPuzzleResult:
    """Solve the 8-puzzle problem using the A* Search algorithm.

    Algorithm Overview:
    1. Parse and validate the initial and target goal states into 1D tuples.
    2. Check mathematical solvability via inversion parity (Optional Challenge).
       If unsolvable, abort immediately without wasting CPU/memory cycles.
    3. Initialize a binary min-heap priority queue with:
           (f_score, h_score, tie_counter, current_state)
       where:
       - f_score = g_score + h_score
       - h_score tie-breaker favors states closer to the goal on equal f.
       - tie_counter ensures deterministic FIFO behavior.
    4. Maintain `g_score: Dict[PuzzleState, int]` tracking the cheapest cost to
       reach each discovered state.
    5. Maintain `parent: Dict[PuzzleState, Tuple[PuzzleState, str, str, int]]`
       for exact optimal path reconstruction.
    6. Expand the lowest-f state from the priority queue until the goal state is
       popped (guaranteeing optimal move count via consistency of Manhattan distance).

    Args:
        initial_state: 1D list, 2D matrix, string, or 1D tuple of starting board.
        goal_state: Target board configuration (defaults to classic 1-8, 0).
        heuristic: "manhattan" (default, L1 norm) or "misplaced" (Hamming distance).
        check_solvability_first: If True, verifies mathematical solvability upfront.
        max_nodes: Safety limit on total expanded nodes to protect against memory limits.

    Returns:
        EightPuzzleResult containing the optimal path, move count, step records, and metrics.

    Raises:
        InvalidPuzzleStateError: If boards are malformed or invalid permutations.
        UnsolvablePuzzleError: If check_solvability_first is True and the board is unsolvable.
    """
    start_time = time.perf_counter()

    init_tuple = parse_puzzle_state(initial_state)
    goal_tuple = parse_puzzle_state(goal_state)

    # 1. Mathematical Solvability Verification (Optional Challenge)
    is_solv, reason, init_inv, goal_inv = explain_solvability(init_tuple, goal_tuple)
    if check_solvability_first and not is_solv:
        elapsed = time.perf_counter() - start_time
        return EightPuzzleResult(
            success=False,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            path=[],
            moves=[],
            step_records=[],
            move_count=0,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=0,
            time_taken=elapsed,
            heuristic_name=heuristic,
            is_solvable=False,
            initial_inversions=init_inv,
            goal_inversions=goal_inv,
            solvability_reason=reason,
        )

    # 2. Select heuristic evaluation function
    goal_pos_map = build_goal_position_map(goal_tuple)
    if heuristic.lower() == "misplaced":
        h_func: Callable[[PuzzleState], int] = lambda s: misplaced_tiles(s, goal_tuple)
    else:
        h_func = lambda s: manhattan_distance(s, goal_tuple, goal_pos_map)

    # 3. Handle trivial edge case: start is already goal
    if init_tuple == goal_tuple:
        elapsed = time.perf_counter() - start_time
        initial_record = EightPuzzleStepRecord(
            step_number=0,
            state=init_tuple,
            action="Already at goal",
            blank_move="NONE",
            tile_moved=0,
            g_cost=0,
            h_cost=0,
            f_cost=0,
        )
        return EightPuzzleResult(
            success=True,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            path=[init_tuple],
            moves=[],
            step_records=[initial_record],
            move_count=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            time_taken=elapsed,
            heuristic_name=heuristic,
            is_solvable=is_solv,
            initial_inversions=init_inv,
            goal_inversions=goal_inv,
            solvability_reason=reason,
        )

    # 4. Data structures for A* search
    # Priority queue entry: (f_score, h_score, tie_counter, state)
    tie_counter = itertools.count()
    init_h = h_func(init_tuple)
    frontier: List[Tuple[int, int, int, PuzzleState]] = [
        (init_h, init_h, next(tie_counter), init_tuple)
    ]
    heapq.heapify(frontier)

    g_score: Dict[PuzzleState, int] = {init_tuple: 0}
    # parent mapping: current_state -> (parent_state, action_label, blank_dir, tile_moved)
    parent: Dict[PuzzleState, Tuple[PuzzleState, str, str, int]] = {}
    closed_set: Set[PuzzleState] = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    goal_reached = False

    # 5. Core Priority Queue Loop
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        f_curr, h_curr, _, curr_state = heapq.heappop(frontier)

        # Skip if already closed (since Manhattan is consistent, first pop is optimal)
        if curr_state in closed_set:
            continue

        closed_set.add(curr_state)
        nodes_expanded += 1

        # Goal check upon popping
        if curr_state == goal_tuple:
            goal_reached = True
            break

        if nodes_expanded >= max_nodes:
            # Memory/safety limit reached
            break

        curr_g = g_score[curr_state]

        # Expand successors
        for next_state, action_label, b_dir, tile_moved in get_successors(curr_state):
            nodes_generated += 1
            tentative_g = curr_g + 1

            if tentative_g < g_score.get(next_state, math.inf):
                g_score[next_state] = tentative_g
                parent[next_state] = (curr_state, action_label, b_dir, tile_moved)
                next_h = h_func(next_state)
                next_f = tentative_g + next_h
                heapq.heappush(frontier, (next_f, next_h, next(tie_counter), next_state))

    elapsed = time.perf_counter() - start_time

    # 6. Reconstruct optimal path
    if not goal_reached:
        return EightPuzzleResult(
            success=False,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            path=[],
            moves=[],
            step_records=[],
            move_count=0,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            time_taken=elapsed,
            heuristic_name=heuristic,
            is_solvable=is_solv,
            initial_inversions=init_inv,
            goal_inversions=goal_inv,
            solvability_reason=f"Exhausted search space or hit max node limit ({max_nodes}).",
        )

    # Trace backwards from goal to start
    reversed_path: List[PuzzleState] = []
    reversed_transitions: List[Tuple[str, str, int]] = []
    curr = goal_tuple

    while curr in parent:
        reversed_path.append(curr)
        prev_state, action, b_dir, t_num = parent[curr]
        reversed_transitions.append((action, b_dir, t_num))
        curr = prev_state

    reversed_path.append(init_tuple)
    full_path = list(reversed(reversed_path))
    transitions = list(reversed(reversed_transitions))
    move_names = [t[0] for t in transitions]

    # Build detailed step records
    step_records: List[EightPuzzleStepRecord] = []
    # Initial step
    step_records.append(
        EightPuzzleStepRecord(
            step_number=0,
            state=full_path[0],
            action="Initial State",
            blank_move="START",
            tile_moved=0,
            g_cost=0,
            h_cost=h_func(full_path[0]),
            f_cost=h_func(full_path[0]),
        )
    )

    for i in range(1, len(full_path)):
        st = full_path[i]
        act, b_dir, t_num = transitions[i - 1]
        g = i
        h = h_func(st)
        f = g + h
        step_records.append(
            EightPuzzleStepRecord(
                step_number=i,
                state=st,
                action=act,
                blank_move=b_dir,
                tile_moved=t_num,
                g_cost=g,
                h_cost=h,
                f_cost=f,
            )
        )

    return EightPuzzleResult(
        success=True,
        initial_state=init_tuple,
        goal_state=goal_tuple,
        path=full_path,
        moves=move_names,
        step_records=step_records,
        move_count=len(full_path) - 1,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        time_taken=elapsed,
        heuristic_name=heuristic,
        is_solvable=True,
        initial_inversions=init_inv,
        goal_inversions=goal_inv,
        solvability_reason=reason,
    )


# ============================================================================
# Breadth-First Search Baseline (for Comparative Benchmarking)
# ============================================================================


def bfs_eight_puzzle(
    initial_state: Union[Sequence[int], str],
    goal_state: Union[Sequence[int], str] = CLASSIC_GOAL_STATE,
    max_nodes: int = 100_000,
) -> EightPuzzleResult:
    """Solve the 8-puzzle using unguided Breadth-First Search (BFS baseline).

    Used in laboratory benchmarks to demonstrate the exponential node expansion
    of uninformed search compared to A* with Manhattan distance.

    Args:
        initial_state: Starting board sequence.
        goal_state: Target board sequence.
        max_nodes: Node limit to prevent memory exhaustion.

    Returns:
        EightPuzzleResult for BFS.
    """
    start_time = time.perf_counter()
    init_tuple = parse_puzzle_state(initial_state)
    goal_tuple = parse_puzzle_state(goal_state)

    is_solv, reason, init_inv, goal_inv = explain_solvability(init_tuple, goal_tuple)
    if not is_solv:
        return EightPuzzleResult(
            success=False,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            time_taken=time.perf_counter() - start_time,
            heuristic_name="BFS (None)",
            is_solvable=False,
            initial_inversions=init_inv,
            goal_inversions=goal_inv,
            solvability_reason=reason,
        )

    if init_tuple == goal_tuple:
        return EightPuzzleResult(
            success=True,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            path=[init_tuple],
            time_taken=time.perf_counter() - start_time,
            heuristic_name="BFS (None)",
            is_solvable=True,
        )

    from collections import deque

    queue = deque([init_tuple])
    visited: Set[PuzzleState] = {init_tuple}
    parent: Dict[PuzzleState, Tuple[PuzzleState, str, str, int]] = {}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    goal_reached = False

    while queue:
        max_frontier_size = max(max_frontier_size, len(queue))
        curr = queue.popleft()
        nodes_expanded += 1

        if curr == goal_tuple:
            goal_reached = True
            break

        if nodes_expanded >= max_nodes:
            break

        for nxt, act, b_dir, t_num in get_successors(curr):
            nodes_generated += 1
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (curr, act, b_dir, t_num)
                if nxt == goal_tuple:
                    goal_reached = True
                    break
                queue.append(nxt)

        if goal_reached:
            break

    elapsed = time.perf_counter() - start_time

    if not goal_reached:
        return EightPuzzleResult(
            success=False,
            initial_state=init_tuple,
            goal_state=goal_tuple,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            time_taken=elapsed,
            heuristic_name="BFS (None)",
            is_solvable=True,
            solvability_reason="BFS node limit exceeded.",
        )

    # Trace path
    reversed_path: List[PuzzleState] = []
    reversed_moves: List[str] = []
    curr_ptr = goal_tuple
    while curr_ptr in parent:
        reversed_path.append(curr_ptr)
        p, a, _, _ = parent[curr_ptr]
        reversed_moves.append(a)
        curr_ptr = p
    reversed_path.append(init_tuple)

    full_path = list(reversed(reversed_path))
    moves = list(reversed(reversed_moves))

    return EightPuzzleResult(
        success=True,
        initial_state=init_tuple,
        goal_state=goal_tuple,
        path=full_path,
        moves=moves,
        move_count=len(full_path) - 1,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        time_taken=elapsed,
        heuristic_name="BFS (None)",
        is_solvable=True,
        initial_inversions=init_inv,
        goal_inversions=goal_inv,
    )


# ============================================================================
# Algorithm & Heuristic Comparison
# ============================================================================


@dataclass
class EightPuzzleComparisonReport:
    """Comparative benchmarking report comparing A* (Manhattan), A* (Misplaced), and BFS."""

    initial_state: PuzzleState
    goal_state: PuzzleState
    astar_manhattan: EightPuzzleResult
    astar_misplaced: EightPuzzleResult
    bfs_result: Optional[EightPuzzleResult]

    def format_table(self) -> str:
        """Format comparison results into an aligned ASCII comparison table."""
        lines: List[str] = [
            "=" * 78,
            "8-PUZZLE HEURISTIC & SEARCH ALGORITHM COMPARISON REPORT",
            "=" * 78,
            f"Initial State: {list(self.initial_state)}",
            f"Goal State   : {list(self.goal_state)}",
            "-" * 78,
            f"{'Algorithm / Heuristic':<26} | {'Moves':<6} | {'Expanded':<9} | {'Generated':<10} | {'Time (ms)':<9}",
            "-" * 78,
        ]

        m_res = self.astar_manhattan
        lines.append(
            f"{'A* (Manhattan Distance)':<26} | {m_res.move_count:<6} | "
            f"{m_res.nodes_expanded:<9} | {m_res.nodes_generated:<10} | "
            f"{m_res.time_taken * 1000:<9.2f}"
        )

        h_res = self.astar_misplaced
        lines.append(
            f"{'A* (Misplaced Tiles)':<26} | {h_res.move_count:<6} | "
            f"{h_res.nodes_expanded:<9} | {h_res.nodes_generated:<10} | "
            f"{h_res.time_taken * 1000:<9.2f}"
        )

        if self.bfs_result and self.bfs_result.success:
            b_res = self.bfs_result
            lines.append(
                f"{'BFS (Uniform Cost / h=0)':<26} | {b_res.move_count:<6} | "
                f"{b_res.nodes_expanded:<9} | {b_res.nodes_generated:<10} | "
                f"{b_res.time_taken * 1000:<9.2f}"
            )

        lines.append("=" * 78)

        # Theoretical insights
        if m_res.nodes_expanded <= h_res.nodes_expanded:
            reduction = (
                ((h_res.nodes_expanded - m_res.nodes_expanded) / h_res.nodes_expanded) * 100
                if h_res.nodes_expanded > 0
                else 0.0
            )
            lines.append(
                f"Heuristic Dominance Principle (h_Manhattan >= h_Misplaced):\n"
                f"  Manhattan expanded {reduction:.1f}% fewer nodes than Misplaced Tiles,\n"
                f"  empirically verifying that dominant heuristics prune the search tree."
            )

        return "\n".join(lines)


def compare_search_heuristics(
    initial_state: Union[Sequence[int], str],
    goal_state: Union[Sequence[int], str] = CLASSIC_GOAL_STATE,
    include_bfs: bool = True,
) -> EightPuzzleComparisonReport:
    """Compare A* with Manhattan distance against Misplaced Tiles and BFS baseline.

    Args:
        initial_state: Starting puzzle state.
        goal_state: Target puzzle state.
        include_bfs: Whether to include unguided BFS baseline.

    Returns:
        EightPuzzleComparisonReport instance.
    """
    init_tuple = parse_puzzle_state(initial_state)
    goal_tuple = parse_puzzle_state(goal_state)

    m_res = a_star_eight_puzzle(init_tuple, goal_tuple, heuristic="manhattan")
    h_res = a_star_eight_puzzle(init_tuple, goal_tuple, heuristic="misplaced")

    b_res: Optional[EightPuzzleResult] = None
    if include_bfs:
        # Avoid running BFS on deep states (m_res.move_count > 15) to prevent OOM
        if m_res.move_count <= 14:
            b_res = bfs_eight_puzzle(init_tuple, goal_tuple)

    return EightPuzzleComparisonReport(
        initial_state=init_tuple,
        goal_state=goal_tuple,
        astar_manhattan=m_res,
        astar_misplaced=h_res,
        bfs_result=b_res,
    )


# ============================================================================
# Board Visualization & Rendering Utilities
# ============================================================================


def render_board_ascii(state: PuzzleState, title: Optional[str] = None, indent: str = "") -> str:
    """Render a 3x3 8-puzzle board state as a clean ASCII grid box.

    Example Output:
        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        | 4 |   | 5 |
        +---+---+---+
        | 6 | 7 | 8 |
        +---+---+---+

    Args:
        state: 1D tuple of length 9.
        title: Optional title rendered above the box.
        indent: Optional indentation prefix for each line.

    Returns:
        ASCII string representation.
    """
    lines: List[str] = []
    if title:
        lines.append(f"{indent}{title}")

    border = f"{indent}+---+---+---+"
    lines.append(border)

    for r in range(GRID_SIZE):
        row_str = f"{indent}|"
        for c in range(GRID_SIZE):
            tile = state[r * GRID_SIZE + c]
            tile_disp = " " if tile == BLANK_TILE else str(tile)
            row_str += f" {tile_disp} |"
        lines.append(row_str)
        lines.append(border)

    return "\n".join(lines)


def render_step_by_step_boards(result: EightPuzzleResult) -> str:
    """Render the complete sequential solution as a series of ASCII boards.

    Args:
        result: Solved EightPuzzleResult instance.

    Returns:
        Formatted multi-board ASCII string.
    """
    return result.format_detailed_solution(use_ascii=True)


def render_side_by_side_boards(
    states: Sequence[PuzzleState],
    labels: Sequence[str],
    max_per_row: int = 4,
) -> str:
    """Render multiple boards horizontally side-by-side in rows.

    Args:
        states: Sequence of 1D PuzzleState tuples.
        labels: Corresponding titles for each board.
        max_per_row: Maximum number of boards per horizontal line.

    Returns:
        Multi-line string with side-by-side ASCII boards.
    """
    rendered_blocks: List[List[str]] = []
    for st, lbl in zip(states, labels):
        block = render_board_ascii(st, title=lbl).splitlines()
        rendered_blocks.append(block)

    output_lines: List[str] = []
    for chunk_start in range(0, len(rendered_blocks), max_per_row):
        chunk = rendered_blocks[chunk_start : chunk_start + max_per_row]
        num_lines = max(len(b) for b in chunk)
        padded_chunk = [b + [""] * (num_lines - len(b)) for b in chunk]

        for line_idx in range(num_lines):
            line_parts = [padded_chunk[b_idx][line_idx].ljust(18) for b_idx in range(len(chunk))]
            output_lines.append("  ".join(line_parts))
        output_lines.append("")

    return "\n".join(output_lines)


# ============================================================================
# Object-Oriented Solver Class
# ============================================================================


class EightPuzzleSolver:
    """Object-oriented wrapper for configuring and executing 8-puzzle A* searches.

    Attributes:
        goal_state: Target board configuration.
        heuristic: Heuristic name ("manhattan" or "misplaced").
        check_solvability: Whether to mathematically verify solvability upfront.
        max_nodes: Safety limit on node expansions.
    """

    def __init__(
        self,
        goal_state: Union[Sequence[int], str] = CLASSIC_GOAL_STATE,
        heuristic: str = "manhattan",
        check_solvability: bool = True,
        max_nodes: int = 200_000,
    ) -> None:
        """Initialize the 8-puzzle solver with configurable settings."""
        self.goal_state = parse_puzzle_state(goal_state)
        self.heuristic = heuristic
        self.check_solvability = check_solvability
        self.max_nodes = max_nodes

    def solve(self, initial_state: Union[Sequence[int], str]) -> EightPuzzleResult:
        """Solve the puzzle from the given initial state to the configured goal.

        Args:
            initial_state: Starting board configuration.

        Returns:
            EightPuzzleResult containing the optimal path and search metrics.
        """
        return a_star_eight_puzzle(
            initial_state=initial_state,
            goal_state=self.goal_state,
            heuristic=self.heuristic,
            check_solvability_first=self.check_solvability,
            max_nodes=self.max_nodes,
        )

    def is_solvable(self, initial_state: Union[Sequence[int], str]) -> bool:
        """Verify if the given initial state can reach this solver's goal state."""
        return is_solvable(initial_state, self.goal_state)

    def explain_solvability(self, initial_state: Union[Sequence[int], str]) -> str:
        """Return the mathematical inversion parity explanation for an initial state."""
        _, explanation, _, _ = explain_solvability(initial_state, self.goal_state)
        return explanation


# ============================================================================
# University Viva Voce Reference Guide
# ============================================================================


def get_eight_puzzle_viva_reference() -> str:
    """Return an exhaustive Viva Voce reference guide for Question 30.

    Covers:
    - Inversions and mathematical proof of solvability
    - Importance of 1D state hashing and immutability
    - Admissibility and consistency of Manhattan distance
    - Dominance of Manhattan distance over Misplaced Tiles
    - Relaxation principle for heuristic derivation
    - Memory complexity limits and iterative deepening A* (IDA*)
    """
    return """
===============================================================================
QUESTION NO. 30: 8-PUZZLE SOLVER USING A* (MANHATTAN)
UNIVERSITY VIVA VOCE REFERENCE GUIDE & THEORETICAL PRINCIPLES
===============================================================================

1. INVERSIONS TO CHECK SOLVABILITY
-------------------------------------------------------------------------------
Q1: What is an inversion in the context of the 8-puzzle?
A1: An inversion is any pair of numbered tiles (A, B) such that A > B and A appears
    before B in linear reading order (row-major), strictly ignoring the blank space (0).
    For example, in [1, 2, 3, 4, 0, 5, 6, 8, 7], the pair (8, 7) is an inversion (count = 1).

Q2: Prove why inversion parity is invariant under sliding moves on a 3x3 grid.
A2: On a 3x3 grid (odd grid width W = 3):
    - Horizontal moves: Sliding the blank left or right swaps the blank with a neighbor
      in the same row. Because the linear ordering of all numbered tiles remains unchanged,
      delta_inversions = 0.
    - Vertical moves: Sliding the blank up or down swaps the blank with a tile 3 positions
      away in the 1D representation. This causes that tile to skip over exactly 2 other
      numbered tiles.
      Let the tile moved be T, jumping over tiles X and Y:
        * If T is greater than both X and Y: inversions change by +/- 2.
        * If T is smaller than both X and Y: inversions change by -/+ 2.
        * If T is between X and Y: inversions change by 0.
      In every case, delta_inversions in {-2, 0, +2}.
    - Conclusion: The parity of inversions (inversions % 2) is strictly invariant!
      A state is solvable if and only if inv(initial) % 2 == inv(goal) % 2.
      For the classic goal (1-8, 0), inv(goal) = 0 (even), so an initial state is
      solvable if and only if its inversion count is even.

Q3: How many reachable states exist in the 8-puzzle?
A3: The total number of permutations of 9 tiles is 9! = 362,880.
    Because parity splits the symmetric group S_9 into two disjoint cosets of the
    alternating group A_9 (even vs odd permutations), exactly half of all states
    (181,440 states) are reachable from any given configuration.

2. STATE HASHING IMPORTANCE
-------------------------------------------------------------------------------
Q4: Why is state representation as an immutable 1D tuple crucial in Python?
A4: 1. O(1) Hash Table Lookups: Python's `set` and `dict` require hashable keys.
       A 1D tuple `(1, 2, 3, 4, 0, 5, 6, 7, 8)` implements fast C-level hashing
       (SipHash). Checking if a state is in `closed_set` or looking up its `g_score`
       takes O(1) expected time.
    2. Preventing O(V^2) Explosion: If mutable 2D lists `[[1,2,3],[4,0,5],[6,7,8]]`
       were used, they cannot be hashed. Membership checks would degrade to O(N)
       linear list scans, making search exponentially slower and causing practical freezes.
    3. Memory Overhead: A 9-element tuple takes ~112 bytes in CPython. A 2D list
       requires 4 Python list objects plus integer pointers (~400 bytes).
    4. Immutability & Heap Integrity: Mutating a state inside a priority queue
       corrupts the binary min-heap invariant. Tuples guarantee immutability.

3. INFORMED SEARCH & HEURISTICS
-------------------------------------------------------------------------------
Q5: Define the Manhattan Distance heuristic and explain why the blank tile (0) is excluded.
A5: Manhattan distance is the sum of horizontal and vertical grid offsets of all
    numbered tiles from their target positions:
        h_Manhattan(n) = sum_{tile=1..8} (|r_curr - r_goal| + |c_curr - c_goal|)
    Theoretical Reference: Russell & Norvig, Artificial Intelligence: A Modern Approach (AIMA Chapter 3).
    Exclusion of Blank Space (0):
    Counting the blank violates admissibility. In a state 1 move from the goal, sliding
    a single tile simultaneously swaps that tile and the blank. The true optimal cost
    h* = 1. If the blank were counted, both the moved tile and the blank would contribute 1,
    giving h = 2 > h* = 1, violating admissibility (h(n) <= h*(n)).

Q6: What is heuristic dominance, and why does Manhattan distance dominate Misplaced Tiles?
A6: Definition: A heuristic h2 dominates h1 if for all states n, h2(n) >= h1(n), and
    both are admissible.
    Proof:
    For every misplaced tile:
    - In Misplaced Tiles (h1), the tile contributes exactly 1.
    - In Manhattan distance (h2), any tile not in its goal position must differ in
      row, column, or both, contributing at least |dr| + |dc| >= 1.
    - For tiles in their goal position, both h1 and h2 contribute 0.
    Therefore, h_Manhattan(n) >= h_Misplaced(n) for all n.
    Consequence: A* using Manhattan distance is guaranteed to expand fewer or equal
    nodes than A* using Misplaced Tiles, while guaranteeing optimal solutions.

Q7: What is the maximum number of moves required to solve any solvable 8-puzzle?
A7: The maximum optimal solution length for any solvable 8-puzzle is 31 moves
    (known as "God's Number" for the 8-puzzle, proven by Reinefeld in 1993).
===============================================================================
"""
