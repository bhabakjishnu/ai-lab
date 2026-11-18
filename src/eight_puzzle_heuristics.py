"""8-Puzzle Heuristic Evaluator (Informed Search and Heuristics — Question 25).

This module implements heuristic evaluation functions for the classic 8-puzzle
sliding-tile problem on a 3x3 grid (with ``0`` denoting the blank space):

- **h1 (Misplaced Tiles / Hamming Distance)**: Counts the number of numbered
  tiles (1..8) that are not in their goal position, strictly ignoring the
  blank space (0).
- **h2 (Manhattan Distance / City-Block Distance)**: Computes the sum of
  orthogonal grid offsets ``|r_curr - r_goal| + |c_curr - c_goal|`` for every
  numbered tile (1..8) from its current position to its goal position, strictly
  ignoring the blank space (0).

Theoretical Background & Core AI Concepts:
- **Admissibility**: Both ``h1`` and ``h2`` are admissible heuristics
  (``0 <= h(n) <= h*(n)`` for every state ``n``), because each corresponds to
  the exact optimal solution cost of a relaxed version of the 8-puzzle:
    * ``h1`` relaxes both adjacency and empty-slot constraints (a tile may
      teleport directly to any square in 1 move).
    * ``h2`` relaxes only the empty-slot constraint (a tile may slide to any
      orthogonally adjacent square even if occupied).
- **Dominance**: For every 8-puzzle state ``n``, ``h2(n) >= h1(n)``. Each
  misplaced tile contributes ``1`` to ``h1`` and at least ``1`` to ``h2``.
  Therefore, ``h2`` dominates ``h1`` and guarantees that A* search using ``h2``
  expands fewer or equal nodes than A* using ``h1`` while preserving optimality.
- **Exclusion of Blank Space (0)**: Counting the blank tile violates
  admissibility. In a state 1 move from the goal, sliding a single tile swaps
  that tile and the blank simultaneously (true cost ``h* = 1``), yet counting
  the blank yields ``h1' = 2 > 1`` and ``h2' = 2 > 1``.
- **Solvability & Inversion Parity**: For any 3x3 board, horizontal blank moves
  leave the linear tile order unchanged and vertical blank moves shift one tile
  past 2 tiles (preserving inversion count parity). Thus, a state is reachable
  from a goal state if and only if both share the same inversion parity.

Complexity:
- Time Complexity: O(N) where N = 9 (single pass to index goal coordinates and
  single pass to evaluate all tiles).
- Space Complexity: O(N) where N = 9 (flattened representation and goal lookup
  dictionary).

Recommended Python Libraries:
- Pure Python standard library only (no third-party dependencies required).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# =====================================================================
# Constants & Canonical Board Presets
# =====================================================================

BLANK_TILE: int = 0
DEFAULT_GRID_SIZE: int = 3

BoardGrid = Sequence[Sequence[int]]
BoardTuple = Tuple[Tuple[int, ...], ...]

# Canonical Syllabus Sample (Russell & Norvig, AIMA Fig. 3.28; optimal h* = 26)
SAMPLE_CURRENT_STATE: BoardTuple = (
    (7, 2, 4),
    (5, 0, 6),
    (8, 3, 1),
)
SAMPLE_GOAL_STATE: BoardTuple = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
)

# Alternative Classic Goal (blank in bottom-right corner)
CLASSIC_GOAL_STATE: BoardTuple = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0),
)

# Benchmark & Pedagogical Presets (relative to SAMPLE_GOAL_STATE)
ONE_MOVE_STATE: BoardTuple = (
    (1, 0, 2),
    (3, 4, 5),
    (6, 7, 8),
)
EASY_STATE: BoardTuple = (
    (1, 2, 5),
    (3, 4, 0),
    (6, 7, 8),
)
UNSOLVABLE_STATE: BoardTuple = (
    (0, 2, 1),
    (3, 4, 5),
    (6, 7, 8),
)
HARD_STATE: BoardTuple = (
    (8, 7, 6),
    (5, 4, 3),
    (2, 1, 0),
)


# =====================================================================
# Structured Result Dataclasses
# =====================================================================


@dataclass(frozen=True)
class TileHeuristicDetail:
    """Per-tile heuristic breakdown for a single non-blank 8-puzzle tile.

    Attributes:
        tile: Integer tile label (1..8 for a 3x3 puzzle).
        current_pos: Current ``(row, col)`` coordinate obtained via ``divmod``.
        goal_pos: Target ``(row, col)`` coordinate in the goal state.
        is_misplaced: Whether ``current_pos != goal_pos`` (contributes 1 to h1).
        row_offset: Vertical Manhattan component ``|curr_row - goal_row|``.
        col_offset: Horizontal Manhattan component ``|curr_col - goal_col|``.
        manhattan_distance: Total orthogonal distance ``row_offset + col_offset``.
    """

    tile: int
    current_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    is_misplaced: bool
    row_offset: int
    col_offset: int
    manhattan_distance: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tile heuristic detail into a JSON-compatible dictionary."""
        return {
            "tile": self.tile,
            "current_pos": list(self.current_pos),
            "goal_pos": list(self.goal_pos),
            "is_misplaced": self.is_misplaced,
            "row_offset": self.row_offset,
            "col_offset": self.col_offset,
            "manhattan_distance": self.manhattan_distance,
        }


@dataclass(frozen=True)
class HeuristicEvaluationResult:
    """Complete evaluation report for an 8-puzzle state against a goal state.

    Attributes:
        current_state: Validated immutable 2D tuple representing the current board.
        goal_state: Validated immutable 2D tuple representing the goal board.
        h1_misplaced: Number of misplaced non-blank tiles (Hamming distance).
        h2_manhattan: Sum of Manhattan distances of all non-blank tiles.
        is_goal: True when ``current_state == goal_state`` (``h1 == h2 == 0``).
        is_solvable: True if ``current_state`` is reachable from ``goal_state``.
        current_inversions: Inversion count of non-blank tiles in ``current_state``.
        goal_inversions: Inversion count of non-blank tiles in ``goal_state``.
        dominance_holds: True when ``h2_manhattan >= h1_misplaced``.
        dominance_gap: Non-negative excess ``h2_manhattan - h1_misplaced``.
        tile_details: Ordered tuple of ``TileHeuristicDetail`` sorted by tile value.
    """

    current_state: BoardTuple
    goal_state: BoardTuple
    h1_misplaced: int
    h2_manhattan: int
    is_goal: bool
    is_solvable: bool
    current_inversions: int
    goal_inversions: int
    dominance_holds: bool
    dominance_gap: int
    tile_details: Tuple[TileHeuristicDetail, ...]

    def format_sample_output(self) -> str:
        """Return the exact syllabus sample output string.

        Returns:
            Formatted string ``"h1 (Misplaced): <h1_val>, h2 (Manhattan): <h2_val>"``.
        """
        return f"h1 (Misplaced): {self.h1_misplaced}, h2 (Manhattan): {self.h2_manhattan}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the evaluation result to a JSON-compatible dictionary."""
        return {
            "current_state": [list(row) for row in self.current_state],
            "goal_state": [list(row) for row in self.goal_state],
            "h1_misplaced": self.h1_misplaced,
            "h2_manhattan": self.h2_manhattan,
            "sample_output": self.format_sample_output(),
            "is_goal": self.is_goal,
            "is_solvable": self.is_solvable,
            "current_inversions": self.current_inversions,
            "goal_inversions": self.goal_inversions,
            "dominance_holds": self.dominance_holds,
            "dominance_gap": self.dominance_gap,
            "tile_details": [detail.to_dict() for detail in self.tile_details],
        }


# =====================================================================
# Board Validation, Flattening & Index Math (divmod)
# =====================================================================


def validate_board(
    board: BoardGrid,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> BoardTuple:
    """Validate an NxN sliding-puzzle board and convert it to an immutable tuple.

    Ensures the matrix has exact dimensions ``expected_size x expected_size``
    and contains each integer tile from ``0`` through ``expected_size**2 - 1``
    exactly once (rejecting booleans, non-integers, ragged rows, duplicates, or
    missing tiles).

    Args:
        board: 2D sequence of integers representing the puzzle grid.
        expected_size: Side length of the square puzzle (default: 3 for 8-puzzle).

    Returns:
        Canonical immutable ``BoardTuple`` of shape ``(expected_size, expected_size)``.

    Raises:
        TypeError: If ``board``, any row, or any tile has an invalid type.
        ValueError: If dimensions are wrong or tile values are not a valid permutation.
    """
    if expected_size < 2:
        raise ValueError(f"Puzzle grid size must be at least 2, got {expected_size}.")

    if not isinstance(board, Sequence) or isinstance(board, (str, bytes)):
        raise TypeError(
            f"Board must be a 2D sequence of integers, got {type(board).__name__}."
        )

    if len(board) != expected_size:
        raise ValueError(
            f"Expected a {expected_size}x{expected_size} board, "
            f"but received {len(board)} row(s)."
        )

    normalized_rows: List[Tuple[int, ...]] = []
    seen_tiles: set[int] = set()
    total_cells = expected_size * expected_size

    for row_idx, row in enumerate(board):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(
                f"Row {row_idx} must be a sequence of integers, got {type(row).__name__}."
            )
        if len(row) != expected_size:
            raise ValueError(
                f"Row {row_idx} must contain exactly {expected_size} elements, "
                f"got {len(row)}."
            )

        clean_row: List[int] = []
        for col_idx, tile in enumerate(row):
            if isinstance(tile, bool) or not isinstance(tile, int):
                raise TypeError(
                    f"Tile at ({row_idx}, {col_idx}) must be an int, "
                    f"got {type(tile).__name__}: {tile!r}."
                )
            if tile < 0 or tile >= total_cells:
                raise ValueError(
                    f"Tile value {tile} at ({row_idx}, {col_idx}) is out of range "
                    f"[0, {total_cells - 1}]."
                )
            if tile in seen_tiles:
                raise ValueError(
                    f"Duplicate tile value {tile} found at ({row_idx}, {col_idx})."
                )
            seen_tiles.add(tile)
            clean_row.append(tile)

        normalized_rows.append(tuple(clean_row))

    expected_tiles = set(range(total_cells))
    if seen_tiles != expected_tiles:
        missing = sorted(expected_tiles - seen_tiles)
        raise ValueError(f"Board is missing required tile(s): {missing}.")

    return tuple(normalized_rows)


def flatten_board(board: BoardGrid, expected_size: int = DEFAULT_GRID_SIZE) -> Tuple[int, ...]:
    """Flatten a validated 2D puzzle matrix into a 1D row-major tuple.

    Time Complexity: O(N), where N = size^2 (9 for 8-puzzle).
    Space Complexity: O(N).

    Args:
        board: 2D matrix representing the puzzle state.
        expected_size: Expected side length of the square grid (default: 3).

    Returns:
        1D tuple of length ``expected_size**2`` in row-major order.
    """
    validated = validate_board(board, expected_size=expected_size)
    return tuple(tile for row in validated for tile in row)


def unflatten_board(
    flat: Sequence[int],
    size: int = DEFAULT_GRID_SIZE,
) -> BoardTuple:
    """Reconstruct a validated 2D square puzzle board from a 1D sequence.

    Args:
        flat: Flat sequence of ``size**2`` integers in row-major order.
        size: Side length of the square grid (default: 3).

    Returns:
        Validated immutable ``BoardTuple``.

    Raises:
        TypeError: If ``flat`` is not a sequence of integers.
        ValueError: If length != ``size**2`` or tiles are not a valid permutation.
    """
    if not isinstance(flat, Sequence) or isinstance(flat, (str, bytes)):
        raise TypeError(f"Flat board must be a sequence, got {type(flat).__name__}.")
    if len(flat) != size * size:
        raise ValueError(
            f"Expected {size * size} tiles for a {size}x{size} board, got {len(flat)}."
        )
    matrix = tuple(tuple(flat[r * size : (r + 1) * size]) for r in range(size))
    return validate_board(matrix, expected_size=size)


def index_to_coords(index: int, size: int = DEFAULT_GRID_SIZE) -> Tuple[int, int]:
    """Convert a 1D row-major flat index into 2D ``(row, col)`` coordinates using ``divmod``.

    Why ``divmod``?
        ``divmod(index, size)`` computes ``(index // size, index % size)`` in a
        single C-level operation, cleanly mapping flat array index ``0..8`` into
        3x3 grid coordinates ``(0, 0)..(2, 2)``.

    Args:
        index: Zero-based linear index in ``[0, size**2 - 1]``.
        size: Side length of the square grid (default: 3).

    Returns:
        Tuple ``(row, col)``.

    Raises:
        ValueError: If ``size < 1`` or ``index`` is outside ``[0, size**2 - 1]``.
    """
    if size < 1:
        raise ValueError(f"Grid size must be positive, got {size}.")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"Index must be an integer, got {type(index).__name__}.")
    if index < 0 or index >= size * size:
        raise ValueError(
            f"Flat index {index} out of bounds for {size}x{size} grid [0, {size * size - 1}]."
        )
    return divmod(index, size)


def build_position_map(
    board: BoardGrid,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> Dict[int, Tuple[int, int]]:
    """Build an O(1) lookup table mapping each tile value to its ``(row, col)`` position.

    Uses ``flatten_board`` and ``divmod`` so that subsequent coordinate lookups
    for Manhattan distance run in constant O(1) time per tile, keeping overall
    heuristic computation strictly O(N).

    Args:
        board: 2D puzzle matrix.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        Dictionary mapping each tile integer ``0..(N-1)`` to ``(row, col)``.
    """
    flat = flatten_board(board, expected_size=expected_size)
    return {tile: divmod(idx, expected_size) for idx, tile in enumerate(flat)}


def parse_board(text: str, expected_size: int = DEFAULT_GRID_SIZE) -> BoardTuple:
    """Parse a string representation of an 8-puzzle board into a validated ``BoardTuple``.

    Supported formats:
    1. Python nested list literal: ``"[[7,2,4],[5,0,6],[8,3,1]]"``
    2. Semicolon-separated rows: ``"7 2 4; 5 0 6; 8 3 1"`` or ``"7,2,4;5,0,6;8,3,1"``
    3. Flat sequence of integers: ``"7 2 4 5 0 6 8 3 1"`` or ``"7,2,4,5,0,6,8,3,1"``
    4. Named goal presets: ``"standard"`` / ``"zero-first"``, ``"classic"`` / ``"zero-last"``,
       ``"sample"``

    Args:
        text: User-supplied board specification string.
        expected_size: Expected side length of the square grid (default: 3).

    Returns:
        Validated immutable ``BoardTuple``.

    Raises:
        ValueError: If the string cannot be parsed into a valid puzzle board.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Board input string must be non-empty.")

    cleaned = text.strip()
    lowered = cleaned.lower()

    if expected_size == 3:
        presets: Dict[str, BoardTuple] = {
            "sample": SAMPLE_CURRENT_STATE,
            "standard": SAMPLE_GOAL_STATE,
            "zero-first": SAMPLE_GOAL_STATE,
            "goal": SAMPLE_GOAL_STATE,
            "classic": CLASSIC_GOAL_STATE,
            "zero-last": CLASSIC_GOAL_STATE,
            "easy": EASY_STATE,
            "one-move": ONE_MOVE_STATE,
            "unsolvable": UNSOLVABLE_STATE,
            "hard": HARD_STATE,
        }
        if lowered in presets:
            return presets[lowered]

    # 1. Try literal Python matrix syntax (safe ast.literal_eval)
    if cleaned.startswith("[") or cleaned.startswith("("):
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid matrix literal syntax: {text!r}") from exc
        if isinstance(parsed, (list, tuple)) and parsed and isinstance(parsed[0], (list, tuple)):
            return validate_board(parsed, expected_size=expected_size)
        if isinstance(parsed, (list, tuple)):
            return unflatten_board(parsed, size=expected_size)
        raise ValueError(f"Unsupported literal structure for board: {text!r}")

    # 2. Semicolon-separated rows
    if ";" in cleaned:
        row_strs = [segment.strip() for segment in cleaned.split(";") if segment.strip()]
        rows: List[List[int]] = []
        for row_str in row_strs:
            tokens = [tok for tok in re.split(r"[\s,]+", row_str) if tok]
            try:
                rows.append([int(tok) for tok in tokens])
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer token found in row {row_str!r}."
                ) from exc
        return validate_board(rows, expected_size=expected_size)

    # 3. Flat space/comma separated integers
    tokens = [tok for tok in re.split(r"[\s,]+", cleaned) if tok]
    try:
        flat_values = [int(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError(f"Unable to parse integers from board string: {text!r}") from exc

    return unflatten_board(flat_values, size=expected_size)


# =====================================================================
# Core Heuristic Algorithms: h1 (Misplaced Tiles) & h2 (Manhattan)
# =====================================================================


def misplaced_tiles_heuristic(
    current: BoardGrid,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> int:
    """Calculate ``h1``: the number of misplaced tiles (ignoring the blank space ``0``).

    Algorithm:
        1. Flatten both ``current`` and ``goal`` matrices into 1D sequences of length N.
        2. Iterate over paired positions ``(curr_tile, goal_tile)``.
        3. Skip ``curr_tile == 0`` (the blank space must never be counted).
        4. Increment the count whenever ``curr_tile != goal_tile``.

    Complexity:
        - Time: O(N), where N = 9 for a 3x3 8-puzzle.
        - Space: O(N) for flattened tuples.

    Args:
        current: 3x3 (or NxN) matrix representing the current puzzle state.
        goal: 3x3 (or NxN) matrix representing the target goal state.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        Non-negative integer count of misplaced non-blank tiles.
    """
    flat_current = flatten_board(current, expected_size=expected_size)
    flat_goal = flatten_board(goal, expected_size=expected_size)

    return sum(
        1
        for curr_tile, goal_tile in zip(flat_current, flat_goal)
        if curr_tile != BLANK_TILE and curr_tile != goal_tile
    )


def manhattan_distance_heuristic(
    current: BoardGrid,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> int:
    """Calculate ``h2``: sum of Manhattan distances of tiles from goal positions (ignoring ``0``).

    Algorithm:
        1. Precompute a hash map ``goal_positions[tile] = divmod(idx, size)`` in O(N).
        2. Flatten ``current`` and iterate with ``enumerate``.
        3. Skip ``tile == 0`` (the blank space must never be counted).
        4. Compute current coordinates ``(curr_r, curr_c) = divmod(idx, size)``.
        5. Accumulate ``|curr_r - goal_r| + |curr_c - goal_c|``.

    Complexity:
        - Time: O(N), where N = 9 for a 3x3 8-puzzle (avoids O(N^2) nested scanning).
        - Space: O(N) for the goal position lookup map.

    Args:
        current: 3x3 (or NxN) matrix representing the current puzzle state.
        goal: 3x3 (or NxN) matrix representing the target goal state.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        Non-negative integer sum of Manhattan distances across all non-blank tiles.
    """
    flat_current = flatten_board(current, expected_size=expected_size)
    goal_positions = build_position_map(goal, expected_size=expected_size)

    total_distance = 0
    for idx, tile in enumerate(flat_current):
        if tile == BLANK_TILE:
            continue
        curr_row, curr_col = divmod(idx, expected_size)
        goal_row, goal_col = goal_positions[tile]
        total_distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)

    return total_distance


# =====================================================================
# Solvability Analysis & Optional Challenge Generator
# =====================================================================


def count_inversions(
    board: BoardGrid,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> int:
    """Count the number of inversions among non-blank tiles in row-major order.

    An inversion is a pair of indices ``(i, j)`` with ``i < j`` such that
    ``tiles[i] > tiles[j]`` (after removing the blank space ``0``).

    Args:
        board: 2D puzzle matrix.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        Non-negative integer inversion count.
    """
    flat = flatten_board(board, expected_size=expected_size)
    non_blank = [tile for tile in flat if tile != BLANK_TILE]

    inversions = 0
    num_tiles = len(non_blank)
    for i in range(num_tiles):
        tile_i = non_blank[i]
        for j in range(i + 1, num_tiles):
            if tile_i > non_blank[j]:
                inversions += 1
    return inversions


def _solvability_invariant(board: BoardTuple) -> int:
    """Compute the parity invariant (0 or 1) determining the connected component of a board.

    - For odd grid width (e.g., 3x3 8-puzzle): ``inversions % 2``.
    - For even grid width (e.g., 4x4 15-puzzle): ``(inversions + blank_row_from_bottom) % 2``.
    """
    size = len(board)
    inversions = count_inversions(board, expected_size=size)
    if size % 2 == 1:
        return inversions % 2

    blank_row = next(r for r, row in enumerate(board) if BLANK_TILE in row)
    blank_row_from_bottom = size - blank_row
    return (inversions + blank_row_from_bottom) % 2


def is_solvable(
    current: BoardGrid,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> bool:
    """Determine whether ``current`` can be legally transformed into ``goal``.

    The state space of the 8-puzzle partitions into two disjoint connected
    components of equal size (``9! / 2 = 181,440`` states each). Two 3x3
    configurations belong to the same reachable component if and only if they
    have identical inversion parity.

    Args:
        current: Starting puzzle configuration.
        goal: Target goal configuration (default: ``SAMPLE_GOAL_STATE``).
        expected_size: Side length of the square grid (default: 3).

    Returns:
        True if a finite sequence of legal tile slides connects ``current`` to ``goal``.
    """
    validated_current = validate_board(current, expected_size=expected_size)
    validated_goal = validate_board(goal, expected_size=expected_size)
    return _solvability_invariant(validated_current) == _solvability_invariant(validated_goal)


def generate_random_solvable_board(
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    seed: Optional[int] = None,
    ensure_non_goal: bool = True,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> BoardTuple:
    """Optional Challenge: Generate a uniformly random solvable puzzle configuration.

    Algorithm:
        1. Create a random permutation of ``0..(N-1)``.
        2. Check solvability parity against ``goal``.
        3. If parity differs, swap any two non-blank tiles. Because every
           transposition flips permutation parity, this creates an exact
           2-to-1 bijection onto the reachable half of the state space,
           guaranteeing uniform sampling over all ``181,440`` solvable 8-puzzle
           boards without rejection loops.

    Args:
        goal: Target goal configuration relative to which solvability is guaranteed.
        seed: Optional random seed for deterministic reproducibility.
        ensure_non_goal: If True, avoids returning a board identical to ``goal``.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        A validated, guaranteed-solvable ``BoardTuple``.
    """
    validated_goal = validate_board(goal, expected_size=expected_size)
    rng = random.Random(seed)
    total_cells = expected_size * expected_size

    while True:
        tiles = list(range(total_cells))
        rng.shuffle(tiles)
        candidate = unflatten_board(tiles, size=expected_size)

        if not is_solvable(candidate, validated_goal, expected_size=expected_size):
            # Swap the first two non-blank tiles to flip inversion parity
            non_blank_indices = [i for i, t in enumerate(tiles) if t != BLANK_TILE]
            i0, i1 = non_blank_indices[0], non_blank_indices[1]
            tiles[i0], tiles[i1] = tiles[i1], tiles[i0]
            candidate = unflatten_board(tiles, size=expected_size)

        if not ensure_non_goal or candidate != validated_goal:
            return candidate


# =====================================================================
# Full Evaluator Function & OOP Evaluator Class
# =====================================================================


def evaluate_heuristics(
    current: BoardGrid,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    expected_size: int = DEFAULT_GRID_SIZE,
) -> HeuristicEvaluationResult:
    """Evaluate both ``h1`` and ``h2`` along with detailed per-tile and theoretical diagnostics.

    Args:
        current: 2D matrix representing the current puzzle state.
        goal: 2D matrix representing the target goal state.
        expected_size: Side length of the square grid (default: 3).

    Returns:
        ``HeuristicEvaluationResult`` containing ``h1_misplaced``, ``h2_manhattan``,
        solvability metadata, dominance verification, and per-tile offsets.
    """
    validated_current = validate_board(current, expected_size=expected_size)
    validated_goal = validate_board(goal, expected_size=expected_size)

    flat_current = flatten_board(validated_current, expected_size=expected_size)
    goal_positions = build_position_map(validated_goal, expected_size=expected_size)

    h1 = 0
    h2 = 0
    details: List[TileHeuristicDetail] = []

    for idx, tile in enumerate(flat_current):
        if tile == BLANK_TILE:
            continue

        curr_row, curr_col = divmod(idx, expected_size)
        goal_row, goal_col = goal_positions[tile]
        row_offset = abs(curr_row - goal_row)
        col_offset = abs(curr_col - goal_col)
        dist = row_offset + col_offset
        misplaced = (curr_row, curr_col) != (goal_row, goal_col)

        if misplaced:
            h1 += 1
        h2 += dist

        details.append(
            TileHeuristicDetail(
                tile=tile,
                current_pos=(curr_row, curr_col),
                goal_pos=(goal_row, goal_col),
                is_misplaced=misplaced,
                row_offset=row_offset,
                col_offset=col_offset,
                manhattan_distance=dist,
            )
        )

    details.sort(key=lambda item: item.tile)

    curr_inv = count_inversions(validated_current, expected_size=expected_size)
    goal_inv = count_inversions(validated_goal, expected_size=expected_size)
    solvable = _solvability_invariant(validated_current) == _solvability_invariant(validated_goal)

    return HeuristicEvaluationResult(
        current_state=validated_current,
        goal_state=validated_goal,
        h1_misplaced=h1,
        h2_manhattan=h2,
        is_goal=(validated_current == validated_goal),
        is_solvable=solvable,
        current_inversions=curr_inv,
        goal_inversions=goal_inv,
        dominance_holds=(h2 >= h1),
        dominance_gap=(h2 - h1),
        tile_details=tuple(details),
    )


class EightPuzzleHeuristicEvaluator:
    """Reusable OOP evaluator for 8-puzzle heuristic computations against a fixed goal.

    Precomputes the goal state's tile-to-coordinate lookup map once at initialization
    so repeated evaluations during A* / IDA* search avoid rebuilding the goal map.

    Attributes:
        goal_state: Validated immutable target board.
        size: Side length of the square puzzle (default: 3).
    """

    def __init__(
        self,
        goal_state: BoardGrid = SAMPLE_GOAL_STATE,
        expected_size: int = DEFAULT_GRID_SIZE,
    ) -> None:
        """Initialize the evaluator with a validated goal state.

        Args:
            goal_state: Target 3x3 (or NxN) puzzle matrix.
            expected_size: Side length of the square grid (default: 3).
        """
        self.size: int = expected_size
        self.goal_state: BoardTuple = validate_board(goal_state, expected_size=expected_size)
        self._goal_flat: Tuple[int, ...] = flatten_board(
            self.goal_state, expected_size=expected_size
        )
        self._goal_positions: Dict[int, Tuple[int, int]] = build_position_map(
            self.goal_state, expected_size=expected_size
        )

    def h1(self, state: BoardGrid) -> int:
        """Compute ``h1`` (misplaced non-blank tiles) for ``state``."""
        flat_state = flatten_board(state, expected_size=self.size)
        return sum(
            1
            for curr_tile, goal_tile in zip(flat_state, self._goal_flat)
            if curr_tile != BLANK_TILE and curr_tile != goal_tile
        )

    def h2(self, state: BoardGrid) -> int:
        """Compute ``h2`` (Manhattan distance sum of non-blank tiles) for ``state``."""
        flat_state = flatten_board(state, expected_size=self.size)
        total = 0
        for idx, tile in enumerate(flat_state):
            if tile == BLANK_TILE:
                continue
            curr_r, curr_c = divmod(idx, self.size)
            goal_r, goal_c = self._goal_positions[tile]
            total += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return total

    def evaluate(self, state: BoardGrid) -> HeuristicEvaluationResult:
        """Return a complete ``HeuristicEvaluationResult`` for ``state``."""
        return evaluate_heuristics(state, self.goal_state, expected_size=self.size)

    def evaluate_batch(
        self,
        states: Iterable[BoardGrid],
    ) -> List[HeuristicEvaluationResult]:
        """Evaluate a sequence or iterable of puzzle states against the goal."""
        return [self.evaluate(state) for state in states]

    def is_solvable(self, state: BoardGrid) -> bool:
        """Check whether ``state`` can legally reach ``self.goal_state``."""
        return is_solvable(state, self.goal_state, expected_size=self.size)

    def generate_random_solvable_board(
        self,
        seed: Optional[int] = None,
        ensure_non_goal: bool = True,
    ) -> BoardTuple:
        """Generate a uniformly random board solvable to ``self.goal_state``."""
        return generate_random_solvable_board(
            goal=self.goal_state,
            seed=seed,
            ensure_non_goal=ensure_non_goal,
            expected_size=self.size,
        )


# =====================================================================
# Theoretical Demonstrations (Admissibility & Dominance)
# =====================================================================


def demonstrate_blank_space_inadmissibility(
    state: BoardGrid = ONE_MOVE_STATE,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    true_optimal_cost: int = 1,
) -> Dict[str, Any]:
    """Demonstrate why counting the blank space (0) destroys heuristic admissibility.

    Consider ``state = [[1, 0, 2], [3, 4, 5], [6, 7, 8]]`` and
    ``goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]``. Exactly ``1`` legal move
    (sliding tile ``1`` right into the blank) solves the puzzle (``h* = 1``).
    Ignoring blank ``0`` yields ``h1 = 1 <= 1`` and ``h2 = 1 <= 1`` (admissible).
    Erroneously counting blank ``0`` yields ``h1_with_blank = 2 > 1`` and
    ``h2_with_blank = 2 > 1`` (inadmissible overestimation!).

    Args:
        state: State to evaluate (default: 1-move state ``ONE_MOVE_STATE``).
        goal: Target goal state (default: ``SAMPLE_GOAL_STATE``).
        true_optimal_cost: Known true optimal solution length ``h*(state)``.

    Returns:
        Dictionary comparing valid heuristics (blank excluded) against flawed
        heuristics (blank included) and reporting admissibility flags.
    """
    validated_state = validate_board(state)
    validated_goal = validate_board(goal)
    flat_state = flatten_board(validated_state)
    flat_goal = flatten_board(validated_goal)
    goal_pos = build_position_map(validated_goal)

    h1_valid = misplaced_tiles_heuristic(validated_state, validated_goal)
    h2_valid = manhattan_distance_heuristic(validated_state, validated_goal)

    h1_with_blank = sum(
        1 for curr_t, goal_t in zip(flat_state, flat_goal) if curr_t != goal_t
    )
    h2_with_blank = sum(
        abs(divmod(idx, 3)[0] - goal_pos[tile][0])
        + abs(divmod(idx, 3)[1] - goal_pos[tile][1])
        for idx, tile in enumerate(flat_state)
    )

    return {
        "state": validated_state,
        "goal": validated_goal,
        "true_optimal_cost": true_optimal_cost,
        "h1_valid": h1_valid,
        "h2_valid": h2_valid,
        "h1_valid_admissible": h1_valid <= true_optimal_cost,
        "h2_valid_admissible": h2_valid <= true_optimal_cost,
        "h1_with_blank": h1_with_blank,
        "h2_with_blank": h2_with_blank,
        "h1_with_blank_admissible": h1_with_blank <= true_optimal_cost,
        "h2_with_blank_admissible": h2_with_blank <= true_optimal_cost,
    }


def analyze_dominance_over_samples(
    num_samples: int = 50,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Empirically verify ``h2 >= h1`` dominance across random solvable 8-puzzle boards.

    Args:
        num_samples: Number of random solvable boards to generate and evaluate.
        goal: Target goal board (default: ``SAMPLE_GOAL_STATE``).
        seed: Optional random seed for reproducible sampling.

    Returns:
        Summary dictionary with dominance verification rates, averages, ranges,
        and evaluated ``HeuristicEvaluationResult`` list.

    Raises:
        ValueError: If ``num_samples < 1``.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    validated_goal = validate_board(goal)
    rng = random.Random(seed)
    evaluator = EightPuzzleHeuristicEvaluator(validated_goal)

    results: List[HeuristicEvaluationResult] = []
    for _ in range(num_samples):
        board_seed = rng.randint(0, 10**9)
        board = evaluator.generate_random_solvable_board(seed=board_seed, ensure_non_goal=True)
        results.append(evaluator.evaluate(board))

    h1_values = [r.h1_misplaced for r in results]
    h2_values = [r.h2_manhattan for r in results]
    gaps = [r.dominance_gap for r in results]

    all_dominate = all(r.dominance_holds for r in results)
    strict_count = sum(1 for r in results if r.h2_manhattan > r.h1_misplaced)

    return {
        "num_samples": num_samples,
        "all_solvable": all(r.is_solvable for r in results),
        "all_h2_dominates_h1": all_dominate,
        "strict_dominance_count": strict_count,
        "strict_dominance_pct": (strict_count / num_samples) * 100.0,
        "mean_h1": sum(h1_values) / num_samples,
        "mean_h2": sum(h2_values) / num_samples,
        "mean_dominance_gap": sum(gaps) / num_samples,
        "min_h1": min(h1_values),
        "max_h1": max(h1_values),
        "min_h2": min(h2_values),
        "max_h2": max(h2_values),
        "results": results,
    }


# =====================================================================
# ASCII Board Visualization & Formatting Utilities
# =====================================================================


def render_board_ascii(board: BoardGrid, title: Optional[str] = None) -> str:
    """Render a 3x3 (or NxN) 8-puzzle board as a clean boxed ASCII grid.

    The blank tile (``0``) is rendered as ``_`` for clear visual distinction.

    Args:
        board: 2D puzzle matrix.
        title: Optional header label placed above the grid.

    Returns:
        Multi-line ASCII string representing the puzzle grid.
    """
    validated = validate_board(board, expected_size=len(board))
    size = len(validated)
    border = "+" + "+".join(["---"] * size) + "+"

    lines: List[str] = []
    if title:
        lines.append(title)
    lines.append(border)
    for row in validated:
        cells = [f" {'_' if tile == BLANK_TILE else tile} " for tile in row]
        lines.append("|" + "|".join(cells) + "|")
        lines.append(border)
    return "\n".join(lines)


def render_side_by_side_boards(
    current: BoardGrid,
    goal: BoardGrid = SAMPLE_GOAL_STATE,
) -> str:
    """Render Current State and Goal State side by side with a transition arrow.

    Args:
        current: Current 3x3 puzzle matrix.
        goal: Goal 3x3 puzzle matrix.

    Returns:
        Formatted multi-line ASCII comparison block.
    """
    curr_lines = render_board_ascii(current, title="Current State").splitlines()
    goal_lines = render_board_ascii(goal, title="Goal State").splitlines()

    width = max(len(line) for line in curr_lines)
    mid_idx = len(curr_lines) // 2

    combined: List[str] = []
    for idx, (left, right) in enumerate(zip(curr_lines, goal_lines)):
        connector = "  --->  " if idx == mid_idx else "        "
        combined.append(f"{left:<{width}}{connector}{right}")
    return "\n".join(combined)


def format_tile_breakdown_table(result: HeuristicEvaluationResult) -> str:
    """Format a clean ASCII table decomposing ``h1`` and ``h2`` tile by tile.

    Args:
        result: ``HeuristicEvaluationResult`` from ``evaluate_heuristics``.

    Returns:
        Formatted multi-line table string with totals and blank-exclusion note.
    """
    sep = "+------+-------------+----------+------+------+------------+---------------+"
    header = "| Tile | Current Pos | Goal Pos | |dr| | |dc| | h1 (Misp.) | h2 (Manhattan)|"
    lines = [sep, header, sep]

    for detail in result.tile_details:
        misp_int = 1 if detail.is_misplaced else 0
        lines.append(
            f"|  {detail.tile:<3} | {str(detail.current_pos):<11} | "
            f"{str(detail.goal_pos):<8} |  {detail.row_offset:<3} |  "
            f"{detail.col_offset:<3} |     {misp_int:<6} |      {detail.manhattan_distance:<8} |"
        )

    lines.append(sep)
    lines.append(
        f"| TOTAL (Blank '0' Excluded)                 |     "
        f"{result.h1_misplaced:<6} |      {result.h2_manhattan:<8} |"
    )
    lines.append(sep)
    return "\n".join(lines)


def format_evaluation_report(result: HeuristicEvaluationResult) -> str:
    """Build a complete human-readable report for an 8-puzzle heuristic evaluation.

    Args:
        result: ``HeuristicEvaluationResult`` object.

    Returns:
        Multi-line formatted report suitable for CLI and notebook output.
    """
    solvable_str = "SOLVABLE" if result.is_solvable else "UNSOLVABLE (Opposite Inversion Parity)"
    dominance_str = (
        f"VERIFIED (h2={result.h2_manhattan} >= h1={result.h1_misplaced}, gap=+{result.dominance_gap})"
        if result.dominance_holds
        else "VIOLATED"
    )

    lines = [
        "=" * 76,
        "QUESTION NO. 25: 8-PUZZLE HEURISTIC EVALUATOR (h1 & h2)",
        "=" * 76,
        "",
        render_side_by_side_boards(result.current_state, result.goal_state),
        "",
        f"Result           : {result.format_sample_output()}",
        f"Goal Reached?    : {'YES (h1 = h2 = 0)' if result.is_goal else 'NO'}",
        f"Solvability      : {solvable_str}",
        f"Inversion Counts : Current = {result.current_inversions}, Goal = {result.goal_inversions}",
        f"Dominance Check  : {dominance_str}",
        "",
        "PER-TILE HEURISTIC DECOMPOSITION (divmod Coordinate Math):",
        format_tile_breakdown_table(result),
        "=" * 76,
    ]
    return "\n".join(lines)


# =====================================================================
# University Viva Voce Reference Guide
# =====================================================================


def get_eight_puzzle_viva_reference() -> Dict[str, Dict[str, str]]:
    """Return structured Viva Voce Q&A covering all Question 25 syllabus topics.

    Topics Covered:
        1. Dominance of heuristics (``h2`` dominates ``h1``)
        2. Admissibility and consistency of ``h1`` and ``h2``
        3. Why the blank space (``0``) must not be counted
        4. Solvability and inversion count parity in the 8-puzzle
        5. Algorithmic complexity and ``divmod`` index math

    Returns:
        Dictionary mapping topic keys to structured Q&A entries.
    """
    return {
        "dominance_of_heuristics": {
            "question": "What does it mean that h2 (Manhattan distance) dominates h1 (Misplaced tiles), and why does it matter in A* search?",
            "answer": (
                "An admissible heuristic h2 dominates an admissible heuristic h1 if h2(n) >= h1(n) "
                "for every state n in the state space (with strict inequality h2(n) > h1(n) for at "
                "least some states).\n"
                "Proof for the 8-puzzle: Any tile that is already in its goal cell contributes 0 to "
                "both h1 and h2. Any misplaced tile contributes exactly 1 to h1, whereas its Manhattan "
                "distance |r_curr - r_goal| + |c_curr - c_goal| is at least 1 (since either row or "
                "column differs by at least 1). Summing across all 8 numbered tiles yields h2(n) >= h1(n).\n"
                "Impact on A*: Because h2 is strictly closer to the true cost h*(n) without overestimating, "
                "every node expanded by A* with h2 (having f2(n) < C*) must also be expanded by A* with h1 "
                "(since f1(n) <= f2(n) < C*). Thus, A* with h2 expands dramatically fewer nodes while "
                "preserving optimality."
            ),
            "key_formula": "0 <= h1(n) <= h2(n) <= h*(n) for all states n",
        },
        "admissibility_of_h1_and_h2": {
            "question": "Why are both h1 and h2 guaranteed to be admissible (and consistent) heuristics for the 8-puzzle?",
            "answer": (
                "A heuristic h(n) is admissible if it never overestimates the true minimum number of moves "
                "h*(n) to reach the goal: 0 <= h(n) <= h*(n). Both h1 and h2 are derived as exact costs "
                "of relaxed versions of the 8-puzzle:\n"
                "  • Standard rule: A tile can move from cell A to cell B iff A and B are orthogonally "
                "adjacent AND B is blank.\n"
                "  • Relaxation for h1: Remove both restrictions (a tile can move anywhere in 1 step). "
                "Each misplaced tile needs at least 1 move, so h1(n) <= h*(n).\n"
                "  • Relaxation for h2: Remove only the blank-cell restriction (a tile can slide to any "
                "adjacent cell in 1 step). Each tile must take at least |dr| + |dc| orthogonal steps, "
                "and each legal move moves only one numbered tile by 1 step, so h2(n) <= h*(n).\n"
                "Consistency: Every single legal move shifts exactly one numbered tile by 1 cell, so "
                "h1 and h2 can decrease by at most 1 per step: h(n) <= 1 + h(n')."
            ),
            "key_property": "Relaxed-problem exact costs are always both admissible and consistent.",
        },
        "why_blank_space_is_not_counted": {
            "question": "Why must the blank space (0) NEVER be counted in h1 or h2?",
            "answer": (
                "Counting the blank space (0) violates admissibility by double-counting moves.\n"
                "Counterexample: Consider Current = [[1, 0, 2], [3, 4, 5], [6, 7, 8]] and "
                "Goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. The true optimal cost is h* = 1 move "
                "(slide tile 1 left into the blank).\n"
                "  • Ignoring blank (0): h1 = 1, h2 = 1 -> Admissible (1 <= 1).\n"
                "  • Counting blank (0): Both tile 1 and blank 0 are out of place at Manhattan distance 1, "
                "yielding h1' = 2 > 1 and h2' = 2 > 1 -> INADMISSIBLE!\n"
                "Physically, the blank space is not a tile; it is the absence of a tile. Every move "
                "relocates one numbered tile and simultaneously shifts the blank as a side effect, so "
                "counting the blank double-counts every move and can cause A* to return suboptimal paths."
            ),
            "counterexample": "[[1,0,2],[3,4,5],[6,7,8]] -> h*=1, valid h1=h2=1, with-blank h1'=h2'=2 (INADMISSIBLE)",
        },
        "solvability_and_inversion_parity": {
            "question": "How do we check if a randomly generated 8-puzzle board is solvable without running a search?",
            "answer": (
                "Flatten the 3x3 board in row-major order and remove the blank (0). Count inversions: "
                "pairs (i, j) with i < j where tile[i] > tile[j].\n"
                "Because the grid width is 3 (odd):\n"
                "  • Moving the blank left or right does not change the linear order of numbered tiles "
                "(delta inversions = 0).\n"
                "  • Moving the blank up or down moves one tile past 2 neighbors (delta inversions is "
                "-2, 0, or +2, preserving parity mod 2).\n"
                "Therefore, Current is solvable to Goal if and only if inversions(Current) % 2 == "
                "inversions(Goal) % 2. Exactly half of all 9! = 362,880 permutations (181,440 states) "
                "are solvable."
            ),
            "state_space_size": "9! = 362,880 total permutations; 9!/2 = 181,440 reachable states",
        },
        "complexity_and_divmod_math": {
            "question": "How do we achieve O(N) time and space complexity using Python's divmod?",
            "answer": (
                "A naive Manhattan implementation searches the 3x3 goal grid for each tile, taking "
                "O(N^2) time. Instead, we build a dictionary mapping each tile -> divmod(goal_idx, 3) "
                "in a single O(N) pass, then iterate over the flattened current board once, converting "
                "each flat index idx to (row, col) via divmod(idx, 3) and looking up its goal position "
                "in O(1). Total time and space are both O(N) where N = 9."
            ),
            "complexity": "Time: O(N), Space: O(N) where N = 9",
        },
    }
