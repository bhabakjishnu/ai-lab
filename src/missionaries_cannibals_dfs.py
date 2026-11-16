"""Missionaries and Cannibals Problem Solver using Depth-First Search (DFS).

This module implements the classic Missionaries and Cannibals river-crossing problem
using Depth-First Search (DFS) with cycle detection over an implicitly generated
state space, as covered in standard Artificial Intelligence curricula
(Russell & Norvig, Rich & Knight).

Theoretical Background & AI Concepts:
- Problem Formulation:
    Three missionaries (M) and three cannibals (C) must cross from the left bank of a
    river to the right bank using a boat that can hold at most two people (boat capacity = 2).
    The boat cannot cross the river without at least one person rowing (minimum passengers = 1).
    At no time, neither on either riverbank nor in the boat, may the number of cannibals
    outnumber the number of missionaries present on that bank, unless there are zero
    missionaries on that bank (if M == 0, cannibals have no missionaries to eat).
- State-Space Representation:
    A state is represented as a 3-tuple: (M_left, C_left, Boat_pos) where:
        M_left: Number of missionaries on the left bank (0 <= M_left <= 3)
        C_left: Number of cannibals on the left bank (0 <= C_left <= 3)
        Boat_pos: Position of the boat ('Left' or 'Right')
    The number of people on the right bank is implicitly determined by conservation of mass:
        M_right = total_missionaries - M_left = 3 - M_left
        C_right = total_cannibals - C_left = 3 - C_left
    Initial State: (3, 3, 'Left')
    Goal State: (0, 0, 'Right')
- The 5 Valid Boat Moves:
    With boat capacity 2, any move transfers between 1 and 2 individuals across the river:
    1. 1M   : (1, 0) - 1 Missionary crosses
    2. 2M   : (2, 0) - 2 Missionaries cross
    3. 1C   : (0, 1) - 1 Cannibal crosses
    4. 2C   : (0, 2) - 2 Cannibals cross
    5. 1M1C : (1, 1) - 1 Missionary and 1 Cannibal cross
- Transition Model:
    When Boat_pos == 'Left':
        Move from Left to Right: (M_left - m, C_left - c, 'Right')
    When Boat_pos == 'Right':
        Move from Right to Left: (M_left + m, C_left + c, 'Left')
- Safety Constraints:
    For any state (M_left, C_left, Boat_pos):
    1. Bounds: 0 <= M_left <= 3 and 0 <= C_left <= 3
    2. Left Bank Safety: if M_left > 0, then M_left >= C_left
    3. Right Bank Safety: if M_right > 0, then M_right >= C_right (where M_right = 3 - M_left, C_right = 3 - C_left)
- Constraint Validation during Generation vs Testing:
    - Validation during generation (Precondition Pruning / Forward Checking):
      Checks constraints before generating the child node, pruning invalid moves
      before allocating memory or pushing to the search stack.
    - Validation during testing (Generate-and-Test):
      Generates all 5 arithmetic successor states blindly and filters afterwards.
    Evaluating constraints during state generation significantly reduces the effective
    branching factor from 5 to ~1-2 valid transitions per state.
- Depth-First Search with Cycle Detection:
    DFS traverses the state tree by following a branch to maximum depth before backtracking.
    Because river-crossing moves are reversible (an action can be undone immediately),
    DFS in a graph without cycle detection WILL enter an infinite loop (e.g. crossing back and
    forth indefinitely). To prevent infinite loops, cycle detection is mandatory:
    - Path-based cycle avoidance: verifying candidate states against the current search path.
    - Visited-set graph-search: tracking all previously explored states in a hash set.
- State Space Characteristics:
    - Combinatorial unconstrained states: (3 + 1) * (3 + 1) * 2 = 32 states.
    - Safe (valid) states satisfying cannibal constraints: exactly 20 states.
    - Reachable safe states from (3, 3, 'Left'): 16 states.
    - Shortest optimal solution length: 11 boat crossings (12 states including initial).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO, Tuple, Union

# ---------------------------------------------------------------------------
# Canonical Constants & State Aliases
# ---------------------------------------------------------------------------

DEFAULT_TOTAL_MISSIONARIES: int = 3
DEFAULT_TOTAL_CANNIBALS: int = 3
DEFAULT_BOAT_CAPACITY: int = 2

BOAT_LEFT: str = "Left"
BOAT_RIGHT: str = "Right"
VALID_BOAT_POSITIONS: Tuple[str, ...] = (BOAT_LEFT, BOAT_RIGHT)

# State representation: (M_left, C_left, Boat_pos)
State = Tuple[int, int, str]

INITIAL_STATE: State = (DEFAULT_TOTAL_MISSIONARIES, DEFAULT_TOTAL_CANNIBALS, BOAT_LEFT)
GOAL_STATE: State = (0, 0, BOAT_RIGHT)


# ---------------------------------------------------------------------------
# Move Definitions (The 5 Canonical Moves)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoatMove:
    """Represents one of the 5 canonical boat crossing moves."""
    m: int
    c: int
    label: str
    description: str

    @property
    def total_passengers(self) -> int:
        """Return total number of passengers in the boat."""
        return self.m + self.c

    @property
    def move_tuple(self) -> Tuple[int, int]:
        """Return (missionaries, cannibals) passenger tuple."""
        return (self.m, self.c)

    def __repr__(self) -> str:
        return f"BoatMove({self.label}: {self.m}M, {self.c}C)"


# The 5 canonical moves matching the syllabus requirements: 1M, 2M, 1C, 2C, 1M1C
MOVE_1M = BoatMove(m=1, c=0, label="1M", description="1 Missionary")
MOVE_2M = BoatMove(m=2, c=0, label="2M", description="2 Missionaries")
MOVE_1C = BoatMove(m=0, c=1, label="1C", description="1 Cannibal")
MOVE_2C = BoatMove(m=0, c=2, label="2C", description="2 Cannibals")
MOVE_1M1C = BoatMove(m=1, c=1, label="1M1C", description="1 Missionary and 1 Cannibal")

CANONICAL_5_MOVES: Tuple[BoatMove, ...] = (
    MOVE_1M,
    MOVE_2M,
    MOVE_1C,
    MOVE_2C,
    MOVE_1M1C,
)

# Ordered move sequence that guides DFS directly to the standard 11-step optimal solution
# matching sample output: Step 1: 2C cross right...
OPTIMAL_DFS_MOVE_ORDER: Tuple[BoatMove, ...] = (
    MOVE_2C,
    MOVE_1M1C,
    MOVE_2M,
    MOVE_1C,
    MOVE_1M,
)


# ---------------------------------------------------------------------------
# Transition and Search Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TransitionStep:
    """Record of a single river crossing step."""
    step_number: int
    action: str
    move: BoatMove
    from_state: State
    to_state: State
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transition record to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "move": self.move.label,
            "from_state": list(self.from_state),
            "to_state": list(self.to_state),
            "description": self.description,
        }


@dataclass
class MissionariesCannibalsResult:
    """Comprehensive result of Missionaries and Cannibals search."""
    success: bool
    path: List[State] = field(default_factory=list)
    transitions: List[TransitionStep] = field(default_factory=list)
    states_explored: int = 0
    states_generated: int = 0
    pruned_by_capacity: int = 0
    pruned_by_safety: int = 0
    pruned_by_cycle: int = 0
    max_depth: int = 0
    execution_time_ms: float = 0.0
    algorithm: str = "DFS"
    method: str = "iterative"
    validation_mode: str = "generation"

    @property
    def total_trips(self) -> int:
        """Return the number of boat trips taken (transitions count)."""
        return len(self.transitions)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search result to dictionary."""
        return {
            "success": self.success,
            "algorithm": self.algorithm,
            "method": self.method,
            "validation_mode": self.validation_mode,
            "total_trips": self.total_trips,
            "path": [list(s) for s in self.path],
            "transitions": [t.to_dict() for t in self.transitions],
            "states_explored": self.states_explored,
            "states_generated": self.states_generated,
            "pruned_by_capacity": self.pruned_by_capacity,
            "pruned_by_safety": self.pruned_by_safety,
            "pruned_by_cycle": self.pruned_by_cycle,
            "max_depth": self.max_depth,
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# State Constraint Checking & Validation
# ---------------------------------------------------------------------------

def is_valid_state(
    state: State,
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> bool:
    """Validate whether a state satisfies all environmental and safety constraints.

    Safety Rules:
    1. Numerical bounds: 0 <= M_left <= total_m and 0 <= C_left <= total_c.
    2. Left bank safety: If M_left > 0, then M_left >= C_left.
    3. Right bank safety: If M_right > 0, then M_right >= C_right.
       (where M_right = total_m - M_left and C_right = total_c - C_left).
    4. Valid boat position: Must be 'Left' or 'Right'.

    Args:
        state: State 3-tuple (M_left, C_left, Boat_pos).
        total_m: Total number of missionaries in the problem.
        total_c: Total number of cannibals in the problem.

    Returns:
        True if the state is safe and valid, False otherwise.
    """
    if len(state) != 3:
        return False

    m_left, c_left, boat_pos = state

    if not isinstance(m_left, int) or not isinstance(c_left, int):
        return False

    if boat_pos not in VALID_BOAT_POSITIONS:
        return False

    # 1. Bounds checking on left bank
    if m_left < 0 or m_left > total_m or c_left < 0 or c_left > total_c:
        return False

    # 2. Left bank cannibal constraint: cannibals cannot outnumber missionaries
    if m_left > 0 and c_left > m_left:
        return False

    # 3. Right bank cannibal constraint
    m_right = total_m - m_left
    c_right = total_c - c_left

    if m_right < 0 or m_right > total_m or c_right < 0 or c_right > total_c:
        return False

    if m_right > 0 and c_right > m_right:
        return False

    return True


def validate_state_detailed(
    state: State,
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> Tuple[bool, str]:
    """Provide detailed diagnostic feedback explaining why a state is valid or invalid.

    Args:
        state: State 3-tuple (M_left, C_left, Boat_pos).
        total_m: Total number of missionaries.
        total_c: Total number of cannibals.

    Returns:
        Tuple (is_valid, diagnostic_reason_string).
    """
    if len(state) != 3:
        return False, "Invalid state format: state must be a 3-tuple (M_left, C_left, Boat_pos)"

    m_left, c_left, boat_pos = state

    if boat_pos not in VALID_BOAT_POSITIONS:
        return False, f"Invalid boat position: '{boat_pos}' (expected 'Left' or 'Right')"

    if m_left < 0 or m_left > total_m:
        return False, f"Missionaries on Left bank out of bounds: {m_left} (must be 0 to {total_m})"

    if c_left < 0 or c_left > total_c:
        return False, f"Cannibals on Left bank out of bounds: {c_left} (must be 0 to {total_c})"

    m_right = total_m - m_left
    c_right = total_c - c_left

    if m_left > 0 and c_left > m_left:
        return False, (
            f"Cannibals outnumber missionaries on Left bank: {c_left} cannibals > {m_left} missionaries"
        )

    if m_right > 0 and c_right > m_right:
        return False, (
            f"Cannibals outnumber missionaries on Right bank: {c_right} cannibals > {m_right} missionaries"
        )

    return True, "Safe state: missionaries are not outnumbered on either bank"


def is_move_physically_possible(
    state: State,
    move: BoatMove,
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> bool:
    """Check whether sufficient people exist on the departure bank to board the boat.

    Args:
        state: Current state (M_left, C_left, Boat_pos).
        move: BoatMove specification with passenger counts.
        total_m: Total missionaries.
        total_c: Total cannibals.

    Returns:
        True if the required missionaries and cannibals are physically on the departure bank.
    """
    m_left, c_left, boat_pos = state

    if boat_pos == BOAT_LEFT:
        return m_left >= move.m and c_left >= move.c
    elif boat_pos == BOAT_RIGHT:
        m_right = total_m - m_left
        c_right = total_c - c_left
        return m_right >= move.m and c_right >= move.c

    return False


def apply_move(
    state: State,
    move: BoatMove,
) -> State:
    """Apply a boat move to a state and return the raw successor state.

    Args:
        state: Current state (M_left, C_left, Boat_pos).
        move: BoatMove to execute.

    Returns:
        New state (M_left', C_left', Boat_pos').
    """
    m_left, c_left, boat_pos = state

    if boat_pos == BOAT_LEFT:
        return (m_left - move.m, c_left - move.c, BOAT_RIGHT)
    else:
        return (m_left + move.m, c_left + move.c, BOAT_LEFT)


def get_action_label(move: BoatMove, boat_pos: str) -> str:
    """Generate human-readable action string (e.g. '2C cross right')."""
    direction = "right" if boat_pos == BOAT_LEFT else "left"
    return f"{move.label} cross {direction}"


# ---------------------------------------------------------------------------
# State Space Explorer & Analyzer
# ---------------------------------------------------------------------------

def analyze_state_space(
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> Dict[str, Any]:
    """Exhaustively enumerate and analyze the Missionaries and Cannibals state space.

    Calculates:
    - Total theoretical unconstrained states: (total_m + 1) * (total_c + 1) * 2
    - Total valid (safe) states satisfying safety constraints
    - Total reachable safe states from initial state (3, 3, 'Left')
    - Unreachable valid states
    - Adjacency structure of the state transition graph

    Returns:
        Dictionary containing state space metrics and categorized state lists.
    """
    total_theoretical = (total_m + 1) * (total_c + 1) * 2
    all_states: List[State] = []
    safe_states: List[State] = []
    unsafe_states: List[Tuple[State, str]] = []

    for m in range(total_m + 1):
        for c in range(total_c + 1):
            for b in VALID_BOAT_POSITIONS:
                st = (m, c, b)
                all_states.append(st)
                valid, reason = validate_state_detailed(st, total_m, total_c)
                if valid:
                    safe_states.append(st)
                else:
                    unsafe_states.append((st, reason))

    # Determine reachability via BFS from INITIAL_STATE
    start_state = (total_m, total_c, BOAT_LEFT)
    queue: deque[State] = deque([start_state])
    reachable_safe: Set[State] = {start_state}
    graph_edges: Dict[State, List[Tuple[BoatMove, State]]] = {}

    for s in safe_states:
        graph_edges[s] = []

    while queue:
        current = queue.popleft()
        for move in CANONICAL_5_MOVES:
            if is_move_physically_possible(current, move, total_m, total_c):
                nxt = apply_move(current, move)
                if is_valid_state(nxt, total_m, total_c):
                    graph_edges[current].append((move, nxt))
                    if nxt not in reachable_safe:
                        reachable_safe.add(nxt)
                        queue.append(nxt)

    unreachable_safe = [s for s in safe_states if s not in reachable_safe]

    return {
        "total_theoretical_states": total_theoretical,
        "total_safe_states": len(safe_states),
        "total_unsafe_states": len(unsafe_states),
        "total_reachable_safe_states": len(reachable_safe),
        "total_unreachable_safe_states": len(unreachable_safe),
        "safe_states": safe_states,
        "reachable_safe_states": sorted(list(reachable_safe)),
        "unreachable_safe_states": unreachable_safe,
        "unsafe_states_sample": unsafe_states[:5],
    }


# ---------------------------------------------------------------------------
# Core Missionaries and Cannibals Solver Class
# ---------------------------------------------------------------------------

class MissionariesCannibalsSolver:
    """Depth-First Search (DFS) Solver for the Missionaries and Cannibals Problem.

    Supports:
    - Recursive and iterative DFS with loop/cycle avoidance
    - Constraint checking during generation (forward checking) or during testing
    - Configurable move ordering (defaulting to the canonical optimal 11-step sequence)
    - Optional Breadth-First Search (BFS) for theoretical comparison against DFS
    """

    def __init__(
        self,
        total_missionaries: int = DEFAULT_TOTAL_MISSIONARIES,
        total_cannibals: int = DEFAULT_TOTAL_CANNIBALS,
        boat_capacity: int = DEFAULT_BOAT_CAPACITY,
        move_ordering: Optional[Sequence[BoatMove]] = None,
    ) -> None:
        """Initialize the solver with problem parameters.

        Args:
            total_missionaries: Total missionaries (standard 3).
            total_cannibals: Total cannibals (standard 3).
            boat_capacity: Maximum people the boat can hold (standard 2).
            move_ordering: Priority sequence of boat moves to evaluate.
        """
        self.total_m = total_missionaries
        self.total_c = total_cannibals
        self.boat_capacity = boat_capacity
        self.moves: Tuple[BoatMove, ...] = (
            tuple(move_ordering) if move_ordering is not None else OPTIMAL_DFS_MOVE_ORDER
        )
        self.start_state: State = (self.total_m, self.total_c, BOAT_LEFT)
        self.goal_state: State = (0, 0, BOAT_RIGHT)

    def is_safe(self, state: State) -> bool:
        """Convenience method checking safety constraints of a state."""
        return is_valid_state(state, self.total_m, self.total_c)

    def generate_successors_during_generation(
        self,
        state: State,
    ) -> Tuple[List[Tuple[BoatMove, State, str]], int, int]:
        """Generate valid successors using constraint checking during state generation.

        Preconditions and safety rules are evaluated before instantiating and accepting
        successor states, preventing unviable states from entering search memory.

        Returns:
            Tuple of:
            - List of (move, next_state, action_label)
            - Count of moves pruned due to insufficient passengers on bank
            - Count of moves pruned due to cannibal safety violation
        """
        successors: List[Tuple[BoatMove, State, str]] = []
        pruned_cap = 0
        pruned_safe = 0

        for move in self.moves:
            # 1. Precondition check: are there enough people on this bank?
            if not is_move_physically_possible(state, move, self.total_m, self.total_c):
                pruned_cap += 1
                continue

            # 2. Forward checking: will the resulting state be safe?
            candidate_state = apply_move(state, move)
            if not self.is_safe(candidate_state):
                pruned_safe += 1
                continue

            action_label = get_action_label(move, state[2])
            successors.append((move, candidate_state, action_label))

        return successors, pruned_cap, pruned_safe

    def generate_successors_during_testing(
        self,
        state: State,
    ) -> Tuple[List[Tuple[BoatMove, State, str]], int, int]:
        """Generate all arithmetic successors first and test validity afterwards.

        This demonstrates the generate-and-test paradigm where constraint checking
        happens on candidate states after blind generation.

        Returns:
            Tuple of:
            - List of valid (move, next_state, action_label)
            - Count of candidates discarded by capacity bounds
            - Count of candidates discarded by cannibal safety constraint
        """
        valid_successors: List[Tuple[BoatMove, State, str]] = []
        pruned_cap = 0
        pruned_safe = 0

        for move in self.moves:
            # Unconditionally generate arithmetic candidate
            candidate_state = apply_move(state, move)
            m_left, c_left, _ = candidate_state

            # Test capacity bounds
            if m_left < 0 or m_left > self.total_m or c_left < 0 or c_left > self.total_c:
                pruned_cap += 1
                continue

            # Test cannibal safety
            if not self.is_safe(candidate_state):
                pruned_safe += 1
                continue

            action_label = get_action_label(move, state[2])
            valid_successors.append((move, candidate_state, action_label))

        return valid_successors, pruned_cap, pruned_safe

    def solve_dfs(
        self,
        method: str = "iterative",
        validation_mode: str = "generation",
    ) -> MissionariesCannibalsResult:
        """Solve the Missionaries and Cannibals problem using Depth-First Search.

        Args:
            method: 'iterative' (LIFO stack) or 'recursive' (call stack).
            validation_mode: 'generation' (precondition check) or 'testing' (generate-and-test).

        Returns:
            MissionariesCannibalsResult containing path, transitions, and metrics.
        """
        start_time = time.perf_counter()

        if method == "recursive":
            res = self._solve_dfs_recursive(validation_mode)
        else:
            res = self._solve_dfs_iterative(validation_mode)

        res.execution_time_ms = round((time.perf_counter() - start_time) * 1000, 3)
        res.validation_mode = validation_mode
        return res

    def _solve_dfs_iterative(
        self,
        validation_mode: str = "generation",
    ) -> MissionariesCannibalsResult:
        """Iterative DFS using an explicit LIFO stack with cycle detection."""
        # Stack items: (current_state, path_states, transitions_so_far, visited_set_in_path)
        initial_visited = {self.start_state}
        stack: List[Tuple[State, List[State], List[TransitionStep], Set[State]]] = [
            (self.start_state, [self.start_state], [], initial_visited)
        ]

        states_explored = 0
        states_generated = 1
        pruned_by_capacity = 0
        pruned_by_safety = 0
        pruned_by_cycle = 0
        max_depth = 0

        successor_fn = (
            self.generate_successors_during_generation
            if validation_mode == "generation"
            else self.generate_successors_during_testing
        )

        while stack:
            state, path, transitions, visited_in_path = stack.pop()
            states_explored += 1
            max_depth = max(max_depth, len(path))

            if state == self.goal_state:
                return MissionariesCannibalsResult(
                    success=True,
                    path=path,
                    transitions=transitions,
                    states_explored=states_explored,
                    states_generated=states_generated,
                    pruned_by_capacity=pruned_by_capacity,
                    pruned_by_safety=pruned_by_safety,
                    pruned_by_cycle=pruned_by_cycle,
                    max_depth=max_depth,
                    algorithm="DFS",
                    method="iterative",
                )

            # Generate valid successors
            succs, p_cap, p_safe = successor_fn(state)
            pruned_by_capacity += p_cap
            pruned_by_safety += p_safe
            states_generated += len(succs)

            # Push successors onto the stack in REVERSE order so that the first move in
            # self.moves is evaluated first (LIFO order).
            for move, nxt_state, action_label in reversed(succs):
                # Cycle detection: check if state already exists along the current path
                if nxt_state in visited_in_path:
                    pruned_by_cycle += 1
                    continue

                step_num = len(transitions) + 1
                desc = (
                    f"Step {step_num}: {action_label} -> "
                    f"({nxt_state[0]},{nxt_state[1]},{nxt_state[2]})"
                )
                step = TransitionStep(
                    step_number=step_num,
                    action=action_label,
                    move=move,
                    from_state=state,
                    to_state=nxt_state,
                    description=desc,
                )

                new_path = path + [nxt_state]
                new_transitions = transitions + [step]
                new_visited = visited_in_path | {nxt_state}

                stack.append((nxt_state, new_path, new_transitions, new_visited))

        return MissionariesCannibalsResult(
            success=False,
            states_explored=states_explored,
            states_generated=states_generated,
            pruned_by_capacity=pruned_by_capacity,
            pruned_by_safety=pruned_by_safety,
            pruned_by_cycle=pruned_by_cycle,
            max_depth=max_depth,
            algorithm="DFS",
            method="iterative",
        )

    def _solve_dfs_recursive(
        self,
        validation_mode: str = "generation",
    ) -> MissionariesCannibalsResult:
        """Recursive DFS implementation with cycle detection."""
        states_explored = 0
        states_generated = 1
        pruned_by_capacity = 0
        pruned_by_safety = 0
        pruned_by_cycle = 0
        max_depth = 0

        successor_fn = (
            self.generate_successors_during_generation
            if validation_mode == "generation"
            else self.generate_successors_during_testing
        )

        def _dfs_helper(
            current: State,
            path: List[State],
            transitions: List[TransitionStep],
            visited: Set[State],
        ) -> Optional[Tuple[List[State], List[TransitionStep]]]:
            nonlocal states_explored, states_generated, pruned_by_capacity
            nonlocal pruned_by_safety, pruned_by_cycle, max_depth

            states_explored += 1
            max_depth = max(max_depth, len(path))

            if current == self.goal_state:
                return path, transitions

            succs, p_cap, p_safe = successor_fn(current)
            pruned_by_capacity += p_cap
            pruned_by_safety += p_safe
            states_generated += len(succs)

            for move, nxt_state, action_label in succs:
                if nxt_state in visited:
                    pruned_by_cycle += 1
                    continue

                step_num = len(transitions) + 1
                desc = (
                    f"Step {step_num}: {action_label} -> "
                    f"({nxt_state[0]},{nxt_state[1]},{nxt_state[2]})"
                )
                step = TransitionStep(
                    step_number=step_num,
                    action=action_label,
                    move=move,
                    from_state=current,
                    to_state=nxt_state,
                    description=desc,
                )

                visited.add(nxt_state)
                result = _dfs_helper(
                    nxt_state,
                    path + [nxt_state],
                    transitions + [step],
                    visited,
                )
                if result is not None:
                    return result
                # Backtrack: remove from current path visited set
                visited.remove(nxt_state)

            return None

        res = _dfs_helper(
            self.start_state,
            [self.start_state],
            [],
            {self.start_state},
        )

        if res is not None:
            path, transitions = res
            return MissionariesCannibalsResult(
                success=True,
                path=path,
                transitions=transitions,
                states_explored=states_explored,
                states_generated=states_generated,
                pruned_by_capacity=pruned_by_capacity,
                pruned_by_safety=pruned_by_safety,
                pruned_by_cycle=pruned_by_cycle,
                max_depth=max_depth,
                algorithm="DFS",
                method="recursive",
            )

        return MissionariesCannibalsResult(
            success=False,
            states_explored=states_explored,
            states_generated=states_generated,
            pruned_by_capacity=pruned_by_capacity,
            pruned_by_safety=pruned_by_safety,
            pruned_by_cycle=pruned_by_cycle,
            max_depth=max_depth,
            algorithm="DFS",
            method="recursive",
        )

    def solve_bfs(self) -> MissionariesCannibalsResult:
        """Solve using Breadth-First Search (BFS) for optimality comparison against DFS.

        BFS expands level-by-level using a FIFO queue, guaranteeing the minimum
        number of boat trips to reach the goal.

        Returns:
            MissionariesCannibalsResult containing optimal BFS solution.
        """
        start_time = time.perf_counter()
        queue: deque[Tuple[State, List[State], List[TransitionStep]]] = deque(
            [(self.start_state, [self.start_state], [])]
        )
        visited: Set[State] = {self.start_state}

        states_explored = 0
        states_generated = 1
        pruned_by_capacity = 0
        pruned_by_safety = 0
        pruned_by_cycle = 0
        max_depth = 0

        while queue:
            state, path, transitions = queue.popleft()
            states_explored += 1
            max_depth = max(max_depth, len(path))

            if state == self.goal_state:
                elapsed = round((time.perf_counter() - start_time) * 1000, 3)
                return MissionariesCannibalsResult(
                    success=True,
                    path=path,
                    transitions=transitions,
                    states_explored=states_explored,
                    states_generated=states_generated,
                    pruned_by_capacity=pruned_by_capacity,
                    pruned_by_safety=pruned_by_safety,
                    pruned_by_cycle=pruned_by_cycle,
                    max_depth=max_depth,
                    execution_time_ms=elapsed,
                    algorithm="BFS",
                    method="queue",
                )

            succs, p_cap, p_safe = self.generate_successors_during_generation(state)
            pruned_by_capacity += p_cap
            pruned_by_safety += p_safe
            states_generated += len(succs)

            for move, nxt_state, action_label in succs:
                if nxt_state in visited:
                    pruned_by_cycle += 1
                    continue

                visited.add(nxt_state)
                step_num = len(transitions) + 1
                desc = (
                    f"Step {step_num}: {action_label} -> "
                    f"({nxt_state[0]},{nxt_state[1]},{nxt_state[2]})"
                )
                step = TransitionStep(
                    step_number=step_num,
                    action=action_label,
                    move=move,
                    from_state=state,
                    to_state=nxt_state,
                    description=desc,
                )
                queue.append((nxt_state, path + [nxt_state], transitions + [step]))

        elapsed = round((time.perf_counter() - start_time) * 1000, 3)
        return MissionariesCannibalsResult(
            success=False,
            states_explored=states_explored,
            states_generated=states_generated,
            pruned_by_capacity=pruned_by_capacity,
            pruned_by_safety=pruned_by_safety,
            pruned_by_cycle=pruned_by_cycle,
            max_depth=max_depth,
            execution_time_ms=elapsed,
            algorithm="BFS",
            method="queue",
        )

    def compare_dfs_vs_bfs(self) -> Dict[str, Any]:
        """Perform side-by-side empirical comparison between DFS and BFS.

        Directly provides quantitative data for the Viva Voce examination:
        'Why BFS might be preferred over DFS here'.

        Returns:
            Comparison dictionary with path length, nodes explored, and optimality analysis.
        """
        dfs_res = self.solve_dfs(method="iterative")
        bfs_res = self.solve_bfs()

        return {
            "dfs": {
                "success": dfs_res.success,
                "trips": dfs_res.total_trips,
                "states_explored": dfs_res.states_explored,
                "states_generated": dfs_res.states_generated,
                "execution_time_ms": dfs_res.execution_time_ms,
                "optimal": dfs_res.total_trips == bfs_res.total_trips,
            },
            "bfs": {
                "success": bfs_res.success,
                "trips": bfs_res.total_trips,
                "states_explored": bfs_res.states_explored,
                "states_generated": bfs_res.states_generated,
                "execution_time_ms": bfs_res.execution_time_ms,
                "optimal": True,
            },
            "optimality_notes": (
                "BFS guarantees the minimum number of river crossings (11 trips). "
                "DFS path optimality depends on move ordering; DFS may find longer non-optimal "
                "paths or explore unnecessary subtrees before discovering the goal."
            ),
        }


# ---------------------------------------------------------------------------
# Visual Formatting & Output Traces
# ---------------------------------------------------------------------------

def format_solution_trace(
    result: MissionariesCannibalsResult,
    format_type: str = "standard",
) -> str:
    """Format the solution transitions according to desired laboratory format.

    Args:
        result: MissionariesCannibalsResult object.
        format_type:
            - 'standard': Step 1: 2C cross right... [Full path trace]
            - 'arrows': (3,3,Left) -> [2C right] -> (3,1,Right) -> ...
            - 'table': Formatted ASCII table with bank state distributions
            - 'detailed': Multi-line river view for each transition

    Returns:
        Formatted string trace.
    """
    if not result.success:
        return "Search failed: No solution found."

    lines: List[str] = []

    if format_type == "standard":
        lines.append(f"Initial State: {result.path[0]}")
        for t in result.transitions:
            lines.append(f"Step {t.step_number}: {t.action} -> {t.to_state}")
        lines.append(f"Goal Reached: {result.path[-1]} in {result.total_trips} boat trips!")

    elif format_type == "arrows":
        parts: List[str] = [f"({result.path[0][0]},{result.path[0][1]},{result.path[0][2]})"]
        for t in result.transitions:
            parts.append(f"--[{t.action}]--> ({t.to_state[0]},{t.to_state[1]},{t.to_state[2]})")
        lines.append(" ".join(parts))

    elif format_type == "table":
        header = (
            f"+------+--------------------+--------------------+--------------------+--------------------+\n"
            f"| Step | Action             | Left Bank (M, C)   | River / Boat       | Right Bank (M, C)  |\n"
            f"+------+--------------------+--------------------+--------------------+--------------------+"
        )
        lines.append(header)

        # Initial state row
        init_m, init_c, init_b = result.path[0]
        lines.append(
            f"|   0  | Initial State      | {init_m}M, {init_c}C [BOAT]      | ~~~~~~~~~~~~~~~~~~ | 0M, 0C             |"
        )

        for t in result.transitions:
            m_l, c_l, b_pos = t.to_state
            m_r = DEFAULT_TOTAL_MISSIONARIES - m_l
            c_r = DEFAULT_TOTAL_CANNIBALS - c_l

            if b_pos == BOAT_RIGHT:
                boat_river = f"~~~ [{t.move.label}] >>>> ~~~"
                left_str = f"{m_l}M, {c_l}C"
                right_str = f"{m_r}M, {c_r}C [BOAT]"
            else:
                boat_river = f"~~~ <<<< [{t.move.label}] ~~~"
                left_str = f"{m_l}M, {c_l}C [BOAT]"
                right_str = f"{m_r}M, {c_r}C"

            lines.append(
                f"| {t.step_number:4d} | {t.action:<18} | {left_str:<18} | {boat_river:<18} | {right_str:<18} |"
            )

        lines.append(
            f"+------+--------------------+--------------------+--------------------+--------------------+"
        )

    elif format_type == "detailed":
        lines.append("=" * 76)
        lines.append(f"MISSIONARIES & CANNIBALS RIVER CROSSING TRACE ({result.total_trips} TRIPS)")
        lines.append("=" * 76)
        lines.append(f"Step 0 (Start): Left={result.path[0][:2]}, Boat=Left, Right=(0, 0)")
        lines.append(render_ascii_river(result.path[0]))

        for t in result.transitions:
            lines.append("-" * 76)
            lines.append(f"Step {t.step_number}: {t.action}")
            lines.append(render_ascii_river(t.to_state, t.move, t.action))

        lines.append("=" * 76)
        lines.append(f"SUCCESS: All missionaries and cannibals crossed safely in {result.total_trips} trips!")

    return "\n".join(lines)


def render_ascii_river(
    state: State,
    move: Optional[BoatMove] = None,
    action: str = "",
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> str:
    """Render a visual ASCII representation of both riverbanks and the boat.

    Args:
        state: Current state (M_left, C_left, Boat_pos).
        move: Most recent move executed (if any).
        action: Descriptive action string.
        total_m: Total missionaries.
        total_c: Total cannibals.

    Returns:
        Multi-line ASCII illustration of the river scenario.
    """
    m_left, c_left, boat_pos = state
    m_right = total_m - m_left
    c_right = total_c - c_left

    m_left_str = "M" * m_left + "-" * (total_m - m_left)
    c_left_str = "C" * c_left + "-" * (total_c - c_left)

    m_right_str = "M" * m_right + "-" * (total_m - m_right)
    c_right_str = "C" * c_right + "-" * (total_c - c_right)

    if boat_pos == BOAT_LEFT:
        boat_l = "[BOAT]"
        boat_r = "      "
        river = "~ ~ ~ ~ ~ ~ ~ ~"
    else:
        boat_l = "      "
        boat_r = "[BOAT]"
        river = "~ ~ ~ ~ ~ ~ ~ ~"

    line1 = f"Left Bank: [{m_left_str} {c_left_str}] {boat_l} | {river} | {boat_r} [{m_right_str} {c_right_str}] :Right Bank"
    return line1


# ---------------------------------------------------------------------------
# Optional Challenge: Text Animation with Delays
# ---------------------------------------------------------------------------

def animate_solution(
    result: MissionariesCannibalsResult,
    delay: float = 0.35,
    stream: Optional[TextIO] = None,
    total_m: int = DEFAULT_TOTAL_MISSIONARIES,
    total_c: int = DEFAULT_TOTAL_CANNIBALS,
) -> None:
    """Animate the river crossing step-by-step with ASCII visualization and configurable delay.

    Optional Challenge requirement: 'Animate the text output with delays.'

    Args:
        result: MissionariesCannibalsResult to animate.
        delay: Time delay in seconds between frames (default 0.35s).
        stream: Output text stream (defaults to sys.stdout).
        total_m: Total missionaries.
        total_c: Total cannibals.
    """
    if stream is None:
        stream = sys.stdout

    if not result.success:
        stream.write("Cannot animate: No solution found.\n")
        stream.flush()
        return

    banner = "\n" + "=" * 76 + "\n" + "  MISSIONARIES & CANNIBALS RIVER CROSSING ANIMATION\n" + "=" * 76 + "\n"
    stream.write(banner)
    stream.flush()

    # Initial frame
    stream.write("\n[Step 0 / 11] Initial State: (3, 3, 'Left')\n")
    stream.write(render_ascii_river(result.path[0], total_m=total_m, total_c=total_c) + "\n")
    stream.flush()
    if delay > 0:
        time.sleep(delay)

    for step in result.transitions:
        stream.write("\n" + "-" * 76 + "\n")
        stream.write(f"[{step.step_number:02d} / {result.total_trips:02d}] Crossing Action: {step.action}\n")

        # Crossing motion sub-frames
        move_label = step.move.label
        if step.from_state[2] == BOAT_LEFT:
            frames = [
                f"~~~ [{move_label}] >>>> ~~~",
                f"~~~~~ >>>> [{move_label}] ~",
            ]
        else:
            frames = [
                f"~~~ <<<< [{move_label}] ~~~",
                f"~ [{move_label}] <<<< ~~~~~",
            ]

        for frame in frames:
            stream.write(f"  River transit: {frame}\r")
            stream.flush()
            if delay > 0:
                time.sleep(delay * 0.4)

        stream.write(f"\nArrived State: ({step.to_state[0]}, {step.to_state[1]}, '{step.to_state[2]}')\n")
        stream.write(render_ascii_river(step.to_state, step.move, step.action, total_m, total_c) + "\n")
        stream.flush()

        if delay > 0:
            time.sleep(delay)

    stream.write("\n" + "=" * 76 + "\n")
    stream.write(f"  ALL 3 MISSIONARIES AND 3 CANNIBALS SAFELY CROSSED THE RIVER!\n")
    stream.write("=" * 76 + "\n")
    stream.flush()


# ---------------------------------------------------------------------------
# Convenience Wrapper Functions
# ---------------------------------------------------------------------------

def solve_missionaries_cannibals_dfs(
    total_missionaries: int = DEFAULT_TOTAL_MISSIONARIES,
    total_cannibals: int = DEFAULT_TOTAL_CANNIBALS,
    method: str = "iterative",
    validation_mode: str = "generation",
    move_ordering: Optional[Sequence[BoatMove]] = None,
) -> MissionariesCannibalsResult:
    """Solve the Missionaries and Cannibals problem using Depth-First Search.

    Args:
        total_missionaries: Number of missionaries (default 3).
        total_cannibals: Number of cannibals (default 3).
        method: 'iterative' or 'recursive'.
        validation_mode: 'generation' or 'testing'.
        move_ordering: Optional custom move sequence.

    Returns:
        MissionariesCannibalsResult object.
    """
    solver = MissionariesCannibalsSolver(
        total_missionaries=total_missionaries,
        total_cannibals=total_cannibals,
        move_ordering=move_ordering,
    )
    return solver.solve_dfs(method=method, validation_mode=validation_mode)


# ---------------------------------------------------------------------------
# Viva Voce Examination Reference Guide
# ---------------------------------------------------------------------------

def get_missionaries_cannibals_viva_reference() -> Dict[str, Dict[str, str]]:
    """Return authoritative answers to standard University Viva Voce examination questions.

    Covers:
    1. Constraint validation during generation vs testing
    2. State space size of this problem (combinatorial, valid, and reachable states)
    3. Why BFS might be preferred over DFS here
    4. Generalization to N Missionaries and N Cannibals
    5. Loop / Cycle detection mechanisms in DFS

    Returns:
        Dictionary mapping topic titles to question and answer dictionaries.
    """
    return {
        "Constraint Validation: Generation vs Testing": {
            "question": (
                "Explain the difference between constraint validation during state generation "
                "versus constraint validation during state testing in state-space search."
            ),
            "answer": (
                "In state-space search, constraints can be evaluated at two distinct phases:\n"
                "1. Validation During Generation (Precondition Checking / Forward Checking):\n"
                "   The search agent evaluates physical preconditions (e.g. are there enough people on "
                "the departure bank?) and safety rules (e.g. will cannibals outnumber missionaries on either bank?) "
                "BEFORE creating or queueing the child node. If a move is invalid or unsafe, it is pruned immediately. "
                "This prunes unviable branches before memory allocation or stack insertion, reducing the effective "
                "branching factor from 5 to ~1-2 valid transitions per state.\n"
                "2. Validation During Testing (Generate-and-Test):\n"
                "   The generator blindly creates all 5 mathematical successor states (M +/- m, C +/- c) "
                "and only later tests each state against constraint rules. While conceptually simpler, this wastes "
                "CPU cycles generating out-of-bounds states (e.g. negative people) and allocating stack objects for "
                "states that are instantly discarded.\n"
                "Conclusion: In AI constraint satisfaction and state-space search, validation during generation "
                "is strictly superior in terms of time and memory complexity."
            ),
        },
        "State Space Size and Reachability": {
            "question": (
                "What is the theoretical state space size of the 3M, 3C problem, "
                "how many states are safe, and how many are actually reachable?"
            ),
            "answer": (
                "For 3 missionaries, 3 cannibals, and 1 boat:\n"
                "1. Theoretical Combinatorial Size:\n"
                "   M_left in {0, 1, 2, 3} (4 values)\n"
                "   C_left in {0, 1, 2, 3} (4 values)\n"
                "   Boat_pos in {'Left', 'Right'} (2 values)\n"
                "   Total unrestricted state space = 4 x 4 x 2 = 32 states.\n"
                "2. Valid (Safe) States:\n"
                "   A state is safe if M=0 or M >= C on both banks:\n"
                "   - M_left = 0: C_left in {0, 1, 2, 3} (4 valid configurations)\n"
                "   - M_left = 1: C_left can only be 1 (Right has 2M, 2C) (1 valid configuration)\n"
                "   - M_left = 2: C_left can only be 2 (Right has 1M, 1C) (1 valid configuration)\n"
                "   - M_left = 3: C_left in {0, 1, 2, 3} (4 valid configurations)\n"
                "   Total safe configurations = 4 + 1 + 1 + 4 = 10 configurations.\n"
                "   Since the boat can be on Left or Right: 10 x 2 = 20 safe states in total.\n"
                "3. Reachable Safe States:\n"
                "   From the start state (3, 3, 'Left'), exactly 16 of the 20 safe states are reachable. "
                "The remaining 4 safe states (such as (3, 3, 'Right') or (0, 0, 'Left')) are unreachable because "
                "they violate river conservation parity (the boat cannot be on the other side without people crossing)."
            ),
        },
        "Why BFS is Preferred Over DFS": {
            "question": (
                "Why might Breadth-First Search (BFS) be preferred over Depth-First Search (DFS) "
                "for solving the Missionaries and Cannibals problem?"
            ),
            "answer": (
                "BFS is generally preferred over DFS for this problem for several key reasons:\n"
                "1. Shortest Path Optimality:\n"
                "   BFS explores states level-by-level (by path length). The first time it encounters the goal "
                "(0, 0, 'Right'), it is mathematically guaranteed to have found the shortest path (minimum number "
                "of boat trips, which is exactly 11 trips for 3M, 3C). DFS does NOT guarantee shortest path; "
                "depending on move ordering, DFS may follow a long, convoluted path of 15, 20, or more crossings.\n"
                "2. Risk of Infinite Loops in DFS:\n"
                "   Because river-crossing moves are reversible (an action can be undone immediately), DFS without "
                "cycle detection will enter an infinite loop ping-ponging between two states. BFS naturally terminates "
                "even with simpler tracking because it does not chase deep paths.\n"
                "3. Small State Space:\n"
                "   The memory benefit of DFS (O(b*d) vs O(b^d)) is irrelevant here because the total state space is "
                "tiny (only 16 reachable states). BFS easily fits in memory (a few kilobytes) while providing the optimal solution."
            ),
        },
        "Generalization to N Missionaries and N Cannibals": {
            "question": (
                "Can the Missionaries and Cannibals problem be solved for N = 4 or higher with a 2-person boat?"
            ),
            "answer": (
                "No. It is a famous mathematical theorem in Artificial Intelligence (Amarel, 1968) that for a "
                "boat capacity of K = 2, the Missionaries and Cannibals problem is solvable if and only if N <= 3.\n"
                "For N >= 4, the problem is provably UNSOLVABLE with a 2-person boat unless an intermediate island "
                "in the river is introduced or the boat capacity is increased to K >= 3.\n"
                "Reason: To transport missionaries across without being eaten, the boat must shuttle back and forth. "
                "With N >= 4 and K = 2, every possible crossing leaves at least one bank where cannibals outnumber "
                "missionaries during the intermediate transition stage."
            ),
        },
        "Cycle Detection in DFS": {
            "question": (
                "How does cycle detection work in DFS for state-space search?"
            ),
            "answer": (
                "Cycle detection can be implemented via two primary approaches:\n"
                "1. Path-based cycle detection (Current Recursion Stack):\n"
                "   Maintains the set of states currently on the path from the root to the current node. "
                "Before expanding a successor, the search checks if the successor is already in the path set. "
                "If it is, the branch is pruned as a loop. This requires O(d) space where d is search depth.\n"
                "2. Graph-search visited set (Closed List):\n"
                "   Maintains a global hash set of all states ever explored. A state is never re-expanded even "
                "if reached via a different branch. This guarantees O(V) termination but in DFS may occasionally "
                "block alternative valid paths if depth bounding is used.\n"
                "For Missionaries and Cannibals, tracking the visited set along the active path guarantees avoidance "
                "of infinite loops while exploring all valid river-crossing sequences."
            ),
        },
    }
