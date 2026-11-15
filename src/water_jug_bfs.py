"""Water Jug Problem Solver using Breadth-First Search (BFS).

This module implements the classic Water Jug Problem solved using Breadth-First
Search (BFS) over an implicitly generated state space, as covered in standard
Artificial Intelligence curricula (Russell & Norvig, Rich & Knight).

Theoretical Background & AI Concepts:
- Problem Formulation:
    Given two jugs of integer capacities X and Y liters without intermediate markings,
    and an infinite water supply (tap) and infinite drain (sink), find the optimal
    (shortest) sequence of operations to measure exactly Z liters in one of the jugs.
- State-Space Representation:
    A state is represented as a 2-tuple: (jug1_amt, jug2_amt) where:
        0 <= jug1_amt <= X
        0 <= jug2_amt <= Y
    The initial state is typically (0, 0) (both jugs empty).
    The goal condition is:
        jug1_amt == Z  (when measuring in Jug 1, e.g. classic 4-gallon jug)
        jug2_amt == Z  (when measuring in Jug 2)
        or jug1_amt == Z or jug2_amt == Z (when measuring in either jug)
- Transition Model (The 6 Canonical Production Rules):
    1. Fill Jug 1: (X, y)
    2. Fill Jug 2: (x, Y)
    3. Empty Jug 1: (0, y)
    4. Empty Jug 2: (x, 0)
    5. Pour Jug 1 -> Jug 2:
       amount = min(x, Y - y)
       new_state = (x - amount, y + amount)
    6. Pour Jug 2 -> Jug 1:
       amount = min(y, X - x)
       new_state = (x + amount, y - amount)
- FIFO Frontier Queue (collections.deque):
    BFS explores the state space level-by-level, ensuring that the first time a
    goal state is reached, the path discovered has the minimum number of transitions.
- Visited Set & Cycle Avoidance:
    Maintains a set of visited states. Because state spaces are discrete and finite
    (at most (X + 1) * (Y + 1) unique states), tracking visited states guarantees
    termination and prevents infinite loops.
- Algebraic Reachability & GCD Rule (Bézout's Identity):
    Before initiating search, algebraic reachability can be verified in O(log(min(X, Y)))
    time using Bézout's Identity:
    Any state reachable by filling, emptying, and pouring operations is an integer linear
    combination of X and Y: a*X + b*Y = Z for integers a, b.
    Therefore, target Z is reachable in one of the jugs IF AND ONLY IF:
    1. Z <= max(X, Y) (capacity upper bound; or Z <= X if target is Jug 1).
    2. Z % gcd(X, Y) == 0 (divisibility condition).
    If these conditions fail, the problem is provably unsolvable.
- Complexity:
    Time Complexity: O(X * Y) - at most (X + 1) * (Y + 1) states are visited,
    and each state evaluates 6 constant-time production rules.
    Space Complexity: O(X * Y) - auxiliary memory for the FIFO queue and visited set.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Canonical Benchmark Configurations
# ---------------------------------------------------------------------------

SAMPLE_WATER_JUG_PARAMS: Dict[str, int] = {
    "capacity_x": 4,
    "capacity_y": 3,
    "target_z": 2,
}
"""Canonical sample problem matching Question 20 specification:
Capacities: 4, 3. Target: 2.
Expected Output: (0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)
"""

DIE_HARD_WATER_JUG_PARAMS: Dict[str, int] = {
    "capacity_x": 5,
    "capacity_y": 3,
    "target_z": 4,
}
"""Classic puzzle from the film 'Die Hard with a Vengeance' (Capacities 5, 3; Target 4)."""

COPRIME_WATER_JUG_PARAMS: Dict[str, int] = {
    "capacity_x": 7,
    "capacity_y": 5,
    "target_z": 4,
}
"""Co-prime capacities with larger state space."""

UNSOLVABLE_WATER_JUG_PARAMS: Dict[str, int] = {
    "capacity_x": 6,
    "capacity_y": 4,
    "target_z": 5,
}
"""Unsolvable configuration: gcd(6, 4) = 2, which does not divide target 5."""

EQUAL_CAPACITY_PARAMS: Dict[str, int] = {
    "capacity_x": 4,
    "capacity_y": 4,
    "target_z": 4,
}
"""Edge case: equal capacities."""

# State alias
WaterJugState = Tuple[int, int]


# ---------------------------------------------------------------------------
# Action Enumeration & Production Rules
# ---------------------------------------------------------------------------

class WaterJugAction(str, Enum):
    """The 6 canonical production rules for the two water jugs."""
    FILL_1 = "Fill 1"
    FILL_2 = "Fill 2"
    EMPTY_1 = "Empty 1"
    EMPTY_2 = "Empty 2"
    POUR_1_TO_2 = "Pour 1->2"
    POUR_2_TO_1 = "Pour 2->1"

    def __str__(self) -> str:
        return self.value


RULE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    WaterJugAction.FILL_1.value: {
        "rule_number": 1,
        "name": "Fill 1",
        "description": "Fill Jug 1 completely from the water supply until full.",
        "precondition": "jug1 < X",
        "effect": "(X, jug2)",
    },
    WaterJugAction.FILL_2.value: {
        "rule_number": 2,
        "name": "Fill 2",
        "description": "Fill Jug 2 completely from the water supply until full.",
        "precondition": "jug2 < Y",
        "effect": "(jug1, Y)",
    },
    WaterJugAction.EMPTY_1.value: {
        "rule_number": 3,
        "name": "Empty 1",
        "description": "Empty Jug 1 completely into the drain.",
        "precondition": "jug1 > 0",
        "effect": "(0, jug2)",
    },
    WaterJugAction.EMPTY_2.value: {
        "rule_number": 4,
        "name": "Empty 2",
        "description": "Empty Jug 2 completely into the drain.",
        "precondition": "jug2 > 0",
        "effect": "(jug1, 0)",
    },
    WaterJugAction.POUR_1_TO_2.value: {
        "rule_number": 5,
        "name": "Pour 1->2",
        "description": "Pour water from Jug 1 into Jug 2 until Jug 2 is full or Jug 1 is empty.",
        "precondition": "jug1 > 0 and jug2 < Y",
        "effect": "(jug1 - amount, jug2 + amount) where amount = min(jug1, Y - jug2)",
    },
    WaterJugAction.POUR_2_TO_1.value: {
        "rule_number": 6,
        "name": "Pour 2->1",
        "description": "Pour water from Jug 2 into Jug 1 until Jug 1 is full or Jug 2 is empty.",
        "precondition": "jug2 > 0 and jug1 < X",
        "effect": "(jug1 + amount, jug2 - amount) where amount = min(jug2, X - jug1)",
    },
}


# ---------------------------------------------------------------------------
# Transition and Algebraic Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TransitionRecord:
    """Detailed record of a single step transition between water jug states."""
    step: int
    from_state: WaterJugState
    action: str
    rule_number: int
    to_state: WaterJugState
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition record to dictionary format."""
        return {
            "step": self.step,
            "from_state": list(self.from_state),
            "action": self.action,
            "rule_number": self.rule_number,
            "to_state": list(self.to_state),
            "description": self.description,
        }


@dataclass
class AlgebraicReachabilityResult:
    """Detailed analysis of algebraic reachability via Bézout's identity and GCD."""
    is_reachable: bool
    capacity_x: int
    capacity_y: int
    target_z: int
    gcd_value: int
    target_jug: str
    bezout_x: int
    bezout_y: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert reachability result to dictionary."""
        return {
            "is_reachable": self.is_reachable,
            "capacity_x": self.capacity_x,
            "capacity_y": self.capacity_y,
            "target_z": self.target_z,
            "target_jug": self.target_jug,
            "gcd_value": self.gcd_value,
            "bezout_x": self.bezout_x,
            "bezout_y": self.bezout_y,
            "reason": self.reason,
        }


@dataclass
class WaterJugSearchResult:
    """Container holding full results of the Water Jug BFS search."""
    capacities: Tuple[int, int]
    target_z: int
    target_jug: str
    is_solvable: bool
    algebraic_check: AlgebraicReachabilityResult
    path: List[WaterJugState] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    transitions: List[TransitionRecord] = field(default_factory=list)
    step_count: int = 0
    nodes_explored: int = 0
    nodes_generated: int = 0
    max_queue_size: int = 0
    execution_time_ms: float = 0.0

    @property
    def arrow_path_string(self) -> str:
        """Format the path of states as an arrow-separated string without spaces in tuples."""
        if not self.path:
            return "No solution path found."
        return " -> ".join(f"({s[0]},{s[1]})" for s in self.path)

    def format_table(self) -> str:
        """Generate a formatted ASCII table of the step-by-step transition trace."""
        if not self.transitions and len(self.path) <= 1:
            if self.is_solvable and len(self.path) == 1:
                return (
                    "+------+--------------+--------+--------------+---------------------------------------+\n"
                    "| Step | Action       | Rule # | State (J1,J2)| Description                           |\n"
                    "+------+--------------+--------+--------------+---------------------------------------+\n"
                    f"|    0 | Initial      |    -   | ({self.path[0][0]:>2},{self.path[0][1]:>2})      | Initial state is already at the goal. |\n"
                    "+------+--------------+--------+--------------+---------------------------------------+"
                )
            return "No transitions executed."

        header = (
            "+------+--------------+--------+--------------+-------------------------------------------------------------+\n"
            "| Step | Action       | Rule # | State (J1,J2)| Description                                                 |\n"
            "+------+--------------+--------+--------------+-------------------------------------------------------------+"
        )
        footer = (
            "+------+--------------+--------+--------------+-------------------------------------------------------------+"
        )
        lines = [header]

        # Initial state row
        init_state = self.path[0]
        lines.append(
            f"|    0 | Initial      |    -   | ({init_state[0]:>2},{init_state[1]:>2})      | Initial state: both jugs empty                              |"
        )

        for tr in self.transitions:
            lines.append(
                f"| {tr.step:>4} | {tr.action:<12} | {tr.rule_number:>6} | ({tr.to_state[0]:>2},{tr.to_state[1]:>2})      | {tr.description:<59} |"
            )

        lines.append(footer)
        return "\n".join(lines)

    def format_ascii_visual_path(self) -> str:
        """Render step-by-step ASCII visual diagrams showing water levels."""
        cap_x, cap_y = self.capacities
        output_blocks = []
        for i, st in enumerate(self.path):
            action_label = "Initial State" if i == 0 else f"Step {i}: {self.actions[i-1]}"
            visual = render_ascii_water_jugs(cap_x, cap_y, st[0], st[1])
            output_blocks.append(f"[{action_label}]\n{visual}")
        return "\n\n".join(output_blocks)

    def format_summary(self) -> str:
        """Return a structured multi-line summary of the search results."""
        cap_x, cap_y = self.capacities
        lines = [
            "=" * 70,
            "WATER JUG PROBLEM BFS SEARCH SUMMARY",
            "=" * 70,
            f"Jug 1 Capacity (X)   : {cap_x} L",
            f"Jug 2 Capacity (Y)   : {cap_y} L",
            f"Target Water (Z)     : {self.target_z} L",
            f"Target Jug Option    : {self.target_jug}",
            f"Algebraic Reachable  : {'Yes' if self.algebraic_check.is_reachable else 'No'}",
            f"GCD(X, Y)            : {self.algebraic_check.gcd_value}",
            f"Bézout Identity      : {self.algebraic_check.bezout_x}*({cap_x}) + {self.algebraic_check.bezout_y}*({cap_y}) = {self.algebraic_check.gcd_value}",
            f"Solvable via BFS     : {'Yes' if self.is_solvable else 'No'}",
            f"Shortest Path Length : {self.step_count} steps ({len(self.path)} states)",
            f"Nodes Explored       : {self.nodes_explored}",
            f"Nodes Generated      : {self.nodes_generated}",
            f"Max Queue Size       : {self.max_queue_size}",
            f"Execution Time       : {self.execution_time_ms:.3f} ms",
            "-" * 70,
            "State Sequence:",
            f"  {self.arrow_path_string}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full search result into serializable dictionary."""
        return {
            "capacities": list(self.capacities),
            "target_z": self.target_z,
            "target_jug": self.target_jug,
            "is_solvable": self.is_solvable,
            "algebraic_check": self.algebraic_check.to_dict(),
            "path": [list(s) for s in self.path],
            "actions": self.actions,
            "transitions": [tr.to_dict() for tr in self.transitions],
            "step_count": self.step_count,
            "nodes_explored": self.nodes_explored,
            "nodes_generated": self.nodes_generated,
            "max_queue_size": self.max_queue_size,
            "execution_time_ms": self.execution_time_ms,
            "arrow_path_string": self.arrow_path_string,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export search result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Mathematical Utilities: Extended GCD & Reachability
# ---------------------------------------------------------------------------

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm.

    Computes integers (gcd, x, y) such that:
        a * x + b * y = gcd(a, b)

    Args:
        a: First non-negative integer.
        b: Second non-negative integer.

    Returns:
        Tuple of (gcd, x, y).
    """
    if a == 0:
        return b, 0, 1
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd_val, x, y


def check_reachability_algebraic(
    capacity_x: int,
    capacity_y: int,
    target_z: int,
    target_jug: Union[int, str] = 1,
) -> AlgebraicReachabilityResult:
    """Verify algebraic reachability of target Z using Bézout's identity and GCD.

    In the classic two-jug problem, any sequence of water operations produces
    amounts that are integer linear combinations of X and Y:
        amount = a * X + b * Y
    By Bézout's Identity, the smallest positive linear combination is gcd(X, Y).
    Therefore, target Z can be measured if and only if:
    1. Z <= max(X, Y) (capacity bound; or Z <= X if target is Jug 1; Z <= Y if Jug 2).
    2. Z % gcd(X, Y) == 0 (divisibility condition).

    Args:
        capacity_x: Capacity of Jug 1 in liters.
        capacity_y: Capacity of Jug 2 in liters.
        target_z: Desired target amount in liters.
        target_jug: Target jug selector (1, 2, or 'any').

    Returns:
        AlgebraicReachabilityResult detailing reachability and mathematical justification.
    """
    jug_str = str(target_jug).lower()
    if jug_str not in ("1", "2", "any"):
        jug_str = "1"

    if capacity_x <= 0 or capacity_y <= 0:
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=0,
            target_jug=jug_str,
            bezout_x=0,
            bezout_y=0,
            reason="Invalid capacities: Capacities X and Y must be strictly positive integers.",
        )

    if target_z < 0:
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=math.gcd(capacity_x, capacity_y),
            target_jug=jug_str,
            bezout_x=0,
            bezout_y=0,
            reason="Invalid target: Target amount Z cannot be negative.",
        )

    # Edge case: Target 0 is always reachable at (0, 0)
    g, bez_x, bez_y = extended_gcd(capacity_x, capacity_y)
    if target_z == 0:
        return AlgebraicReachabilityResult(
            is_reachable=True,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=g,
            target_jug=jug_str,
            bezout_x=bez_x,
            bezout_y=bez_y,
            reason="Target 0 is trivially reachable in the initial empty state (0, 0).",
        )

    # Capacity constraints
    if jug_str == "1" and target_z > capacity_x:
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=g,
            target_jug=jug_str,
            bezout_x=bez_x,
            bezout_y=bez_y,
            reason=f"Capacity exceeded: Target {target_z}L exceeds Jug 1 capacity {capacity_x}L.",
        )
    if jug_str == "2" and target_z > capacity_y:
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=g,
            target_jug=jug_str,
            bezout_x=bez_x,
            bezout_y=bez_y,
            reason=f"Capacity exceeded: Target {target_z}L exceeds Jug 2 capacity {capacity_y}L.",
        )
    if jug_str == "any" and target_z > max(capacity_x, capacity_y):
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=g,
            target_jug=jug_str,
            bezout_x=bez_x,
            bezout_y=bez_y,
            reason=f"Capacity exceeded: Target {target_z}L exceeds maximum jug capacity {max(capacity_x, capacity_y)}L.",
        )

    # GCD divisibility constraint (Bézout's identity)
    if target_z % g != 0:
        return AlgebraicReachabilityResult(
            is_reachable=False,
            capacity_x=capacity_x,
            capacity_y=capacity_y,
            target_z=target_z,
            gcd_value=g,
            target_jug=jug_str,
            bezout_x=bez_x,
            bezout_y=bez_y,
            reason=(
                f"GCD divisibility violation: Target {target_z} is not divisible by "
                f"gcd({capacity_x}, {capacity_y}) = {g}. By Bézout's identity, all measurable "
                f"quantities must be integer multiples of gcd(X, Y)."
            ),
        )

    return AlgebraicReachabilityResult(
        is_reachable=True,
        capacity_x=capacity_x,
        capacity_y=capacity_y,
        target_z=target_z,
        gcd_value=g,
        target_jug=jug_str,
        bezout_x=bez_x,
        bezout_y=bez_y,
        reason=(
            f"Algebraically reachable: Target {target_z} <= capacity bounds and "
            f"{target_z} % gcd({capacity_x}, {capacity_y}) == 0 (gcd = {g}). "
            f"Bézout representation: {bez_x}*({capacity_x}) + {bez_y}*({capacity_y}) = {g}."
        ),
    )


# ---------------------------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------------------------

def format_state_arrow_path(path: Sequence[WaterJugState]) -> str:
    """Format state path into standard arrow format without internal spaces.

    Example:
        [(0,0), (4,0), (1,3)] -> "(0,0) -> (4,0) -> (1,3)"
    """
    if not path:
        return "()"
    return " -> ".join(f"({s[0]},{s[1]})" for s in path)


def render_ascii_water_jugs(cap_x: int, cap_y: int, amt_x: int, amt_y: int) -> str:
    """Render a visual ASCII representation of the two water jugs.

    Args:
        cap_x: Capacity of Jug 1.
        cap_y: Capacity of Jug 2.
        amt_x: Current water volume in Jug 1.
        amt_y: Current water volume in Jug 2.

    Returns:
        Multi-line string containing the side-by-side ASCII art.
    """
    max_height = max(cap_x, cap_y)
    lines = []
    lines.append(f"  Jug 1 ({amt_x}/{cap_x} L)       Jug 2 ({amt_y}/{cap_y} L)")
    lines.append(f"  +-------------+       +-------------+")

    for level in range(max_height, 0, -1):
        # Jug 1 cell
        if level > cap_x:
            j1_str = "               "
        elif level <= amt_x:
            j1_str = "  |~~~~~~~~~~~|  "
        else:
            j1_str = "  |           |  "

        # Jug 2 cell
        if level > cap_y:
            j2_str = "               "
        elif level <= amt_y:
            j2_str = "  |~~~~~~~~~~~|  "
        else:
            j2_str = "  |           |  "

        lines.append(f"{j1_str}     {j2_str}")

    lines.append("  +-------------+       +-------------+")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Solver: WaterJugSolver
# ---------------------------------------------------------------------------

class WaterJugSolver:
    """Breadth-First Search (BFS) Solver for the classic Water Jug Problem.

    This class encapsulates state-space exploration, the 6 canonical production
    rules, cycle detection, algebraic reachability checks, and shortest path
    reconstruction.
    """

    def __init__(
        self,
        capacity_x: int,
        capacity_y: int,
        target_z: int,
        target_jug: Union[int, str] = 1,
    ) -> None:
        """Initialize the water jug solver.

        Args:
            capacity_x: Capacity of Jug 1 in liters.
            capacity_y: Capacity of Jug 2 in liters.
            target_z: Target amount of water in liters.
            target_jug: Which jug must hold the target amount:
                - 1: Target must be in Jug 1 (default, matching classic syllabus).
                - 2: Target must be in Jug 2.
                - 'any': Target can be in either Jug 1 or Jug 2.
        """
        if capacity_x <= 0 or capacity_y <= 0:
            raise ValueError(f"Jug capacities must be strictly positive integers (got X={capacity_x}, Y={capacity_y}).")
        if target_z < 0:
            raise ValueError(f"Target Z must be non-negative (got Z={target_z}).")

        self.capacity_x = int(capacity_x)
        self.capacity_y = int(capacity_y)
        self.target_z = int(target_z)

        # Normalize target_jug option
        t_str = str(target_jug).strip().lower()
        if t_str in ("1", "jug1", "x"):
            self.target_jug: str = "1"
        elif t_str in ("2", "jug2", "y"):
            self.target_jug = "2"
        elif t_str in ("any", "either", "*", "both"):
            self.target_jug = "any"
        else:
            self.target_jug = "1"

    def is_goal_state(self, state: WaterJugState) -> bool:
        """Evaluate if the given state satisfies the target goal condition."""
        j1, j2 = state
        if self.target_jug == "1":
            return j1 == self.target_z
        elif self.target_jug == "2":
            return j2 == self.target_z
        else:  # 'any'
            return j1 == self.target_z or j2 == self.target_z

    def check_algebraic_reachability(self) -> AlgebraicReachabilityResult:
        """Check reachability algebraically using Bézout's identity and GCD."""
        return check_reachability_algebraic(
            self.capacity_x, self.capacity_y, self.target_z, self.target_jug
        )

    def get_successors(
        self, state: WaterJugState
    ) -> List[Tuple[WaterJugState, str, int, str]]:
        """Generate valid successors for the 6 canonical production rules.

        Rules are generated in strict priority order:
        1. Fill 1: Fill Jug 1 to capacity X -> (X, j2)
        2. Fill 2: Fill Jug 2 to capacity Y -> (j1, Y)
        3. Empty 1: Empty Jug 1 completely -> (0, j2)
        4. Empty 2: Empty Jug 2 completely -> (j1, 0)
        5. Pour 1->2: Pour Jug 1 into Jug 2 until full or empty
        6. Pour 2->1: Pour Jug 2 into Jug 1 until full or empty

        Preconditions are checked to discard redundant states (self-loops)
        and invalid transitions (negative water, overflow).

        Args:
            state: Current tuple state (jug1_amt, jug2_amt).

        Returns:
            List of tuples: (next_state, action_name, rule_number, description).
        """
        j1, j2 = state
        X = self.capacity_x
        Y = self.capacity_y
        successors: List[Tuple[WaterJugState, str, int, str]] = []

        # Rule 1: Fill 1
        if j1 < X:
            nxt = (X, j2)
            desc = f"Fill Jug 1 from tap to capacity ({X}L). Result: ({X}, {j2})"
            successors.append((nxt, WaterJugAction.FILL_1.value, 1, desc))

        # Rule 2: Fill 2
        if j2 < Y:
            nxt = (j1, Y)
            desc = f"Fill Jug 2 from tap to capacity ({Y}L). Result: ({j1}, {Y})"
            successors.append((nxt, WaterJugAction.FILL_2.value, 2, desc))

        # Rule 3: Empty 1
        if j1 > 0:
            nxt = (0, j2)
            desc = f"Empty Jug 1 into the drain. Result: (0, {j2})"
            successors.append((nxt, WaterJugAction.EMPTY_1.value, 3, desc))

        # Rule 4: Empty 2
        if j2 > 0:
            nxt = (j1, 0)
            desc = f"Empty Jug 2 into the drain. Result: ({j1}, 0)"
            successors.append((nxt, WaterJugAction.EMPTY_2.value, 4, desc))

        # Rule 5: Pour 1 -> 2
        if j1 > 0 and j2 < Y:
            amount = min(j1, Y - j2)
            if amount > 0:
                nxt = (j1 - amount, j2 + amount)
                desc = (
                    f"Pour {amount}L from Jug 1 into Jug 2 (Jug 2 {'filled' if nxt[1] == Y else 'partial'}). "
                    f"Result: ({nxt[0]}, {nxt[1]})"
                )
                successors.append((nxt, WaterJugAction.POUR_1_TO_2.value, 5, desc))

        # Rule 6: Pour 2 -> 1
        if j2 > 0 and j1 < X:
            amount = min(j2, X - j1)
            if amount > 0:
                nxt = (j1 + amount, j2 - amount)
                desc = (
                    f"Pour {amount}L from Jug 2 into Jug 1 (Jug 1 {'filled' if nxt[0] == X else 'partial'}). "
                    f"Result: ({nxt[0]}, {nxt[1]})"
                )
                successors.append((nxt, WaterJugAction.POUR_2_TO_1.value, 6, desc))

        return successors

    def solve_bfs(
        self, start_state: WaterJugState = (0, 0)
    ) -> WaterJugSearchResult:
        """Execute Breadth-First Search to find the shortest action sequence.

        Explores states using a FIFO queue (`collections.deque`), recording parent
        pointers and action descriptions to reconstruct the optimal state sequence.

        Args:
            start_state: Initial starting state (default: (0, 0)).

        Returns:
            WaterJugSearchResult object containing path, transitions, and metrics.
        """
        start_time = time.perf_counter()
        algebraic = self.check_algebraic_reachability()

        # If algebraically impossible, return immediately without wasting search cycles
        if not algebraic.is_reachable:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return WaterJugSearchResult(
                capacities=(self.capacity_x, self.capacity_y),
                target_z=self.target_z,
                target_jug=self.target_jug,
                is_solvable=False,
                algebraic_check=algebraic,
                path=[],
                actions=[],
                transitions=[],
                step_count=0,
                nodes_explored=0,
                nodes_generated=0,
                max_queue_size=0,
                execution_time_ms=elapsed_ms,
            )

        # Validate start state boundaries
        s1, s2 = start_state
        if s1 < 0 or s1 > self.capacity_x or s2 < 0 or s2 > self.capacity_y:
            raise ValueError(
                f"Start state ({s1}, {s2}) violates capacity constraints (X={self.capacity_x}, Y={self.capacity_y})."
            )

        # Check if initial state is already at the goal
        if self.is_goal_state(start_state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return WaterJugSearchResult(
                capacities=(self.capacity_x, self.capacity_y),
                target_z=self.target_z,
                target_jug=self.target_jug,
                is_solvable=True,
                algebraic_check=algebraic,
                path=[start_state],
                actions=[],
                transitions=[],
                step_count=0,
                nodes_explored=1,
                nodes_generated=1,
                max_queue_size=1,
                execution_time_ms=elapsed_ms,
            )

        # FIFO Queue holds current frontier states
        queue: deque[WaterJugState] = deque([start_state])

        # Parent map tracks: child_state -> (parent_state, action, rule_number, description)
        parent_map: Dict[
            WaterJugState,
            Optional[Tuple[WaterJugState, str, int, str]],
        ] = {start_state: None}

        nodes_explored = 0
        nodes_generated = 1
        max_queue_size = 1
        goal_state: Optional[WaterJugState] = None

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current = queue.popleft()
            nodes_explored += 1

            # Generate successors for 6 rules
            successors = self.get_successors(current)

            for nxt_state, action, rule_num, desc in successors:
                if nxt_state not in parent_map:
                    parent_map[nxt_state] = (current, action, rule_num, desc)
                    nodes_generated += 1

                    # Goal test on node generation (standard optimization in unit-cost BFS)
                    if self.is_goal_state(nxt_state):
                        goal_state = nxt_state
                        break

                    queue.append(nxt_state)

            if goal_state is not None:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if goal_state is None:
            return WaterJugSearchResult(
                capacities=(self.capacity_x, self.capacity_y),
                target_z=self.target_z,
                target_jug=self.target_jug,
                is_solvable=False,
                algebraic_check=algebraic,
                path=[],
                actions=[],
                transitions=[],
                step_count=0,
                nodes_explored=nodes_explored,
                nodes_generated=nodes_generated,
                max_queue_size=max_queue_size,
                execution_time_ms=elapsed_ms,
            )

        # Backtrack to reconstruct path and transitions
        path: List[WaterJugState] = []
        actions: List[str] = []
        raw_transitions: List[Tuple[WaterJugState, str, int, WaterJugState, str]] = []

        curr = goal_state
        while curr is not None:
            path.append(curr)
            entry = parent_map[curr]
            if entry is not None:
                p_state, act, r_num, desc = entry
                actions.append(act)
                raw_transitions.append((p_state, act, r_num, curr, desc))
                curr = p_state
            else:
                curr = None

        path.reverse()
        actions.reverse()
        raw_transitions.reverse()

        transitions: List[TransitionRecord] = [
            TransitionRecord(
                step=i + 1,
                from_state=t[0],
                action=t[1],
                rule_number=t[2],
                to_state=t[3],
                description=t[4],
            )
            for i, t in enumerate(raw_transitions)
        ]

        return WaterJugSearchResult(
            capacities=(self.capacity_x, self.capacity_y),
            target_z=self.target_z,
            target_jug=self.target_jug,
            is_solvable=True,
            algebraic_check=algebraic,
            path=path,
            actions=actions,
            transitions=transitions,
            step_count=len(actions),
            nodes_explored=nodes_explored,
            nodes_generated=nodes_generated,
            max_queue_size=max_queue_size,
            execution_time_ms=elapsed_ms,
        )

    def find_all_reachable_states(
        self, start_state: WaterJugState = (0, 0)
    ) -> Set[WaterJugState]:
        """Explore and return all reachable states from start_state.

        Args:
            start_state: Initial state (default: (0, 0)).

        Returns:
            Set of all reachable (jug1, jug2) tuple states.
        """
        queue: deque[WaterJugState] = deque([start_state])
        visited: Set[WaterJugState] = {start_state}

        while queue:
            current = queue.popleft()
            for nxt_state, _, _, _ in self.get_successors(current):
                if nxt_state not in visited:
                    visited.add(nxt_state)
                    queue.append(nxt_state)

        return visited


# ---------------------------------------------------------------------------
# High-Level Helper Functions
# ---------------------------------------------------------------------------

def solve_water_jug_bfs(
    capacity_x: int,
    capacity_y: int,
    target_z: int,
    target_jug: Union[int, str] = 1,
    start_state: WaterJugState = (0, 0),
) -> WaterJugSearchResult:
    """Convenience function to solve the Water Jug Problem using BFS.

    Args:
        capacity_x: Capacity of Jug 1 in liters.
        capacity_y: Capacity of Jug 2 in liters.
        target_z: Target water volume in liters.
        target_jug: Target jug selector (1, 2, or 'any'). Default is 1.
        start_state: Starting state tuple (default: (0, 0)).

    Returns:
        WaterJugSearchResult object containing the solution sequence.
    """
    solver = WaterJugSolver(
        capacity_x=capacity_x,
        capacity_y=capacity_y,
        target_z=target_z,
        target_jug=target_jug,
    )
    return solver.solve_bfs(start_state=start_state)


def get_water_jug_viva_reference() -> Dict[str, Dict[str, str]]:
    """Authoritative Viva Voce study guide for the Water Jug Problem."""
    return {
        "1. State Space Representation": {
            "question": "How is the state space represented in the Water Jug Problem, and what is its size?",
            "answer": (
                "A state is represented as an ordered tuple of integers (x, y), where x denotes "
                "the current water volume in Jug 1 (0 <= x <= X) and y denotes the volume in "
                "Jug 2 (0 <= y <= Y). The initial state is (0, 0). The theoretical upper bound "
                "of the state space is (X + 1) * (Y + 1). However, in any valid sequence of filling, "
                "emptying, and pouring, at least one jug is always empty or full at each step. "
                "Thus, the reachable state space lies on the perimeter of the grid and specific diagonal "
                "chords, drastically reducing the effective number of reachable states."
            ),
        },
        "2. Action Rules Definition": {
            "question": "Define the 6 production rules used in the Water Jug Problem transition model.",
            "answer": (
                "The 6 deterministic production rules are:\n"
                "1. Fill 1: (x, y) -> (X, y) if x < X (fill Jug 1 from tap).\n"
                "2. Fill 2: (x, y) -> (x, Y) if y < Y (fill Jug 2 from tap).\n"
                "3. Empty 1: (x, y) -> (0, y) if x > 0 (empty Jug 1 into drain).\n"
                "4. Empty 2: (x, y) -> (0, y) if y > 0 (empty Jug 2 into drain).\n"
                "5. Pour 1->2: (x, y) -> (x - d, y + d) where d = min(x, Y - y) if x > 0 and y < Y.\n"
                "6. Pour 2->1: (x, y) -> (x + d, y - d) where d = min(y, X - x) if y > 0 and x < X.\n"
                "Preconditions prune self-loops (e.g., pouring into a full jug or emptying an empty jug), "
                "and negative amounts are strictly discarded."
            ),
        },
        "3. Unsolvable States & GCD Rule": {
            "question": "What is the GCD rule for unsolvable states, and how is it proven algebraically?",
            "answer": (
                "By Bézout's Identity, for any positive integers X and Y, the set of integer linear "
                "combinations {a*X + b*Y | a, b in Z} equals the set of all integer multiples of gcd(X, Y). "
                "Every fill operation adds X or Y; every empty operation subtracts X or Y; pouring conserves total volume. "
                "Therefore, the water content in either jug at any step must be an integer linear combination "
                "of X and Y, and hence must be divisible by gcd(X, Y).\n"
                "Thus, target Z is solvable in one of the jugs IF AND ONLY IF:\n"
                "1. Z <= max(X, Y) (capacity bound; Z <= X if target is Jug 1).\n"
                "2. Z % gcd(X, Y) == 0.\n"
                "For example, with capacities 6L and 4L, gcd(6, 4) = 2. Any odd target (such as 1L, 3L, 5L) "
                "is mathematically impossible to measure."
            ),
        },
        "4. BFS Optimality and Completeness": {
            "question": "Why is Breadth-First Search (BFS) the preferred algorithm for this problem?",
            "answer": (
                "BFS explores states level-by-level (shallowest depth first). Because all operations have uniform "
                "step cost (c = 1), BFS guarantees finding the optimal (shortest) sequence of operations to reach "
                "the target. Furthermore, because the state space is finite (bounded by (X+1)*(Y+1)), maintaining a "
                "visited set guarantees completeness (it will find the solution if one exists, or terminate with "
                "failure without getting trapped in cycles)."
            ),
        },
        "5. Time and Space Complexity": {
            "question": "What are the time and space complexities of the BFS Water Jug solver?",
            "answer": (
                "Time Complexity: O(X * Y). There are at most (X + 1) * (Y + 1) reachable states, and for each state, "
                "at most 6 transitions are evaluated in O(1) time.\n"
                "Space Complexity: O(X * Y). The FIFO queue and the visited set / parent map store at most "
                "(X + 1) * (Y + 1) state tuples in memory."
            ),
        },
    }
