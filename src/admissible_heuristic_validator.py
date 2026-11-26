"""Admissible Heuristic Validator (Question 33).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟡 Intermediate
Practical Type: Validation
Core AI Concept: Admissibility (h(n) <= h*(n))
Python Concepts Tested: Dictionary iteration, reverse graph search, priority queues.
Expected Algorithm: Reverse UCS (Dijkstra's from goal) & Admissibility Validation.

Problem Statement:
    Given a graph, the true minimum costs from every node to the goal, and a
    heuristic dictionary h(n), write a Python function to programmatically
    determine if the provided heuristic is Admissible.

Task Requirements:
    1. Calculate true cost h*(n) for all nodes using a reverse UCS from the goal.
    2. Compare provided h(n) against h*(n).
    3. Identify any node where h(n) > h*(n).
    4. Output True (Admissible) or False (Not Admissible with counterexample).

Sample Input:
    h(A) = 10, True Cost A->Goal = 8
Sample Output:
    False. Heuristic overestimates at node A (10 > 8).

Optional Challenge:
    Automatically lower the heuristic values to make them admissible:
    h_repaired(n) = min(h(n), h*(n)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import json
import math
import time
from typing import (
    Any,
    Dict,
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
# Type Aliases & Constant Specifications
# ============================================================================

NodeId = str
Cost = float
Weight = float

# Graph formats supported:
# 1. Dict[NodeId, Dict[NodeId, Cost]]: {'A': {'B': 3.0, 'C': 6.0}}
# 2. Dict[NodeId, List[Tuple[NodeId, Cost]]]: {'A': [('B', 3.0), ('C', 6.0)]}
GraphAdjacency = Dict[NodeId, Dict[NodeId, Cost]]
GraphInput = Union[
    Mapping[NodeId, Mapping[NodeId, Union[int, float]]],
    Mapping[NodeId, Sequence[Tuple[NodeId, Union[int, float]]]],
]

# Canonical syllabus sample specification
CANONICAL_GOAL: NodeId = "Goal"

# Directed graph where A -> B (3), B -> Goal (5) => Total cost A->Goal = 8
# Also includes alternative path A -> C (6), C -> Goal (4) => Total cost 10
# Node D is an independent path D -> Goal (7)
CANONICAL_GRAPH: GraphAdjacency = {
    "A": {"B": 3.0, "C": 6.0},
    "B": {"Goal": 5.0},
    "C": {"Goal": 4.0},
    "D": {"Goal": 7.0},
    "Goal": {},
}

# Sample heuristic with overestimation at A: h(A)=10 > h*(A)=8
CANONICAL_HEURISTIC_OVERESTIMATING: Dict[NodeId, float] = {
    "A": 10.0,
    "B": 4.0,
    "C": 3.0,
    "D": 6.0,
    "Goal": 0.0,
}

# Admissible counterpart heuristic where h(A) = 7.0 <= 8.0
CANONICAL_HEURISTIC_ADMISSIBLE: Dict[NodeId, float] = {
    "A": 7.0,
    "B": 4.0,
    "C": 3.0,
    "D": 6.0,
    "Goal": 0.0,
}

SYLLABUS_SAMPLE_OUTPUT: str = (
    "False. Heuristic overestimates at node A (10 > 8)."
)


# ============================================================================
# Dataclasses & Violation Records
# ============================================================================

@dataclass(frozen=True)
class NodeViolation:
    """Represents an admissibility violation where h(n) > h*(n).

    Attributes:
        node: The identifier of the violating graph node.
        h_val: The user-supplied heuristic estimate h(n).
        h_star: The true minimum cost to the goal h*(n).
        difference: The magnitude of overestimation (h(n) - h*(n)).
    """

    node: NodeId
    h_val: float
    h_star: float
    difference: float

    def format_counterexample(self) -> str:
        """Format counterexample string matching syllabus specification.

        Returns:
            Formatted string such as 'Heuristic overestimates at node A (10 > 8).'
        """
        h_str = f"{self.h_val:g}"
        h_star_str = f"{self.h_star:g}" if not math.isinf(self.h_star) else "inf"
        return f"Heuristic overestimates at node {self.node} ({h_str} > {h_star_str})."


@dataclass
class AdmissibilityReport:
    """Complete validation report for a heuristic function against a graph.

    Attributes:
        is_admissible: True if h(n) <= h*(n) for all evaluated nodes; False otherwise.
        violations: List of all nodes where admissibility is violated.
        true_costs: Dictionary mapping each node to its true shortest cost h*(n).
        heuristics: Dictionary of user-provided heuristic values h(n).
        goal_node: The target goal node identifier.
        counterexample: Concise counterexample string or success confirmation.
        repaired_heuristics: Admissible heuristic obtained by clamping h(n) to h*(n).
        execution_time_ms: Wall-clock time taken for validation in milliseconds.
    """

    is_admissible: bool
    violations: List[NodeViolation]
    true_costs: Dict[NodeId, float]
    heuristics: Dict[NodeId, float]
    goal_node: NodeId
    counterexample: str
    repaired_heuristics: Dict[NodeId, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "is_admissible": self.is_admissible,
            "goal_node": self.goal_node,
            "violations_count": len(self.violations),
            "violations": [asdict(v) for v in self.violations],
            "counterexample": self.counterexample,
            "true_costs": {
                k: ("inf" if math.isinf(v) else v) for k, v in self.true_costs.items()
            },
            "heuristics": self.heuristics,
            "repaired_heuristics": {
                k: ("inf" if math.isinf(v) else v)
                for k, v in self.repaired_heuristics.items()
            },
            "execution_time_ms": round(self.execution_time_ms, 4),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Graph Normalization & Validation Functions
# ============================================================================

def normalize_graph(graph_input: GraphInput) -> GraphAdjacency:
    """Normalize and validate various graph representation formats.

    Supports:
        - Nested dictionary: {'A': {'B': 3.0, 'C': 5.0}}
        - Adjacency list: {'A': [('B', 3.0), ('C', 5.0)]}

    Args:
        graph_input: Raw graph mapping.

    Returns:
        Standardized nested dictionary GraphAdjacency.

    Raises:
        ValueError: If edge weights are non-positive or structure is invalid.
    """
    normalized: GraphAdjacency = {}

    # First pass: collect all explicit source nodes
    for src_raw, neighbors in graph_input.items():
        src = str(src_raw)
        if src not in normalized:
            normalized[src] = {}

        if isinstance(neighbors, Mapping):
            for dst_raw, weight in neighbors.items():
                dst = str(dst_raw)
                w = float(weight)
                if w <= 0:
                    raise ValueError(
                        f"Graph constraint violated: Edge ({src} -> {dst}) has non-positive "
                        f"weight {w}. Graph must have strictly positive weights."
                    )
                normalized[src][dst] = w
                if dst not in normalized:
                    normalized[dst] = {}
        elif isinstance(neighbors, Sequence):
            for edge in neighbors:
                if not (isinstance(edge, (tuple, list)) and len(edge) == 2):
                    raise ValueError(
                        f"Invalid edge format in node {src}: expected (destination, weight) tuple."
                    )
                dst = str(edge[0])
                w = float(edge[1])
                if w <= 0:
                    raise ValueError(
                        f"Graph constraint violated: Edge ({src} -> {dst}) has non-positive "
                        f"weight {w}. Graph must have strictly positive weights."
                    )
                normalized[src][dst] = w
                if dst not in normalized:
                    normalized[dst] = {}
        else:
            raise ValueError(
                f"Unsupported neighbors type for node {src}: {type(neighbors)}"
            )

    return normalized


def compute_reverse_graph(graph: GraphAdjacency) -> GraphAdjacency:
    """Compute the transpose (reverse) graph G^R = (V, E^R).

    In a directed graph, if an edge u -> v has cost c, then in the reversed
    graph there is an edge v -> u with the exact same cost c.

    Args:
        graph: Forward graph adjacency.

    Returns:
        Reversed graph adjacency mapping where all edge directions are flipped.
    """
    reversed_graph: GraphAdjacency = {node: {} for node in graph}

    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if v not in reversed_graph:
                reversed_graph[v] = {}
            reversed_graph[v][u] = weight

    return reversed_graph


# ============================================================================
# Reverse Uniform Cost Search (Reverse UCS / Dijkstra)
# ============================================================================

def compute_true_costs(
    graph: GraphInput,
    goal_node: NodeId,
) -> Dict[NodeId, float]:
    """Calculate true minimum cost h*(n) from every node to the goal using reverse UCS.

    Running Dijkstra's algorithm from the goal on the reversed graph solves the
    Single-Target Shortest Path problem for all nodes simultaneously in
    O((V + E) log V) time.

    Args:
        graph: Forward graph adjacency input.
        goal_node: The designated target node.

    Returns:
        Dictionary mapping each node in the graph to its true shortest cost h*(n).
        Nodes unable to reach the goal are mapped to float('inf').

    Raises:
        ValueError: If goal_node is not found in the graph.
    """
    norm_graph = normalize_graph(graph)
    goal = str(goal_node)

    if goal not in norm_graph:
        raise ValueError(
            f"Goal node '{goal}' does not exist in graph nodes: {list(norm_graph.keys())}"
        )

    # Step 1: Transpose the graph
    reversed_graph = compute_reverse_graph(norm_graph)

    # Step 2: Initialize Dijkstra distances from goal on reversed graph
    true_costs: Dict[NodeId, float] = {node: float("inf") for node in norm_graph}
    true_costs[goal] = 0.0

    # Priority queue stores tuples: (distance, insertion_counter, node)
    counter = 0
    pq: List[Tuple[float, int, NodeId]] = [(0.0, counter, goal)]
    visited: Set[NodeId] = set()

    while pq:
        curr_dist, _, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        # Explore incoming edges in forward graph (outgoing in reversed graph)
        for predecessor, edge_cost in reversed_graph.get(u, {}).items():
            new_dist = curr_dist + edge_cost
            if new_dist < true_costs[predecessor]:
                true_costs[predecessor] = new_dist
                counter += 1
                heapq.heappush(pq, (new_dist, counter, predecessor))

    return true_costs


# ============================================================================
# Heuristic Validation & Repair Engine
# ============================================================================

def make_admissible(
    heuristic: Mapping[NodeId, float],
    true_costs: Mapping[NodeId, float],
) -> Dict[NodeId, float]:
    """Automatically lower heuristic values to make them admissible (Optional Challenge).

    For every node n:
        h_repaired(n) = min(h(n), h*(n))
    Ensures h_repaired(n) <= h*(n) and h_repaired(n) >= 0.

    Args:
        heuristic: Provided heuristic dictionary.
        true_costs: Computed true minimum costs h*(n).

    Returns:
        Repaired heuristic dictionary with overestimations clamped to h*(n).
    """
    repaired: Dict[NodeId, float] = {}
    for node, h_val in heuristic.items():
        h_star = true_costs.get(node, float("inf"))
        # Clamping logic: Cannot exceed h*(n), cannot be negative
        if math.isinf(h_star):
            # If unreachable, keep original non-negative estimate
            repaired[node] = max(0.0, float(h_val))
        else:
            clamped = min(float(h_val), float(h_star))
            repaired[node] = max(0.0, clamped)
    return repaired


def validate_admissibility(
    graph: GraphInput,
    goal_node: NodeId,
    heuristic: Mapping[NodeId, Union[int, float]],
    check_all_nodes: bool = True,
) -> AdmissibilityReport:
    """Programmatically determine if the provided heuristic is Admissible.

    Core Algorithm:
        1. Calculate true cost h*(n) for all nodes using a reverse UCS from goal.
        2. Compare provided h(n) against h*(n).
        3. Identify any node where h(n) > h*(n).
        4. Output True (Admissible) or False (Not Admissible with counterexample).

    Args:
        graph: Forward graph structure.
        goal_node: Target goal node.
        heuristic: Dictionary containing heuristic estimates h(n).
        check_all_nodes: If True, evaluates all nodes in heuristic dict.

    Returns:
        AdmissibilityReport containing boolean result, list of violations,
        true costs, counterexample string, and repaired heuristic.
    """
    start_time = time.perf_counter()

    # Step 1: Calculate true costs h*(n) via Reverse UCS
    true_costs = compute_true_costs(graph, goal_node)
    goal = str(goal_node)

    # Clean heuristics to standard float dictionary
    clean_heuristic: Dict[NodeId, float] = {
        str(k): float(v) for k, v in heuristic.items()
    }

    # Step 2 & 3: Compare h(n) against h*(n) and identify violations
    violations: List[NodeViolation] = []

    # Sort nodes for deterministic counterexample reporting
    evaluated_nodes = sorted(clean_heuristic.keys())

    for node in evaluated_nodes:
        h_val = clean_heuristic[node]
        h_star = true_costs.get(node, float("inf"))

        # Floating-point tolerance check
        if h_val > h_star and not math.isclose(h_val, h_star, rel_tol=1e-9, abs_tol=1e-9):
            diff = h_val - h_star
            violations.append(
                NodeViolation(
                    node=node,
                    h_val=h_val,
                    h_star=h_star,
                    difference=diff,
                )
            )

    is_admissible = len(violations) == 0

    # Step 4: Construct concise counterexample matching sample output format
    if is_admissible:
        counterexample_str = "True. Heuristic is admissible (h(n) <= h*(n) for all nodes)."
    else:
        # Highlight primary counterexample
        primary = violations[0]
        counterexample_str = f"False. {primary.format_counterexample()}"

    # Optional Challenge: Repair heuristics to make them admissible
    repaired = make_admissible(clean_heuristic, true_costs)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return AdmissibilityReport(
        is_admissible=is_admissible,
        violations=violations,
        true_costs=true_costs,
        heuristics=clean_heuristic,
        goal_node=goal,
        counterexample=counterexample_str,
        repaired_heuristics=repaired,
        execution_time_ms=elapsed_ms,
    )


# ============================================================================
# Canonical Sample Generators & Formatting Helpers
# ============================================================================

def get_canonical_sample_graph() -> Tuple[GraphAdjacency, NodeId, Dict[NodeId, float]]:
    """Return the canonical graph, goal, and overestimating heuristic.

    Returns:
        Tuple of (canonical_graph, canonical_goal, canonical_heuristic).
    """
    return CANONICAL_GRAPH, CANONICAL_GOAL, CANONICAL_HEURISTIC_OVERESTIMATING


def format_comparison_table(report: AdmissibilityReport) -> str:
    """Format an ASCII comparison table showing h(n), h*(n), and admissibility status.

    Args:
        report: Completed AdmissibilityReport.

    Returns:
        Nicely aligned Markdown / ASCII table string.
    """
    lines: List[str] = [
        "| Node | h(n) (Heuristic) | h*(n) (True Cost) | Status | Deviation (h - h*) | Repaired h(n) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_nodes = sorted(
        set(report.heuristics.keys()).union(set(report.true_costs.keys()))
    )

    for n in all_nodes:
        h_val = report.heuristics.get(n, None)
        h_star = report.true_costs.get(n, float("inf"))
        repaired_val = report.repaired_heuristics.get(n, None)

        h_str = f"{h_val:g}" if h_val is not None else "N/A"
        h_star_str = f"{h_star:g}" if not math.isinf(h_star) else "inf"
        repaired_str = f"{repaired_val:g}" if repaired_val is not None else "N/A"

        if h_val is None:
            status = "Unchecked"
            diff_str = "-"
        elif h_val > h_star and not math.isclose(h_val, h_star, rel_tol=1e-9):
            status = "❌ Overestimates"
            diff = h_val - h_star
            diff_str = f"+{diff:g}"
        else:
            status = "✅ Admissible"
            diff_str = f"{h_val - h_star:g}" if not math.isinf(h_star) else "-inf"

        lines.append(
            f"| {n} | {h_str} | {h_star_str} | {status} | {diff_str} | {repaired_str} |"
        )

    return "\n".join(lines)


def render_ascii_graph(graph: GraphAdjacency, goal: NodeId) -> str:
    """Render a text representation of the graph connections and weights.

    Args:
        graph: Forward graph adjacency.
        goal: Target goal node identifier.

    Returns:
        Formatted ASCII layout of nodes, edges, and target goal.
    """
    lines: List[str] = [
        "Directed Graph Topological Layout:",
        "==================================",
    ]
    for u in sorted(graph.keys()):
        targets = graph[u]
        if not targets:
            goal_marker = " [GOAL]" if u == goal else ""
            lines.append(f"  ({u}){goal_marker} -> (None)")
        else:
            edge_strs = [f"-> ({v}) [cost={w:g}]" for v, w in sorted(targets.items())]
            lines.append(f"  ({u}) " + ", ".join(edge_strs))
    return "\n".join(lines)


# ============================================================================
# Theoretical Analysis & University Viva Voce Reference
# ============================================================================

def get_theoretical_analysis() -> str:
    """Provide in-depth theoretical analysis of heuristic admissibility."""
    return """================================================================================
THEORETICAL FOUNDATIONS: ADMISSIBLE HEURISTIC VALIDATION
================================================================================

1. Mathematical Definition of Admissibility:
   - A heuristic function h(n) is said to be ADMISSIBLE if for every state n in
     the state space:
         0 <= h(n) <= h*(n)
     where h*(n) denotes the true minimum cost of the optimal path from node n
     to the nearest goal state.
   - Intuitively, an admissible heuristic is OPTIMISTIC. It never overestimates
     the true cost remaining to achieve the goal.

2. Single-Target Reverse UCS (Dijkstra on Transpose Graph):
   - Problem: Given directed graph G = (V, E) and goal G, compute h*(n) for all n.
   - Forward Naive Approach: Run Dijkstra |V| times, once from each node n to G.
     Time Complexity: O(|V| * (|V| + |E|) log |V|). Highly redundant!
   - Optimal Reverse Approach: Transpose graph G to G^R by reversing all directed
     edges (u -> v becomes v -> u with identical weight c(u, v)).
   - Run a single Dijkstra (UCS) on G^R originating at node G.
     The distance d(G, u) in G^R is mathematically identical to the shortest path
     distance from u to G in G.
     Time Complexity: O((|V| + |E|) log |V|). Single pass!

3. Why Overestimation Breaks Optimality in A*:
   - Recall the A* evaluation function:
         f(n) = g(n) + h(n)
   - Suppose path P* is the true optimal path with cost C*, and P' is a suboptimal
     path leading to a goal with cost C' > C*.
   - Suppose node n lies on the optimal path P*, but its heuristic severely
     overestimates: h(n) > h*(n), such that f(n) = g(n) + h(n) > C'.
   - A* expands nodes in order of non-decreasing f(n). Because f(n) > C', the
     suboptimal goal node (with f = C') will be popped from the priority queue
     and returned BEFORE node n is ever expanded!
   - Result: A* returns a suboptimal solution path, breaking its fundamental
     guarantee of optimality.

4. Repairing Non-Admissible Heuristics (Optional Challenge):
   - To minimally modify an overestimating heuristic without losing domain
     guidance, clamp each heuristic value to the upper admissible bound:
         h_repaired(n) = min(h(n), h*(n))
   - This ensures:
     (a) Admissibility: h_repaired(n) <= h*(n) holds unconditionally.
     (b) Maximal Informedness: h_repaired preserves as much heuristic guidance
         as possible without violating admissibility.
"""


def get_viva_reference() -> str:
    """Return university viva voce preparation guide for heuristic validation."""
    return """================================================================================
UNIVERSITY VIVA VOCE PREPARATION GUIDE: ADMISSIBLE HEURISTICS
================================================================================

Q1: What is the formal definition of an Admissible Heuristic?
--------------------------------------------------------------------------------
Answer:
An admissible heuristic h(n) is an evaluation function that never overestimates
the actual minimum cost to reach the goal from node n:
    0 <= h(n) <= h*(n),  for all n in S
where h*(n) is the true cost of the shortest path from n to the goal.
In simple words: an admissible heuristic is always optimistic or perfect.

Q2: Why does an overestimating heuristic break the optimality of A*?
--------------------------------------------------------------------------------
Answer:
A* selects the node with the lowest f(n) = g(n) + h(n). If a node n on the true
optimal path has an overestimated heuristic such that f(n) > C* (where C* is the
cost of an optimal path), f(n) may exceed the cost of an alternative, suboptimal
path to the goal. A* will expand the suboptimal path to completion and terminate
prematurely upon reaching the goal, returning a suboptimal path.

Q3: What is the difference between Admissibility and Consistency (Monotonicity)?
--------------------------------------------------------------------------------
Answer:
- Admissibility: Global condition comparing h(n) to true remaining cost:
      h(n) <= h*(n)
- Consistency (Monotonicity): Local condition adhering to the triangle inequality:
      h(u) <= c(u, v) + h(v)   for every transition u -> v
      h(Goal) = 0
- Theorem: Every consistent heuristic is admissible. However, an admissible
  heuristic is not necessarily consistent (especially in graph search without
  reopening closed nodes).

Q4: Why do we run Reverse UCS from the goal instead of searching forward?
--------------------------------------------------------------------------------
Answer:
We need the shortest path from ALL nodes to a SINGLE goal (Single-Target Shortest
Path). Running forward UCS requires |V| independent searches (one per node).
By reversing all directed edges in the graph, the problem becomes Single-Source
Shortest Path (SSSP) starting from the goal node. A single run of Dijkstra / UCS
computes h*(n) for all nodes simultaneously in O((V + E) log V) time.

Q5: What should h(Goal) be for an admissible heuristic?
--------------------------------------------------------------------------------
Answer:
h(Goal) MUST be 0. The true cost from the goal to itself is h*(Goal) = 0.
If h(Goal) > 0, it overestimates the cost at the goal and is NOT admissible.
If h(Goal) < 0, it violates the non-negativity constraint of standard heuristics.

Q6: What happens if h(n) = 0 for all nodes? Is it admissible?
--------------------------------------------------------------------------------
Answer:
Yes! If h(n) = 0 for all nodes, 0 <= h*(n) always holds (since edge costs are
positive). Therefore, h(n) = 0 is unconditionally admissible.
Under h(n) = 0, f(n) = g(n) + 0 = g(n), meaning A* reduces identically to
Uniform Cost Search (UCS / Dijkstra's algorithm). It guarantees optimality but
expands the maximum number of states because it lacks heuristic guidance.

Q7: What happens if h(n) = h*(n) for all nodes?
--------------------------------------------------------------------------------
Answer:
h(n) = h*(n) is the 'perfect heuristic'. It is strictly admissible. With a perfect
heuristic, A* will expand only nodes that lie directly on the optimal solution
path, with zero false expansions or exploratory detours.
"""
