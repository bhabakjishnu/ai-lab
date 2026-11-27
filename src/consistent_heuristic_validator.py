"""Consistent Heuristic Validator (Question 34).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟠 Advanced
Practical Type: Validation
Problem Statement:
    Write a Python program to check if a given heuristic h(n) for a graph
    is Consistent (Monotonic). The program must verify the triangle inequality:
        h(n) <= c(n, a, n') + h(n')
    for every node and its successors.

Task Requirements:
    1. Iterate through every edge (u, v) with cost c.
    2. Check if h(u) <= c + h(v).
    3. Output True if consistent, else pinpoint the violating edge.

Input Requirements:
    Graph with edge costs, Heuristic dictionary.
Expected Output:
    Boolean and details of violations.
Constraints / Conditions:
    Must check all directed edges.
Core AI Concept:
    Consistency / Monotonicity in Heuristics.
Python Concepts Tested:
    Nested loops, graph traversal, typing, dataclasses.
Expected Algorithm / Technique:
    Edge validation iteration over directed graph edges.
Recommended Python Libraries:
    None (Standard Library only).
Complexity Expectation:
    Time Complexity: O(E) where E is the number of directed edges.
    Space Complexity: O(V) for node heuristic lookup.

Sample Input:
    h(A) = 5, h(B) = 2, cost(A, B) = 2
Sample Output:
    False. Violation at edge A->B: 5 > 2 + 2.

University Practical Importance:
    Advanced
Viva Topics:
    • Consistency vs Admissibility
    • Why consistency prevents A* from reopening nodes
Optional Challenge:
    Prove programmatically that all consistent heuristics are admissible.
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

# Graph representations supported:
# 1. Nested Dict: {'A': {'B': 2.0}}
# 2. Adjacency List: {'A': [('B', 2.0)]}
# 3. Flat Edge List: [('A', 'B', 2.0)]
GraphAdjacency = Dict[NodeId, Dict[NodeId, Cost]]
GraphInput = Union[
    Mapping[NodeId, Mapping[NodeId, Union[int, float]]],
    Mapping[NodeId, Sequence[Tuple[NodeId, Union[int, float]]]],
    Sequence[Tuple[NodeId, NodeId, Union[int, float]]],
]

# Canonical syllabus sample specification
CANONICAL_SAMPLE_GRAPH: GraphAdjacency = {
    "A": {"B": 2.0},
    "B": {},
}

CANONICAL_SAMPLE_HEURISTIC_VIOLATING: Dict[NodeId, float] = {
    "A": 5.0,
    "B": 2.0,
}

CANONICAL_SAMPLE_HEURISTIC_CONSISTENT: Dict[NodeId, float] = {
    "A": 4.0,
    "B": 2.0,
}

SYLLABUS_SAMPLE_OUTPUT: str = "False. Violation at edge A->B: 5 > 2 + 2."

# Extended Canonical Multi-Node Graph for comprehensive demonstrations:
# A -> B (cost 1), A -> C (cost 4), B -> C (cost 2), B -> Goal (cost 5), C -> Goal (cost 1)
CANONICAL_EXTENDED_GRAPH: GraphAdjacency = {
    "A": {"B": 1.0, "C": 4.0},
    "B": {"C": 2.0, "Goal": 5.0},
    "C": {"Goal": 1.0},
    "Goal": {},
}

# Consistent heuristic on extended graph (all edges satisfy triangle inequality)
CANONICAL_EXTENDED_HEURISTIC_CONSISTENT: Dict[NodeId, float] = {
    "A": 3.0,
    "B": 2.0,
    "C": 1.0,
    "Goal": 0.0,
}

# Admissible BUT Inconsistent heuristic on extended graph
# Demonstrates that Admissibility does NOT imply Consistency!
CANONICAL_EXTENDED_HEURISTIC_INCONSISTENT_ADMISSIBLE: Dict[NodeId, float] = {
    "A": 3.0,
    "B": 0.5,  # Undervalued: h(A)=3 > c(A,B)+h(B)=1+0.5=1.5 (Inconsistent!), yet <= true cost
    "C": 1.0,
    "Goal": 0.0,
}


# ============================================================================
# Dataclasses & Violation Records
# ============================================================================

@dataclass(frozen=True)
class EdgeViolation:
    """Represents a single consistency (triangle inequality) violation on an edge.

    Inequality checked:
        h(u) <= c(u, v) + h(v)
    Violation occurs when:
        h(u) > c(u, v) + h(v)

    Attributes:
        u: Source node identifier.
        v: Successor node identifier.
        cost: Step cost c(u, v) from u to v.
        h_u: Heuristic estimate h(u) of the source node.
        h_v: Heuristic estimate h(v) of the destination node.
        difference: Overestimation magnitude h(u) - (c(u, v) + h(v)).
    """

    u: NodeId
    v: NodeId
    cost: float
    h_u: float
    h_v: float
    difference: float

    def format_violation(self) -> str:
        """Format violation string matching the syllabus specification.

        Returns:
            Formatted string such as: 'Violation at edge A->B: 5 > 2 + 2.'
        """
        cost_str = f"{self.cost:g}"
        hu_str = f"{self.h_u:g}"
        hv_str = f"{self.h_v:g}"
        return f"Violation at edge {self.u}->{self.v}: {hu_str} > {cost_str} + {hv_str}."

    def format_syllabus_output(self) -> str:
        """Format complete single-line output matching sample output."""
        return f"False. {self.format_violation()}"


@dataclass
class ConsistencyReport:
    """Complete validation report for a heuristic function evaluated for consistency.

    Attributes:
        is_consistent: True if h(u) <= c(u, v) + h(v) holds for all directed edges; False otherwise.
        violations: List of all detected EdgeViolation instances.
        total_edges_checked: Number of directed edges examined.
        graph: Normalized graph adjacency structure.
        heuristics: Clean dictionary of node heuristic values.
        primary_violation: The first encountered or most significant violation.
        summary_message: Concise textual summary matching syllabus expectations.
        execution_time_ms: Wall-clock verification duration in milliseconds.
    """

    is_consistent: bool
    violations: List[EdgeViolation]
    total_edges_checked: int
    graph: GraphAdjacency
    heuristics: Dict[NodeId, float]
    primary_violation: Optional[EdgeViolation] = None
    summary_message: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "is_consistent": self.is_consistent,
            "total_edges_checked": self.total_edges_checked,
            "violations_count": len(self.violations),
            "violations": [asdict(v) for v in self.violations],
            "primary_violation": asdict(self.primary_violation) if self.primary_violation else None,
            "summary_message": self.summary_message,
            "heuristics": self.heuristics,
            "graph": self.graph,
            "execution_time_ms": round(self.execution_time_ms, 4),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class AdmissibilityProofStep:
    """Record of a single edge step in the telescoping path induction proof.

    Attributes:
        step_index: Sequential 1-based step counter along the shortest path.
        u: Current node.
        v: Successor node towards goal.
        edge_cost: Step cost c(u, v).
        h_u: Heuristic h(u).
        h_v: Heuristic h(v).
        inequality_str: Formatted inequality string for this step.
        is_valid: Whether h(u) <= c(u, v) + h(v) holds.
    """

    step_index: int
    u: NodeId
    v: NodeId
    edge_cost: float
    h_u: float
    h_v: float
    inequality_str: str
    is_valid: bool


@dataclass
class ConsistencyProofReport:
    """Container for programmatic proof that consistency implies admissibility.

    Attributes:
        goal_node: Identifier of the target goal node.
        is_consistent: Whether the heuristic satisfies the triangle inequality on all edges.
        is_admissible: Whether h(n) <= h*(n) holds for all reachable nodes.
        proof_holds: True if the mathematical induction holds across all shortest paths.
        telescoping_steps: Dictionary mapping each node to its step-by-step induction trace.
        true_shortest_costs: Dictionary mapping each node to true minimum cost h*(n).
        heuristics: Clean dictionary of heuristic values.
        explanation: Detailed prose explanation of the proof.
    """

    goal_node: NodeId
    is_consistent: bool
    is_admissible: bool
    proof_holds: bool
    telescoping_steps: Dict[NodeId, List[AdmissibilityProofStep]]
    true_shortest_costs: Dict[NodeId, float]
    heuristics: Dict[NodeId, float]
    explanation: str


# ============================================================================
# Graph Normalization & Extraction Utilities
# ============================================================================

def normalize_graph(graph_input: GraphInput) -> GraphAdjacency:
    """Normalize various graph representations into a standardized nested dictionary.

    Supports:
        - Nested dictionary: {'A': {'B': 2.0}}
        - Adjacency list: {'A': [('B', 2.0)]}
        - Edge list / tuples: [('A', 'B', 2.0)]

    Args:
        graph_input: Raw graph data in any supported format.

    Returns:
        Standardized nested dictionary GraphAdjacency: {u: {v: cost}}.

    Raises:
        ValueError: If edge costs are non-positive or structure format is invalid.
    """
    normalized: GraphAdjacency = {}

    # Case 1: Sequence of 3-tuples: [('A', 'B', 2.0), ...]
    if isinstance(graph_input, Sequence) and not isinstance(graph_input, (str, bytes)):
        for item in graph_input:
            if not (isinstance(item, (tuple, list)) and len(item) == 3):
                raise ValueError(
                    f"Invalid edge sequence item: expected (u, v, cost) 3-tuple, got {item}."
                )
            u, v, weight = str(item[0]), str(item[1]), float(item[2])
            if weight <= 0:
                raise ValueError(
                    f"Graph constraint violated: Edge ({u} -> {v}) has non-positive cost {weight}. "
                    "Edge costs must be strictly positive."
                )
            if u not in normalized:
                normalized[u] = {}
            if v not in normalized:
                normalized[v] = {}
            normalized[u][v] = weight
        return normalized

    # Case 2: Mapping of source node to neighbors (dict or list)
    if isinstance(graph_input, Mapping):
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
                            f"cost {w}. Edge costs must be strictly positive."
                        )
                    normalized[src][dst] = w
                    if dst not in normalized:
                        normalized[dst] = {}
            elif isinstance(neighbors, Sequence) and not isinstance(neighbors, (str, bytes)):
                for edge in neighbors:
                    if not (isinstance(edge, (tuple, list)) and len(edge) == 2):
                        raise ValueError(
                            f"Invalid edge format in node '{src}': expected (destination, cost) tuple."
                        )
                    dst = str(edge[0])
                    w = float(edge[1])
                    if w <= 0:
                        raise ValueError(
                            f"Graph constraint violated: Edge ({src} -> {dst}) has non-positive "
                            f"cost {w}. Edge costs must be strictly positive."
                        )
                    normalized[src][dst] = w
                    if dst not in normalized:
                        normalized[dst] = {}
            else:
                raise ValueError(
                    f"Unsupported neighbors type for node '{src}': {type(neighbors)}"
                )
        return normalized

    raise ValueError(f"Unsupported graph input format: {type(graph_input)}")


def extract_all_directed_edges(
    graph: GraphAdjacency,
) -> List[Tuple[NodeId, NodeId, float]]:
    """Extract all directed edges from a normalized graph adjacency structure.

    Args:
        graph: Standardized nested adjacency dictionary.

    Returns:
        List of (u, v, cost) tuples ordered deterministically by source then destination.
    """
    edges: List[Tuple[NodeId, NodeId, float]] = []
    for u in sorted(graph.keys()):
        for v in sorted(graph[u].keys()):
            edges.append((u, v, float(graph[u][v])))
    return edges


# ============================================================================
# Core Consistency Validation Engine
# ============================================================================

def validate_consistency(
    graph: GraphInput,
    heuristic: Mapping[NodeId, Union[int, float]],
    tolerance: float = 1e-9,
) -> ConsistencyReport:
    """Check if a given heuristic h(n) for a graph is Consistent (Monotonic).

    Algorithm & Requirements:
        1. Iterate through every directed edge (u, v) with step cost c.
        2. Verify the triangle inequality:
               h(u) <= c(u, v) + h(v)
        3. Output True if consistent for all edges, else pinpoint the violating edge(s).

    Args:
        graph: Directed graph input representation.
        heuristic: Dictionary mapping node identifiers to heuristic estimates h(n).
        tolerance: Numerical floating-point tolerance for equality comparisons.

    Returns:
        ConsistencyReport containing boolean is_consistent, all violations,
        primary violation, and summary text.

    Raises:
        KeyError: If an edge references a node missing from the heuristic dictionary.
        ValueError: If graph edges have invalid formats or non-positive costs.
    """
    start_time = time.perf_counter()

    # Step 1: Normalize graph and extract clean heuristic dictionary
    norm_graph = normalize_graph(graph)
    clean_heuristic: Dict[NodeId, float] = {
        str(k): float(v) for k, v in heuristic.items()
    }

    # Verify that all graph nodes have an associated heuristic value
    for u in norm_graph:
        if u not in clean_heuristic:
            raise KeyError(
                f"Heuristic dictionary is missing value for node '{u}'. "
                f"Available nodes: {list(clean_heuristic.keys())}"
            )
        for v in norm_graph[u]:
            if v not in clean_heuristic:
                raise KeyError(
                    f"Heuristic dictionary is missing value for successor node '{v}'. "
                    f"Available nodes: {list(clean_heuristic.keys())}"
                )

    edges = extract_all_directed_edges(norm_graph)
    violations: List[EdgeViolation] = []

    # Step 2: Iterate through every directed edge and check the triangle inequality
    for u, v, cost in edges:
        h_u = clean_heuristic[u]
        h_v = clean_heuristic[v]
        allowed_upper_bound = cost + h_v

        # Check if triangle inequality is violated beyond floating point tolerance
        if h_u > allowed_upper_bound and not math.isclose(
            h_u, allowed_upper_bound, rel_tol=tolerance, abs_tol=tolerance
        ):
            diff = h_u - allowed_upper_bound
            violations.append(
                EdgeViolation(
                    u=u,
                    v=v,
                    cost=cost,
                    h_u=h_u,
                    h_v=h_v,
                    difference=diff,
                )
            )

    is_consistent = len(violations) == 0
    primary_violation = violations[0] if violations else None

    # Step 3: Format summary message matching sample output
    if is_consistent:
        summary_msg = (
            "True. All edges satisfy the triangle inequality (h(u) <= c(u, v) + h(v))."
        )
    else:
        assert primary_violation is not None
        summary_msg = primary_violation.format_syllabus_output()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return ConsistencyReport(
        is_consistent=is_consistent,
        violations=violations,
        total_edges_checked=len(edges),
        graph=norm_graph,
        heuristics=clean_heuristic,
        primary_violation=primary_violation,
        summary_message=summary_msg,
        execution_time_ms=elapsed_ms,
    )


def is_consistent_heuristic(
    graph: GraphInput,
    heuristic: Mapping[NodeId, Union[int, float]],
) -> bool:
    """Convenience predicate returning strictly True or False for consistency.

    Args:
        graph: Directed graph input.
        heuristic: Node heuristic dictionary.

    Returns:
        True if all directed edges satisfy the triangle inequality; False otherwise.
    """
    report = validate_consistency(graph, heuristic)
    return report.is_consistent


# ============================================================================
# Optional Challenge: Programmatic Proof (Consistency => Admissibility)
# ============================================================================

def compute_shortest_paths_and_costs(
    graph: GraphAdjacency,
    goal_node: NodeId,
) -> Tuple[Dict[NodeId, float], Dict[NodeId, Optional[NodeId]]]:
    """Calculate true minimum costs h*(n) and optimal parent pointers using Dijkstra from goal.

    Runs Dijkstra's algorithm on the transposed graph starting from goal_node.

    Args:
        graph: Forward graph adjacency.
        goal_node: Identifier of the target goal node.

    Returns:
        Tuple of (true_costs, parent_towards_goal).
        - true_costs: {node: h*(node)}
        - parent_towards_goal: {node: next_node_on_optimal_path_to_goal}
    """
    goal = str(goal_node)
    if goal not in graph:
        raise ValueError(f"Goal node '{goal}' is not present in graph.")

    # Transpose graph
    reversed_graph: GraphAdjacency = {n: {} for n in graph}
    for u, neighbors in graph.items():
        for v, cost in neighbors.items():
            if v not in reversed_graph:
                reversed_graph[v] = {}
            reversed_graph[v][u] = cost

    true_costs: Dict[NodeId, float] = {n: float("inf") for n in graph}
    true_costs[goal] = 0.0
    parent_towards_goal: Dict[NodeId, Optional[NodeId]] = {n: None for n in graph}

    # Priority queue: (cost, insertion_counter, node)
    counter = 0
    pq: List[Tuple[float, int, NodeId]] = [(0.0, counter, goal)]
    visited: Set[NodeId] = set()

    while pq:
        curr_dist, _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for predecessor, cost in reversed_graph.get(u, {}).items():
            new_dist = curr_dist + cost
            if new_dist < true_costs[predecessor]:
                true_costs[predecessor] = new_dist
                parent_towards_goal[predecessor] = u
                counter += 1
                heapq.heappush(pq, (new_dist, counter, predecessor))

    return true_costs, parent_towards_goal


def prove_consistency_implies_admissibility(
    graph: GraphInput,
    goal_node: NodeId,
    heuristic: Mapping[NodeId, Union[int, float]],
) -> ConsistencyProofReport:
    """Programmatically prove that consistency implies admissibility on any given graph.

    Mathematical Proof by Path Induction:
        For any node n_0, let P = (n_0, n_1, n_2, ..., n_k = Goal) be the
        optimal shortest path from n_0 to Goal with total cost h*(n_0).

        Since h is consistent:
            h(n_0)     <= c(n_0, n_1)     + h(n_1)
            h(n_1)     <= c(n_1, n_2)     + h(n_2)
            ...
            h(n_{k-1}) <= c(n_{k-1}, Goal) + h(Goal)

        Summing these k inequalities cancels internal terms (telescoping sum):
            h(n_0) <= sum_{i=0}^{k-1} c(n_i, n_{i+1}) + h(Goal)
            h(n_0) <= h*(n_0) + h(Goal)

        Given standard goal condition h(Goal) = 0:
            h(n_0) <= h*(n_0)
        Therefore, consistency guarantees admissibility for all reachable nodes.

    Args:
        graph: Directed graph input.
        goal_node: Target goal node.
        heuristic: Node heuristic dictionary.

    Returns:
        ConsistencyProofReport with complete step-by-step induction traces.
    """
    norm_graph = normalize_graph(graph)
    clean_h: Dict[NodeId, float] = {str(k): float(v) for k, v in heuristic.items()}
    goal = str(goal_node)

    consistency_report = validate_consistency(norm_graph, clean_h)
    true_costs, next_ptrs = compute_shortest_paths_and_costs(norm_graph, goal)

    telescoping_steps: Dict[NodeId, List[AdmissibilityProofStep]] = {}
    admissibility_violations: List[Tuple[NodeId, float, float]] = []

    for node in sorted(norm_graph.keys()):
        h_val = clean_h[node]
        h_star = true_costs[node]

        if math.isinf(h_star):
            # Node cannot reach goal; admissibility trivially holds
            telescoping_steps[node] = []
            continue

        if node == goal:
            telescoping_steps[node] = []
            if h_val > 0.0 and not math.isclose(h_val, 0.0, abs_tol=1e-9):
                admissibility_violations.append((goal, h_val, 0.0))
            continue

        # Reconstruct optimal path from node to goal
        path_steps: List[AdmissibilityProofStep] = []
        curr = node
        step_idx = 1
        path_valid = True

        while curr != goal:
            nxt = next_ptrs[curr]
            if nxt is None:
                path_valid = False
                break
            edge_cost = norm_graph[curr][nxt]
            curr_h = clean_h[curr]
            nxt_h = clean_h[nxt]
            is_step_valid = curr_h <= (edge_cost + nxt_h) or math.isclose(
                curr_h, edge_cost + nxt_h, rel_tol=1e-9, abs_tol=1e-9
            )

            ineq_str = f"h({curr}) = {curr_h:g} <= c({curr},{nxt}) + h({nxt}) = {edge_cost:g} + {nxt_h:g} = {edge_cost + nxt_h:g}"
            path_steps.append(
                AdmissibilityProofStep(
                    step_index=step_idx,
                    u=curr,
                    v=nxt,
                    edge_cost=edge_cost,
                    h_u=curr_h,
                    h_v=nxt_h,
                    inequality_str=ineq_str,
                    is_valid=is_step_valid,
                )
            )
            curr = nxt
            step_idx += 1

        telescoping_steps[node] = path_steps

        # Admissibility check: h(n) <= h*(n)
        if h_val > h_star and not math.isclose(h_val, h_star, rel_tol=1e-9, abs_tol=1e-9):
            admissibility_violations.append((node, h_val, h_star))

    is_admissible = len(admissibility_violations) == 0
    proof_holds = consistency_report.is_consistent and is_admissible

    explanation_lines = [
        "MATHEMATICAL INDUCTION PROOF: Consistency => Admissibility",
        "=" * 60,
        f"Goal Node: {goal} with h({goal}) = {clean_h.get(goal, 0.0):g}",
        f"Consistency on all edges: {'PASSED (True)' if consistency_report.is_consistent else 'FAILED (False)'}",
        f"Admissibility on all nodes: {'PASSED (True)' if is_admissible else 'FAILED (False)'}",
        "",
        "Formal Induction Step:",
        "For any node n_0, let (n_0, n_1, ..., n_k = Goal) be an optimal path.",
        "Summing the triangle inequality h(n_i) <= c(n_i, n_{i+1}) + h(n_{i+1})",
        "along all k edges telescopes into: h(n_0) <= sum(costs) + h(Goal) = h*(n_0).",
        f"Result: Programmatic induction verified across {len(norm_graph)} nodes.",
    ]

    return ConsistencyProofReport(
        goal_node=goal,
        is_consistent=consistency_report.is_consistent,
        is_admissible=is_admissible,
        proof_holds=proof_holds,
        telescoping_steps=telescoping_steps,
        true_shortest_costs=true_costs,
        heuristics=clean_h,
        explanation="\n".join(explanation_lines),
    )


# ============================================================================
# Visual & Tabular Presentation Helpers
# ============================================================================

def format_consistency_table(report: ConsistencyReport) -> str:
    """Format an ASCII comparison table of all graph edges and consistency checks.

    Args:
        report: ConsistencyReport instance.

    Returns:
        Formatted ASCII table string showing edge evaluations.
    """
    edges = extract_all_directed_edges(report.graph)
    if not edges:
        return "No edges to display in graph."

    col_edge = 14
    col_cost = 8
    col_hu = 8
    col_hv = 8
    col_bound = 12
    col_ineq = 20
    col_status = 14

    header = (
        f"| {'Edge (u->v)':<{col_edge}} "
        f"| {'Cost c':<{col_cost}} "
        f"| {'h(u)':<{col_hu}} "
        f"| {'h(v)':<{col_hv}} "
        f"| {'c + h(v)':<{col_bound}} "
        f"| {'h(u) <= c + h(v)':<{col_ineq}} "
        f"| {'Status':<{col_status}} |"
    )
    separator = (
        f"|{'-' * (col_edge + 2)}"
        f"|{'-' * (col_cost + 2)}"
        f"|{'-' * (col_hu + 2)}"
        f"|{'-' * (col_hv + 2)}"
        f"|{'-' * (col_bound + 2)}"
        f"|{'-' * (col_ineq + 2)}"
        f"|{'-' * (col_status + 2)}|"
    )

    rows: List[str] = [
        "",
        "CONSISTENCY VERIFICATION TABLE (Triangle Inequality: h(u) <= c + h(v))",
        "=" * len(header),
        header,
        separator,
    ]

    for u, v, cost in edges:
        hu = report.heuristics[u]
        hv = report.heuristics[v]
        bound = cost + hv
        holds = hu <= bound or math.isclose(hu, bound, rel_tol=1e-9, abs_tol=1e-9)

        edge_str = f"{u} -> {v}"
        ineq_str = f"{hu:g} <= {bound:g}" if holds else f"{hu:g} > {bound:g}"
        status_str = "✅ Valid" if holds else "❌ Violation"

        row = (
            f"| {edge_str:<{col_edge}} "
            f"| {cost:<{col_cost}.2f} "
            f"| {hu:<{col_hu}.2f} "
            f"| {hv:<{col_hv}.2f} "
            f"| {bound:<{col_bound}.2f} "
            f"| {ineq_str:<{col_ineq}} "
            f"| {status_str:<{col_status}} |"
        )
        rows.append(row)

    rows.append(separator)
    rows.append(f"Summary: {report.summary_message}")
    rows.append(f"Total Edges: {report.total_edges_checked} | Violations: {len(report.violations)}")
    rows.append("")

    return "\n".join(rows)


def render_ascii_graph(
    graph: GraphAdjacency,
    heuristics: Mapping[NodeId, float],
) -> str:
    """Render a text-based ASCII representation of graph nodes, edges, and heuristic values.

    Args:
        graph: Normalized graph adjacency.
        heuristics: Mapping of node to heuristic estimate.

    Returns:
        Multi-line string diagram.
    """
    lines: List[str] = [
        "GRAPH TOPOLOGY & HEURISTIC ESTIMATES",
        "-" * 45,
    ]
    for u in sorted(graph.keys()):
        h_val = heuristics.get(u, 0.0)
        lines.append(f"Node [{u}] (h = {h_val:g}):")
        if not graph[u]:
            lines.append("    └── (terminal / no outgoing edges)")
        else:
            neighbors = sorted(graph[u].items())
            for idx, (v, cost) in enumerate(neighbors):
                prefix = "    └──" if idx == len(neighbors) - 1 else "    ├──"
                hv = heuristics.get(v, 0.0)
                lines.append(f"{prefix} --(cost: {cost:g})--> Node [{v}] (h = {hv:g})")
    return "\n".join(lines)


# ============================================================================
# Theoretical Foundation & University Viva Reference
# ============================================================================

def get_theoretical_analysis() -> str:
    """Return comprehensive academic and theoretical foundation for consistency in heuristics."""
    return """================================================================================
THEORETICAL FOUNDATIONS: CONSISTENT (MONOTONIC) HEURISTICS IN AI
================================================================================

1. FORMAL MATHEMATICAL DEFINITION
---------------------------------
In informed search (Russell & Norvig, AIMA 4th Ed.), a heuristic function h(n)
is defined as CONSISTENT (or MONOTONIC) if, for every node n and every successor
n' generated by any action a with step cost c(n, a, n'):

    h(n) <= c(n, a, n') + h(n')

with the terminal boundary condition:
    h(Goal) = 0

This condition expresses the classic TRIANGLE INEQUALITY:
The direct estimated distance from node n to the goal cannot be strictly greater
than the step cost to a neighbor n' plus the estimated distance from n' to the goal.

2. CONSISTENCY VS. ADMISSIBILITY: THE CRUCIAL DISTINCTION
---------------------------------------------------------
- Admissibility: h(n) <= h*(n) for all nodes n.
  The heuristic never overestimates the true minimum remaining cost to the goal.
- Consistency:   h(n) <= c(n, a, n') + h(n') for every edge (n -> n').
  Local monotonicity holds along every individual transition step.

KEY THEORETICAL THEOREMS:
a) CONSISTENCY IMPLIES ADMISSIBILITY:
   Every consistent heuristic with h(Goal) = 0 is guaranteed to be admissible.
   (Proof: By induction/telescoping summation along the shortest path).
b) ADMISSIBILITY DOES NOT IMPLY CONSISTENCY:
   A heuristic can be everywhere admissible (underestimating the true cost)
   and yet violate the triangle inequality on specific transitions.
   Example:
       Path A --(cost 1)--> B --(cost 5)--> Goal.
       True costs: h*(Goal)=0, h*(B)=5, h*(A)=6.
       Heuristic:  h(Goal)=0, h(B)=1, h(A)=4.
       Admissibility: h(A)=4 <= 6 (OK), h(B)=1 <= 5 (OK).
       Consistency:   h(A) = 4 > c(A,B) + h(B) = 1 + 1 = 2 (VIOLATION!).

3. WHY CONSISTENCY PREVENTS A* FROM REOPENING NODES
----------------------------------------------------
In Graph Search (where an explored/closed set is maintained):
- For an admissible but inconsistent heuristic, A* may discover a cheaper path
  to an already closed node, requiring the node to be REOPENED (moved from Closed
  back to Open). In the worst case, reopening leads to exponential expansions!
- When h(n) is CONSISTENT:
  Along any path, the evaluation function f(n) = g(n) + h(n) is MONOTONICALLY
  NON-DECREASING:
      f(n') = g(n') + h(n') = g(n) + c(n, n') + h(n') >= g(n) + h(n) = f(n).
  Therefore, the sequence of nodes expanded by A* is non-decreasing in f-cost.
  When A* selects a node n for expansion, the path found to n is GUARANTEED TO BE
  OPTIMAL: g(n) = g*(n).
  NO CHEAPER PATH TO n CAN EVER BE FOUND LATER.
  Hence, explored nodes NEVER need to be reopened, ensuring optimal graph-search
  efficiency O(V) node expansions.
================================================================================"""


def get_viva_reference() -> str:
    """Return academic Viva Voce preparation guide for Question 34."""
    return """================================================================================
UNIVERSITY VIVA VOCE EXAMINATION GUIDE: CONSISTENT HEURISTIC VALIDATOR
================================================================================

Q1: What is the triangle inequality in the context of heuristic search?
A1: The triangle inequality states that the estimated distance from node u to the
    goal h(u) cannot exceed the cost of moving from u to a successor v plus the
    estimated distance from v to the goal h(v):
        h(u) <= c(u, v) + h(v)
    If this holds for all edges, the heuristic is consistent (monotonic).

Q2: Does consistency imply admissibility, or vice versa?
A2: Consistency STRICTLY IMPLIES admissibility (assuming h(Goal) = 0).
    The reverse is NOT true: an admissible heuristic can be inconsistent if it
    underestimates unevenly across adjacent nodes.

Q3: Why is consistency so important for A* Graph Search?
A3: If h is consistent, the f-costs of expanded nodes are non-decreasing. When A*
    expands a node, its g-cost is guaranteed to be optimal. Consequently, closed
    nodes NEVER need to be reopened, preserving O(V) search complexity rather than
    exponential node reopenings.

Q4: What is the time complexity of the Consistent Heuristic Validator?
A4: Time complexity is O(E) where E is the number of directed edges in the graph.
    The algorithm iterates through every edge (u, v) once, looks up h(u) and h(v)
    in O(1) time using hash tables, and evaluates the inequality in O(1) arithmetic.

Q5: What happens if an edge has a negative weight?
A5: In standard heuristic search, edge costs must be strictly positive (c > 0).
    Negative weights violate basic A* assumptions, allow negative cost cycles,
    and invalidate the standard monotonicity proof. Our validator enforces c > 0.
================================================================================"""
