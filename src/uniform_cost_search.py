"""Uniform Cost Search (UCS) Implementation on Weighted Directed Graphs.

This module implements Uniform Cost Search (UCS) for finding the optimal (minimum-cost)
path between a source node and a destination node in a weighted directed graph, adhering to
Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA) and Dijkstra's algorithm.

Theoretical Background & AI Concepts:
- Uninformed (Blind) Search with Path Cost:
    Uniform Cost Search is an uninformed search strategy that guides exploration using the
    cumulative path cost g(n) from the start state to node n. Unlike informed search algorithms
    (such as A*), UCS operates without any domain-specific heuristic function h(n) (effectively
    equivalent to A* with h(n) = 0).
- Priority Queue Frontier:
    UCS maintains its frontier (open list) as a priority queue ordered by g(n). Python's
    built-in `heapq` module implements a binary min-heap on top of a standard list, providing
    O(log N) insertion (`heappush`) and O(log N) minimum extraction (`heappop`).
- Python Tuple Sorting Mechanics:
    When tuples are placed in a min-heap via `heapq`, Python compares elements lexicographically
    position by position: (cost, tie_breaker, node_id, ...).
    If two nodes have identical cumulative cost g(n), Python falls back to comparing the second
    element. By using an explicit tie-breaking token (e.g. node identifier for alphabetical order
    or a monotonic insertion sequence counter for FIFO ordering), we prevent comparison errors
    and guarantee deterministic, reproducible behavior.
- Goal Test Placement (Expansion vs. Generation):
    In UCS, the goal test MUST be applied when a node is selected for expansion (popped from
    the priority queue), NOT when it is first generated (added to the frontier). This is because
    the first path discovered to a goal may not be the cheapest; a cheaper path might still reside
    in the frontier through an unexpanded intermediate node.
- Explored Set and Lazy Deletion:
    Because Python's `heapq` does not natively support an efficient O(log N) `decrease_key`
    operation, UCS employs lazy deletion. When a cheaper path to an already queued node is found,
    the cheaper entry is pushed into the heap. When an entry is popped, if its cost exceeds the
    known minimum cost for that node (`cost > best_cost[node]`), it is recognized as a superseded
    entry and discarded in O(1) time.
- Completeness and Optimality:
    - Completeness: UCS is complete on graphs with finite branching factor if every step cost
      is bounded away from zero by some positive epsilon (c(u, v) >= eps > 0).
    - Optimality: UCS is guaranteed to return an optimal (minimum-cost) path because it expands
      nodes in strictly non-decreasing order of path cost g(n).
- Negative Edge Weights & Negative Cycles (Optional Challenge):
    UCS relies on non-decreasing path costs. Negative edge weights break the greedy property,
    as nodes already marked explored may later receive cheaper paths. If a negative cycle exists,
    path costs can decrease indefinitely toward negative infinity. This module provides a
    dedicated Bellman-Ford based cycle detection and diagnostic handler to detect, trace, and
    report negative cycles gracefully.

Complexity:
- Time Complexity: O(E log V) with a binary heap, where V is the number of vertices and E is
  the number of edges. Each vertex is expanded at most once (O(V log V)) and each edge relaxation
  pushes to the heap at most once (O(E log V)).
- Space Complexity: O(V) auxiliary memory for the explored set, best cost table, and frontier.
"""

from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Benchmark and sample graphs for testing, education, and demonstrations
# Default sample input from Question 14: A->B:1, B->C:2, A->C:4. Start: A, Goal: C
DEFAULT_SAMPLE_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2)],
    "C": [],
}

# Classic Diamond graph: direct expensive path vs multi-hop cheap path
DIAMOND_BENCHMARK_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "S": [("A", 1), ("B", 5)],
    "A": [("C", 2)],
    "B": [("G", 2)],
    "C": [("G", 1)],
    "G": [],
}

# Cyclic graph with positive weights to verify cycle avoidance
CYCLIC_WEIGHTED_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 2), ("C", 5)],
    "B": [("A", 2), ("C", 1), ("D", 4)],
    "C": [("D", 1)],
    "D": [("B", 3)],
}

# Disconnected graph to verify handling of unreachable goals
DISCONNECTED_WEIGHTED_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 2)],
    "B": [("C", 3)],
    "C": [],
    "D": [("E", 1)],
    "E": [],
}

# AIMA Romania Map Subgraph (Arad to Bucharest)
ROMANIA_MAP_SUBGRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "Arad": [("Zerind", 75), ("Sibiu", 140), ("Timisoara", 118)],
    "Zerind": [("Arad", 75), ("Oradea", 71)],
    "Oradea": [("Zerind", 71), ("Sibiu", 151)],
    "Timisoara": [("Arad", 118), ("Lugoj", 111)],
    "Lugoj": [("Timisoara", 111), ("Mehadia", 70)],
    "Mehadia": [("Lugoj", 70), ("Drobeta", 75)],
    "Drobeta": [("Mehadia", 75), ("Craiova", 120)],
    "Sibiu": [("Arad", 140), ("Oradea", 151), ("Fagaras", 99), ("Rimnicu Vilcea", 80)],
    "Fagaras": [("Sibiu", 99), ("Bucharest", 211)],
    "Rimnicu Vilcea": [("Sibiu", 80), ("Craiova", 146), ("Pitesti", 97)],
    "Craiova": [("Drobeta", 120), ("Rimnicu Vilcea", 146), ("Pitesti", 138)],
    "Pitesti": [("Rimnicu Vilcea", 97), ("Craiova", 138), ("Bucharest", 101)],
    "Bucharest": [("Fagaras", 211), ("Pitesti", 101), ("Giurgiu", 90)],
    "Giurgiu": [("Bucharest", 90)],
}

# Graph containing negative edges but NO negative cycles (for challenge testing)
NEGATIVE_EDGE_DAG: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "S": [("A", 4), ("B", 2)],
    "A": [("C", -1)],
    "B": [("C", 2)],
    "C": [("G", 3)],
    "G": [],
}

# Graph containing a reachable negative cycle (for challenge testing)
NEGATIVE_CYCLE_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 1)],
    "B": [("C", 2)],
    "C": [("D", -5)],
    "D": [("B", 1), ("Goal", 3)],
    "Goal": [],
}


@dataclass(frozen=True)
class UCSStepRecord:
    """Immutable record capturing an individual step of Uniform Cost Search.

    Attributes:
        step_number: 1-based chronological expansion sequence index.
        current_node: Node identifier extracted from the priority queue.
        cumulative_cost: Path cost g(n) from start to current_node.
        action: Semantic action label (e.g. 'EXPAND', 'GOAL_REACHED', 'SKIP_SUPERSEDED').
        frontier_snapshot: Snapshot of (cost, node) pairs in the priority queue after this step.
        explored_nodes: Sorted snapshot of all nodes closed/explored so far.
        neighbors_evaluated: List of (neighbor, edge_cost) pairs evaluated from current_node.
        neighbors_enqueued: List of (neighbor, new_cost) pairs pushed to the frontier.
        path_so_far: Sequence of nodes traversed to reach current_node.
        parent: Predecessor node from which current_node was reached.
    """

    step_number: int
    current_node: str
    cumulative_cost: Union[int, float]
    action: str
    frontier_snapshot: List[Tuple[Union[int, float], str]]
    explored_nodes: List[str]
    neighbors_evaluated: List[Tuple[str, Union[int, float]]]
    neighbors_enqueued: List[Tuple[str, Union[int, float]]]
    path_so_far: List[str]
    parent: Optional[str] = None

    def format_row(self) -> str:
        """Format the step into a structured table row for terminal presentation.

        Returns:
            Formatted row string.
        """
        frontier_items = [f"{node}:{cost}" for cost, node in self.frontier_snapshot]
        frontier_str = "[" + ", ".join(frontier_items) + "]" if frontier_items else "[]"
        enqueued_items = [f"{node}:{cost}" for node, cost in self.neighbors_enqueued]
        enqueued_str = "[" + ", ".join(enqueued_items) + "]" if enqueued_items else "-"
        parent_str = str(self.parent) if self.parent is not None else "START"
        path_str = "->".join(self.path_so_far)
        cost_str = f"{self.cumulative_cost:g}" if isinstance(self.cumulative_cost, float) else str(self.cumulative_cost)
        return (
            f"{self.step_number:<5} | {self.current_node:<6} | {cost_str:<5} | "
            f"{self.action:<14} | {parent_str:<6} | {enqueued_str:<16} | {frontier_str}"
        )


@dataclass
class UCSResult:
    """Encapsulates the complete outcome of a Uniform Cost Search execution.

    Attributes:
        path: Ordered list of nodes forming the optimal path from start to goal (None if unreachable).
        total_cost: Minimum cumulative cost g(goal) (infinity if unreachable).
        start_node: Origin node for the search.
        goal_node: Target destination node.
        goal_found: Boolean flag indicating if goal was reached.
        nodes_expanded: Chronological sequence of nodes popped and expanded.
        steps: Chronological list of UCSStepRecord objects detailing the search progression.
        cost_to_nodes: Mapping of all reached nodes to their minimum confirmed path cost.
        parents: Mapping of each node to its predecessor in the shortest-path tree.
        graph: Adjacency list representation of the searched graph.
        execution_time_ms: Wall-clock execution time in milliseconds.
        negative_cycle_detected: Flag indicating whether a negative cycle was discovered.
        negative_cycle_path: List of nodes forming the detected negative cycle, if present.
        warning_message: Optional diagnostic notice (e.g. regarding negative weights or cycle).
    """

    path: Optional[List[str]]
    total_cost: Union[int, float]
    start_node: str
    goal_node: str
    goal_found: bool
    nodes_expanded: List[str]
    steps: List[UCSStepRecord]
    cost_to_nodes: Dict[str, Union[int, float]]
    parents: Dict[str, Optional[str]]
    graph: Dict[str, List[Tuple[str, Union[int, float]]]]
    execution_time_ms: float = 0.0
    negative_cycle_detected: bool = False
    negative_cycle_path: Optional[List[str]] = None
    warning_message: Optional[str] = None

    def format_path(self, delimiter: str = "->") -> str:
        """Format the path and cost matching the university standard output.

        Example:
            'Path: A->B->C, Total Cost: 3'

        Args:
            delimiter: Separator between successive node identifiers.

        Returns:
            Human-readable path and cost summary string.
        """
        if self.path is None or not self.goal_found:
            return "Path: None (Unreachable), Total Cost: Infinity"
        path_str = delimiter.join(self.path)
        cost_str = f"{self.total_cost:g}" if isinstance(self.total_cost, float) else str(self.total_cost)
        return f"Path: {path_str}, Total Cost: {cost_str}"

    def format_step_table(self) -> str:
        """Return a formatted ASCII execution trace table of the UCS process.

        Returns:
            Multi-line ASCII table string.
        """
        headers = f"{'Step':<5} | {'Node':<6} | {'g(n)':<5} | {'Action':<14} | {'Parent':<6} | {'Enqueued':<16} | {'Frontier Priority Queue':<24}"
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary.

        Returns:
            Dictionary containing search statistics and outcomes.
        """
        return {
            "path": self.path,
            "total_cost": self.total_cost if self.total_cost != float("inf") else "Infinity",
            "start_node": self.start_node,
            "goal_node": self.goal_node,
            "goal_found": self.goal_found,
            "nodes_expanded": self.nodes_expanded,
            "total_expansions": len(self.nodes_expanded),
            "step_count": len(self.steps),
            "cost_to_nodes": {
                k: (v if v != float("inf") else "Infinity") for k, v in self.cost_to_nodes.items()
            },
            "execution_time_ms": round(self.execution_time_ms, 4),
            "negative_cycle_detected": self.negative_cycle_detected,
            "negative_cycle_path": self.negative_cycle_path,
            "warning_message": self.warning_message,
        }


def _to_numeric_weight(weight: Any) -> Union[int, float]:
    """Convert a weight value or string to integer if whole, otherwise float.

    Args:
        weight: Raw weight value.

    Returns:
        Integer or float numeric weight.

    Raises:
        ValueError: If value is not a valid number.
    """
    try:
        f = float(weight)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid non-numeric edge weight '{weight}'.") from exc


def normalize_weighted_graph(
    graph: Mapping[str, Any],
) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
    """Normalize a weighted graph mapping into a standardized canonical dictionary.

    Ensures:
    1. Every node referenced as either a source or destination has an entry in the graph.
    2. Edge costs are strictly numeric (int or float).
    3. Successors are stored as a list of (neighbor_id, weight) tuples.

    Args:
        graph: Raw graph dictionary or mapping. Supports formats such as:
               `{'A': [('B', 1), ('C', 4)]}` or `{'A': {'B': 1, 'C': 4}}`.

    Returns:
        Canonical adjacency dictionary mapping node strings to lists of (neighbor, weight) tuples.

    Raises:
        ValueError: If edge weights are not valid numbers or format is unrecognized.
    """
    normalized: Dict[str, List[Tuple[str, Union[int, float]]]] = {}

    for source_node, outgoing in graph.items():
        src = str(source_node).strip()
        if src not in normalized:
            normalized[src] = []

        if isinstance(outgoing, dict):
            # Format: {'A': {'B': 1, 'C': 4}}
            for target_node, weight in outgoing.items():
                tgt = str(target_node).strip()
                num_weight = _to_numeric_weight(weight)
                normalized[src].append((tgt, num_weight))
                if tgt not in normalized:
                    normalized[tgt] = []
        elif isinstance(outgoing, (list, tuple, set)):
            # Format: {'A': [('B', 1), ('C', 4)]} or ['B:1', 'C:4']
            for edge in outgoing:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    tgt = str(edge[0]).strip()
                    num_weight = _to_numeric_weight(edge[1])
                    normalized[src].append((tgt, num_weight))
                    if tgt not in normalized:
                        normalized[tgt] = []
                elif isinstance(edge, str) and ":" in edge:
                    parts = edge.split(":", 1)
                    tgt = parts[0].strip()
                    num_weight = _to_numeric_weight(parts[1].strip())
                    normalized[src].append((tgt, num_weight))
                    if tgt not in normalized:
                        normalized[tgt] = []
                else:
                    raise ValueError(f"Unrecognized edge structure '{edge}' in node '{src}'.")
        else:
            raise ValueError(f"Unsupported outgoing edge container type '{type(outgoing)}' for node '{src}'.")

    return normalized


def parse_weighted_graph(
    input_data: Union[str, Dict[str, Any], Sequence[Any]],
) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
    """Parse a flexible user representation of a weighted directed graph into canonical form.

    Supports:
    1. Arrow syntax matching prompt:
       `"A->B:1, B->C:2, A->C:4"`
       or `"A->B:1.5, B->C:2"`
    2. Adjacency list string:
       `"A: B:1, C:4; B: C:2; C:"`
    3. JSON dictionary string:
       `'{"A": [["B", 1], ["C", 4]], "B": [["C", 2]], "C": []}'`
    4. Python literal dictionary:
       `{"A": [("B", 1), ("C", 4)], "B": [("C", 2)], "C": []}`
    5. List of 3-tuples:
       `[("A", "B", 1), ("B", "C", 2), ("A", "C", 4)]`

    Args:
        input_data: Graph specification in any supported format.

    Returns:
        Canonical normalized graph dictionary.

    Raises:
        ValueError: If input format cannot be parsed or weights are non-numeric.
    """
    if isinstance(input_data, dict):
        return normalize_weighted_graph(input_data)

    if isinstance(input_data, (list, tuple)):
        # List of (u, v, cost) tuples
        graph: Dict[str, List[Tuple[str, Union[int, float]]]] = {}
        for item in input_data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                u, v, w = str(item[0]).strip(), str(item[1]).strip(), item[2]
                num_w = _to_numeric_weight(w)
                graph.setdefault(u, []).append((v, num_w))
                graph.setdefault(v, [])
            else:
                raise ValueError(f"Expected 3-tuple (u, v, weight), got {item}")
        return normalize_weighted_graph(graph)

    if not isinstance(input_data, str):
        raise ValueError(f"Unsupported input data type: {type(input_data)}")

    text = input_data.strip()
    if not text:
        raise ValueError("Graph specification string cannot be empty.")

    # Try parsing as JSON first
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, dict):
                return normalize_weighted_graph(parsed_json)
        except json.JSONDecodeError:
            # Fall back to custom string syntax parser
            pass

    # Pattern for arrow notation: Source -> Target : Weight
    # Example: A->B:1, B->C:2, A->C:4
    arrow_pattern = re.compile(
        r"([a-zA-Z0-9_\-]+)\s*(?:->|–>|—>)\s*([a-zA-Z0-9_\-]+)\s*:\s*([+-]?[0-9]*\.?[0-9]+)"
    )
    arrow_matches = list(arrow_pattern.finditer(text))
    if arrow_matches:
        graph = {}
        for match in arrow_matches:
            u, v, w_str = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            num_w = _to_numeric_weight(w_str)
            graph.setdefault(u, []).append((v, num_w))
            graph.setdefault(v, [])
        return normalize_weighted_graph(graph)

    # Pattern for semicolon-separated adjacency lists: A: B:1, C:4; B: C:2; C:
    graph = {}
    clauses = [c.strip() for c in text.split(";") if c.strip()]
    if clauses and any(":" in c for c in clauses):
        for clause in clauses:
            parts = clause.split(":", 1)
            u = parts[0].strip()
            graph.setdefault(u, [])
            rest = parts[1].strip() if len(parts) > 1 else ""
            if rest:
                targets = [t.strip() for t in rest.split(",") if t.strip()]
                for target_item in targets:
                    if ":" in target_item:
                        v, w_str = target_item.split(":", 1)
                        v = v.strip()
                        num_w = _to_numeric_weight(w_str.strip())
                        graph[u].append((v, num_w))
                        graph.setdefault(v, [])
                    elif "(" in target_item and target_item.endswith(")"):
                        # Format: B(1)
                        v, w_str = target_item[:-1].split("(", 1)
                        v = v.strip()
                        num_w = _to_numeric_weight(w_str.strip())
                        graph[u].append((v, num_w))
                        graph.setdefault(v, [])
                    else:
                        raise ValueError(f"Missing weight for neighbor in clause: '{target_item}'.")
        return normalize_weighted_graph(graph)

    raise ValueError(
        f"Unable to parse graph input: '{text}'. "
        "Expected arrow format 'A->B:1, B->C:2', JSON, or adjacency list."
    )


def validate_strictly_positive_weights(
    graph: Dict[str, List[Tuple[str, Union[int, float]]]],
) -> List[Tuple[str, str, Union[int, float]]]:
    """Inspect graph and return any edges violating the strictly positive weight constraint.

    Args:
        graph: Canonical adjacency dictionary of weighted edges.

    Returns:
        List of (source, target, weight) tuples where weight <= 0.
    """
    non_positive: List[Tuple[str, str, Union[int, float]]] = []
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            if weight <= 0:
                non_positive.append((u, v, weight))
    return non_positive


def detect_negative_cycle(
    graph: Dict[str, List[Tuple[str, Union[int, float]]]],
    start_node: str,
) -> Tuple[bool, Optional[List[str]], Dict[str, float]]:
    """Detect reachable negative weight cycles using the Bellman-Ford algorithm (Optional Challenge).

    Uniform Cost Search assumes monotonically non-decreasing path costs g(n). If negative
    edge weights form a cycle reachable from start_node, path costs can decrease indefinitely (-inf).
    This function performs |V| - 1 edge relaxations followed by a cycle-detection pass to locate
    and extract any reachable negative cycle.

    Args:
        graph: Normalized weighted directed graph.
        start_node: Source node from which reachable vertices are searched.

    Returns:
        A 3-tuple:
        - has_negative_cycle: Boolean indicating presence of a reachable negative cycle.
        - cycle_path: Ordered list of nodes forming the negative cycle (or None).
        - distances: Dictionary of estimated distances to reachable vertices.
    """
    if start_node not in graph:
        return False, None, {}

    all_nodes = list(graph.keys())
    num_vertices = len(all_nodes)
    distance: Dict[str, float] = {node: float("inf") for node in all_nodes}
    predecessor: Dict[str, Optional[str]] = {node: None for node in all_nodes}
    distance[start_node] = 0.0

    # Relax all edges |V| - 1 times
    for _ in range(max(1, num_vertices - 1)):
        updated = False
        for u in all_nodes:
            if distance[u] == float("inf"):
                continue
            for v, weight in graph.get(u, []):
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u
                    updated = True
        if not updated:
            # Reached fixed-point early; no negative cycle reachable
            break

    # Check for negative weight cycle on |V|-th iteration
    cycle_node: Optional[str] = None
    for u in all_nodes:
        if distance[u] == float("inf"):
            continue
        for v, weight in graph.get(u, []):
            if distance[u] + weight < distance[v]:
                # Negative cycle detected!
                cycle_node = v
                break
        if cycle_node is not None:
            break

    if cycle_node is None:
        return False, None, distance

    # Trace back |V| steps to guarantee landing inside the cycle
    curr: str = cycle_node
    for _ in range(num_vertices):
        pred = predecessor[curr]
        if pred is None:
            break
        curr = pred

    # Now trace the cycle vertices
    cycle: List[str] = [curr]
    step_node = predecessor[curr]
    while step_node is not None and step_node != curr and step_node not in cycle:
        cycle.append(step_node)
        step_node = predecessor[step_node]

    cycle.append(curr)
    cycle.reverse()
    return True, cycle, distance


def uniform_cost_search(
    graph: Union[Dict[str, Any], str, Sequence[Any]],
    start: str,
    goal: str,
    tie_breaker: str = "lexicographical",
    detect_cycles: bool = True,
    sort_neighbors: bool = True,
) -> UCSResult:
    """Execute Uniform Cost Search (UCS) on a weighted directed graph to find the minimum cost path.

    Algorithm Overview (AIMA Dijkstra/UCS):
    1. Initialize a priority queue (min-heap) `frontier` containing:
       (cumulative_cost, tie_breaking_key, current_node, path_history).
    2. Maintain `best_cost` dictionary mapping each node to the lowest path cost discovered so far.
    3. Maintain `explored` set tracking nodes whose minimal path cost has been finalized.
    4. While `frontier` is not empty:
       a. Pop the lowest-cost tuple (c, ..., u, path) from `frontier`.
       b. If u was already popped with a strictly lower cost, skip (lazy deletion).
       c. APPLY GOAL TEST: If u == goal, return the optimal path and total cost c!
       d. Add u to `explored`.
       e. For each neighbor (v, edge_cost) of u:
          Calculate new_cost = c + edge_cost.
          If v is not in `explored` and new_cost < best_cost.get(v, inf):
              best_cost[v] = new_cost
              Push (new_cost, tie_key, v, path + [v]) to `frontier`.
    5. If frontier empties without reaching goal, return failure (unreachable).

    Args:
        graph: Weighted directed graph representation (dict, string, or edge tuples).
        start: Identifier of the origin/start node.
        goal: Identifier of the destination/target node.
        tie_breaker: Deterministic tie-breaking rule when cumulative costs are identical:
            - 'lexicographical' (default): breaks ties alphabetically by node identifier.
            - 'fifo': breaks ties by chronological insertion sequence counter.
            - 'hops': breaks ties by shortest hop length (fewest edges), then lexicographical.
        detect_cycles: If True, executes pre-flight Bellman-Ford validation for negative cycles.
        sort_neighbors: If True, evaluates outgoing neighbors in sorted order for determinism.

    Returns:
        UCSResult encapsulating the optimal path, minimum cost, step-by-step trace, and diagnostics.

    Raises:
        ValueError: If start node does not exist in graph or graph format is invalid.
    """
    start_time = time.perf_counter()
    normalized_graph = parse_weighted_graph(graph)

    start_str = str(start).strip()
    goal_str = str(goal).strip()

    if start_str not in normalized_graph:
        raise ValueError(
            f"Start node '{start_str}' does not exist in the graph. Available nodes: {sorted(normalized_graph.keys())}"
        )

    # Optional Challenge: Check for negative edges and negative cycles
    neg_cycle_detected = False
    neg_cycle_path: Optional[List[str]] = None
    warning_msg: Optional[str] = None

    non_positive_edges = validate_strictly_positive_weights(normalized_graph)
    if non_positive_edges:
        if detect_cycles:
            has_neg_cycle, cycle_nodes, _ = detect_negative_cycle(normalized_graph, start_str)
            if has_neg_cycle:
                neg_cycle_detected = True
                neg_cycle_path = cycle_nodes
                warning_msg = (
                    f"Reachable negative cycle detected: {' -> '.join(cycle_nodes or [])}. "
                    "Path cost can decrease indefinitely to -infinity; optimal path undefined."
                )
                total_duration = (time.perf_counter() - start_time) * 1000.0
                return UCSResult(
                    path=None,
                    total_cost=float("-inf"),
                    start_node=start_str,
                    goal_node=goal_str,
                    goal_found=False,
                    nodes_expanded=[],
                    steps=[],
                    cost_to_nodes={},
                    parents={},
                    graph=normalized_graph,
                    execution_time_ms=total_duration,
                    negative_cycle_detected=True,
                    negative_cycle_path=neg_cycle_path,
                    warning_message=warning_msg,
                )
        warning_msg = (
            f"Graph contains {len(non_positive_edges)} non-positive edge(s): {non_positive_edges}. "
            "UCS standard optimality assumes strictly positive edge costs."
        )

    # Priority queue item format:
    # (cost, tie_primary, tie_secondary, node_id, path_list)
    # This ensures Python tuple sorting uses cost first, then user-specified tie-breaker,
    # then sequence counter to never compare path lists or unsupported types.
    frontier: List[Tuple[Any, ...]] = []
    counter = 0

    def make_tie_key(cost: Union[int, float], node: str, path_len: int, seq: int) -> Tuple[Any, ...]:
        if tie_breaker == "fifo":
            return (cost, seq, node)
        elif tie_breaker == "hops":
            return (cost, path_len, node, seq)
        else:  # lexicographical default
            return (cost, node, seq)

    # Push initial start node: cumulative cost g(start) = 0
    initial_tuple = make_tie_key(0, start_str, 1, counter) + (start_str, [start_str])
    heapq.heappush(frontier, initial_tuple)

    # Best known cumulative cost to reach each node
    best_cost: Dict[str, Union[int, float]] = {start_str: 0}
    parents: Dict[str, Optional[str]] = {start_str: None}
    explored: Set[str] = set()
    nodes_expanded: List[str] = []
    steps: List[UCSStepRecord] = []
    step_num = 1

    # Main Uniform Cost Search Loop
    while frontier:
        # Extract the node with lowest cumulative path cost g(n)
        popped = heapq.heappop(frontier)
        # Reconstruct extracted elements based on tie_breaker tuple layout
        curr_cost: Union[int, float] = popped[0]
        curr_node: str = popped[-2]
        curr_path: List[str] = popped[-1]

        # Lazy deletion: if current popped cost exceeds recorded best cost, this entry is stale
        if curr_cost > best_cost.get(curr_node, float("inf")):
            snapshot = [(item[0], item[-2]) for item in sorted(frontier)]
            steps.append(
                UCSStepRecord(
                    step_number=step_num,
                    current_node=curr_node,
                    cumulative_cost=curr_cost,
                    action="SKIP_SUPERSEDED",
                    frontier_snapshot=snapshot,
                    explored_nodes=sorted(explored),
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    path_so_far=curr_path,
                    parent=curr_path[-2] if len(curr_path) >= 2 else None,
                )
            )
            step_num += 1
            continue

        # GOAL TEST: Applied upon selection for expansion, NOT upon generation!
        if curr_node == goal_str:
            snapshot = [(item[0], item[-2]) for item in sorted(frontier)]
            nodes_expanded.append(curr_node)
            steps.append(
                UCSStepRecord(
                    step_number=step_num,
                    current_node=curr_node,
                    cumulative_cost=curr_cost,
                    action="GOAL_REACHED",
                    frontier_snapshot=snapshot,
                    explored_nodes=sorted(explored),
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    path_so_far=curr_path,
                    parent=curr_path[-2] if len(curr_path) >= 2 else None,
                )
            )
            total_duration = (time.perf_counter() - start_time) * 1000.0
            return UCSResult(
                path=curr_path,
                total_cost=curr_cost,
                start_node=start_str,
                goal_node=goal_str,
                goal_found=True,
                nodes_expanded=nodes_expanded,
                steps=steps,
                cost_to_nodes=best_cost,
                parents=parents,
                graph=normalized_graph,
                execution_time_ms=total_duration,
                negative_cycle_detected=False,
                negative_cycle_path=None,
                warning_message=warning_msg,
            )

        # Mark node as explored
        explored.add(curr_node)
        nodes_expanded.append(curr_node)

        # Evaluate outgoing neighbors
        raw_neighbors = normalized_graph.get(curr_node, [])
        if sort_neighbors:
            # Sort neighbors by (edge_weight, target_node) for deterministic execution
            neighbors = sorted(raw_neighbors, key=lambda pair: (pair[1], str(pair[0])))
        else:
            neighbors = list(raw_neighbors)

        neighbors_evaluated: List[Tuple[str, Union[int, float]]] = []
        neighbors_enqueued: List[Tuple[str, Union[int, float]]] = []

        for target_node, edge_weight in neighbors:
            neighbors_evaluated.append((target_node, edge_weight))
            new_path_cost = curr_cost + edge_weight

            # Only consider node if not already finalized in explored set
            # or if a cheaper path is discovered (crucial if non-positive weights exist)
            if target_node not in explored or new_path_cost < best_cost.get(target_node, float("inf")):
                if new_path_cost < best_cost.get(target_node, float("inf")):
                    best_cost[target_node] = new_path_cost
                    parents[target_node] = curr_node
                    counter += 1
                    next_item = make_tie_key(new_path_cost, target_node, len(curr_path) + 1, counter) + (
                        target_node,
                        curr_path + [target_node],
                    )
                    heapq.heappush(frontier, next_item)
                    neighbors_enqueued.append((target_node, new_path_cost))
                    if target_node in explored:
                        explored.remove(target_node)

        snapshot = [(item[0], item[-2]) for item in sorted(frontier)]
        steps.append(
            UCSStepRecord(
                step_number=step_num,
                current_node=curr_node,
                cumulative_cost=curr_cost,
                action="EXPAND",
                frontier_snapshot=snapshot,
                explored_nodes=sorted(explored),
                neighbors_evaluated=neighbors_evaluated,
                neighbors_enqueued=neighbors_enqueued,
                path_so_far=curr_path,
                parent=curr_path[-2] if len(curr_path) >= 2 else None,
            )
        )
        step_num += 1

    # Frontier became empty without reaching goal (unreachable target)
    total_duration = (time.perf_counter() - start_time) * 1000.0
    return UCSResult(
        path=None,
        total_cost=float("inf"),
        start_node=start_str,
        goal_node=goal_str,
        goal_found=False,
        nodes_expanded=nodes_expanded,
        steps=steps,
        cost_to_nodes=best_cost,
        parents=parents,
        graph=normalized_graph,
        execution_time_ms=total_duration,
        negative_cycle_detected=False,
        negative_cycle_path=None,
        warning_message=warning_msg or f"Goal '{goal_str}' is unreachable from start '{start_str}'.",
    )


class UniformCostSearch:
    """Object-oriented interface for Uniform Cost Search and weighted graph exploration."""

    def __init__(
        self,
        graph: Union[Dict[str, Any], str, Sequence[Any]],
        tie_breaker: str = "lexicographical",
        detect_cycles: bool = True,
        sort_neighbors: bool = True,
    ) -> None:
        """Initialize the UniformCostSearch instance with a weighted directed graph.

        Args:
            graph: Graph dictionary, arrow syntax string, or sequence of edge tuples.
            tie_breaker: Tie-breaking strategy ('lexicographical', 'fifo', 'hops').
            detect_cycles: Whether to detect negative cycles automatically.
            sort_neighbors: Whether to evaluate neighbors in sorted order.
        """
        self.graph = parse_weighted_graph(graph)
        self.tie_breaker = tie_breaker
        self.detect_cycles = detect_cycles
        self.sort_neighbors = sort_neighbors

    def search(self, start: str, goal: str) -> UCSResult:
        """Execute UCS to find the optimal path from start to goal.

        Args:
            start: Starting vertex.
            goal: Target goal vertex.

        Returns:
            UCSResult containing optimal path, cost, and execution telemetry.
        """
        return uniform_cost_search(
            graph=self.graph,
            start=start,
            goal=goal,
            tie_breaker=self.tie_breaker,
            detect_cycles=self.detect_cycles,
            sort_neighbors=self.sort_neighbors,
        )

    def find_all_shortest_paths(self, start: str) -> Dict[str, Tuple[Optional[List[str]], Union[int, float]]]:
        """Compute shortest path and minimum cost from start to all reachable vertices.

        Args:
            start: Origin node.

        Returns:
            Dictionary mapping each vertex to a tuple of (path, total_cost).
        """
        results: Dict[str, Tuple[Optional[List[str]], Union[int, float]]] = {}
        for target in sorted(self.graph.keys()):
            res = self.search(start, target)
            results[target] = (res.path, res.total_cost)
        return results

    def check_negative_cycle(self, start: str) -> Tuple[bool, Optional[List[str]]]:
        """Check for reachable negative cycles from a specified source vertex.

        Args:
            start: Origin node.

        Returns:
            Tuple (has_negative_cycle, cycle_path).
        """
        has_cycle, cycle, _ = detect_negative_cycle(self.graph, start)
        return has_cycle, cycle


def visualize_ucs(
    result: UCSResult,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
) -> Path:
    """Generate a clean NetworkX/Matplotlib visual diagram of the UCS search graph.

    Highlights:
    - Start node (distinct light green color)
    - Goal node (distinct gold/orange color)
    - Optimal path edges (thick green arrows)
    - Explored nodes vs unreached nodes
    - Edge weights labeled clearly on every directed edge

    Args:
        result: Completed UCSResult object containing path, costs, and graph.
        output_path: File system path where the rendered PNG should be saved.
        show_plot: If True, calls plt.show() (used in interactive sessions).

    Returns:
        Path object pointing to the written image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph_dict = result.graph
    G = nx.DiGraph()

    # Populate graph nodes and edges with weights
    for u, neighbors in graph_dict.items():
        G.add_node(u)
        for v, weight in neighbors:
            G.add_edge(u, v, weight=weight)

    # Calculate optimal path edge set
    path_edges: Set[Tuple[str, str]] = set()
    if result.path and len(result.path) >= 2:
        for i in range(len(result.path) - 1):
            path_edges.add((result.path[i], result.path[i + 1]))

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.set_title(
        f"Uniform Cost Search (UCS): {result.start_node} -> {result.goal_node}\n"
        f"{result.format_path()} | Nodes Expanded: {len(result.nodes_expanded)}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Determine node positions using spring layout with fixed seed
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Color nodes according to role
    node_colors = []
    node_border_colors = []
    for node in G.nodes():
        if node == result.start_node:
            node_colors.append("#81C784")  # Soft Green
            node_border_colors.append("#2E7D32")
        elif node == result.goal_node:
            node_colors.append("#FFD54F")  # Gold
            node_border_colors.append("#F57F17")
        elif result.path and node in result.path:
            node_colors.append("#A5D6A7")  # Light Path Green
            node_border_colors.append("#388E3C")
        elif node in result.nodes_expanded:
            node_colors.append("#BBDEFB")  # Explored Blue
            node_border_colors.append("#1976D2")
        else:
            node_colors.append("#E0E0E0")  # Unreached Gray
            node_border_colors.append("#757575")

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=2.0,
        node_size=1100,
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=11,
        font_weight="bold",
        font_color="#212121",
    )

    # Draw normal edges
    normal_edges = [edge for edge in G.edges() if edge not in path_edges]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=normal_edges,
        edge_color="#9E9E9E",
        arrows=True,
        arrowsize=16,
        width=1.5,
        connectionstyle="arc3,rad=0.08",
    )

    # Draw optimal path edges highlighted in bold green
    if path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=list(path_edges),
            edge_color="#2E7D32",
            arrows=True,
            arrowsize=22,
            width=3.2,
            connectionstyle="arc3,rad=0.08",
        )

    # Draw edge weights
    edge_labels = {
        (u, v): f"{w:g}" if isinstance(w, float) else str(w)
        for u, v, w in G.edges(data="weight")
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_color="#C2185B",
        font_size=10,
        font_weight="bold",
        label_pos=0.45,
    )

    # Legend annotations
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#81C784", markersize=12, label="Start Node"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD54F", markersize=12, label="Goal Node"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#A5D6A7", markersize=12, label="Optimal Path Node"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBDEFB", markersize=12, label="Explored Node"),
        plt.Line2D([0], [0], color="#2E7D32", lw=3, label="Optimal Path Edge"),
        plt.Line2D([0], [0], color="#9E9E9E", lw=1.5, label="Standard Edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    if output_path is None:
        target_path = Path("reports") / "ucs_optimal_path.png"
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
    return target_path


def get_ucs_viva_reference() -> Dict[str, Any]:
    """Return comprehensive academic reference material for laboratory viva voce examinations.

    Returns:
        Structured dictionary containing core topics, in-depth technical answers,
        and comparison matrices across standard AI search algorithms.
    """
    return {
        "syllabus_area": "Uninformed Search Techniques",
        "algorithm": "Uniform Cost Search (UCS) / Dijkstra's Algorithm Variant",
        "core_topics": [
            {
                "topic": "UCS vs BFS",
                "question": "What is the key difference between Uniform Cost Search (UCS) and Breadth-First Search (BFS)?",
                "answer": (
                    "1. Step Costs: BFS assumes all step costs are identical (uniform cost c = 1 per hop), "
                    "expanding nodes level by level according to edge count. UCS handles arbitrary non-negative "
                    "edge weights c(u, v) >= eps > 0, expanding nodes in order of cumulative path cost g(n).\n"
                    "2. Frontier Data Structure: BFS uses a First-In, First-Out (FIFO) queue (O(1) popleft), whereas "
                    "UCS uses a Priority Queue (min-heap via heapq) ordered by g(n) (O(log N) heappop).\n"
                    "3. Optimality: BFS is optimal only when step costs are equal. When costs vary, BFS may return "
                    "a path with fewer hops but a much higher total cost. UCS is guaranteed to find the minimal cost "
                    "path for arbitrary non-negative weights.\n"
                    "4. Equivalence: When all edge weights in a graph are identical, UCS behaves identically to BFS "
                    "in expansion order (subject to tie-breaking)."
                ),
            },
            {
                "topic": "Priority Queue mechanisms & Python heapq",
                "question": "How does Python's heapq manage the frontier, and how does tuple sorting prevent comparison errors?",
                "answer": (
                    "1. Binary Min-Heap: `heapq` maintains a binary min-heap on top of a standard list where parent "
                    "index i has children at 2*i + 1 and 2*i + 2. The minimum element is always at index 0.\n"
                    "2. Tuple Sorting: When pushing tuples (g(n), tie_breaker, node_id, path), Python compares "
                    "elements sequentially: first g(n), then tie_breaker, then node_id. If g(n) values match, Python "
                    "uses the tie_breaker rather than comparing non-comparable data or paths.\n"
                    "3. Monotonic Counter / Lexicographical Ordering: Passing an integer counter or string node id as "
                    "the tie-breaker guarantees strict determinism and prevents Python from raising TypeError when "
                    "comparing complex objects.\n"
                    "4. Lazy Deletion: Since standard binary heaps do not support O(log N) decrease-key, whenever a "
                    "cheaper path is found, the new tuple is pushed to the heap. When popped, if its cost exceeds the "
                    "node's known best cost, it is skipped as a stale entry in O(1) time."
                ),
            },
            {
                "topic": "Effect of negative edge weights on UCS",
                "question": "What happens when a graph contains negative edge weights or negative cycles in UCS?",
                "answer": (
                    "1. Greedy Property Violation: UCS relies on the invariant that path costs g(n) are monotonically "
                    "non-decreasing along any path (g(v) >= g(u) since weight >= 0). Once a node is popped and added "
                    "to the explored set, its optimal cost is assumed final. A negative edge weight invalidates this "
                    "assumption because an explored node could later receive a cheaper path via an edge with a negative cost.\n"
                    "2. Suboptimal Paths: UCS with an explored set will fail to find the optimal path in graphs with "
                    "negative weights because it will not reopen closed nodes.\n"
                    "3. Negative Cycles: If a cycle has a net negative sum, looping through it reduces path cost toward "
                    "-infinity. An algorithm without negative cycle detection may enter an infinite loop or produce "
                    "meaningless results.\n"
                    "4. Proper Alternative: For graphs with negative edge weights, the Bellman-Ford algorithm (O(V * E)) "
                    "or SPFA (Shortest Path Faster Algorithm) should be used, both of which reliably detect negative cycles."
                ),
            },
            {
                "topic": "Goal Test Placement: Expansion vs Generation",
                "question": "Why MUST the goal test in Uniform Cost Search be performed upon expansion rather than generation?",
                "answer": (
                    "In Breadth-First Search, testing for the goal upon generation is sound because all step costs are 1, "
                    "so the first time a goal node is generated, it is guaranteed to be at the shallowest depth.\n"
                    "In Uniform Cost Search, the first path generated to a goal node might have a high cost (e.g. A->Goal: 10). "
                    "An alternate path through an unexpanded frontier node might be significantly cheaper (e.g. A->B: 1, B->Goal: 2). "
                    "If UCS terminated upon generating Goal at cost 10, it would fail to discover the optimal path of cost 3! "
                    "Therefore, the goal test must occur when the goal node is POPPED (expanded) from the priority queue, at "
                    "which point the min-heap guarantees that no cheaper path to the goal remains anywhere in the state space."
                ),
            },
            {
                "topic": "Time and Space Complexity Derivation",
                "question": "What are the time and space complexities of Uniform Cost Search, and how are they derived?",
                "answer": (
                    "1. Adjacency List Graph Formulation:\n"
                    "   - Time Complexity: O(E log V). Every vertex is popped from the min-heap at most once (V * log V). "
                    "For every outgoing edge, we perform a heap push in O(log V) time (E * log V). Total: O((V + E) log V) = O(E log V) "
                    "for connected graphs.\n"
                    "   - Space Complexity: O(V) auxiliary memory to store the priority queue frontier, best_cost table, and "
                    "explored set.\n"
                    "2. Russell & Norvig State-Space Formulation:\n"
                    "   - Time & Space: O(b^(1 + floor(C* / eps))), where C* is the optimal solution cost, eps is the minimum "
                    "step cost, and b is the branching factor. When step costs are very small compared to C*, UCS can explore "
                    "a vast number of small-step paths before reaching the goal, making space the primary bottleneck."
                ),
            },
        ],
        "comparison_matrix": {
            "headers": ["Algorithm", "Frontier Data Structure", "Optimal?", "Complete?", "Time Complexity", "Space Complexity"],
            "rows": [
                ["BFS", "FIFO Queue (collections.deque)", "Yes (if uniform cost)", "Yes (finite b)", "O(b^d) / O(V + E)", "O(b^d) / O(V)"],
                ["DFS", "LIFO Stack (list / recursion)", "No", "No (infinite paths)", "O(b^m) / O(V + E)", "O(b * m) / O(V)"],
                ["UCS", "Priority Queue (heapq min-heap)", "Yes (cost >= eps > 0)", "Yes (cost >= eps > 0)", "O(E log V)", "O(V)"],
                ["Greedy Best-First", "Priority Queue (by h(n))", "No", "No (loops possible)", "O(b^m)", "O(b^m)"],
                ["A* Search", "Priority Queue (by f = g + h)", "Yes (admissible h)", "Yes (cost >= eps > 0)", "O(b^d) to O(E log V)", "O(b^d) / O(V)"],
            ],
        },
    }
