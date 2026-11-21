"""A* Search Algorithm on Weighted Graphs.

This module implements the A* Search algorithm for finding optimal paths
from a start node to a goal node in a weighted graph using heuristic estimates,
adhering to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA Chapter 3).

Core AI Concept:
- A* Search is an informed (heuristic) search strategy that evaluates nodes using the
  evaluation function:
      f(n) = g(n) + h(n)
  where:
      g(n): Exact accumulated path cost from the start node to node n.
      h(n): Heuristic estimated cost from node n to the nearest goal state.
      f(n): Estimated total cost of the cheapest solution path through node n.

Priority Queue & Frontier Mechanics:
- The frontier (open list) is maintained as a min-heap using Python's `heapq` module.
- Each heap entry is structured as a tuple:
      (f_cost, tie_breaker_key, g_cost, current_node, path)
  This guarantees O(log N) insertion (`heappush`) and O(log N) extraction (`heappop`)
  of the minimum-f node.
- Tie-Breaking Strategies:
  When two nodes have identical f-values (f(u) == f(v)), Python compares subsequent
  tuple elements. We provide explicit, configurable tie-breaking strategies:
  1. 'fifo' (default): Uses a monotonic insertion sequence counter to preserve FIFO ordering.
  2. 'alphabetical' / 'lexicographical': Compares node identifiers alphabetically.
  3. 'lifo': Uses a decreasing sequence counter to prefer recently added nodes.

Node Re-Evaluation and Reopening:
- Maintain g_cost for each visited node in a dynamic dictionary (`best_g_costs`).
- If a new path to an already discovered node v is found with a strictly cheaper g_cost
  (`tentative_g < best_g_costs[v]`):
  1. The node's g_cost and f_cost are dynamically updated.
  2. A new entry with the cheaper cost is pushed into the priority queue.
  3. If the node was previously expanded and added to the closed set, and the heuristic
     is non-monotonic (inconsistent), A* allows reopening of the node by removing it
     from the closed set and re-inserting it into the frontier to guarantee optimality.
- Lazy Deletion:
  When popping an entry `(f, _, g, node, path)` from the min-heap, if its g-cost exceeds
  the recorded minimum cost for that node (`g > best_g_costs[node]`), it is recognized as
  superseded/stale and discarded in O(1) time without expansion.

Theoretical Properties & University Viva Topics:
1. Components:
   - g(n): Cost so far from start to n.
   - h(n): Estimated remaining cost from n to goal.
   - f(n) = g(n) + h(n): Estimated total path cost.
2. Optimality Conditions:
   - Tree Search: Optimal if h(n) is admissible (never overestimates: 0 <= h(n) <= h*(n)).
   - Graph Search: Optimal if h(n) is consistent / monotonic:
     h(u) <= c(u, v) + h(v) for every edge (u, v), which implies admissibility and
     guarantees that f(n) never decreases along any path. If h(n) is admissible but
     not consistent, graph search remains optimal only if closed nodes can be reopened
     upon discovering a cheaper g_cost.
3. Completeness:
   - Complete on graphs with finite branching factor and positive edge costs (c(u, v) >= eps > 0).
4. Complexity:
   - Time Complexity: O(E log V) in standard graph implementations with min-heap.
   - Space Complexity: O(V) to store nodes in the frontier and explored set.

Optional Challenge:
- Track and print the frontier contents at each step of the search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# ============================================================================
# Benchmark Graphs and Heuristic Dictionaries
# ============================================================================

# Default Sample Input from Syllabus (Question 28):
# Topology:
#   Start 'S' has edges:
#     - ('A', 3), ('B', 1), ('C', 4)
#   Node 'B' has edge:
#     - ('A', 1)  -> S -> B -> A costs 1 + 1 = 2, cheaper than S -> A (3)! (Re-evaluation)
#   Node 'A' has edge:
#     - ('G', 8)  -> S -> B -> A -> G costs 2 + 8 = 10
#   Node 'C' has edge:
#     - ('G', 5)  -> S -> C -> G costs 4 + 5 = 9!
#   Goal 'G' has no outgoing edges.
#
# Admissible Heuristics:
#   Actual shortest distances to G:
#     h*(G) = 0, h*(C) = 5, h*(A) = 8, h*(B) = 9, h*(S) = 9
#   Heuristics:
#     {'S': 7, 'A': 6, 'B': 7, 'C': 4, 'G': 0}
#   Since h(n) <= h*(n) for all n, heuristics are strictly admissible!
#
# Optimal Path: S -> C -> G, Total Cost: 9
DEFAULT_SAMPLE_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "S": [("A", 3), ("B", 1), ("C", 4)],
    "A": [("G", 8)],
    "B": [("A", 1)],
    "C": [("G", 5)],
    "G": [],
}

DEFAULT_SAMPLE_HEURISTICS: Dict[str, Union[int, float]] = {
    "S": 7,
    "A": 6,
    "B": 7,
    "C": 4,
    "G": 0,
}

# Classic Russell & Norvig Romania Map Subgraph (Arad to Bucharest)
# Straight-Line Distance to Bucharest (h_SLD)
ROMANIA_MAP_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "Arad": [("Zerind", 75), ("Sibiu", 140), ("Timisoara", 118)],
    "Zerind": [("Arad", 75), ("Oradea", 71)],
    "Oradea": [("Zerind", 71), ("Sibiu", 151)],
    "Timisoara": [("Arad", 118), ("Lugoj", 111)],
    "Lugoj": [("Timisoara", 111), ("Mehadia", 70)],
    "Mehadia": [("Lugoj", 70), ("Drobeta", 75)],
    "Drobeta": [("Mehadia", 75), ("Craiova", 120)],
    "Craiova": [("Drobeta", 120), ("Rimnicu Vilcea", 146), ("Pitesti", 138)],
    "Sibiu": [("Arad", 140), ("Oradea", 151), ("Fagaras", 99), ("Rimnicu Vilcea", 80)],
    "Fagaras": [("Sibiu", 99), ("Bucharest", 211)],
    "Rimnicu Vilcea": [("Sibiu", 80), ("Craiova", 146), ("Pitesti", 97)],
    "Pitesti": [("Rimnicu Vilcea", 97), ("Craiova", 138), ("Bucharest", 101)],
    "Bucharest": [("Fagaras", 211), ("Pitesti", 101), ("Giurgiu", 90)],
    "Giurgiu": [("Bucharest", 90)],
}

ROMANIA_SLD_HEURISTICS: Dict[str, Union[int, float]] = {
    "Arad": 366,
    "Bucharest": 0,
    "Craiova": 160,
    "Drobeta": 242,
    "Eforie": 161,
    "Fagaras": 176,
    "Giurgiu": 77,
    "Hirsova": 151,
    "Iasi": 226,
    "Lugoj": 244,
    "Mehadia": 241,
    "Neamt": 234,
    "Oradea": 380,
    "Pitesti": 100,
    "Rimnicu Vilcea": 193,
    "Sibiu": 253,
    "Timisoara": 329,
    "Urziceni": 80,
    "Vaslui": 199,
    "Zerind": 374,
}

# Reopening Demonstration Benchmark:
# Demonstrates an admissible but inconsistent heuristic that requires reopening
# an already-explored closed node to find the optimal path.
# Topology:
#   Start S -> A (cost 5), S -> B (cost 1)
#   B -> A (cost 2) (so path S -> B -> A has cost 3 < 5!)
#   A -> G (cost 4)
# True shortest path: S -> B -> A -> G (cost 1 + 2 + 4 = 7).
# Heuristics:
#   h(G) = 0
#   h(A) = 0 (admissible: 0 <= 4)
#   h(B) = 10 (admissible: 10 <= 6... wait, if h*(B) = 6, h(B)=10 is inadmissible!
#   Let h*(B) = 6. To be admissible, h(B) <= 6.
#   Let h(B) = 6, h(A) = 0.
#   Then f(A via S) = 5 + 0 = 5.
#   f(B via S) = 1 + 6 = 7.
#   Since f(A) = 5 < f(B) = 7, A is expanded BEFORE B!
#   When A is expanded, it enters closed_set.
#   Later, B is expanded, and edge B -> A (cost 2) gives tentative g(A) = 1 + 2 = 3 < 5!
#   If closed nodes cannot be reopened, A* fails to update A and misses optimal path.
#   With reopening enabled, A is reopened and optimal path S -> B -> A -> G (cost 7) is found!
REOPENING_DEMO_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "S": [("A", 5), ("B", 1)],
    "B": [("A", 2)],
    "A": [("G", 4)],
    "G": [],
}

REOPENING_INCONSISTENT_HEURISTICS: Dict[str, Union[int, float]] = {
    "S": 5,
    "A": 0,
    "B": 6,
    "G": 0,
}

# Classic Diamond Graph:
# Direct path vs multi-hop path
DIAMOND_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "S": [("A", 1), ("B", 5)],
    "A": [("C", 2)],
    "B": [("G", 2)],
    "C": [("G", 1)],
    "G": [],
}

DIAMOND_HEURISTICS: Dict[str, Union[int, float]] = {
    "S": 3,
    "A": 2,
    "B": 2,
    "C": 1,
    "G": 0,
}

# Cyclic Graph to verify cycle avoidance:
CYCLIC_WEIGHTED_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 2), ("C", 5)],
    "B": [("A", 2), ("C", 1), ("D", 4)],
    "C": [("D", 1)],
    "D": [("B", 3), ("G", 2)],
    "G": [],
}
CYCLIC_GRAPH = CYCLIC_WEIGHTED_GRAPH

CYCLIC_HEURISTICS: Dict[str, Union[int, float]] = {
    "A": 5,
    "B": 3,
    "C": 2,
    "D": 1,
    "G": 0,
}

# Disconnected Graph to verify unreachable goal handling:
DISCONNECTED_GRAPH: Dict[str, List[Tuple[str, Union[int, float]]]] = {
    "A": [("B", 2)],
    "B": [],
    "C": [("G", 3)],
    "G": [],
}

DISCONNECTED_HEURISTICS: Dict[str, Union[int, float]] = {
    "A": 5,
    "B": 4,
    "C": 2,
    "G": 0,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class AStarStepRecord:
    """Immutable record capturing an individual step of A* Search.

    Attributes:
        step_number: 1-based chronological expansion sequence index.
        current_node: Node identifier extracted from the priority queue.
        g_cost: Path cost g(n) from start to current_node.
        h_cost: Heuristic cost h(n) from current_node to goal.
        f_cost: Evaluation function value f(n) = g(n) + h(n).
        action: Semantic action label (e.g. 'START', 'EXPAND', 'GOAL_REACHED', 'SKIP_SUPERSEDED').
        frontier_snapshot: Snapshot of (f, g, node) tuples in the priority queue after this step.
        explored_nodes: Sorted snapshot of all nodes in the closed set after this step.
        neighbors_evaluated: List of (neighbor, edge_cost, tentative_g) tuples evaluated.
        neighbors_enqueued: List of (neighbor, g_cost, f_cost) tuples pushed into frontier.
        reopened_nodes: List of nodes that were previously closed and reopened during this step.
        path_so_far: Sequence of nodes traversed to reach current_node.
        parent: Predecessor node from which current_node was reached.
    """

    step_number: int
    current_node: str
    g_cost: Union[int, float]
    h_cost: Union[int, float]
    f_cost: Union[int, float]
    action: str
    frontier_snapshot: List[Tuple[Union[int, float], Union[int, float], str]]
    explored_nodes: List[str]
    neighbors_evaluated: List[Tuple[str, Union[int, float], Union[int, float]]]
    neighbors_enqueued: List[Tuple[str, Union[int, float], Union[int, float]]]
    reopened_nodes: List[str]
    path_so_far: List[str]
    parent: Optional[str] = None

    def format_row(self) -> str:
        """Format the step into a structured table row for terminal presentation.

        Returns:
            Formatted row string.
        """
        frontier_items = [f"{node}(f={f:g},g={g:g})" if isinstance(f, float) or isinstance(g, float)
                          else f"{node}(f={f},g={g})" for f, g, node in self.frontier_snapshot]
        frontier_str = "[" + ", ".join(frontier_items) + "]" if frontier_items else "[]"
        enqueued_items = [f"{node}(f={f},g={g})" for node, g, f in self.neighbors_enqueued]
        enqueued_str = "[" + ", ".join(enqueued_items) + "]" if enqueued_items else "-"
        parent_str = str(self.parent) if self.parent is not None else "START"

        g_str = f"{self.g_cost:g}" if isinstance(self.g_cost, float) else str(self.g_cost)
        h_str = f"{self.h_cost:g}" if isinstance(self.h_cost, float) else str(self.h_cost)
        f_str = f"{self.f_cost:g}" if isinstance(self.f_cost, float) else str(self.f_cost)

        return (
            f"{self.step_number:<5} | {self.current_node:<6} | {g_str:<5} | {h_str:<5} | {f_str:<5} | "
            f"{self.action:<14} | {parent_str:<6} | {enqueued_str:<18} | {frontier_str}"
        )


@dataclass
class AStarResult:
    """Encapsulates the complete outcome of an A* Search execution.

    Attributes:
        path: Ordered list of nodes forming the optimal path from start to goal (None if unreachable).
        total_cost: Minimum cumulative cost g(goal) (infinity if unreachable).
        start_node: Origin node for the search.
        goal_node: Target destination node.
        goal_found: Boolean flag indicating if goal was reached.
        nodes_expanded: Chronological sequence of nodes popped and expanded.
        steps: Chronological list of AStarStepRecord objects detailing the search progression.
        g_costs: Mapping of all reached nodes to their minimum confirmed g(n) cost.
        f_costs: Mapping of all reached nodes to their final f(n) cost.
        parents: Mapping of each node to its predecessor in the optimal path tree.
        graph: Adjacency list representation of the searched graph.
        heuristics: Heuristic dictionary used during search.
        re_evaluations_count: Number of times a node's g_cost was improved.
        nodes_reopened_count: Number of times a closed node was reopened.
        execution_time_ms: Wall-clock execution time in milliseconds.
    """

    path: Optional[List[str]]
    total_cost: Union[int, float]
    start_node: str
    goal_node: str
    goal_found: bool
    nodes_expanded: List[str]
    steps: List[AStarStepRecord]
    g_costs: Dict[str, Union[int, float]]
    f_costs: Dict[str, Union[int, float]]
    parents: Dict[str, Optional[str]]
    graph: Dict[str, List[Tuple[str, Union[int, float]]]]
    heuristics: Dict[str, Union[int, float]]
    re_evaluations_count: int = 0
    nodes_reopened_count: int = 0
    execution_time_ms: float = 0.0

    def format_path(self, delimiter: str = " -> ") -> str:
        """Format the path and cost matching the exact university sample output.

        Example:
            'Optimal Path: S -> C -> G, Total Cost: 9'

        Args:
            delimiter: Separator between successive node identifiers.

        Returns:
            Human-readable path and cost summary string.
        """
        if self.path is None or not self.goal_found:
            return "Optimal Path: None (Unreachable), Total Cost: Infinity"
        path_str = delimiter.join(self.path)
        cost_str = f"{self.total_cost:g}" if isinstance(self.total_cost, float) else str(self.total_cost)
        return f"Optimal Path: {path_str}, Total Cost: {cost_str}"

    def format_step_table(self) -> str:
        """Return a formatted ASCII execution trace table of the A* search process.

        Returns:
            Multi-line ASCII table string.
        """
        headers = (
            f"{'Step':<5} | {'Node':<6} | {'g(n)':<5} | {'h(n)':<5} | {'f(n)':<5} | "
            f"{'Action':<14} | {'Parent':<6} | {'Enqueued':<18} | {'Frontier Priority Queue':<24}"
        )
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def format_frontier_at_steps(self) -> str:
        """Format the frontier contents at each step (Optional Challenge).

        Returns:
            Formatted multi-line string detailing the priority queue contents at each step.
        """
        lines = ["=== A* Frontier (Priority Queue) Contents at Each Step ==="]
        for step in self.steps:
            frontier_items = [
                f"{node}(f={f:g}, g={g:g})" if isinstance(f, float) or isinstance(g, float)
                else f"{node}(f={f}, g={g})"
                for f, g, node in step.frontier_snapshot
            ]
            frontier_str = "[" + ", ".join(frontier_items) + "]" if frontier_items else "[]"
            lines.append(
                f"Step {step.step_number:02d} | Popped: {step.current_node:<4} (Action: {step.action:<14}) | "
                f"Frontier: {frontier_str}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary.

        Returns:
            Dictionary containing result metrics and execution metadata.
        """
        return {
            "path": self.path,
            "total_cost": self.total_cost,
            "start_node": self.start_node,
            "goal_node": self.goal_node,
            "goal_found": self.goal_found,
            "nodes_expanded": self.nodes_expanded,
            "re_evaluations_count": self.re_evaluations_count,
            "nodes_reopened_count": self.nodes_reopened_count,
            "execution_time_ms": round(self.execution_time_ms, 4),
            "best_g_costs": {k: (v if not isinstance(v, float) or v.is_integer() else round(v, 4))
                             for k, v in self.g_costs.items()},
            "steps_count": len(self.steps),
        }


# ============================================================================
# Normalization & Parsing Utilities
# ============================================================================

def normalize_graph(
    graph: Mapping[str, Sequence[Any]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Normalize user-supplied graph representations into a standard adjacency list.

    Converts edge pairs `(neighbor, weight)` or single neighbors (default weight 1.0)
    into a canonical `Dict[str, List[Tuple[str, float]]]`.

    Args:
        graph: Raw input graph dictionary.

    Returns:
        Standardized adjacency dictionary with float edge costs.

    Raises:
        ValueError: If graph contains negative edge weights or malformed edges.
    """
    normalized: Dict[str, List[Tuple[str, float]]] = {}
    for node, edges in graph.items():
        node_str = str(node).strip()
        edge_list: List[Tuple[str, float]] = []
        for edge in edges:
            if isinstance(edge, (tuple, list)):
                if len(edge) < 2:
                    neighbor = str(edge[0]).strip()
                    cost = 1.0
                else:
                    neighbor = str(edge[0]).strip()
                    cost = float(edge[1])
            else:
                neighbor = str(edge).strip()
                cost = 1.0

            if cost < 0:
                raise ValueError(
                    f"Negative edge weight {cost} detected on edge {node_str} -> {neighbor}. "
                    "A* requires non-negative edge costs."
                )
            edge_list.append((neighbor, cost))

        normalized[node_str] = edge_list

    # Ensure all referenced target nodes exist as keys
    for edges in list(normalized.values()):
        for target, _ in edges:
            if target not in normalized:
                normalized[target] = []

    return normalized


def normalize_heuristics(
    heuristics: Mapping[str, Union[int, float]],
) -> Dict[str, float]:
    """Normalize heuristic dictionary keys to stripped strings and values to floats.

    Args:
        heuristics: Raw heuristic dictionary mapping nodes to estimated distances.

    Returns:
        Cleaned dictionary mapping node strings to float heuristic values.

    Raises:
        ValueError: If heuristic value is negative.
    """
    normalized: Dict[str, float] = {}
    for node, h_val in heuristics.items():
        node_str = str(node).strip()
        val = float(h_val)
        if val < 0:
            raise ValueError(
                f"Negative heuristic value {val} for node '{node_str}'. "
                "Heuristic h(n) must be non-negative."
            )
        normalized[node_str] = val
    return normalized


def parse_graph(text: str) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
    """Parse graph definition from JSON string, semicolon notation, or edge list.

    Supported Formats:
    1. JSON object: '{"S": [["A", 3], ["B", 1]], "A": [["G", 8]]}'
    2. Adjacency notation: 'S: A=3, B=1; A: G=8; G:'
    3. Arrow notation: 'S->A: 3, S->B: 1, A->G: 8'

    Args:
        text: String describing graph topology and weights.

    Returns:
        Adjacency dictionary mapping node to list of (neighbor, weight) tuples.

    Raises:
        ValueError: If string format cannot be parsed.
    """
    text = text.strip()
    if not text:
        return {}

    # 1. Attempt JSON parse
    if text.startswith("{") and text.endswith("}"):
        try:
            raw = json.loads(text)
            parsed: Dict[str, List[Tuple[str, Union[int, float]]]] = {}
            for k, v in raw.items():
                parsed[str(k).strip()] = [
                    (str(item[0]).strip(), float(item[1]) if len(item) > 1 else 1.0)
                    if isinstance(item, (list, tuple)) else (str(item).strip(), 1.0)
                    for item in v
                ]
            return parsed
        except Exception:
            pass

    graph: Dict[str, List[Tuple[str, Union[int, float]]]] = {}

    # 2. Attempt Arrow notation: 'S->A: 3, S->B: 1'
    if "->" in text:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        for tok in tokens:
            m = re.match(r"^([A-Za-z0-9_-]+)\s*->\s*([A-Za-z0-9_-]+)(?:\s*:\s*([0-9.]+))?$", tok)
            if m:
                u, v, w = m.group(1), m.group(2), m.group(3)
                cost = float(w) if w is not None else 1.0
                graph.setdefault(u, []).append((v, int(cost) if cost.is_integer() else cost))
                graph.setdefault(v, [])
            else:
                raise ValueError(f"Unable to parse edge token: '{tok}'")
        return graph

    # 3. Attempt Semicolon notation: 'S: A=3, B=1; A: G=8'
    segments = [s.strip() for s in text.split(";") if s.strip()]
    for seg in segments:
        if ":" not in seg:
            continue
        node, rest = seg.split(":", 1)
        node = node.strip()
        graph.setdefault(node, [])
        rest = rest.strip()
        if not rest:
            continue
        neighbors = [n.strip() for n in rest.split(",") if n.strip()]
        for neigh in neighbors:
            if "=" in neigh or ":" in neigh:
                sep = "=" if "=" in neigh else ":"
                n_id, w = neigh.split(sep, 1)
                cost = float(w.strip())
                graph[node].append((n_id.strip(), int(cost) if cost.is_integer() else cost))
                graph.setdefault(n_id.strip(), [])
            else:
                graph[node].append((neigh, 1.0))
                graph.setdefault(neigh, [])

    if not graph:
        raise ValueError(f"Could not parse graph string: '{text}'")
    return graph


def parse_heuristics(text: str) -> Dict[str, Union[int, float]]:
    """Parse heuristic dictionary from JSON string or comma-delimited key-value pairs.

    Supported Formats:
    1. JSON object: '{"S": 7, "A": 6, "B": 7, "C": 4, "G": 0}'
    2. Comma pairs: 'S: 7, A: 6, B: 7, C: 4, G: 0'
    3. Equality pairs: 'S=7, A=6, B=7, C=4, G=0'

    Args:
        text: String describing node heuristic estimates.

    Returns:
        Dictionary mapping node names to heuristic values.

    Raises:
        ValueError: If string format cannot be parsed.
    """
    text = text.strip()
    if not text:
        return {}

    # 1. JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            raw = json.loads(text)
            return {str(k).strip(): (int(v) if float(v).is_integer() else float(v)) for k, v in raw.items()}
        except Exception:
            pass

    heuristics: Dict[str, Union[int, float]] = {}
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    for tok in tokens:
        if ":" in tok or "=" in tok:
            sep = ":" if ":" in tok else "="
            k, v = tok.split(sep, 1)
            val = float(v.strip())
            heuristics[k.strip()] = int(val) if val.is_integer() else val
        else:
            raise ValueError(f"Invalid heuristic token: '{tok}'")

    if not heuristics:
        raise ValueError(f"Could not parse heuristic string: '{text}'")
    return heuristics


# ============================================================================
# Core A* Search Algorithm
# ============================================================================

def a_star_search(
    graph: Mapping[str, Sequence[Any]],
    heuristics: Mapping[str, Union[int, float]],
    start: str,
    goal: str,
    tie_breaking: str = "fifo",
    allow_reopen: bool = True,
) -> AStarResult:
    """Execute A* Search on a weighted graph with heuristic estimates.

    Evaluates nodes using f(n) = g(n) + h(n), finding the optimal path from
    start to goal.

    Algorithmic Guarantees & Operations:
    1. Priority Queue: Managed as a min-heap via Python's `heapq`.
    2. Dynamic g_cost Dictionary: Tracks the best known cumulative cost from start.
    3. Re-evaluation: If an alternate route to an active or visited node is found
       with a cheaper g_cost, its costs are updated and re-enqueued.
    4. Reopening: If `allow_reopen` is True, closed nodes can be reopened upon
       discovering a cheaper path (vital when heuristics are admissible but non-consistent).
    5. Lazy Deletion: Obsolete frontier entries with stale higher g-costs are discarded in O(1).
    6. Goal Test: Applied when a node is extracted from the priority queue, guaranteeing optimality.

    Args:
        graph: Adjacency list mapping node to list of (neighbor, weight) pairs.
        heuristics: Mapping from node name to heuristic estimate h(n) to goal.
        start: Starting node identifier.
        goal: Destination goal node identifier.
        tie_breaking: Tie-breaking policy when f-costs collide ('fifo', 'alphabetical', 'lifo').
        allow_reopen: Whether to reopen nodes from the closed set if a lower g_cost is found.

    Returns:
        AStarResult containing the optimal path, total cost, step trace, and metrics.

    Raises:
        ValueError: If start or goal node is missing or edge costs are negative.
        KeyError: If heuristic estimate for a reached node is missing.
    """
    start_time = time.perf_counter()

    norm_graph = normalize_graph(graph)
    norm_heuristics = normalize_heuristics(heuristics)

    start_node = str(start).strip()
    goal_node = str(goal).strip()

    if start_node not in norm_graph:
        raise ValueError(f"Start node '{start_node}' does not exist in graph.")
    if goal_node not in norm_graph:
        raise ValueError(f"Goal node '{goal_node}' does not exist in graph.")
    if start_node not in norm_heuristics:
        raise KeyError(f"Heuristic estimate for start node '{start_node}' is missing.")
    if goal_node not in norm_heuristics:
        raise KeyError(f"Heuristic estimate for goal node '{goal_node}' is missing.")

    # Base Case: start == goal
    if start_node == goal_node:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        h_start = norm_heuristics[start_node]
        step0 = AStarStepRecord(
            step_number=1,
            current_node=start_node,
            g_cost=0.0,
            h_cost=h_start,
            f_cost=h_start,
            action="GOAL_REACHED",
            frontier_snapshot=[],
            explored_nodes=[start_node],
            neighbors_evaluated=[],
            neighbors_enqueued=[],
            reopened_nodes=[],
            path_so_far=[start_node],
            parent=None,
        )
        return AStarResult(
            path=[start_node],
            total_cost=0,
            start_node=start_node,
            goal_node=goal_node,
            goal_found=True,
            nodes_expanded=[start_node],
            steps=[step0],
            g_costs={start_node: 0},
            f_costs={start_node: norm_heuristics[start_node]},
            parents={start_node: None},
            graph={k: [(n, int(c) if c.is_integer() else c) for n, c in v] for k, v in norm_graph.items()},
            heuristics={k: int(v) if v.is_integer() else v for k, v in norm_heuristics.items()},
            re_evaluations_count=0,
            nodes_reopened_count=0,
            execution_time_ms=elapsed,
        )

    # Priority queue min-heap entries:
    # (f_cost, tie_breaker_key, g_cost, current_node, path)
    frontier: List[Tuple[float, Any, float, str, List[str]]] = []
    insertion_counter = 0

    def make_tie_key(node: str, step_idx: int) -> Any:
        if tie_breaking == "alphabetical":
            return node
        elif tie_breaking == "lifo":
            return -step_idx
        return step_idx  # 'fifo' default

    start_g = 0.0
    start_h = norm_heuristics[start_node]
    start_f = start_g + start_h

    heapq.heappush(
        frontier,
        (start_f, make_tie_key(start_node, insertion_counter), start_g, start_node, [start_node]),
    )
    insertion_counter += 1

    # Dynamic best g_costs dictionary
    best_g_costs: Dict[str, float] = {start_node: 0.0}
    parents: Dict[str, Optional[str]] = {start_node: None}
    closed_set: Set[str] = set()

    nodes_expanded: List[str] = []
    step_records: List[AStarStepRecord] = []
    re_evaluations_count = 0
    nodes_reopened_count = 0
    step_counter = 0

    while frontier:
        f_val, _, g_val, current_node, current_path = heapq.heappop(frontier)

        # Lazy Deletion Check:
        # If g_val is greater than best_g_costs[current_node], this entry is superseded.
        if g_val > best_g_costs[current_node]:
            # Record superseded step
            step_counter += 1
            h_val = norm_heuristics.get(current_node, 0.0)
            snapshot = [(f, g, n) for f, _, g, n, _ in frontier]
            step_records.append(
                AStarStepRecord(
                    step_number=step_counter,
                    current_node=current_node,
                    g_cost=int(g_val) if g_val.is_integer() else g_val,
                    h_cost=int(h_val) if h_val.is_integer() else h_val,
                    f_cost=int(f_val) if f_val.is_integer() else f_val,
                    action="SKIP_SUPERSEDED",
                    frontier_snapshot=[(int(f) if f.is_integer() else f,
                                        int(g) if g.is_integer() else g, n) for f, g, n in snapshot],
                    explored_nodes=sorted(list(closed_set)),
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    reopened_nodes=[],
                    path_so_far=current_path,
                    parent=parents.get(current_node),
                )
            )
            continue

        step_counter += 1
        h_val = norm_heuristics.get(current_node, 0.0)

        # Goal Test upon expansion
        if current_node == goal_node:
            closed_set.add(current_node)
            nodes_expanded.append(current_node)
            snapshot = [(f, g, n) for f, _, g, n, _ in frontier]
            step_records.append(
                AStarStepRecord(
                    step_number=step_counter,
                    current_node=current_node,
                    g_cost=int(g_val) if g_val.is_integer() else g_val,
                    h_cost=int(h_val) if h_val.is_integer() else h_val,
                    f_cost=int(f_val) if f_val.is_integer() else f_val,
                    action="GOAL_REACHED",
                    frontier_snapshot=[(int(f) if f.is_integer() else f,
                                        int(g) if g.is_integer() else g, n) for f, g, n in snapshot],
                    explored_nodes=sorted(list(closed_set)),
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    reopened_nodes=[],
                    path_so_far=current_path,
                    parent=parents.get(current_node),
                )
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            int_cost = int(g_val) if g_val.is_integer() else g_val
            return AStarResult(
                path=current_path,
                total_cost=int_cost,
                start_node=start_node,
                goal_node=goal_node,
                goal_found=True,
                nodes_expanded=nodes_expanded,
                steps=step_records,
                g_costs={k: (int(v) if v.is_integer() else v) for k, v in best_g_costs.items()},
                f_costs={k: (int(best_g_costs[k] + norm_heuristics.get(k, 0))
                             if (best_g_costs[k] + norm_heuristics.get(k, 0)).is_integer()
                             else best_g_costs[k] + norm_heuristics.get(k, 0)) for k in best_g_costs},
                parents=parents,
                graph={k: [(n, int(c) if c.is_integer() else c) for n, c in v] for k, v in norm_graph.items()},
                heuristics={k: int(v) if v.is_integer() else v for k, v in norm_heuristics.items()},
                re_evaluations_count=re_evaluations_count,
                nodes_reopened_count=nodes_reopened_count,
                execution_time_ms=elapsed,
            )

        # Mark current node as explored
        closed_set.add(current_node)
        nodes_expanded.append(current_node)

        neighbors_evaluated: List[Tuple[str, Union[int, float], Union[int, float]]] = []
        neighbors_enqueued: List[Tuple[str, Union[int, float], Union[int, float]]] = []
        reopened_in_step: List[str] = []

        # Evaluate successor neighbors
        for neighbor, edge_cost in norm_graph.get(current_node, []):
            if neighbor not in norm_heuristics:
                raise KeyError(f"Heuristic estimate for node '{neighbor}' is missing from heuristics dictionary.")

            tentative_g = g_val + edge_cost
            neighbors_evaluated.append((
                neighbor,
                int(edge_cost) if edge_cost.is_integer() else edge_cost,
                int(tentative_g) if tentative_g.is_integer() else tentative_g,
            ))

            is_already_visited = neighbor in best_g_costs
            is_in_closed = neighbor in closed_set

            # Check if this new path to neighbor is strictly cheaper
            if not is_already_visited or tentative_g < best_g_costs[neighbor]:
                if is_already_visited:
                    re_evaluations_count += 1

                # Reopening logic:
                # If neighbor is in closed_set, but we found a strictly cheaper g_cost:
                if is_in_closed:
                    if allow_reopen:
                        closed_set.remove(neighbor)
                        reopened_in_step.append(neighbor)
                        nodes_reopened_count += 1
                    else:
                        # Reopening not allowed; skip closed node
                        continue

                # Update best known g_cost and parent
                best_g_costs[neighbor] = tentative_g
                parents[neighbor] = current_node
                h_neighbor = norm_heuristics[neighbor]
                f_neighbor = tentative_g + h_neighbor

                tie_key = make_tie_key(neighbor, insertion_counter)
                insertion_counter += 1

                heapq.heappush(
                    frontier,
                    (f_neighbor, tie_key, tentative_g, neighbor, current_path + [neighbor]),
                )
                neighbors_enqueued.append((
                    neighbor,
                    int(tentative_g) if tentative_g.is_integer() else tentative_g,
                    int(f_neighbor) if f_neighbor.is_integer() else f_neighbor,
                ))

        snapshot = [(f, g, n) for f, _, g, n, _ in frontier]
        step_records.append(
            AStarStepRecord(
                step_number=step_counter,
                current_node=current_node,
                g_cost=int(g_val) if g_val.is_integer() else g_val,
                h_cost=int(h_val) if h_val.is_integer() else h_val,
                f_cost=int(f_val) if f_val.is_integer() else f_val,
                action="EXPAND",
                frontier_snapshot=[(int(f) if f.is_integer() else f,
                                    int(g) if g.is_integer() else g, n) for f, g, n in snapshot],
                explored_nodes=sorted(list(closed_set)),
                neighbors_evaluated=neighbors_evaluated,
                neighbors_enqueued=neighbors_enqueued,
                reopened_nodes=reopened_in_step,
                path_so_far=current_path,
                parent=parents.get(current_node),
            )
        )

    # Goal unreachable
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return AStarResult(
        path=None,
        total_cost=float("inf"),
        start_node=start_node,
        goal_node=goal_node,
        goal_found=False,
        nodes_expanded=nodes_expanded,
        steps=step_records,
        g_costs={k: (int(v) if v.is_integer() else v) for k, v in best_g_costs.items()},
        f_costs={k: (int(best_g_costs[k] + norm_heuristics.get(k, 0))
                     if (best_g_costs[k] + norm_heuristics.get(k, 0)).is_integer()
                     else best_g_costs[k] + norm_heuristics.get(k, 0)) for k in best_g_costs},
        parents=parents,
        graph={k: [(n, int(c) if c.is_integer() else c) for n, c in v] for k, v in norm_graph.items()},
        heuristics={k: int(v) if v.is_integer() else v for k, v in norm_heuristics.items()},
        re_evaluations_count=re_evaluations_count,
        nodes_reopened_count=nodes_reopened_count,
        execution_time_ms=elapsed,
    )


# ============================================================================
# Object-Oriented Wrapper Class
# ============================================================================

class AStarSearch:
    """Object-Oriented interface for executing and analyzing A* Search workloads.

    Encapsulates graph topology, heuristic dictionaries, and search parameters.

    Example:
        >>> solver = AStarSearch(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS)
        >>> result = solver.search(start='S', goal='G')
        >>> print(result.format_path())
        Optimal Path: S -> C -> G, Total Cost: 9
    """

    def __init__(
        self,
        graph: Mapping[str, Sequence[Any]],
        heuristics: Mapping[str, Union[int, float]],
        tie_breaking: str = "fifo",
        allow_reopen: bool = True,
    ) -> None:
        """Initialize solver with graph and heuristic definitions.

        Args:
            graph: Graph adjacency structure.
            heuristics: Heuristic dictionary mapping node names to estimates.
            tie_breaking: Priority queue tie-breaker ('fifo', 'alphabetical', 'lifo').
            allow_reopen: Flag controlling whether closed nodes may be reopened.
        """
        self.graph = normalize_graph(graph)
        self.heuristics = normalize_heuristics(heuristics)
        self.tie_breaking = tie_breaking
        self.allow_reopen = allow_reopen

    def search(self, start: str, goal: str) -> AStarResult:
        """Run A* search and return complete structured result.

        Args:
            start: Start node identifier.
            goal: Goal node identifier.

        Returns:
            AStarResult object.
        """
        return a_star_search(
            graph=self.graph,
            heuristics=self.heuristics,
            start=start,
            goal=goal,
            tie_breaking=self.tie_breaking,
            allow_reopen=self.allow_reopen,
        )

    def find_path(
        self,
        start: str,
        goal: str,
    ) -> Tuple[Optional[List[str]], Union[int, float]]:
        """Convenience method returning optimal path list and total cost tuple.

        Args:
            start: Start node.
            goal: Goal node.

        Returns:
            Tuple of (optimal_path, total_cost).
        """
        res = self.search(start, goal)
        return res.path, res.total_cost


# ============================================================================
# Theoretical Analysis & Heuristic Verification Utilities
# ============================================================================

def compute_all_pairs_shortest_paths(
    graph: Mapping[str, Sequence[Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute true shortest path costs h*(u, v) between all node pairs using Dijkstra.

    Args:
        graph: Graph adjacency dictionary.

    Returns:
        Mapping from source node to dictionary of shortest path costs to all reachable nodes.
    """
    norm = normalize_graph(graph)
    nodes = list(norm.keys())
    all_dists: Dict[str, Dict[str, float]] = {}

    for src in nodes:
        dists: Dict[str, float] = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dists[u]:
                continue
            for v, w in norm.get(u, []):
                new_d = d + w
                if v not in dists or new_d < dists[v]:
                    dists[v] = new_d
                    heapq.heappush(pq, (new_d, v))
        all_dists[src] = dists

    return all_dists


def check_admissibility(
    graph: Mapping[str, Sequence[Any]],
    heuristics: Mapping[str, Union[int, float]],
    goal: str,
) -> Dict[str, Any]:
    """Verify if the heuristic dictionary is admissible with respect to target goal.

    A heuristic h(n) is admissible if for all nodes n:
        0 <= h(n) <= h*(n)
    where h*(n) is the true minimal cost from n to goal.

    Args:
        graph: Graph adjacency structure.
        heuristics: Heuristic dictionary.
        goal: Destination goal state.

    Returns:
        Dictionary containing:
        - is_admissible: Boolean flag
        - violations: List of nodes where h(n) > h*(n)
        - comparisons: Table of {node: (h(n), h*(n))}
    """
    norm_graph = normalize_graph(graph)
    norm_heuristics = normalize_heuristics(heuristics)
    goal = str(goal).strip()

    all_dists = compute_all_pairs_shortest_paths(norm_graph)

    violations = []
    comparisons = {}

    for node in norm_graph:
        h_val = norm_heuristics.get(node, 0.0)
        h_star = all_dists.get(node, {}).get(goal, float("inf"))
        comparisons[node] = {
            "h": h_val,
            "h_star": h_star if h_star != float("inf") else "Infinity",
            "is_valid": h_val <= h_star,
        }
        if h_val > h_star:
            violations.append({
                "node": node,
                "h": h_val,
                "h_star": h_star,
                "overestimation": h_val - h_star,
            })

    return {
        "is_admissible": len(violations) == 0,
        "violations": violations,
        "comparisons": comparisons,
    }


def check_consistency(
    graph: Mapping[str, Sequence[Any]],
    heuristics: Mapping[str, Union[int, float]],
) -> Dict[str, Any]:
    """Verify if heuristic function satisfies the consistency (monotonicity) condition.

    A heuristic is consistent if for every edge (u, v) with step cost c(u, v):
        h(u) <= c(u, v) + h(v)   (Triangle Inequality)

    Consistency guarantees that f(n) is non-decreasing along any path and that
    when a node is selected for expansion, its discovered path is guaranteed optimal,
    meaning closed nodes never need to be reopened.

    Args:
        graph: Graph adjacency structure.
        heuristics: Heuristic dictionary.

    Returns:
        Dictionary detailing monotonicity status and any edge violations.
    """
    norm_graph = normalize_graph(graph)
    norm_heuristics = normalize_heuristics(heuristics)

    violations = []
    edge_checks = []

    for u, edges in norm_graph.items():
        h_u = norm_heuristics.get(u, 0.0)
        for v, cost in edges:
            h_v = norm_heuristics.get(v, 0.0)
            lhs = h_u
            rhs = cost + h_v
            is_consistent = lhs <= rhs + 1e-9
            edge_checks.append({
                "edge": f"{u} -> {v}",
                "cost": cost,
                "h(u)": h_u,
                "c(u,v) + h(v)": rhs,
                "consistent": is_consistent,
            })
            if not is_consistent:
                violations.append({
                    "edge": f"{u} -> {v}",
                    "h(u)": h_u,
                    "c(u,v)": cost,
                    "h(v)": h_v,
                    "deficit": round(lhs - rhs, 4),
                })

    return {
        "is_consistent": len(violations) == 0,
        "violations": violations,
        "edge_checks": edge_checks,
    }


def compare_astar_vs_ucs_and_greedy(
    graph: Mapping[str, Sequence[Any]],
    heuristics: Mapping[str, Union[int, float]],
    start: str,
    goal: str,
) -> Dict[str, Any]:
    """Compare A* Search against Uniform Cost Search (f=g) and Greedy Best-First Search (f=h).

    Demonstrates how A* combines the optimality guarantee of UCS with the targeted
    search guidance of Greedy Best-First Search.

    Args:
        graph: Graph structure.
        heuristics: Heuristic dictionary.
        start: Start node.
        goal: Destination goal node.

    Returns:
        Dictionary summarizing path, cost, and nodes expanded across all three algorithms.
    """
    # 1. A* Search: f(n) = g(n) + h(n)
    astar_res = a_star_search(graph, heuristics, start, goal)

    # 2. Uniform Cost Search: f(n) = g(n) + 0
    zero_heuristics = {k: 0 for k in heuristics}
    ucs_res = a_star_search(graph, zero_heuristics, start, goal)

    # 3. Greedy Best-First Search: f(n) = 0*g(n) + h(n) (approximated by pure h(n) priority queue)
    norm_graph = normalize_graph(graph)
    norm_heuristics = normalize_heuristics(heuristics)

    gbfs_expanded: List[str] = []
    gbfs_frontier = [(norm_heuristics[start], 0, start, [start])]
    gbfs_closed = set()
    gbfs_path = None
    gbfs_counter = 0

    while gbfs_frontier:
        _, _, curr, path = heapq.heappop(gbfs_frontier)
        if curr in gbfs_closed:
            continue
        gbfs_closed.add(curr)
        gbfs_expanded.append(curr)

        if curr == goal:
            gbfs_path = path
            break

        for neighbor, _ in norm_graph.get(curr, []):
            if neighbor not in gbfs_closed:
                gbfs_counter += 1
                heapq.heappush(gbfs_frontier, (norm_heuristics[neighbor], gbfs_counter, neighbor, path + [neighbor]))

    # Compute GBFS path cost
    gbfs_cost = 0.0
    if gbfs_path:
        for i in range(len(gbfs_path) - 1):
            u, v = gbfs_path[i], gbfs_path[i + 1]
            for target, w in norm_graph.get(u, []):
                if target == v:
                    gbfs_cost += w
                    break

    return {
        "start": start,
        "goal": goal,
        "a_star": {
            "evaluation": "f(n) = g(n) + h(n)",
            "path": astar_res.path,
            "total_cost": astar_res.total_cost,
            "nodes_expanded": astar_res.nodes_expanded,
            "expansion_count": len(astar_res.nodes_expanded),
            "is_optimal": True,
        },
        "uniform_cost_search": {
            "evaluation": "f(n) = g(n)",
            "path": ucs_res.path,
            "total_cost": ucs_res.total_cost,
            "nodes_expanded": ucs_res.nodes_expanded,
            "expansion_count": len(ucs_res.nodes_expanded),
            "is_optimal": True,
        },
        "greedy_best_first_search": {
            "evaluation": "f(n) = h(n)",
            "path": gbfs_path,
            "total_cost": int(gbfs_cost) if gbfs_cost.is_integer() else gbfs_cost,
            "nodes_expanded": gbfs_expanded,
            "expansion_count": len(gbfs_expanded),
            "is_optimal": astar_res.total_cost == gbfs_cost,
        },
    }


def visualize_a_star(
    result: AStarResult,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Render the graph, edge weights, heuristics, and optimal path as an image.

    Requires `matplotlib` and `networkx`. If libraries are not installed, skips
    rendering gracefully and returns None.

    Args:
        result: AStarResult instance.
        output_path: File path to save the generated image (PNG).
        title: Optional custom chart title.

    Returns:
        Absolute path to the saved image file, or None if plotting unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        return None

    G = nx.DiGraph()
    for u, edges in result.graph.items():
        G.add_node(u)
        for v, w in edges:
            G.add_edge(u, v, weight=w)

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, seed=42)

    path_edges = []
    if result.path and len(result.path) > 1:
        path_edges = list(zip(result.path[:-1], result.path[1:]))

    # Node coloring
    node_colors = []
    for node in G.nodes():
        if node == result.start_node:
            node_colors.append("#4CAF50")  # Green
        elif node == result.goal_node:
            node_colors.append("#F44336")  # Red
        elif result.path and node in result.path:
            node_colors.append("#FF9800")  # Orange
        elif node in result.nodes_expanded:
            node_colors.append("#64B5F6")  # Light Blue
        else:
            node_colors.append("#CFD8DC")  # Grey

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, edgecolors="black", linewidths=1.5)

    # Labels with heuristic h(n)
    node_labels = {
        n: f"{n}\nh={result.heuristics.get(n, 0)}"
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")

    # Regular edges
    non_path_edges = [e for e in G.edges() if e not in path_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=non_path_edges, edge_color="#B0BEC5",
        arrows=True, arrowstyle="-|>", arrowsize=18, width=1.5,
    )

    # Highlight optimal path edges
    if path_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color="#D32F2F",
            arrows=True, arrowstyle="-|>", arrowsize=22, width=3.0,
        )

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="#37474F")

    chart_title = title or f"A* Search: {result.format_path()}"
    plt.title(chart_title, fontsize=12, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()

    out_file = Path(output_path) if output_path else Path("a_star_graph.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out_file.resolve())


# ============================================================================
# Viva Voce Examination Knowledge Base
# ============================================================================

def get_a_star_viva_reference() -> Dict[str, Any]:
    """Return a comprehensive study guide for university viva voce examinations.

    Returns:
        Structured dictionary of key definitions, formulas, and Q&A entries.
    """
    return {
        "title": "A* Search Algorithm University Viva Voce Reference Guide",
        "syllabus_topic": "Question 28: Informed Search and Heuristics",
        "core_formula": {
            "f_n": "f(n) = g(n) + h(n)",
            "components": {
                "g_n": "Path cost incurred from the start state to current node n.",
                "h_n": "Heuristic estimate of the cheapest path cost from node n to the goal.",
                "f_n": "Estimated total cost of the cheapest solution path through node n.",
            },
        },
        "questions_and_answers": [
            {
                "question": "What are the three components f(n), g(n), and h(n) in A* Search?",
                "answer": (
                    "In A* Search, f(n) = g(n) + h(n). g(n) represents the actual cost incurred from "
                    "the start node to the current node n. h(n) is the heuristic function that estimates "
                    "the remaining cost to travel from node n to the nearest goal state. f(n) is the "
                    "estimated total cost of a complete solution path going through node n. At each step, "
                    "A* expands the node minimizing f(n), balancing past path cost against estimated remaining distance."
                ),
            },
            {
                "question": "What are the exact mathematical conditions required for A* optimality?",
                "answer": (
                    "1. Tree Search: A* is optimal if h(n) is ADMISSIBLE (it never overestimates the true cost: "
                    "0 <= h(n) <= h*(n)). An admissible heuristic guarantees that the goal node on the optimal "
                    "path will always look cheaper or equal to any suboptimal goal candidate.\n"
                    "2. Graph Search: A* is optimal without reopening closed nodes if h(n) is CONSISTENT "
                    "(monotonic), satisfying the triangle inequality: h(u) <= c(u, v) + h(v) for every edge (u, v). "
                    "Consistency ensures that the f-values along any path are monotonically non-decreasing, so "
                    "whenever a node is expanded, its path cost g(n) is guaranteed to be optimal."
                ),
            },
            {
                "question": "How does A* handle visited nodes when a lower g_cost path is discovered?",
                "answer": (
                    "When evaluating an edge (u, v) with tentative cost g(u) + c(u, v) < g(v):\n"
                    "- Dynamic updating: The algorithm updates best_g_costs[v] with the lower cost and updates "
                    "its parent pointer to u.\n"
                    "- Priority Queue: Because binary min-heaps in Python (heapq) do not support O(1) or O(log N) "
                    "decrease_key, a new tuple with the lower (f, g) is pushed to the heap. Superseded entries "
                    "are pruned via lazy deletion upon being popped.\n"
                    "- Reopening Closed Nodes: If v has already been expanded and added to the closed set, and the "
                    "heuristic is merely admissible but NOT consistent, v MUST be removed from the closed set "
                    "(reopened) and re-enqueued to guarantee that optimal downstream costs propagate."
                ),
            },
            {
                "question": "How does A* Search compare to Uniform Cost Search (UCS) and Greedy Best-First Search?",
                "answer": (
                    "- Uniform Cost Search uses f(n) = g(n) (equivalent to A* with h(n) = 0). It is optimal and complete "
                    "but explores blindly in concentric circles, expanding many unnecessary states.\n"
                    "- Greedy Best-First Search uses f(n) = h(n). It explores aggressively towards the goal but ignores "
                    "accumulated edge costs, making it suboptimal and vulnerable to local minima traps.\n"
                    "- A* balances both: f(n) = g(n) + h(n). It combines the goal-directed focus of greedy search with "
                    "the strict optimality guarantees of uniform cost search."
                ),
            },
            {
                "question": "What is the time and space complexity of A* Search?",
                "answer": (
                    "- Time Complexity: In standard graph search with min-heap priority queue, time complexity is "
                    "O(E log V). In the worst case where the heuristic is uninformative, it is O(b^d) where b is the "
                    "effective branching factor and d is the solution depth.\n"
                    "- Space Complexity: O(b^d) or O(V), because all generated nodes must be stored in either the "
                    "frontier (open list) or explored set (closed list) to prevent cycles and maintain optimality. "
                    "Memory exhaustion is the primary practical bottleneck of A*, inspiring memory-bounded variants like "
                    "IDA* and SMA*."
                ),
            },
        ],
    }
