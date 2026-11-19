"""Greedy Best-First Search (GBFS) Implementation on Graphs.

This module implements Greedy Best-First Search (GBFS) for finding a path
from a start node to a goal node in a graph using heuristic estimates, adhering
to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA Chapter 3).

Core AI Concept:
- Greedy Best-First Search is an informed (heuristic) search strategy that evaluates
  nodes using solely the heuristic function:
      f(n) = h(n)
  where h(n) is an estimate of the cost from node n to the nearest goal state.
- Pure Heuristic Search: Unlike Uniform Cost Search (which uses f(n) = g(n)) and
  A* Search (which uses f(n) = g(n) + h(n)), Greedy Best-First Search completely ignores
  the accumulated step costs g(n) incurred so far.
- Expansion Criterion: At each step, the algorithm expands the node that appears to be
  closest to the goal according to the heuristic value h(n), selecting the node minimizing h(n).

Priority Queue & Frontier Mechanics:
- The frontier (open list) is maintained as a min-heap using Python's `heapq` module.
- Each heap entry is structured as a tuple:
      (heuristic_value, tie_breaker_key, current_node, path)
  This guarantees O(log N) insertion (`heappush`) and O(log N) extraction (`heappop`)
  of the minimum-heuristic node.
- Tie-Breaking Strategies:
  When two nodes have identical heuristic values (h(u) == h(v)), Python compares the next
  tuple element. We provide explicit, configurable tie-breaking strategies:
  1. 'fifo' (default): Uses a monotonic insertion sequence counter to preserve FIFO ordering.
  2. 'alphabetical' / 'lexicographical': Compares node names alphabetically.
  3. 'lifo': Uses a decreasing sequence counter to prefer recently added nodes.

Theoretical Properties & University Viva Topics:
1. Optimality:
   - Greedy Best-First Search is NOT optimal. Because it ignores path cost g(n), it can be
     misled into selecting a path that seems immediately promising (small h) but accumulates
     massive actual edge costs, or bypasses a slightly less attractive initial step that leads
     to a much cheaper overall path.
2. Completeness:
   - Graph Search (with explored set): Complete in finite state spaces because each state
     is expanded at most once and cycles cannot cause infinite loops.
   - Tree Search (without cycle checking): Incomplete; can get caught in infinite loops along
     cycles or paths with oscillating heuristic values.
3. Local Minima & Deceptive Heuristics:
   - A local minimum occurs when a search state has a lower heuristic value than all its
     unexpanded successors, but is not the goal. Greedy search eagerly plunges into dead ends
     or local traps before backtracking.
4. Complexity:
   - Time Complexity: O(b^m) in the worst case, where b is branching factor and m is maximum depth.
   - Space Complexity: O(b^m) to maintain all nodes in memory (frontier and explored set).

Optional Challenge:
- Counterexample graph construction where Greedy chooses a severely suboptimal path compared
  to the optimal path found by Uniform Cost Search or A*.
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

# Default Sample Input from Syllabus / Question 26:
# Heuristics: {'A': 5, 'B': 2, 'C': 0}
# Start: 'A', Goal: 'C'
# Path chosen based on minimum h(n): A -> B -> C
DEFAULT_SAMPLE_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": ["C"],
    "C": [],
}

DEFAULT_SAMPLE_HEURISTICS: Dict[str, float] = {
    "A": 5.0,
    "B": 2.0,
    "C": 0.0,
}

# Optional Challenge Benchmark: Suboptimality Counterexample
# Demonstrates that Greedy Best-First Search is NOT optimal.
# Topology:
#   Start 'S' can go to 'A' (h=2, cost=10) or 'B' (h=8, cost=1).
#   'A' leads to 'G' with cost=100 (total path S->A->G cost = 110).
#   'B' leads to 'G' with cost=1 (total path S->B->G cost = 2).
# Greedy looks only at h(n): sees h(A)=2 < h(B)=8, so it chooses S -> A -> G (Cost: 110).
# Optimal path (found by UCS/A*): S -> B -> G (Cost: 2).
# Suboptimality ratio: 110 / 2 = 55.0x worse!
SUBOPTIMAL_CHALLENGE_GRAPH: Dict[str, List[Tuple[str, float]]] = {
    "S": [("A", 10.0), ("B", 1.0)],
    "A": [("G", 100.0)],
    "B": [("G", 1.0)],
    "G": [],
}

SUBOPTIMAL_CHALLENGE_HEURISTICS: Dict[str, float] = {
    "S": 10.0,
    "A": 2.0,
    "B": 8.0,
    "G": 0.0,
}

# Local Minimum Trap Benchmark:
# Demonstrates Greedy getting tricked by a deceptive local minimum dead-end.
# Start 'S' (h=10) has branches:
# - Branch 1 (Deceptive Trap): 'Trap_1' (h=1.0) -> 'Trap_2' (h=0.5) -> 'Dead_End' (h=0.1)
# - Branch 2 (True Path): 'Path_1' (h=7.0) -> 'Path_2' (h=4.0) -> 'Goal' (h=0.0)
# Greedy eagerly explores the entire Trap branch first before backtracking to Path_1.
LOCAL_MINIMUM_GRAPH: Dict[str, List[str]] = {
    "S": ["Trap_1", "Path_1"],
    "Trap_1": ["Trap_2"],
    "Trap_2": ["Dead_End"],
    "Dead_End": [],
    "Path_1": ["Path_2"],
    "Path_2": ["Goal"],
    "Goal": [],
}

LOCAL_MINIMUM_HEURISTICS: Dict[str, float] = {
    "S": 10.0,
    "Trap_1": 1.0,
    "Trap_2": 0.5,
    "Dead_End": 0.1,
    "Path_1": 7.0,
    "Path_2": 4.0,
    "Goal": 0.0,
}

# Cyclic Graph Benchmark to test graph search cycle avoidance
CYCLIC_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C", "G"],
    "G": [],
}

CYCLIC_HEURISTICS: Dict[str, float] = {
    "A": 6.0,
    "B": 4.0,
    "C": 4.0,
    "D": 2.0,
    "G": 0.0,
}

# Disconnected Graph Benchmark to test unreachable goal handling
DISCONNECTED_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": [],
    "C": ["Goal"],
    "Goal": [],
}

DISCONNECTED_HEURISTICS: Dict[str, float] = {
    "A": 5.0,
    "B": 3.0,
    "C": 1.0,
    "Goal": 0.0,
}

# Classic Russell & Norvig Romania Map Subgraph (Arad to Bucharest)
# Straight-Line Distance to Bucharest (h_SLD)
ROMANIA_MAP_GRAPH: Dict[str, List[Tuple[str, float]]] = {
    "Arad": [("Zerind", 75.0), ("Sibiu", 140.0), ("Timisoara", 118.0)],
    "Zerind": [("Arad", 75.0), ("Oradea", 71.0)],
    "Oradea": [("Zerind", 71.0), ("Sibiu", 151.0)],
    "Timisoara": [("Arad", 118.0), ("Lugoj", 111.0)],
    "Lugoj": [("Timisoara", 111.0), ("Mehadia", 70.0)],
    "Mehadia": [("Lugoj", 70.0), ("Drobeta", 75.0)],
    "Drobeta": [("Mehadia", 75.0), ("Craiova", 120.0)],
    "Craiova": [("Drobeta", 120.0), ("Rimnicu Vilcea", 146.0), ("Pitesti", 138.0)],
    "Sibiu": [("Arad", 140.0), ("Oradea", 151.0), ("Fagaras", 99.0), ("Rimnicu Vilcea", 80.0)],
    "Fagaras": [("Sibiu", 99.0), ("Bucharest", 211.0)],
    "Rimnicu Vilcea": [("Sibiu", 80.0), ("Craiova", 146.0), ("Pitesti", 97.0)],
    "Pitesti": [("Rimnicu Vilcea", 97.0), ("Craiova", 138.0), ("Bucharest", 101.0)],
    "Bucharest": [("Fagaras", 211.0), ("Pitesti", 101.0), ("Giurgiu", 90.0)],
    "Giurgiu": [("Bucharest", 90.0)],
}

ROMANIA_SLD_HEURISTICS: Dict[str, float] = {
    "Arad": 366.0,
    "Bucharest": 0.0,
    "Craiova": 160.0,
    "Drobeta": 242.0,
    "Eforie": 161.0,
    "Fagaras": 176.0,
    "Giurgiu": 77.0,
    "Hirsova": 151.0,
    "Iasi": 226.0,
    "Lugoj": 244.0,
    "Mehadia": 241.0,
    "Neamt": 234.0,
    "Oradea": 380.0,
    "Pitesti": 100.0,
    "Rimnicu Vilcea": 193.0,
    "Sibiu": 253.0,
    "Timisoara": 329.0,
    "Urziceni": 80.0,
    "Vaslui": 199.0,
    "Zerind": 374.0,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class GBFSStepRecord:
    """Immutable record capturing a single step in Greedy Best-First Search.

    Attributes:
        step_number: 1-based chronological expansion step index.
        current_node: Node identifier extracted from the priority queue.
        heuristic_val: The h(n) value of the current node.
        action: Semantic action label ('EXPAND', 'GOAL_REACHED', 'SKIP_EXPLORED').
        frontier_snapshot: Snapshot of (h, node) pairs remaining in the frontier.
        explored_nodes: Snapshot of nodes closed/explored so far.
        neighbors_evaluated: List of (neighbor, h_val) evaluated from current_node.
        neighbors_enqueued: List of (neighbor, h_val) pushed to the frontier.
        path_so_far: Sequence of nodes traversed to reach current_node.
        parent: Predecessor node from which current_node was reached.
    """

    step_number: int
    current_node: str
    heuristic_val: float
    action: str
    frontier_snapshot: List[Tuple[float, str]]
    explored_nodes: List[str]
    neighbors_evaluated: List[Tuple[str, float]]
    neighbors_enqueued: List[Tuple[str, float]]
    path_so_far: List[str]
    parent: Optional[str] = None

    def format_row(self) -> str:
        """Format the step into a structured table row for terminal presentation."""
        frontier_items = [f"{node}(h={h:g})" for h, node in self.frontier_snapshot]
        frontier_str = "[" + ", ".join(frontier_items) + "]" if frontier_items else "[]"
        enqueued_items = [f"{node}(h={h:g})" for node, h in self.neighbors_enqueued]
        enqueued_str = "[" + ", ".join(enqueued_items) + "]" if enqueued_items else "-"
        parent_str = self.parent if self.parent is not None else "START"
        h_str = f"{self.heuristic_val:g}"
        return (
            f"{self.step_number:<5} | {self.current_node:<10} | {h_str:<6} | "
            f"{self.action:<14} | {parent_str:<8} | {enqueued_str:<22} | {frontier_str}"
        )


@dataclass
class GBFSResult:
    """Encapsulates the complete outcome of a Greedy Best-First Search run.

    Attributes:
        path: Ordered list of nodes forming the path from start to goal (None if unreachable).
        path_cost: Actual cumulative path cost if edge weights exist, else hop count.
        start_node: Origin node for the search.
        goal_node: Target destination node.
        goal_found: Boolean flag indicating if goal was reached.
        nodes_expanded: Chronological sequence of nodes popped and expanded.
        steps: Chronological list of GBFSStepRecord objects detailing search progression.
        heuristics: The heuristic dictionary used.
        graph: The graph representation used.
        frontier_peak: Maximum number of entries stored concurrently in the priority queue.
        execution_time_ms: Wall-clock execution time in milliseconds.
        warning_message: Optional diagnostic notice.
    """

    path: Optional[List[str]]
    path_cost: Optional[float]
    start_node: str
    goal_node: str
    goal_found: bool
    nodes_expanded: List[str]
    steps: List[GBFSStepRecord]
    heuristics: Dict[str, float]
    graph: Dict[str, List[Any]]
    frontier_peak: int = 0
    execution_time_ms: float = 0.0
    warning_message: Optional[str] = None

    def format_path(self, delimiter: str = " -> ") -> str:
        """Format the path matching the university practical syllabus standard output.

        Example:
            'Path chosen based on minimum h(n): A -> B -> C'
        """
        if self.path is None or not self.goal_found:
            return "Path chosen based on minimum h(n): None (Goal unreachable)"
        path_str = delimiter.join(self.path)
        return f"Path chosen based on minimum h(n): {path_str}"

    def format_summary(self) -> str:
        """Produce a comprehensive multi-line text summary of the search result."""
        lines = [
            "=" * 70,
            "Greedy Best-First Search (GBFS) Execution Summary",
            "=" * 70,
            f"Start Node      : {self.start_node}",
            f"Goal Node       : {self.goal_node}",
            f"Goal Found      : {self.goal_found}",
            f"Path Found      : {self.format_path()}",
            f"Path Cost       : {self.path_cost if self.path_cost is not None else 'N/A'}",
            f"Nodes Expanded  : {len(self.nodes_expanded)} ({' -> '.join(self.nodes_expanded) if self.nodes_expanded else 'None'})",
            f"Frontier Peak   : {self.frontier_peak} nodes",
            f"Execution Time  : {self.execution_time_ms:.4f} ms",
        ]
        if self.warning_message:
            lines.append(f"Warning         : {self.warning_message}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def format_table(self) -> str:
        """Produce an ASCII table detailing every step of the search."""
        header = (
            f"{'Step':<5} | {'Node':<10} | {'h(n)':<6} | {'Action':<14} | "
            f"{'Parent':<8} | {'Enqueued':<22} | {'Frontier (Min-Heap)'}"
        )
        separator = "-" * len(header)
        rows = [header, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)


@dataclass
class SuboptimalityComparisonResult:
    """Encapsulates the comparative analysis between GBFS and Optimal Search (UCS).

    Attributes:
        greedy_result: GBFSResult from Greedy Best-First Search.
        optimal_path: Optimal path found by Uniform Cost Search.
        optimal_cost: Minimum path cost found by Uniform Cost Search.
        is_suboptimal: True if Greedy path cost > Optimal path cost.
        cost_difference: greedy_cost - optimal_cost.
        suboptimality_ratio: greedy_cost / optimal_cost (if optimal_cost > 0).
        greedy_expansions: Number of nodes expanded by GBFS.
        optimal_expansions: Number of nodes expanded by UCS.
        analysis: Educational breakdown explaining why Greedy failed or succeeded.
    """

    greedy_result: GBFSResult
    optimal_path: Optional[List[str]]
    optimal_cost: Optional[float]
    is_suboptimal: bool
    cost_difference: float
    suboptimality_ratio: float
    greedy_expansions: int
    optimal_expansions: int
    analysis: str

    def format_report(self) -> str:
        """Format an educational comparison report for viva voce and lab submissions."""
        g_path_str = " -> ".join(self.greedy_result.path) if self.greedy_result.path else "None"
        o_path_str = " -> ".join(self.optimal_path) if self.optimal_path else "None"
        g_cost_str = f"{self.greedy_result.path_cost:g}" if self.greedy_result.path_cost is not None else "N/A"
        o_cost_str = f"{self.optimal_cost:g}" if self.optimal_cost is not None else "N/A"

        lines = [
            "=" * 75,
            "Greedy Best-First Search vs. Optimal Search (UCS/Dijkstra) Comparison",
            "=" * 75,
            f"Algorithm              | {'Greedy Best-First Search':<26} | {'Uniform Cost Search (Optimal)':<26}",
            "-" * 75,
            f"Evaluation Function    | f(n) = h(n)                | f(n) = g(n)",
            f"Path Found             | {g_path_str:<26} | {o_path_str:<26}",
            f"Path Cost              | {g_cost_str:<26} | {o_cost_str:<26}",
            f"Nodes Expanded         | {self.greedy_expansions:<26} | {self.optimal_expansions:<26}",
            "-" * 75,
            f"Is Greedy Suboptimal?  : {'YES (Optimal Failure Demonstrated)' if self.is_suboptimal else 'NO (Greedy matched optimal cost)'}",
            f"Excess Cost Incurred   : {self.cost_difference:g}",
            f"Suboptimality Ratio    : {self.suboptimality_ratio:.2f}x",
            "=" * 75,
            "Algorithmic Analysis & Why Greedy Failed:",
            self.analysis,
            "=" * 75,
        ]
        return "\n".join(lines)


# ============================================================================
# Normalization and Parsing Utilities
# ============================================================================

def normalize_graph(
    raw_graph: Any,
) -> Dict[str, List[Union[str, Tuple[str, float]]]]:
    """Normalize various graph representations into a standard adjacency list.

    Supports:
    - Dict[str, List[str]]: Unweighted adjacency list.
    - Dict[str, List[Tuple[str, Union[int, float]]]]: Weighted adjacency list.
    - Dict[str, Dict[str, Union[int, float]]]: Mapping of node to neighbor cost dict.

    Args:
        raw_graph: The raw graph structure.

    Returns:
        Standardized dictionary mapping string node names to neighbor lists.

    Raises:
        TypeError: If graph format is unsupported.
    """
    if not isinstance(raw_graph, dict):
        raise TypeError(f"Graph must be a dictionary, got {type(raw_graph).__name__}")

    normalized: Dict[str, List[Union[str, Tuple[str, float]]]] = {}

    for node, neighbors in raw_graph.items():
        node_key = str(node).strip()
        if isinstance(neighbors, dict):
            # Form: {"A": {"B": 1.0, "C": 2.0}}
            normalized[node_key] = [
                (str(neigh).strip(), float(cost)) for neigh, cost in neighbors.items()
            ]
        elif isinstance(neighbors, (list, tuple)):
            norm_list: List[Union[str, Tuple[str, float]]] = []
            for item in neighbors:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    # Form: ("B", cost)
                    neigh_name = str(item[0]).strip()
                    cost = float(item[1])
                    norm_list.append((neigh_name, cost))
                else:
                    # Unweighted neighbor name
                    norm_list.append(str(item).strip())
            normalized[node_key] = norm_list
        else:
            raise TypeError(
                f"Invalid neighbors for node '{node_key}': expected list, tuple, or dict, got {type(neighbors).__name__}"
            )

    return normalized


def normalize_heuristics(raw_heuristics: Any) -> Dict[str, float]:
    """Normalize heuristic dictionary into Dict[str, float].

    Args:
        raw_heuristics: Mapping of node identifiers to heuristic estimates.

    Returns:
        Dictionary mapping string node names to float heuristic values.

    Raises:
        TypeError: If heuristics is not a dictionary.
        ValueError: If any heuristic value cannot be cast to a float.
    """
    if not isinstance(raw_heuristics, dict):
        raise TypeError(
            f"Heuristics must be a dictionary, got {type(raw_heuristics).__name__}"
        )

    normalized: Dict[str, float] = {}
    for node, h_val in raw_heuristics.items():
        node_key = str(node).strip()
        try:
            val = float(h_val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Heuristic value for node '{node_key}' must be numeric, got {h_val!r}"
            ) from exc
        normalized[node_key] = val

    return normalized


def parse_graph(text: str) -> Dict[str, List[Union[str, Tuple[str, float]]]]:
    """Parse graph from a string (JSON or arrow/colon notation).

    Examples of supported formats:
    - JSON: '{"A": ["B", "C"], "B": ["C"], "C": []}'
    - Arrow notation: 'A -> B, B -> C' or 'A -> B:1, B -> C:2'
    - Adjacency list: 'A: B, C; B: C; C:'

    Args:
        text: String representation of the graph.

    Returns:
        Normalized graph dictionary.
    """
    text = text.strip()
    if not text:
        return {}

    # Attempt JSON parsing first
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            data = json.loads(text)
            return normalize_graph(data)
        except json.JSONDecodeError:
            pass

    graph: Dict[str, List[Union[str, Tuple[str, float]]]] = {}

    # Split by semicolon or newline for multi-line or statement-based inputs
    statements = [s.strip() for s in re.split(r"[;\n]", text) if s.strip()]

    # Case 1: Arrow notation 'A -> B, B -> C' or 'A -> B:5'
    if any("->" in s for s in statements):
        # Flatten comma-separated edge clauses if present
        all_edges: List[str] = []
        for stmt in statements:
            for part in stmt.split(","):
                part_clean = part.strip()
                if part_clean:
                    all_edges.append(part_clean)

        for edge in all_edges:
            if "->" in edge:
                src_part, dst_part = edge.split("->", 1)
                src = src_part.strip()
                dst_part = dst_part.strip()
                graph.setdefault(src, [])
                if ":" in dst_part:
                    dst_raw, cost_str = dst_part.split(":", 1)
                    dst_clean = dst_raw.strip()
                    cost = float(cost_str.strip())
                    graph.setdefault(src, []).append((dst_clean, cost))
                    graph.setdefault(dst_clean, [])
                else:
                    graph.setdefault(src, []).append(dst_part)
                    graph.setdefault(dst_part, [])
        return graph

    # Case 2: Colon notation 'A: B, C'
    for stmt in statements:
        if ":" in stmt:
            node_part, neighbors_part = stmt.split(":", 1)
            node = node_part.strip()
            neighbors_list: List[Union[str, Tuple[str, float]]] = []
            for n_item in neighbors_part.split(","):
                n_clean = n_item.strip()
                if not n_clean:
                    continue
                if "(" in n_clean and ")" in n_clean:
                    # e.g. "B(10)"
                    match = re.match(r"([^\(]+)\(([\d\.]+)\)", n_clean)
                    if match:
                        n_name = match.group(1).strip()
                        n_cost = float(match.group(2))
                        neighbors_list.append((n_name, n_cost))
                    else:
                        neighbors_list.append(n_clean)
                else:
                    neighbors_list.append(n_clean)
            graph[node] = neighbors_list
        else:
            # Standalone node
            graph.setdefault(stmt.strip(), [])

    return graph


def parse_heuristics(text: str) -> Dict[str, float]:
    """Parse heuristics from a string (JSON, Python dict literal, or comma-separated pairs).

    Examples:
    - JSON: '{"A": 5, "B": 2, "C": 0}'
    - Pairs: 'A: 5, B: 2, C: 0' or 'A=5, B=2, C=0'

    Args:
        text: String representation of heuristics.

    Returns:
        Normalized dictionary of heuristics.
    """
    text = text.strip()
    if not text:
        return {}

    # Attempt JSON parsing first
    clean_json = text.replace("'", '"')
    if clean_json.startswith("{") and clean_json.endswith("}"):
        try:
            data = json.loads(clean_json)
            return normalize_heuristics(data)
        except json.JSONDecodeError:
            pass

    heuristics: Dict[str, float] = {}
    # Remove outer braces if user typed '{A: 5, B: 2}'
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()

    # Split by comma or semicolon
    items = re.split(r"[,;\n]", text)
    for item in items:
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            k, v = item.split(":", 1)
            heuristics[k.strip().strip("'\"")] = float(v.strip())
        elif "=" in item:
            k, v = item.split("=", 1)
            heuristics[k.strip().strip("'\"")] = float(v.strip())
        else:
            # e.g. "A 5"
            parts = item.split()
            if len(parts) == 2:
                heuristics[parts[0].strip().strip("'\"")] = float(parts[1].strip())
            else:
                raise ValueError(f"Could not parse heuristic entry: '{item}'")

    return heuristics


def compute_path_cost(
    path: Sequence[str],
    graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
) -> Optional[float]:
    """Calculate the cumulative cost of a path in the graph.

    If the graph contains weighted edges, the actual sum of edge weights is returned.
    If the graph is unweighted, the number of steps (hops = len(path) - 1) is returned.

    Args:
        path: Ordered list of nodes forming the path.
        graph: The graph representation.

    Returns:
        Total path cost as float, or None if path is empty.
    """
    if not path or len(path) <= 1:
        return 0.0

    total_cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        neighbors = graph.get(u, [])
        found_edge = False
        edge_weight: float = 1.0

        for item in neighbors:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0].strip() == v:
                    edge_weight = float(item[1])
                    found_edge = True
                    break
            elif item.strip() == v:
                edge_weight = 1.0
                found_edge = True
                break

        if not found_edge:
            # Discontinuous edge in path
            return None
        total_cost += edge_weight

    return total_cost


# ============================================================================
# Core Algorithm: Greedy Best-First Search
# ============================================================================

def greedy_best_first_search(
    graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
    heuristics: Mapping[str, float],
    start: str,
    goal: str,
    tie_breaker: str = "fifo",
    allow_revisit: bool = False,
    record_steps: bool = True,
    max_expansions: int = 10000,
) -> GBFSResult:
    """Execute Greedy Best-First Search on a graph using heuristic estimates.

    The algorithm maintains a min-heap priority queue ordered strictly by h(n),
    ignoring step costs g(n). It selects and expands nodes minimizing h(n).

    Args:
        graph: Adjacency list mapping node names to neighbor lists.
        heuristics: Mapping of node names to numerical heuristic estimates h(n).
        start: Starting node identifier.
        goal: Target goal node identifier.
        tie_breaker: Strategy for breaking ties when h(u) == h(v).
                     Options: 'fifo' (default), 'alphabetical', 'lifo'.
        allow_revisit: If False (default), standard graph search with explored set.
                       If True, pure tree search (allows cycles, used for demonstrations).
        record_steps: Whether to record detailed step-by-step trace.
        max_expansions: Safety threshold to prevent infinite loops in infinite graphs.

    Returns:
        GBFSResult containing the path, metrics, and step-by-step history.

    Raises:
        ValueError: If start or goal is missing from heuristics or graph.
        KeyError: If an evaluated neighbor lacks a heuristic estimate.
    """
    start_time = time.perf_counter()

    norm_graph = normalize_graph(graph)
    norm_heuristics = normalize_heuristics(heuristics)
    start_node = start.strip()
    goal_node = goal.strip()

    # Pre-flight validation
    if start_node not in norm_heuristics:
        raise KeyError(
            f"Start node '{start_node}' has no heuristic value in heuristics dictionary!"
        )
    if goal_node not in norm_heuristics:
        raise KeyError(
            f"Goal node '{goal_node}' has no heuristic value in heuristics dictionary!"
        )
    if start_node not in norm_graph:
        # Implicitly add start if missing with no neighbors
        norm_graph[start_node] = []

    # Handle trivial base case: start is already goal
    if start_node == goal_node:
        h_start = norm_heuristics[start_node]
        steps = []
        if record_steps:
            steps.append(
                GBFSStepRecord(
                    step_number=1,
                    current_node=start_node,
                    heuristic_val=h_start,
                    action="GOAL_REACHED",
                    frontier_snapshot=[],
                    explored_nodes=[start_node],
                    neighbors_evaluated=[],
                    neighbors_enqueued=[],
                    path_so_far=[start_node],
                    parent=None,
                )
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return GBFSResult(
            path=[start_node],
            path_cost=0.0,
            start_node=start_node,
            goal_node=goal_node,
            goal_found=True,
            nodes_expanded=[start_node],
            steps=steps,
            heuristics=norm_heuristics,
            graph=norm_graph,
            frontier_peak=1,
            execution_time_ms=elapsed_ms,
        )

    # Priority queue min-heap entries:
    # (heuristic_val, tie_breaker_key, current_node, path)
    frontier: List[Tuple[float, Any, str, List[str]]] = []
    counter = 0

    def get_tie_key(node_id: str, count: int) -> Any:
        if tie_breaker == "alphabetical" or tie_breaker == "lexicographical":
            return node_id
        if tie_breaker == "lifo":
            return -count
        # Default: FIFO
        return count

    # Push start node to frontier
    h_start = norm_heuristics[start_node]
    heapq.heappush(frontier, (h_start, get_tie_key(start_node, counter), start_node, [start_node]))

    explored: Set[str] = set()
    nodes_expanded: List[str] = []
    steps: List[GBFSStepRecord] = []
    step_num = 0
    frontier_peak = 1
    goal_found = False
    solution_path: Optional[List[str]] = None
    warning_msg: Optional[str] = None

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))

        if len(nodes_expanded) >= max_expansions:
            warning_msg = f"Expansion limit reached ({max_expansions} expansions). Possible infinite loop."
            break

        # Pop node with minimum h(n)
        h_val, _, current, path = heapq.heappop(frontier)

        # Graph search explored check
        if not allow_revisit and current in explored:
            if record_steps:
                step_num += 1
                frontier_snapshot = [(h, n) for h, _, n, _ in frontier]
                steps.append(
                    GBFSStepRecord(
                        step_number=step_num,
                        current_node=current,
                        heuristic_val=h_val,
                        action="SKIP_EXPLORED",
                        frontier_snapshot=sorted(frontier_snapshot),
                        explored_nodes=sorted(list(explored)),
                        neighbors_evaluated=[],
                        neighbors_enqueued=[],
                        path_so_far=path,
                        parent=path[-2] if len(path) >= 2 else None,
                    )
                )
            continue

        # Goal test when node is selected for expansion (standard Russell & Norvig best-first search)
        if current == goal_node:
            goal_found = True
            solution_path = path
            nodes_expanded.append(current)
            if not allow_revisit:
                explored.add(current)

            if record_steps:
                step_num += 1
                frontier_snapshot = [(h, n) for h, _, n, _ in frontier]
                steps.append(
                    GBFSStepRecord(
                        step_number=step_num,
                        current_node=current,
                        heuristic_val=h_val,
                        action="GOAL_REACHED",
                        frontier_snapshot=sorted(frontier_snapshot),
                        explored_nodes=sorted(list(explored)),
                        neighbors_evaluated=[],
                        neighbors_enqueued=[],
                        path_so_far=path,
                        parent=path[-2] if len(path) >= 2 else None,
                    )
                )
            break

        # Mark explored and expanded
        if not allow_revisit:
            explored.add(current)
        nodes_expanded.append(current)

        # Evaluate successors from graph
        raw_neighbors = norm_graph.get(current, [])
        neighbors_evaluated: List[Tuple[str, float]] = []
        neighbors_enqueued: List[Tuple[str, float]] = []

        for item in raw_neighbors:
            neigh_name = item[0].strip() if isinstance(item, (list, tuple)) else item.strip()

            # Verify neighbor has a valid heuristic
            if neigh_name not in norm_heuristics:
                raise KeyError(
                    f"Neighbor node '{neigh_name}' (from '{current}') has no heuristic value in heuristics dictionary!"
                )

            h_neigh = norm_heuristics[neigh_name]
            neighbors_evaluated.append((neigh_name, h_neigh))

            # In graph search, skip already explored nodes
            if not allow_revisit and neigh_name in explored:
                continue

            counter += 1
            tie_key = get_tie_key(neigh_name, counter)
            heapq.heappush(frontier, (h_neigh, tie_key, neigh_name, path + [neigh_name]))
            neighbors_enqueued.append((neigh_name, h_neigh))

        if record_steps:
            step_num += 1
            frontier_snapshot = [(h, n) for h, _, n, _ in frontier]
            steps.append(
                GBFSStepRecord(
                    step_number=step_num,
                    current_node=current,
                    heuristic_val=h_val,
                    action="EXPAND",
                    frontier_snapshot=sorted(frontier_snapshot),
                    explored_nodes=sorted(list(explored)),
                    neighbors_evaluated=neighbors_evaluated,
                    neighbors_enqueued=neighbors_enqueued,
                    path_so_far=path,
                    parent=path[-2] if len(path) >= 2 else None,
                )
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    actual_path_cost = compute_path_cost(solution_path, norm_graph) if solution_path else None

    return GBFSResult(
        path=solution_path,
        path_cost=actual_path_cost,
        start_node=start_node,
        goal_node=goal_node,
        goal_found=goal_found,
        nodes_expanded=nodes_expanded,
        steps=steps,
        heuristics=norm_heuristics,
        graph=norm_graph,
        frontier_peak=frontier_peak,
        execution_time_ms=elapsed_ms,
        warning_message=warning_msg,
    )


# ============================================================================
# Optimal Baseline: Uniform Cost Search (Dijkstra) for Comparison
# ============================================================================

def uniform_cost_search_baseline(
    graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], Optional[float], int]:
    """Execute Uniform Cost Search (Dijkstra) to obtain the true optimal path and cost.

    Args:
        graph: Adjacency list representation.
        start: Start node.
        goal: Target goal node.

    Returns:
        Tuple of (optimal_path, optimal_cost, nodes_expanded_count).
    """
    norm_graph = normalize_graph(graph)
    start_node = start.strip()
    goal_node = goal.strip()

    if start_node == goal_node:
        return [start_node], 0.0, 1

    # Min-heap entries: (cumulative_g, counter, current_node, path)
    frontier: List[Tuple[float, int, str, List[str]]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, start_node, [start_node]))

    best_cost: Dict[str, float] = {start_node: 0.0}
    nodes_expanded = 0

    while frontier:
        g, _, current, path = heapq.heappop(frontier)

        if g > best_cost.get(current, float("inf")):
            continue

        nodes_expanded += 1

        if current == goal_node:
            return path, g, nodes_expanded

        for item in norm_graph.get(current, []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                neigh = item[0].strip()
                edge_cost = float(item[1])
            else:
                neigh = item.strip()
                edge_cost = 1.0

            new_g = g + edge_cost
            if new_g < best_cost.get(neigh, float("inf")):
                best_cost[neigh] = new_g
                counter += 1
                heapq.heappush(frontier, (new_g, counter, neigh, path + [neigh]))

    return None, None, nodes_expanded


# ============================================================================
# Optional Challenge: Demonstrating Suboptimality & Comparative Analysis
# ============================================================================

def compare_greedy_vs_optimal(
    graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
    heuristics: Mapping[str, float],
    start: str,
    goal: str,
) -> SuboptimalityComparisonResult:
    """Compare Greedy Best-First Search with Uniform Cost Search on the same graph.

    Constructs a thorough empirical and educational analysis highlighting whether
    Greedy found an optimal path or fell into a suboptimal heuristic trap.

    Args:
        graph: Adjacency list representation.
        heuristics: Heuristics dictionary.
        start: Start node.
        goal: Goal node.

    Returns:
        SuboptimalityComparisonResult with quantitative metrics and qualitative analysis.
    """
    gbfs_res = greedy_best_first_search(graph, heuristics, start, goal)
    opt_path, opt_cost, ucs_expansions = uniform_cost_search_baseline(graph, start, goal)

    greedy_cost = gbfs_res.path_cost if gbfs_res.path_cost is not None else float("inf")
    optimal_cost = opt_cost if opt_cost is not None else float("inf")

    is_suboptimal = False
    cost_diff = 0.0
    subopt_ratio = 1.0

    if gbfs_res.goal_found and opt_path is not None:
        cost_diff = greedy_cost - optimal_cost
        is_suboptimal = cost_diff > 1e-6
        if optimal_cost > 0:
            subopt_ratio = greedy_cost / optimal_cost

    # Construct educational analysis text
    if is_suboptimal:
        analysis = (
            f"SUBOPTIMALITY CONFIRMED:\n"
            f"- Greedy Best-First Search chose path '{' -> '.join(gbfs_res.path or [])}' "
            f"with total cost {greedy_cost:g}.\n"
            f"- The true optimal path is '{' -> '.join(opt_path or [])}' with total cost {optimal_cost:g}.\n"
            f"- Greedy was misled because its evaluation function f(n) = h(n) completely ignores "
            f"accumulated path costs g(n). It eagerly selected immediate successors with alluringly small "
            f"h(n) values, completely blinded to the massive edge costs required to traverse those links."
        )
    else:
        analysis = (
            f"OPTIMALITY ACHIEVED (ON THIS INSTANCE):\n"
            f"- Greedy Best-First Search matched the optimal path cost ({greedy_cost:g}).\n"
            f"- Note: Even when Greedy happens to find the optimal path on specific graph topologies, "
            f"it offers NO theoretical guarantee of optimality in the general case."
        )

    return SuboptimalityComparisonResult(
        greedy_result=gbfs_res,
        optimal_path=opt_path,
        optimal_cost=opt_cost,
        is_suboptimal=is_suboptimal,
        cost_difference=cost_diff,
        suboptimality_ratio=subopt_ratio,
        greedy_expansions=len(gbfs_res.nodes_expanded),
        optimal_expansions=ucs_expansions,
        analysis=analysis,
    )


def demonstrate_local_minimum() -> Dict[str, Any]:
    """Demonstrate the local minimum phenomenon where Greedy is lured into a dead-end.

    Returns:
        Dictionary containing the search trace and educational summary.
    """
    res = greedy_best_first_search(
        LOCAL_MINIMUM_GRAPH,
        LOCAL_MINIMUM_HEURISTICS,
        start="S",
        goal="Goal",
    )
    explanation = (
        "LOCAL MINIMUM EXPLANATION:\n"
        "At node 'S', the algorithm sees two alternatives:\n"
        "  - 'Trap_1' with h=1.0\n"
        "  - 'Path_1' with h=7.0\n"
        "Because Greedy minimizes h(n), it eagerly pursues 'Trap_1' (h=1.0), then 'Trap_2' (h=0.5),\n"
        "and finally 'Dead_End' (h=0.1). None of these lead to the Goal!\n"
        "Only after completely exhausting this deceptive branch does it fall back to 'Path_1',\n"
        "eventually finding 'Goal'. This demonstrates that Greedy search can get heavily delayed\n"
        "or stuck in local minima when heuristic values decrease away from the actual solution."
    )
    return {
        "result": res,
        "nodes_expanded": res.nodes_expanded,
        "path": res.path,
        "explanation": explanation,
    }


# ============================================================================
# Object-Oriented Interface: GreedyBestFirstSearch
# ============================================================================

class GreedyBestFirstSearch:
    """Object-oriented solver for Greedy Best-First Search on a graph.

    Encapsulates graph topology and heuristic values, providing high-level
    query methods for path finding, step-by-step reporting, and suboptimality analysis.
    """

    def __init__(
        self,
        graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
        heuristics: Mapping[str, float],
        tie_breaker: str = "fifo",
    ) -> None:
        """Initialize the solver with graph and heuristic dictionaries.

        Args:
            graph: Adjacency list representation of the graph.
            heuristics: Mapping from node identifiers to heuristic estimates.
            tie_breaker: Tie-breaking policy ('fifo', 'alphabetical', 'lifo').
        """
        self.graph = normalize_graph(graph)
        self.heuristics = normalize_heuristics(heuristics)
        self.tie_breaker = tie_breaker

    def search(self, start: str, goal: str) -> GBFSResult:
        """Find a path from start to goal using Greedy Best-First Search.

        Args:
            start: Start node.
            goal: Goal node.

        Returns:
            GBFSResult instance.
        """
        return greedy_best_first_search(
            self.graph,
            self.heuristics,
            start=start,
            goal=goal,
            tie_breaker=self.tie_breaker,
        )

    def get_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Convenience method returning just the list of traversed nodes."""
        result = self.search(start, goal)
        return result.path

    def compare_with_optimal(self, start: str, goal: str) -> SuboptimalityComparisonResult:
        """Compare Greedy result against optimal Uniform Cost Search."""
        return compare_greedy_vs_optimal(
            self.graph,
            self.heuristics,
            start=start,
            goal=goal,
        )


# ============================================================================
# Graph Visualization (NetworkX + Matplotlib)
# ============================================================================

def visualize_gbfs(
    graph: Mapping[str, Sequence[Union[str, Tuple[str, float]]]],
    heuristics: Mapping[str, float],
    result: Optional[GBFSResult] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Greedy Best-First Search Traversal",
) -> Optional[str]:
    """Generate a visual graph diagram showing nodes, heuristics, and the search path.

    Requires matplotlib and networkx. If not installed, gracefully returns None.

    Args:
        graph: Graph dictionary.
        heuristics: Heuristics dictionary.
        result: Optional GBFSResult whose path will be highlighted in bright green.
        output_path: File path to save the generated image (PNG).
        title: Title for the diagram.

    Returns:
        Absolute file path to the saved image if saved, or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless rendering safe for server/CLI
        import matplotlib.pyplot as plt
        import networkx as nx  # type: ignore[import-untyped]
    except ImportError:
        return None

    norm_g = normalize_graph(graph)
    norm_h = normalize_heuristics(heuristics)

    G = nx.DiGraph()
    edge_labels: Dict[Tuple[str, str], str] = {}

    for u, neighbors in norm_g.items():
        G.add_node(u)
        for item in neighbors:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                v = item[0].strip()
                w = float(item[1])
                G.add_edge(u, v, weight=w)
                edge_labels[(u, v)] = f"{w:g}"
            else:
                v = item.strip()
                G.add_edge(u, v)

    # Compute node colors
    path_edges: Set[Tuple[str, str]] = set()
    path_nodes: Set[str] = set()
    start_node = result.start_node if result else None
    goal_node = result.goal_node if result else None

    if result and result.path:
        path_nodes = set(result.path)
        for u, v in zip(result.path[:-1], result.path[1:]):
            path_edges.add((u, v))

    node_colors: List[str] = []
    labels: Dict[str, str] = {}

    for node in G.nodes():
        h_val = norm_h.get(node, None)
        h_text = f"h={h_val:g}" if h_val is not None else "h=?"
        labels[node] = f"{node}\n({h_text})"

        if node == start_node:
            node_colors.append("#388E3C")  # Dark Green
        elif node == goal_node:
            node_colors.append("#D32F2F")  # Red
        elif node in path_nodes:
            node_colors.append("#81C784")  # Light Green
        elif result and node in result.nodes_expanded:
            node_colors.append("#FFF59D")  # Yellow (Explored)
        else:
            node_colors.append("#E0E0E0")  # Grey (Unvisited)

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, seed=42)

    # Draw standard edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")

    non_path_edges = [e for e in G.edges() if e not in path_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=non_path_edges, edge_color="#9E9E9E", arrowstyle="->", arrowsize=18, width=1.5
    )

    # Highlight path edges
    if path_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=list(path_edges), edge_color="#2E7D32", arrowstyle="->", arrowsize=22, width=3.5
        )

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()

    if output_path is None:
        save_file = Path("gbfs_traversal.png")
    else:
        save_file = Path(output_path)

    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close()

    return str(save_file.resolve())


# ============================================================================
# Viva Voce Examination Reference
# ============================================================================

def get_gbfs_viva_reference() -> Dict[str, Any]:
    """Retrieve comprehensive Viva Voce preparation material for Question 26.

    Returns:
        Structured dictionary covering optimality, completeness, local minima,
        complexity analysis, and comparison matrices for viva examinations.
    """
    return {
        "title": "Question 26: Greedy Best-First Search on a Graph - Viva Voce Guide",
        "core_concept": "Greedy Best-First Search (GBFS) is an informed search strategy guided strictly by h(n).",
        "evaluation_function": "f(n) = h(n) (Step costs g(n) are completely ignored).",
        "key_questions": [
            {
                "question": "Is Greedy Best-First Search optimal?",
                "answer": (
                    "NO. Greedy Best-First Search is strictly NOT optimal. Because it completely ignores the "
                    "accumulated path cost g(n) and relies solely on the heuristic estimate h(n), it can be "
                    "lured into taking steps that appear locally promising (low h(n)) but lead to enormous "
                    "actual edge costs. For example, in a graph where S->A has cost 10 and A->G has cost 100 "
                    "(with h(A)=2), while S->B has cost 1 and B->G has cost 1 (with h(B)=8), Greedy will choose "
                    "S->A->G (total cost 110) instead of the optimal S->B->G (total cost 2)."
                ),
            },
            {
                "question": "Is Greedy Best-First Search complete?",
                "answer": (
                    "It depends on the search formulation:\n"
                    "1. Tree Search: INCOMPLETE. It can enter an infinite loop in state spaces with cycles or "
                    "infinite paths where repeated states have lower heuristic values than alternative branches.\n"
                    "2. Graph Search (with explored set): COMPLETE in finite state spaces, because every finite "
                    "graph has a finite number of states, and the explored set guarantees each state is expanded "
                    "at most once. In infinite state spaces, it remains incomplete."
                ),
            },
            {
                "question": "What is a 'Local Minimum' in Greedy Search, and how does it affect traversal?",
                "answer": (
                    "A local minimum is a state or region whose heuristic value h(n) is lower than all its "
                    "unexpanded successor states, but which does NOT correspond to the goal state. Because Greedy "
                    "always chases the lowest heuristic value, it eagerly plunges into local minima traps or "
                    "dead-end branches. It wastes significant time expanding these deceptive nodes before the "
                    "priority queue finally exhausts them and backtracks to the true path leading to the goal."
                ),
            },
            {
                "question": "What are the Time and Space Complexities of Greedy Best-First Search?",
                "answer": (
                    "Worst-case Time Complexity: O(b^m)\n"
                    "Worst-case Space Complexity: O(b^m)\n"
                    "where b is the maximum branching factor and m is the maximum depth of the search space. "
                    "In the worst case (with a deceptive or uninformative heuristic), Greedy behaves like "
                    "an unguided Depth-First Search wandering down the wrong path, retaining all generated nodes "
                    "in memory in the priority queue and explored set."
                ),
            },
            {
                "question": "How does Greedy Best-First Search compare to Uniform Cost Search and A* Search?",
                "answer": (
                    "Comparison of Evaluation Functions:\n"
                    "- Uniform Cost Search (UCS): f(n) = g(n). Ignores heuristic h(n). Guaranteed optimal and complete.\n"
                    "- Greedy Best-First Search (GBFS): f(n) = h(n). Ignores path cost g(n). Fast heuristic guidance, but not optimal.\n"
                    "- A* Search: f(n) = g(n) + h(n). Combines actual cost g(n) with estimated future cost h(n). "
                    "Guaranteed optimal and complete if h(n) is admissible (tree search) or consistent (graph search)."
                ),
            },
            {
                "question": "Why is heapq used, and how does Python handle tie-breaking?",
                "answer": (
                    "Python's `heapq` module implements a binary min-heap over a standard list with O(log N) "
                    "insertion and deletion of the minimum. When tuples (h, tie_key, node, path) are inserted, "
                    "heapq compares element by element. To avoid invalid comparisons between non-comparable "
                    "objects and to enforce deterministic FIFO ordering when heuristics are identical (h(u) == h(v)), "
                    "a monotonic sequence counter is placed as the second tuple element."
                ),
            },
        ],
        "comparison_matrix": [
            {
                "algorithm": "Uniform Cost Search (UCS)",
                "evaluation_fn": "f(n) = g(n)",
                "optimal": "Yes",
                "complete": "Yes (step cost >= eps > 0)",
                "time_complexity": "O(b^(1 + floor(C*/eps)))",
                "space_complexity": "O(b^(1 + floor(C*/eps)))",
            },
            {
                "algorithm": "Greedy Best-First Search (GBFS)",
                "evaluation_fn": "f(n) = h(n)",
                "optimal": "No",
                "complete": "No (Tree) / Yes (Finite Graph)",
                "time_complexity": "O(b^m)",
                "space_complexity": "O(b^m)",
            },
            {
                "algorithm": "A* Search",
                "evaluation_fn": "f(n) = g(n) + h(n)",
                "optimal": "Yes (with admissible h)",
                "complete": "Yes",
                "time_complexity": "O(b^d) to O(b^m)",
                "space_complexity": "O(b^d) to O(b^m)",
            },
        ],
    }
