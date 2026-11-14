"""Bidirectional Search (BDS) Simulation for State-Space Exploration.

This module implements Bidirectional Search for unweighted graphs and grid mazes,
running two simultaneous Breadth-First Searches (BFS) — one forward from the start
node and one backward from the goal node — halting the moment their search frontiers
intersect, as described in Russell & Norvig's *Artificial Intelligence: A Modern Approach*
(AIMA 3rd/4th ed., Chapter 3: Solving Problems by Searching).

Theoretical Background & AI Concepts:
- Search Space Halving & Complexity Reduction:
    Standard forward BFS explores a search ball of radius d with time and space
    complexity O(b^d), where b is the branching factor and d is the distance to the goal.
    Bidirectional search replaces this single exponential ball with two smaller balls
    of radius d/2, reducing the complexity to:
        Time Complexity:  O(b^(d/2) + b^(d/2)) = O(b^(d/2))
        Space Complexity: O(b^(d/2) + b^(d/2)) = O(b^(d/2))
    For example, with b = 10 and d = 10:
        Standard BFS:      10^10 = 10,000,000,000 nodes (~10 billion nodes)
        Bidirectional BFS: 2 * 10^5 = 200,000 nodes
        Reduction factor:  50,000x faster execution and memory reduction!

- Dual Frontier & Visited Tracking:
    The algorithm maintains:
    1. Forward queue (FIFO) and forward visited dictionary (storing parent pointers and depth).
    2. Backward queue (FIFO) and backward visited dictionary.
    Expansions alternate symmetrically between forward and backward frontiers.

- Intersection Detection & Path Splicing:
    When an expanding node in one direction discovers a neighbor already recorded
    in the opposite visited dictionary, an intersection node is detected.
    The complete path is reconstructed by:
    1. Tracing forward parent pointers from the intersection node back to the start node (reversed).
    2. Tracing backward parent pointers from the intersection node down to the goal node.
    3. Splicing the two halves together at the intersection node.

- Predecessor vs. Successor Generation:
    Forward search explores forward transitions (successors). Backward search in undirected
    graphs uses symmetric adjacency; in directed graphs, backward search requires the inverse
    adjacency relation (predecessors: u -> v inverted to v <- u).

- Optional Challenge:
    Includes 2D Grid Maze Bidirectional Search supporting obstacles, coordinate boundaries,
    and visual path rendering.
"""

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Canonical sample benchmark graphs for Question 19 testing and demonstration
SAMPLE_BDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B"],
    "D": ["B", "E", "F"],
    "E": ["D", "F"],
    "F": ["D", "E", "G"],
    "G": ["F"],
}
"""Canonical sample graph matching Question 19 specification:
Start: 'A', Goal: 'G'
Expected Output: Intersection at D. Path: A-B-D-F-G
"""

LINEAR_CHAIN_BDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": ["A", "C"],
    "C": ["B", "D"],
    "D": ["C", "E"],
    "E": ["D", "F"],
    "F": ["E"],
}
"""Linear chain of 6 nodes to test odd/even path length splicing."""

DISCONNECTED_BDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A"],
    "D": ["B"],
    "X": ["Y"],
    "Y": ["X", "Z"],
    "Z": ["Y"],
}
"""Disconnected graph with two disjoint components to test unreachable search termination."""

EXPONENTIAL_TREE_BDS_GRAPH: Dict[str, List[str]] = {
    "root": ["L1_1", "L1_2", "L1_3"],
    "L1_1": ["root", "L2_1", "L2_2"],
    "L1_2": ["root", "L2_3", "L2_4"],
    "L1_3": ["root", "L2_5", "L2_6"],
    "L2_1": ["L1_1", "mid_1"],
    "L2_2": ["L1_1", "mid_1"],
    "L2_3": ["L1_2", "mid_2"],
    "L2_4": ["L1_2", "mid_2"],
    "L2_5": ["L1_3", "mid_3"],
    "L2_6": ["L1_3", "mid_3"],
    "mid_1": ["L2_1", "L2_2", "R2_1"],
    "mid_2": ["L2_3", "L2_4", "R2_2"],
    "mid_3": ["L2_5", "L2_6", "R2_3"],
    "R2_1": ["mid_1", "R1_1"],
    "R2_2": ["mid_2", "R1_2"],
    "R2_3": ["mid_3", "R1_3"],
    "R1_1": ["R2_1", "goal"],
    "R1_2": ["R2_2", "goal"],
    "R1_3": ["R2_3", "goal"],
    "goal": ["R1_1", "R1_2", "R1_3"],
}
"""High-branching graph to demonstrate O(b^(d/2)) exponential expansion reduction."""

DIRECTED_REVERSIBLE_BDS_GRAPH: Dict[str, List[str]] = {
    "S": ["A", "B"],
    "A": ["C"],
    "B": ["D"],
    "C": ["E"],
    "D": ["E"],
    "E": ["G"],
    "G": [],
}
"""Directed graph used to demonstrate predecessor inversion for backward search."""

# Optional Challenge: 2D Grid Maze Benchmark
DEFAULT_BDS_MAZE: List[List[int]] = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
"""7x7 grid maze where 0 = open path and 1 = wall obstacle."""


@dataclass(frozen=True)
class BidirectionalStepRecord:
    """Telemetry record capturing a single expansion step in bidirectional search.

    Attributes:
        step_number: 1-based sequential step counter.
        direction: Direction of current expansion ('FORWARD' or 'BACKWARD').
        expanded_node: Node label popped from queue and expanded.
        neighbors_evaluated: Adjacent nodes considered during this step.
        newly_visited: Nodes added to the active visited set during this step.
        frontier_size_forward: Number of elements in forward queue after step.
        frontier_size_backward: Number of elements in backward queue after step.
        visited_count_forward: Total nodes marked visited by forward search.
        visited_count_backward: Total nodes marked visited by backward search.
        intersection_found: Whether this step triggered frontier collision.
        intersection_node: Node label where frontiers intersected, if found.
    """

    step_number: int
    direction: str
    expanded_node: str
    neighbors_evaluated: List[str]
    newly_visited: List[str]
    frontier_size_forward: int
    frontier_size_backward: int
    visited_count_forward: int
    visited_count_backward: int
    intersection_found: bool = False
    intersection_node: Optional[str] = None

    def format_row(self) -> str:
        """Format step telemetry as an aligned ASCII table row."""
        neighbors_str = ", ".join(self.neighbors_evaluated) if self.neighbors_evaluated else "-"
        intersect_str = f"YES ({self.intersection_node})" if self.intersection_found else "No"
        return (
            f"{self.step_number:<4} | {self.direction:<8} | {self.expanded_node:<8} | "
            f"{neighbors_str:<18} | F:{self.frontier_size_forward:<3} B:{self.frontier_size_backward:<3} | "
            f"F:{self.visited_count_forward:<3} B:{self.visited_count_backward:<3} | {intersect_str}"
        )


@dataclass
class BidirectionalSearchResult:
    """Encapsulates the complete outcome and execution metrics of Bidirectional Search.

    Attributes:
        status: Search outcome ('Success' or 'Failure').
        start_node: Origin state label.
        goal_node: Target state label.
        intersection_node: Node where forward and backward frontiers intersected.
        path: Ordered list of nodes forming the shortest path from start to goal.
        path_length: Number of edges in the reconstructed path (len(path) - 1).
        forward_path: Path segment from start node to intersection node.
        backward_path: Path segment from intersection node to goal node.
        nodes_expanded_forward: Number of nodes expanded in the forward direction.
        nodes_expanded_backward: Number of nodes expanded in the backward direction.
        total_nodes_expanded: Combined count of expanded nodes across both directions.
        nodes_generated_forward: Total nodes added to forward queue/visited.
        nodes_generated_backward: Total nodes added to backward queue/visited.
        total_nodes_generated: Combined count of generated nodes.
        visited_forward: Set of all nodes visited by the forward search.
        visited_backward: Set of all nodes visited by the backward search.
        steps: Step-by-step telemetry history of alternating expansions.
        execution_time_ms: Wall-clock execution duration in milliseconds.
        graph: Adjacency list of the graph searched.
        alternation_strategy: Strategy used ('turn' or 'balanced').
    """

    status: str
    start_node: str
    goal_node: str
    intersection_node: Optional[str]
    path: Optional[List[str]]
    path_length: int
    forward_path: Optional[List[str]] = None
    backward_path: Optional[List[str]] = None
    nodes_expanded_forward: int = 0
    nodes_expanded_backward: int = 0
    total_nodes_expanded: int = 0
    nodes_generated_forward: int = 0
    nodes_generated_backward: int = 0
    total_nodes_generated: int = 0
    visited_forward: Set[str] = field(default_factory=set)
    visited_backward: Set[str] = field(default_factory=set)
    steps: List[BidirectionalStepRecord] = field(default_factory=list)
    execution_time_ms: float = 0.0
    graph: Dict[str, List[str]] = field(default_factory=dict)
    alternation_strategy: str = "turn"

    def format_sample_output(self) -> str:
        """Format output matching exact Question 19 sample output specification.

        Returns:
            String formatted as 'Intersection at <Node>. Path: <A-B-D-F-G>'
            or 'Failure: No path found between <start> and <goal>.'
        """
        if self.status == "Success" and self.path and self.intersection_node:
            path_str = "-".join(self.path)
            return f"Intersection at {self.intersection_node}. Path: {path_str}"
        return f"Failure: No path found between {self.start_node} and {self.goal_node}."

    def format_summary(self) -> str:
        """Generate a structured, human-readable summary report of search metrics."""
        path_str = "-".join(self.path) if self.path else "None"
        fwd_str = "-".join(self.forward_path) if self.forward_path else "None"
        bwd_str = "-".join(self.backward_path) if self.backward_path else "None"
        lines = [
            "======================================================================",
            "        BIDIRECTIONAL SEARCH (BDS) SIMULATION REPORT",
            "======================================================================",
            f"  • Start State            : {self.start_node}",
            f"  • Goal State             : {self.goal_node}",
            f"  • Search Status          : {self.status.upper()}",
            f"  • Intersection Node      : {self.intersection_node or 'None'}",
            f"  • Combined Shortest Path : {path_str}",
            f"  • Path Length (Edges)    : {self.path_length}",
            f"  • Forward Half-Path      : {fwd_str}",
            f"  • Backward Half-Path     : {bwd_str}",
            "----------------------------------------------------------------------",
            "  SEARCH COMPLEXITY & TELEMETRY:",
            f"  • Forward Expansions     : {self.nodes_expanded_forward}",
            f"  • Backward Expansions    : {self.nodes_expanded_backward}",
            f"  • Total Nodes Expanded   : {self.total_nodes_expanded}",
            f"  • Total Nodes Generated  : {self.total_nodes_generated}",
            f"  • Forward Visited Size   : {len(self.visited_forward)} states",
            f"  • Backward Visited Size  : {len(self.visited_backward)} states",
            f"  • Execution Time         : {self.execution_time_ms:.4f} ms",
            "======================================================================",
        ]
        return "\n".join(lines)

    def format_trace_table(self) -> str:
        """Format an aligned tabular trace of all alternating expansion steps."""
        if not self.steps:
            return "(No expansion steps recorded - start and goal were identical or invalid)"

        header = (
            "Step | Direct.  | Node     | Neighbors Evaluated| Frontier  | Visited   | Intersect?\n"
            "-----+----------+----------+--------------------+-----------+-----------+-----------"
        )
        rows = [step.format_row() for step in self.steps]
        return "\n".join([header] + rows)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "start_node": self.start_node,
            "goal_node": self.goal_node,
            "intersection_node": self.intersection_node,
            "path": self.path,
            "path_length": self.path_length,
            "forward_path": self.forward_path,
            "backward_path": self.backward_path,
            "nodes_expanded_forward": self.nodes_expanded_forward,
            "nodes_expanded_backward": self.nodes_expanded_backward,
            "total_nodes_expanded": self.total_nodes_expanded,
            "nodes_generated_forward": self.nodes_generated_forward,
            "nodes_generated_backward": self.nodes_generated_backward,
            "total_nodes_generated": self.total_nodes_generated,
            "visited_forward": sorted(list(self.visited_forward)),
            "visited_backward": sorted(list(self.visited_backward)),
            "execution_time_ms": self.execution_time_ms,
            "steps": [
                {
                    "step": s.step_number,
                    "direction": s.direction,
                    "node": s.expanded_node,
                    "neighbors": s.neighbors_evaluated,
                    "intersect": s.intersection_found,
                    "intersect_node": s.intersection_node,
                }
                for s in self.steps
            ],
        }


# ============================================================================
# Core Graph Utilities: Normalization, Inversion, Parsing
# ============================================================================


def normalize_graph(raw_graph: Dict[Any, Sequence[Any]]) -> Dict[str, List[str]]:
    """Normalize any graph dictionary to clean string keys and string lists.

    Args:
        raw_graph: Mapping from node keys to iterables of neighboring node keys.

    Returns:
        Normalized dictionary mapping uppercase/stripped string nodes to neighbor lists.
    """
    normalized: Dict[str, List[str]] = {}
    for node, neighbors in raw_graph.items():
        node_str = str(node).strip()
        normalized[node_str] = [str(nbr).strip() for nbr in neighbors]
    return normalized


def invert_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Compute the reverse (predecessor) graph for backward search in directed graphs.

    For each edge u -> v in ``graph``, the inverted graph contains an edge v -> u.

    Args:
        graph: Forward adjacency list mapping u -> [v1, v2, ...].

    Returns:
        Predecessor adjacency list mapping v -> [u1, u2, ...].
    """
    predecessors: Dict[str, List[str]] = {node: [] for node in graph}
    for u, neighbors in graph.items():
        for v in neighbors:
            if v not in predecessors:
                predecessors[v] = []
            if u not in predecessors[v]:
                predecessors[v].append(u)
    return predecessors


def make_undirected(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Convert an arbitrary adjacency dictionary into an undirected symmetric graph.

    Args:
        graph: Adjacency list representation.

    Returns:
        New adjacency list where (u, v) implies (v, u).
    """
    undirected: Dict[str, List[str]] = {node: list(nbrs) for node, nbrs in graph.items()}
    for u, neighbors in graph.items():
        for v in neighbors:
            if v not in undirected:
                undirected[v] = []
            if u not in undirected[v]:
                undirected[v].append(u)
    return undirected


def parse_graph_input(raw_input: Union[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Parse user-provided graph string or dictionary into a normalized adjacency map.

    Supports:
    1. JSON format: '{"A": ["B", "C"], "B": ["D"]}'
    2. Semicolon-delimited format: 'A: B, C; B: D; C:; D:'

    Args:
        raw_input: Graph representation as JSON string, delimited string, or dictionary.

    Returns:
        Normalized adjacency list dictionary.

    Raises:
        ValueError: If graph format cannot be parsed or is invalid.
    """
    if isinstance(raw_input, dict):
        return normalize_graph(raw_input)

    text = raw_input.strip()
    if not text:
        raise ValueError("Graph specification cannot be empty.")

    # Try parsing as JSON first
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return normalize_graph(parsed)
        except json.JSONDecodeError:
            pass

    # Parse semicolon-delimited adjacency list
    graph: Dict[str, List[str]] = {}
    entries = text.split(";")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid adjacency entry '{entry}'. Expected format 'Node: neighbor1, neighbor2'.")
        node_part, neighbors_part = entry.split(":", 1)
        node = node_part.strip()
        neighbors = [nbr.strip() for nbr in neighbors_part.split(",") if nbr.strip()]
        graph[node] = neighbors

    if not graph:
        raise ValueError(f"Could not parse any valid nodes or edges from input: '{raw_input}'")

    return graph


# ============================================================================
# Bidirectional Search Implementation (Graph)
# ============================================================================


class BidirectionalSearch:
    """Object-oriented engine for Bidirectional Search simulation on unweighted graphs.

    Maintains dual FIFO frontiers and visited sets, alternating expansion turns
    between forward and backward explorations until frontiers collide.
    """

    def __init__(
        self,
        graph: Optional[Dict[str, Sequence[str]]] = None,
        is_directed: bool = False,
        alternation_strategy: str = "turn",
    ) -> None:
        """Initialize Bidirectional Search runner.

        Args:
            graph: Optional adjacency list. If None, default sample graph is used.
            is_directed: Whether edges are directed. If True, backward search uses
                inverted predecessor edges. If False (default), symmetric edges are assumed.
            alternation_strategy: 'turn' (alternate 1 node per turn) or 'balanced'
                (expand the queue with fewer pending elements).
        """
        raw = graph if graph is not None else SAMPLE_BDS_GRAPH
        self._raw_graph = normalize_graph(raw)
        self.is_directed = is_directed
        self.alternation_strategy = alternation_strategy

        if self.is_directed:
            self.forward_graph = self._raw_graph
            self.backward_graph = invert_graph(self._raw_graph)
        else:
            self.forward_graph = make_undirected(self._raw_graph)
            self.backward_graph = self.forward_graph

    def search(self, start: str, goal: str) -> BidirectionalSearchResult:
        """Execute simultaneous bidirectional BFS from start to goal.

        Args:
            start: Origin node label.
            goal: Destination node label.

        Returns:
            BidirectionalSearchResult containing intersection node, full spliced path,
            and step-by-step telemetry.

        Raises:
            ValueError: If start or goal node does not exist in graph.
        """
        start = str(start).strip()
        goal = str(goal).strip()

        start_time = time.perf_counter()

        # Validate node existence
        all_nodes = set(self.forward_graph.keys()) | set(self.backward_graph.keys())
        if start not in all_nodes:
            raise ValueError(f"Start node '{start}' does not exist in graph nodes: {sorted(list(all_nodes))}")
        if goal not in all_nodes:
            raise ValueError(f"Goal node '{goal}' does not exist in graph nodes: {sorted(list(all_nodes))}")

        # Base case: start and goal are the same node
        if start == goal:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return BidirectionalSearchResult(
                status="Success",
                start_node=start,
                goal_node=goal,
                intersection_node=start,
                path=[start],
                path_length=0,
                forward_path=[start],
                backward_path=[goal],
                nodes_expanded_forward=0,
                nodes_expanded_backward=0,
                total_nodes_expanded=0,
                nodes_generated_forward=1,
                nodes_generated_backward=1,
                total_nodes_generated=1,
                visited_forward={start},
                visited_backward={goal},
                steps=[],
                execution_time_ms=elapsed,
                graph=self.forward_graph,
                alternation_strategy=self.alternation_strategy,
            )

        # Dual FIFO queues
        queue_forward: deque[str] = deque([start])
        queue_backward: deque[str] = deque([goal])

        # Dual visited records: map node -> parent node
        parents_forward: Dict[str, Optional[str]] = {start: None}
        parents_backward: Dict[str, Optional[str]] = {goal: None}

        steps: List[BidirectionalStepRecord] = []
        step_counter = 0

        nodes_expanded_fwd = 0
        nodes_expanded_bwd = 0
        nodes_generated_fwd = 1
        nodes_generated_bwd = 1

        intersection_node: Optional[str] = None

        # Determine turn order
        current_direction = "FORWARD"

        while queue_forward and queue_backward:
            step_counter += 1

            # Decide which direction to expand
            if self.alternation_strategy == "balanced":
                # Expand direction with smaller frontier
                if len(queue_forward) <= len(queue_backward):
                    active_direction = "FORWARD"
                else:
                    active_direction = "BACKWARD"
            else:
                # Strict turn-based alternation
                active_direction = current_direction
                current_direction = "BACKWARD" if current_direction == "FORWARD" else "FORWARD"

            if active_direction == "FORWARD":
                curr_node = queue_forward.popleft()
                nodes_expanded_fwd += 1
                neighbors = self.forward_graph.get(curr_node, [])
                newly_visited_list: List[str] = []

                for neighbor in neighbors:
                    # Check frontier intersection immediately upon generation (AIMA standard)
                    if neighbor in parents_backward:
                        # Collision detected!
                        parents_forward[neighbor] = curr_node
                        newly_visited_list.append(neighbor)
                        intersection_node = neighbor
                        nodes_generated_fwd += 1

                        steps.append(
                            BidirectionalStepRecord(
                                step_number=step_counter,
                                direction="FORWARD",
                                expanded_node=curr_node,
                                neighbors_evaluated=list(neighbors),
                                newly_visited=newly_visited_list,
                                frontier_size_forward=len(queue_forward),
                                frontier_size_backward=len(queue_backward),
                                visited_count_forward=len(parents_forward),
                                visited_count_backward=len(parents_backward),
                                intersection_found=True,
                                intersection_node=intersection_node,
                            )
                        )
                        break

                    if neighbor not in parents_forward:
                        parents_forward[neighbor] = curr_node
                        queue_forward.append(neighbor)
                        newly_visited_list.append(neighbor)
                        nodes_generated_fwd += 1

                if intersection_node is not None:
                    break

                steps.append(
                    BidirectionalStepRecord(
                        step_number=step_counter,
                        direction="FORWARD",
                        expanded_node=curr_node,
                        neighbors_evaluated=list(neighbors),
                        newly_visited=newly_visited_list,
                        frontier_size_forward=len(queue_forward),
                        frontier_size_backward=len(queue_backward),
                        visited_count_forward=len(parents_forward),
                        visited_count_backward=len(parents_backward),
                        intersection_found=False,
                        intersection_node=None,
                    )
                )

            else:  # BACKWARD
                curr_node = queue_backward.popleft()
                nodes_expanded_bwd += 1
                neighbors = self.backward_graph.get(curr_node, [])
                newly_visited_list = []

                for neighbor in neighbors:
                    # Check frontier intersection immediately upon generation
                    if neighbor in parents_forward:
                        parents_backward[neighbor] = curr_node
                        newly_visited_list.append(neighbor)
                        intersection_node = neighbor
                        nodes_generated_bwd += 1

                        steps.append(
                            BidirectionalStepRecord(
                                step_number=step_counter,
                                direction="BACKWARD",
                                expanded_node=curr_node,
                                neighbors_evaluated=list(neighbors),
                                newly_visited=newly_visited_list,
                                frontier_size_forward=len(queue_forward),
                                frontier_size_backward=len(queue_backward),
                                visited_count_forward=len(parents_forward),
                                visited_count_backward=len(parents_backward),
                                intersection_found=True,
                                intersection_node=intersection_node,
                            )
                        )
                        break

                    if neighbor not in parents_backward:
                        parents_backward[neighbor] = curr_node
                        queue_backward.append(neighbor)
                        newly_visited_list.append(neighbor)
                        nodes_generated_bwd += 1

                if intersection_node is not None:
                    break

                steps.append(
                    BidirectionalStepRecord(
                        step_number=step_counter,
                        direction="BACKWARD",
                        expanded_node=curr_node,
                        neighbors_evaluated=list(neighbors),
                        newly_visited=newly_visited_list,
                        frontier_size_forward=len(queue_forward),
                        frontier_size_backward=len(queue_backward),
                        visited_count_forward=len(parents_forward),
                        visited_count_backward=len(parents_backward),
                        intersection_found=False,
                        intersection_node=None,
                    )
                )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        # If frontiers collided, reconstruct combined shortest path
        if intersection_node is not None:
            forward_path = self._reconstruct_forward_path(parents_forward, intersection_node)
            backward_path = self._reconstruct_backward_path(parents_backward, intersection_node)
            # Splice paths at intersection node: forward_path ends with intersection,
            # backward_path starts with intersection.
            combined_path = forward_path[:-1] + backward_path

            return BidirectionalSearchResult(
                status="Success",
                start_node=start,
                goal_node=goal,
                intersection_node=intersection_node,
                path=combined_path,
                path_length=len(combined_path) - 1,
                forward_path=forward_path,
                backward_path=backward_path,
                nodes_expanded_forward=nodes_expanded_fwd,
                nodes_expanded_backward=nodes_expanded_bwd,
                total_nodes_expanded=nodes_expanded_fwd + nodes_expanded_bwd,
                nodes_generated_forward=nodes_generated_fwd,
                nodes_generated_backward=nodes_generated_bwd,
                total_nodes_generated=nodes_generated_fwd + nodes_generated_bwd,
                visited_forward=set(parents_forward.keys()),
                visited_backward=set(parents_backward.keys()),
                steps=steps,
                execution_time_ms=elapsed,
                graph=self.forward_graph,
                alternation_strategy=self.alternation_strategy,
            )

        # No intersection found (graph disconnected or goal unreachable)
        return BidirectionalSearchResult(
            status="Failure",
            start_node=start,
            goal_node=goal,
            intersection_node=None,
            path=None,
            path_length=0,
            forward_path=None,
            backward_path=None,
            nodes_expanded_forward=nodes_expanded_fwd,
            nodes_expanded_backward=nodes_expanded_bwd,
            total_nodes_expanded=nodes_expanded_fwd + nodes_expanded_bwd,
            nodes_generated_forward=nodes_generated_fwd,
            nodes_generated_backward=nodes_generated_bwd,
            total_nodes_generated=nodes_generated_fwd + nodes_generated_bwd,
            visited_forward=set(parents_forward.keys()),
            visited_backward=set(parents_backward.keys()),
            steps=steps,
            execution_time_ms=elapsed,
            graph=self.forward_graph,
            alternation_strategy=self.alternation_strategy,
        )

    def _reconstruct_forward_path(
        self, parents_forward: Dict[str, Optional[str]], meet_node: str
    ) -> List[str]:
        """Reconstruct the forward path from start node to intersection node."""
        path: List[str] = []
        curr: Optional[str] = meet_node
        while curr is not None:
            path.append(curr)
            curr = parents_forward.get(curr)
        path.reverse()
        return path

    def _reconstruct_backward_path(
        self, parents_backward: Dict[str, Optional[str]], meet_node: str
    ) -> List[str]:
        """Reconstruct the backward path from intersection node to goal node."""
        path: List[str] = []
        curr: Optional[str] = meet_node
        while curr is not None:
            path.append(curr)
            curr = parents_backward.get(curr)
        return path


def bidirectional_search(
    graph: Optional[Dict[str, Sequence[str]]] = None,
    start: str = "A",
    goal: str = "G",
    is_directed: bool = False,
    alternation_strategy: str = "turn",
) -> BidirectionalSearchResult:
    """Convenience procedural wrapper to execute Bidirectional Search.

    Args:
        graph: Optional adjacency list. Defaults to canonical SAMPLE_BDS_GRAPH.
        start: Start node label. Defaults to 'A'.
        goal: Goal node label. Defaults to 'G'.
        is_directed: Whether graph is directed. Defaults to False (undirected).
        alternation_strategy: 'turn' or 'balanced'.

    Returns:
        BidirectionalSearchResult.
    """
    solver = BidirectionalSearch(
        graph=graph,
        is_directed=is_directed,
        alternation_strategy=alternation_strategy,
    )
    return solver.search(start=start, goal=goal)


# ============================================================================
# Optional Challenge: 2D Grid Maze Bidirectional Search
# ============================================================================

Coord = Tuple[int, int]
MAZE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


@dataclass
class GridMazeBidirectionalResult:
    """Result of Bidirectional Search applied to a 2D Grid Maze.

    Attributes:
        status: 'Success' or 'Failure'.
        start_coord: Starting cell coordinate (row, col).
        goal_coord: Goal cell coordinate (row, col).
        intersection_coord: Meeting coordinate where forward and backward frontiers met.
        path: Ordered list of coordinates from start to goal.
        path_length: Number of step transitions in the solution path.
        nodes_expanded: Combined count of grid cells expanded.
        visited_forward: Set of coordinates visited by forward search.
        visited_backward: Set of coordinates visited by backward search.
        maze: 2D binary grid representation (0 = path, 1 = wall).
        execution_time_ms: Search execution duration in milliseconds.
    """

    status: str
    start_coord: Coord
    goal_coord: Coord
    intersection_coord: Optional[Coord]
    path: Optional[List[Coord]]
    path_length: int
    nodes_expanded: int
    visited_forward: Set[Coord]
    visited_backward: Set[Coord]
    maze: List[List[int]]
    execution_time_ms: float = 0.0

    def render_ascii_maze(self) -> str:
        """Render the maze with paths, frontiers, start, goal, and intersection."""
        rows = len(self.maze)
        cols = len(self.maze[0])
        path_set = set(self.path) if self.path else set()

        output_rows: List[str] = []
        for r in range(rows):
            line_chars: List[str] = []
            for c in range(cols):
                coord = (r, c)
                if coord == self.start_coord:
                    line_chars.append(" S ")
                elif coord == self.goal_coord:
                    line_chars.append(" G ")
                elif coord == self.intersection_coord:
                    line_chars.append(" X ")
                elif coord in path_set:
                    line_chars.append(" • ")
                elif self.maze[r][c] == 1:
                    line_chars.append("███")
                elif coord in self.visited_forward:
                    line_chars.append(" f ")
                elif coord in self.visited_backward:
                    line_chars.append(" b ")
                else:
                    line_chars.append(" . ")
            output_rows.append("".join(line_chars))

        legend = (
            "Legend: [S]=Start, [G]=Goal, [X]=Intersection, [•]=Solution Path, "
            "[f]=Forward Visited, [b]=Backward Visited, [███]=Wall"
        )
        return "\n".join(output_rows) + "\n" + legend


def solve_grid_maze_bidirectional(
    maze: Sequence[Sequence[int]],
    start: Coord,
    goal: Coord,
) -> GridMazeBidirectionalResult:
    """Solve a 2D grid maze using Bidirectional Breadth-First Search.

    Args:
        maze: 2D matrix where 0 represents open space and 1 represents wall.
        start: (row, col) start coordinates.
        goal: (row, col) goal coordinates.

    Returns:
        GridMazeBidirectionalResult with meeting coordinate and reconstructed path.

    Raises:
        ValueError: If start or goal coordinates are out of bounds or on a wall.
    """
    rows = len(maze)
    if rows == 0:
        raise ValueError("Maze grid cannot be empty.")
    cols = len(maze[0])

    r_s, c_s = start
    r_g, c_g = goal

    if not (0 <= r_s < rows and 0 <= c_s < cols):
        raise ValueError(f"Start coordinate {start} is out of maze bounds ({rows}x{cols}).")
    if not (0 <= r_g < rows and 0 <= c_g < cols):
        raise ValueError(f"Goal coordinate {goal} is out of maze bounds ({rows}x{cols}).")
    if maze[r_s][c_s] == 1:
        raise ValueError(f"Start coordinate {start} lies inside a wall obstacle.")
    if maze[r_g][c_g] == 1:
        raise ValueError(f"Goal coordinate {goal} lies inside a wall obstacle.")

    start_time = time.perf_counter()

    # Base case: start is goal
    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GridMazeBidirectionalResult(
            status="Success",
            start_coord=start,
            goal_coord=goal,
            intersection_coord=start,
            path=[start],
            path_length=0,
            nodes_expanded=0,
            visited_forward={start},
            visited_backward={goal},
            maze=[list(row) for row in maze],
            execution_time_ms=elapsed,
        )

    queue_fwd: deque[Coord] = deque([start])
    queue_bwd: deque[Coord] = deque([goal])

    parents_fwd: Dict[Coord, Optional[Coord]] = {start: None}
    parents_bwd: Dict[Coord, Optional[Coord]] = {goal: None}

    intersection: Optional[Coord] = None
    expansions = 0

    while queue_fwd and queue_bwd:
        # Forward expansion
        curr_fwd = queue_fwd.popleft()
        expansions += 1
        rf, cf = curr_fwd

        for dr, dc in MAZE_MOVES:
            nr, nc = rf + dr, cf + dc
            nbr = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if nbr in parents_bwd:
                    parents_fwd[nbr] = curr_fwd
                    intersection = nbr
                    break
                if nbr not in parents_fwd:
                    parents_fwd[nbr] = curr_fwd
                    queue_fwd.append(nbr)

        if intersection is not None:
            break

        # Backward expansion
        curr_bwd = queue_bwd.popleft()
        expansions += 1
        rb, cb = curr_bwd

        for dr, dc in MAZE_MOVES:
            nr, nc = rb + dr, cb + dc
            nbr = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if nbr in parents_fwd:
                    parents_bwd[nbr] = curr_bwd
                    intersection = nbr
                    break
                if nbr not in parents_bwd:
                    parents_bwd[nbr] = curr_bwd
                    queue_bwd.append(nbr)

        if intersection is not None:
            break

    elapsed = (time.perf_counter() - start_time) * 1000.0

    if intersection is not None:
        # Reconstruct path
        path_fwd: List[Coord] = []
        curr: Optional[Coord] = intersection
        while curr is not None:
            path_fwd.append(curr)
            curr = parents_fwd.get(curr)
        path_fwd.reverse()

        path_bwd: List[Coord] = []
        curr = parents_bwd.get(intersection)
        while curr is not None:
            path_bwd.append(curr)
            curr = parents_bwd.get(curr)

        full_path = path_fwd + path_bwd

        return GridMazeBidirectionalResult(
            status="Success",
            start_coord=start,
            goal_coord=goal,
            intersection_coord=intersection,
            path=full_path,
            path_length=len(full_path) - 1,
            nodes_expanded=expansions,
            visited_forward=set(parents_fwd.keys()),
            visited_backward=set(parents_bwd.keys()),
            maze=[list(row) for row in maze],
            execution_time_ms=elapsed,
        )

    return GridMazeBidirectionalResult(
        status="Failure",
        start_coord=start,
        goal_coord=goal,
        intersection_coord=None,
        path=None,
        path_length=0,
        nodes_expanded=expansions,
        visited_forward=set(parents_fwd.keys()),
        visited_backward=set(parents_bwd.keys()),
        maze=[list(row) for row in maze],
        execution_time_ms=elapsed,
    )


# ============================================================================
# Theoretical Comparison & Viva Voce Reference Guide
# ============================================================================


def compare_unidirectional_vs_bidirectional(
    graph: Dict[str, List[str]], start: str, goal: str
) -> Dict[str, Any]:
    """Empirically compare standard Unidirectional BFS against Bidirectional BFS.

    Returns:
        Dictionary detailing expansions, path optimality, and reduction ratio.
    """
    # 1. Bidirectional BFS
    bds_result = bidirectional_search(graph=graph, start=start, goal=goal)

    # 2. Standard Forward BFS
    start_time = time.perf_counter()
    queue_uni: deque[str] = deque([start])
    parents_uni: Dict[str, Optional[str]] = {start: None}
    uni_expansions = 0
    found_uni = False

    while queue_uni:
        curr = queue_uni.popleft()
        uni_expansions += 1
        if curr == goal:
            found_uni = True
            break
        for nbr in graph.get(curr, []):
            if nbr not in parents_uni:
                parents_uni[nbr] = curr
                queue_uni.append(nbr)

    uni_time_ms = (time.perf_counter() - start_time) * 1000.0

    uni_path: Optional[List[str]] = None
    if found_uni:
        p: List[str] = []
        c: Optional[str] = goal
        while c is not None:
            p.append(c)
            c = parents_uni.get(c)
        p.reverse()
        uni_path = p

    expansion_reduction = (
        (uni_expansions - bds_result.total_nodes_expanded) / uni_expansions * 100.0
        if uni_expansions > 0
        else 0.0
    )

    return {
        "start": start,
        "goal": goal,
        "unidirectional_expansions": uni_expansions,
        "bidirectional_expansions": bds_result.total_nodes_expanded,
        "expansion_reduction_percent": expansion_reduction,
        "unidirectional_path_length": len(uni_path) - 1 if uni_path else None,
        "bidirectional_path_length": bds_result.path_length,
        "optimal_path_match": uni_path == bds_result.path,
        "intersection_node": bds_result.intersection_node,
        "unidirectional_time_ms": uni_time_ms,
        "bidirectional_time_ms": bds_result.execution_time_ms,
    }


def get_bidirectional_search_viva_reference() -> Dict[str, Dict[str, str]]:
    """Return comprehensive Viva Voce questions, answers, and algorithmic concepts.

    Topics covered:
    1. Why O(b^(d/2)) is massively faster than O(b^d).
    2. Difficulties of backward search in directed graphs.
    3. Predecessor vs Successor generation.
    """
    return {
        "complexity_advantage": {
            "question": "Why is O(b^(d/2)) massively faster than O(b^d)?",
            "answer": (
                "In a search space with branching factor b and solution depth d, a standard forward "
                "BFS explores a single hypersphere of radius d, generating roughly b^0 + b^1 + ... + b^d "
                "= O(b^d) states. In contrast, Bidirectional Search grows two simultaneous hyperspheres of "
                "radius d/2—one from the start state and one from the goal state. When they meet in the middle, "
                "the total states generated is b^(d/2) + b^(d/2) = 2 * b^(d/2) = O(b^(d/2)). "
                "Because exponential functions dominate polynomial constants, halving the exponent yields "
                "exponential savings. For example, when b = 10 and d = 10, O(b^d) is 10^10 (10 billion nodes), "
                "whereas 2 * b^(d/2) is 2 * 10^5 (200,000 nodes)—a 50,000x reduction in memory and execution time."
            ),
        },
        "backward_search_difficulties": {
            "question": "What are the major difficulties of backward search in directed graphs?",
            "answer": (
                "Backward search faces three fundamental engineering and theoretical hurdles in directed graphs: "
                "1. Non-reversible edges & predecessor computation: If graph edges are one-way (u -> v), "
                "exploring backward requires inverting the graph or computing predecessors on the fly. "
                "If only outgoing edges are indexed, finding incoming edges requires an exhaustive graph scan. "
                "2. Multiple or abstract goal states: Forward search typically begins with a single well-defined "
                "initial state. If the goal is specified by a condition/predicate (e.g., checkmate in chess, "
                "any state where clean_ratio == 1.0) rather than an explicit single state, backward search cannot "
                "instantiate a single backward root without creating a dummy super-goal node connected to all valid goals. "
                "3. Irreversible actions: In physical or robotic state spaces, operations may be strictly non-invertible "
                "(e.g., dropping an egg, burning fuel, irreversible chemical reactions)."
            ),
        },
        "predecessor_vs_successor": {
            "question": "Explain Predecessor vs Successor generation and their impact on search efficiency.",
            "answer": (
                "The Successor function generates all states reachable by applying valid forward actions to state s: "
                "Successors(s) = { s' | s -> s' }. The Predecessor function computes all states that could lead to state s': "
                "Predecessors(s') = { s | s -> s' }. In an undirected graph, Predecessors(s) = Successors(s). "
                "In directed domains, the forward branching factor b_fwd (out-degree) and backward branching factor "
                "b_bwd (in-degree) can differ dramatically. For instance, in theorem proving or abstract games, "
                "many distinct antecedent states may collapse into the same simplified theorem (high in-degree, low out-degree). "
                "If b_bwd >> b_fwd, backward search will suffer combinatorial explosion much faster than forward search. "
                "Dynamic frontier balancing (always expanding whichever frontier has fewer elements) helps mitigate this asymmetry."
            ),
        },
    }


# ============================================================================
# Visualizations: Graph Plotting & Maze Heatmaps
# ============================================================================


def plot_bidirectional_search(
    result: BidirectionalSearchResult,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Any:
    """Render and optionally save a visual diagram of the bidirectional graph search.

    Args:
        result: BidirectionalSearchResult containing graph, path, and visited states.
        save_path: Optional file path (PNG/PDF) to save figure.
        title: Optional plot title override.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib or networkx is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise ImportError("matplotlib and networkx are required for graphical plotting.") from e

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # Build NetworkX graph
    G = nx.Graph()
    for u, nbrs in result.graph.items():
        G.add_node(u)
        for v in nbrs:
            G.add_edge(u, v)

    # Use deterministic spring layout with fixed seed
    pos = nx.spring_layout(G, seed=42)

    # Determine node colors
    node_colors: List[str] = []
    path_set = set(result.path) if result.path else set()

    for node in G.nodes():
        if node == result.start_node:
            node_colors.append("#2ecc71")  # Green for Start
        elif node == result.goal_node:
            node_colors.append("#e74c3c")  # Red for Goal
        elif node == result.intersection_node:
            node_colors.append("#f39c12")  # Gold for Intersection
        elif node in result.visited_forward and node in result.visited_backward:
            node_colors.append("#9b59b6")  # Purple for both
        elif node in result.visited_forward:
            node_colors.append("#3498db")  # Blue for forward visited
        elif node in result.visited_backward:
            node_colors.append("#e67e22")  # Orange for backward visited
        else:
            node_colors.append("#bdc3c7")  # Gray for unvisited

    # Draw all edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bdc3c7", width=1.5, style="dashed")

    # Highlight path edges if solution exists
    if result.path and len(result.path) > 1:
        path_edges = list(zip(result.path[:-1], result.path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax, edge_color="#2980b9", width=4.0)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=750, edgecolors="#2c3e50", linewidths=1.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold", font_color="#ffffff")

    plot_title = title or (
        f"Bidirectional Search: {result.start_node} -> {result.goal_node}\n"
        f"Intersection: {result.intersection_node or 'None'} | Path Length: {result.path_length} edges"
    )
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"Start ({result.start_node})", markerfacecolor="#2ecc71", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Goal ({result.goal_node})", markerfacecolor="#e74c3c", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Intersection ({result.intersection_node})", markerfacecolor="#f39c12", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Forward Visited", markerfacecolor="#3498db", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Backward Visited", markerfacecolor="#e67e22", markersize=10),
        Line2D([0], [0], color="#2980b9", lw=3.5, label="Shortest Path"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig


def plot_grid_maze_bidirectional(
    result: GridMazeBidirectionalResult,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Any:
    """Render a heatmap visualization of a 2D Grid Maze solved via Bidirectional BFS.

    Args:
        result: GridMazeBidirectionalResult.
        save_path: Optional file path to save plot.
        title: Optional plot title override.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib or numpy is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError("matplotlib and numpy are required for graphical plotting.") from e

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    rows = len(result.maze)
    cols = len(result.maze[0])

    # Visual map: 0 = unvisited, 1 = wall, 2 = forward, 3 = backward, 4 = path
    grid_img = np.zeros((rows, cols, 3), dtype=float)

    # Color definitions (RGB)
    COLOR_BG = np.array([0.96, 0.96, 0.96])       # Light gray
    COLOR_WALL = np.array([0.15, 0.15, 0.15])     # Dark charcoal
    COLOR_FWD = np.array([0.53, 0.81, 0.98])      # Light sky blue
    COLOR_BWD = np.array([1.00, 0.72, 0.42])      # Light orange
    COLOR_BOTH = np.array([0.75, 0.60, 0.90])     # Light purple

    for r in range(rows):
        for c in range(cols):
            coord = (r, c)
            if result.maze[r][c] == 1:
                grid_img[r, c] = COLOR_WALL
            elif coord in result.visited_forward and coord in result.visited_backward:
                grid_img[r, c] = COLOR_BOTH
            elif coord in result.visited_forward:
                grid_img[r, c] = COLOR_FWD
            elif coord in result.visited_backward:
                grid_img[r, c] = COLOR_BWD
            else:
                grid_img[r, c] = COLOR_BG

    ax.imshow(grid_img, origin="upper")

    # Plot solution path
    if result.path and len(result.path) > 1:
        pr = [p[0] for p in result.path]
        pc = [p[1] for p in result.path]
        ax.plot(pc, pr, color="#e67e22", linewidth=3.5, linestyle="-", label=f"Path ({result.path_length} steps)", zorder=4)
        ax.scatter(pc, pr, color="#d35400", s=35, zorder=5)

    # Plot Start, Goal, and Intersection
    sr, sc = result.start_coord
    gr, gc = result.goal_coord
    ax.plot(sc, sr, marker="o", markersize=12, color="#2ecc71", markeredgecolor="#27ae60", markeredgewidth=2, label=f"Start {result.start_coord}", zorder=6)
    ax.plot(gc, gr, marker="*", markersize=17, color="#e74c3c", markeredgecolor="#c0392b", markeredgewidth=1.5, label=f"Goal {result.goal_coord}", zorder=6)

    if result.intersection_coord:
        ir, ic = result.intersection_coord
        ax.plot(ic, ir, marker="X", markersize=13, color="#f1c40f", markeredgecolor="#f39c12", markeredgewidth=2, label=f"Meet {result.intersection_coord}", zorder=7)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)
    ax.grid(color="#7f8c8d", linestyle=":", linewidth=0.5, alpha=0.5)

    plot_title = title or (
        f"Grid Maze Bidirectional Search Simulation\n"
        f"Length: {result.path_length} steps | Expanded: {result.nodes_expanded} cells"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig

