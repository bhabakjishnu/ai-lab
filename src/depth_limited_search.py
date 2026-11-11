"""Depth-Limited Search (DLS) Implementation for State-Space Exploration.

This module implements Depth-Limited Search (DLS) for graph and tree traversal,
adhering to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA 3rd/4th ed.,
Chapter 3: Solving Problems by Searching).

Theoretical Background & AI Concepts:
- Uninformed (Blind) Bounded Search:
    Depth-Limited Search (DLS) addresses the fundamental vulnerability of standard Depth-First
    Search (DFS) in infinite state spaces or graphs with loops. DFS can plunge down an infinite
    branch forever (incompleteness), never backtracking to explore shallow goal states on
    alternate branches. DLS imposes a predetermined maximum depth limit `l` on the search depth.
    Nodes at depth `l` are treated as leaf nodes with no successors, guaranteeing termination.
- Recursive Bounded Depth-First Search:
    The search is formulated as a recursive function accepting the current state/node, target goal,
    and a remaining depth limit budget `limit`:
    - On each recursive call to a child node, `limit` is decremented by 1 (`limit - 1`).
    - At depth 0 (relative remaining limit), no further expansion is permitted.
- Tri-State Outcome Model:
    Unlike binary search algorithms (found / not found), DLS produces three distinct semantic states:
    1. Success:
       The goal node was discovered at or within depth limit `l`. Returns the reconstructed solution
       path from start to goal (e.g. `['1', '2', '4']`).
    2. Cutoff:
       The goal was NOT found within depth limit `l`, BUT at least one branch was truncated because
       the depth limit was reached (`limit == 0`). This indicates search incompleteness at the
       current limit: the goal might still exist at a deeper level (`depth > l`).
    3. Failure:
       The search space reachable from the start node was exhaustively traversed to natural dead-ends
       or leaf nodes without finding the goal AND without hitting the depth limit. This guarantees
       that the goal does NOT exist anywhere in the reachable component.
- Fundamental Importance for Iterative Deepening Search (IDS):
    The distinction between 'Cutoff' and 'Failure' is the bedrock of Iterative Deepening Search (IDS).
    IDS repeatedly invokes DLS with limits `l = 0, 1, 2, ...`.
    - If DLS returns Cutoff: IDS increases `l` and continues searching.
    - If DLS returns Failure: IDS terminates immediately, knowing no solution exists at any depth.
    - If DLS returns Success: IDS terminates and returns the optimal shallowest path.
- Incompleteness and Non-Optimality:
    - Incompleteness: If the shallowest goal lies at depth `d > l`, DLS will never find it (returns Cutoff).
    - Non-Optimality: If `l > d`, DLS may find a path deeper than `d` first due to DFS branch order.
- Complexity Analysis:
    - Time Complexity: O(b^l), where `b` is the branching factor and `l` is the depth limit.
    - Space Complexity: O(b * l), requiring storage only for the active recursion stack and sibling
      alternatives, retaining DFS's linear memory advantage over BFS's exponential O(b^d).
- Optional Challenge:
    Tracks the deepest node(s) reached before cutoff occurs, capturing the furthest exploration frontier
    and pinpointing exactly where the depth boundary severed tree expansion.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Standard DLS Return State Constants
CUTOFF: str = "Cutoff"
FAILURE: str = "Failure"
SUCCESS: str = "Success"


class DLSStatus(str, Enum):
    """Enumeration of the three possible outcomes of Depth-Limited Search."""

    SUCCESS = SUCCESS
    CUTOFF = CUTOFF
    FAILURE = FAILURE


# Benchmark Graphs for Testing, Demonstration, and Education
SAMPLE_DLS_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4"],
    "3": ["5"],
    "4": ["6"],
    "5": [],
    "6": ["7"],
    "7": [],
}
"""Canonical sample graph from lab question:
Depth 0: 1
Depth 1: 2, 3
Depth 2: 4, 5
Depth 3: 6
Depth 4: 7
With Limit=2 and Goal='6' (at depth 3), search must terminate with Result: Cutoff.
"""

BINARY_TREE_DLS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H", "I"],
    "E": [],
    "F": [],
    "G": ["J"],
    "H": [],
    "I": [],
    "J": [],
}

CYCLIC_DLS_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4"],
    "3": ["1", "5"],
    "4": ["3"],
    "5": [],
}

DISCONNECTED_DLS_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": ["C"],
    "C": [],
    "X": ["Y"],
    "Y": ["Z"],
    "Z": [],
}

LINEAR_CHAIN_DLS_GRAPH: Dict[str, List[str]] = {
    "N0": ["N1"],
    "N1": ["N2"],
    "N2": ["N3"],
    "N3": ["N4"],
    "N4": ["N5"],
    "N5": [],
}


@dataclass(frozen=True)
class DLSStepRecord:
    """Immutable audit log entry recording a single decision step during DLS.

    Attributes:
        step_number: 1-based chronological index of the execution step.
        current_node: Node currently being visited or evaluated.
        current_depth: Distance from the root start node along the active branch.
        remaining_limit: Depth budget remaining for expanding successors from this node.
        action: Semantic label ('GOAL_TEST_PASS', 'CUTOFF_HIT', 'EXPAND_CHILDREN',
                'CYCLE_SKIP', 'BACKTRACK', 'LEAF_FAILURE').
        path_to_node: Sequence of nodes from start node to current node.
        result_returned: Local return value ('Success', 'Cutoff', 'Failure', or 'CONTINUE').
        details: Human-readable technical commentary explaining the decision.
    """

    step_number: int
    current_node: str
    current_depth: int
    remaining_limit: int
    action: str
    path_to_node: List[str]
    result_returned: str
    details: str

    def format_row(self) -> str:
        """Format the step into a tabular row for terminal presentation.

        Returns:
            Formatted row string.
        """
        path_str = "->".join(self.path_to_node)
        return (
            f"{self.step_number:<5} | {self.current_node:<6} | {self.current_depth:<5} | "
            f"{self.remaining_limit:<9} | {self.action:<16} | {self.result_returned:<9} | {path_str}"
        )


@dataclass
class DLSResult:
    """Encapsulates the complete outcome and performance telemetry of DLS.

    Attributes:
        status: The final search outcome: 'Success', 'Cutoff', or 'Failure'.
        path: Ordered list of nodes from start to goal if status is 'Success', else None.
        depth: Number of edges in the solution path if found, else None.
        limit: The user-specified depth limit constraint.
        start_node: Node where the search originated.
        goal_node: Node sought by the search.
        graph: Adjacency list representation of the explored graph.
        max_depth_reached: Maximum depth reached by any explored node.
        deepest_nodes: List of nodes reached at the maximum depth (Optional Challenge).
        cutoff_nodes: Nodes where expansion was explicitly blocked by the depth limit (limit == 0).
        nodes_expanded: Count of internal nodes whose outgoing neighbors were examined.
        nodes_generated: Count of total node visits made during recursive calls.
        cutoff_occurred: Boolean indicating if at least one branch hit the depth limit.
        steps: Chronological list of DLSStepRecord objects capturing the full execution trace.
        execution_time_ms: Wall-clock execution time in milliseconds.
    """

    status: str
    path: Optional[List[str]]
    depth: Optional[int]
    limit: int
    start_node: str
    goal_node: str
    graph: Dict[str, List[str]]
    max_depth_reached: int
    deepest_nodes: List[str]
    cutoff_nodes: List[str]
    nodes_expanded: int
    nodes_generated: int
    cutoff_occurred: bool
    steps: List[DLSStepRecord]
    execution_time_ms: float = 0.0

    @property
    def deepest_node(self) -> Optional[str]:
        """Return the primary deepest node reached before cutoff (Optional Challenge).

        Returns:
            String label of the first deepest node recorded, or None if no nodes visited.
        """
        return self.deepest_nodes[0] if self.deepest_nodes else None

    def is_success(self) -> bool:
        """Check whether the goal was successfully discovered.

        Returns:
            True if status is 'Success', False otherwise.
        """
        return self.status == SUCCESS

    def is_cutoff(self) -> bool:
        """Check whether the search was prematurely truncated by the depth limit.

        Returns:
            True if status is 'Cutoff', False otherwise.
        """
        return self.status == CUTOFF

    def is_failure(self) -> bool:
        """Check whether the search proved the goal is unreachable.

        Returns:
            True if status is 'Failure', False otherwise.
        """
        return self.status == FAILURE

    def format_result(self) -> str:
        """Return the standard output format matching university practical specifications.

        Returns:
            Formatted result string: 'Result: <Path>', 'Result: Cutoff', or 'Result: Failure'.
        """
        if self.is_success() and self.path is not None:
            return f"Result: Path: {self.path}"
        return f"Result: {self.status}"

    def format_path(self, delimiter: str = " -> ") -> str:
        """Return the formatted solution path.

        Args:
            delimiter: Separator between successive nodes.

        Returns:
            Formatted path string, or 'No Path' if not found.
        """
        if self.path:
            return delimiter.join(self.path)
        return "No Path Found"

    def format_step_table(self) -> str:
        """Return a formatted ASCII execution trace table.

        Returns:
            Table string showing step, node, depth, limit remaining, action, result, and path.
        """
        headers = (
            f"{'Step':<5} | {'Node':<6} | {'Depth':<5} | {'Rem.Limit':<9} | "
            f"{'Action':<16} | {'Return':<9} | {'Active Path'}"
        )
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def format_summary(self) -> str:
        """Return a formatted multi-line summary of DLS execution metrics.

        Returns:
            Descriptive string report.
        """
        deepest_str = ", ".join(self.deepest_nodes) if self.deepest_nodes else "None"
        cutoff_str = ", ".join(self.cutoff_nodes) if self.cutoff_nodes else "None"
        path_str = self.format_path()

        lines = [
            "=" * 70,
            "DEPTH-LIMITED SEARCH (DLS) EXECUTION SUMMARY",
            "=" * 70,
            f"Search Problem           : Start='{self.start_node}' -> Goal='{self.goal_node}'",
            f"Depth Limit (l)          : {self.limit}",
            f"Final Outcome            : {self.status.upper()} ({self.format_result()})",
            f"Solution Path            : {path_str}",
            f"Solution Depth (edges)   : {self.depth if self.depth is not None else 'N/A'}",
            f"Nodes Generated (Visits) : {self.nodes_generated}",
            f"Nodes Expanded           : {self.nodes_expanded}",
            f"Max Depth Reached        : {self.max_depth_reached}",
            f"Deepest Node(s) Reached  : {deepest_str} (Optional Challenge)",
            f"Cutoff Boundary Nodes    : {cutoff_str}",
            f"Cutoff Occurred Flag     : {self.cutoff_occurred}",
            f"Execution Time           : {self.execution_time_ms:.4f} ms",
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a serializable dictionary.

        Returns:
            Dictionary containing all result properties.
        """
        return {
            "status": self.status,
            "path": self.path,
            "depth": self.depth,
            "limit": self.limit,
            "start_node": self.start_node,
            "goal_node": self.goal_node,
            "max_depth_reached": self.max_depth_reached,
            "deepest_nodes": self.deepest_nodes,
            "cutoff_nodes": self.cutoff_nodes,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "cutoff_occurred": self.cutoff_occurred,
            "execution_time_ms": self.execution_time_ms,
        }


def normalize_graph(graph: Dict[Any, Sequence[Any]]) -> Dict[str, List[str]]:
    """Sanitize and normalize an arbitrary graph dictionary into standard format.

    Ensures:
    - All node keys and neighbor values are converted to stripped strings.
    - Every neighbor node referenced in adjacency lists has an entry in the graph dictionary.
    - Redundant duplicate edges to the same neighbor are deduplicated while preserving order.

    Args:
        graph: Raw graph dictionary mapping nodes to sequences of neighbors.

    Returns:
        Normalized dictionary mapping str node labels to lists of str neighbors.

    Raises:
        ValueError: If graph is not a dictionary or is empty.
    """
    if not isinstance(graph, dict):
        raise ValueError(f"Graph must be a dictionary, got {type(graph).__name__}")
    if not graph:
        raise ValueError("Graph dictionary cannot be empty.")

    normalized: Dict[str, List[str]] = {}

    for raw_node, raw_neighbors in graph.items():
        node_str = str(raw_node).strip()
        if not node_str:
            continue
        cleaned_neighbors: List[str] = []
        seen_neighbors: Set[str] = set()
        for nbr in raw_neighbors:
            nbr_str = str(nbr).strip()
            if nbr_str and nbr_str not in seen_neighbors:
                seen_neighbors.add(nbr_str)
                cleaned_neighbors.append(nbr_str)
        normalized[node_str] = cleaned_neighbors

    # Ensure all referenced neighbor nodes exist as keys
    all_referenced_neighbors = {nbr for nbrs in normalized.values() for nbr in nbrs}
    for nbr in all_referenced_neighbors:
        if nbr not in normalized:
            normalized[nbr] = []

    return normalized


def parse_graph_input(raw_input: Union[str, Dict[str, Sequence[str]]]) -> Dict[str, List[str]]:
    """Parse user-provided graph input from multiple supported formats.

    Supported Formats:
    1. Python dict directly: `{'1': ['2', '3'], '2': ['4']}`
    2. JSON string: `'{"1": ["2", "3"], "2": ["4"]}'`
    3. Python dict literal string: `"{'1': ['2', '3'], '2': ['4']}"`
    4. Compact adjacency notation: `"1: 2, 3; 2: 4; 3: ; 4: "`
    5. Edge-list notation: `"1-2, 1-3, 2-4"` or `"1->2, 1->3, 2->4"`

    Args:
        raw_input: String or dictionary representation of the graph.

    Returns:
        Normalized adjacency list dictionary.

    Raises:
        ValueError: If input format cannot be parsed or is invalid.
    """
    if isinstance(raw_input, dict):
        return normalize_graph(raw_input)

    text = str(raw_input).strip()
    if not text:
        raise ValueError("Input string cannot be empty.")

    # Try JSON parsing
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, dict):
                return normalize_graph(parsed_json)
        except json.JSONDecodeError:
            pass

        # Try Python dict literal parsing safely
        try:
            import ast

            evaluated = ast.literal_eval(text)
            if isinstance(evaluated, dict):
                return normalize_graph(evaluated)
        except (ValueError, SyntaxError):
            pass

    # Try compact adjacency notation: "1: 2, 3; 2: 4; 3: ; 4: "
    if ":" in text:
        graph: Dict[str, List[str]] = {}
        entries = [segment.strip() for segment in text.split(";") if segment.strip()]
        for entry in entries:
            if ":" not in entry:
                continue
            src, dsts = entry.split(":", 1)
            src = src.strip()
            neighbors = [d.strip() for d in dsts.split(",") if d.strip()]
            graph[src] = neighbors
        if graph:
            return normalize_graph(graph)

    # Try edge-list notation: "1-2, 1-3, 2-4" or "1->2, 1->3"
    if "-" in text:
        graph = {}
        raw_edges = [edge.strip() for edge in text.split(",") if edge.strip()]
        for edge_str in raw_edges:
            cleaned = edge_str.replace("->", "-")
            if "-" in cleaned:
                parts = cleaned.split("-", 1)
                u = parts[0].strip()
                v = parts[1].strip()
                if u not in graph:
                    graph[u] = []
                if v and v not in graph[u]:
                    graph[u].append(v)
                if v and v not in graph:
                    graph[v] = []
        if graph:
            return normalize_graph(graph)

    raise ValueError(
        f"Unable to parse graph input '{raw_input}'. Please supply a valid dictionary, "
        "JSON object, or semicolon-delimited adjacency list (e.g., '1: 2, 3; 2: 4')."
    )


class DepthLimitedSearch:
    """Object-oriented engine for Depth-Limited Search (DLS).

    Provides fine-grained control over execution, step recording, cycle detection,
    and performance tracking.
    """

    def __init__(
        self,
        graph: Dict[str, Sequence[str]],
        cycle_prevention: bool = True,
    ) -> None:
        """Initialize the DLS search engine with a normalized graph.

        Args:
            graph: Graph adjacency structure.
            cycle_prevention: If True, prevents revisiting ancestor nodes on the active
                recursion path. Defaults to True to guarantee termination on cyclic graphs.
        """
        self.graph = normalize_graph(graph)
        self.cycle_prevention = cycle_prevention

    def search(
        self,
        start: str,
        goal: str,
        limit: int,
    ) -> DLSResult:
        """Execute Depth-Limited Search from start to goal bounded by limit.

        Args:
            start: Label of the root start node.
            goal: Label of the target destination node.
            limit: Maximum non-negative search depth permitted.

        Returns:
            DLSResult object containing search status, path, telemetry, and steps.

        Raises:
            ValueError: If limit is negative, or start/goal nodes are invalid.
        """
        return depth_limited_search(
            graph=self.graph,
            start=start,
            goal=goal,
            limit=limit,
            cycle_prevention=self.cycle_prevention,
        )


def _recursive_dls_core(
    current_node: str,
    goal: str,
    remaining_limit: int,
    graph: Dict[str, List[str]],
    path: List[str],
    steps: List[DLSStepRecord],
    cutoff_nodes: List[str],
    deepest_nodes: List[str],
    state: Dict[str, Any],
    cycle_prevention: bool = True,
) -> Union[List[str], str]:
    """Internal recursive worker executing the bounded depth-first traversal.

    Decreases remaining_limit by 1 on each recursive invocation to successors.
    Handles the three fundamental return states: Success (path), Cutoff, Failure.

    Args:
        current_node: State currently being evaluated.
        goal: Target destination state.
        remaining_limit: Depth budget remaining for expanding children.
        graph: Adjacency list representation of graph.
        path: Ordered list of nodes along the active branch up to current_node.
        steps: Mutable audit list collecting step records.
        cutoff_nodes: Mutable list recording nodes where cutoff was triggered.
        deepest_nodes: Mutable list tracking nodes reached at max depth (Optional Challenge).
        state: Mutable dictionary holding telemetry counters ('step_count', 'nodes_generated',
               'nodes_expanded', 'max_depth', 'cutoff_occurred').
        cycle_prevention: Whether to prune ancestors already in active path.

    Returns:
        List[str] representing the solution path if found, or 'Cutoff', or 'Failure'.
    """
    state["nodes_generated"] += 1
    current_depth = len(path) - 1

    # Update deepest node tracking (Optional Challenge)
    if current_depth > state["max_depth"]:
        state["max_depth"] = current_depth
        deepest_nodes.clear()
        deepest_nodes.append(current_node)
    elif current_depth == state["max_depth"]:
        if current_node not in deepest_nodes:
            deepest_nodes.append(current_node)

    # State 1: Goal Test Evaluation
    if current_node == goal:
        state["step_count"] += 1
        steps.append(
            DLSStepRecord(
                step_number=state["step_count"],
                current_node=current_node,
                current_depth=current_depth,
                remaining_limit=remaining_limit,
                action="GOAL_TEST_PASS",
                path_to_node=list(path),
                result_returned=SUCCESS,
                details=f"Target goal '{goal}' discovered at depth {current_depth} within limit.",
            )
        )
        return list(path)

    # State 2: Depth Limit Boundary Check (Cutoff Trigger)
    if remaining_limit == 0:
        state["cutoff_occurred"] = True
        if current_node not in cutoff_nodes:
            cutoff_nodes.append(current_node)
        state["step_count"] += 1
        steps.append(
            DLSStepRecord(
                step_number=state["step_count"],
                current_node=current_node,
                current_depth=current_depth,
                remaining_limit=0,
                action="CUTOFF_HIT",
                path_to_node=list(path),
                result_returned=CUTOFF,
                details=f"Depth limit reached at depth {current_depth}; successor expansion halted.",
            )
        )
        return CUTOFF

    # State 3: Node Expansion and Successor Recursion
    neighbors = graph.get(current_node, [])
    if not neighbors:
        state["step_count"] += 1
        steps.append(
            DLSStepRecord(
                step_number=state["step_count"],
                current_node=current_node,
                current_depth=current_depth,
                remaining_limit=remaining_limit,
                action="LEAF_FAILURE",
                path_to_node=list(path),
                result_returned=FAILURE,
                details=f"Node '{current_node}' has no outgoing edges (leaf).",
            )
        )
        return FAILURE

    state["nodes_expanded"] += 1
    state["step_count"] += 1
    steps.append(
        DLSStepRecord(
            step_number=state["step_count"],
            current_node=current_node,
            current_depth=current_depth,
            remaining_limit=remaining_limit,
            action="EXPAND_CHILDREN",
            path_to_node=list(path),
            result_returned="CONTINUE",
            details=f"Expanding {len(neighbors)} successor(s): {neighbors} with limit budget {remaining_limit - 1}.",
        )
    )

    cutoff_occurred_in_subtree = False

    for child in neighbors:
        # Prevent active branch cycles if enabled
        if cycle_prevention and child in path:
            state["step_count"] += 1
            steps.append(
                DLSStepRecord(
                    step_number=state["step_count"],
                    current_node=child,
                    current_depth=current_depth + 1,
                    remaining_limit=remaining_limit - 1,
                    action="CYCLE_SKIP",
                    path_to_node=path + [child],
                    result_returned="SKIP",
                    details=f"Pruned cyclic edge '{current_node}' -> '{child}' already in active branch.",
                )
            )
            continue

        # Recursive invocation with decremented limit parameter
        result = _recursive_dls_core(
            current_node=child,
            goal=goal,
            remaining_limit=remaining_limit - 1,
            graph=graph,
            path=path + [child],
            steps=steps,
            cutoff_nodes=cutoff_nodes,
            deepest_nodes=deepest_nodes,
            state=state,
            cycle_prevention=cycle_prevention,
        )

        if result == CUTOFF:
            cutoff_occurred_in_subtree = True
        elif result != FAILURE:
            # Result is the verified solution path list
            return result

    # Evaluate aggregated outcome of all children
    if cutoff_occurred_in_subtree:
        state["step_count"] += 1
        steps.append(
            DLSStepRecord(
                step_number=state["step_count"],
                current_node=current_node,
                current_depth=current_depth,
                remaining_limit=remaining_limit,
                action="BACKTRACK_CUTOFF",
                path_to_node=list(path),
                result_returned=CUTOFF,
                details=f"Backtracking from '{current_node}' with Cutoff (at least one descendant branch truncated).",
            )
        )
        return CUTOFF

    state["step_count"] += 1
    steps.append(
        DLSStepRecord(
            step_number=state["step_count"],
            current_node=current_node,
            current_depth=current_depth,
            remaining_limit=remaining_limit,
            action="BACKTRACK_FAIL",
            path_to_node=list(path),
            result_returned=FAILURE,
            details=f"All branches under '{current_node}' completely explored with Failure; no cutoff hit.",
        )
    )
    return FAILURE


def depth_limited_search(
    graph: Union[str, Dict[str, Sequence[str]]],
    start: str,
    goal: str,
    limit: int,
    cycle_prevention: bool = True,
) -> DLSResult:
    """Execute Depth-Limited Search and return complete telemetry encapsulated in DLSResult.

    Args:
        graph: Graph adjacency dictionary or parseable string representation.
        start: Start node label.
        goal: Target destination goal node label.
        limit: Non-negative maximum search depth integer.
        cycle_prevention: Whether to avoid exploring active path ancestors (default: True).

    Returns:
        DLSResult object containing status, path, depth, telemetry, and execution steps.

    Raises:
        ValueError: If limit is negative, or if start node is missing from graph.
    """
    if limit < 0:
        raise ValueError(f"Depth limit must be non-negative (>= 0), got limit={limit}.")

    normalized = parse_graph_input(graph) if isinstance(graph, str) else normalize_graph(graph)

    start_str = str(start).strip()
    goal_str = str(goal).strip()

    if not start_str:
        raise ValueError("Start node cannot be empty.")
    if not goal_str:
        raise ValueError("Goal node cannot be empty.")

    if start_str not in normalized:
        raise ValueError(
            f"Start node '{start_str}' does not exist in the graph. "
            f"Available nodes: {sorted(list(normalized.keys()))}"
        )

    # Ensure goal node is represented in normalized graph keys
    if goal_str not in normalized:
        normalized[goal_str] = []

    steps: List[DLSStepRecord] = []
    cutoff_nodes: List[str] = []
    deepest_nodes: List[str] = []
    state: Dict[str, Any] = {
        "step_count": 0,
        "nodes_generated": 0,
        "nodes_expanded": 0,
        "max_depth": 0,
        "cutoff_occurred": False,
    }

    t0 = time.perf_counter()
    outcome = _recursive_dls_core(
        current_node=start_str,
        goal=goal_str,
        remaining_limit=limit,
        graph=normalized,
        path=[start_str],
        steps=steps,
        cutoff_nodes=cutoff_nodes,
        deepest_nodes=deepest_nodes,
        state=state,
        cycle_prevention=cycle_prevention,
    )
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    if isinstance(outcome, list):
        final_status = SUCCESS
        final_path: Optional[List[str]] = outcome
        solution_depth: Optional[int] = len(outcome) - 1
    elif outcome == CUTOFF:
        final_status = CUTOFF
        final_path = None
        solution_depth = None
    else:
        final_status = FAILURE
        final_path = None
        solution_depth = None

    return DLSResult(
        status=final_status,
        path=final_path,
        depth=solution_depth,
        limit=limit,
        start_node=start_str,
        goal_node=goal_str,
        graph=normalized,
        max_depth_reached=state["max_depth"],
        deepest_nodes=sorted(deepest_nodes),
        cutoff_nodes=sorted(cutoff_nodes),
        nodes_expanded=state["nodes_expanded"],
        nodes_generated=state["nodes_generated"],
        cutoff_occurred=state["cutoff_occurred"],
        steps=steps,
        execution_time_ms=elapsed_ms,
    )


def dls(
    graph: Union[str, Dict[str, Sequence[str]]],
    start: str,
    goal: str,
    limit: int,
    cycle_prevention: bool = True,
) -> Union[List[str], str]:
    """Convenience functional interface matching university practical signature and output.

    Returns the solution path if found, 'Cutoff' if goal lies beyond the limit,
    or 'Failure' if goal is unreachable.

    Args:
        graph: Graph adjacency structure.
        start: Label of start node.
        goal: Label of goal node.
        limit: Non-negative maximum search depth.
        cycle_prevention: Whether to prune active-branch cycles.

    Returns:
        List[str] (e.g. ['1', '2', '4']) on Success,
        'Cutoff' on Cutoff,
        'Failure' on Failure.

    Example:
        >>> graph = {'1': ['2', '3'], '2': ['4'], '3': [], '4': []}
        >>> dls(graph, start='1', goal='4', limit=2)
        ['1', '2', '4']
        >>> dls(graph, start='1', goal='4', limit=1)
        'Cutoff'
        >>> dls(graph, start='1', goal='99', limit=2)
        'Failure'
    """
    result = depth_limited_search(
        graph=graph,
        start=start,
        goal=goal,
        limit=limit,
        cycle_prevention=cycle_prevention,
    )
    if result.is_success() and result.path is not None:
        return result.path
    return result.status


def visualize_dls(
    result: DLSResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Path:
    """Generate and save a visual graph rendering of Depth-Limited Search.

    Highlights:
    - Nodes explored within the depth limit.
    - Cutoff boundary nodes where expansion was halted.
    - Solution path (if status is 'Success').
    - Deepest nodes reached (Optional Challenge).

    Args:
        result: DLSResult from a completed DLS run.
        output_path: Destination filepath for image (e.g. reports/dls_search.png).
        show: Whether to display interactive window via plt.show().
        title: Custom chart title.

    Returns:
        Path to the saved image file.
    """
    import matplotlib

    matplotlib.use("Agg")  # Headless rendering
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"dls_search_{result.start_node}_to_{result.goal_node}_limit{result.limit}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    for u, neighbors in result.graph.items():
        g.add_node(u)
        for v in neighbors:
            g.add_edge(u, v)

    # Compute node depths from start_node via shortest unweighted path for layout
    node_depths: Dict[str, int] = {}
    for node in g.nodes():
        if nx.has_path(g, result.start_node, node):
            node_depths[node] = nx.shortest_path_length(g, result.start_node, node)
        else:
            node_depths[node] = -1

    # Hierarchical tree or spring layout
    try:
        # Layered multipartite layout based on distance
        for n, d in node_depths.items():
            g.nodes[n]["layer"] = d if d >= 0 else 99
        pos = nx.multipartite_layout(g, subset_key="layer")
    except Exception:
        pos = nx.spring_layout(g, seed=42)

    solution_edges: Set[Tuple[str, str]] = set()
    if result.is_success() and result.path:
        for i in range(len(result.path) - 1):
            solution_edges.add((result.path[i], result.path[i + 1]))

    # Node coloring
    node_colors: List[str] = []
    node_border_colors: List[str] = []
    node_sizes: List[int] = []

    for node in g.nodes():
        if node == result.start_node and node == result.goal_node and result.is_success():
            node_colors.append("#2ECC71")  # Green
            node_border_colors.append("#196F3D")
            node_sizes.append(1100)
        elif node == result.start_node:
            node_colors.append("#3498DB")  # Blue Start
            node_border_colors.append("#1B4F72")
            node_sizes.append(1100)
        elif node == result.goal_node and result.is_success():
            node_colors.append("#2ECC71")  # Green Success Goal
            node_border_colors.append("#196F3D")
            node_sizes.append(1100)
        elif node == result.goal_node:
            node_colors.append("#F39C12")  # Orange Goal
            node_border_colors.append("#B9770E")
            node_sizes.append(1100)
        elif result.is_success() and result.path and node in result.path:
            node_colors.append("#A9DFBF")  # Light Green Path
            node_border_colors.append("#1E8449")
            node_sizes.append(900)
        elif node in result.cutoff_nodes:
            node_colors.append("#E74C3C")  # Red Cutoff Node
            node_border_colors.append("#922B21")
            node_sizes.append(950)
        elif node in result.deepest_nodes:
            node_colors.append("#9B59B6")  # Purple Deepest
            node_border_colors.append("#6C3483")
            node_sizes.append(900)
        elif node_depths.get(node, -1) != -1 and node_depths[node] <= result.limit:
            node_colors.append("#EBF5FB")  # Explored within limit
            node_border_colors.append("#85929E")
            node_sizes.append(750)
        else:
            node_colors.append("#EAEDED")  # Unexplored / Beyond limit
            node_border_colors.append("#BDC3C7")
            node_sizes.append(650)

    # Edge styling
    edge_colors: List[str] = []
    edge_widths: List[float] = []
    edge_styles: List[str] = []

    for u, v in g.edges():
        if (u, v) in solution_edges:
            edge_colors.append("#27AE60")
            edge_widths.append(3.2)
            edge_styles.append("solid")
        else:
            edge_colors.append("#BDC3C7")
            edge_widths.append(1.2)
            edge_styles.append("solid")

    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)

    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=2.0,
        node_size=node_sizes,
    )

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_labels(
        g,
        pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
        font_family="sans-serif",
    )

    # Legend patches
    legend_elements = [
        mpatches.Patch(color="#3498DB", label=f"Start Node ('{result.start_node}')"),
        mpatches.Patch(
            color="#2ECC71" if result.is_success() else "#F39C12",
            label=f"Goal Node ('{result.goal_node}' - {'FOUND' if result.is_success() else 'NOT FOUND'})",
        ),
        mpatches.Patch(color="#E74C3C", label=f"Cutoff Boundary (Limit={result.limit})"),
        mpatches.Patch(color="#9B59B6", label=f"Deepest Node(s) Reached: {', '.join(result.deepest_nodes)}"),
    ]
    if result.is_success():
        legend_elements.append(mpatches.Patch(color="#27AE60", label=f"Solution Path: {result.format_path()}"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5, framealpha=0.92)

    chart_title = title or (
        f"Depth-Limited Search (DLS) — {result.format_result()}\n"
        f"Start='{result.start_node}', Goal='{result.goal_node}', Limit={result.limit} | "
        f"Generated={result.nodes_generated}, MaxDepth={result.max_depth_reached}"
    )
    ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return output_path.resolve()


def get_dls_viva_reference() -> Dict[str, Any]:
    """Return comprehensive Viva Voce theory notes, questions, and comparison tables.

    Returns:
        Structured dictionary containing theoretical concepts, syllabus viva topics,
        comparisons with BFS/DFS/IDS, and common interview questions.
    """
    return {
        "title": "Depth-Limited Search (DLS) Viva Voce Reference Guide",
        "core_concept": "Bounded Depth-First Search with Tri-State Outcomes",
        "viva_topics": {
            "infinite_state_spaces": {
                "definition": (
                    "A state space with an infinite number of states or unbounded paths (e.g., "
                    "integers with operations +1/*2, continuous navigation, or unbounded recursive generation)."
                ),
                "dfs_failure_mode": (
                    "Standard DFS plunges along a single infinite branch to infinity (m = infinity). "
                    "It never backtracks, failing to discover a goal even if one exists at depth 1 on "
                    "a neighbouring branch."
                ),
                "dls_resolution": (
                    "DLS enforces a hard cutoff limit 'l'. Exploration halts at depth 'l', transforming "
                    "an unbounded infinite exploration into a strictly finite search of depth <= l."
                ),
            },
            "drawback_of_standard_dfs": {
                "incompleteness": (
                    "Standard DFS is incomplete in infinite state spaces or finite state spaces with cycles "
                    "(unless graph search tracking is maintained)."
                ),
                "non_optimality": (
                    "DFS is not optimal. It returns the first goal found on its deepest path, which may have "
                    "cost/length much greater than a shallow alternative."
                ),
                "recursion_overflow": (
                    "In deep or infinite graphs, standard recursive DFS crashes with Python's RecursionError "
                    "(exceeding maximum recursion limit)."
                ),
            },
            "cutoff_vs_failure": {
                "cutoff_meaning": (
                    "Indicates the goal was NOT found within depth limit 'l', BUT at least one node at "
                    "depth 'l' was truncated before its children could be expanded. The goal MIGHT exist "
                    "at depth > l."
                ),
                "failure_meaning": (
                    "Indicates the entire reachable state space was exhaustively traversed without finding "
                    "the goal AND without hitting any depth cutoffs. Proves definitively that the goal "
                    "does NOT exist in this connected component."
                ),
                "importance_for_ids": (
                    "Iterative Deepening Search (IDS) relies strictly on this distinction:\n"
                    "- If DLS returns Cutoff: IDS increases limit (l = l + 1) and continues.\n"
                    "- If DLS returns Failure: IDS halts immediately because searching deeper cannot "
                    "discover nonexistent states.\n"
                    "- If DLS returns Success: IDS halts and returns the optimal shallowest path."
                ),
            },
        },
        "complexity_comparison": {
            "DLS": {
                "time": "O(b^l)",
                "space": "O(b * l)",
                "complete": "Yes, if goal depth d <= l; Incomplete if d > l",
                "optimal": "No (may find deeper path before shallow one)",
            },
            "DFS": {
                "time": "O(b^m)",
                "space": "O(b * m)",
                "complete": "No (fails in infinite spaces or cycles)",
                "optimal": "No",
            },
            "BFS": {
                "time": "O(b^d)",
                "space": "O(b^d) [Exponential Memory Bottleneck!]",
                "complete": "Yes (if b is finite)",
                "optimal": "Yes (for uniform step costs)",
            },
            "IDS": {
                "time": "O(b^d)",
                "space": "O(b * d) [Linear Memory Advantage of DFS!]",
                "complete": "Yes (if b is finite)",
                "optimal": "Yes (for uniform step costs)",
            },
        },
        "key_questions": [
            {
                "question": "What is the primary motivation for introducing Depth-Limited Search?",
                "answer": (
                    "To eliminate the infinite-path failure mode of DFS in infinite state spaces or graphs "
                    "with loops by establishing a maximum depth limit constraint."
                ),
            },
            {
                "question": "What are the three possible return states of DLS and what does each signify?",
                "answer": (
                    "1. Success: Solution path found at depth <= l.\n"
                    "2. Cutoff: Goal not found, but search was truncated at limit l (goal may exist deeper).\n"
                    "3. Failure: Entire state space exhausted without finding goal; goal does not exist."
                ),
            },
            {
                "question": "Why does DLS fail to be complete?",
                "answer": (
                    "Because if the user selects a depth limit l smaller than the actual goal depth d (l < d), "
                    "the algorithm will never reach the goal and will return Cutoff instead of finding it."
                ),
            },
            {
                "question": "How does Iterative Deepening Search (IDS) overcome the incompleteness of DLS?",
                "answer": (
                    "IDS systematically runs DLS with incrementally increasing limits (l = 0, 1, 2, ...). "
                    "It combines the completeness and optimality of BFS with the O(b*d) linear memory of DFS."
                ),
            },
            {
                "question": "Why is the overhead of repeated node regeneration in IDS negligible?",
                "answer": (
                    "Because in an exponential tree, the bottom level contains the vast majority of nodes "
                    "(fraction (b-1)/b). Regenerating upper levels adds only a small constant factor overhead: "
                    "O(b^d) overall."
                ),
            },
        ],
    }
