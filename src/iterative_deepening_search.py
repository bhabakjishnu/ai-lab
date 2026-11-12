"""Iterative Deepening Search (IDS) Implementation for State-Space Exploration.

This module implements Iterative Deepening Search (IDS) — also known as Iterative
Deepening Depth-First Search (IDDFS) — by repeatedly invoking Depth-Limited Search
(DLS) with incrementally increasing depth limits ``l = 0, 1, 2, ...``, adhering to
Russell & Norvig's *Artificial Intelligence: A Modern Approach* (AIMA 3rd/4th ed.,
Chapter 3: Solving Problems by Searching).

Theoretical Background & AI Concepts:
- Optimal + Complete Uninformed Search with Linear Memory:
    Breadth-First Search (BFS) guarantees completeness and shortest-path optimality in
    unweighted graphs, but requires exponential memory O(b^d) to store the entire
    explored frontier. Depth-First Search (DFS) uses linear memory O(b * m), but is
    incomplete in infinite or cyclic state spaces and non-optimal.
    Iterative Deepening Search combines the best of both worlds:
    1. **Completeness**: Guaranteed to find a solution if one exists (for finite b).
    2. **Optimality**: Guaranteed to find the shallowest (shortest-hop) path first,
       identical to BFS on unweighted graphs.
    3. **Memory Optimization**: Uses only O(b * d) space on the recursion stack,
       avoiding BFS's exponential memory bottleneck.
- Wrapping DLS in an Incrementing Loop:
    At each iteration ``l = 0, 1, 2, ...``, IDS invokes ``depth_limited_search``:
    - If DLS returns ``Success``: IDS terminates and returns the optimal path.
    - If DLS returns ``Cutoff``: At least one branch was truncated at depth ``l``;
      IDS increments ``l <- l + 1`` and continues.
    - If DLS returns ``Failure``: The reachable state space was exhaustively explored
      without hitting any depth cutoff; IDS terminates immediately, proving the goal
      is unreachable.
- Redundant Node Generation & Asymptotic Overhead:
    While IDS regenerates upper-level nodes across iterations, most nodes in a tree
    with branching factor ``b > 1`` reside at the bottom level ``d``. The total number
    of nodes generated across all iterations is bounded by ``O(b^d)``, adding only a
    small constant multiplicative factor ``b / (b - 1)`` (e.g., ~11% for ``b = 10``).
- Cycle Handling:
    By leveraging active-branch ancestor pruning (``cycle_prevention=True``), IDS
    prevents infinite loops on cyclic graphs while keeping memory strictly ``O(b * d)``
    and preserving shortest-path optimality.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.breadth_first_search import breadth_first_search
from src.depth_limited_search import (
    CUTOFF,
    FAILURE,
    SUCCESS,
    DLSResult,
    depth_limited_search,
    normalize_graph,
    parse_graph_input,
)

# Canonical benchmark graphs for Question 17 testing, demonstrations, and education
SAMPLE_IDS_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4", "5"],
    "3": ["6"],
    "4": [],
    "5": [],
    "6": [],
}
"""Canonical sample graph with goal at depth 2 (e.g., Start='1', Goal='4'):
Depth 0: 1
Depth 1: 2, 3
Depth 2: 4, 5, 6
Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found ['1', '2', '4'].
"""

SHORTCUT_IDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D"],
    "D": ["E"],
    "E": ["G"],
    "C": ["G"],
    "G": [],
}
"""Demonstrates why IDS finds the shortest path (A -> C -> G, depth 2) like BFS,
whereas standard DFS plunges down the left branch (A -> B -> D -> E -> G, depth 4).
"""

CYCLIC_IDS_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["1", "4"],
    "3": ["1", "5"],
    "4": ["2", "6"],
    "5": ["3"],
    "6": [],
}
"""Graph containing bidirectional cycles to verify cycle-safe termination."""

DISCONNECTED_IDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": [],
    "D": [],
    "X": ["Y"],
    "Y": [],
}
"""Disconnected graph to verify exhaustive graph Failure termination."""

BINARY_TREE_IDS_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H", "I"],
    "E": ["J", "K"],
    "F": ["L", "M"],
    "G": ["N", "O"],
    "H": [],
    "I": [],
    "J": [],
    "K": [],
    "L": [],
    "M": [],
    "N": [],
    "O": [],
}
"""Complete binary tree of depth 3 (15 nodes) for studying redundant node generation."""

_ENTRY_STEP_ACTIONS: Set[str] = {
    "GOAL_TEST_PASS",
    "CUTOFF_HIT",
    "LEAF_FAILURE",
    "EXPAND_CHILDREN",
}


@dataclass(frozen=True)
class IDSIterationRecord:
    """Immutable telemetry record for a single Depth-Limited Search iteration inside IDS.

    Attributes:
        limit: Depth limit ``l`` used for this DLS invocation.
        status: Tri-state outcome returned by DLS ('Success', 'Cutoff', or 'Failure').
        path: Solution path if discovered at this limit, else None.
        nodes_expanded: Number of internal nodes whose successors were expanded.
        nodes_generated: Number of node visits (recursive entries) made in this iteration.
        expanded_nodes: Ordered sequence of node labels expanded during this iteration.
        visited_nodes: Ordered sequence of node labels visited during this iteration.
        cutoff_nodes: Nodes truncated at the depth boundary ``limit``.
        max_depth_reached: Deepest level reached during this iteration.
        execution_time_ms: Wall-clock duration of this iteration in milliseconds.
        dls_result: Full underlying ``DLSResult`` for fine-grained step inspection.
    """

    limit: int
    status: str
    path: Optional[List[str]]
    nodes_expanded: int
    nodes_generated: int
    expanded_nodes: List[str]
    visited_nodes: List[str]
    cutoff_nodes: List[str]
    max_depth_reached: int
    execution_time_ms: float
    dls_result: DLSResult

    def format_outcome(self) -> str:
        """Return the practical outcome summary for this depth limit iteration.

        Returns:
            String formatted as 'Limit <l>: Found <path>', 'Limit <l>: Cutoff',
            or 'Limit <l>: Failure'.
        """
        if self.status == SUCCESS and self.path is not None:
            return f"Limit {self.limit}: Found {self.path}"
        return f"Limit {self.limit}: {self.status}"

    def format_row(self) -> str:
        """Format iteration telemetry as a single tabular row.

        Returns:
            Formatted ASCII table row string.
        """
        expanded_str = ", ".join(self.expanded_nodes) if self.expanded_nodes else "(none)"
        visited_str = ", ".join(self.visited_nodes) if self.visited_nodes else "(none)"
        return (
            f"{self.limit:<5} | {self.status:<8} | {self.nodes_expanded:<8} | "
            f"{self.nodes_generated:<9} | {expanded_str:<22} | {visited_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert this iteration record into a JSON-serializable dictionary."""
        return {
            "limit": self.limit,
            "status": self.status,
            "path": self.path,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "expanded_nodes": list(self.expanded_nodes),
            "visited_nodes": list(self.visited_nodes),
            "cutoff_nodes": list(self.cutoff_nodes),
            "max_depth_reached": self.max_depth_reached,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class IDSResult:
    """Encapsulates the complete outcome, iteration history, and telemetry of IDS.

    Attributes:
        status: Final IDS outcome ('Success', 'Failure', or 'Cutoff' if capped by max_depth).
        path: Shortest solution path from start to goal if found, else None.
        depth: Optimal solution depth (edge count) if found, else None.
        start_node: Root node label where search originated.
        goal_node: Target destination node label.
        graph: Normalized adjacency list representation of the graph.
        final_limit: Last depth limit attempted before termination.
        max_depth_cap: Optional upper bound placed on depth iterations.
        iterations: Chronological list of ``IDSIterationRecord`` objects for limits ``0..final_limit``.
        total_nodes_expanded: Sum of node expansions across all iterations (Optional Challenge).
        total_nodes_generated: Sum of node visits/generations across all iterations.
        peak_stack_depth: Maximum active recursion path length (``max_depth + 1``), reflecting O(d) memory.
        execution_time_ms: Total wall-clock execution time in milliseconds.
    """

    status: str
    path: Optional[List[str]]
    depth: Optional[int]
    start_node: str
    goal_node: str
    graph: Dict[str, List[str]]
    final_limit: int
    max_depth_cap: Optional[int]
    iterations: List[IDSIterationRecord] = field(default_factory=list)
    total_nodes_expanded: int = 0
    total_nodes_generated: int = 0
    peak_stack_depth: int = 1
    execution_time_ms: float = 0.0

    @property
    def final_iteration_generated(self) -> int:
        """Number of node generations performed in the final depth iteration."""
        return self.iterations[-1].nodes_generated if self.iterations else 0

    @property
    def redundant_regenerations(self) -> int:
        """Count of node visits performed in earlier iterations before the final depth."""
        if len(self.iterations) <= 1:
            return 0
        return sum(it.nodes_generated for it in self.iterations[:-1])

    def is_success(self) -> bool:
        """Return True if the goal was found."""
        return self.status == SUCCESS

    def is_failure(self) -> bool:
        """Return True if the graph was exhausted without finding the goal."""
        return self.status == FAILURE

    def is_cutoff(self) -> bool:
        """Return True if an optional max_depth cap terminated the search prematurely."""
        return self.status == CUTOFF

    def format_sequence(self) -> str:
        """Format the chronological sequence of limits tried matching practical output.

        Example:
            ``Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found ['1', '2', '4'].``

        Returns:
            Period-separated sequence string across all attempted limits.
        """
        if not self.iterations:
            return f"Result: {self.status}"
        return ". ".join(it.format_outcome() for it in self.iterations) + "."

    def format_path(self, delimiter: str = " -> ") -> str:
        """Return formatted solution path or 'No Path Found'."""
        if self.path:
            return delimiter.join(self.path)
        return "No Path Found"

    def format_iteration_table(self) -> str:
        """Return an ASCII breakdown table of nodes expanded and visited per depth limit.

        Returns:
            Formatted multi-line table string.
        """
        headers = (
            f"{'Limit':<5} | {'Outcome':<8} | {'Expanded':<8} | "
            f"{'Generated':<9} | {'Expanded Nodes':<22} | {'Visited Order'}"
        )
        separator = "-" * max(len(headers), 85)
        rows = [headers, separator]
        for it in self.iterations:
            rows.append(it.format_row())
        return "\n".join(rows)

    def format_summary(self) -> str:
        """Return a comprehensive multi-line summary of IDS execution and complexity metrics."""
        lines = [
            "=" * 78,
            "ITERATIVE DEEPENING SEARCH (IDS) EXECUTION SUMMARY",
            "=" * 78,
            f"Search Problem              : Start='{self.start_node}' -> Goal='{self.goal_node}'",
            f"Final Outcome               : {self.status.upper()}",
            f"Sequence of Limits Tried    : {self.format_sequence()}",
            f"Solution Path               : {self.format_path()}",
            f"Optimal Solution Depth      : {self.depth if self.depth is not None else 'N/A'}",
            f"Iterations Executed         : {len(self.iterations)} (Limits 0..{self.final_limit})",
            f"Total Node Expansions       : {self.total_nodes_expanded} (Optional Challenge)",
            f"Total Node Generations      : {self.total_nodes_generated}",
            f"Redundant Prior Generations : {self.redundant_regenerations}",
            f"Peak Recursion Stack Depth  : {self.peak_stack_depth} nodes (O(b * d) memory)",
            f"Execution Wall Time         : {self.execution_time_ms:.4f} ms",
            "=" * 78,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full IDS result into a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "path": self.path,
            "depth": self.depth,
            "start_node": self.start_node,
            "goal_node": self.goal_node,
            "final_limit": self.final_limit,
            "max_depth_cap": self.max_depth_cap,
            "total_nodes_expanded": self.total_nodes_expanded,
            "total_nodes_generated": self.total_nodes_generated,
            "redundant_regenerations": self.redundant_regenerations,
            "peak_stack_depth": self.peak_stack_depth,
            "execution_time_ms": self.execution_time_ms,
            "sequence": self.format_sequence(),
            "iterations": [it.to_dict() for it in self.iterations],
        }


def _extract_iteration_nodes(dls_result: DLSResult) -> Tuple[List[str], List[str]]:
    """Extract ordered expanded and visited node lists from a DLS execution trace.

    Args:
        dls_result: Completed ``DLSResult`` for one depth limit iteration.

    Returns:
        Tuple ``(expanded_nodes, visited_nodes)`` preserving exact DFS chronological order.
    """
    expanded_nodes: List[str] = []
    visited_nodes: List[str] = []
    for step in dls_result.steps:
        if step.action == "EXPAND_CHILDREN":
            expanded_nodes.append(step.current_node)
        if step.action in _ENTRY_STEP_ACTIONS:
            visited_nodes.append(step.current_node)
    return expanded_nodes, visited_nodes


def iterative_deepening_search(
    graph: Union[str, Dict[str, Sequence[str]]],
    start: str,
    goal: str,
    cycle_prevention: bool = True,
    max_depth: Optional[int] = None,
    verbose: bool = False,
) -> IDSResult:
    """Execute Iterative Deepening Search (IDS) by wrapping DLS in an incrementing loop.

    Starts at depth limit ``l = 0`` and calls ``depth_limited_search`` with ``l = 0, 1, 2, ...``
    until one of the following stopping conditions is reached:
    - DLS returns ``Success``: returns the shallowest (shortest) solution path.
    - DLS returns ``Failure``: the reachable graph is exhausted; returns ``Failure``.
    - ``max_depth`` is specified and ``limit == max_depth``: returns ``Cutoff``.

    Args:
        graph: Adjacency list dictionary or parseable string representation.
        start: Start node label.
        goal: Target goal node label.
        cycle_prevention: If True, prunes ancestors in the active recursion path to
            guarantee termination on cyclic graphs while maintaining O(b * d) space.
        max_depth: Optional non-negative maximum depth limit cap. If None, runs until
            the goal is found or the reachable graph is exhausted.
        verbose: If True, prints the nodes expanded and outcome at each depth iteration.

    Returns:
        ``IDSResult`` encapsulating iteration telemetry, path, and total expansions.

    Raises:
        ValueError: If ``max_depth`` is negative, graph is invalid, or start/goal are invalid.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError(f"max_depth must be non-negative (>= 0), got {max_depth}.")

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

    # Ensure goal node exists in normalized adjacency dict for consistent reporting
    if goal_str not in normalized:
        normalized[goal_str] = []

    iterations: List[IDSIterationRecord] = []
    total_expanded = 0
    total_generated = 0
    peak_depth = 0
    final_status = FAILURE
    final_path: Optional[List[str]] = None
    solution_depth: Optional[int] = None

    t0 = time.perf_counter()
    limit = 0

    while True:
        dls_res = depth_limited_search(
            graph=normalized,
            start=start_str,
            goal=goal_str,
            limit=limit,
            cycle_prevention=cycle_prevention,
        )

        expanded_nodes, visited_nodes = _extract_iteration_nodes(dls_res)
        total_expanded += dls_res.nodes_expanded
        total_generated += dls_res.nodes_generated
        if dls_res.max_depth_reached > peak_depth:
            peak_depth = dls_res.max_depth_reached

        record = IDSIterationRecord(
            limit=limit,
            status=dls_res.status,
            path=dls_res.path,
            nodes_expanded=dls_res.nodes_expanded,
            nodes_generated=dls_res.nodes_generated,
            expanded_nodes=expanded_nodes,
            visited_nodes=visited_nodes,
            cutoff_nodes=list(dls_res.cutoff_nodes),
            max_depth_reached=dls_res.max_depth_reached,
            execution_time_ms=dls_res.execution_time_ms,
            dls_result=dls_res,
        )
        iterations.append(record)

        if verbose:
            expanded_display = expanded_nodes if expanded_nodes else ["(none)"]
            print(
                f"Depth Limit {limit}: {record.format_outcome()} | "
                f"Expanded ({dls_res.nodes_expanded}): {expanded_display} | "
                f"Visited ({dls_res.nodes_generated}): {visited_nodes}"
            )

        if dls_res.is_success():
            final_status = SUCCESS
            final_path = dls_res.path
            solution_depth = dls_res.depth
            break

        if dls_res.is_failure():
            final_status = FAILURE
            break

        if max_depth is not None and limit >= max_depth:
            final_status = CUTOFF
            break

        limit += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return IDSResult(
        status=final_status,
        path=final_path,
        depth=solution_depth,
        start_node=start_str,
        goal_node=goal_str,
        graph=normalized,
        final_limit=limit,
        max_depth_cap=max_depth,
        iterations=iterations,
        total_nodes_expanded=total_expanded,
        total_nodes_generated=total_generated,
        peak_stack_depth=peak_depth + 1,
        execution_time_ms=elapsed_ms,
    )


def ids(
    graph: Union[str, Dict[str, Sequence[str]]],
    start: str,
    goal: str,
    cycle_prevention: bool = True,
    max_depth: Optional[int] = None,
) -> Optional[List[str]]:
    """Convenience functional interface returning the shortest path or None.

    Args:
        graph: Graph adjacency structure or parseable string.
        start: Label of start node.
        goal: Label of goal node.
        cycle_prevention: Whether to prune active-branch cycles.
        max_depth: Optional depth limit cap.

    Returns:
        Shortest path ``List[str]`` if found, else ``None``.

    Example:
        >>> graph = {'1': ['2', '3'], '2': ['4', '5'], '3': ['6'], '4': [], '5': [], '6': []}
        >>> ids(graph, start='1', goal='4')
        ['1', '2', '4']
    """
    result = iterative_deepening_search(
        graph=graph,
        start=start,
        goal=goal,
        cycle_prevention=cycle_prevention,
        max_depth=max_depth,
    )
    return result.path


class IterativeDeepeningSearch:
    """Object-oriented engine for Iterative Deepening Search (IDS)."""

    def __init__(
        self,
        graph: Union[str, Dict[str, Sequence[str]]],
        cycle_prevention: bool = True,
        max_depth: Optional[int] = None,
    ) -> None:
        """Initialize the IDS solver with a normalized graph.

        Args:
            graph: Adjacency list dictionary or parseable string representation.
            cycle_prevention: Prune ancestors in active recursion path (default: True).
            max_depth: Optional default depth cap across searches.
        """
        self.graph = parse_graph_input(graph) if isinstance(graph, str) else normalize_graph(graph)
        self.cycle_prevention = cycle_prevention
        self.max_depth = max_depth

    def search(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
        verbose: bool = False,
    ) -> IDSResult:
        """Run Iterative Deepening Search from start to goal.

        Args:
            start: Start node label.
            goal: Target goal node label.
            max_depth: Override default max_depth cap for this run.
            verbose: Print per-iteration node expansion details.

        Returns:
            ``IDSResult`` encapsulating full search telemetry and path.
        """
        effective_cap = max_depth if max_depth is not None else self.max_depth
        return iterative_deepening_search(
            graph=self.graph,
            start=start,
            goal=goal,
            cycle_prevention=self.cycle_prevention,
            max_depth=effective_cap,
            verbose=verbose,
        )


def compare_ids_with_bfs(
    graph: Union[str, Dict[str, Sequence[str]]],
    start: str,
    goal: str,
    cycle_prevention: bool = True,
) -> Dict[str, Any]:
    """Compare IDS against BFS on optimality (path depth) and memory footprint.

    Demonstrates empirically that IDS discovers the same minimum-hop shortest path
    depth as BFS while keeping peak active memory linear O(b * d) instead of storing
    an exponential O(b^d) frontier and visited table.

    Args:
        graph: Graph adjacency dictionary or parseable string.
        start: Start node label.
        goal: Target goal node label.
        cycle_prevention: Cycle-prevention setting for IDS.

    Returns:
        Structured comparison dictionary containing IDS and BFS metrics.
    """
    normalized = parse_graph_input(graph) if isinstance(graph, str) else normalize_graph(graph)
    start_str = str(start).strip()
    goal_str = str(goal).strip()

    ids_res = iterative_deepening_search(
        graph=normalized,
        start=start_str,
        goal=goal_str,
        cycle_prevention=cycle_prevention,
    )

    bfs_graph = dict(normalized)
    if goal_str not in bfs_graph:
        bfs_graph[goal_str] = []

    bfs_res = breadth_first_search(
        graph=bfs_graph,
        start_node=start_str,
        sort_neighbors=False,
    )
    bfs_path = bfs_res.shortest_path_to(goal_str)
    bfs_depth = (len(bfs_path) - 1) if bfs_path is not None else None
    bfs_peak_frontier = max((len(step.frontier_after) for step in bfs_res.steps), default=1)
    bfs_total_stored = len(bfs_res.visited_set)

    # Number of BFS expansions up to discovering goal
    bfs_expansions_to_goal = 0
    for step in bfs_res.steps:
        bfs_expansions_to_goal += 1
        if step.current_node == goal_str or goal_str in step.neighbors_enqueued:
            break

    return {
        "start_node": start_str,
        "goal_node": goal_str,
        "ids_found": ids_res.is_success(),
        "ids_path": ids_res.path,
        "ids_depth": ids_res.depth,
        "ids_iterations": len(ids_res.iterations),
        "ids_total_expansions": ids_res.total_nodes_expanded,
        "ids_total_generations": ids_res.total_nodes_generated,
        "ids_peak_stack_depth": ids_res.peak_stack_depth,
        "bfs_found": bfs_path is not None,
        "bfs_path": bfs_path,
        "bfs_depth": bfs_depth,
        "bfs_expansions_to_goal": bfs_expansions_to_goal,
        "bfs_total_visited_stored": bfs_total_stored,
        "bfs_peak_frontier_size": bfs_peak_frontier,
        "same_optimal_depth": ids_res.depth == bfs_depth,
        "ids_result": ids_res,
        "bfs_result": bfs_res,
    }


def visualize_ids(
    result: IDSResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Path:
    """Render a two-panel visualization of IDS path discovery and per-depth expansions.

    Left panel: Directed graph highlighting start node, goal node, and optimal path.
    Right panel: Grouped bar chart of nodes expanded and nodes visited at each depth limit.

    Args:
        result: Completed ``IDSResult``.
        output_path: Destination image path (defaults to ``reports/ids_<start>_to_<goal>.png``).
        show: Whether to invoke ``plt.show()``.
        title: Optional suptitle override.

    Returns:
        Resolved ``Path`` of the saved image file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ids_search_{result.start_node}_to_{result.goal_node}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    for u, neighbors in result.graph.items():
        g.add_node(u)
        for v in neighbors:
            g.add_edge(u, v)

    node_depths: Dict[str, int] = {}
    for node in g.nodes():
        if nx.has_path(g, result.start_node, node):
            node_depths[node] = nx.shortest_path_length(g, result.start_node, node)
        else:
            node_depths[node] = -1

    try:
        for n, d in node_depths.items():
            g.nodes[n]["layer"] = d if d >= 0 else 99
        pos = nx.multipartite_layout(g, subset_key="layer")
    except Exception:
        pos = nx.spring_layout(g, seed=42)

    solution_edges: Set[Tuple[str, str]] = set()
    if result.is_success() and result.path:
        for i in range(len(result.path) - 1):
            solution_edges.add((result.path[i], result.path[i + 1]))

    node_colors: List[str] = []
    node_borders: List[str] = []
    node_sizes: List[int] = []

    for node in g.nodes():
        if node == result.start_node and node == result.goal_node and result.is_success():
            node_colors.append("#2ECC71")
            node_borders.append("#196F3D")
            node_sizes.append(1100)
        elif node == result.start_node:
            node_colors.append("#3498DB")
            node_borders.append("#1B4F72")
            node_sizes.append(1050)
        elif node == result.goal_node and result.is_success():
            node_colors.append("#2ECC71")
            node_borders.append("#196F3D")
            node_sizes.append(1050)
        elif node == result.goal_node:
            node_colors.append("#F39C12")
            node_borders.append("#B9770E")
            node_sizes.append(1050)
        elif result.is_success() and result.path and node in result.path:
            node_colors.append("#A9DFBF")
            node_borders.append("#1E8449")
            node_sizes.append(900)
        else:
            node_colors.append("#EBF5FB")
            node_borders.append("#85929E")
            node_sizes.append(750)

    edge_colors = [
        "#27AE60" if (u, v) in solution_edges else "#BDC3C7" for u, v in g.edges()
    ]
    edge_widths = [3.0 if (u, v) in solution_edges else 1.2 for u, v in g.edges()]

    fig, (ax_graph, ax_bars) = plt.subplots(1, 2, figsize=(14, 6), dpi=140)

    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax_graph,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=2.0,
        node_size=node_sizes,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax_graph,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(g, pos, ax=ax_graph, font_size=10, font_weight="bold")

    legend_items = [
        mpatches.Patch(color="#3498DB", label=f"Start ('{result.start_node}')"),
        mpatches.Patch(
            color="#2ECC71" if result.is_success() else "#F39C12",
            label=f"Goal ('{result.goal_node}' - {'FOUND' if result.is_success() else result.status.upper()})",
        ),
    ]
    if result.is_success():
        legend_items.append(mpatches.Patch(color="#27AE60", label=f"Optimal Path: {result.format_path()}"))
    ax_graph.legend(handles=legend_items, loc="upper right", fontsize=8.5, framealpha=0.92)
    ax_graph.set_title("State-Space Graph & Optimal Path", fontsize=11, fontweight="bold")
    ax_graph.axis("off")

    limits = [it.limit for it in result.iterations]
    expanded_counts = [it.nodes_expanded for it in result.iterations]
    generated_counts = [it.nodes_generated for it in result.iterations]
    x_positions = list(range(len(limits)))
    width = 0.36

    ax_bars.bar(
        [x - width / 2 for x in x_positions],
        expanded_counts,
        width=width,
        color="#2E86C1",
        label="Nodes Expanded",
    )
    ax_bars.bar(
        [x + width / 2 for x in x_positions],
        generated_counts,
        width=width,
        color="#F39C12",
        label="Nodes Visited (Generated)",
    )
    ax_bars.set_xticks(x_positions)
    ax_bars.set_xticklabels([f"Limit {l}\n({it.status})" for l, it in zip(limits, result.iterations)])
    ax_bars.set_ylabel("Node Count")
    ax_bars.set_title(
        f"Per-Iteration Work (Total Expanded={result.total_nodes_expanded}, "
        f"Total Visited={result.total_nodes_generated})",
        fontsize=11,
        fontweight="bold",
    )
    ax_bars.legend(fontsize=8.5)
    ax_bars.grid(axis="y", linestyle="--", alpha=0.4)

    suptitle = title or (
        f"Iterative Deepening Search (IDS) — {result.format_sequence()}\n"
        f"Optimal Depth={result.depth if result.depth is not None else 'N/A'} | "
        f"Peak Stack Depth={result.peak_stack_depth} (O(b*d))"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return output_path.resolve()


def get_ids_viva_reference() -> Dict[str, Any]:
    """Return structured Viva Voce reference notes covering all Question 17 syllabus topics.

    Returns:
        Dictionary containing syllabus viva topics, complexity comparison matrix,
        and high-frequency viva Q&As.
    """
    return {
        "title": "Iterative Deepening Search (IDS) Viva Voce Reference Guide",
        "core_concept": "Completeness + BFS Optimality + DFS Linear Memory O(b * d)",
        "viva_topics": {
            "why_ids_preferred_in_large_state_spaces": {
                "memory_bottleneck_of_bfs": (
                    "In large state spaces (e.g., 15-puzzle, Rubik's Cube, chess search trees), "
                    "BFS must keep every generated node in its frontier and explored table: O(b^d) "
                    "space. For b = 10 and d = 12, BFS requires ~10^12 nodes in RAM, exhausting "
                    "memory long before time runs out."
                ),
                "ids_advantage": (
                    "IDS uses only O(b * d) memory (storing just the active recursion path and "
                    "sibling choices along depth d). For b = 10, d = 12, IDS needs only ~120 nodes "
                    "in memory while still guaranteeing the shortest path and completeness."
                ),
                "unknown_solution_depth": (
                    "Unlike DLS, which requires guessing the depth limit l in advance, IDS "
                    "automatically discovers the shallowest goal depth d without prior knowledge."
                ),
            },
            "redundant_node_generation": {
                "apparent_concern": (
                    "Because IDS restarts DLS from the root at limits 0, 1, 2, ..., d, nodes at "
                    "depth 0 are generated (d + 1) times, depth 1 nodes d times, and so on."
                ),
                "mathematical_proof": (
                    "In a uniform tree of branching factor b and goal depth d:\n"
                    "  N_IDS = (d + 1)*1 + d*b + (d - 1)*b^2 + ... + 1*b^d\n"
                    "Since the bottom level b^d dominates, N_IDS / b^d approaches b / (b - 1).\n"
                    "Examples:\n"
                    "  - For b = 10, d = 5: N_BFS = 111,111 vs N_IDS = 123,456 (~11% overhead).\n"
                    "  - For b = 2, d = 10: overhead factor is at most ~2x.\n"
                    "Thus asymptotic time remains O(b^d)."
                ),
            },
            "time_vs_space_tradeoff": {
                "summary": (
                    "IDS trades a small constant-factor increase in CPU time (regenerating upper "
                    "levels) for an exponential reduction in memory from O(b^d) down to O(b * d). "
                    "In practice, modern AI systems become memory-bound before becoming CPU-bound, "
                    "making IDS the preferred uninformed search method for large state spaces."
                ),
            },
        },
        "complexity_comparison": {
            "BFS": {
                "time": "O(b^d)",
                "space": "O(b^d) [Exponential RAM bottleneck]",
                "complete": "Yes (finite b)",
                "optimal": "Yes (uniform step cost)",
            },
            "DFS": {
                "time": "O(b^m)",
                "space": "O(b * m) [Linear]",
                "complete": "No (infinite/cyclic spaces)",
                "optimal": "No",
            },
            "DLS": {
                "time": "O(b^l)",
                "space": "O(b * l) [Linear]",
                "complete": "No (fails if l < d)",
                "optimal": "No",
            },
            "IDS": {
                "time": "O(b^d)",
                "space": "O(b * d) [Linear]",
                "complete": "Yes (finite b)",
                "optimal": "Yes (uniform step cost)",
            },
        },
        "key_questions": [
            {
                "question": "Why is Iterative Deepening Search preferred over BFS in large state spaces?",
                "answer": (
                    "BFS requires O(b^d) space to store the entire frontier, causing out-of-memory "
                    "failures at moderate depths. IDS achieves the same completeness and shortest-path "
                    "optimality using only O(b * d) linear space."
                ),
            },
            {
                "question": "Does regenerating upper-level nodes make IDS slower asymptotically than BFS?",
                "answer": (
                    "No. Both BFS and IDS have O(b^d) time complexity. In an exponential tree with "
                    "branching factor b > 1, most nodes live at the bottom level, so regenerating "
                    "upper levels adds only a constant factor b / (b - 1)."
                ),
            },
            {
                "question": "How does IDS know when to stop if a goal does not exist in the graph?",
                "answer": (
                    "It relies on the tri-state return of DLS. When DLS returns 'Cutoff', IDS "
                    "increments the limit and continues. When DLS returns 'Failure' (no branch hit "
                    "the depth limit), the graph has been exhausted and IDS halts immediately."
                ),
            },
            {
                "question": "How does IDS handle graphs containing cycles without losing O(b * d) space?",
                "answer": (
                    "By checking for cycles only against ancestors on the current recursion path "
                    "(O(d) active stack), rather than maintaining a global O(|V|) visited set."
                ),
            },
        ],
    }
