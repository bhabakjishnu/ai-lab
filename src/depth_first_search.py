"""Depth-First Search (DFS) for Graph Traversal.

This module implements Depth-First Search (DFS) for graph traversal and state-space
exploration, adhering to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA)
and Cormen, Leiserson, Rivest, Stein's 'Introduction to Algorithms' (CLRS).

Theoretical Background & AI Concepts:
- Uninformed (Blind) Search:
    DFS explores a graph by venturing as deep as possible along each branch before backtracking.
    It operates without domain-specific heuristics, relying strictly on state-space topology.
- LIFO Frontier Stack:
    DFS maintains a Last-In, First-Out (LIFO) frontier. In Python:
    - Iterative DFS uses an explicit list as a stack with O(1) amortized `append()` and `pop()`.
    - Recursive DFS uses Python's runtime call stack (implicit LIFO).
- Visited / Explored Set:
    Graph-search DFS maintains a `visited` set to track discovered/expanded vertices.
    This guarantees termination on graphs containing directed or undirected cycles and prevents
    infinite loops.
- Backtracking Mechanism:
    When a branch terminates at a dead-end (leaf node or all neighbors already visited), DFS
    backtracks to the most recent predecessor node with unexplored branches.
- Non-Optimality of DFS:
    DFS is NOT an optimal search strategy. It does not guarantee finding the shallowest goal
    or minimum-cost path. DFS may plunge down a deep or infinite branch while a shallow solution
    exists nearby on an alternate branch.
- Space Complexity Advantage over BFS:
    - BFS must store entire levels in memory, requiring O(b^d) space where b is branching factor
      and d is goal depth (exponential space bottleneck).
    - DFS needs to store only the active branch path plus unexpanded sibling alternatives:
      O(b * m) space for tree search (where m is maximum tree depth), and O(V) auxiliary space
      for graph search with a visited set. This makes DFS vastly more memory-efficient.
- Edge Classification (CLRS Formulation):
    During DFS, graph edges can be classified into:
    1. Tree Edges: Edges leading to newly discovered vertices in the DFS forest.
    2. Back Edges: Edges connecting a vertex to an ancestor in the DFS tree. A back edge
       proves the existence of a cycle in directed graphs!
    3. Forward Edges: Non-tree edges connecting a vertex to a descendant in the DFS tree.
    4. Cross Edges: All other edges connecting vertices across different branches/trees.
- Iterative vs. Recursive Traversal Order (Optional Challenge):
    - In iterative DFS, when neighbors are pushed onto the stack in forward order
      (e.g., node 1 pushes neighbor '2' then '3'), neighbor '3' sits at the top of the LIFO
      stack and is popped first, yielding order: 1 -> 3 -> 2 -> 4 (matching the sample output).
    - In standard recursive DFS, iterating through neighbors in forward order explores '2'
      first, yielding order: 1 -> 2 -> 4 -> 3.
    - If neighbors are pushed in reverse order (`reversed(neighbors)`), iterative DFS produces
      the identical sequence to standard recursive DFS (`1, 2, 4, 3`). Both ordering modes
      are supported and directly compared in this implementation.

Complexity:
- Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges.
  Each reachable vertex is visited once, and each outgoing edge is examined once.
- Space Complexity: O(V) auxiliary memory for the visited set and explicit/implicit stack.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Benchmark and sample graphs for testing, education, and demonstrations
DEFAULT_SAMPLE_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4"],
    "3": [],
    "4": [],
}

CYCLIC_BENCHMARK_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4"],
    "3": ["1", "5"],
    "4": ["3"],
    "5": [],
}

BINARY_TREE_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4", "5"],
    "3": ["6", "7"],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
}

DISCONNECTED_GRAPH: Dict[str, List[str]] = {
    "1": ["2"],
    "2": ["3"],
    "3": [],
    "4": ["5"],
    "5": [],
}

DAG_GRAPH: Dict[str, List[str]] = {
    "1": ["2", "3"],
    "2": ["4"],
    "3": ["4", "5"],
    "4": ["6"],
    "5": ["6"],
    "6": [],
}


@dataclass(frozen=True)
class DFSStepRecord:
    """Immutable record capturing an individual step during Depth-First Search.

    Attributes:
        step_number: 1-based chronological expansion or visitation sequence index.
        current_node: Node identifier being processed.
        action: Semantic action label (e.g., 'EXPAND', 'BACKTRACK', 'VISIT', 'SKIP_ALREADY_VISITED').
        stack_before: Snapshot of the LIFO stack (or call stack) prior to processing current node.
        stack_after: Snapshot of the LIFO stack after pushing eligible unvisited neighbors.
        visited_nodes: Ordered list of nodes marked visited up to this step.
        neighbors_evaluated: All outgoing neighbors from the current node in the graph.
        neighbors_pushed: Subset of neighbors that were unvisited and pushed to stack.
        parent: The node from which current_node was discovered (None for root).
        depth: Distance (number of edges) from the start node in the current active DFS branch.
    """

    step_number: int
    current_node: str
    action: str
    stack_before: List[str]
    stack_after: List[str]
    visited_nodes: List[str]
    neighbors_evaluated: List[str]
    neighbors_pushed: List[str]
    parent: Optional[str] = None
    depth: int = 0

    def format_row(self) -> str:
        """Format the step into a structured table row for terminal presentation.

        Returns:
            Formatted row string.
        """
        stack_str = "[" + ", ".join(self.stack_after) + "]" if self.stack_after else "[]"
        pushed_str = "[" + ", ".join(self.neighbors_pushed) + "]" if self.neighbors_pushed else "-"
        parent_str = str(self.parent) if self.parent is not None else "ROOT"
        return (
            f"{self.step_number:<5} | {self.current_node:<6} | {self.action:<10} | "
            f"{self.depth:<5} | {parent_str:<6} | {pushed_str:<16} | {stack_str}"
        )


@dataclass
class DFSResult:
    """Encapsulates the complete outcome of a Depth-First Search graph traversal.

    Attributes:
        traversal_order: Chronological sequence of visited node labels.
        mode: Implementation method ('iterative' with explicit stack or 'recursive').
        order_preserving: Whether neighbor pushing/expansion preserves input list order.
        discovery_time: Mapping of node labels to their discovery timestep (d[u]).
        finish_time: Mapping of node labels to their finish/backtrack timestep (f[u]).
        node_depths: Mapping of node labels to their depth in the DFS tree.
        max_depth: Maximum depth reached along any explored branch.
        parents: Mapping of each node to its predecessor parent in the DFS tree.
        edge_classification: Categorization of inspected edges ('TREE', 'BACK', 'FORWARD', 'CROSS').
        cycles_detected: List of directed back-edges indicating cycle presence.
        steps: Chronological list of DFSStepRecord objects for fine-grained inspection.
        graph: The normalized adjacency list representation of the searched graph.
        start_node: Node where DFS traversal originated.
        visited_set: Set of all nodes reached and explored.
        unreachable_nodes: Nodes defined in the graph that could not be reached from start_node.
        execution_time_ms: Measured execution wall-clock time in milliseconds.
    """

    traversal_order: List[str]
    mode: str
    order_preserving: bool
    discovery_time: Dict[str, int]
    finish_time: Dict[str, int]
    node_depths: Dict[str, int]
    max_depth: int
    parents: Dict[str, Optional[str]]
    edge_classification: Dict[Tuple[str, str], str]
    cycles_detected: List[Tuple[str, str]]
    steps: List[DFSStepRecord]
    graph: Dict[str, List[str]]
    start_node: str
    visited_set: Set[str] = field(default_factory=set)
    unreachable_nodes: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def format_traversal(self, delimiter: str = ", ") -> str:
        """Return the standard traversal sequence string.

        Args:
            delimiter: Separator between successive node labels.

        Returns:
            Formatted traversal order string.
        """
        return delimiter.join(self.traversal_order)

    def has_cycle(self) -> bool:
        """Check whether at least one cycle was discovered during traversal.

        Returns:
            True if any back-edge was encountered, False otherwise.
        """
        return len(self.cycles_detected) > 0

    def format_step_table(self) -> str:
        """Return a formatted ASCII execution trace table of the DFS search process.

        Returns:
            Table string showing step, expanded node, action, depth, parent, pushed, and stack.
        """
        headers = (
            f"{'Step':<5} | {'Node':<6} | {'Action':<10} | {'Depth':<5} | "
            f"{'Parent':<6} | {'Pushed':<16} | {'Stack (LIFO)':<22}"
        )
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def format_summary(self) -> str:
        """Return a formatted multi-line summary of traversal metrics and results.

        Returns:
            Descriptive string report.
        """
        cycle_status = "Detected" if self.has_cycle() else "None (Acyclic)"
        cycle_detail = ", ".join([f"{u}->{v}" for u, v in self.cycles_detected]) if self.has_cycle() else "N/A"
        unreachable_str = ", ".join(self.unreachable_nodes) if self.unreachable_nodes else "None (All Reached)"

        lines = [
            "=" * 65,
            f"DEPTH-FIRST SEARCH (DFS) TRAVERSAL SUMMARY ({self.mode.upper()} MODE)",
            "=" * 65,
            f"Start Node               : {self.start_node}",
            f"Traversal Order          : {self.format_traversal()}",
            f"Neighbor Order Preserving: {self.order_preserving}",
            f"Nodes Visited            : {len(self.traversal_order)} of {len(self.graph)}",
            f"Maximum Depth Reached    : {self.max_depth}",
            f"Cycles Detected          : {cycle_status} ({cycle_detail})",
            f"Unreachable Nodes        : {unreachable_str}",
            f"Execution Time           : {self.execution_time_ms:.4f} ms",
            "-" * 65,
            "DFS Spanning Tree Parent Connections:",
        ]
        for node in self.traversal_order:
            parent = self.parents.get(node)
            parent_display = parent if parent is not None else "ROOT"
            depth = self.node_depths.get(node, 0)
            d_time = self.discovery_time.get(node, "-")
            f_time = self.finish_time.get(node, "-")
            lines.append(
                f"  Node {node:<4} | Parent: {parent_display:<6} | Depth: {depth:<3} | "
                f"d/f Timestamps: [{d_time}/{f_time}]"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def format_edge_classification(self) -> str:
        """Return a formatted breakdown of CLRS edge classifications.

        Returns:
            Table string classifying all inspected edges.
        """
        lines = [
            f"{'Edge':<10} | {'Classification':<15} | {'Significance'}",
            "-" * 55,
        ]
        significance_map = {
            "TREE": "Primary traversal edge forming DFS spanning tree",
            "BACK": "Edge to an ancestor; indicates presence of a cycle!",
            "FORWARD": "Non-tree shortcut edge to a descendant",
            "CROSS": "Edge between branches or disjoint trees",
        }
        for (u, v), edge_type in sorted(self.edge_classification.items()):
            sig = significance_map.get(edge_type, "")
            lines.append(f"{f'{u}->{v}':<10} | {edge_type:<15} | {sig}")
        return "\n".join(lines)

    def path_to(self, target_node: str) -> Optional[List[str]]:
        """Reconstruct the explored branch path from start_node to target_node.

        Uses the DFS tree `parents` mapping backward from target to start.

        Args:
            target_node: Destination node label.

        Returns:
            List of node labels along the path, or None if target is unreachable.
        """
        if target_node not in self.parents:
            return None

        path = []
        curr: Optional[str] = target_node
        visited_in_path: Set[str] = set()

        while curr is not None:
            if curr in visited_in_path:
                # Defensive check against circular parent links
                break
            visited_in_path.add(curr)
            path.append(curr)
            curr = self.parents.get(curr)

        path.reverse()
        if path and path[0] == self.start_node:
            return path
        return None


def normalize_graph(graph: Dict[Any, Sequence[Any]]) -> Dict[str, List[str]]:
    """Sanitize and normalize an arbitrary graph dictionary into standard format.

    Ensures:
    - All node keys and neighbor values are converted to stripped strings.
    - Every neighbor node mentioned in adjacency lists has an entry in the graph dictionary.
    - Redundant duplicate edges to the same neighbor are deduplicated while preserving order.

    Args:
        graph: Raw graph dictionary mapping nodes to sequences of neighbors.

    Returns:
        Normalized dictionary mapping str node labels to lists of str neighbors.

    Raises:
        ValueError: If graph is empty or has non-mapping structure.
    """
    if not isinstance(graph, dict):
        raise ValueError(f"Graph must be a dictionary, got {type(graph).__name__}")
    if not graph:
        raise ValueError("Graph dictionary cannot be empty.")

    normalized: Dict[str, List[str]] = {}

    # First pass: normalize existing keys and their neighbors
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

    # Second pass: ensure any referenced neighbor node exists as a dictionary key
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

    # Format 1: Try JSON parsing
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, dict):
                return normalize_graph(parsed_json)
        except json.JSONDecodeError:
            pass

        # Format 2: Try Python dict literal parsing safely
        try:
            import ast

            evaluated = ast.literal_eval(text)
            if isinstance(evaluated, dict):
                return normalize_graph(evaluated)
        except (ValueError, SyntaxError):
            pass

    # Format 3: Compact adjacency list notation ("1: 2, 3; 2: 4; 3: ; 4: ")
    if ":" in text:
        graph: Dict[str, List[str]] = {}
        entries = [e.strip() for e in text.split(";") if e.strip()]
        for entry in entries:
            parts = entry.split(":", 1)
            node = parts[0].strip()
            neighbors = []
            if len(parts) > 1 and parts[1].strip():
                neighbors = [n.strip() for n in parts[1].split(",") if n.strip()]
            graph[node] = neighbors
        if graph:
            return normalize_graph(graph)

    # Format 4: Edge-list notation ("1->2, 1->3, 2->4" or "1-2, 1-3, 2-4")
    if "->" in text or "-" in text:
        graph = {}
        delimiters = [",", "\n"]
        raw_edges = [text]
        for delim in delimiters:
            raw_edges = [sub for item in raw_edges for sub in item.split(delim)]

        for edge_str in raw_edges:
            edge_str = edge_str.strip()
            if not edge_str:
                continue
            if "->" in edge_str:
                u, v = edge_str.split("->", 1)
            elif "-" in edge_str:
                u, v = edge_str.split("-", 1)
            else:
                continue
            u, v = u.strip(), v.strip()
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            if v and v not in graph[u]:
                graph[u].append(v)
        if graph:
            return normalize_graph(graph)

    raise ValueError(
        f"Unable to parse graph input: '{text}'. Please provide a dictionary, "
        "JSON string, compact 'u: v1, v2; ...' notation, or edge list 'u->v, ...'."
    )


def dfs_iterative(
    graph: Dict[str, List[str]],
    start: str,
    order_preserving: bool = False,
    sort_neighbors: bool = False,
) -> DFSResult:
    """Execute Depth-First Search iteratively using an explicit LIFO stack (Python list).

    Exploration Mechanism:
    - Employs a Python `list` as a stack (`append()` and `pop()`).
    - Tracks discovered/visited nodes in a `visited` set to eliminate cycles and redundant work.
    - When `order_preserving` is False (default for sample input matching):
      Neighbors are pushed in forward order `[nbr1, nbr2]`. Because the stack is LIFO,
      the last neighbor pushed (`nbr2`) is popped first!
      For graph `{'1': ['2', '3'], '2': ['4']}`, start `'1'`, pushing `'2'` then `'3'` means
      `'3'` is popped first, producing traversal: `1, 3, 2, 4` (exact sample output).
    - When `order_preserving` is True:
      Neighbors are pushed in reversed order (`reversed(neighbors)`), ensuring that the first
      neighbor (`nbr1`) remains at the top of the stack and is popped first, producing `1, 2, 4, 3`.

    Args:
        graph: Adjacency list representation of graph.
        start: Start node label.
        order_preserving: If True, pushes neighbors in reverse so first neighbor is popped first.
        sort_neighbors: If True, sorts neighbors before pushing.

    Returns:
        DFSResult containing traversal sequence, step records, cycles, and tree structures.

    Raises:
        KeyError: If start node is not present in graph.
    """
    normalized = normalize_graph(graph)
    start_str = str(start).strip()
    if start_str not in normalized:
        raise KeyError(f"Start node '{start_str}' does not exist in graph.")

    start_time = time.perf_counter()

    traversal_order: List[str] = []
    visited_set: Set[str] = set()
    parents: Dict[str, Optional[str]] = {start_str: None}
    node_depths: Dict[str, int] = {start_str: 0}
    discovery_time: Dict[str, int] = {}
    finish_time: Dict[str, int] = {}
    edge_classification: Dict[Tuple[str, str], str] = {}
    cycles_detected: List[Tuple[str, str]] = []
    steps: List[DFSStepRecord] = []

    # Stack holds tuples: (current_node, parent_node, current_depth)
    stack: List[Tuple[str, Optional[str], int]] = [(start_str, None, 0)]
    timer = 0
    step_num = 1

    while stack:
        stack_before_labels = [item[0] for item in stack]
        curr_node, parent_node, curr_depth = stack.pop()

        if curr_node in visited_set:
            # Already visited via another path
            continue

        # Mark as visited and record discovery
        visited_set.add(curr_node)
        traversal_order.append(curr_node)
        timer += 1
        discovery_time[curr_node] = timer
        if parent_node is not None and curr_node not in parents:
            parents[curr_node] = parent_node
            node_depths[curr_node] = curr_depth

        # Retrieve active ancestor path to identify back-edges (cycles)
        ancestors: Set[str] = set()
        p: Optional[str] = curr_node
        while p is not None:
            ancestors.add(p)
            p = parents.get(p)

        # Retrieve and format neighbors
        raw_neighbors = list(normalized.get(curr_node, []))
        if sort_neighbors:
            raw_neighbors = sorted(raw_neighbors)

        # Classify edges & filter unvisited neighbors
        eligible_neighbors: List[str] = []
        for nbr in raw_neighbors:
            edge = (curr_node, nbr)
            if nbr in ancestors:
                edge_classification[edge] = "BACK"
                if edge not in cycles_detected:
                    cycles_detected.append(edge)
            elif nbr not in visited_set:
                edge_classification[edge] = "TREE"
                eligible_neighbors.append(nbr)
            elif discovery_time.get(nbr, 0) > discovery_time.get(curr_node, 0):
                edge_classification[edge] = "FORWARD"
            else:
                edge_classification[edge] = "CROSS"

        # Determine push ordering onto LIFO stack
        if order_preserving:
            # Pushing in reverse allows the first neighbor to sit on top
            neighbors_to_push = list(reversed(eligible_neighbors))
        else:
            # Standard push: pushing in forward order places the last neighbor on top
            neighbors_to_push = eligible_neighbors

        for nbr in neighbors_to_push:
            if nbr not in parents:
                parents[nbr] = curr_node
                node_depths[nbr] = curr_depth + 1
            stack.append((nbr, curr_node, curr_depth + 1))

        stack_after_labels = [item[0] for item in stack]

        # Record fine-grained execution step
        steps.append(
            DFSStepRecord(
                step_number=step_num,
                current_node=curr_node,
                action="EXPAND",
                stack_before=stack_before_labels,
                stack_after=stack_after_labels,
                visited_nodes=list(traversal_order),
                neighbors_evaluated=raw_neighbors,
                neighbors_pushed=neighbors_to_push,
                parent=parent_node,
                depth=curr_depth,
            )
        )
        step_num += 1

    # Record finish times for completed nodes
    for idx, node in enumerate(traversal_order, start=1):
        finish_time[node] = len(traversal_order) + idx

    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    all_graph_nodes = sorted(normalized.keys())
    unreachable = [n for n in all_graph_nodes if n not in visited_set]
    max_depth = max(node_depths.values()) if node_depths else 0

    return DFSResult(
        traversal_order=traversal_order,
        mode="iterative",
        order_preserving=order_preserving,
        discovery_time=discovery_time,
        finish_time=finish_time,
        node_depths=node_depths,
        max_depth=max_depth,
        parents=parents,
        edge_classification=edge_classification,
        cycles_detected=cycles_detected,
        steps=steps,
        graph=normalized,
        start_node=start_str,
        visited_set=visited_set,
        unreachable_nodes=unreachable,
        execution_time_ms=execution_time_ms,
    )


def dfs_recursive(
    graph: Dict[str, List[str]],
    start: str,
    order_preserving: bool = True,
    sort_neighbors: bool = False,
) -> DFSResult:
    """Execute Depth-First Search recursively using the runtime call stack.

    Exploration Mechanism:
    - Uses Python functions calls as the implicit LIFO recursion stack.
    - Tracks visited nodes in a set to prevent infinite loops in cyclic graphs.
    - Demonstrates true depth-first backtracking upon hitting leaf or visited nodes.
    - When `order_preserving` is True (standard recursive convention):
      Neighbors are traversed in natural forward order `[nbr1, nbr2]`, exploring `nbr1` first.
    - When `order_preserving` is False:
      Neighbors are traversed in reverse order `reversed(neighbors)`, exploring `nbr2` first,
      which matches the exact LIFO pop behavior of the standard iterative stack (`1, 3, 2, 4`).

    Args:
        graph: Adjacency list representation of graph.
        start: Start node label.
        order_preserving: If True, visits neighbors in list order; if False, visits in reverse.
        sort_neighbors: If True, sorts neighbors before iteration.

    Returns:
        DFSResult containing traversal sequence, recursion traces, cycles, and tree structures.

    Raises:
        KeyError: If start node is not present in graph.
    """
    normalized = normalize_graph(graph)
    start_str = str(start).strip()
    if start_str not in normalized:
        raise KeyError(f"Start node '{start_str}' does not exist in graph.")

    start_time = time.perf_counter()

    traversal_order: List[str] = []
    visited_set: Set[str] = set()
    parents: Dict[str, Optional[str]] = {start_str: None}
    node_depths: Dict[str, int] = {start_str: 0}
    discovery_time: Dict[str, int] = {}
    finish_time: Dict[str, int] = {}
    edge_classification: Dict[Tuple[str, str], str] = {}
    cycles_detected: List[Tuple[str, str]] = []
    steps: List[DFSStepRecord] = []

    active_call_stack: List[str] = []
    timer = 0
    step_num = 1

    def _dfs(curr_node: str, parent_node: Optional[str], curr_depth: int) -> None:
        nonlocal timer, step_num
        timer += 1
        discovery_time[curr_node] = timer
        visited_set.add(curr_node)
        traversal_order.append(curr_node)
        node_depths[curr_node] = curr_depth
        if parent_node is not None:
            parents[curr_node] = parent_node

        active_call_stack.append(curr_node)
        stack_before = list(active_call_stack)

        raw_neighbors = list(normalized.get(curr_node, []))
        if sort_neighbors:
            raw_neighbors = sorted(raw_neighbors)

        eligible_neighbors: List[str] = []
        for nbr in raw_neighbors:
            edge = (curr_node, nbr)
            if nbr in active_call_stack:
                edge_classification[edge] = "BACK"
                if edge not in cycles_detected:
                    cycles_detected.append(edge)
            elif nbr not in visited_set:
                edge_classification[edge] = "TREE"
                eligible_neighbors.append(nbr)
            elif discovery_time.get(nbr, 0) > discovery_time.get(curr_node, 0):
                edge_classification[edge] = "FORWARD"
            else:
                edge_classification[edge] = "CROSS"

        traversal_neighbors = eligible_neighbors if order_preserving else list(reversed(eligible_neighbors))

        steps.append(
            DFSStepRecord(
                step_number=step_num,
                current_node=curr_node,
                action="EXPAND",
                stack_before=stack_before,
                stack_after=list(active_call_stack),
                visited_nodes=list(traversal_order),
                neighbors_evaluated=raw_neighbors,
                neighbors_pushed=traversal_neighbors,
                parent=parent_node,
                depth=curr_depth,
            )
        )
        step_num += 1

        for nbr in traversal_neighbors:
            if nbr not in visited_set:
                _dfs(nbr, curr_node, curr_depth + 1)

        timer += 1
        finish_time[curr_node] = timer

        # Backtracking step record
        steps.append(
            DFSStepRecord(
                step_number=step_num,
                current_node=curr_node,
                action="BACKTRACK",
                stack_before=list(active_call_stack),
                stack_after=[n for n in active_call_stack if n != curr_node],
                visited_nodes=list(traversal_order),
                neighbors_evaluated=[],
                neighbors_pushed=[],
                parent=parent_node,
                depth=curr_depth,
            )
        )
        step_num += 1
        active_call_stack.pop()

    _dfs(start_str, None, 0)

    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    all_graph_nodes = sorted(normalized.keys())
    unreachable = [n for n in all_graph_nodes if n not in visited_set]
    max_depth = max(node_depths.values()) if node_depths else 0

    return DFSResult(
        traversal_order=traversal_order,
        mode="recursive",
        order_preserving=order_preserving,
        discovery_time=discovery_time,
        finish_time=finish_time,
        node_depths=node_depths,
        max_depth=max_depth,
        parents=parents,
        edge_classification=edge_classification,
        cycles_detected=cycles_detected,
        steps=steps,
        graph=normalized,
        start_node=start_str,
        visited_set=visited_set,
        unreachable_nodes=unreachable,
        execution_time_ms=execution_time_ms,
    )


def depth_first_search(
    graph: Dict[str, List[str]],
    start: str,
    mode: str = "iterative",
    order_preserving: bool = False,
    sort_neighbors: bool = False,
) -> DFSResult:
    """Unified top-level entry point to execute Depth-First Search.

    Args:
        graph: Graph dictionary mapping node labels to neighbor sequences.
        start: Initial starting node.
        mode: Implementation choice: 'iterative' (explicit stack) or 'recursive'.
        order_preserving: If True, preserves left-to-right neighbor ordering.
        sort_neighbors: If True, sorts neighbor lists before processing.

    Returns:
        DFSResult encapsulating traversal order, metrics, and trace.
    """
    mode_clean = mode.lower().strip()
    if mode_clean in ("recursive", "rec"):
        return dfs_recursive(
            graph=graph,
            start=start,
            order_preserving=order_preserving,
            sort_neighbors=sort_neighbors,
        )
    return dfs_iterative(
        graph=graph,
        start=start,
        order_preserving=order_preserving,
        sort_neighbors=sort_neighbors,
    )


def compare_dfs_modes(graph: Dict[str, List[str]], start: str) -> Dict[str, Any]:
    """Execute both iterative and recursive DFS and provide a comprehensive comparison.

    Addresses the Optional Challenge:
    'Optional Challenge: Implement both iterative and recursive versions and compare output.'

    Compares:
    1. Iterative DFS (Standard Stack Push, LIFO pop order): Matches question sample output.
    2. Iterative DFS (Order-Preserving Reverse Push): Pops neighbors in left-to-right order.
    3. Recursive DFS (Standard Left-to-Right Recursion).
    4. Recursive DFS (Reverse Recursion): Matches standard stack pop order.

    Args:
        graph: Graph adjacency dictionary.
        start: Starting node.

    Returns:
        Dictionary summarizing comparison metrics, sequences, and explanations.
    """
    iter_standard = dfs_iterative(graph, start, order_preserving=False)
    iter_preserving = dfs_iterative(graph, start, order_preserving=True)
    rec_standard = dfs_recursive(graph, start, order_preserving=True)
    rec_reversed = dfs_recursive(graph, start, order_preserving=False)

    identical_standard_iter_vs_reversed_rec = iter_standard.traversal_order == rec_reversed.traversal_order
    identical_preserving_iter_vs_standard_rec = iter_preserving.traversal_order == rec_standard.traversal_order

    explanation = (
        "Pedagogical Analysis of Traversal Sequences:\n"
        "1. Iterative (Standard Forward-Push): Pushes neighbors [A, B] onto a LIFO stack in order.\n"
        "   Because the stack is Last-In First-Out, 'B' is popped and explored before 'A'.\n"
        f"   Output: {iter_standard.format_traversal()} (matches Question 12 sample output!).\n"
        "2. Recursive (Standard Forward-Call): Calls recursive dfs(A) before dfs(B).\n"
        "   Therefore 'A' and all its deep descendants are visited before 'B'.\n"
        f"   Output: {rec_standard.format_traversal()}.\n"
        "3. Iterative (Order-Preserving Reverse-Push): Pushes neighbors in reversed order [B, A].\n"
        "   'A' is on top of the stack and popped first, perfectly aligning with standard recursion!\n"
        f"   Output: {iter_preserving.format_traversal()} (Identical to Standard Recursive: {identical_preserving_iter_vs_standard_rec})."
    )

    return {
        "graph": normalize_graph(graph),
        "start_node": str(start).strip(),
        "iterative_standard": {
            "order": iter_standard.traversal_order,
            "formatted": iter_standard.format_traversal(),
            "steps_count": len(iter_standard.steps),
            "time_ms": iter_standard.execution_time_ms,
        },
        "iterative_order_preserving": {
            "order": iter_preserving.traversal_order,
            "formatted": iter_preserving.format_traversal(),
            "steps_count": len(iter_preserving.steps),
            "time_ms": iter_preserving.execution_time_ms,
        },
        "recursive_standard": {
            "order": rec_standard.traversal_order,
            "formatted": rec_standard.format_traversal(),
            "steps_count": len(rec_standard.steps),
            "time_ms": rec_standard.execution_time_ms,
        },
        "recursive_reversed": {
            "order": rec_reversed.traversal_order,
            "formatted": rec_reversed.format_traversal(),
            "steps_count": len(rec_reversed.steps),
            "time_ms": rec_reversed.execution_time_ms,
        },
        "identical_standard_iter_vs_reversed_rec": identical_standard_iter_vs_reversed_rec,
        "identical_preserving_iter_vs_standard_rec": identical_preserving_iter_vs_standard_rec,
        "explanation": explanation,
    }


class DepthFirstSearch:
    """Object-Oriented Depth-First Search Graph Search Engine.

    Encapsulates graph topology, state tracking, cycle detection, path reconstruction,
    and side-by-side iterative and recursive analysis.
    """

    def __init__(self, graph: Optional[Dict[str, Sequence[str]]] = None) -> None:
        """Initialize the DepthFirstSearch engine.

        Args:
            graph: Optional initial graph dictionary. Defaults to DEFAULT_SAMPLE_GRAPH.
        """
        raw_graph = graph if graph is not None else DEFAULT_SAMPLE_GRAPH
        self._graph: Dict[str, List[str]] = normalize_graph(raw_graph)

    @property
    def graph(self) -> Dict[str, List[str]]:
        """Return the normalized graph adjacency list."""
        return self._graph

    def set_graph(self, graph: Union[str, Dict[str, Sequence[str]]]) -> None:
        """Update the searched graph topology.

        Args:
            graph: String or dictionary representation of the new graph.
        """
        self._graph = parse_graph_input(graph)

    def search(
        self,
        start_node: str,
        mode: str = "iterative",
        order_preserving: bool = False,
        sort_neighbors: bool = False,
    ) -> DFSResult:
        """Execute DFS on the configured graph.

        Args:
            start_node: Starting node label.
            mode: 'iterative' or 'recursive'.
            order_preserving: Whether to preserve left-to-right neighbor ordering.
            sort_neighbors: Whether to sort neighbor lists.

        Returns:
            DFSResult instance.
        """
        return depth_first_search(
            graph=self._graph,
            start=start_node,
            mode=mode,
            order_preserving=order_preserving,
            sort_neighbors=sort_neighbors,
        )

    def compare(self, start_node: str) -> Dict[str, Any]:
        """Compare iterative and recursive implementations from start_node.

        Args:
            start_node: Starting node label.

        Returns:
            Comparison dictionary.
        """
        return compare_dfs_modes(self._graph, start_node)

    def find_path(self, start_node: str, target_node: str, mode: str = "iterative") -> Optional[List[str]]:
        """Find an explored path from start_node to target_node using DFS.

        Args:
            start_node: Origin node label.
            target_node: Destination node label.
            mode: 'iterative' or 'recursive'.

        Returns:
            List of node labels along the path, or None if target is unreachable.
        """
        result = self.search(start_node=start_node, mode=mode)
        return result.path_to(target_node)

    def detect_cycles(self, start_node: Optional[str] = None) -> List[Tuple[str, str]]:
        """Identify all directed cycle back-edges reachable from the starting node.

        If start_node is None, examines all connected components across the graph.

        Args:
            start_node: Optional root node for cycle detection.

        Returns:
            List of (u, v) back-edges confirming cycles.
        """
        if start_node is not None:
            result = self.search(start_node=start_node, mode="iterative")
            return result.cycles_detected

        all_cycles: List[Tuple[str, str]] = []
        visited_global: Set[str] = set()
        for node in sorted(self._graph.keys()):
            if node not in visited_global:
                res = self.search(start_node=node, mode="iterative")
                visited_global.update(res.visited_set)
                all_cycles.extend(res.cycles_detected)
        return all_cycles


def visualize_dfs(
    result: DFSResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Any:
    """Render a publication-quality NetworkX & Matplotlib visualization of the DFS traversal.

    Highlights:
    - Root/Start node in distinct accent color.
    - Visited nodes with discovery sequence badges (e.g., 'Node (1st)').
    - DFS spanning tree edges in bold solid styling.
    - Back-edges (cycles) highlighted in dashed red.
    - Unreachable nodes in muted gray.

    Args:
        result: DFSResult instance from depth_first_search().
        save_path: Optional file path to save image (e.g. 'reports/dfs_traversal.png').
        show: If True, calls plt.show() to display interactive window.
        title: Optional custom figure title.

    Returns:
        The matplotlib Figure object, or None if matplotlib/networkx are unavailable.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")  # Non-interactive backend for headless execution
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("[Warning] Matplotlib and/or NetworkX are not installed. Visualization skipped.")
        return None

    G = nx.DiGraph()
    for u, neighbors in result.graph.items():
        G.add_node(u)
        for v in neighbors:
            G.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    # Determine graph layout
    try:
        # Hierarchical tree layout if acyclic, otherwise spring layout
        if nx.is_directed_acyclic_graph(G):
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.2)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Partition nodes into categories
    start_node = result.start_node
    visited_nodes = [n for n in result.traversal_order if n != start_node]
    unreachable_nodes = result.unreachable_nodes

    # Colors
    color_start = "#10B981"  # Emerald green
    color_visited = "#3B82F6"  # Royal blue
    color_unreached = "#9CA3AF"  # Slate gray

    # Draw nodes
    if start_node in G:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[start_node], node_color=color_start,
            node_size=1400, ax=ax, label=f"Start Node: {start_node}"
        )
    if visited_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=visited_nodes, node_color=color_visited,
            node_size=1100, ax=ax, label="Visited Nodes (DFS)"
        )
    if unreachable_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=unreachable_nodes, node_color=color_unreached,
            node_size=800, ax=ax, label="Unreachable Nodes"
        )

    # Create node label annotations showing traversal sequence
    order_map = {node: idx + 1 for idx, node in enumerate(result.traversal_order)}
    labels = {}
    for node in G.nodes():
        if node in order_map:
            labels[node] = f"{node}\n[#{order_map[node]}]"
        else:
            labels[node] = f"{node}\n[-]"

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax, font_color="#FFFFFF")

    # Partition edges: Tree edges vs Back edges vs Other
    tree_edges = [(u, v) for (u, v), e_type in result.edge_classification.items() if e_type == "TREE"]
    back_edges = result.cycles_detected
    other_edges = [e for e in G.edges() if e not in tree_edges and e not in back_edges]

    # Draw edges
    if other_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=other_edges, edge_color="#D1D5DB",
            arrows=True, arrowstyle="-|>", arrowsize=15, width=1.2, ax=ax,
            connectionstyle="arc3,rad=0.08"
        )
    if tree_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=tree_edges, edge_color="#2563EB",
            arrows=True, arrowstyle="-|>", arrowsize=18, width=2.8, ax=ax,
            label="DFS Spanning Tree Edges", connectionstyle="arc3,rad=0.05"
        )
    if back_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=back_edges, edge_color="#DC2626",
            arrows=True, arrowstyle="-|>", arrowsize=18, width=2.5, style="dashed",
            ax=ax, label="Back Edges (Cycles)", connectionstyle="arc3,rad=0.2"
        )

    plot_title = title or (
        f"Depth-First Search (DFS) Traversal ({result.mode.capitalize()} Mode)\n"
        f"Sequence: {result.format_traversal()} | Cycles: {len(result.cycles_detected)}"
    )
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[Success] Saved DFS visualization to: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def get_dfs_viva_reference() -> Dict[str, Any]:
    """Retrieve comprehensive Viva Voce questions, answers, and algorithmic comparisons.

    Returns:
        Structured dictionary covering essential AI examination topics for DFS.
    """
    return {
        "title": "Depth-First Search (DFS) Viva Voce & Theory Reference Guide",
        "syllabus_area": "Uninformed Search Techniques",
        "core_topics": [
            {
                "topic": "Stack vs Queue (LIFO vs FIFO)",
                "question": "What is the fundamental difference between the frontier data structure of DFS and BFS?",
                "answer": (
                    "DFS relies on a Last-In, First-Out (LIFO) Stack, while BFS relies on a First-In, "
                    "First-Out (FIFO) Queue.\n"
                    "- In DFS (LIFO Stack), the most recently discovered successor is expanded first. "
                    "This forces the search to explore along a single branch down to its deepest leaf "
                    "before backtracking to shallower unexpanded siblings.\n"
                    "- In BFS (FIFO Queue), nodes are expanded in the strict chronological order of their "
                    "discovery. This guarantees level-by-level exploration where all nodes at depth k are "
                    "expanded before any node at depth k+1.\n"
                    "- In Python, an iterative DFS stack is efficiently implemented with a standard list "
                    "using amortized O(1) append() and pop(), whereas BFS requires collections.deque "
                    "for amortized O(1) popleft()."
                ),
            },
            {
                "topic": "Why DFS is Not Optimal",
                "question": "Why is Depth-First Search considered non-optimal, even when step costs are uniform?",
                "answer": (
                    "An algorithm is optimal if it guarantees finding the lowest-cost solution path.\n"
                    "- DFS is NOT optimal because it commits deeply down the first branch it encounters. "
                    "If a goal state exists at depth 1 on the second branch, but the first branch contains "
                    "a deep path of length 1,000 leading to another goal, DFS will happily traverse all 1,000 "
                    "edges and return the deep solution without ever checking the shallow 1-edge goal.\n"
                    "- Even when all edge costs are identical (c = 1), BFS finds the shallowest goal "
                    "(optimal path length), whereas DFS finds an arbitrary path determined strictly by "
                    "neighbor exploration ordering."
                ),
            },
            {
                "topic": "Space Complexity Advantage of DFS over BFS",
                "question": "Why does DFS have a massive space complexity advantage over BFS?",
                "answer": (
                    "Space complexity is the primary practical bottleneck of BFS, and the primary strength of DFS.\n"
                    "- In Tree Search formulation with branching factor 'b' and maximum depth 'm':\n"
                    "  * BFS must store the entire current frontier level in memory, requiring O(b^d) space. "
                    "For b = 10 and d = 8, BFS must store 10^8 nodes (~10-100 GB RAM), exhausting memory.\n"
                    "  * DFS only needs to store the current active branch path from root to leaf, plus the "
                    "unexpanded sibling alternatives at each level along that path. This requires only O(b * m) "
                    "linear space! For b = 10 and m = 8, DFS stores only 10 * 8 = 80 nodes in memory.\n"
                    "- In Graph Search with finite vertices V and edges E:\n"
                    "  * DFS auxiliary memory is O(V) for the visited set and explicit/implicit stack."
                ),
            },
            {
                "topic": "Completeness of DFS",
                "question": "Under what conditions is DFS complete or incomplete?",
                "answer": (
                    "- Graph Search DFS (with a visited set): COMPLETE on finite state spaces, because the "
                    "visited set guarantees that no vertex is expanded twice, preventing cycles.\n"
                    "- Tree Search DFS (without a visited set): INCOMPLETE on cyclic graphs (it loops "
                    "infinitely) and INCOMPLETE on infinite state spaces (it can plunge down an infinite path "
                    "even if a shallow goal exists on an alternate branch).\n"
                    "- Remedy: Iterative Deepening Search (IDS / IDDFS) combines the O(b*d) linear memory "
                    "advantage of DFS with the completeness and optimality of BFS by incrementally increasing "
                    "the maximum search depth limit."
                ),
            },
            {
                "topic": "Cycle Detection via Back Edges",
                "question": "How does DFS detect cycles in directed graphs?",
                "answer": (
                    "In CLRS DFS edge classification, an edge (u, v) is a BACK EDGE if vertex 'v' is an "
                    "ancestor of vertex 'u' in the current active DFS recursion tree (i.e., 'v' is currently "
                    "on the active call stack / un-finished).\n"
                    "Theorem: A directed graph G contains a directed cycle IF AND ONLY IF a DFS traversal of G "
                    "encounters at least one back edge."
                ),
            },
        ],
        "comparison_matrix": {
            "headers": ["Metric / Property", "Breadth-First Search (BFS)", "Depth-First Search (DFS)"],
            "rows": [
                ["Frontier Data Structure", "FIFO Queue (collections.deque)", "LIFO Stack (list or call stack)"],
                ["Exploration Strategy", "Level-by-level (shallowest first)", "Branch-by-branch (deepest first)"],
                ["Time Complexity (Tree)", "O(b^d)", "O(b^m)"],
                ["Space Complexity (Tree)", "O(b^d) [Exponential bottleneck]", "O(b * m) [Linear space!]"],
                ["Time Complexity (Graph)", "O(V + E)", "O(V + E)"],
                ["Space Complexity (Graph)", "O(V)", "O(V)"],
                ["Completeness (Finite Graph)", "Yes (with visited set)", "Yes (with visited set)"],
                ["Completeness (Infinite/No Set)", "Yes (if b is finite)", "No (can get lost in infinite paths)"],
                ["Optimality (Uniform Cost)", "Yes (shallowest path)", "No (returns first found path)"],
                ["Optimality (Arbitrary Cost)", "No (requires UCS/Dijkstra)", "No"],
                ["Best Used For", "Shortest path, closest targets", "Maze solving, topological sort, cycle detection, exhaustive search"],
            ],
        },
    }
