"""Breadth-First Search (BFS) on User-Defined Directed Graph.

This module implements Breadth-First Search (BFS) for graph traversal and state-space
exploration, adhering to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA).

Theoretical Background & AI Concepts:
- Uninformed (Blind) Search:
    BFS explores the state space systematically level by level, without using domain-specific
    heuristics to estimate proximity to a goal.
- FIFO Frontier Queue:
    BFS uses a First-In, First-Out (FIFO) queue for the frontier (open list). In Python,
    `collections.deque` provides O(1) amortized enqueue (`append`) and dequeue (`popleft`),
    which is asymptotically superior to list-based `pop(0)` which incurs O(N) memory shifts.
- Visited / Explored Set:
    Graph-search BFS maintains a `visited` set to track all discovered vertices.
    This prevents infinite loops in graphs with directed or undirected cycles, and avoids
    redundant node expansions in directed acyclic graphs (DAGs) with multiple paths.
- Deterministic Tie-Breaking:
    When multiple successor nodes are discovered at the same parent, they are expanded
    in lexicographical (alphabetical) order, satisfying strict reproducibility requirements.
- Completeness and Optimality:
    - Completeness: BFS is complete on finite graphs (and on infinite graphs if branching
      factor b is finite).
    - Optimality: BFS is optimal when all step costs are identical/uniform (e.g., unit cost 1).
      It is guaranteed to find the shallowest path with the minimum number of edges.
- Level-Order / Depth Tracking (Optional Challenge):
    BFS naturally calculates the shortest distance (in number of edges / depth level)
    from the source node to every reachable node.

Complexity:
- Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges.
  Every vertex enters the frontier at most once, and every incident edge is evaluated once.
- Space Complexity: O(V) auxiliary memory for the frontier queue and visited set.
"""

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Benchmark and sample graphs for testing, education, and demonstrations
DEFAULT_SAMPLE_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": [],
    "D": [],
}

CYCLIC_BENCHMARK_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

BINARY_TREE_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": [],
    "F": [],
    "G": [],
}

DISCONNECTED_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": ["C"],
    "C": [],
    "D": ["E"],
    "E": [],
}


@dataclass(frozen=True)
class BFSStepRecord:
    """Immutable record capturing an individual node expansion step during BFS.

    Attributes:
        step_number: 1-based chronological expansion sequence index.
        current_node: Node identifier being popped from the frontier and expanded.
        depth_level: Distance (number of edges) of the current node from the source node.
        frontier_before: Snapshot of nodes in the FIFO frontier before expansion.
        frontier_after: Snapshot of nodes in the FIFO frontier after enqueuing valid neighbors.
        visited_nodes: Sorted snapshot of all nodes discovered and marked visited so far.
        neighbors_evaluated: All outgoing neighbors from the current node in the graph.
        neighbors_enqueued: Successor neighbors that were unvisited and appended to frontier.
        parent: The node from which the current node was first discovered (None for start).
    """

    step_number: int
    current_node: str
    depth_level: int
    frontier_before: List[str]
    frontier_after: List[str]
    visited_nodes: List[str]
    neighbors_evaluated: List[str]
    neighbors_enqueued: List[str]
    parent: Optional[str] = None

    def format_row(self) -> str:
        """Format the step into a structured table row for terminal presentation.

        Returns:
            Formatted row string.
        """
        frontier_str = "[" + ", ".join(self.frontier_after) + "]" if self.frontier_after else "[]"
        enqueued_str = "[" + ", ".join(self.neighbors_enqueued) + "]" if self.neighbors_enqueued else "-"
        parent_str = str(self.parent) if self.parent is not None else "ROOT"
        return (
            f"{self.step_number:<5} | {self.current_node:<6} | {self.depth_level:<5} | "
            f"{parent_str:<6} | {enqueued_str:<18} | {frontier_str}"
        )


@dataclass
class BFSResult:
    """Encapsulates the complete outcome of a Breadth-First Search graph traversal.

    Attributes:
        traversal_order: Chronological sequence of expanded node labels.
        levels: Mapping of node labels to their discrete distance / depth from start.
        nodes_by_level: Grouping of nodes partitioned by their depth level index.
        total_levels: Total count of discrete depth layers (depth + 1).
        max_depth: Maximum tree depth reached from the start node.
        parents: Mapping of each node to its parent node in the BFS spanning tree.
        steps: Chronological list of BFSStepRecord objects for fine-grained execution inspection.
        graph: The normalized adjacency list representation of the searched graph.
        start_node: Node where BFS traversal originated.
        visited_set: Set of all nodes reached and explored.
        unreachable_nodes: Nodes present in the graph definition that could not be reached.
        execution_time_ms: Measured execution wall-clock time in milliseconds.
    """

    traversal_order: List[str]
    levels: Dict[str, int]
    nodes_by_level: Dict[int, List[str]]
    total_levels: int
    max_depth: int
    parents: Dict[str, Optional[str]]
    steps: List[BFSStepRecord]
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

    def format_levels_summary(self) -> str:
        """Return a formatted level-by-level breakdown of the BFS tree.

        Returns:
            Multi-line string detailing nodes per depth layer.
        """
        lines = [
            f"Total Levels (Depth Count): {self.total_levels} | Maximum Depth: {self.max_depth}",
            "-" * 60,
        ]
        for lvl in sorted(self.nodes_by_level.keys()):
            nodes_at_level = ", ".join(self.nodes_by_level[lvl])
            lines.append(f"Level {lvl} (Depth {lvl}): [{nodes_at_level}]")
        return "\n".join(lines)

    def format_step_table(self) -> str:
        """Return a formatted ASCII execution trace table of the BFS search process.

        Returns:
            Table string showing step, expanded node, depth, parent, enqueued, and frontier.
        """
        headers = f"{'Step':<5} | {'Node':<6} | {'Depth':<5} | {'Parent':<6} | {'Enqueued':<18} | {'Frontier Queue':<20}"
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def shortest_path_to(self, target_node: str) -> Optional[List[str]]:
        """Reconstruct the minimum-hop shortest path from start_node to target_node.

        Because BFS traverses level-by-level, the path traced backward via the `parents`
        mapping is guaranteed to have the minimum number of edges (shortest path in an
        unweighted graph).

        Args:
            target_node: Destination node label.

        Returns:
            List of node labels representing the path from start to target,
            or None if the target is unreachable.
        """
        if target_node not in self.parents:
            return None

        path = []
        curr: Optional[str] = target_node
        while curr is not None:
            path.append(curr)
            curr = self.parents.get(curr)
        path.reverse()
        return path

    def format_ascii_tree(self) -> str:
        """Render a clean ASCII hierarchy of the BFS traversal spanning tree.

        Returns:
            Hierarchical tree representation string.
        """
        # Build children adjacency from parents map
        children_map: Dict[str, List[str]] = {}
        for child, parent in self.parents.items():
            if parent is not None:
                children_map.setdefault(parent, []).append(child)
        for p in children_map:
            children_map[p].sort()

        lines: List[str] = []

        def _build_tree(node: str, prefix: str = "", is_tail: bool = True) -> None:
            depth = self.levels.get(node, 0)
            connector = "└── " if is_tail else "├── "
            lines.append(f"{prefix}{connector}{node} (Level {depth})")
            children = children_map.get(node, [])
            for i, ch in enumerate(children):
                ch_tail = (i == len(children) - 1)
                extension = "    " if is_tail else "│   "
                _build_tree(ch, prefix + extension, ch_tail)

        lines.append(f"{self.start_node} (Root / Level 0)")
        root_children = children_map.get(self.start_node, [])
        for i, ch in enumerate(root_children):
            _build_tree(ch, "", i == len(root_children) - 1)

        return "\n".join(lines)


def normalize_graph(graph: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Normalize and validate a graph adjacency dictionary.

    Ensures all referenced nodes (including targets of directed edges) exist as keys
    in the adjacency list, stringifies identifiers, and converts sequences to lists.

    Args:
        graph: Raw input graph dictionary mapping nodes to sequences of neighbors.

    Returns:
        Clean, standardized dictionary mapping node strings to lists of neighbor strings.

    Raises:
        ValueError: If graph is empty or malformed.
    """
    if not isinstance(graph, dict):
        raise TypeError(f"Graph must be a dictionary (adjacency list), received {type(graph).__name__}")
    if not graph:
        raise ValueError("Graph cannot be empty. At least one node is required.")

    normalized: Dict[str, List[str]] = {}
    all_referenced_nodes: Set[str] = set()

    for node, neighbors in graph.items():
        node_str = str(node).strip()
        if not isinstance(neighbors, (list, tuple, set)):
            raise TypeError(f"Neighbors for node '{node_str}' must be a sequence, received {type(neighbors).__name__}")
        clean_neighbors = [str(n).strip() for n in neighbors]
        normalized[node_str] = clean_neighbors
        all_referenced_nodes.add(node_str)
        all_referenced_nodes.update(clean_neighbors)

    # Ensure every referenced destination node is present in keys with an empty list if not defined
    for ref_node in all_referenced_nodes:
        if ref_node not in normalized:
            normalized[ref_node] = []

    return normalized


def breadth_first_search(
    graph: Dict[str, Sequence[str]],
    start_node: str,
    sort_neighbors: bool = True,
) -> BFSResult:
    """Execute Breadth-First Search (BFS) on a directed graph starting from a source node.

    Adheres strictly to FIFO frontier dynamics (`collections.deque`), visited tracking to
    prevent cycles, deterministic alphabetical tie-breaking, and level/depth computation.

    Algorithm Steps:
        1. Normalize the graph adjacency list and validate that `start_node` exists.
        2. Initialize the FIFO frontier queue with `start_node`.
        3. Initialize `visited` set containing `start_node`.
        4. Initialize depth levels: `levels[start_node] = 0`.
        5. While frontier is not empty:
            a. Dequeue current node using `popleft()` (FIFO order).
            b. Append current node to `traversal_order`.
            c. Retrieve outgoing neighbors; sort alphabetically if `sort_neighbors` is True.
            d. For each neighbor not yet in `visited`:
                i. Mark neighbor as visited.
                ii. Record parent pointer: `parents[neighbor] = current_node`.
                iii. Record level: `levels[neighbor] = levels[current_node] + 1`.
                iv. Enqueue neighbor using `append()`.
        6. Compute depth metrics, level groupings, and unreachable vertices.

    Args:
        graph: Adjacency list dictionary mapping nodes to lists/tuples of successor nodes.
        start_node: Node identifier from which traversal begins.
        sort_neighbors: Whether to sort tied successor nodes alphabetically (default: True).

    Returns:
        BFSResult instance containing traversal order, levels, parents, steps, and diagnostics.

    Raises:
        KeyError: If `start_node` is not present in the graph.
        ValueError: If graph is empty or invalid.
    """
    t_start = time.perf_counter()
    clean_graph = normalize_graph(graph)
    start_str = str(start_node).strip()

    if start_str not in clean_graph:
        available = ", ".join(sorted(clean_graph.keys()))
        raise KeyError(f"Start node '{start_str}' not found in graph. Available nodes: [{available}]")

    frontier: deque[str] = deque([start_str])
    visited: Set[str] = {start_str}
    traversal_order: List[str] = []
    levels: Dict[str, int] = {start_str: 0}
    parents: Dict[str, Optional[str]] = {start_str: None}
    steps: List[BFSStepRecord] = []
    step_count = 0

    while frontier:
        step_count += 1
        frontier_before = list(frontier)
        current = frontier.popleft()
        traversal_order.append(current)
        current_level = levels[current]

        # Retrieve successors and sort if tie-breaking condition is active
        raw_neighbors = clean_graph.get(current, [])
        evaluated_neighbors = sorted(raw_neighbors) if sort_neighbors else list(raw_neighbors)

        enqueued_this_step: List[str] = []
        for neighbor in evaluated_neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
                levels[neighbor] = current_level + 1
                parents[neighbor] = current
                enqueued_this_step.append(neighbor)

        steps.append(
            BFSStepRecord(
                step_number=step_count,
                current_node=current,
                depth_level=current_level,
                frontier_before=frontier_before,
                frontier_after=list(frontier),
                visited_nodes=sorted(list(visited)),
                neighbors_evaluated=evaluated_neighbors,
                neighbors_enqueued=enqueued_this_step,
                parent=parents.get(current),
            )
        )

    # Group nodes by discrete depth level
    nodes_by_level: Dict[int, List[str]] = {}
    for node in traversal_order:
        lvl = levels[node]
        nodes_by_level.setdefault(lvl, []).append(node)

    total_levels = len(nodes_by_level)
    max_depth = max(levels.values()) if levels else 0

    # Determine unreachable nodes from the defined graph
    all_graph_nodes = set(clean_graph.keys())
    unreachable = sorted(list(all_graph_nodes - visited))
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return BFSResult(
        traversal_order=traversal_order,
        levels=levels,
        nodes_by_level=nodes_by_level,
        total_levels=total_levels,
        max_depth=max_depth,
        parents=parents,
        steps=steps,
        graph=clean_graph,
        start_node=start_str,
        visited_set=visited,
        unreachable_nodes=unreachable,
        execution_time_ms=elapsed_ms,
    )


def parse_graph_input(raw_input: Union[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Parse user-defined graph input from various formats.

    Supports:
        1. Python literal dict syntax: "{'A': ['B', 'C'], 'B': ['D']}"
        2. JSON dict syntax: '{"A": ["B", "C"], "B": ["D"]}'
        3. Compact adjacency syntax: "A: B, C; B: D; C:; D:"
        4. Edge list format: "A->B, A->C, B->D"

    Args:
        raw_input: String or dictionary representation of the graph.

    Returns:
        Normalized dictionary representing the graph adjacency list.

    Raises:
        ValueError: If parsing fails or graph syntax is invalid.
    """
    if isinstance(raw_input, dict):
        return normalize_graph(raw_input)

    if not isinstance(raw_input, str):
        raise TypeError(f"Expected str or dict for graph input, received {type(raw_input).__name__}")

    text = raw_input.strip()
    if not text:
        raise ValueError("Graph input string is empty.")

    # 1. Attempt JSON parsing (standard or with single quotes converted to double quotes)
    for candidate in (text, text.replace("'", '"')):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return normalize_graph(parsed)
        except Exception:
            pass

    # 2. Attempt Compact Adjacency List Format: "A: B, C; B: D" or "A: B, C | B: D"
    if ":" in text:
        graph: Dict[str, List[str]] = {}
        # Split on semicolon, pipe, or newline
        delimiters = [";", "|", "\n"]
        segments = [text]
        for d in delimiters:
            new_segments = []
            for seg in segments:
                new_segments.extend(seg.split(d))
            segments = new_segments

        has_valid_pair = False
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if ":" not in seg:
                continue
            parts = seg.split(":", 1)
            src = parts[0].strip()
            target_str = parts[1].strip()
            targets = [t.strip() for t in target_str.split(",") if t.strip()] if target_str else []
            graph[src] = targets
            has_valid_pair = True

        if has_valid_pair and graph:
            return normalize_graph(graph)

    # 3. Attempt Edge-List Format: "A->B, A->C, B->D"
    if "->" in text or "-->" in text:
        graph = {}
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        for token in tokens:
            arrow = "-->" if "-->" in token else "->"
            if arrow in token:
                u, v = token.split(arrow, 1)
                u = u.strip()
                v = v.strip()
                graph.setdefault(u, []).append(v)
                graph.setdefault(v, [])
        if graph:
            return normalize_graph(graph)

    raise ValueError(
        f"Unable to parse graph input: '{text}'.\n"
        "Supported formats:\n"
        "  - Python dict: {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\n"
        "  - Compact syntax: A: B, C; B: D; C:; D:\n"
        "  - Directed edges: A->B, A->C, B->D"
    )


class BreadthFirstSearch:
    """High-level object-oriented orchestrator for BFS graph traversal and visualization.

    Provides intuitive configuration, execution, inspection, and visualization
    methods for laboratory and classroom use.
    """

    def __init__(
        self,
        graph: Optional[Dict[str, Sequence[str]]] = None,
        sort_neighbors: bool = True,
    ) -> None:
        """Initialize the BreadthFirstSearch runner.

        Args:
            graph: Optional initial graph dictionary. Defaults to DEFAULT_SAMPLE_GRAPH.
            sort_neighbors: Whether to visit tie-broken neighbors alphabetically.
        """
        self.graph = normalize_graph(graph if graph is not None else DEFAULT_SAMPLE_GRAPH)
        self.sort_neighbors = sort_neighbors
        self.last_result: Optional[BFSResult] = None

    def run(self, start_node: str) -> BFSResult:
        """Execute BFS traversal from the given start node.

        Args:
            start_node: Node identifier where traversal begins.

        Returns:
            BFSResult instance containing detailed search results.
        """
        result = breadth_first_search(
            graph=self.graph,
            start_node=start_node,
            sort_neighbors=self.sort_neighbors,
        )
        self.last_result = result
        return result

    def get_traversal_string(self, start_node: str, delimiter: str = ", ") -> str:
        """Execute BFS and return the traversal order as a formatted string.

        Args:
            start_node: Source node identifier.
            delimiter: Separator between successive node names.

        Returns:
            Formatted string, e.g., 'A, B, C, D'.
        """
        res = self.run(start_node)
        return res.format_traversal(delimiter)

    def get_level_count(self, start_node: str) -> int:
        """Execute BFS and return the number of discrete depth levels (Optional Challenge).

        Args:
            start_node: Source node identifier.

        Returns:
            Count of depth levels (depth + 1).
        """
        res = self.run(start_node)
        return res.total_levels


def visualize_bfs(
    result: BFSResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Any:
    """Generate a publication-grade visualization of the BFS traversal and spanning tree.

    Highlights:
        - Source node highlighted with distinct gold border and star indicator.
        - BFS tree edges displayed as solid, prominent directed arrows.
        - Non-tree / cross edges rendered as dashed lines.
        - Node colors mapped by BFS depth level using a clean gradient.
        - Clear annotation of expansion sequence order (e.g., '1. A (L0)').

    Args:
        result: BFSResult object containing search metadata.
        save_path: Optional file path where the figure should be saved (.png/.svg).
        show: Whether to invoke plt.show() interactively.
        title: Optional custom figure title.

    Returns:
        The generated matplotlib Figure object, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("[Warning] networkx or matplotlib is not installed. Skipping graphical plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    G = nx.DiGraph()

    for node, neighbors in result.graph.items():
        G.add_node(node)
        for nbr in neighbors:
            G.add_edge(node, nbr)

    # Distinguish BFS tree edges from non-tree edges
    bfs_tree_edges: Set[Tuple[str, str]] = set()
    for child, parent in result.parents.items():
        if parent is not None:
            bfs_tree_edges.add((parent, child))

    non_tree_edges = [edge for edge in G.edges() if edge not in bfs_tree_edges]

    # Create layout: hierarchical if possible, otherwise spring layout
    try:
        # Hierarchical node positioning based on BFS levels
        pos: Dict[str, Tuple[float, float]] = {}
        for lvl, nodes in result.nodes_by_level.items():
            num_nodes = len(nodes)
            for idx, node in enumerate(nodes):
                x = (idx + 1) / (num_nodes + 1) * 10.0
                y = 10.0 - (lvl * 3.0)
                pos[node] = (x, y)
        # Position any unreachable nodes at the bottom
        for idx, u in enumerate(result.unreachable_nodes):
            pos[u] = ((idx + 1) * 2.0, 10.0 - ((result.total_levels + 1) * 3.0))
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Color palette based on levels
    level_colors = ["#2ECC71", "#3498DB", "#9B59B6", "#F39C12", "#E74C3C", "#1ABC9C"]
    node_colors = []
    labels = {}
    expansion_order_map = {node: i + 1 for i, node in enumerate(result.traversal_order)}

    for node in G.nodes():
        if node in result.levels:
            lvl = result.levels[node]
            color = level_colors[lvl % len(level_colors)]
            order = expansion_order_map.get(node, "?")
            labels[node] = f"{node}\n(Step {order}, L{lvl})"
        else:
            color = "#BDC3C7"  # Unreachable node
            labels[node] = f"{node}\n(Unreachable)"
        node_colors.append(color)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2400,
        edgecolors="#2C3E50",
        linewidths=2.5,
    )

    # Highlight root node
    if result.start_node in pos:
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[result.start_node],
            node_color="#2ECC71",
            node_size=2800,
            edgecolors="#F1C40F",
            linewidths=4.0,
        )

    # Draw BFS Tree Edges (Prominent solid green/blue)
    if bfs_tree_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=list(bfs_tree_edges),
            edge_color="#2980B9",
            width=2.8,
            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
        )

    # Draw Non-Tree Edges (Dashed grey)
    if non_tree_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=non_tree_edges,
            edge_color="#7F8C8D",
            width=1.5,
            style="dashed",
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
        )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_family="sans-serif",
    )

    chart_title = (
        title
        or f"Breadth-First Search (BFS) Traversal Tree\n"
        f"Start: '{result.start_node}' | Order: [{result.format_traversal()}] | "
        f"Total Levels: {result.total_levels} | Max Depth: {result.max_depth}"
    )
    ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=15)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight")
        print(f"[Success] Saved BFS visualization to: {out_file.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def get_viva_reference() -> Dict[str, Any]:
    """Retrieve structured theoretical notes and model answers for viva voce examinations.

    Returns:
        Dictionary containing key questions, concepts, and authoritative explanations.
    """
    return {
        "title": "Question 11: Breadth-First Search (BFS) Viva Voce Study Guide",
        "syllabus_area": "Uninformed Search Techniques",
        "core_topics": [
            "FIFO Queue Mechanism",
            "Completeness and Optimality",
            "Time and Space Complexity Derivation",
            "Graph Search vs. Tree Search",
            "BFS vs. DFS Comparison",
            "Level-Order Depth Tracking",
        ],
        "qna": [
            {
                "question": "What is the frontier data structure in BFS, how does the FIFO queue mechanism operate, and why is collections.deque preferred in Python?",
                "answer": (
                    "BFS requires a First-In, First-Out (FIFO) queue for its frontier. "
                    "`collections.deque` is implemented as a doubly linked list or block of arrays in CPython, "
                    "giving O(1) amortized time complexity for both `append()` (enqueue) and `popleft()` (dequeue). "
                    "In contrast, a Python `list` stores contiguous references, making `list.pop(0)` an O(N) operation "
                    "because all remaining elements must be shifted in memory, degrading overall BFS to O(V^2 + E)."
                ),
            },
            {
                "question": "Is Breadth-First Search complete and optimal? Under what conditions?",
                "answer": (
                    "Completeness: Yes. BFS is complete provided the branching factor b is finite, and either "
                    "the state space is finite (using graph search with an explored/visited set) or a shallowest goal exists.\n"
                    "Optimality: BFS is optimal if all step costs are identical/uniform (e.g., unit cost c = 1). "
                    "In that case, the shallowest goal is the least-cost goal. If step costs vary, BFS is NOT optimal, "
                    "and Uniform-Cost Search (Dijkstra's algorithm) must be used instead."
                ),
            },
            {
                "question": "Derive the Time Complexity of BFS on a graph and on a uniform search tree.",
                "answer": (
                    "Graph Formulation: O(V + E), where V is the number of vertices and E is the number of directed edges. "
                    "Every reachable vertex is inserted into the FIFO queue and visited set exactly once (O(V)), and its outgoing "
                    "adjacency list is scanned once during expansion, summing to all edges (O(E)).\n"
                    "Tree Formulation with branching factor b and goal depth d: 1 + b + b^2 + ... + b^d = O(b^d). "
                    "If testing for goal on expansion rather than generation, it expands up to level d+1, yielding O(b^(d+1))."
                ),
            },
            {
                "question": "What is the Space Complexity of BFS, and why is memory usually the critical bottleneck?",
                "answer": (
                    "Space Complexity is O(V) in graph search, or O(b^d) in tree search. "
                    "Every generated state must be retained in memory either in the frontier queue or the visited set. "
                    "At depth d, the frontier alone holds O(b^d) nodes. In typical AI problems where b = 10, "
                    "depth 8 requires storing ~100 million nodes, which exhausts RAM long before processor time runs out. "
                    "This makes memory the primary practical limiting factor of BFS compared to DFS (which requires only O(b*d))."
                ),
            },
            {
                "question": "Why is an explored/visited set necessary in graph search BFS?",
                "answer": (
                    "Without an explored/visited set (pure tree search), cycles in the graph (e.g., A -> B -> A) "
                    "cause the algorithm to enter an infinite loop. Furthermore, in directed acyclic graphs with redundant "
                    "paths (e.g., rectangular grids), tree search experiences exponential blowup by repeatedly expanding "
                    "the same state reached through different paths."
                ),
            },
            {
                "question": "How does BFS determine the number of levels (depth) of a graph?",
                "answer": (
                    "By initializing the start node with level 0 (depth = 0), and assigning each newly discovered "
                    "successor neighbor the level: level[child] = level[parent] + 1. "
                    "The total number of levels is the count of distinct depth values (or max_depth + 1)."
                ),
            },
        ],
    }
