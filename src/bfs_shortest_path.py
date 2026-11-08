"""BFS Shortest Path Constructor on Unweighted Directed/Undirected Graphs.

This module implements the Breadth-First Search (BFS) Shortest Path Constructor,
extending standard graph traversal into a goal-directed pathfinding algorithm
adhering to Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA).

Theoretical Background & AI Concepts:
- Path vs Traversal:
    - Traversal explores all reachable vertices systematically, producing a complete
      enumeration of the reachable state space (exploration order).
    - Path Construction finds a specific sequence of state transitions (edges)
      connecting a distinct start node to a target goal node:
      P = <v_0, v_1, v_2, ..., v_k> where v_0 = start, v_k = goal.
- Optimality in Unweighted Graphs:
    - In unweighted graphs, every edge has uniform step cost (c = 1).
    - The FIFO queue guarantees that nodes are discovered in non-decreasing order of
      their distance (depth / number of edges) from the start node: all nodes at depth d
      are expanded before any node at depth d + 1.
    - Mathematical Induction: The first time a goal node is generated, no path of shorter
      length can exist, because all shorter paths have already been explored and no longer
      paths have been considered. Thus, BFS is guaranteed to return the exact shortest path.
- Early Goal Testing vs Late Goal Testing:
    - Early Goal Testing (Node Generation): The goal test is evaluated immediately when a
      neighbor node is generated (before enqueuing into the FIFO frontier). Because all step
      costs are uniform, the first time the goal is generated, the shortest path is found.
      This saves an entire layer of node expansions (a factor of b in time and space).
    - Late Goal Testing (Node Expansion): The goal test is deferred until a node is dequeued
      and expanded. This is required in algorithms with non-uniform edge costs (like UCS / A*),
      but in BFS it performs unnecessary expansions.
- Path Reconstruction:
    - Parent Dictionary Backtracking (Recommended): Tracks a mapping `parent[child] = parent_node`.
      Upon reaching the goal, the path is reconstructed by walking backward from goal to start:
      goal -> parent[goal] -> ... -> start, and reversing the list.
      Auxiliary space: O(V).
    - Queue-Stored Paths: The FIFO queue stores full path lists `[v_0, ..., v_i]`.
      Simpler to write, but incurs O(V * b^d) space overhead due to path replication.
- Optional Challenge (All Shortest Paths):
    - Identifies all tied minimal-length paths between start and goal by maintaining
      multi-parent mappings for nodes reached at equal minimum depths.

Complexity:
- Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges.
  With early goal testing, execution stops immediately upon discovering the goal,
  frequently running in O(b^d) << O(V + E).
- Space Complexity: O(V) auxiliary memory for the FIFO frontier and parent mapping.
"""

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.breadth_first_search import normalize_graph, parse_graph_input


# Standard benchmark graphs for testing, education, and demonstrations
DEFAULT_SHORTEST_PATH_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": ["E"],
    "E": ["F"],
    "F": [],
}

CYCLIC_BENCHMARK_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

MULTI_PATH_DIAMOND_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}

MULTI_PATH_GRID_GRAPH: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["E", "F"],
    "D": ["G"],
    "E": ["G"],
    "F": ["G"],
    "G": [],
}

DISCONNECTED_SHORTEST_PATH_GRAPH: Dict[str, List[str]] = {
    "A": ["B"],
    "B": ["C"],
    "C": [],
    "D": ["E"],
    "E": ["F"],
    "F": [],
}


@dataclass(frozen=True)
class PathStepRecord:
    """Immutable record capturing a single step during BFS shortest path search.

    Attributes:
        step_number: 1-based chronological step sequence index.
        action_type: Category of action performed ('EXPAND', 'EARLY_GOAL_FOUND',
                     'LATE_GOAL_FOUND', 'ENQUEUE', 'START').
        current_node: Node being processed or expanded.
        depth_level: Distance (number of edges) from start node.
        frontier_snapshot: Snapshot of node labels in the FIFO queue.
        parent_map_snapshot: Copy of parent mappings recorded up to this step.
        candidate_neighbor: Neighbor node evaluated during this step (if applicable).
        goal_tested: Whether the goal test condition was executed in this step.
        goal_matched: Whether the goal test evaluated to True.
        explanation: Human-readable narrative of the step's logic.
    """

    step_number: int
    action_type: str
    current_node: str
    depth_level: int
    frontier_snapshot: List[str]
    parent_map_snapshot: Dict[str, Optional[str]]
    candidate_neighbor: Optional[str] = None
    goal_tested: bool = False
    goal_matched: bool = False
    explanation: str = ""

    def format_row(self) -> str:
        """Format the step record as a structured table row.

        Returns:
            Formatted row string.
        """
        frontier_str = "[" + ", ".join(self.frontier_snapshot) + "]" if self.frontier_snapshot else "[]"
        neighbor_str = self.candidate_neighbor if self.candidate_neighbor is not None else "-"
        goal_status = "MATCH" if self.goal_matched else ("TEST" if self.goal_tested else "-")
        return (
            f"{self.step_number:<5} | {self.action_type:<18} | {self.current_node:<6} | "
            f"{self.depth_level:<5} | {neighbor_str:<8} | {goal_status:<6} | {frontier_str}"
        )


@dataclass
class ShortestPathResult:
    """Encapsulates the complete outcome of a BFS shortest path search.

    Attributes:
        path: List of node labels from start to goal representing the shortest path,
              or None if no path exists.
        start_node: Originating source node.
        goal_node: Target destination node.
        cost_edges: Number of edges along the shortest path (-1 if unreachable).
        cost_nodes: Number of vertices along the shortest path (0 if unreachable).
        nodes_expanded: Total number of vertices popped from the queue and expanded.
        nodes_generated: Total number of successor vertices generated during search.
        parents: Mapping of discovered nodes to their direct parent in the BFS tree.
        steps: Chronological log of PathStepRecord events.
        graph: Normalized adjacency list of the graph searched.
        goal_found: True if a path to the goal was found, False otherwise.
        early_goal_test: True if early goal testing (at generation) was employed.
        tracking_mode: Mechanism used ('parent_dict' or 'queue_paths').
        all_shortest_paths: List of all tied shortest paths if multi-path mode was requested.
        execution_time_ms: Elapsed wall-clock time in milliseconds.
    """

    path: Optional[List[str]]
    start_node: str
    goal_node: str
    cost_edges: int
    cost_nodes: int
    nodes_expanded: int
    nodes_generated: int
    parents: Dict[str, Optional[str]]
    steps: List[PathStepRecord]
    graph: Dict[str, List[str]]
    goal_found: bool
    early_goal_test: bool = True
    tracking_mode: str = "parent_dict"
    all_shortest_paths: List[List[str]] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def format_path(self, arrow: str = " -> ") -> str:
        """Return the formatted shortest path string.

        Args:
            arrow: Separator string placed between sequential nodes.

        Returns:
            Formatted path string, or 'No path exists' if unreachable.
        """
        if self.path is None:
            return "No path exists (Goal is unreachable)"
        return arrow.join(self.path)

    def format_summary(self) -> str:
        """Produce a comprehensive summary of the shortest path search results.

        Returns:
            Formatted multiline summary string.
        """
        status_str = "SUCCESS (Goal Found)" if self.goal_found else "FAILURE (Goal Unreachable)"
        lines = [
            "=" * 70,
            "BFS SHORTEST PATH CONSTRUCTOR EXECUTION SUMMARY",
            "=" * 70,
            f"Start Node           : {self.start_node}",
            f"Goal Node            : {self.goal_node}",
            f"Status               : {status_str}",
            f"Shortest Path (List) : {self.path}",
            f"Shortest Path (Nav)  : {self.format_path()}",
            f"Path Cost (Edges)    : {self.cost_edges if self.cost_edges >= 0 else 'N/A'}",
            f"Path Length (Nodes)  : {self.cost_nodes}",
            f"Nodes Expanded       : {self.nodes_expanded}",
            f"Nodes Generated      : {self.nodes_generated}",
            f"Goal Testing Mode    : {'Early (on generation)' if self.early_goal_test else 'Late (on expansion)'}",
            f"Tracking Mechanism   : {self.tracking_mode}",
            f"Execution Time       : {self.execution_time_ms:.4f} ms",
        ]

        if self.all_shortest_paths:
            lines.append(f"All Shortest Paths ({len(self.all_shortest_paths)} found):")
            for idx, p in enumerate(self.all_shortest_paths, start=1):
                lines.append(f"  [{idx}] {' -> '.join(p)} (Length: {len(p) - 1} edges)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def format_step_table(self) -> str:
        """Format the execution step logs into a readable ASCII table.

        Returns:
            Formatted table string.
        """
        if not self.steps:
            return "No execution steps recorded."
        headers = f"{'Step':<5} | {'Action':<18} | {'Node':<6} | {'Depth':<5} | {'Cand.':<8} | {'Goal':<6} | Frontier"
        separator = "-" * len(headers)
        rows = [headers, separator]
        for step in self.steps:
            rows.append(step.format_row())
        return "\n".join(rows)

    def format_parent_map(self) -> str:
        """Format the reconstructed parent dictionary mapping.

        Returns:
            Formatted parent map string.
        """
        if not self.parents:
            return "Parent map is empty."
        items = []
        for child in sorted(self.parents.keys()):
            parent = self.parents[child]
            parent_str = parent if parent is not None else "None (Start)"
            items.append(f"{child} -> {parent_str}")
        return ", ".join(items)


def reconstruct_path_from_parents(
    parents: Dict[str, Optional[str]],
    start: str,
    goal: str,
) -> List[str]:
    """Reconstruct the sequence of vertices from start to goal using parent pointers.

    The path is reconstructed by walking backward from the goal node toward the
    start node using the parent dictionary, and then reversing the resulting list.

    Args:
        parents: Dictionary mapping each visited node to its predecessor.
        start: Originating start vertex.
        goal: Target destination vertex.

    Returns:
        List of vertex labels from start to goal.

    Raises:
        ValueError: If goal is not reachable from start or parent pointers are broken.
    """
    if goal not in parents:
        raise ValueError(f"Goal node '{goal}' is not present in parents dictionary.")

    path: List[str] = []
    current: Optional[str] = goal

    while current is not None:
        path.append(current)
        if current == start:
            break
        current = parents.get(current)

    if not path or path[-1] != start:
        raise ValueError(f"Could not reconstruct complete path: reached {path[-1]} instead of {start}.")

    path.reverse()
    return path


def bfs_shortest_path(
    graph: Dict[str, Sequence[str]],
    start: str,
    goal: str,
    early_goal_test: bool = True,
    tracking_mode: str = "parent_dict",
) -> ShortestPathResult:
    """Find the exact shortest path between start and goal using Breadth-First Search.

    Extends standard BFS traversal to solve the shortest-path construction problem
    on unweighted graphs.

    Features:
        - Exact shortest path guarantee: BFS explores layer by layer (increasing depth).
        - Early Goal Testing (default): Evaluates goal condition at node generation time,
          stopping immediately and preventing expansion of the remaining level.
        - Late Goal Testing (optional): Evaluates goal condition when dequeuing from frontier.
        - Tracking Modes:
            - 'parent_dict': Maintains a parent dictionary and reconstructs backward. O(V) space.
            - 'queue_paths': Enqueues full path lists into the FIFO queue.
        - Robust Edge Case Handling: Start == Goal, disconnected components, cycles,
          self-loops, unreachable goals, and nonexistent node validation.

    Args:
        graph: Adjacency list mapping node labels to sequences of outgoing neighbors.
        start: Starting source node label.
        goal: Target destination node label.
        early_goal_test: If True, tests goal at node generation; if False, tests at expansion.
        tracking_mode: Either 'parent_dict' (recommended) or 'queue_paths'.

    Returns:
        ShortestPathResult encapsulating path, costs, metrics, parents, and step log.

    Raises:
        ValueError: If start or goal node does not exist in graph definition.
    """
    t_start = time.perf_counter()
    normalized = normalize_graph(graph)

    if start not in normalized:
        raise ValueError(f"Start node '{start}' does not exist in the graph.")
    if goal not in normalized:
        raise ValueError(f"Goal node '{goal}' does not exist in the graph.")

    steps: List[PathStepRecord] = []
    step_counter = 0

    # Base Case: Start is already the Goal (Path length = 0 edges)
    if start == goal:
        step_counter += 1
        parents_map: Dict[str, Optional[str]] = {start: None}
        steps.append(
            PathStepRecord(
                step_number=step_counter,
                action_type="EARLY_GOAL_FOUND",
                current_node=start,
                depth_level=0,
                frontier_snapshot=[start],
                parent_map_snapshot=dict(parents_map),
                candidate_neighbor=None,
                goal_tested=True,
                goal_matched=True,
                explanation=f"Start node '{start}' is already the goal node.",
            )
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return ShortestPathResult(
            path=[start],
            start_node=start,
            goal_node=goal,
            cost_edges=0,
            cost_nodes=1,
            nodes_expanded=0,
            nodes_generated=1,
            parents=parents_map,
            steps=steps,
            graph=normalized,
            goal_found=True,
            early_goal_test=early_goal_test,
            tracking_mode=tracking_mode,
            all_shortest_paths=[[start]],
            execution_time_ms=elapsed_ms,
        )

    # Dispatch based on tracking mode
    if tracking_mode == "queue_paths":
        return _bfs_shortest_path_queue_tracking_internal(
            normalized, start, goal, early_goal_test, t_start
        )

    # Primary Implementation: Parent Dictionary Tracking
    frontier: deque[Tuple[str, int]] = deque([(start, 0)])
    visited: Set[str] = {start}
    parents: Dict[str, Optional[str]] = {start: None}
    nodes_expanded = 0
    nodes_generated = 1  # Start node generated

    step_counter += 1
    steps.append(
        PathStepRecord(
            step_number=step_counter,
            action_type="START",
            current_node=start,
            depth_level=0,
            frontier_snapshot=[start],
            parent_map_snapshot=dict(parents),
            candidate_neighbor=None,
            goal_tested=False,
            goal_matched=False,
            explanation=f"Initialized FIFO frontier with start node '{start}'.",
        )
    )

    goal_reached = False
    reconstructed_path: Optional[List[str]] = None

    while frontier:
        current, depth = frontier.popleft()
        nodes_expanded += 1

        # Late Goal Testing: Evaluate goal condition upon expansion
        if not early_goal_test:
            step_counter += 1
            is_goal = current == goal
            steps.append(
                PathStepRecord(
                    step_number=step_counter,
                    action_type="LATE_GOAL_TEST",
                    current_node=current,
                    depth_level=depth,
                    frontier_snapshot=[node for node, _ in frontier],
                    parent_map_snapshot=dict(parents),
                    candidate_neighbor=None,
                    goal_tested=True,
                    goal_matched=is_goal,
                    explanation=f"Late goal test on popped node '{current}': {'MATCH' if is_goal else 'No match'}.",
                )
            )
            if is_goal:
                goal_reached = True
                reconstructed_path = reconstruct_path_from_parents(parents, start, goal)
                break

        # Log node expansion
        step_counter += 1
        steps.append(
            PathStepRecord(
                step_number=step_counter,
                action_type="EXPAND",
                current_node=current,
                depth_level=depth,
                frontier_snapshot=[node for node, _ in frontier],
                parent_map_snapshot=dict(parents),
                candidate_neighbor=None,
                goal_tested=False,
                goal_matched=False,
                explanation=f"Expanding node '{current}' at depth {depth}. Evaluating outgoing edges.",
            )
        )

        neighbors = normalized.get(current, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = current
                nodes_generated += 1

                # Early Goal Testing: Evaluate goal condition immediately upon generation
                if early_goal_test:
                    step_counter += 1
                    is_goal = neighbor == goal
                    steps.append(
                        PathStepRecord(
                            step_number=step_counter,
                            action_type="EARLY_GOAL_FOUND" if is_goal else "ENQUEUE",
                            current_node=current,
                            depth_level=depth + 1,
                            frontier_snapshot=[node for node, _ in frontier] + [neighbor],
                            parent_map_snapshot=dict(parents),
                            candidate_neighbor=neighbor,
                            goal_tested=True,
                            goal_matched=is_goal,
                            explanation=(
                                f"Generated neighbor '{neighbor}'. Goal test: MATCH! Immediate termination."
                                if is_goal
                                else f"Generated neighbor '{neighbor}'. Enqueued into FIFO frontier."
                            ),
                        )
                    )
                    if is_goal:
                        goal_reached = True
                        reconstructed_path = reconstruct_path_from_parents(parents, start, goal)
                        break
                    frontier.append((neighbor, depth + 1))
                else:
                    # Late goal testing mode: just enqueue without goal check
                    frontier.append((neighbor, depth + 1))
                    step_counter += 1
                    steps.append(
                        PathStepRecord(
                            step_number=step_counter,
                            action_type="ENQUEUE",
                            current_node=current,
                            depth_level=depth + 1,
                            frontier_snapshot=[node for node, _ in frontier],
                            parent_map_snapshot=dict(parents),
                            candidate_neighbor=neighbor,
                            goal_tested=False,
                            goal_matched=False,
                            explanation=f"Generated unvisited neighbor '{neighbor}'. Enqueued into FIFO frontier.",
                        )
                    )

        if goal_reached:
            break

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if goal_reached and reconstructed_path is not None:
        cost_edges = len(reconstructed_path) - 1
        cost_nodes = len(reconstructed_path)
        all_paths = [reconstructed_path]
    else:
        cost_edges = -1
        cost_nodes = 0
        all_paths = []

    return ShortestPathResult(
        path=reconstructed_path,
        start_node=start,
        goal_node=goal,
        cost_edges=cost_edges,
        cost_nodes=cost_nodes,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        parents=parents,
        steps=steps,
        graph=normalized,
        goal_found=goal_reached,
        early_goal_test=early_goal_test,
        tracking_mode=tracking_mode,
        all_shortest_paths=all_paths,
        execution_time_ms=elapsed_ms,
    )


def _bfs_shortest_path_queue_tracking_internal(
    normalized: Dict[str, List[str]],
    start: str,
    goal: str,
    early_goal_test: bool,
    t_start: float,
) -> ShortestPathResult:
    """Internal implementation storing full path sequences inside the FIFO queue.

    Task Requirement: Queue must store paths (lists of nodes).

    Args:
        normalized: Normalized adjacency dictionary.
        start: Start node label.
        goal: Goal node label.
        early_goal_test: Whether to test goal at node generation or expansion.
        t_start: Performance counter timestamp from entry point.

    Returns:
        ShortestPathResult encapsulating the search outcome.
    """
    frontier: deque[List[str]] = deque([[start]])
    visited: Set[str] = {start}
    parents: Dict[str, Optional[str]] = {start: None}
    steps: List[PathStepRecord] = []
    step_counter = 1
    nodes_expanded = 0
    nodes_generated = 1

    steps.append(
        PathStepRecord(
            step_number=step_counter,
            action_type="START",
            current_node=start,
            depth_level=0,
            frontier_snapshot=[start],
            parent_map_snapshot=dict(parents),
            candidate_neighbor=None,
            goal_tested=False,
            goal_matched=False,
            explanation=f"Queue-Tracking initialized with path [{start}].",
        )
    )

    goal_reached = False
    reconstructed_path: Optional[List[str]] = None

    while frontier:
        path = frontier.popleft()
        current = path[-1]
        depth = len(path) - 1
        nodes_expanded += 1

        if not early_goal_test:
            step_counter += 1
            is_goal = current == goal
            steps.append(
                PathStepRecord(
                    step_number=step_counter,
                    action_type="LATE_GOAL_TEST",
                    current_node=current,
                    depth_level=depth,
                    frontier_snapshot=[p[-1] for p in frontier],
                    parent_map_snapshot=dict(parents),
                    candidate_neighbor=None,
                    goal_tested=True,
                    goal_matched=is_goal,
                    explanation=f"Late goal test on path ending at '{current}'.",
                )
            )
            if is_goal:
                goal_reached = True
                reconstructed_path = path
                break

        step_counter += 1
        steps.append(
            PathStepRecord(
                step_number=step_counter,
                action_type="EXPAND",
                current_node=current,
                depth_level=depth,
                frontier_snapshot=[p[-1] for p in frontier],
                parent_map_snapshot=dict(parents),
                candidate_neighbor=None,
                goal_tested=False,
                goal_matched=False,
                explanation=f"Expanding path: {' -> '.join(path)}.",
            )
        )

        neighbors = normalized.get(current, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = current
                nodes_generated += 1
                new_path = path + [neighbor]

                if early_goal_test:
                    step_counter += 1
                    is_goal = neighbor == goal
                    steps.append(
                        PathStepRecord(
                            step_number=step_counter,
                            action_type="EARLY_GOAL_FOUND" if is_goal else "ENQUEUE",
                            current_node=current,
                            depth_level=len(new_path) - 1,
                            frontier_snapshot=[p[-1] for p in frontier] + [neighbor],
                            parent_map_snapshot=dict(parents),
                            candidate_neighbor=neighbor,
                            goal_tested=True,
                            goal_matched=is_goal,
                            explanation=(
                                f"Generated neighbor '{neighbor}'. Goal test: MATCH! Path found: {new_path}"
                                if is_goal
                                else f"Enqueued extended path ending at '{neighbor}' into frontier."
                            ),
                        )
                    )
                    if is_goal:
                        goal_reached = True
                        reconstructed_path = new_path
                        break
                    frontier.append(new_path)
                else:
                    frontier.append(new_path)
                    step_counter += 1
                    steps.append(
                        PathStepRecord(
                            step_number=step_counter,
                            action_type="ENQUEUE",
                            current_node=current,
                            depth_level=len(new_path) - 1,
                            frontier_snapshot=[p[-1] for p in frontier],
                            parent_map_snapshot=dict(parents),
                            candidate_neighbor=neighbor,
                            goal_tested=False,
                            goal_matched=False,
                            explanation=f"Enqueued extended path ending at '{neighbor}'.",
                        )
                    )

        if goal_reached:
            break

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if goal_reached and reconstructed_path is not None:
        cost_edges = len(reconstructed_path) - 1
        cost_nodes = len(reconstructed_path)
        all_paths = [reconstructed_path]
    else:
        cost_edges = -1
        cost_nodes = 0
        all_paths = []

    return ShortestPathResult(
        path=reconstructed_path,
        start_node=start,
        goal_node=goal,
        cost_edges=cost_edges,
        cost_nodes=cost_nodes,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        parents=parents,
        steps=steps,
        graph=normalized,
        goal_found=goal_reached,
        early_goal_test=early_goal_test,
        tracking_mode="queue_paths",
        all_shortest_paths=all_paths,
        execution_time_ms=elapsed_ms,
    )


def find_all_shortest_paths(
    graph: Dict[str, Sequence[str]],
    start: str,
    goal: str,
) -> List[List[str]]:
    """Find all possible shortest paths between start and goal (Optional Challenge).

    When multiple paths have identical minimum edge counts (e.g. in diamond graphs
    or symmetric grids), standard BFS only returns the single path encountered first.
    This algorithm discovers ALL paths of minimal length by tracking multi-parent
    relationships for nodes reached at equal minimal distance.

    Algorithm:
        1. BFS assigns exact minimum distance `dist[v]` from start to every node.
        2. A multi-parent dictionary `parents_list[v]` records every parent `u`
           satisfying `dist[v] == dist[u] + 1`.
        3. Search terminates once the level strictly exceeds the goal's depth.
        4. Backtracking DFS builds all paths from goal back to start.

    Args:
        graph: Adjacency list representation of the unweighted graph.
        start: Starting source node label.
        goal: Target destination node label.

    Returns:
        List of paths (each path is a list of node labels), sorted lexicographically.
        Returns an empty list if goal is unreachable.

    Raises:
        ValueError: If start or goal node does not exist in graph definition.
    """
    normalized = normalize_graph(graph)

    if start not in normalized:
        raise ValueError(f"Start node '{start}' does not exist in the graph.")
    if goal not in normalized:
        raise ValueError(f"Goal node '{goal}' does not exist in the graph.")

    if start == goal:
        return [[start]]

    # Map each node to its shortest distance from start
    dist: Dict[str, int] = {start: 0}
    # Map each node to all parents that reach it via a shortest edge
    multi_parents: Dict[str, List[str]] = {start: []}

    frontier: deque[str] = deque([start])
    goal_depth: Optional[int] = None

    while frontier:
        curr = frontier.popleft()
        curr_dist = dist[curr]

        # If we have already reached the goal and current node is at or beyond goal depth,
        # we do not need to expand its children further.
        if goal_depth is not None and curr_dist >= goal_depth:
            continue

        for neighbor in normalized.get(curr, []):
            # Neighbor reached for the first time
            if neighbor not in dist:
                dist[neighbor] = curr_dist + 1
                multi_parents[neighbor] = [curr]
                frontier.append(neighbor)
                if neighbor == goal and goal_depth is None:
                    goal_depth = curr_dist + 1
            # Neighbor reached at the same optimal distance via an alternative branch
            elif dist[neighbor] == curr_dist + 1:
                if curr not in multi_parents[neighbor]:
                    multi_parents[neighbor].append(curr)

    if goal not in dist:
        return []

    # Backtrack all paths from goal to start
    results: List[List[str]] = []

    def _backtrack(current_node: str, accumulator: List[str]) -> None:
        if current_node == start:
            results.append(list(reversed(accumulator)))
            return
        for parent_node in multi_parents.get(current_node, []):
            accumulator.append(parent_node)
            _backtrack(parent_node, accumulator)
            accumulator.pop()

    _backtrack(goal, [goal])
    results.sort()
    return results


def compare_goal_testing_strategies(
    graph: Dict[str, Sequence[str]],
    start: str,
    goal: str,
) -> Dict[str, Any]:
    """Compare Early Goal Testing vs Late Goal Testing on the exact same graph problem.

    Demonstrates the efficiency advantage of early goal testing in uninformed BFS:
    - Early testing evaluates goal upon node generation (skipping frontier enqueue).
    - Late testing evaluates goal upon node expansion (requiring queue dequeue).

    Args:
        graph: Adjacency list representation of graph.
        start: Start node label.
        goal: Destination node label.

    Returns:
        Dictionary containing comparative metrics and results.
    """
    early_res = bfs_shortest_path(graph, start, goal, early_goal_test=True)
    late_res = bfs_shortest_path(graph, start, goal, early_goal_test=False)

    expansions_saved = late_res.nodes_expanded - early_res.nodes_expanded
    generations_diff = late_res.nodes_generated - early_res.nodes_generated

    return {
        "start_node": start,
        "goal_node": goal,
        "shortest_path": early_res.path,
        "path_cost_edges": early_res.cost_edges,
        "early_goal_test": {
            "nodes_expanded": early_res.nodes_expanded,
            "nodes_generated": early_res.nodes_generated,
            "execution_time_ms": early_res.execution_time_ms,
            "total_steps": len(early_res.steps),
        },
        "late_goal_test": {
            "nodes_expanded": late_res.nodes_expanded,
            "nodes_generated": late_res.nodes_generated,
            "execution_time_ms": late_res.execution_time_ms,
            "total_steps": len(late_res.steps),
        },
        "expansions_saved": max(0, expansions_saved),
        "generations_saved": max(0, generations_diff),
    }


def compare_tracking_mechanisms(
    graph: Dict[str, Sequence[str]],
    start: str,
    goal: str,
) -> Dict[str, Any]:
    """Compare Parent Dictionary Backtracking vs Queue Path Storing.

    Evaluates:
        1. Parent Dictionary (`parent[child] = parent`): O(V) auxiliary memory.
        2. Queue Storing Paths (`queue.append(path + [nbr])`): O(V * b^d) memory.

    Args:
        graph: Graph adjacency list.
        start: Start node label.
        goal: Destination node label.

    Returns:
        Comparative dictionary of performance and correctness metrics.
    """
    parent_dict_res = bfs_shortest_path(graph, start, goal, tracking_mode="parent_dict")
    queue_paths_res = bfs_shortest_path(graph, start, goal, tracking_mode="queue_paths")

    return {
        "start": start,
        "goal": goal,
        "identical_paths": parent_dict_res.path == queue_paths_res.path,
        "path": parent_dict_res.path,
        "parent_dict": {
            "execution_time_ms": parent_dict_res.execution_time_ms,
            "nodes_expanded": parent_dict_res.nodes_expanded,
            "nodes_generated": parent_dict_res.nodes_generated,
            "space_complexity": "O(V) auxiliary parent dictionary",
        },
        "queue_paths": {
            "execution_time_ms": queue_paths_res.execution_time_ms,
            "nodes_expanded": queue_paths_res.nodes_expanded,
            "nodes_generated": queue_paths_res.nodes_generated,
            "space_complexity": "O(V * b^d) queue path duplications",
        },
    }


class BFSShortestPathConstructor:
    """Object-Oriented BFS Shortest Path Constructor.

    Provides a clean, reusable interface encapsulating graph state, search configuration,
    and path reconstruction operations.

    Example:
        >>> constructor = BFSShortestPathConstructor(DEFAULT_SHORTEST_PATH_GRAPH)
        >>> result = constructor.find_shortest_path("A", "F")
        >>> result.path
        ['A', 'C', 'F']
        >>> result.cost_edges
        2
    """

    def __init__(
        self,
        graph: Union[str, Dict[str, Any]],
        early_goal_test: bool = True,
        tracking_mode: str = "parent_dict",
    ) -> None:
        """Initialize the BFS Shortest Path Constructor.

        Args:
            graph: Graph specification as dict, JSON, or compact adjacency string.
            early_goal_test: Whether to perform early goal testing (default True).
            tracking_mode: 'parent_dict' or 'queue_paths'.
        """
        self.raw_graph = graph
        self.graph = parse_graph_input(graph)
        self.early_goal_test = early_goal_test
        self.tracking_mode = tracking_mode

    def find_shortest_path(self, start: str, goal: str) -> ShortestPathResult:
        """Execute BFS to find and reconstruct the shortest path between start and goal.

        Args:
            start: Start node label.
            goal: Goal node label.

        Returns:
            ShortestPathResult encapsulating path and diagnostic metrics.
        """
        return bfs_shortest_path(
            self.graph,
            start=start,
            goal=goal,
            early_goal_test=self.early_goal_test,
            tracking_mode=self.tracking_mode,
        )

    def find_all_shortest_paths(self, start: str, goal: str) -> List[List[str]]:
        """Find all tied minimal-edge shortest paths between start and goal.

        Args:
            start: Start node label.
            goal: Goal node label.

        Returns:
            List of shortest path lists.
        """
        return find_all_shortest_paths(self.graph, start=start, goal=goal)

    def compare_goal_tests(self, start: str, goal: str) -> Dict[str, Any]:
        """Compare early vs late goal testing on this graph instance."""
        return compare_goal_testing_strategies(self.graph, start, goal)


def visualize_shortest_path(
    result: ShortestPathResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Any:
    """Generate a publication-grade visualization highlighting the shortest path.

    Highlights:
        - Source node highlighted in emerald green with start marker.
        - Destination goal node highlighted in crimson red with goal marker.
        - Shortest path edges rendered with prominent thick ruby/coral arrows.
        - Non-path edges displayed with muted dashed gray arrows.
        - Unvisited / unreachable nodes displayed with subtle neutral styling.

    Args:
        result: ShortestPathResult instance from a completed search.
        save_path: Optional file path to save the generated image (.png / .svg).
        show: Whether to display the plot interactively with plt.show().
        title: Optional custom figure title.

    Returns:
        Matplotlib Figure object, or None if matplotlib/networkx is missing.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("[Warning] matplotlib or networkx is not installed. Skipping plot generation.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    G = nx.DiGraph()

    for node, neighbors in result.graph.items():
        G.add_node(node)
        for nbr in neighbors:
            G.add_edge(node, nbr)

    # Identify edges on the shortest path
    path_edges: Set[Tuple[str, str]] = set()
    if result.path and len(result.path) > 1:
        for idx in range(len(result.path) - 1):
            path_edges.add((result.path[idx], result.path[idx + 1]))

    non_path_edges = [e for e in G.edges() if e not in path_edges]

    # Node coloring based on path roles
    node_colors: List[str] = []
    path_set = set(result.path or [])
    for node in G.nodes():
        if node == result.start_node:
            node_colors.append("#2ECC71")  # Green for Start
        elif node == result.goal_node:
            node_colors.append("#E74C3C")  # Red for Goal
        elif node in path_set:
            node_colors.append("#F39C12")  # Orange for path intermediates
        elif node in result.parents:
            node_colors.append("#3498DB")  # Blue for visited non-path nodes
        else:
            node_colors.append("#BDC3C7")  # Gray for unreached nodes

    # Determine layout
    try:
        pos = nx.spring_layout(G, seed=42, k=1.2)
    except Exception:
        pos = nx.shell_layout(G)

    # Draw non-path edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=non_path_edges,
        edge_color="#BDC3C7",
        width=1.5,
        style="dashed",
        arrows=True,
        arrowsize=14,
        ax=ax,
        connectionstyle="arc3,rad=0.08",
    )

    # Draw shortest path edges prominently
    if path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=list(path_edges),
            edge_color="#E74C3C",
            width=3.5,
            arrows=True,
            arrowsize=22,
            ax=ax,
            connectionstyle="arc3,rad=0.08",
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=1200,
        edgecolors="#2C3E50",
        linewidths=2.0,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=11,
        font_weight="bold",
        font_color="#FFFFFF",
        ax=ax,
    )

    fig_title = (
        title
        or f"BFS Shortest Path Constructor: {result.start_node} -> {result.goal_node}\n"
        f"Path: {result.format_path()} | Cost: {result.cost_edges} edges"
    )
    ax.set_title(fig_title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    # Legend annotations
    legend_text = (
        "Green: Start | Red: Goal | Orange: Path Intermediates | Blue: Visited Non-Path | Gray: Unreached\n"
        f"Algorithm: BFS with {result.tracking_mode} | Goal Test: {'Early' if result.early_goal_test else 'Late'}"
    )
    plt.figtext(
        0.5,
        0.02,
        legend_text,
        wrap=True,
        horizontalalignment="center",
        fontsize=9,
        bbox={"facecolor": "whitesmoke", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[Success] Saved shortest path visualization to: {out_path.resolve()}")

    if show:
        plt.show()

    return fig


def get_viva_reference() -> Dict[str, Any]:
    """Provide comprehensive academic viva-voce reference guide for Question 13.

    Covers:
        1. Path vs Traversal
        2. Early goal testing vs Late goal testing
        3. Why BFS finds the shortest path in unweighted graphs
        4. Time and Space Complexity analysis
        5. Path reconstruction techniques (Parent dict vs queue storing paths)
        6. Handling unreachable goals and disconnected components

    Returns:
        Structured dictionary of viva questions, concise summaries, and in-depth answers.
    """
    return {
        "question_number": 13,
        "title": "BFS Shortest Path Constructor",
        "syllabus_area": "Uninformed Search Techniques",
        "topics": {
            "path_vs_traversal": {
                "question": "What is the key difference between a Graph Traversal and a Shortest Path Constructor?",
                "summary": "Traversal visits all reachable nodes; Path Constructor finds a minimal-cost sequence between two specific vertices.",
                "in_depth_answer": (
                    "1. Objective:\n"
                    "   - Graph Traversal (e.g., standard BFS/DFS) is an exploratory process that systematically "
                    "visits every reachable vertex in a component to discover graph structure or collect nodes.\n"
                    "   - Path Construction is a goal-directed search designed to find a specific valid sequence "
                    "of alternating vertices and edges connecting a specified start node v_0 to a goal node v_k.\n\n"
                    "2. Termination:\n"
                    "   - Traversal terminates only when the frontier becomes completely empty (all reachable nodes visited).\n"
                    "   - Path Construction terminates as soon as the goal test succeeds, returning the reconstructed path.\n\n"
                    "3. Output:\n"
                    "   - Traversal outputs an ordering/list of all explored vertices.\n"
                    "   - Path Constructor outputs an ordered sequence representing the shortest path: ['A', 'C', 'F']."
                ),
            },
            "early_vs_late_goal_testing": {
                "question": "What is the difference between Early Goal Testing and Late Goal Testing, and why does BFS use early testing?",
                "summary": "Early testing checks the goal when a node is generated (saves an entire level b); late testing checks when popped.",
                "in_depth_answer": (
                    "1. Early Goal Testing (Generation Time):\n"
                    "   - The goal test is performed as soon as a successor node is generated by expanding its parent, "
                    "before the child is enqueued into the FIFO frontier.\n"
                    "   - In unweighted graphs (all step costs = 1), the first time a goal node is generated, "
                    "it is mathematically guaranteed to be at the minimum possible depth.\n"
                    "   - Immediate termination saves expanding all remaining nodes at depth d and avoids generating "
                    "up to b^(d+1) successor nodes, yielding a massive b-factor speedup in both time and memory!\n\n"
                    "2. Late Goal Testing (Expansion Time):\n"
                    "   - The goal test is deferred until the node is popped from the front of the queue.\n"
                    "   - Required for non-uniform cost search (e.g., Dijkstra, Uniform-Cost Search, A*), where "
                    "generating a node does not guarantee the cheapest path has been discovered.\n"
                    "   - In uniform-cost BFS, late goal testing is redundant and unnecessarily expands an extra layer."
                ),
            },
            "why_bfs_finds_shortest_path": {
                "question": "Why is BFS guaranteed to find the shortest path in an unweighted graph?",
                "summary": "FIFO queue enforces level-by-level exploration; all paths of length d are explored before any path of length d+1.",
                "in_depth_answer": (
                    "1. Uniform Edge Cost:\n"
                    "   - In an unweighted graph, all edge weights are equal (unit cost c = 1). Therefore, path cost "
                    "is identical to path depth (number of edges / hops).\n\n"
                    "2. FIFO Frontier Invariant:\n"
                    "   - A FIFO queue guarantees that vertices are processed strictly in monotonic non-decreasing order "
                    "of their distance from the start node: depth 0, then depth 1, then depth 2, ..., depth d.\n\n"
                    "3. Inductive Proof of Optimality:\n"
                    "   - Assume the shortest path from start to goal has length d*.\n"
                    "   - BFS explores every vertex at depth 0, 1, ..., d* - 1 before any vertex at depth d*.\n"
                    "   - Therefore, no path of length > d* can ever be returned before a path of length d*, and no path "
                    "of length < d* exists (otherwise the goal would have been reached at that earlier depth).\n"
                    "   - Hence, the first path constructed to the goal is guaranteed to be optimal."
                ),
            },
            "path_reconstruction_techniques": {
                "question": "Compare Parent Dictionary Backtracking with Storing Full Paths in the Queue.",
                "summary": "Parent dictionary uses O(V) space by mapping child to parent; queue path storing duplicates lists taking O(V * b^d) space.",
                "in_depth_answer": (
                    "1. Parent Dictionary Backtracking (Standard / Optimal):\n"
                    "   - Maintain `parents: Dict[str, Optional[str]]` mapping child -> parent.\n"
                    "   - Each discovered node stores exactly one predecessor pointer.\n"
                    "   - Auxiliary space is strictly O(V).\n"
                    "   - Reconstruction takes O(d) time by tracing from goal back to start: "
                    "`curr = goal; while curr: path.append(curr); curr = parents[curr]` followed by `path.reverse()`.\n\n"
                    "2. Storing Full Paths in Queue:\n"
                    "   - The queue stores complete lists: `[['A'], ['A', 'B'], ['A', 'C']]`.\n"
                    "   - For each neighbor, creates `path + [neighbor]`.\n"
                    "   - Highly intuitive, but wastes significant memory: each queued path consumes O(d) space, "
                    "leading to O(d * b^d) frontier memory consumption."
                ),
            },
            "complexity_analysis": {
                "question": "What are the Time and Space Complexities of BFS Shortest Path Constructor?",
                "summary": "Time: O(V + E) worst case, O(b^d) with early goal test. Space: O(V) for visited set and parent dictionary.",
                "in_depth_answer": (
                    "1. Time Complexity: O(V + E)\n"
                    "   - In the worst case (goal unreachable or at maximum depth), BFS visits every vertex V once "
                    "and examines every incident directed edge E once.\n"
                    "   - With Early Goal Testing on a graph with branching factor b and shallowest goal at depth d, "
                    "time complexity is O(b^d), which is often vastly smaller than O(V + E).\n\n"
                    "2. Space Complexity: O(V) or O(b^d)\n"
                    "   - Auxiliary space is dominated by the FIFO frontier queue and the `parents` dictionary / `visited` set.\n"
                    "   - Each vertex appears in the visited set and parent map at most once: O(V).\n"
                    "   - In terms of branching factor b and solution depth d, frontier memory is O(b^d)."
                ),
            },
            "handling_unreachable_goals": {
                "question": "How does the algorithm detect and handle cases where no path exists?",
                "summary": "When the FIFO frontier becomes empty without finding the goal, return None with clear status.",
                "in_depth_answer": (
                    "1. Condition for Failure:\n"
                    "   - If all reachable nodes in the connected component containing the start node are expanded, "
                    "and the FIFO frontier empties (`len(frontier) == 0`) without reaching the goal, no path exists.\n\n"
                    "2. Graceful Return:\n"
                    "   - The function returns a `ShortestPathResult` with `path=None`, `goal_found=False`, and `cost_edges=-1`.\n"
                    "   - This prevents crashes and allows callers to differentiate between an empty graph, a 0-hop path "
                    "(start == goal), and an unreachable destination."
                ),
            },
        },
    }
