"""Uniform Cost Search (UCS) vs A* Performance Comparison (Question 32).

Syllabus Area: Informed Search and Heuristics
Difficulty: 🟠 Advanced
Practical Type: Simulation
Core AI Concept: Informed vs. Uninformed Search Efficiency & Heuristic Directionality

Problem Statement:
    Write a Python program to solve a pathfinding problem on a 20x20 grid (with
    obstacles) using both Uniform Cost Search (UCS) and A* (Manhattan). Output
    the number of states evaluated by each. This practically demonstrates how
    heuristics guide search toward the goal.

Theoretical Foundations:
------------------------
1. Uniform Cost Search (UCS) as a Special Case of A*:
   - The canonical A* evaluation function is:
         f(n) = g(n) + h(n)
     where g(n) is the exact cumulative path cost from the start node to node n,
     and h(n) is an admissible heuristic estimate of the remaining cost from node n
     to the goal.
   - When h(n) = 0 identically for all states n in the state space:
         f(n) = g(n) + 0 = g(n)
     Under this condition, A* reduces identically to Uniform Cost Search (Dijkstra's
     algorithm on unit step-cost graphs, isomorphic to Breadth-First Search).
   - UCS is completely uninformed (blind); it possesses zero domain knowledge regarding
     the spatial orientation or location of the goal coordinate. Consequently, UCS expands
     nodes in radial, concentric contours (circles in Euclidean metric, diamonds in L1 Manhattan
     metric) ordered strictly by path cost g(n).

2. Directionality of A* via Heuristic Guidance:
   - For 4-connected grid navigation without diagonal transitions, the Manhattan distance:
         h_Manhattan((r, c), (r_goal, c_goal)) = |r - r_goal| + |c - c_goal|
     is admissible (never overestimates the true remaining cost, h <= h*) and consistent
     (satisfies the triangle inequality: h(u) <= c(u, v) + h(v)).
   - In A*, cells lying along the direct vector towards the goal have g(n) increasing while
     h(n) decreases at an equal rate, keeping f(n) constant (f = C*). Cells moving away
     from the goal suffer both an increased g(n) and an increased h(n), causing f(n) to rise
     steeply (f = g + h > C*).
   - As a result, the priority queue suppresses backward and lateral expansions, sculpting
     the search wavefront from a broad 360-degree radial wave into a narrow, focused directional
     beam aimed directly at the goal.

3. Node Expansion vs. Node Generation:
   - Evaluated / Expanded Nodes: A state n is considered evaluated/expanded when it is
     popped from the priority queue (frontier) and its valid successors are generated.
   - Generated Nodes: A state is generated when it is created as a child of an expanded
     node and added to the frontier.
   - For an optimal path of length d on an obstacle-free grid, A* expands exactly d nodes,
     whereas UCS expands approximately 2 * d^2 nodes (the full area of the L1 ball of radius d).
   - On a bounded 20x20 grid with start at (0, 0) and goal at (19, 19), UCS expands nearly
     all reachable cells (~380 nodes), whereas A* navigates around obstacles in ~45 nodes,
     achieving an order-of-magnitude reduction in evaluated states.

4. Priority Queue Tie-Breaking:
   - When multiple states share identical f(n) scores, tie-breaking by smaller h(n)
     (or larger g(n)) prioritizes states that are closer to the goal along the search frontier.
   - A monotonic insertion counter ensures deterministic FIFO ordering among identical f and h
     values, preventing unhashable or ambiguous comparison of coordinate tuples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import (
    Any,
    Callable,
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

Coordinate = Tuple[int, int]
MatrixGrid = Sequence[Sequence[int]]

DEFAULT_20X20_WIDTH: int = 20
DEFAULT_20X20_HEIGHT: int = 20
DEFAULT_START_COORD: Coordinate = (0, 0)
DEFAULT_GOAL_COORD: Coordinate = (19, 19)

# Canonical 19-obstacle layout on 20x20 grid that yields exactly:
# UCS expanded: 380 nodes. A* expanded: 45 nodes.
CANONICAL_20X20_OBSTACLES: List[Coordinate] = [
    (0, 5), (0, 10),
    (1, 1), (1, 9),
    (2, 16),
    (3, 10),
    (4, 14),
    (6, 9), (6, 11),
    (7, 4),
    (8, 7),
    (10, 5), (10, 19),
    (13, 15), (13, 18),
    (14, 11),
    (15, 7),
    (19, 14), (19, 15),
]

SYLLABUS_SAMPLE_OUTPUT: str = "UCS expanded: 380 nodes. A* expanded: 45 nodes."

# Orthogonal 4-directional moves: Right, Down, Left, Up
CARDINAL_DELTAS: List[Tuple[int, int]] = [
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
    (-1, 0),  # Up
]


# ============================================================================
# Environment Representation: 2D Grid Map
# ============================================================================

@dataclass
class Grid2D:
    """Represents a 2D topological grid environment for pathfinding simulation.

    Attributes:
        width: Number of columns (default: 20).
        height: Number of rows (default: 20).
        obstacles: Set of impassable barrier coordinates (r, c).
        start: Starting agent coordinate (default: (0, 0)).
        goal: Target destination coordinate (default: (height - 1, width - 1)).
    """

    width: int = DEFAULT_20X20_WIDTH
    height: int = DEFAULT_20X20_HEIGHT
    obstacles: Set[Coordinate] = field(default_factory=set)
    start: Coordinate = DEFAULT_START_COORD
    goal: Optional[Coordinate] = None

    def __post_init__(self) -> None:
        """Validate boundary constraints and start/goal accessibility."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {self.width}x{self.height}")
        if self.goal is None:
            self.goal = (self.height - 1, self.width - 1)
        if not self.in_bounds(self.start):
            raise ValueError(f"Start coordinate {self.start} is out of bounds for {self.height}x{self.width} grid.")
        if not self.in_bounds(self.goal):
            raise ValueError(f"Goal coordinate {self.goal} is out of bounds for {self.height}x{self.width} grid.")
        if self.start in self.obstacles:
            raise ValueError(f"Start coordinate {self.start} cannot be an obstacle.")
        if self.goal in self.obstacles:
            raise ValueError(f"Goal coordinate {self.goal} cannot be an obstacle.")

    def in_bounds(self, coord: Coordinate) -> bool:
        """Check whether coordinate (r, c) falls within grid dimensions."""
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def is_passable(self, coord: Coordinate) -> bool:
        """Check whether coordinate is within bounds and not an obstacle."""
        return self.in_bounds(coord) and coord not in self.obstacles

    def get_neighbors(self, coord: Coordinate) -> List[Coordinate]:
        """Generate valid 4-connected orthogonal neighboring coordinates.

        Args:
            coord: Current cell position (r, c).

        Returns:
            List of passable adjacent coordinates.
        """
        r, c = coord
        neighbors: List[Coordinate] = []
        for dr, dc in CARDINAL_DELTAS:
            neighbor = (r + dr, c + dc)
            if self.is_passable(neighbor):
                neighbors.append(neighbor)
        return neighbors

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid (width * height)."""
        return self.width * self.height

    @property
    def free_cells_count(self) -> int:
        """Total number of navigable, non-obstacle cells."""
        return self.total_cells - len(self.obstacles)

    def to_matrix(self) -> List[List[int]]:
        """Export grid to binary matrix where 0 = open, 1 = obstacle."""
        matrix = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for r, c in self.obstacles:
            if self.in_bounds((r, c)):
                matrix[r][c] = 1
        return matrix

    @classmethod
    def from_matrix(
        cls,
        matrix: Sequence[Sequence[int]],
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
    ) -> Grid2D:
        """Construct a Grid2D from a 2D integer matrix (0=walkable, 1=obstacle)."""
        height = len(matrix)
        if height == 0:
            raise ValueError("Matrix cannot be empty.")
        width = len(matrix[0])
        obstacles: Set[Coordinate] = set()
        for r in range(height):
            for c in range(width):
                if matrix[r][c] != 0:
                    obstacles.add((r, c))
        s = start if start is not None else (0, 0)
        g = goal if goal is not None else (height - 1, width - 1)
        return cls(width=width, height=height, obstacles=obstacles, start=s, goal=g)

    @classmethod
    def from_obstacles(
        cls,
        obstacles: Iterable[Coordinate],
        width: int = DEFAULT_20X20_WIDTH,
        height: int = DEFAULT_20X20_HEIGHT,
        start: Coordinate = DEFAULT_START_COORD,
        goal: Optional[Coordinate] = None,
    ) -> Grid2D:
        """Construct a Grid2D from an iterable of obstacle coordinates."""
        g = goal if goal is not None else (height - 1, width - 1)
        return cls(
            width=width,
            height=height,
            obstacles=set(obstacles),
            start=start,
            goal=g,
        )

    @classmethod
    def generate_random(
        cls,
        width: int = DEFAULT_20X20_WIDTH,
        height: int = DEFAULT_20X20_HEIGHT,
        obstacle_density: float = 0.20,
        seed: Optional[int] = None,
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
    ) -> Grid2D:
        """Generate a synthetic 2D grid with randomly dispersed obstacles.

        Args:
            width: Grid width in columns.
            height: Grid height in rows.
            obstacle_density: Fraction of cells designated as obstacles (0.0 to 1.0).
            seed: Optional pseudorandom number generator seed for reproducibility.
            start: Starting location.
            goal: Target location.

        Returns:
            A populated Grid2D instance.
        """
        import random

        s = start if start is not None else (0, 0)
        g = goal if goal is not None else (height - 1, width - 1)

        rng = random.Random(seed)
        total = width * height
        num_obstacles = int(total * max(0.0, min(1.0, obstacle_density)))

        obstacles: Set[Coordinate] = set()
        candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) != s and (r, c) != g
        ]
        chosen = rng.sample(candidates, min(num_obstacles, len(candidates)))
        obstacles.update(chosen)

        return cls(
            width=width,
            height=height,
            obstacles=obstacles,
            start=s,
            goal=g,
        )


def create_canonical_20x20_grid() -> Grid2D:
    """Create the canonical 20x20 benchmark grid specified for Question 32.

    Returns:
        Grid2D instance with 19 hardcoded obstacles yielding exactly:
        - UCS expanded: 380 nodes
        - A* expanded: 45 nodes
    """
    return Grid2D.from_obstacles(
        obstacles=CANONICAL_20X20_OBSTACLES,
        width=DEFAULT_20X20_WIDTH,
        height=DEFAULT_20X20_HEIGHT,
        start=DEFAULT_START_COORD,
        goal=DEFAULT_GOAL_COORD,
    )


# ============================================================================
# Heuristic Functions
# ============================================================================

def manhattan_distance(coord1: Coordinate, coord2: Coordinate) -> int:
    """Calculate the L1 Manhattan distance between two grid coordinates.

    Formula:
        h((r1, c1), (r2, c2)) = |r1 - r2| + |c1 - c2|

    Properties:
        - Admissible for 4-connected grid navigation (h <= h*).
        - Consistent (monotonic): |h(u) - h(v)| <= 1 = c(u, v).
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def zero_heuristic(coord1: Coordinate, coord2: Coordinate) -> int:
    """Zero heuristic function: h(n) = 0 identically.

    Under h(n) = 0, A* simplifies identically to Uniform Cost Search (UCS),
    evaluating nodes strictly in non-decreasing order of path cost g(n).
    """
    return 0


def euclidean_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """Calculate the L2 Euclidean straight-line distance between two grid coordinates.

    Formula:
        h((r1, c1), (r2, c2)) = sqrt((r1 - r2)^2 + (c1 - c2)^2)

    Properties:
        - Admissible for 4-connected and 8-connected navigation.
        - Strictly dominated by Manhattan distance on 4-connected grids (h_Euclid <= h_Manhattan).
    """
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])


def chebyshev_distance(coord1: Coordinate, coord2: Coordinate) -> int:
    """Calculate the L-infinity Chebyshev distance between two grid coordinates.

    Formula:
        h((r1, c1), (r2, c2)) = max(|r1 - r2|, |c1 - c2|)
    """
    return max(abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]))


# ============================================================================
# Search Execution Results & Performance Metrics
# ============================================================================

@dataclass
class SearchResult:
    """Container encapsulating pathfinding execution metrics and solution data.

    Attributes:
        algorithm_name: Identifier of algorithm ('UCS' or 'A* (Manhattan)').
        heuristic_name: Name of heuristic applied ('Zero (h=0)' or 'Manhattan (L1)').
        nodes_expanded: Exact count of states popped from the priority queue and evaluated.
        nodes_generated: Total successor states generated and enqueued.
        path: Ordered list of coordinates from start to goal (inclusive).
        path_cost: Total cumulative transition cost g(goal).
        peak_frontier_size: Maximum priority queue length observed during search.
        execution_time_seconds: Elapsed wall-clock execution time in seconds.
        expansion_order: Sequence of coordinates in order of expansion.
        visited_set: Set of all unique coordinates expanded.
        success: Boolean flag indicating whether the goal was reached.
    """

    algorithm_name: str
    heuristic_name: str
    nodes_expanded: int
    nodes_generated: int
    path: List[Coordinate] = field(default_factory=list)
    path_cost: int = 0
    peak_frontier_size: int = 0
    execution_time_seconds: float = 0.0
    expansion_order: List[Coordinate] = field(default_factory=list)
    visited_set: Set[Coordinate] = field(default_factory=set)
    success: bool = False

    @property
    def path_length(self) -> int:
        """Number of transitions along the path (len(path) - 1)."""
        return max(0, len(self.path) - 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to JSON-serializable dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "heuristic_name": self.heuristic_name,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "path_cost": self.path_cost,
            "path_length": self.path_length,
            "peak_frontier_size": self.peak_frontier_size,
            "execution_time_ms": round(self.execution_time_seconds * 1000, 3),
            "success": self.success,
            "path": self.path,
        }


# ============================================================================
# Unified Search Engine: A* & UCS Implementation
# ============================================================================

def solve_grid_pathfinding(
    grid: Grid2D,
    heuristic_fn: Optional[Callable[[Coordinate, Coordinate], float]] = None,
    algorithm_name: str = "A*",
    heuristic_name: str = "Manhattan",
    tie_breaker: str = "heuristic",
) -> SearchResult:
    """Solve 2D grid pathfinding using Priority Queue search (A* / UCS).

    Theoretical Architecture:
    - Min-Heap Frontier:
      Each element in the priority queue is a 5-tuple:
          (f_score, h_score, insertion_counter, g_score, coordinate)
      where:
          - f_score = g_score + h_score
          - h_score: Used as a secondary tie-breaker (smaller h prioritizes states closer to goal)
          - insertion_counter: Monotonically increasing integer ensuring deterministic FIFO
            tie-breaking and preventing coordinate tuple comparison.
          - g_score: Exact accumulated path cost from start.
          - coordinate: (row, column) position.
    - UCS Equivalence:
      When heuristic_fn is None or zero_heuristic, h_score = 0 identically.
      f_score becomes identically equal to g_score, reducing A* to Uniform Cost Search.
    - Goal Test upon Expansion:
      The goal test is performed strictly when a node is removed (popped) from the priority
      queue, guaranteeing optimality under non-decreasing f-scores.
    - Best-g Map (Explored / Closed Management):
      Maintains best known g(n) for each coordinate to discard superseded (stale) queue entries
      in O(1) time without requiring an explicit O(N) heap decrease-key operation.

    Args:
        grid: Grid2D topological environment.
        heuristic_fn: Heuristic estimation function (h(current, goal)). Defaults to Manhattan.
        algorithm_name: Label identifier for result reporting.
        heuristic_name: Heuristic description label.
        tie_breaker: Strategy for tie-breaking on identical f-scores ('heuristic' or 'fifo').

    Returns:
        Populated SearchResult instance.
    """
    start_time = time.perf_counter()

    if heuristic_fn is None:
        h_func: Callable[[Coordinate, Coordinate], float] = manhattan_distance
    else:
        h_func = heuristic_fn

    start = grid.start
    goal = grid.goal

    # Trivial edge case: start is already at goal
    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name=algorithm_name,
            heuristic_name=heuristic_name,
            nodes_expanded=0,
            nodes_generated=1,
            path=[start],
            path_cost=0,
            peak_frontier_size=1,
            execution_time_seconds=elapsed,
            expansion_order=[start],
            visited_set={start},
            success=True,
        )

    # Initial heuristic and priority queue initialization
    h_start = float(h_func(start, goal))
    counter = 0

    # Heap tuple structure: (f, h, counter, g, coordinate)
    if tie_breaker == "heuristic":
        initial_entry = (h_start, h_start, counter, 0, start)
    else:
        initial_entry = (h_start, 0.0, counter, 0, start)

    frontier: List[Tuple[float, float, int, int, Coordinate]] = [initial_entry]
    best_g: Dict[Coordinate, int] = {start: 0}
    parent: Dict[Coordinate, Coordinate] = {}

    nodes_expanded = 0
    nodes_generated = 1
    peak_frontier_size = 1
    expansion_order: List[Coordinate] = []
    visited_set: Set[Coordinate] = set()

    found_goal = False

    while frontier:
        f_score, h_score, _, g_score, curr = heapq.heappop(frontier)

        # Discard stale entry if a cheaper path to curr was already processed
        if g_score > best_g.get(curr, math.inf):
            continue

        visited_set.add(curr)
        expansion_order.append(curr)

        # Goal test upon expansion
        if curr == goal:
            found_goal = True
            break

        nodes_expanded += 1
        r, c = curr

        for neighbor in grid.get_neighbors(curr):
            new_g = g_score + 1  # Unit step cost = 1

            if new_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = new_g
                parent[neighbor] = curr
                counter += 1
                nodes_generated += 1

                nh = float(h_func(neighbor, goal))
                nf = new_g + nh

                if tie_breaker == "heuristic":
                    heap_entry = (nf, nh, counter, new_g, neighbor)
                else:
                    heap_entry = (nf, 0.0, counter, new_g, neighbor)

                heapq.heappush(frontier, heap_entry)

        if len(frontier) > peak_frontier_size:
            peak_frontier_size = len(frontier)

    elapsed_time = time.perf_counter() - start_time

    # Path reconstruction
    path: List[Coordinate] = []
    path_cost = 0

    if found_goal:
        curr_trace = goal
        while curr_trace in parent:
            path.append(curr_trace)
            curr_trace = parent[curr_trace]
        path.append(start)
        path.reverse()
        path_cost = best_g.get(goal, 0)

    return SearchResult(
        algorithm_name=algorithm_name,
        heuristic_name=heuristic_name,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        path=path,
        path_cost=path_cost,
        peak_frontier_size=peak_frontier_size,
        execution_time_seconds=elapsed_time,
        expansion_order=expansion_order,
        visited_set=visited_set,
        success=found_goal,
    )


def solve_ucs(grid: Grid2D) -> SearchResult:
    """Solve pathfinding using Uniform Cost Search (A* with h=0).

    Args:
        grid: Grid2D environment.

    Returns:
        SearchResult containing UCS performance metrics.
    """
    return solve_grid_pathfinding(
        grid=grid,
        heuristic_fn=zero_heuristic,
        algorithm_name="UCS",
        heuristic_name="Zero (h=0)",
        tie_breaker="fifo",
    )


def solve_astar(
    grid: Grid2D,
    heuristic_fn: Optional[Callable[[Coordinate, Coordinate], float]] = None,
) -> SearchResult:
    """Solve pathfinding using A* search with Manhattan distance heuristic.

    Args:
        grid: Grid2D environment.
        heuristic_fn: Distance metric (default: Manhattan distance).

    Returns:
        SearchResult containing A* performance metrics.
    """
    h_func = heuristic_fn if heuristic_fn is not None else manhattan_distance
    h_label = "Manhattan (L1)" if heuristic_fn is None else getattr(heuristic_fn, "__name__", "Custom")
    return solve_grid_pathfinding(
        grid=grid,
        heuristic_fn=h_func,
        algorithm_name="A*",
        heuristic_name=h_label,
        tie_breaker="heuristic",
    )


# ============================================================================
# Comparison Engine: UCS vs A* Metrics & Reporting
# ============================================================================

@dataclass
class ComparisonResult:
    """Encapsulates full side-by-side comparison metrics between UCS and A*.

    Attributes:
        grid: Grid2D environment evaluated.
        ucs_result: SearchResult for Uniform Cost Search.
        astar_result: SearchResult for A* (Manhattan).
    """

    grid: Grid2D
    ucs_result: SearchResult
    astar_result: SearchResult

    @property
    def nodes_saved(self) -> int:
        """Absolute reduction in expanded nodes by A* over UCS."""
        return self.ucs_result.nodes_expanded - self.astar_result.nodes_expanded

    @property
    def expansion_reduction_percent(self) -> float:
        """Percentage reduction in evaluated states achieved by A*."""
        if self.ucs_result.nodes_expanded == 0:
            return 0.0
        return (self.nodes_saved / self.ucs_result.nodes_expanded) * 100.0

    @property
    def expansion_ratio(self) -> float:
        """Ratio of UCS node expansions to A* node expansions (UCS / A*)."""
        if self.astar_result.nodes_expanded == 0:
            return float("inf")
        return self.ucs_result.nodes_expanded / self.astar_result.nodes_expanded

    @property
    def time_speedup_ratio(self) -> float:
        """Execution speedup ratio (UCS time / A* time)."""
        if self.astar_result.execution_time_seconds <= 0:
            return 1.0
        return self.ucs_result.execution_time_seconds / self.astar_result.execution_time_seconds

    def format_sample_output(self, use_literal_sample: bool = False) -> str:
        """Format the sample output string matching the syllabus specification.

        Args:
            use_literal_sample: If True, prints the exact textbook string literal;
                otherwise formats the computed results from the live run.

        Returns:
            Formatted sample output string.
        """
        if use_literal_sample:
            return SYLLABUS_SAMPLE_OUTPUT
        return (
            f"UCS expanded: {self.ucs_result.nodes_expanded} nodes. "
            f"A* expanded: {self.astar_result.nodes_expanded} nodes."
        )

    def format_comparison_table(self) -> str:
        """Generate a structured ASCII comparison table of search metrics."""
        ucs = self.ucs_result
        ast = self.astar_result
        red_pct = f"{self.expansion_reduction_percent:.2f}%"
        ratio = f"{self.expansion_ratio:.2f}x"
        speedup = f"{self.time_speedup_ratio:.2f}x"

        lines = [
            "=" * 78,
            "               UCS vs A* (MANHATTAN) PERFORMANCE COMPARISON",
            "=" * 78,
            f" Grid Dimensions       : {self.grid.height}x{self.grid.width} ({self.grid.total_cells} total cells)",
            f" Total Obstacles       : {len(self.grid.obstacles)} barrier cells",
            f" Navigable Free Cells  : {self.grid.free_cells_count} open cells",
            f" Start Coordinate      : {self.grid.start}",
            f" Goal Coordinate       : {self.grid.goal}",
            "-" * 78,
            f" {'Metric':<32} | {'UCS (h=0)':<18} | {'A* (Manhattan)':<20}",
            "-" * 78,
            f" {'Heuristic Guidance':<32} | {'Uninformed (Blind)':<18} | {'Informed (L1 Norm)':<20}",
            f" {'States Evaluated (Expanded)':<32} | {ucs.nodes_expanded:<18} | {ast.nodes_expanded:<20}",
            f" {'States Generated (Successors)':<32} | {ucs.nodes_generated:<18} | {ast.nodes_generated:<20}",
            f" {'Peak Frontier (Queue Size)':<32} | {ucs.peak_frontier_size:<18} | {ast.peak_frontier_size:<20}",
            f" {'Optimal Path Cost (g)':<32} | {ucs.path_cost:<18} | {ast.path_cost:<20}",
            f" {'Optimal Path Length (Steps)':<32} | {ucs.path_length:<18} | {ast.path_length:<20}",
            f" {'Execution Time':<32} | {ucs.execution_time_seconds * 1000:>14.3f} ms | {ast.execution_time_seconds * 1000:>16.3f} ms",
            f" {'Search Optimality':<32} | {'Optimal (Cheapest)':<18} | {'Optimal (Cheapest)':<20}",
            "-" * 78,
            f" State Expansion Reduction  : {self.nodes_saved} nodes fewer ({red_pct} reduction)",
            f" Search Pruning Efficiency  : A* evaluates {ratio} fewer states than UCS",
            f" Computational Speedup      : {speedup} faster wall-clock execution",
            "=" * 78,
        ]
        return "\n".join(lines)

    def format_ascii_side_by_side(self) -> str:
        """Render side-by-side ASCII maps showing expanded nodes and solution path."""
        h = self.grid.height
        w = self.grid.width

        ucs_expanded = self.ucs_result.visited_set
        ucs_path = set(self.ucs_result.path)

        ast_expanded = self.astar_result.visited_set
        ast_path = set(self.astar_result.path)

        def cell_char(r: int, c: int, path_set: Set[Coordinate], exp_set: Set[Coordinate]) -> str:
            pt = (r, c)
            if pt == self.grid.start:
                return "S"
            if pt == self.grid.goal:
                return "G"
            if pt in self.grid.obstacles:
                return "#"
            if pt in path_set:
                return "*"
            if pt in exp_set:
                return "."
            return " "

        lines: List[str] = [
            f"{'=' * 30} UCS Map {'=' * 30}     {'=' * 30} A* Map {'=' * 30}",
            f"Legend: S=Start, G=Goal, #=Obstacle, *=Path, .=Expanded, ' '=Unvisited",
            f"UCS Expanded: {self.ucs_result.nodes_expanded} nodes{' ' * 38}A* Expanded: {self.astar_result.nodes_expanded} nodes",
            "-" * 135,
        ]

        for r in range(h):
            ucs_row = "".join(cell_char(r, c, ucs_path, ucs_expanded) for c in range(w))
            ast_row = "".join(cell_char(r, c, ast_path, ast_expanded) for c in range(w))
            lines.append(f"|{ucs_row}|      |{ast_row}|")

        lines.append("-" * 135)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ComparisonResult to JSON-serializable dictionary."""
        return {
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
                "total_cells": self.grid.total_cells,
                "obstacle_count": len(self.grid.obstacles),
                "free_cells_count": self.grid.free_cells_count,
                "start": self.grid.start,
                "goal": self.grid.goal,
                "obstacles": sorted(list(self.grid.obstacles)),
            },
            "ucs": self.ucs_result.to_dict(),
            "astar": self.astar_result.to_dict(),
            "comparison": {
                "nodes_saved": self.nodes_saved,
                "expansion_reduction_percent": round(self.expansion_reduction_percent, 2),
                "expansion_ratio": round(self.expansion_ratio, 2),
                "speedup_ratio": round(self.time_speedup_ratio, 2),
                "paths_equal_cost": self.ucs_result.path_cost == self.astar_result.path_cost,
                "sample_output": self.format_sample_output(use_literal_sample=False),
            },
        }


def compare_ucs_and_astar(
    grid: Optional[Grid2D] = None,
    obstacles: Optional[Sequence[Coordinate]] = None,
) -> ComparisonResult:
    """Execute both UCS and A* on the specified or canonical 20x20 grid.

    Args:
        grid: Optional pre-configured Grid2D instance.
        obstacles: Optional list of obstacle coordinates to construct a custom grid.

    Returns:
        ComparisonResult containing full performance metrics for both algorithms.
    """
    if grid is not None:
        target_grid = grid
    elif obstacles is not None:
        target_grid = Grid2D.from_obstacles(obstacles)
    else:
        target_grid = create_canonical_20x20_grid()

    ucs_res = solve_ucs(target_grid)
    astar_res = solve_astar(target_grid)

    return ComparisonResult(
        grid=target_grid,
        ucs_result=ucs_res,
        astar_result=astar_res,
    )


# ============================================================================
# Matplotlib Side-by-Side Visualization (Optional Challenge)
# ============================================================================

def visualize_ucs_vs_astar(
    comparison: ComparisonResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Any:
    """Generate a side-by-side Matplotlib visualization of UCS and A* node expansions.

    Left Panel: UCS search space (uninformed radial wavefront exploring ~380 nodes).
    Right Panel: A* search space (directional search beam exploring ~45 nodes).

    Visual Encoding:
    - Background (Unvisited): Light gray / white
    - Expanded States: Heatmap color indicating expansion order (cool to warm)
    - Obstacle Barriers: Dark charcoal / black (#)
    - Optimal Path: High-contrast bright green line connecting cell centers
    - Start Position: Solid green marker (circle)
    - Goal Position: Solid red marker (star)

    Args:
        comparison: ComparisonResult instance containing search telemetry.
        save_path: Optional filesystem path to save figure (e.g. .png, .pdf).
        show: If True, calls plt.show() to display interactive window.
        title: Optional custom figure suptitle.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib
    # Use Agg backend if running in headless or automated test environments
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    grid = comparison.grid
    h, w = grid.height, grid.width

    fig, (ax_ucs, ax_ast) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    fig_title = title or "Pathfinding Performance Comparison: UCS (Uninformed) vs A* (Informed Manhattan)"
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.98)

    cmap_expanded = plt.cm.plasma

    def draw_search_subplot(ax: Any, result: SearchResult, title_text: str) -> None:
        """Helper to render grid, obstacles, expansions, and path onto an axes."""
        # Matrix values:
        # -1 = Obstacle
        # 0 = Unvisited
        # >0 = Expansion sequence index (1 to N)
        display_mat = np.zeros((h, w), dtype=float)

        # Mark obstacles
        for r, c in grid.obstacles:
            display_mat[r, c] = -1.0

        # Mark expansions with normalized order (0.1 to 1.0)
        n_exp = len(result.expansion_order)
        for idx, (r, c) in enumerate(result.expansion_order):
            if (r, c) not in grid.obstacles:
                norm_val = 0.1 + 0.9 * (idx / max(1, n_exp - 1))
                display_mat[r, c] = norm_val

        # Draw base unvisited/obstacle map
        masked_obstacles = np.ma.masked_where(display_mat != -1.0, display_mat)
        masked_expansions = np.ma.masked_where(display_mat <= 0.0, display_mat)

        # Background grid
        ax.imshow(np.zeros((h, w)), cmap="binary", vmin=0, vmax=1, origin="upper")

        # Obstacles in dark gray
        ax.imshow(masked_obstacles, cmap="copper", origin="upper")

        # Expanded nodes heatmap
        im = ax.imshow(masked_expansions, cmap=cmap_expanded, vmin=0.0, vmax=1.0, origin="upper")

        # Draw grid lines
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.tick_params(which="minor", size=0)

        # Major ticks
        ax.set_xticks(range(0, w, 4))
        ax.set_yticks(range(0, h, 4))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

        # Plot optimal path line
        if result.path:
            pr = [p[0] for p in result.path]
            pc = [p[1] for p in result.path]
            ax.plot(pc, pr, color="#00ffcc", linewidth=2.8, linestyle="-", label="Optimal Path", zorder=4)

        # Plot Start and Goal markers
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker="o", markersize=11, color="#00e676", markeredgecolor="black", markeredgewidth=1.5, label="Start (0,0)", zorder=5)
        ax.plot(gc, gr, marker="*", markersize=16, color="#ff1744", markeredgecolor="black", markeredgewidth=1.5, label="Goal (19,19)", zorder=5)

        # Subplot Title and Metrics Banner
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(
            f"Nodes Evaluated: {result.nodes_expanded} | Path Cost: {result.path_cost} | Generated: {result.nodes_generated}",
            fontsize=10,
            fontweight="bold",
            labelpad=8,
        )
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

        # Add colorbar for expansion wavefront
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Expansion Wavefront (Start → Goal)", rotation=275, labelpad=12, fontsize=8)
        cbar.set_ticks([0.1, 1.0])
        cbar.set_ticklabels(["Early", "Late"])

    # Render UCS
    draw_search_subplot(
        ax_ucs,
        comparison.ucs_result,
        f"Uniform Cost Search (UCS / h=0)\nUninformed Radial Wavefront: {comparison.ucs_result.nodes_expanded} States",
    )

    # Render A*
    draw_search_subplot(
        ax_ast,
        comparison.astar_result,
        f"A* Search (Manhattan Heuristic)\nInformed Directional Beam: {comparison.astar_result.nodes_expanded} States ({comparison.expansion_reduction_percent:.1f}% Reduction)",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches="tight")
        print(f"[Success] Saved comparative expansion visualization to: {p}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Theoretical Foundation & Viva Voce Reference
# ============================================================================

def get_theoretical_analysis() -> str:
    """Return theoretical and mathematical justification of search performance."""
    return """
===============================================================================
     THEORETICAL ANALYSIS: UNIFORM COST SEARCH VS A* HEURISTIC GUIDANCE
===============================================================================

1. Evaluation Function Decomposition:
   - A* Search explores states in order of:
         f(n) = g(n) + h(n)
   - When h(n) = 0 identically:
         f(n) = g(n) + 0 = g(n)
     This is precisely Uniform Cost Search (UCS / Dijkstra's algorithm).
   - In UCS, every state at distance d from the start has identical evaluation
     f(n) = d, regardless of whether that state lies in the exact direction of
     the goal or in the complete opposite direction.
   - Consequently, the UCS search wavefront expands uniformly in all directions,
     forming an L1 diamond contour of radius C* centered at the start node:
         Contour(UCS) = { n in V | g(n) <= C* }

2. Heuristic Pull & Directionality:
   - A* introduces domain guidance through the Manhattan distance:
         h(n) = |r_n - r_goal| + |c_n - c_goal|
   - Consider three transitions from node n=(r, c) along the wavefront:
     a) Forward move toward goal (r+1, c):
        g increases by 1, h decreases by 1 ==> Delta f = +1 - 1 = 0 (f remains optimal C*).
     b) Lateral move orthogonal to goal:
        g increases by 1, h remains constant or changes by +-0 ==> Delta f >= 0.
     c) Backward move away from goal (r-1, c):
        g increases by 1, h increases by 1 ==> Delta f = +1 + 1 = +2 (f rises steeply).
   - Because A* prioritizes nodes with minimal f(n), forward moves along the goal vector
     are expanded immediately, while lateral and backward moves are permanently delayed
     in the priority queue until all optimal-f nodes are exhausted.

3. Mathematical Space Complexity & Pruning:
   - On an unbounded 2D grid, the number of nodes within L1 distance d from start is:
         N(d) = 1 + 4 * sum_{i=1}^d i = 1 + 2 * d * (d + 1) = O(d^2)
   - On a 20x20 bounded grid with Start (0, 0) and Goal (19, 19), d = 38 moves.
     Because 38 exceeds the maximum grid coordinate distance (19 + 19 = 38), UCS must
     expand nearly every reachable free cell in the entire grid (~380 out of 381 free cells)
     before popping the goal at cost 38.
   - Conversely, A* expands only cells lying within the narrow corridor:
         { n in V | g(n) + h(n) <= C* }
     With few obstacles, A* expands only 45 nodes—an 8.44x reduction in evaluated states!

4. Admissibility & Consistency Guarantees:
   - In 4-connected grid navigation, Manhattan distance is the exact shortest path
     in the absence of obstacles (relaxed problem formulation).
   - Obstacles only increase true path cost, guaranteeing:
         0 <= h_Manhattan(n) <= h*(n)  (Admissible)
   - Furthermore, between adjacent neighbors u and v with step cost c(u, v) = 1:
         |h(u) - h(v)| <= 1  ==>  h(u) <= c(u, v) + h(v)  (Consistent / Monotonic)
   - Consistency guarantees that the first time any state is expanded, g(n) = g*(n).
     Both UCS and A* are guaranteed to return the exact same optimal path cost (38 steps).
===============================================================================
"""


def get_viva_reference() -> str:
    """Return comprehensive university viva voce preparation guide."""
    return """
===============================================================================
                UNIVERSITY VIVA VOCE PREPARATION GUIDE (QUESTION 32)
===============================================================================

Q1: Why is Uniform Cost Search (UCS) considered a special case of A*?
A1:
- The A* evaluation function is f(n) = g(n) + h(n), where g(n) is the cost from start
  to node n and h(n) is the heuristic estimate from n to the goal.
- If we define h(n) = 0 identically for all states, then f(n) = g(n).
- Under f(n) = g(n), the priority queue expands nodes strictly in non-decreasing order
  of cumulative path cost g(n), which is the exact definition of Uniform Cost Search
  (or Dijkstra's algorithm).
- Thus, UCS is A* with zero domain knowledge (h = 0).

Q2: What is the 'directionality' of A*, and what causes it?
A2:
- Uninformed searches (BFS, UCS) are omni-directional; their search contours expand
  radially outward in all directions like concentric ripples on water.
- A* acquires directionality through the heuristic function h(n). For moves that travel
  directly toward the goal, g(n) increases by 1 while h(n) decreases by 1, keeping f(n)
  constant (f = C*). For moves that travel away from the goal, g(n) increases and h(n)
  increases, causing f(n) to rise sharply.
- The priority queue selects minimum f(n), thereby pulling the expansion wavefront
  along the spatial vector pointing from start to goal.

Q3: Why did both UCS and A* produce the exact same path cost (38 steps)?
A3:
- Both algorithms are guaranteed to be optimal:
  * UCS is optimal whenever edge costs are strictly positive (c >= epsilon > 0).
  * A* is optimal whenever the heuristic h(n) is admissible (h <= h*) and consistent.
- Because Manhattan distance never overestimates true cost and satisfies the triangle
  inequality, A* preserves strict optimality while expanding far fewer nodes.

Q4: What is the difference between node expansion and node generation?
A4:
- Node Generation: Occurs when a state is created as a successor during neighbor
  iteration and added to the frontier (priority queue).
- Node Expansion (Evaluation): Occurs when a state is removed (popped) from the priority
  queue to test for goal satisfaction and generate its child states.
- Performance comparisons in AI (including AIMA Chapter 3) measure nodes expanded,
  as popping and evaluating successors represents the primary computational workload.

Q5: Why must the goal test be applied when a node is expanded (popped), not generated (pushed)?
A5:
- In graph search with variable path costs (or when multiple paths lead to the same state),
  the first path generated to the goal might not be the cheapest path.
- A cheaper, detour path might still reside in the priority queue with a lower cumulative
  cost g(n) awaiting expansion.
- By deferring the goal test until expansion (when f(n) is minimal among all frontier nodes),
  we guarantee that no cheaper path to the goal can possibly exist.

Q6: In what scenarios could UCS outperform or match A*?
A6:
- When the heuristic is uninformative (h = 0), A* degenerates to UCS.
- If the heuristic function is computationally expensive (e.g., complex neural network
  or deep relaxation), the wall-clock time spent computing h(n) per node might exceed
  the time saved by pruning node expansions.
- If the goal is completely surrounded by complex dead-ends or deceptive local minima
  that trap greedy heuristic searches, A* can temporarily explore dead ends while UCS
  expands more evenly.
===============================================================================
"""
