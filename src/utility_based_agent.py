"""Utility-Based Agent Simulation with Terrain Costs and Optimization.

This module implements a Utility-Based Agent tasked with evaluating multiple
candidate paths to a goal in a weighted graph environment. Different terrain
types (e.g., Grass=1, Mud=3, Water=5) impose varying traversal costs on the edges.
The agent formulates a utility function U(path) = -1 * total_cost to evaluate
paths and selects the path that maximizes utility (strictly minimizing total cost).

Core AI Concepts:
- Utility-Based Agent: An intelligent agent that employs an explicit real-valued
  utility function U(s) to evaluate candidate actions, states, or paths.
  Unlike goal-based agents which only distinguish between achieving a goal (True/False),
  a utility-based agent quantifies *how desirable* a particular trajectory is,
  enabling trade-offs between speed, cost, risk, and resource consumption.
- Cost Minimization vs. Utility Maximization:
  In decision theory, minimizing a cost function C(p) is equivalent to maximizing
  a utility function U(p) = -C(p). This formulation adheres to the Principle of
  Maximum Expected Utility (MEU).
- Terrain Traversal Cost:
  Edges in the environment correspond to physical terrain types with distinct costs:
  * Grass: Cost 1 (firm ground, low resistance)
  * Sand: Cost 2 (loose ground, moderate resistance)
  * Mud: Cost 3 (sticky soil, heavy drag)
  * Rock: Cost 4 (uneven surface, obstacle navigation)
  * Water: Cost 5 (deep obstacle, high energy consumption)
- Battery Constraint (Optional Challenge):
  An energy budget that filters out infeasible paths whose total cost exceeds
  the agent's remaining battery capacity, regardless of their theoretical utility.

Complexity:
- Time Complexity: O(P * L) where P is the number of candidate paths and
  L is the average path length (number of nodes in the path).
- Space Complexity: O(1) auxiliary space beyond storing the evaluations.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Standard Terrain Cost Mapping
DEFAULT_TERRAIN_COSTS: Dict[str, float] = {
    "Grass": 1.0,
    "Sand": 2.0,
    "Mud": 3.0,
    "Rock": 4.0,
    "Water": 5.0,
}

# Type aliases for graph representation
Node = str
EdgeCost = Union[float, int]
TerrainInfo = Union[EdgeCost, str, Dict[str, Any]]
GraphDict = Dict[Node, Dict[Node, TerrainInfo]]
Path = Sequence[Node]


@dataclass(frozen=True)
class EdgeDetail:
    """Detailed record of a single edge transition along a path.

    Attributes:
        from_node: Starting node of the edge.
        to_node: Target node of the edge.
        cost: Traversal cost of this edge segment.
        terrain: Terrain label (e.g., 'Grass', 'Mud', 'Water', or 'Direct').
    """

    from_node: Node
    to_node: Node
    cost: float
    terrain: str = "Direct"


@dataclass
class PathEvaluation:
    """Evaluation metrics for a candidate path evaluated by the utility-based agent.

    Attributes:
        path: Ordered sequence of nodes from start to goal.
        total_cost: Summed terrain traversal cost across all edges.
        utility: Real-valued utility score, formulated as -1 * total_cost.
        edge_details: Granular breakdown of each edge in the path.
        is_valid: Whether the path exists continuously in the graph.
        is_feasible: Whether the path satisfies battery constraints (cost <= battery).
        rejection_reason: Explanation if the path is invalid or exceeds battery budget.
    """

    path: List[Node]
    total_cost: float
    utility: float
    edge_details: List[EdgeDetail] = field(default_factory=list)
    is_valid: bool = True
    is_feasible: bool = True
    rejection_reason: Optional[str] = None

    @property
    def path_str(self) -> str:
        """Return human-readable path representation e.g. 'A-C-D'."""
        return "-".join(self.path)

    @property
    def formatted_summary(self) -> str:
        """Formatted one-line summary of path evaluation."""
        status = "FEASIBLE" if self.is_feasible else f"REJECTED ({self.rejection_reason})"
        return (
            f"Path: {self.path_str:<12} | Cost: {self.total_cost:>4.1f} | "
            f"Utility: {self.utility:>5.1f} | Status: {status}"
        )


@dataclass
class UtilitySimulationResult:
    """Summary result of evaluating candidate paths for a Utility-Based Agent.

    Attributes:
        graph: The environment graph used for evaluation.
        evaluated_paths: All evaluated candidate paths with costs and utilities.
        selected_path: The optimal path selected by the agent (highest utility).
        battery_limit: Optional battery constraint applied during evaluation.
        feasible_paths: Subset of paths satisfying the battery constraint.
        rejected_paths: Subset of paths rejected due to battery depletion or invalidity.
    """

    graph: GraphDict
    evaluated_paths: List[PathEvaluation]
    selected_path: Optional[PathEvaluation]
    battery_limit: Optional[float] = None
    feasible_paths: List[PathEvaluation] = field(default_factory=list)
    rejected_paths: List[PathEvaluation] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Indicates whether an optimal feasible path was found."""
        return self.selected_path is not None


# Default university sample graph representing nodes and terrain edge costs
# Matches Sample Input:
# - Path A-B-D: A->B (Grass=1) + B->D (Water=5) = 6
# - Path A-C-D: A->C (Grass=1) + C->D (Mud=3) = 4
# - Path A-E-D: A->E (Mud=3) + E->D (Water=5) = 8 (Predefined Path 3)
DEFAULT_GRAPH: GraphDict = {
    "A": {
        "B": {"terrain": "Grass", "cost": 1.0},
        "C": {"terrain": "Grass", "cost": 1.0},
        "E": {"terrain": "Mud", "cost": 3.0},
    },
    "B": {
        "D": {"terrain": "Water", "cost": 5.0},
        "C": {"terrain": "Grass", "cost": 1.0},
    },
    "C": {
        "D": {"terrain": "Mud", "cost": 3.0},
        "B": {"terrain": "Grass", "cost": 1.0},
    },
    "E": {
        "D": {"terrain": "Water", "cost": 5.0},
    },
    "D": {},
}

# The 3 Predefined Paths specified in Task Requirement 3
DEFAULT_PREDEFINED_PATHS: List[List[Node]] = [
    ["A", "B", "D"],  # Cost: 1 + 5 = 6, Utility: -6.0
    ["A", "C", "D"],  # Cost: 1 + 3 = 4, Utility: -4.0 (Optimal)
    ["A", "E", "D"],  # Cost: 3 + 5 = 8, Utility: -8.0
]


def resolve_edge_cost(
    edge_data: TerrainInfo,
    terrain_costs: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Extract numeric traversal cost and terrain description from edge specification.

    Handles multiple representation formats:
    - Numeric cost: 4 or 4.0 -> cost: 4.0, terrain: "Direct"
    - Terrain string: "Mud" -> looks up cost in terrain_costs dict (e.g., 3.0)
    - Dictionary: {"cost": 3.0, "terrain": "Mud"} or {"terrain": "Mud"}

    Args:
        edge_data: Edge weight or terrain specification.
        terrain_costs: Lookup mapping for terrain names to numerical costs.

    Returns:
        Tuple of (numeric_cost, terrain_name).

    Raises:
        ValueError: If terrain type is unrecognized or cost is invalid.
    """
    costs = terrain_costs if terrain_costs is not None else DEFAULT_TERRAIN_COSTS

    if isinstance(edge_data, (int, float)):
        return float(edge_data), "Direct"

    if isinstance(edge_data, str):
        if edge_data in costs:
            return float(costs[edge_data]), edge_data
        raise ValueError(
            f"Unknown terrain type '{edge_data}'. Allowed terrains: {list(costs.keys())}"
        )

    if isinstance(edge_data, dict):
        terrain_label = edge_data.get("terrain", "Direct")
        if "cost" in edge_data:
            return float(edge_data["cost"]), str(terrain_label)
        if terrain_label in costs:
            return float(costs[terrain_label]), str(terrain_label)
        raise ValueError(
            f"Invalid edge dictionary {edge_data}; must contain 'cost' or recognized 'terrain'."
        )

    raise TypeError(f"Unsupported edge data type: {type(edge_data).__name__}")


def calculate_utility(total_cost: float, multiplier: float = -1.0) -> float:
    """Formulate agent utility as U(path) = -1 * total_cost.

    A utility-based agent evaluates paths based on desirability. In cost-minimization
    problems, utility is inversely related to expenditure. Formulating utility as
    the negative of total cost aligns cost minimization directly with utility maximization:
        argmax_p U(p) = argmin_p Cost(p)

    Args:
        total_cost: Sum of traversal costs along the path.
        multiplier: Linear scaling factor, defaults to -1.0.

    Returns:
        Real-valued utility score (e.g., cost 4.0 -> utility -4.0).
    """
    return multiplier * float(total_cost)


def calculate_path_cost(
    graph: GraphDict,
    path: Path,
    terrain_costs: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[EdgeDetail]]:
    """Calculate the total traversal cost along a path through the graph.

    Iterates sequentially through adjacent node pairs in the path, querying
    the graph dictionary for edge weights and terrain costs.

    Args:
        graph: Adjacency dictionary mapping node -> {neighbor -> edge_info}.
        path: Sequence of nodes defining the path (e.g. ['A', 'C', 'D']).
        terrain_costs: Optional custom mapping of terrain types to costs.

    Returns:
        Tuple of (total_cost, edge_details_list).

    Raises:
        ValueError: If the path contains fewer than 2 nodes, or an edge does not exist.
        KeyError: If a node is missing from the graph dictionary.
    """
    if len(path) < 2:
        raise ValueError(f"Path must contain at least 2 nodes; got {path}")

    total_cost = 0.0
    edge_details: List[EdgeDetail] = []

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        if u not in graph:
            raise KeyError(f"Node '{u}' is not present in graph.")

        neighbors = graph[u]
        if v not in neighbors:
            raise ValueError(
                f"No directed edge exists from node '{u}' to '{v}' in graph."
            )

        edge_cost, terrain_label = resolve_edge_cost(neighbors[v], terrain_costs)
        total_cost += edge_cost
        edge_details.append(
            EdgeDetail(
                from_node=u,
                to_node=v,
                cost=edge_cost,
                terrain=terrain_label,
            )
        )

    return total_cost, edge_details


def find_all_paths(
    graph: GraphDict,
    start: Node,
    goal: Node,
    visited: Optional[List[Node]] = None,
) -> List[List[Node]]:
    """Find all simple (acyclic) directed paths from start node to goal node.

    Explores the graph using Depth-First Search (DFS) with backtracking
    to enumerate all possible distinct valid paths between start and goal.

    Args:
        graph: Adjacency dictionary mapping node -> {neighbor -> edge_info}.
        start: Starting node identifier.
        goal: Target destination node identifier.
        visited: Internal accumulator of nodes visited along current branch.

    Returns:
        List of paths, where each path is an ordered list of nodes.
    """
    if visited is None:
        visited = [start]

    if start == goal:
        return [visited]

    if start not in graph:
        return []

    paths: List[List[Node]] = []
    for neighbor in graph[start]:
        if neighbor not in visited:
            extended_paths = find_all_paths(
                graph, neighbor, goal, visited + [neighbor]
            )
            paths.extend(extended_paths)

    return paths


class UtilityBasedAgent:
    """Utility-Based Agent for path evaluation and cost optimization.

    Attributes:
        graph: Weighted graph representing nodes and terrain edge costs.
        terrain_costs: Dictionary mapping terrain names to traversal costs.
        battery_limit: Optional energy threshold. Paths with cost > battery_limit
            are rejected regardless of utility.
        utility_multiplier: Scaling factor for utility calculation (default: -1.0).
    """

    def __init__(
        self,
        graph: Optional[GraphDict] = None,
        terrain_costs: Optional[Dict[str, float]] = None,
        battery_limit: Optional[float] = None,
        utility_multiplier: float = -1.0,
    ) -> None:
        """Initialize the Utility-Based Agent with graph and configuration.

        Args:
            graph: Graph dictionary. Defaults to DEFAULT_GRAPH if None.
            terrain_costs: Terrain-to-cost lookup. Defaults to DEFAULT_TERRAIN_COSTS.
            battery_limit: Maximum permissible cost. If set, paths exceeding
                this cost are marked infeasible.
            utility_multiplier: Multiplier for utility function U = multiplier * cost.
        """
        self.graph: GraphDict = graph if graph is not None else DEFAULT_GRAPH
        self.terrain_costs: Dict[str, float] = (
            dict(terrain_costs) if terrain_costs is not None else dict(DEFAULT_TERRAIN_COSTS)
        )
        self.battery_limit: Optional[float] = (
            float(battery_limit) if battery_limit is not None else None
        )
        self.utility_multiplier: float = utility_multiplier

    def utility_function(self, total_cost: float) -> float:
        """Compute utility score U(path) = -1 * total_cost."""
        return calculate_utility(total_cost, self.utility_multiplier)

    def evaluate_path(self, path: Path) -> PathEvaluation:
        """Evaluate a single candidate path for traversal cost and utility.

        Args:
            path: Ordered sequence of nodes (e.g., ['A', 'C', 'D']).

        Returns:
            PathEvaluation instance with cost, utility, validity, and feasibility.
        """
        path_list = list(path)
        try:
            total_cost, details = calculate_path_cost(
                self.graph, path_list, self.terrain_costs
            )
            utility = self.utility_function(total_cost)

            # Check battery constraint if configured
            is_feasible = True
            rejection_reason: Optional[str] = None
            if self.battery_limit is not None and total_cost > self.battery_limit:
                is_feasible = False
                rejection_reason = (
                    f"Cost {total_cost:.1f} exceeds battery limit {self.battery_limit:.1f}"
                )

            return PathEvaluation(
                path=path_list,
                total_cost=total_cost,
                utility=utility,
                edge_details=details,
                is_valid=True,
                is_feasible=is_feasible,
                rejection_reason=rejection_reason,
            )
        except (ValueError, KeyError) as err:
            return PathEvaluation(
                path=path_list,
                total_cost=float("inf"),
                utility=float("-inf"),
                edge_details=[],
                is_valid=False,
                is_feasible=False,
                rejection_reason=str(err),
            )

    def evaluate_all_paths(
        self, candidate_paths: Sequence[Path]
    ) -> List[PathEvaluation]:
        """Evaluate multiple candidate paths using dictionary iteration.

        Complexity:
            Time: O(P * L) where P is len(candidate_paths) and L is path length.
            Space: O(P) to store evaluation records.

        Args:
            candidate_paths: Sequence of paths to evaluate.

        Returns:
            List of PathEvaluation objects.
        """
        return [self.evaluate_path(p) for p in candidate_paths]

    def rank_paths(
        self,
        evaluations: Sequence[PathEvaluation],
        feasible_only: bool = True,
    ) -> List[PathEvaluation]:
        """Rank paths in descending order of utility using custom lambda sorting.

        Python Concept Tested:
            Custom sorting lambda: `sorted(evals, key=lambda ev: ev.utility, reverse=True)`

        Args:
            evaluations: Sequence of PathEvaluation records.
            feasible_only: If True, filters out paths rejected by battery/validity.

        Returns:
            Sorted list of PathEvaluation instances from highest to lowest utility.
        """
        candidates = [
            ev for ev in evaluations if (ev.is_valid and (not feasible_only or ev.is_feasible))
        ]
        # Custom sorting lambda function: sorts primarily by utility descending,
        # with total cost ascending as tie-breaker
        return sorted(
            candidates,
            key=lambda ev: (ev.utility, -ev.total_cost),
            reverse=True,
        )

    def select_best_path(
        self,
        candidate_paths: Sequence[Path],
    ) -> Optional[PathEvaluation]:
        """Select the path that maximizes utility while adhering to constraints.

        Python Concept Tested:
            Custom selection using `max()` with a key lambda:
            `max(feasible_paths, key=lambda ev: ev.utility)`

        Args:
            candidate_paths: Sequence of paths to evaluate and select from.

        Returns:
            Optimal PathEvaluation with highest utility, or None if no valid/feasible path exists.
        """
        if not candidate_paths:
            return None

        evaluations = self.evaluate_all_paths(candidate_paths)
        feasible = [ev for ev in evaluations if ev.is_valid and ev.is_feasible]

        if not feasible:
            return None

        # Maximize utility: max with key lambda
        return max(feasible, key=lambda ev: ev.utility)

    def run_simulation(
        self,
        candidate_paths: Optional[Sequence[Path]] = None,
    ) -> UtilitySimulationResult:
        """Run full simulation over candidate paths and package detailed results.

        Args:
            candidate_paths: Predefined paths to evaluate. Defaults to
                DEFAULT_PREDEFINED_PATHS (3 predefined paths).

        Returns:
            UtilitySimulationResult with all evaluated paths, selected path,
            feasible set, and rejected set.
        """
        paths = (
            candidate_paths
            if candidate_paths is not None
            else DEFAULT_PREDEFINED_PATHS
        )
        evaluations = self.evaluate_all_paths(paths)

        feasible_paths = [ev for ev in evaluations if ev.is_valid and ev.is_feasible]
        rejected_paths = [ev for ev in evaluations if not (ev.is_valid and ev.is_feasible)]

        # Select best path among feasible paths using custom lambda optimization
        selected: Optional[PathEvaluation] = None
        if feasible_paths:
            selected = max(feasible_paths, key=lambda ev: ev.utility)

        return UtilitySimulationResult(
            graph=self.graph,
            evaluated_paths=evaluations,
            selected_path=selected,
            battery_limit=self.battery_limit,
            feasible_paths=feasible_paths,
            rejected_paths=rejected_paths,
        )


def run_utility_based_simulation(
    graph: Optional[GraphDict] = None,
    paths: Optional[Sequence[Path]] = None,
    battery_limit: Optional[float] = None,
    terrain_costs: Optional[Dict[str, float]] = None,
) -> UtilitySimulationResult:
    """Convenience functional interface to execute a utility-based agent simulation.

    Args:
        graph: Optional graph dictionary. Uses DEFAULT_GRAPH if omitted.
        paths: Candidate paths to evaluate. Uses DEFAULT_PREDEFINED_PATHS if omitted.
        battery_limit: Optional battery constraint threshold.
        terrain_costs: Optional custom terrain cost mappings.

    Returns:
        UtilitySimulationResult dataclass containing evaluated and selected paths.
    """
    agent = UtilityBasedAgent(
        graph=graph,
        terrain_costs=terrain_costs,
        battery_limit=battery_limit,
    )
    return agent.run_simulation(paths)
