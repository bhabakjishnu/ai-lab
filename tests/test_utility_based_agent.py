"""Unit test suite for Question 5: Utility-Based Agent Simulation.

Covers:
- Terrain edge cost resolution (numbers, terrain strings, dicts, edge errors)
- Utility function formulation U(path) = -1 * total_cost
- Path cost summation and dictionary iteration
- Predefined paths evaluation and sample output verification
- Custom lambda sorting and selection
- Optional challenge: Battery capacity constraint filtering
- Graph path discovery (DFS)
- Edge cases: disconnected paths, empty candidates, malformed inputs
"""

import unittest
from typing import Dict, List

from src.utility_based_agent import (
    DEFAULT_GRAPH,
    DEFAULT_PREDEFINED_PATHS,
    DEFAULT_TERRAIN_COSTS,
    EdgeDetail,
    PathEvaluation,
    UtilityBasedAgent,
    UtilitySimulationResult,
    calculate_path_cost,
    calculate_utility,
    find_all_paths,
    resolve_edge_cost,
    run_utility_based_simulation,
)


class TestEdgeCostResolution(unittest.TestCase):
    """Tests for resolving edge costs across different input formats."""

    def test_numeric_cost_resolution(self) -> None:
        """Verify numeric integer and float weights are resolved directly."""
        cost, terrain = resolve_edge_cost(4)
        self.assertEqual(cost, 4.0)
        self.assertEqual(terrain, "Direct")

        cost, terrain = resolve_edge_cost(2.5)
        self.assertEqual(cost, 2.5)
        self.assertEqual(terrain, "Direct")

    def test_terrain_string_lookup(self) -> None:
        """Verify standard terrain types map to designated physical costs."""
        self.assertEqual(resolve_edge_cost("Grass"), (1.0, "Grass"))
        self.assertEqual(resolve_edge_cost("Sand"), (2.0, "Sand"))
        self.assertEqual(resolve_edge_cost("Mud"), (3.0, "Mud"))
        self.assertEqual(resolve_edge_cost("Rock"), (4.0, "Rock"))
        self.assertEqual(resolve_edge_cost("Water"), (5.0, "Water"))

    def test_custom_terrain_mapping(self) -> None:
        """Verify custom terrain dictionary takes precedence when provided."""
        custom_map = {"Lava": 10.0, "Ice": 1.5}
        cost, terrain = resolve_edge_cost("Lava", terrain_costs=custom_map)
        self.assertEqual(cost, 10.0)
        self.assertEqual(terrain, "Lava")

    def test_edge_dict_format(self) -> None:
        """Verify dictionary containing cost and terrain properties is parsed."""
        cost, terrain = resolve_edge_cost({"cost": 3.0, "terrain": "Mud"})
        self.assertEqual(cost, 3.0)
        self.assertEqual(terrain, "Mud")

    def test_unknown_terrain_string_raises_value_error(self) -> None:
        """Verify unrecognized terrain string raises informative ValueError."""
        with self.assertRaises(ValueError) as ctx:
            resolve_edge_cost("UnknownBiome")
        self.assertIn("Unknown terrain type", str(ctx.exception))

    def test_invalid_edge_dict_raises_value_error(self) -> None:
        """Verify malformed edge dictionary raises ValueError."""
        with self.assertRaises(ValueError):
            resolve_edge_cost({"invalid_key": 99})

    def test_unsupported_edge_type_raises_type_error(self) -> None:
        """Verify unsupported type raises TypeError."""
        with self.assertRaises(TypeError):
            resolve_edge_cost([1, 2, 3])  # type: ignore[arg-type]


class TestUtilityFunction(unittest.TestCase):
    """Tests for the agent's utility function U(path) = -1 * total_cost."""

    def test_cost_to_utility_mapping(self) -> None:
        """Verify utility is the exact negative of traversal cost."""
        self.assertEqual(calculate_utility(4.0), -4.0)
        self.assertEqual(calculate_utility(6.0), -6.0)
        self.assertEqual(calculate_utility(0.0), 0.0)

    def test_utility_maximization_reflects_cost_minimization(self) -> None:
        """Verify that smaller cost maps to strictly larger utility."""
        cost_a = 4.0
        cost_b = 6.0
        utility_a = calculate_utility(cost_a)
        utility_b = calculate_utility(cost_b)
        self.assertGreater(utility_a, utility_b)  # -4.0 > -6.0


class TestPathCostCalculation(unittest.TestCase):
    """Tests for summing edge costs along a path in the graph."""

    def test_valid_two_node_path(self) -> None:
        """Verify cost calculation for a single edge transition."""
        cost, details = calculate_path_cost(DEFAULT_GRAPH, ["A", "C"])
        self.assertEqual(cost, 1.0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0].from_node, "A")
        self.assertEqual(details[0].to_node, "C")
        self.assertEqual(details[0].terrain, "Grass")

    def test_valid_multi_edge_path(self) -> None:
        """Verify summation along multi-segment path A-C-D."""
        cost, details = calculate_path_cost(DEFAULT_GRAPH, ["A", "C", "D"])
        self.assertEqual(cost, 4.0)
        self.assertEqual(len(details), 2)
        # Edge 1: A -> C (Grass: 1.0)
        self.assertEqual(details[0].cost, 1.0)
        # Edge 2: C -> D (Mud: 3.0)
        self.assertEqual(details[1].cost, 3.0)

    def test_path_with_fewer_than_two_nodes_raises_value_error(self) -> None:
        """Verify paths with 0 or 1 node are rejected."""
        with self.assertRaises(ValueError):
            calculate_path_cost(DEFAULT_GRAPH, [])
        with self.assertRaises(ValueError):
            calculate_path_cost(DEFAULT_GRAPH, ["A"])

    def test_missing_start_node_raises_key_error(self) -> None:
        """Verify KeyError is raised when a node is absent from the graph."""
        with self.assertRaises(KeyError):
            calculate_path_cost(DEFAULT_GRAPH, ["Z", "A"])

    def test_nonexistent_edge_raises_value_error(self) -> None:
        """Verify ValueError is raised when two adjacent nodes are not connected."""
        with self.assertRaises(ValueError) as ctx:
            calculate_path_cost(DEFAULT_GRAPH, ["A", "D"])  # No direct edge A->D
        self.assertIn("No directed edge exists", str(ctx.exception))


class TestUtilityBasedAgentSimulation(unittest.TestCase):
    """Tests evaluating the 3 predefined paths and sample input/output."""

    def setUp(self) -> None:
        """Initialize agent with default graph."""
        self.agent = UtilityBasedAgent()

    def test_sample_input_two_paths(self) -> None:
        """Verify exact sample input from specification:

        Paths: [A-B-D (cost 6), A-C-D (cost 4)]
        Expected Output: Selected Path: A-C-D, Utility: -4
        """
        sample_paths = [["A", "B", "D"], ["A", "C", "D"]]
        result = self.agent.run_simulation(sample_paths)

        self.assertIsNotNone(result.selected_path)
        self.assertEqual(result.selected_path.path_str, "A-C-D")
        self.assertEqual(result.selected_path.total_cost, 4.0)
        self.assertEqual(result.selected_path.utility, -4.0)

    def test_evaluate_three_predefined_paths(self) -> None:
        """Verify Task Requirement 3: Evaluate 3 predefined paths.

        Path 1: A-B-D (cost 6.0, utility -6.0)
        Path 2: A-C-D (cost 4.0, utility -4.0) [Selected]
        Path 3: A-E-D (cost 8.0, utility -8.0)
        """
        result = self.agent.run_simulation(DEFAULT_PREDEFINED_PATHS)

        self.assertEqual(len(result.evaluated_paths), 3)

        # Check Path 1
        p1 = result.evaluated_paths[0]
        self.assertEqual(p1.path_str, "A-B-D")
        self.assertEqual(p1.total_cost, 6.0)
        self.assertEqual(p1.utility, -6.0)

        # Check Path 2
        p2 = result.evaluated_paths[1]
        self.assertEqual(p2.path_str, "A-C-D")
        self.assertEqual(p2.total_cost, 4.0)
        self.assertEqual(p2.utility, -4.0)

        # Check Path 3
        p3 = result.evaluated_paths[2]
        self.assertEqual(p3.path_str, "A-E-D")
        self.assertEqual(p3.total_cost, 8.0)
        self.assertEqual(p3.utility, -8.0)

        # Output path with highest utility
        self.assertEqual(result.selected_path.path_str, "A-C-D")
        self.assertEqual(result.selected_path.utility, -4.0)

    def test_custom_lambda_sorting(self) -> None:
        """Verify Python custom lambda sorting ranks paths by descending utility."""
        evals = self.agent.evaluate_all_paths(DEFAULT_PREDEFINED_PATHS)
        ranked = self.agent.rank_paths(evals)

        # Expected order: A-C-D (U=-4.0), A-B-D (U=-6.0), A-E-D (U=-8.0)
        self.assertEqual([p.path_str for p in ranked], ["A-C-D", "A-B-D", "A-E-D"])
        self.assertEqual([p.utility for p in ranked], [-4.0, -6.0, -8.0])

    def test_invalid_path_in_candidate_list(self) -> None:
        """Verify disconnected candidate paths are gracefully flagged as invalid."""
        candidates = [["A", "Z", "D"], ["A", "C", "D"]]
        result = self.agent.run_simulation(candidates)

        invalid_path = result.evaluated_paths[0]
        self.assertFalse(invalid_path.is_valid)
        self.assertEqual(invalid_path.utility, float("-inf"))

        # Valid path is still correctly selected
        self.assertEqual(result.selected_path.path_str, "A-C-D")

    def test_empty_candidates_returns_none(self) -> None:
        """Verify empty candidate list returns None without error."""
        result = self.agent.run_simulation([])
        self.assertIsNone(result.selected_path)
        self.assertFalse(result.success)


class TestBatteryConstraintChallenge(unittest.TestCase):
    """Tests for Optional Challenge: Battery Capacity Constraint."""

    def test_battery_budget_sufficient_for_all(self) -> None:
        """When battery=10.0, all 3 paths are feasible and A-C-D is selected."""
        result = run_utility_based_simulation(battery_limit=10.0)
        self.assertEqual(len(result.feasible_paths), 3)
        self.assertEqual(len(result.rejected_paths), 0)
        self.assertEqual(result.selected_path.path_str, "A-C-D")

    def test_battery_budget_rejects_expensive_paths(self) -> None:
        """When battery=5.0:

        A-C-D (Cost 4.0 <= 5.0) -> ACCEPTED
        A-B-D (Cost 6.0 > 5.0)  -> REJECTED
        A-E-D (Cost 8.0 > 5.0)  -> REJECTED
        """
        result = run_utility_based_simulation(battery_limit=5.0)
        self.assertEqual(len(result.feasible_paths), 1)
        self.assertEqual(result.feasible_paths[0].path_str, "A-C-D")
        self.assertEqual(len(result.rejected_paths), 2)
        rejected_strs = {p.path_str for p in result.rejected_paths}
        self.assertEqual(rejected_strs, {"A-B-D", "A-E-D"})
        self.assertEqual(result.selected_path.path_str, "A-C-D")

    def test_battery_budget_insufficient_for_all(self) -> None:
        """When battery=3.0, all paths exceed battery and result.selected_path is None."""
        result = run_utility_based_simulation(battery_limit=3.0)
        self.assertEqual(len(result.feasible_paths), 0)
        self.assertEqual(len(result.rejected_paths), 3)
        self.assertIsNone(result.selected_path)
        self.assertFalse(result.success)

    def test_exact_battery_threshold_is_accepted(self) -> None:
        """Cost exactly equal to battery limit (Cost 4.0 == Battery 4.0) is accepted."""
        result = run_utility_based_simulation(battery_limit=4.0)
        self.assertIn("A-C-D", [p.path_str for p in result.feasible_paths])
        self.assertEqual(result.selected_path.path_str, "A-C-D")


class TestPathDiscovery(unittest.TestCase):
    """Tests for DFS graph path discovery."""

    def test_find_all_paths_between_a_and_d(self) -> None:
        """Verify DFS discovers all simple directed paths from A to D."""
        paths = find_all_paths(DEFAULT_GRAPH, "A", "D")
        # In DEFAULT_GRAPH, valid simple paths from A to D include:
        # A-B-D, A-B-C-D, A-C-D, A-C-B-D, A-E-D
        path_strs = ["-".join(p) for p in paths]
        self.assertIn("A-B-D", path_strs)
        self.assertIn("A-C-D", path_strs)
        self.assertIn("A-E-D", path_strs)
        self.assertIn("A-B-C-D", path_strs)

    def test_find_all_paths_to_unreachable_node(self) -> None:
        """Verify empty list when destination is unreachable."""
        paths = find_all_paths(DEFAULT_GRAPH, "A", "NonExistent")
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
