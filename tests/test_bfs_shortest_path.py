"""Comprehensive Unit Test Suite for BFS Shortest Path Constructor (Question 13).

Validates:
    1. Sample Input/Output: Start A, Goal F -> Shortest Path: ['A', 'C', 'F'] (Cost: 2 edges).
    2. Identity Search: Start == Goal (Cost: 0 edges, path ['A']).
    3. Unreachable Goal & Disconnected Graphs: Clean None return and goal_found=False.
    4. Cyclic Graphs & Cycle Avoidance: Infinite loop prevention and shortest hop guarantee.
    5. Early Goal Testing vs Late Goal Testing:
       - Mathematical equivalence of computed shortest path.
       - Verification that early goal testing expands fewer or equal vertices.
    6. Tracking Mechanisms:
       - Equivalence of Parent Dictionary Backtracking (O(V) space) and Queue Path Tracking.
    7. Optional Challenge (All Possible Shortest Paths):
       - Multi-path diamond graph (A->B->D and A->C->D).
       - Symmetric grid graphs with multiple equidistant branches.
    8. Exception Handling:
       - Nonexistent start or goal nodes raise ValueError.
       - Corrupted parent mappings raise ValueError.
    9. Object-Oriented Interface:
       - BFSShortestPathConstructor class methods.
    10. Graphical Visualization & Step Record Formatting:
       - Matplotlib rendering and step table generation.
    11. Viva Voce Reference Completeness:
       - Coverage of all required theoretical topics.
"""

from pathlib import Path
import tempfile
import unittest

from src.bfs_shortest_path import (
    CYCLIC_BENCHMARK_GRAPH,
    DEFAULT_SHORTEST_PATH_GRAPH,
    DISCONNECTED_SHORTEST_PATH_GRAPH,
    MULTI_PATH_DIAMOND_GRAPH,
    MULTI_PATH_GRID_GRAPH,
    BFSShortestPathConstructor,
    PathStepRecord,
    ShortestPathResult,
    bfs_shortest_path,
    compare_goal_testing_strategies,
    compare_tracking_mechanisms,
    find_all_shortest_paths,
    get_viva_reference,
    reconstruct_path_from_parents,
    visualize_shortest_path,
)


class TestBFSShortestPathConstructor(unittest.TestCase):
    """Unit test suite for BFS Shortest Path Constructor algorithm."""

    def test_sample_input_default_graph(self) -> None:
        """Verify Sample Input: Start: A, Goal: F returns ['A', 'C', 'F'] with cost 2."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "C", "F"])
        self.assertEqual(result.cost_edges, 2)
        self.assertEqual(result.cost_nodes, 3)
        self.assertEqual(result.format_path(), "A -> C -> F")

    def test_sample_input_cyclic_graph(self) -> None:
        """Verify shortest path on cyclic graph is ['A', 'C', 'F'] despite cycles."""
        result = bfs_shortest_path(CYCLIC_BENCHMARK_GRAPH, start="A", goal="F")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "C", "F"])
        self.assertEqual(result.cost_edges, 2)

    def test_start_equals_goal_identity(self) -> None:
        """Verify when start node equals goal node, returns 0 edges and [start]."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="A")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.cost_edges, 0)
        self.assertEqual(result.cost_nodes, 1)
        self.assertEqual(result.nodes_expanded, 0)

    def test_unreachable_goal_disconnected_graph(self) -> None:
        """Verify unreachable destination in disconnected graph returns None."""
        result = bfs_shortest_path(DISCONNECTED_SHORTEST_PATH_GRAPH, start="A", goal="F")
        self.assertFalse(result.goal_found)
        self.assertIsNone(result.path)
        self.assertEqual(result.cost_edges, -1)
        self.assertEqual(result.cost_nodes, 0)
        self.assertEqual(result.format_path(), "No path exists (Goal is unreachable)")

    def test_direct_neighbor_single_hop(self) -> None:
        """Verify 1-hop path to immediate neighbor."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="B")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "B"])
        self.assertEqual(result.cost_edges, 1)

    def test_longer_branch_bypass(self) -> None:
        """Verify BFS chooses shorter branch A->C->F (2 edges) over A->B->E->F (3 edges)."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="E")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "B", "E"])
        self.assertEqual(result.cost_edges, 2)

    def test_early_vs_late_goal_testing_equivalence_and_efficiency(self) -> None:
        """Verify early and late goal testing yield identical paths, but early expands fewer nodes."""
        early_res = bfs_shortest_path(
            DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F", early_goal_test=True
        )
        late_res = bfs_shortest_path(
            DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F", early_goal_test=False
        )

        # Both find the exact same shortest path
        self.assertEqual(early_res.path, late_res.path)
        self.assertEqual(early_res.cost_edges, late_res.cost_edges)

        # Early goal testing terminates earlier without expanding the entire bottom layer
        self.assertLess(early_res.nodes_expanded, late_res.nodes_expanded)
        self.assertLessEqual(len(early_res.steps), len(late_res.steps))

    def test_queue_paths_tracking_mode(self) -> None:
        """Verify queue_paths tracking mode produces identical shortest path."""
        res_queue = bfs_shortest_path(
            DEFAULT_SHORTEST_PATH_GRAPH,
            start="A",
            goal="F",
            tracking_mode="queue_paths",
        )
        res_parent = bfs_shortest_path(
            DEFAULT_SHORTEST_PATH_GRAPH,
            start="A",
            goal="F",
            tracking_mode="parent_dict",
        )
        self.assertTrue(res_queue.goal_found)
        self.assertEqual(res_queue.path, res_parent.path)
        self.assertEqual(res_queue.cost_edges, 2)
        self.assertEqual(res_queue.tracking_mode, "queue_paths")

    def test_queue_paths_tracking_late_goal(self) -> None:
        """Verify queue_paths tracking mode functions with late goal testing."""
        res = bfs_shortest_path(
            DEFAULT_SHORTEST_PATH_GRAPH,
            start="A",
            goal="F",
            early_goal_test=False,
            tracking_mode="queue_paths",
        )
        self.assertTrue(res.goal_found)
        self.assertEqual(res.path, ["A", "C", "F"])

    def test_optional_challenge_all_shortest_paths_diamond(self) -> None:
        """Verify Optional Challenge finds all tied shortest paths in diamond graph."""
        all_paths = find_all_shortest_paths(MULTI_PATH_DIAMOND_GRAPH, start="A", goal="D")
        self.assertEqual(len(all_paths), 2)
        self.assertIn(["A", "B", "D"], all_paths)
        self.assertIn(["A", "C", "D"], all_paths)

    def test_optional_challenge_all_shortest_paths_grid(self) -> None:
        """Verify Optional Challenge on multi-path grid graph finds all minimal paths."""
        all_paths = find_all_shortest_paths(MULTI_PATH_GRID_GRAPH, start="A", goal="G")
        self.assertTrue(len(all_paths) >= 2)
        for p in all_paths:
            self.assertEqual(len(p) - 1, 3)  # All paths must have exactly 3 edges
            self.assertEqual(p[0], "A")
            self.assertEqual(p[-1], "G")

    def test_optional_challenge_unreachable(self) -> None:
        """Verify Optional Challenge returns empty list when goal is unreachable."""
        all_paths = find_all_shortest_paths(DISCONNECTED_SHORTEST_PATH_GRAPH, start="A", goal="F")
        self.assertEqual(all_paths, [])

    def test_optional_challenge_identity(self) -> None:
        """Verify Optional Challenge returns [[start]] when start == goal."""
        all_paths = find_all_shortest_paths(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="A")
        self.assertEqual(all_paths, [["A"]])

    def test_compare_goal_testing_strategies_helper(self) -> None:
        """Verify compare_goal_testing_strategies reports accurate savings."""
        comp = compare_goal_testing_strategies(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F")
        self.assertEqual(comp["shortest_path"], ["A", "C", "F"])
        self.assertEqual(comp["path_cost_edges"], 2)
        self.assertGreater(comp["expansions_saved"], 0)

    def test_compare_tracking_mechanisms_helper(self) -> None:
        """Verify compare_tracking_mechanisms confirms equivalence."""
        comp = compare_tracking_mechanisms(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F")
        self.assertTrue(comp["identical_paths"])
        self.assertEqual(comp["path"], ["A", "C", "F"])

    def test_reconstruct_path_from_parents_success(self) -> None:
        """Verify reconstruct_path_from_parents properly traces predecessors."""
        parents = {"A": None, "B": "A", "C": "B", "D": "C"}
        path = reconstruct_path_from_parents(parents, start="A", goal="D")
        self.assertEqual(path, ["A", "B", "C", "D"])

    def test_reconstruct_path_from_parents_missing_goal_raises(self) -> None:
        """Verify reconstruct_path_from_parents raises ValueError if goal not present."""
        parents = {"A": None, "B": "A"}
        with self.assertRaises(ValueError):
            reconstruct_path_from_parents(parents, start="A", goal="Z")

    def test_invalid_start_node_raises_value_error(self) -> None:
        """Verify search raises ValueError if start node is not in graph."""
        with self.assertRaises(ValueError) as ctx:
            bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="Z", goal="F")
        self.assertIn("Start node 'Z' does not exist", str(ctx.exception))

    def test_invalid_goal_node_raises_value_error(self) -> None:
        """Verify search raises ValueError if goal node is not in graph."""
        with self.assertRaises(ValueError) as ctx:
            bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="Z")
        self.assertIn("Goal node 'Z' does not exist", str(ctx.exception))

    def test_invalid_nodes_in_find_all_shortest_paths(self) -> None:
        """Verify find_all_shortest_paths validates node existence."""
        with self.assertRaises(ValueError):
            find_all_shortest_paths(DEFAULT_SHORTEST_PATH_GRAPH, start="Z", goal="F")
        with self.assertRaises(ValueError):
            find_all_shortest_paths(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="Z")

    def test_oop_constructor_class(self) -> None:
        """Verify BFSShortestPathConstructor class methods and wrapper interface."""
        constructor = BFSShortestPathConstructor(DEFAULT_SHORTEST_PATH_GRAPH)
        res = constructor.find_shortest_path("A", "F")
        self.assertEqual(res.path, ["A", "C", "F"])
        self.assertEqual(res.cost_edges, 2)

        all_paths = constructor.find_all_shortest_paths("A", "F")
        self.assertEqual(all_paths, [["A", "C", "F"]])

        comp = constructor.compare_goal_tests("A", "F")
        self.assertEqual(comp["shortest_path"], ["A", "C", "F"])

    def test_formatting_helpers(self) -> None:
        """Verify format_summary, format_step_table, and format_parent_map."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F")
        summary = result.format_summary()
        self.assertIn("BFS SHORTEST PATH CONSTRUCTOR EXECUTION SUMMARY", summary)
        self.assertIn("Start Node           : A", summary)
        self.assertIn("Goal Node            : F", summary)
        self.assertIn("Shortest Path (List) : ['A', 'C', 'F']", summary)

        table = result.format_step_table()
        self.assertIn("Step  | Action", table)
        self.assertIn("EARLY_GOAL_FOUND", table)

        parent_map = result.format_parent_map()
        self.assertIn("A -> None (Start)", parent_map)
        self.assertIn("C -> A", parent_map)
        self.assertIn("F -> C", parent_map)

    def test_visualization_generates_file(self) -> None:
        """Verify visualize_shortest_path generates and saves figure without errors."""
        result = bfs_shortest_path(DEFAULT_SHORTEST_PATH_GRAPH, start="A", goal="F")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_path.png"
            fig = visualize_shortest_path(result, save_path=out_file, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_viva_reference_structure(self) -> None:
        """Verify get_viva_reference covers all mandatory university examination topics."""
        viva = get_viva_reference()
        self.assertEqual(viva["question_number"], 13)
        self.assertIn("topics", viva)
        topics = viva["topics"]
        self.assertIn("path_vs_traversal", topics)
        self.assertIn("early_vs_late_goal_testing", topics)
        self.assertIn("why_bfs_finds_shortest_path", topics)
        self.assertIn("path_reconstruction_techniques", topics)
        self.assertIn("complexity_analysis", topics)
        self.assertIn("handling_unreachable_goals", topics)

        # Check in-depth answers are thorough
        for topic in topics.values():
            self.assertTrue(len(topic["summary"]) > 10)
            self.assertTrue(len(topic["in_depth_answer"]) > 50)


if __name__ == "__main__":
    unittest.main()
