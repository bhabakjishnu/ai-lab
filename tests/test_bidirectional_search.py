"""Unit tests for Bidirectional Search (BDS) Simulation (Question 19).

Validates:
- Exact syllabus sample input & output: Start 'A', Goal 'G' ->
  'Intersection at D. Path: A-B-D-F-G'
- Dual data structures: two FIFO queues and two visited sets tracking parents
- Alternating expansion between forward and backward searches
- Frontier intersection detection and path splicing at meeting node
- Edge cases: Start is Goal, direct neighbors, linear chains, disconnected components
- Directed graphs with predecessor inversion vs undirected symmetric graphs
- Time & Space complexity halving: O(b^(d/2)) vs O(b^d) empirical validation
- Optional Challenge: 2D Grid Maze Bidirectional Search (solvable, blocked, boundary checks)
- Input parsing & graph normalization (semicolon, JSON, error handling)
- Viva Voce reference guide and visualization generation
"""

import io
from pathlib import Path
import tempfile
import unittest

from src.bidirectional_search import (
    DEFAULT_BDS_MAZE,
    DIRECTED_REVERSIBLE_BDS_GRAPH,
    DISCONNECTED_BDS_GRAPH,
    EXPONENTIAL_TREE_BDS_GRAPH,
    LINEAR_CHAIN_BDS_GRAPH,
    SAMPLE_BDS_GRAPH,
    BidirectionalSearch,
    BidirectionalSearchResult,
    GridMazeBidirectionalResult,
    bidirectional_search,
    compare_unidirectional_vs_bidirectional,
    get_bidirectional_search_viva_reference,
    invert_graph,
    make_undirected,
    normalize_graph,
    parse_graph_input,
    plot_bidirectional_search,
    plot_grid_maze_bidirectional,
    solve_grid_maze_bidirectional,
)


class TestBidirectionalSearchCore(unittest.TestCase):
    """Core algorithmic tests for Bidirectional Search on unweighted graphs."""

    def test_syllabus_sample_input_and_output(self) -> None:
        """Verify exact Question 19 sample input & output specification:
        Sample Input: Start: A, Goal: G
        Sample Output: Intersection at D. Path: A-B-D-F-G
        """
        result = bidirectional_search(
            graph=SAMPLE_BDS_GRAPH,
            start="A",
            goal="G",
        )

        self.assertEqual(result.status, "Success")
        self.assertEqual(result.start_node, "A")
        self.assertEqual(result.goal_node, "G")
        self.assertEqual(result.intersection_node, "D")
        self.assertEqual(result.path, ["A", "B", "D", "F", "G"])
        self.assertEqual(result.path_length, 4)
        self.assertEqual(result.forward_path, ["A", "B", "D"])
        self.assertEqual(result.backward_path, ["D", "F", "G"])

        # Exact target string match
        self.assertEqual(
            result.format_sample_output(),
            "Intersection at D. Path: A-B-D-F-G",
        )

    def test_alternating_expansion_trace(self) -> None:
        """Verify that expansion turns alternate strictly between FORWARD and BACKWARD."""
        result = bidirectional_search(
            graph=SAMPLE_BDS_GRAPH,
            start="A",
            goal="G",
            alternation_strategy="turn",
        )

        self.assertTrue(len(result.steps) >= 4)
        directions = [step.direction for step in result.steps]
        # Should alternate: FORWARD, BACKWARD, FORWARD, BACKWARD
        expected_prefix = ["FORWARD", "BACKWARD", "FORWARD", "BACKWARD"]
        self.assertEqual(directions[:4], expected_prefix)

        # Final step should have detected the intersection
        last_step = result.steps[-1]
        self.assertTrue(last_step.intersection_found)
        self.assertEqual(last_step.intersection_node, "D")

    def test_base_case_start_is_goal(self) -> None:
        """Verify base case when start node is identical to goal node."""
        result = bidirectional_search(
            graph=SAMPLE_BDS_GRAPH,
            start="A",
            goal="A",
        )

        self.assertEqual(result.status, "Success")
        self.assertEqual(result.intersection_node, "A")
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.path_length, 0)
        self.assertEqual(result.total_nodes_expanded, 0)
        self.assertEqual(result.format_sample_output(), "Intersection at A. Path: A")

    def test_direct_neighbor_single_hop(self) -> None:
        """Verify immediate 1-hop path when start and goal are adjacent."""
        result = bidirectional_search(
            graph=SAMPLE_BDS_GRAPH,
            start="A",
            goal="B",
        )

        self.assertEqual(result.status, "Success")
        self.assertIn(result.intersection_node, ["A", "B"])
        self.assertEqual(result.path, ["A", "B"])
        self.assertEqual(result.path_length, 1)

    def test_linear_chain_odd_and_even_distances(self) -> None:
        """Verify path reconstruction on linear chain A-B-C-D-E-F."""
        # 5 hops (even count of nodes, odd edge count)
        result = bidirectional_search(
            graph=LINEAR_CHAIN_BDS_GRAPH,
            start="A",
            goal="F",
        )
        self.assertEqual(result.status, "Success")
        self.assertEqual(result.path, ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(result.path_length, 5)
        self.assertIn(result.intersection_node, ["C", "D"])

        # 4 hops (odd count of nodes, even edge count)
        res_even = bidirectional_search(
            graph=LINEAR_CHAIN_BDS_GRAPH,
            start="A",
            goal="E",
        )
        self.assertEqual(res_even.status, "Success")
        self.assertEqual(res_even.path, ["A", "B", "C", "D", "E"])
        self.assertEqual(res_even.path_length, 4)
        self.assertEqual(res_even.intersection_node, "C")

    def test_disconnected_graph_failure(self) -> None:
        """Verify graceful Failure termination when goal is in an unreachable component."""
        result = bidirectional_search(
            graph=DISCONNECTED_BDS_GRAPH,
            start="A",
            goal="Z",
        )

        self.assertEqual(result.status, "Failure")
        self.assertIsNone(result.intersection_node)
        self.assertIsNone(result.path)
        self.assertEqual(result.path_length, 0)
        self.assertEqual(
            result.format_sample_output(),
            "Failure: No path found between A and Z.",
        )

    def test_directed_graph_with_predecessor_inversion(self) -> None:
        """Verify directed graph search with backward predecessor inversion."""
        result = bidirectional_search(
            graph=DIRECTED_REVERSIBLE_BDS_GRAPH,
            start="S",
            goal="G",
            is_directed=True,
        )

        self.assertEqual(result.status, "Success")
        self.assertIn(result.path, [["S", "A", "C", "E", "G"], ["S", "B", "D", "E", "G"]])
        self.assertEqual(result.path_length, 4)
        self.assertIn(result.intersection_node, ["C", "D", "E"])

    def test_balanced_alternation_strategy(self) -> None:
        """Verify search with balanced frontier alternation strategy."""
        result = bidirectional_search(
            graph=SAMPLE_BDS_GRAPH,
            start="A",
            goal="G",
            alternation_strategy="balanced",
        )

        self.assertEqual(result.status, "Success")
        self.assertEqual(result.intersection_node, "D")
        self.assertEqual(result.path, ["A", "B", "D", "F", "G"])

    def test_invalid_nodes_raise_value_error(self) -> None:
        """Verify ValueError is raised when start or goal does not exist in graph."""
        solver = BidirectionalSearch(SAMPLE_BDS_GRAPH)
        with self.assertRaises(ValueError) as ctx:
            solver.search(start="NON_EXISTENT", goal="G")
        self.assertIn("does not exist in graph nodes", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            solver.search(start="A", goal="UNKNOWN")
        self.assertIn("does not exist in graph nodes", str(ctx2.exception))


class TestBidirectionalComplexityAndComparisons(unittest.TestCase):
    """Empirical tests validating search space halving and theoretical complexity."""

    def test_unidirectional_vs_bidirectional_comparison(self) -> None:
        """Verify that bidirectional search reduces node expansions by ~50% on branching tree."""
        comp = compare_unidirectional_vs_bidirectional(
            graph=EXPONENTIAL_TREE_BDS_GRAPH,
            start="root",
            goal="goal",
        )

        self.assertTrue(comp["optimal_path_match"])
        self.assertEqual(comp["unidirectional_path_length"], 6)
        self.assertEqual(comp["bidirectional_path_length"], 6)
        self.assertLess(comp["bidirectional_expansions"], comp["unidirectional_expansions"])
        self.assertGreaterEqual(comp["expansion_reduction_percent"], 40.0)

    def test_viva_reference_guide_structure(self) -> None:
        """Verify Viva Voce study guide covers all 3 required syllabus topics."""
        viva = get_bidirectional_search_viva_reference()

        self.assertIn("complexity_advantage", viva)
        self.assertIn("backward_search_difficulties", viva)
        self.assertIn("predecessor_vs_successor", viva)

        # Check content depth
        self.assertIn("O(b^(d/2))", viva["complexity_advantage"]["question"])
        self.assertIn("50,000", viva["complexity_advantage"]["answer"])
        self.assertIn("directed graphs", viva["backward_search_difficulties"]["question"])
        self.assertIn("Successors(s)", viva["predecessor_vs_successor"]["answer"])


class TestGridMazeBidirectionalSearch(unittest.TestCase):
    """Tests for the Optional Challenge: 2D Grid Maze Bidirectional Search."""

    def test_default_grid_maze_solution(self) -> None:
        """Verify bidirectional search navigates the 7x7 grid maze from (0,0) to (6,6)."""
        res = solve_grid_maze_bidirectional(
            maze=DEFAULT_BDS_MAZE,
            start=(0, 0),
            goal=(6, 6),
        )

        self.assertEqual(res.status, "Success")
        self.assertIsNotNone(res.intersection_coord)
        self.assertEqual(res.path_length, 12)
        self.assertEqual(len(res.path), 13)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (6, 6))

        # Check ASCII rendering output
        ascii_maze = res.render_ascii_maze()
        self.assertIn("[S]=Start", ascii_maze)
        self.assertIn("[G]=Goal", ascii_maze)
        self.assertIn("[X]=Intersection", ascii_maze)

    def test_grid_maze_start_is_goal(self) -> None:
        """Verify grid maze base case when start equals goal."""
        res = solve_grid_maze_bidirectional(
            maze=DEFAULT_BDS_MAZE,
            start=(2, 2),
            goal=(2, 2),
        )
        self.assertEqual(res.status, "Success")
        self.assertEqual(res.intersection_coord, (2, 2))
        self.assertEqual(res.path, [(2, 2)])
        self.assertEqual(res.path_length, 0)

    def test_grid_maze_blocked_unsolvable(self) -> None:
        """Verify Failure when goal is partitioned behind an impenetrable wall."""
        blocked_maze = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        res = solve_grid_maze_bidirectional(blocked_maze, start=(0, 0), goal=(0, 3))
        self.assertEqual(res.status, "Failure")
        self.assertIsNone(res.path)
        self.assertEqual(res.path_length, 0)

    def test_grid_maze_boundary_and_wall_validation(self) -> None:
        """Verify ValueError is raised for invalid coordinates or wall collisions."""
        # Out of bounds
        with self.assertRaises(ValueError):
            solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, start=(-1, 0), goal=(6, 6))
        with self.assertRaises(ValueError):
            solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, start=(0, 0), goal=(10, 10))

        # Wall obstacle collision
        with self.assertRaises(ValueError):
            solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, start=(1, 1), goal=(6, 6))
        with self.assertRaises(ValueError):
            solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, start=(0, 0), goal=(1, 1))


class TestGraphParsingAndVisualizations(unittest.TestCase):
    """Tests for graph parsing helpers, dictionary export, and plotting."""

    def test_parse_graph_input_formats(self) -> None:
        """Verify semicolon, JSON, and dict graph parsing."""
        # 1. Semicolon format
        raw = "A: B, C; B: D; C:; D:"
        parsed = parse_graph_input(raw)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])
        self.assertEqual(parsed["C"], [])

        # 2. JSON format
        raw_json = '{"X": ["Y", "Z"], "Y": ["X"], "Z": []}'
        parsed_json = parse_graph_input(raw_json)
        self.assertEqual(parsed_json["X"], ["Y", "Z"])

        # 3. Invalid format
        with self.assertRaises(ValueError):
            parse_graph_input("")
        with self.assertRaises(ValueError):
            parse_graph_input("InvalidEntryWithoutColon")

    def test_invert_and_undirected_helpers(self) -> None:
        """Verify graph inversion and undirected conversion."""
        directed = {"A": ["B", "C"], "B": ["C"], "C": []}
        inverted = invert_graph(directed)
        self.assertEqual(sorted(inverted["C"]), ["A", "B"])
        self.assertEqual(inverted["B"], ["A"])

        undirected = make_undirected(directed)
        self.assertIn("A", undirected["B"])
        self.assertIn("A", undirected["C"])
        self.assertIn("B", undirected["C"])

    def test_to_dict_serialization(self) -> None:
        """Verify result to_dict produces valid dictionary."""
        result = bidirectional_search()
        d = result.to_dict()
        self.assertEqual(d["status"], "Success")
        self.assertEqual(d["intersection_node"], "D")
        self.assertEqual(d["path"], ["A", "B", "D", "F", "G"])
        self.assertIn("steps", d)

    def test_plot_generation_headless(self) -> None:
        """Verify Matplotlib graph and maze plots generate and save to temp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_png = Path(tmpdir) / "test_bds_graph.png"
            maze_png = Path(tmpdir) / "test_bds_maze.png"

            # 1. Graph plot
            res_graph = bidirectional_search()
            fig_graph = plot_bidirectional_search(res_graph, save_path=graph_png)
            self.assertIsNotNone(fig_graph)
            self.assertTrue(graph_png.exists())
            self.assertGreater(graph_png.stat().st_size, 0)

            # 2. Maze plot
            res_maze = solve_grid_maze_bidirectional(DEFAULT_BDS_MAZE, (0, 0), (6, 6))
            fig_maze = plot_grid_maze_bidirectional(res_maze, save_path=maze_png)
            self.assertIsNotNone(fig_maze)
            self.assertTrue(maze_png.exists())
            self.assertGreater(maze_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
