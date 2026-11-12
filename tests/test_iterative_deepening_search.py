"""Unit tests for Iterative Deepening Search (IDS) (Question 17).

Validates:
- Exact syllabus sample input: Goal at depth 2 ->
  'Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found [Path].'
- Iterative DLS wrapping: depth limits 0, 1, 2, ... until goal found or graph exhausted.
- Per-iteration node expansion tracking & verbose output.
- Shortest-path optimality matching BFS while maintaining O(b * d) DFS stack depth.
- Cyclic graphs (both reachable goals and unreachable goals terminating with Failure).
- Disconnected graphs and single-node graphs terminating with Failure.
- Boundary conditions (start == goal at depth 0, optional max_depth cap, invalid inputs).
- Optional Challenge: total node expansions across all iterations & redundant regenerations.
- Object-oriented `IterativeDeepeningSearch` and functional `ids(...)` interfaces.
- Two-panel visualization generation and Viva Voce reference structure.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from src.depth_limited_search import CUTOFF, FAILURE, SUCCESS
from src.iterative_deepening_search import (
    BINARY_TREE_IDS_GRAPH,
    CYCLIC_IDS_GRAPH,
    DISCONNECTED_IDS_GRAPH,
    SAMPLE_IDS_GRAPH,
    SHORTCUT_IDS_GRAPH,
    IDSIterationRecord,
    IDSResult,
    IterativeDeepeningSearch,
    compare_ids_with_bfs,
    get_ids_viva_reference,
    ids,
    iterative_deepening_search,
    visualize_ids,
)


class TestIterativeDeepeningSearchCore(unittest.TestCase):
    """Core algorithmic tests for Iterative Deepening Search."""

    def test_sample_input_goal_at_depth_two(self) -> None:
        """Verify canonical sample input: Goal at depth 2 -> Limit 0/1 Cutoff, Limit 2 Found."""
        result = iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="4")

        self.assertEqual(result.status, SUCCESS)
        self.assertTrue(result.is_success())
        self.assertFalse(result.is_failure())
        self.assertFalse(result.is_cutoff())
        self.assertEqual(result.path, ["1", "2", "4"])
        self.assertEqual(result.depth, 2)
        self.assertEqual(result.final_limit, 2)
        self.assertEqual(len(result.iterations), 3)

        # Verify sequence string matches practical sample output format
        self.assertEqual(
            result.format_sequence(),
            "Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found ['1', '2', '4'].",
        )

        # Verify per-iteration statuses
        self.assertEqual([it.limit for it in result.iterations], [0, 1, 2])
        self.assertEqual([it.status for it in result.iterations], [CUTOFF, CUTOFF, SUCCESS])

        # Convenience functional interface
        self.assertEqual(ids(SAMPLE_IDS_GRAPH, start="1", goal="4"), ["1", "2", "4"])

    def test_per_iteration_nodes_expanded_and_verbose_output(self) -> None:
        """Verify nodes expanded at each depth iteration and Optional Challenge total count."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = iterative_deepening_search(
                SAMPLE_IDS_GRAPH,
                start="1",
                goal="4",
                verbose=True,
            )

        printed = buf.getvalue()
        self.assertIn("Depth Limit 0", printed)
        self.assertIn("Depth Limit 1", printed)
        self.assertIn("Depth Limit 2", printed)

        it0, it1, it2 = result.iterations
        # Limit 0: root '1' hits cutoff immediately, 0 expansions, 1 visit
        self.assertEqual(it0.nodes_expanded, 0)
        self.assertEqual(it0.expanded_nodes, [])
        self.assertEqual(it0.visited_nodes, ["1"])

        # Limit 1: '1' expanded -> visits '2', '3' (both hit cutoff)
        self.assertEqual(it1.nodes_expanded, 1)
        self.assertEqual(it1.expanded_nodes, ["1"])
        self.assertEqual(it1.visited_nodes, ["1", "2", "3"])

        # Limit 2: '1' -> '2' -> '4' (goal found!)
        self.assertEqual(it2.nodes_expanded, 2)
        self.assertEqual(it2.expanded_nodes, ["1", "2"])
        self.assertEqual(it2.visited_nodes, ["1", "2", "4"])

        # Optional Challenge: total expansions = 0 + 1 + 2 = 3
        self.assertEqual(result.total_nodes_expanded, 3)
        self.assertEqual(result.total_nodes_generated, 1 + 3 + 3)

    def test_shortest_path_optimality_like_bfs(self) -> None:
        """Verify IDS finds the shortest path on shortcut graphs where DFS takes a longer path."""
        cmp = compare_ids_with_bfs(SHORTCUT_IDS_GRAPH, start="A", goal="G")
        self.assertTrue(cmp["ids_found"])
        self.assertTrue(cmp["bfs_found"])
        self.assertTrue(cmp["same_optimal_depth"])
        self.assertEqual(cmp["ids_depth"], 2)
        self.assertEqual(cmp["bfs_depth"], 2)
        self.assertEqual(cmp["ids_path"], ["A", "C", "G"])
        self.assertEqual(cmp["bfs_path"], ["A", "C", "G"])
        # Linear recursion stack memory is depth + 1 = 3 nodes
        self.assertEqual(cmp["ids_peak_stack_depth"], 3)

    def test_start_is_goal_found_at_limit_zero(self) -> None:
        """Boundary condition: start node equals goal node -> found at limit 0."""
        result = iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="1")
        self.assertEqual(result.status, SUCCESS)
        self.assertEqual(result.path, ["1"])
        self.assertEqual(result.depth, 0)
        self.assertEqual(result.final_limit, 0)
        self.assertEqual(result.total_nodes_expanded, 0)
        self.assertEqual(result.total_nodes_generated, 1)
        self.assertEqual(result.peak_stack_depth, 1)
        self.assertEqual(result.format_sequence(), "Limit 0: Found ['1'].")

    def test_disconnected_graph_terminates_with_failure(self) -> None:
        """Verify IDS stops immediately when DLS exhausts the reachable graph (Failure)."""
        result = iterative_deepening_search(DISCONNECTED_IDS_GRAPH, start="A", goal="Y")
        self.assertEqual(result.status, FAILURE)
        self.assertTrue(result.is_failure())
        self.assertIsNone(result.path)
        self.assertIsNone(result.depth)
        self.assertEqual(result.iterations[-1].status, FAILURE)
        self.assertIsNone(ids(DISCONNECTED_IDS_GRAPH, start="A", goal="Y"))

    def test_single_node_graph_failure(self) -> None:
        """Verify single-node graph with unreachable goal terminates cleanly at limit 1."""
        result = iterative_deepening_search({"A": []}, start="A", goal="Z")
        self.assertEqual(result.status, FAILURE)
        self.assertEqual([it.limit for it in result.iterations], [0, 1])
        self.assertEqual([it.status for it in result.iterations], [CUTOFF, FAILURE])


class TestCyclicGraphsAndBoundaries(unittest.TestCase):
    """Tests for cycle safety, bounded depth caps, and input validation."""

    def test_cyclic_graph_finds_shortest_path(self) -> None:
        """Verify IDS finds the shortest path on a cyclic graph without infinite looping."""
        result = iterative_deepening_search(CYCLIC_IDS_GRAPH, start="1", goal="6")
        self.assertEqual(result.status, SUCCESS)
        self.assertEqual(result.path, ["1", "2", "4", "6"])
        self.assertEqual(result.depth, 3)

    def test_cyclic_graph_unreachable_goal_exhausts_safely(self) -> None:
        """Verify IDS terminates with Failure on a cyclic graph when goal does not exist."""
        cyclic = {
            "A": ["A", "B"],
            "B": ["A", "C"],
            "C": ["B"],
        }
        result = iterative_deepening_search(cyclic, start="A", goal="Z", cycle_prevention=True)
        self.assertEqual(result.status, FAILURE)
        self.assertIsNone(result.path)
        self.assertEqual(result.iterations[-1].status, FAILURE)

    def test_max_depth_cap_returns_cutoff(self) -> None:
        """Verify optional max_depth cap halts IDS early with Cutoff status."""
        result = iterative_deepening_search(
            BINARY_TREE_IDS_GRAPH,
            start="A",
            goal="O",  # located at depth 3
            max_depth=2,
        )
        self.assertEqual(result.status, CUTOFF)
        self.assertTrue(result.is_cutoff())
        self.assertIsNone(result.path)
        self.assertEqual(result.final_limit, 2)
        self.assertEqual(len(result.iterations), 3)

    def test_validation_errors(self) -> None:
        """Verify invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="4", max_depth=-1)
        with self.assertRaises(ValueError):
            iterative_deepening_search(SAMPLE_IDS_GRAPH, start="MISSING", goal="4")
        with self.assertRaises(ValueError):
            iterative_deepening_search(SAMPLE_IDS_GRAPH, start="", goal="4")
        with self.assertRaises(ValueError):
            iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="")


class TestOptionalChallengeAndUtilities(unittest.TestCase):
    """Tests for Optional Challenge metrics, OOP interface, formatting, visualization, and Viva."""

    def test_optional_challenge_binary_tree_expansions_and_redundancy(self) -> None:
        """Verify total node expansions and redundant prior generation counts on binary tree."""
        result = iterative_deepening_search(BINARY_TREE_IDS_GRAPH, start="A", goal="O")
        self.assertEqual(result.status, SUCCESS)
        self.assertEqual(result.path, ["A", "C", "G", "O"])
        self.assertEqual(result.depth, 3)

        # Total expansions = sum of expansions across limits 0..3
        expected_total_expansions = sum(it.nodes_expanded for it in result.iterations)
        self.assertEqual(result.total_nodes_expanded, expected_total_expansions)
        self.assertGreater(result.total_nodes_expanded, 0)
        self.assertEqual(
            result.redundant_regenerations,
            sum(it.nodes_generated for it in result.iterations[:-1]),
        )
        self.assertEqual(result.peak_stack_depth, 4)  # depth 3 + root = 4 nodes

    def test_object_oriented_interface_and_string_graph_input(self) -> None:
        """Verify IterativeDeepeningSearch class works with compact string graph notation."""
        solver = IterativeDeepeningSearch("1: 2, 3; 2: 4, 5; 3: 6")
        res = solver.search(start="1", goal="6")
        self.assertEqual(res.status, SUCCESS)
        self.assertEqual(res.path, ["1", "3", "6"])
        self.assertEqual(res.depth, 2)

    def test_formatting_and_serialization(self) -> None:
        """Verify summary, table, and dictionary serialization helpers."""
        result = iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="4")
        summary = result.format_summary()
        self.assertIn("ITERATIVE DEEPENING SEARCH (IDS) EXECUTION SUMMARY", summary)
        self.assertIn("Total Node Expansions", summary)
        self.assertIn("Peak Recursion Stack Depth", summary)

        table = result.format_iteration_table()
        self.assertIn("Limit", table)
        self.assertIn("Expanded Nodes", table)
        self.assertIn("Cutoff", table)
        self.assertIn("Success", table)

        payload = result.to_dict()
        self.assertEqual(payload["status"], SUCCESS)
        self.assertEqual(payload["path"], ["1", "2", "4"])
        self.assertEqual(payload["total_nodes_expanded"], result.total_nodes_expanded)
        self.assertEqual(len(payload["iterations"]), 3)
        self.assertIsInstance(result.iterations[0], IDSIterationRecord)

    def test_visualization_generation(self) -> None:
        """Verify visualize_ids creates a valid PNG chart on disk."""
        result = iterative_deepening_search(SAMPLE_IDS_GRAPH, start="1", goal="4")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img = Path(tmp_dir) / "test_ids.png"
            saved_path = visualize_ids(result, output_path=out_img)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_viva_reference_topics(self) -> None:
        """Verify Viva Voce guide covers all Question 17 syllabus topics."""
        viva = get_ids_viva_reference()
        topics = viva["viva_topics"]
        self.assertIn("why_ids_preferred_in_large_state_spaces", topics)
        self.assertIn("redundant_node_generation", topics)
        self.assertIn("time_vs_space_tradeoff", topics)
        self.assertEqual(viva["complexity_comparison"]["IDS"]["time"], "O(b^d)")
        self.assertEqual(viva["complexity_comparison"]["IDS"]["space"], "O(b * d) [Linear]")


if __name__ == "__main__":
    unittest.main()
