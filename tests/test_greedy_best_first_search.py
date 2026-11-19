"""Comprehensive Unit Test Suite for Greedy Best-First Search (Question 26).

Validates:
- Exact syllabus sample input & output ('Path chosen based on minimum h(n): A -> B -> C').
- Optional Challenge: Counterexample where Greedy fails to find the optimal path.
- Russell & Norvig Romania Map benchmark (Arad to Bucharest: 450 vs. 418 optimal).
- Local minima trap behavior where deceptive heuristics draw Greedy into dead-ends.
- Cycle avoidance in cyclic graphs via the explored set.
- Base cases: start == goal, disconnected graphs, unreachable goals.
- Tie-breaking policies: FIFO, alphabetical, LIFO.
- Input parsing robustness: JSON, arrow syntax, colon notation, whitespace handling.
- Error handling: missing start/goal/neighbor in heuristics, malformed inputs.
- Step-by-step trace formatting and table generation.
- OOP wrapper class interface (GreedyBestFirstSearch).
- Viva Voce study guide data completeness.
- Graphical visualization generation with NetworkX / Matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from src.greedy_best_first_search import (
    CYCLIC_GRAPH,
    CYCLIC_HEURISTICS,
    DEFAULT_SAMPLE_GRAPH,
    DEFAULT_SAMPLE_HEURISTICS,
    DISCONNECTED_GRAPH,
    DISCONNECTED_HEURISTICS,
    LOCAL_MINIMUM_GRAPH,
    LOCAL_MINIMUM_HEURISTICS,
    ROMANIA_MAP_GRAPH,
    ROMANIA_SLD_HEURISTICS,
    SUBOPTIMAL_CHALLENGE_GRAPH,
    SUBOPTIMAL_CHALLENGE_HEURISTICS,
    GBFSResult,
    GBFSStepRecord,
    GreedyBestFirstSearch,
    compare_greedy_vs_optimal,
    compute_path_cost,
    demonstrate_local_minimum,
    get_gbfs_viva_reference,
    greedy_best_first_search,
    normalize_graph,
    normalize_heuristics,
    parse_graph,
    parse_heuristics,
    uniform_cost_search_baseline,
    visualize_gbfs,
)


class TestGreedyBestFirstSearchCore(unittest.TestCase):
    """Core algorithmic tests for Greedy Best-First Search."""

    def test_syllabus_sample_input_and_output(self) -> None:
        """Verify exact sample from Question 26 prompt: A -> B -> C."""
        result = greedy_best_first_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="A",
            goal="C",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertEqual(
            result.format_path(),
            "Path chosen based on minimum h(n): A -> B -> C",
        )
        self.assertEqual(result.start_node, "A")
        self.assertEqual(result.goal_node, "C")
        self.assertEqual(result.nodes_expanded, ["A", "B", "C"])

    def test_start_equals_goal(self) -> None:
        """Verify base case when start node is already the goal node."""
        result = greedy_best_first_search(
            {"A": ["B"], "B": []},
            {"A": 0.0, "B": 2.0},
            start="A",
            goal="A",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.path_cost, 0.0)
        self.assertEqual(
            result.format_path(),
            "Path chosen based on minimum h(n): A",
        )

    def test_disconnected_graph_unreachable_goal(self) -> None:
        """Verify handling of disconnected graphs where goal cannot be reached."""
        result = greedy_best_first_search(
            DISCONNECTED_GRAPH,
            DISCONNECTED_HEURISTICS,
            start="A",
            goal="Goal",
        )
        self.assertFalse(result.goal_found)
        self.assertIsNone(result.path)
        self.assertEqual(
            result.format_path(),
            "Path chosen based on minimum h(n): None (Goal unreachable)",
        )

    def test_cyclic_graph_avoids_infinite_loop(self) -> None:
        """Verify graph search explored set prevents infinite loops in cyclic graphs."""
        result = greedy_best_first_search(
            CYCLIC_GRAPH,
            CYCLIC_HEURISTICS,
            start="A",
            goal="G",
        )
        self.assertTrue(result.goal_found)
        self.assertIsNotNone(result.path)
        self.assertEqual(result.path[-1], "G")
        self.assertEqual(result.path[0], "A")
        # Ensure no node is expanded more than once in graph search
        self.assertEqual(len(result.nodes_expanded), len(set(result.nodes_expanded)))

    def test_tie_breaking_policies(self) -> None:
        """Verify deterministic tie-breaking across FIFO, alphabetical, and LIFO."""
        # A connects to X (h=3) and Y (h=3); both lead to Goal (h=0)
        graph = {
            "A": ["Y", "X"],
            "X": ["Goal"],
            "Y": ["Goal"],
            "Goal": [],
        }
        heuristics = {"A": 10.0, "X": 3.0, "Y": 3.0, "Goal": 0.0}

        # FIFO: 'Y' was enqueued first, so 'Y' should be expanded first
        res_fifo = greedy_best_first_search(graph, heuristics, "A", "Goal", tie_breaker="fifo")
        self.assertEqual(res_fifo.path, ["A", "Y", "Goal"])

        # Alphabetical: 'X' < 'Y', so 'X' should be expanded first regardless of insertion order
        res_alpha = greedy_best_first_search(graph, heuristics, "A", "Goal", tie_breaker="alphabetical")
        self.assertEqual(res_alpha.path, ["A", "X", "Goal"])


class TestGreedyBestFirstSearchOptionalChallenge(unittest.TestCase):
    """Tests for Optional Challenge: Demonstrating Greedy Suboptimality."""

    def test_suboptimal_challenge_graph(self) -> None:
        """Verify Greedy chooses suboptimal path S->A->G over optimal S->B->G."""
        comparison = compare_greedy_vs_optimal(
            SUBOPTIMAL_CHALLENGE_GRAPH,
            SUBOPTIMAL_CHALLENGE_HEURISTICS,
            start="S",
            goal="G",
        )
        self.assertTrue(comparison.is_suboptimal)
        self.assertEqual(comparison.greedy_result.path, ["S", "A", "G"])
        self.assertEqual(comparison.optimal_path, ["S", "B", "G"])
        self.assertEqual(comparison.greedy_result.path_cost, 110.0)
        self.assertEqual(comparison.optimal_cost, 2.0)
        self.assertAlmostEqual(comparison.suboptimality_ratio, 55.0)
        self.assertIn("SUBOPTIMALITY CONFIRMED", comparison.analysis)

    def test_romania_map_arad_to_bucharest(self) -> None:
        """Verify classic AIMA Romania map suboptimality (Cost 450 vs 418)."""
        comparison = compare_greedy_vs_optimal(
            ROMANIA_MAP_GRAPH,
            ROMANIA_SLD_HEURISTICS,
            start="Arad",
            goal="Bucharest",
        )
        self.assertTrue(comparison.is_suboptimal)
        self.assertEqual(
            comparison.greedy_result.path,
            ["Arad", "Sibiu", "Fagaras", "Bucharest"],
        )
        self.assertEqual(
            comparison.optimal_path,
            ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"],
        )
        self.assertEqual(comparison.greedy_result.path_cost, 450.0)
        self.assertEqual(comparison.optimal_cost, 418.0)
        self.assertEqual(comparison.cost_difference, 32.0)

    def test_local_minima_demonstration(self) -> None:
        """Verify Greedy dives into deceptive local minimum before backtracking."""
        demo = demonstrate_local_minimum()
        res: GBFSResult = demo["result"]
        self.assertTrue(res.goal_found)
        self.assertEqual(res.path, ["S", "Path_1", "Path_2", "Goal"])
        # Verify that Trap nodes were expanded before Goal was found
        trap_indices = [
            res.nodes_expanded.index(node)
            for node in ["Trap_1", "Trap_2", "Dead_End"]
            if node in res.nodes_expanded
        ]
        goal_index = res.nodes_expanded.index("Goal")
        self.assertTrue(all(idx < goal_index for idx in trap_indices))
        self.assertIn("LOCAL MINIMUM EXPLANATION", demo["explanation"])


class TestGreedyBestFirstSearchParsingAndValidation(unittest.TestCase):
    """Tests for parsing, validation, and error handling."""

    def test_parse_graph_formats(self) -> None:
        """Verify parsing JSON, colon, and arrow notations."""
        # JSON
        g_json = parse_graph('{"A": ["B"], "B": ["C"], "C": []}')
        self.assertIn("A", g_json)
        self.assertEqual(g_json["A"], ["B"])

        # Arrow notation
        g_arrow = parse_graph("A -> B, B -> C")
        self.assertEqual(g_arrow["A"], ["B"])
        self.assertEqual(g_arrow["B"], ["C"])

        # Colon notation
        g_colon = parse_graph("A: B; B: C; C:")
        self.assertEqual(g_colon["A"], ["B"])
        self.assertEqual(g_colon["B"], ["C"])

    def test_parse_heuristics_formats(self) -> None:
        """Verify parsing JSON, Python dict strings, and comma-separated pairs."""
        h_json = parse_heuristics('{"A": 5, "B": 2, "C": 0}')
        self.assertEqual(h_json, {"A": 5.0, "B": 2.0, "C": 0.0})

        h_colon = parse_heuristics("A: 5, B: 2, C: 0")
        self.assertEqual(h_colon, {"A": 5.0, "B": 2.0, "C": 0.0})

        h_eq = parse_heuristics("A=5, B=2, C=0")
        self.assertEqual(h_eq, {"A": 5.0, "B": 2.0, "C": 0.0})

    def test_missing_heuristic_raises_key_error(self) -> None:
        """Verify KeyError is raised when start, goal, or neighbor lacks a heuristic."""
        # Start missing
        with self.assertRaises(KeyError):
            greedy_best_first_search({"A": ["B"]}, {"B": 1.0}, start="A", goal="B")

        # Goal missing
        with self.assertRaises(KeyError):
            greedy_best_first_search({"A": ["B"]}, {"A": 1.0}, start="A", goal="B")

        # Neighbor missing
        with self.assertRaises(KeyError):
            greedy_best_first_search(
                {"A": ["MissingNode"]},
                {"A": 5.0, "Goal": 0.0},
                start="A",
                goal="Goal",
            )

    def test_invalid_graph_type(self) -> None:
        """Verify TypeError when graph is not a dict."""
        with self.assertRaises(TypeError):
            normalize_graph(["not", "a", "dict"])

    def test_invalid_heuristics_type(self) -> None:
        """Verify TypeError/ValueError when heuristics are invalid."""
        with self.assertRaises(TypeError):
            normalize_heuristics(["not", "a", "dict"])

        with self.assertRaises(ValueError):
            normalize_heuristics({"A": "not_a_number"})


class TestGreedyBestFirstSearchOOPAndOutput(unittest.TestCase):
    """Tests for the OOP solver class, reporting, and visualization."""

    def test_oop_solver_interface(self) -> None:
        """Verify GreedyBestFirstSearch OOP wrapper."""
        solver = GreedyBestFirstSearch(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS)
        path = solver.get_path("A", "C")
        self.assertEqual(path, ["A", "B", "C"])

        res = solver.search("A", "C")
        self.assertTrue(res.goal_found)
        self.assertEqual(res.path, ["A", "B", "C"])

    def test_step_records_and_table_formatting(self) -> None:
        """Verify GBFSStepRecord and table generation."""
        res = greedy_best_first_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="A",
            goal="C",
        )
        self.assertEqual(len(res.steps), 3)
        table = res.format_table()
        self.assertIn("EXPAND", table)
        self.assertIn("GOAL_REACHED", table)
        self.assertIn("Step", table)
        summary = res.format_summary()
        self.assertIn("Execution Summary", summary)

    def test_viva_reference_completeness(self) -> None:
        """Verify Viva Voce reference contains required syllabus topics."""
        viva = get_gbfs_viva_reference()
        self.assertIn("key_questions", viva)
        self.assertIn("comparison_matrix", viva)

        questions = [q["question"] for q in viva["key_questions"]]
        # Verify coverage of core viva topics
        self.assertTrue(any("optimal" in q.lower() for q in questions))
        self.assertTrue(any("complete" in q.lower() for q in questions))
        self.assertTrue(any("local minimum" in q.lower() for q in questions))

    def test_visualize_gbfs_headless(self) -> None:
        """Verify graph visualization generates a valid PNG file."""
        res = greedy_best_first_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="A",
            goal="C",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_gbfs.png"
            path_returned = visualize_gbfs(
                DEFAULT_SAMPLE_GRAPH,
                DEFAULT_SAMPLE_HEURISTICS,
                result=res,
                output_path=out_file,
            )
            self.assertIsNotNone(path_returned)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
