"""Comprehensive Unit Test Suite for A* Search Algorithm on a Graph (Question 28).

Validates:
- Exact syllabus sample input & output ('Optimal Path: S -> C -> G, Total Cost: 9').
- Optional Challenge: Frontier (priority queue) contents tracking at each step.
- Node re-evaluation: updating g_cost and f_cost when a cheaper path is discovered.
- Node reopening: reopening closed nodes when an admissible but inconsistent heuristic is used.
- Classical AIMA Romania Map benchmark (Arad to Bucharest: optimal path and cost 418).
- Diamond graph benchmark: multi-hop optimal route discovery.
- Base cases: start == goal, disconnected graphs, unreachable goals.
- Cycle avoidance in cyclic graphs.
- Admissibility and Consistency (monotonicity) verification utilities.
- Comparative analysis: A* vs Uniform Cost Search vs Greedy Best-First Search.
- Tie-breaking policies: FIFO, alphabetical, LIFO.
- Input parsing robustness: JSON, arrow syntax, colon notation, whitespace handling.
- Error handling: missing nodes, missing heuristics, negative edge weights, negative heuristics.
- Result formatting: path summary, ASCII step table, frontier challenge view, JSON dictionary.
- OOP wrapper class interface (AStarSearch).
- Viva Voce study guide data completeness.
- Graphical visualization generation with NetworkX / Matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from src.a_star_search import (
    CYCLIC_GRAPH,
    CYCLIC_HEURISTICS,
    DEFAULT_SAMPLE_GRAPH,
    DEFAULT_SAMPLE_HEURISTICS,
    DIAMOND_GRAPH,
    DIAMOND_HEURISTICS,
    DISCONNECTED_GRAPH,
    DISCONNECTED_HEURISTICS,
    REOPENING_DEMO_GRAPH,
    REOPENING_INCONSISTENT_HEURISTICS,
    ROMANIA_MAP_GRAPH,
    ROMANIA_SLD_HEURISTICS,
    AStarResult,
    AStarSearch,
    AStarStepRecord,
    a_star_search,
    check_admissibility,
    check_consistency,
    compare_astar_vs_ucs_and_greedy,
    compute_all_pairs_shortest_paths,
    get_a_star_viva_reference,
    normalize_graph,
    normalize_heuristics,
    parse_graph,
    parse_heuristics,
    visualize_a_star,
)


class TestAStarSearchCore(unittest.TestCase):
    """Core algorithmic tests for A* Search."""

    def test_syllabus_sample_input_and_output(self) -> None:
        """Verify exact sample from Question 28 prompt: S -> C -> G, Total Cost: 9."""
        result = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "C", "G"])
        self.assertEqual(result.total_cost, 9)
        self.assertEqual(
            result.format_path(),
            "Optimal Path: S -> C -> G, Total Cost: 9",
        )
        self.assertEqual(result.start_node, "S")
        self.assertEqual(result.goal_node, "G")
        # Node expansion order: S -> B -> C -> A -> G
        self.assertEqual(result.nodes_expanded, ["S", "B", "C", "A", "G"])
        # Node A was reached via S (g=3), then re-evaluated via B (g=2)
        self.assertEqual(result.re_evaluations_count, 1)

    def test_optional_challenge_frontier_tracking_at_steps(self) -> None:
        """Verify Optional Challenge: frontier contents tracked at each step."""
        result = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
        frontier_text = result.format_frontier_at_steps()
        self.assertIn("=== A* Frontier (Priority Queue) Contents at Each Step ===", frontier_text)
        self.assertIn("Step 01 | Popped: S", frontier_text)
        self.assertIn("Step 06 | Popped: G", frontier_text)
        self.assertIn("Frontier: []", frontier_text)

        # Verify steps structure
        self.assertEqual(len(result.steps), 6)
        # Step 5 is lazy deletion of superseded entry A(g=3)
        self.assertEqual(result.steps[4].action, "SKIP_SUPERSEDED")
        self.assertEqual(result.steps[4].current_node, "A")
        self.assertEqual(result.steps[4].g_cost, 3)

    def test_node_re_evaluation_with_cheaper_path(self) -> None:
        """Verify that when a cheaper g_cost is found, best_g_costs is updated."""
        result = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
        # S -> A is 3, but S -> B -> A is 1 + 1 = 2
        self.assertEqual(result.g_costs["A"], 2)
        self.assertEqual(result.parents["A"], "B")
        self.assertEqual(result.re_evaluations_count, 1)

    def test_node_reopening_admissible_inconsistent_heuristic(self) -> None:
        """Verify reopening of closed nodes when consistency is violated."""
        # With reopening enabled:
        result_reopen = a_star_search(
            REOPENING_DEMO_GRAPH,
            REOPENING_INCONSISTENT_HEURISTICS,
            start="S",
            goal="G",
            allow_reopen=True,
        )
        self.assertTrue(result_reopen.goal_found)
        self.assertEqual(result_reopen.path, ["S", "B", "A", "G"])
        self.assertEqual(result_reopen.total_cost, 7)
        self.assertEqual(result_reopen.nodes_reopened_count, 1)
        # Node A was expanded, then closed, then reopened and expanded again
        self.assertEqual(result_reopen.nodes_expanded, ["S", "A", "B", "A", "G"])

        # With reopening disabled:
        result_no_reopen = a_star_search(
            REOPENING_DEMO_GRAPH,
            REOPENING_INCONSISTENT_HEURISTICS,
            start="S",
            goal="G",
            allow_reopen=False,
        )
        self.assertTrue(result_no_reopen.goal_found)
        # Missed the cheaper route through B -> A, returned suboptimal path S -> A -> G (cost 9)
        self.assertEqual(result_no_reopen.path, ["S", "A", "G"])
        self.assertEqual(result_no_reopen.total_cost, 9)
        self.assertEqual(result_no_reopen.nodes_reopened_count, 0)

    def test_start_equals_goal(self) -> None:
        """Verify base case where start node is the goal node."""
        result = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="S",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S"])
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.format_path(), "Optimal Path: S, Total Cost: 0")
        self.assertEqual(result.nodes_expanded, ["S"])

    def test_disconnected_graph_unreachable_goal(self) -> None:
        """Verify handling of disconnected graphs where goal cannot be reached."""
        result = a_star_search(
            DISCONNECTED_GRAPH,
            DISCONNECTED_HEURISTICS,
            start="A",
            goal="G",
        )
        self.assertFalse(result.goal_found)
        self.assertIsNone(result.path)
        self.assertEqual(result.total_cost, float("inf"))
        self.assertIn("None (Unreachable)", result.format_path())

    def test_cyclic_graph_termination(self) -> None:
        """Verify that cyclic graphs do not produce infinite loops and find the optimal path."""
        result = a_star_search(
            CYCLIC_GRAPH,
            CYCLIC_HEURISTICS,
            start="A",
            goal="G",
        )
        self.assertTrue(result.goal_found)
        # Shortest path in CYCLIC_GRAPH:
        # A -> B: 2, B -> C: 1, C -> D: 1, D -> G: 2 => 2 + 1 + 1 + 2 = 6
        # Or A -> C: 5, C -> D: 1, D -> G: 2 => 8
        self.assertEqual(result.path, ["A", "B", "C", "D", "G"])
        self.assertEqual(result.total_cost, 6)

    def test_diamond_graph_benchmark(self) -> None:
        """Verify diamond graph where multi-hop path is cheaper than single-hop."""
        result = a_star_search(
            DIAMOND_GRAPH,
            DIAMOND_HEURISTICS,
            start="S",
            goal="G",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "A", "C", "G"])
        self.assertEqual(result.total_cost, 4)

    def test_romania_map_benchmark(self) -> None:
        """Verify AIMA Romania Map benchmark (Arad to Bucharest: 418 km)."""
        result = a_star_search(
            ROMANIA_MAP_GRAPH,
            ROMANIA_SLD_HEURISTICS,
            start="Arad",
            goal="Bucharest",
        )
        self.assertTrue(result.goal_found)
        self.assertEqual(
            result.path,
            ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"],
        )
        self.assertEqual(result.total_cost, 418)


class TestAStarHeuristicAnalysis(unittest.TestCase):
    """Tests for admissibility and consistency verification utilities."""

    def test_admissibility_check_valid(self) -> None:
        """Verify admissibility check on valid admissible heuristics."""
        report = check_admissibility(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS, goal="G")
        self.assertTrue(report["is_admissible"])
        self.assertEqual(len(report["violations"]), 0)

    def test_admissibility_check_violation(self) -> None:
        """Verify detection of inadmissible overestimating heuristics."""
        inadmissible_h = dict(DEFAULT_SAMPLE_HEURISTICS)
        inadmissible_h["C"] = 100  # Actual shortest distance from C to G is 5!
        report = check_admissibility(DEFAULT_SAMPLE_GRAPH, inadmissible_h, goal="G")
        self.assertFalse(report["is_admissible"])
        self.assertEqual(len(report["violations"]), 1)
        self.assertEqual(report["violations"][0]["node"], "C")
        self.assertEqual(report["violations"][0]["overestimation"], 95)

    def test_consistency_check_valid(self) -> None:
        """Verify consistency check on monotonic heuristics."""
        report = check_consistency(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS)
        self.assertTrue(report["is_consistent"])
        self.assertEqual(len(report["violations"]), 0)

    def test_consistency_check_violation(self) -> None:
        """Verify consistency check flags triangle inequality violations."""
        report = check_consistency(REOPENING_DEMO_GRAPH, REOPENING_INCONSISTENT_HEURISTICS)
        self.assertFalse(report["is_consistent"])
        self.assertTrue(len(report["violations"]) > 0)

    def test_compute_all_pairs_shortest_paths(self) -> None:
        """Verify Dijkstra all-pairs shortest paths calculation."""
        dists = compute_all_pairs_shortest_paths(DEFAULT_SAMPLE_GRAPH)
        self.assertEqual(dists["S"]["G"], 9)
        self.assertEqual(dists["C"]["G"], 5)
        self.assertEqual(dists["A"]["G"], 8)
        self.assertEqual(dists["B"]["G"], 9)


class TestAStarComparativeAndOOP(unittest.TestCase):
    """Tests for comparative analysis, OOP interface, and serialization."""

    def test_compare_astar_vs_ucs_and_greedy(self) -> None:
        """Verify comparison across A*, UCS, and Greedy Best-First Search."""
        comp = compare_astar_vs_ucs_and_greedy(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
        self.assertEqual(comp["start"], "S")
        self.assertEqual(comp["goal"], "G")
        self.assertEqual(comp["a_star"]["path"], ["S", "C", "G"])
        self.assertEqual(comp["a_star"]["total_cost"], 9)
        self.assertEqual(comp["uniform_cost_search"]["path"], ["S", "C", "G"])
        self.assertEqual(comp["uniform_cost_search"]["total_cost"], 9)
        self.assertEqual(comp["greedy_best_first_search"]["path"], ["S", "C", "G"])

    def test_oop_astar_search_wrapper(self) -> None:
        """Verify object-oriented wrapper class AStarSearch."""
        solver = AStarSearch(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS)
        result = solver.search(start="S", goal="G")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "C", "G"])
        self.assertEqual(result.total_cost, 9)

        path, cost = solver.find_path("S", "G")
        self.assertEqual(path, ["S", "C", "G"])
        self.assertEqual(cost, 9)

    def test_result_to_dict_and_step_table(self) -> None:
        """Verify dictionary serialization and ASCII table formatting."""
        result = a_star_search(
            DEFAULT_SAMPLE_GRAPH,
            DEFAULT_SAMPLE_HEURISTICS,
            start="S",
            goal="G",
        )
        d = result.to_dict()
        self.assertEqual(d["path"], ["S", "C", "G"])
        self.assertEqual(d["total_cost"], 9)
        self.assertEqual(d["goal_found"], True)
        self.assertEqual(d["re_evaluations_count"], 1)

        table = result.format_step_table()
        self.assertIn("Step  | Node   | g(n)  | h(n)  | f(n)", table)
        self.assertIn("GOAL_REACHED", table)
        self.assertIn("SKIP_SUPERSEDED", table)

    def test_tie_breaking_modes(self) -> None:
        """Verify FIFO, alphabetical, and LIFO tie-breaking modes execute without errors."""
        for mode in ("fifo", "alphabetical", "lifo"):
            res = a_star_search(
                DEFAULT_SAMPLE_GRAPH,
                DEFAULT_SAMPLE_HEURISTICS,
                start="S",
                goal="G",
                tie_breaking=mode,
            )
            self.assertTrue(res.goal_found)
            self.assertEqual(res.path, ["S", "C", "G"])
            self.assertEqual(res.total_cost, 9)


class TestAStarParsingAndErrorHandling(unittest.TestCase):
    """Tests for input parsing robustness and error handling."""

    def test_parse_graph_semicolon(self) -> None:
        """Verify parsing semicolon-delimited graph strings."""
        s = "S: A=3, B=1, C=4; A: G=8; B: A=1; C: G=5; G:"
        g = parse_graph(s)
        self.assertEqual(g["S"], [("A", 3), ("B", 1), ("C", 4)])
        self.assertEqual(g["A"], [("G", 8)])
        self.assertEqual(g["G"], [])

    def test_parse_graph_arrow(self) -> None:
        """Verify parsing arrow-delimited graph strings."""
        s = "S->A: 3, S->B: 1, A->G: 8"
        g = parse_graph(s)
        self.assertEqual(g["S"], [("A", 3), ("B", 1)])
        self.assertEqual(g["A"], [("G", 8)])

    def test_parse_graph_json(self) -> None:
        """Verify parsing JSON graph string."""
        s = '{"S": [["A", 3], ["B", 1]], "A": [["G", 8]]}'
        g = parse_graph(s)
        self.assertEqual(g["S"], [("A", 3), ("B", 1)])
        self.assertEqual(g["A"], [("G", 8)])

    def test_parse_heuristics_comma_and_json(self) -> None:
        """Verify parsing heuristic strings."""
        s1 = "S: 7, A: 6, B: 7, C: 4, G: 0"
        h1 = parse_heuristics(s1)
        self.assertEqual(h1["S"], 7)
        self.assertEqual(h1["G"], 0)

        s2 = '{"S": 7, "A": 6, "G": 0}'
        h2 = parse_heuristics(s2)
        self.assertEqual(h2["S"], 7)

    def test_missing_start_or_goal_node(self) -> None:
        """Verify ValueError when start or goal node does not exist in graph."""
        with self.assertRaises(ValueError):
            a_star_search(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS, start="NonExistent", goal="G")
        with self.assertRaises(ValueError):
            a_star_search(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS, start="S", goal="NonExistent")

    def test_missing_heuristic(self) -> None:
        """Verify KeyError when a node's heuristic value is missing."""
        incomplete_h = {"S": 7, "A": 6, "B": 7, "C": 4}  # Missing 'G'
        with self.assertRaises(KeyError):
            a_star_search(DEFAULT_SAMPLE_GRAPH, incomplete_h, start="S", goal="G")

    def test_negative_edge_weight_rejection(self) -> None:
        """Verify ValueError when graph contains negative edge weights."""
        bad_graph = {"S": [("A", -2)], "A": []}
        with self.assertRaises(ValueError):
            a_star_search(bad_graph, {"S": 1, "A": 0}, start="S", goal="A")

    def test_negative_heuristic_rejection(self) -> None:
        """Verify ValueError when heuristic is negative."""
        with self.assertRaises(ValueError):
            a_star_search(DEFAULT_SAMPLE_GRAPH, {"S": -1, "A": 0, "B": 0, "C": 0, "G": 0}, start="S", goal="G")

    def test_viva_reference_completeness(self) -> None:
        """Verify university viva voce guide data completeness."""
        viva = get_a_star_viva_reference()
        self.assertIn("f(n) = g(n) + h(n)", viva["core_formula"]["f_n"])
        self.assertTrue(len(viva["questions_and_answers"]) >= 5)

    def test_visualize_a_star_file_generation(self) -> None:
        """Verify visualize_a_star writes an image file when matplotlib/networkx are present."""
        res = a_star_search(DEFAULT_SAMPLE_GRAPH, DEFAULT_SAMPLE_HEURISTICS, start="S", goal="G")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_astar.png"
            saved = visualize_a_star(res, output_path=out_path)
            if saved is not None:
                self.assertTrue(out_path.exists())
                self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
