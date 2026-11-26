"""Unit tests for Admissible Heuristic Validator (Question 33).

Validates graph normalization, transpose computation, reverse UCS / Dijkstra shortest path
calculation, heuristic admissibility checking, counterexample generation, edge cases,
optional challenge heuristic repairing, JSON serialization, and syllabus sample output.
"""

from __future__ import annotations

import json
import math
import unittest

from src.admissible_heuristic_validator import (
    CANONICAL_GOAL,
    CANONICAL_GRAPH,
    CANONICAL_HEURISTIC_ADMISSIBLE,
    CANONICAL_HEURISTIC_OVERESTIMATING,
    SYLLABUS_SAMPLE_OUTPUT,
    AdmissibilityReport,
    NodeViolation,
    compute_reverse_graph,
    compute_true_costs,
    format_comparison_table,
    get_canonical_sample_graph,
    get_theoretical_analysis,
    get_viva_reference,
    make_admissible,
    normalize_graph,
    render_ascii_graph,
    validate_admissibility,
)


class TestGraphNormalizationAndReverseUCS(unittest.TestCase):
    """Test suite for graph parsing, transposition, and reverse UCS true cost calculation."""

    def test_normalize_dict_and_list_formats(self) -> None:
        """Verify normalization of both nested dict and adjacency list representations."""
        dict_graph = {"A": {"B": 3.0}, "B": {}}
        list_graph = {"A": [("B", 3.0)], "B": []}

        norm_dict = normalize_graph(dict_graph)
        norm_list = normalize_graph(list_graph)

        self.assertEqual(norm_dict, {"A": {"B": 3.0}, "B": {}})
        self.assertEqual(norm_list, {"A": {"B": 3.0}, "B": {}})

    def test_non_positive_weights_raise_error(self) -> None:
        """Verify that zero or negative edge weights raise ValueError per graph constraints."""
        with self.assertRaises(ValueError):
            normalize_graph({"A": {"B": 0.0}})

        with self.assertRaises(ValueError):
            normalize_graph({"A": [("B", -2.5)]})

    def test_compute_reverse_graph(self) -> None:
        """Verify graph transposition: forward edges u->v become reversed edges v->u."""
        graph = {
            "A": {"B": 3.0, "C": 5.0},
            "B": {"Goal": 4.0},
            "C": {"Goal": 2.0},
            "Goal": {},
        }
        rev = compute_reverse_graph(graph)

        self.assertIn("Goal", rev)
        self.assertEqual(rev["Goal"], {"B": 4.0, "C": 2.0})
        self.assertEqual(rev["B"], {"A": 3.0})
        self.assertEqual(rev["C"], {"A": 5.0})
        self.assertEqual(rev["A"], {})

    def test_compute_true_costs_canonical_graph(self) -> None:
        """Verify exact shortest path costs h*(n) on the canonical syllabus graph."""
        true_costs = compute_true_costs(CANONICAL_GRAPH, CANONICAL_GOAL)

        # Expected true minimum costs to Goal:
        # Goal -> Goal: 0
        # B -> Goal: 5
        # C -> Goal: 4
        # D -> Goal: 7
        # A -> Goal: min(3 + 5, 6 + 4) = 8
        self.assertEqual(true_costs["Goal"], 0.0)
        self.assertEqual(true_costs["B"], 5.0)
        self.assertEqual(true_costs["C"], 4.0)
        self.assertEqual(true_costs["D"], 7.0)
        self.assertEqual(true_costs["A"], 8.0)

    def test_missing_goal_raises_error(self) -> None:
        """Verify that a goal node not present in the graph raises ValueError."""
        with self.assertRaises(ValueError):
            compute_true_costs(CANONICAL_GRAPH, "NonExistentGoal")

    def test_unreachable_node_cost_is_infinity(self) -> None:
        """Verify that nodes without any valid path to the goal receive infinity cost."""
        graph_with_island = {
            "A": {"Goal": 5.0},
            "Island": {"Other": 2.0},
            "Other": {},
            "Goal": {},
        }
        costs = compute_true_costs(graph_with_island, "Goal")
        self.assertEqual(costs["A"], 5.0)
        self.assertEqual(costs["Goal"], 0.0)
        self.assertTrue(math.isinf(costs["Island"]))
        self.assertTrue(math.isinf(costs["Other"]))


class TestAdmissibilityValidation(unittest.TestCase):
    """Test suite for heuristic admissibility validation and counterexample reporting."""

    def test_canonical_sample_overestimation(self) -> None:
        """Verify syllabus sample: h(A)=10 > h*(A)=8 returns False with exact counterexample."""
        report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            CANONICAL_HEURISTIC_OVERESTIMATING,
        )

        self.assertFalse(report.is_admissible)
        self.assertEqual(len(report.violations), 1)

        v = report.violations[0]
        self.assertEqual(v.node, "A")
        self.assertEqual(v.h_val, 10.0)
        self.assertEqual(v.h_star, 8.0)
        self.assertEqual(v.difference, 2.0)

        # Counterexample format verification
        self.assertEqual(report.counterexample, SYLLABUS_SAMPLE_OUTPUT)
        self.assertIn("10 > 8", report.counterexample)

    def test_canonical_admissible_heuristic(self) -> None:
        """Verify that an admissible heuristic passes validation with True."""
        report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            CANONICAL_HEURISTIC_ADMISSIBLE,
        )

        self.assertTrue(report.is_admissible)
        self.assertEqual(len(report.violations), 0)
        self.assertIn("True", report.counterexample)
        self.assertIn("admissible", report.counterexample)

    def test_perfect_heuristic_is_admissible(self) -> None:
        """Verify that a perfect heuristic h(n) = h*(n) is admissible."""
        true_costs = compute_true_costs(CANONICAL_GRAPH, CANONICAL_GOAL)
        report = validate_admissibility(CANONICAL_GRAPH, CANONICAL_GOAL, true_costs)

        self.assertTrue(report.is_admissible)
        self.assertEqual(len(report.violations), 0)

    def test_zero_heuristic_is_admissible(self) -> None:
        """Verify that h(n) = 0 for all nodes (UCS equivalent) is strictly admissible."""
        zero_heuristic = {node: 0.0 for node in CANONICAL_GRAPH}
        report = validate_admissibility(CANONICAL_GRAPH, CANONICAL_GOAL, zero_heuristic)

        self.assertTrue(report.is_admissible)
        self.assertEqual(len(report.violations), 0)

    def test_goal_node_overestimation_violation(self) -> None:
        """Verify that h(Goal) > 0 is flagged as a violation since h*(Goal) = 0."""
        heuristics = dict(CANONICAL_HEURISTIC_ADMISSIBLE)
        heuristics["Goal"] = 1.5  # Violation: goal heuristic must be 0!

        report = validate_admissibility(CANONICAL_GRAPH, CANONICAL_GOAL, heuristics)

        self.assertFalse(report.is_admissible)
        violating_nodes = [v.node for v in report.violations]
        self.assertIn("Goal", violating_nodes)

    def test_multiple_violating_nodes_identified(self) -> None:
        """Verify that all nodes overestimating their true costs are captured."""
        heuristics = {
            "A": 15.0,  # true: 8  -> overestimates
            "B": 9.0,   # true: 5  -> overestimates
            "C": 4.0,   # true: 4  -> admissible
            "D": 12.0,  # true: 7  -> overestimates
            "Goal": 0.0,
        }
        report = validate_admissibility(CANONICAL_GRAPH, CANONICAL_GOAL, heuristics)

        self.assertFalse(report.is_admissible)
        self.assertEqual(len(report.violations), 3)
        violating_nodes = {v.node for v in report.violations}
        self.assertEqual(violating_nodes, {"A", "B", "D"})

    def test_floating_point_near_equality_tolerance(self) -> None:
        """Verify that floating point precision artifacts within 1e-9 are not marked as violations."""
        heuristics = {
            "A": 8.000000000000002,  # Floating-point representation artifact
            "B": 5.0,
            "C": 4.0,
            "D": 7.0,
            "Goal": 0.0,
        }
        report = validate_admissibility(CANONICAL_GRAPH, CANONICAL_GOAL, heuristics)
        self.assertTrue(report.is_admissible)


class TestOptionalChallengeHeuristicRepair(unittest.TestCase):
    """Test suite for Optional Challenge: automatically lowering heuristics to make them admissible."""

    def test_make_admissible_lowers_overestimating_nodes(self) -> None:
        """Verify that overestimating nodes are clamped to h*(n) while admissible ones remain unchanged."""
        heuristics = {
            "A": 10.0,  # true: 8.0 -> should become 8.0
            "B": 3.0,   # true: 5.0 -> should remain 3.0
            "C": 4.0,   # true: 4.0 -> should remain 4.0
            "D": 10.0,  # true: 7.0 -> should become 7.0
            "Goal": 1.0,  # true: 0.0 -> should become 0.0
        }
        true_costs = compute_true_costs(CANONICAL_GRAPH, CANONICAL_GOAL)
        repaired = make_admissible(heuristics, true_costs)

        self.assertEqual(repaired["A"], 8.0)
        self.assertEqual(repaired["B"], 3.0)
        self.assertEqual(repaired["C"], 4.0)
        self.assertEqual(repaired["D"], 7.0)
        self.assertEqual(repaired["Goal"], 0.0)

    def test_repaired_heuristic_passes_full_validation(self) -> None:
        """Verify that running validate_admissibility on repaired heuristic yields True."""
        initial_report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            CANONICAL_HEURISTIC_OVERESTIMATING,
        )
        self.assertFalse(initial_report.is_admissible)

        repaired_report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            initial_report.repaired_heuristics,
        )
        self.assertTrue(repaired_report.is_admissible)
        self.assertEqual(len(repaired_report.violations), 0)


class TestFormattingTheoryAndVivaVoce(unittest.TestCase):
    """Test suite for CLI output formatting, JSON serialization, and Viva Voce guides."""

    def test_format_comparison_table(self) -> None:
        """Verify markdown comparison table generation."""
        report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            CANONICAL_HEURISTIC_OVERESTIMATING,
        )
        table = format_comparison_table(report)

        self.assertIn("| Node |", table)
        self.assertIn("Overestimates", table)
        self.assertIn("Admissible", table)
        self.assertIn("10", table)
        self.assertIn("8", table)

    def test_render_ascii_graph(self) -> None:
        """Verify ASCII topology layout rendering."""
        ascii_map = render_ascii_graph(CANONICAL_GRAPH, CANONICAL_GOAL)
        self.assertIn("Directed Graph Topological Layout", ascii_map)
        self.assertIn("Goal", ascii_map)
        self.assertIn("cost=3", ascii_map)

    def test_report_json_serialization(self) -> None:
        """Verify that AdmissibilityReport serializes to valid JSON."""
        report = validate_admissibility(
            CANONICAL_GRAPH,
            CANONICAL_GOAL,
            CANONICAL_HEURISTIC_OVERESTIMATING,
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)

        self.assertFalse(parsed["is_admissible"])
        self.assertEqual(parsed["violations_count"], 1)
        self.assertEqual(parsed["violations"][0]["node"], "A")
        self.assertEqual(parsed["true_costs"]["A"], 8.0)
        self.assertEqual(parsed["repaired_heuristics"]["A"], 8.0)

    def test_theoretical_analysis_and_viva_references(self) -> None:
        """Verify theoretical foundations and viva voce text are comprehensive."""
        theory = get_theoretical_analysis()
        self.assertIn("THEORETICAL FOUNDATIONS", theory)
        self.assertIn("h(n) <= h*(n)", theory)
        self.assertIn("Reverse UCS", theory)

        viva = get_viva_reference()
        self.assertIn("UNIVERSITY VIVA VOCE PREPARATION GUIDE", viva)
        self.assertIn("definition of an Admissible Heuristic", viva)
        self.assertIn("break the optimality of A*", viva)
        self.assertIn("Reverse UCS from the goal", viva)


if __name__ == "__main__":
    unittest.main()
