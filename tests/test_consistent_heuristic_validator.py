"""Unit tests for Consistent Heuristic Validator (Question 34).

Validates graph normalization, edge extraction, triangle inequality verification,
canonical syllabus sample input and output, edge boundary equality cases,
multiple violations detection, missing node error handling, directed edge asymmetry,
admissible-vs-consistent distinction, optional challenge programmatic induction proof,
ASCII formatting, JSON serialization, and Viva voce guide.
"""

from __future__ import annotations

import json
import math
import unittest

from src.consistent_heuristic_validator import (
    CANONICAL_EXTENDED_GRAPH,
    CANONICAL_EXTENDED_HEURISTIC_CONSISTENT,
    CANONICAL_EXTENDED_HEURISTIC_INCONSISTENT_ADMISSIBLE,
    CANONICAL_SAMPLE_GRAPH,
    CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
    CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
    SYLLABUS_SAMPLE_OUTPUT,
    ConsistencyProofReport,
    ConsistencyReport,
    EdgeViolation,
    compute_shortest_paths_and_costs,
    extract_all_directed_edges,
    format_consistency_table,
    get_theoretical_analysis,
    get_viva_reference,
    is_consistent_heuristic,
    normalize_graph,
    prove_consistency_implies_admissibility,
    render_ascii_graph,
    validate_consistency,
)


class TestGraphNormalizationAndEdgeExtraction(unittest.TestCase):
    """Test suite for normalizing graph inputs and extracting directed edges."""

    def test_normalize_nested_dict(self) -> None:
        """Verify normalization from nested dict format."""
        g = {"A": {"B": 2.5}, "B": {}}
        norm = normalize_graph(g)
        self.assertEqual(norm, {"A": {"B": 2.5}, "B": {}})

    def test_normalize_adjacency_list(self) -> None:
        """Verify normalization from adjacency list format."""
        g = {"A": [("B", 2.0), ("C", 4.0)], "B": [], "C": []}
        norm = normalize_graph(g)
        self.assertEqual(norm, {"A": {"B": 2.0, "C": 4.0}, "B": {}, "C": {}})

    def test_normalize_flat_edge_list(self) -> None:
        """Verify normalization from flat edge 3-tuple list."""
        g = [("A", "B", 2.0), ("B", "C", 3.0)]
        norm = normalize_graph(g)
        self.assertEqual(norm["A"]["B"], 2.0)
        self.assertEqual(norm["B"]["C"], 3.0)
        self.assertEqual(norm["C"], {})

    def test_invalid_edge_weights_raise_error(self) -> None:
        """Verify that zero or negative edge costs raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_graph({"A": {"B": 0.0}})
        with self.assertRaises(ValueError):
            normalize_graph({"A": {"B": -1.5}})
        with self.assertRaises(ValueError):
            normalize_graph([("A", "B", -2.0)])

    def test_extract_all_directed_edges(self) -> None:
        """Verify extraction of all directed edges sorted deterministically."""
        norm = {"B": {"C": 1.0}, "A": {"B": 2.0, "C": 5.0}, "C": {}}
        edges = extract_all_directed_edges(norm)
        self.assertEqual(
            edges,
            [("A", "B", 2.0), ("A", "C", 5.0), ("B", "C", 1.0)],
        )


class TestConsistencyValidation(unittest.TestCase):
    """Test suite for core triangle inequality checking and violation reporting."""

    def test_canonical_syllabus_sample_violation(self) -> None:
        """Verify exact sample input and output from syllabus:
        Input: h(A)=5, h(B)=2, cost(A, B)=2
        Output: False. Violation at edge A->B: 5 > 2 + 2.
        """
        report = validate_consistency(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
        )
        self.assertFalse(report.is_consistent)
        self.assertEqual(len(report.violations), 1)
        v = report.violations[0]
        self.assertEqual(v.u, "A")
        self.assertEqual(v.v, "B")
        self.assertEqual(v.cost, 2.0)
        self.assertEqual(v.h_u, 5.0)
        self.assertEqual(v.h_v, 2.0)
        self.assertEqual(v.difference, 1.0)  # 5 - (2 + 2) = 1

        # Check exact syllabus message string
        self.assertEqual(report.summary_message, SYLLABUS_SAMPLE_OUTPUT)
        self.assertEqual(
            report.summary_message,
            "False. Violation at edge A->B: 5 > 2 + 2.",
        )
        self.assertFalse(is_consistent_heuristic(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
        ))

    def test_canonical_sample_consistent(self) -> None:
        """Verify that when h(A) = 4, h(B) = 2, cost(A, B) = 2, consistency holds."""
        report = validate_consistency(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
        )
        self.assertTrue(report.is_consistent)
        self.assertEqual(len(report.violations), 0)
        self.assertIn("True", report.summary_message)
        self.assertTrue(is_consistent_heuristic(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_CONSISTENT,
        ))

    def test_exact_equality_boundary_is_consistent(self) -> None:
        """Verify that h(u) == c + h(v) satisfies the non-strict inequality h(u) <= c + h(v)."""
        graph = {"X": {"Y": 3.0}, "Y": {}}
        heuristic = {"X": 7.0, "Y": 4.0}  # 7 <= 3 + 4 holds exactly
        report = validate_consistency(graph, heuristic)
        self.assertTrue(report.is_consistent)

    def test_strict_underestimate_is_consistent(self) -> None:
        """Verify that h(u) < c + h(v) is valid (e.g., zero heuristic)."""
        graph = {"X": {"Y": 5.0}, "Y": {}}
        heuristic = {"X": 0.0, "Y": 0.0}
        report = validate_consistency(graph, heuristic)
        self.assertTrue(report.is_consistent)

    def test_multiple_violations_detected(self) -> None:
        """Verify detection and ordering of multiple edge violations in a larger graph."""
        graph = {
            "A": {"B": 1.0, "C": 2.0},
            "B": {"Goal": 1.0},
            "C": {"Goal": 1.0},
            "Goal": {},
        }
        # Violations at:
        # A->B: h(A)=10 > 1 + 2 = 3 (diff 7)
        # A->C: h(A)=10 > 2 + 1 = 3 (diff 7)
        # B->Goal: h(B)=2 <= 1 + 0 = 1? 2 > 1 (diff 1)
        heuristic = {"A": 10.0, "B": 2.0, "C": 1.0, "Goal": 0.0}
        report = validate_consistency(graph, heuristic)

        self.assertFalse(report.is_consistent)
        self.assertGreaterEqual(len(report.violations), 2)
        violating_edges = [(v.u, v.v) for v in report.violations]
        self.assertIn(("A", "B"), violating_edges)
        self.assertIn(("A", "C"), violating_edges)
        self.assertIn(("B", "Goal"), violating_edges)

    def test_missing_node_in_heuristic_raises_key_error(self) -> None:
        """Verify that missing node estimates in heuristic raise KeyError."""
        graph = {"A": {"B": 2.0}, "B": {}}
        with self.assertRaises(KeyError):
            validate_consistency(graph, {"A": 5.0})  # Missing 'B'

    def test_directed_edge_asymmetry(self) -> None:
        """Verify that directed edges are evaluated strictly in the forward direction."""
        # A -> B exists with cost 2, but B -> A does NOT exist
        graph = {"A": {"B": 2.0}, "B": {}}
        # h(A) = 3 <= 2 + 5 = 7 (Consistent A->B)
        # Even though h(B)=5 > 2 + 3=5 is boundary, B has no outgoing edge to A
        heuristic = {"A": 3.0, "B": 5.0}
        report = validate_consistency(graph, heuristic)
        self.assertTrue(report.is_consistent)


class TestConsistencyVsAdmissibility(unittest.TestCase):
    """Test suite comparing consistency and admissibility properties."""

    def test_admissible_but_inconsistent_heuristic(self) -> None:
        """Verify that a heuristic can be admissible (underestimating h*) yet inconsistent."""
        report = validate_consistency(
            CANONICAL_EXTENDED_GRAPH,
            CANONICAL_EXTENDED_HEURISTIC_INCONSISTENT_ADMISSIBLE,
        )
        # Triangle inequality fails on edge A->B: h(A)=3 > c(A,B)+h(B)=1+0.5=1.5
        self.assertFalse(report.is_consistent)
        violating_edges = [(v.u, v.v) for v in report.violations]
        self.assertIn(("A", "B"), violating_edges)

    def test_optional_challenge_programmatic_proof(self) -> None:
        """Verify Optional Challenge: Programmatic path induction proving consistency => admissibility."""
        proof_report = prove_consistency_implies_admissibility(
            CANONICAL_EXTENDED_GRAPH,
            goal_node="Goal",
            heuristic=CANONICAL_EXTENDED_HEURISTIC_CONSISTENT,
        )
        self.assertTrue(proof_report.is_consistent)
        self.assertTrue(proof_report.is_admissible)
        self.assertTrue(proof_report.proof_holds)

        # Verify telescoping steps for node 'A'
        steps_a = proof_report.telescoping_steps["A"]
        self.assertGreater(len(steps_a), 0)
        for step in steps_a:
            self.assertTrue(step.is_valid)

        # Check true shortest costs
        self.assertEqual(proof_report.true_shortest_costs["Goal"], 0.0)
        self.assertEqual(proof_report.true_shortest_costs["C"], 1.0)
        self.assertEqual(proof_report.true_shortest_costs["B"], 3.0)  # B->C (2) + C->Goal (1) = 3 < 5
        self.assertEqual(proof_report.true_shortest_costs["A"], 4.0)  # A->B (1) + B->C (2) + C->Goal (1) = 4

    def test_compute_shortest_paths_missing_goal_raises_error(self) -> None:
        """Verify error handling when goal node is absent from graph."""
        with self.assertRaises(ValueError):
            compute_shortest_paths_and_costs(CANONICAL_SAMPLE_GRAPH, "MissingGoal")


class TestPresentationAndSerialization(unittest.TestCase):
    """Test suite for ASCII table formatting, diagram rendering, and JSON serialization."""

    def test_format_consistency_table_contains_key_elements(self) -> None:
        """Verify that ASCII table outputs columns, edge costs, inequalities, and status."""
        report = validate_consistency(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
        )
        table_str = format_consistency_table(report)
        self.assertIn("Edge (u->v)", table_str)
        self.assertIn("A -> B", table_str)
        self.assertIn("❌ Violation", table_str)

    def test_render_ascii_graph(self) -> None:
        """Verify graph ASCII rendering contains nodes, costs, and heuristics."""
        rendered = render_ascii_graph(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
        )
        self.assertIn("Node [A] (h = 5):", rendered)
        self.assertIn("--(cost: 2)--> Node [B] (h = 2)", rendered)

    def test_to_dict_and_to_json(self) -> None:
        """Verify JSON export and dictionary conversions."""
        report = validate_consistency(
            CANONICAL_SAMPLE_GRAPH,
            CANONICAL_SAMPLE_HEURISTIC_VIOLATING,
        )
        d = report.to_dict()
        self.assertFalse(d["is_consistent"])
        self.assertEqual(d["violations_count"], 1)
        self.assertEqual(d["primary_violation"]["u"], "A")

        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["violations_count"], 1)

    def test_theory_and_viva_reference_non_empty(self) -> None:
        """Verify that theoretical analysis and viva voce guides are present and descriptive."""
        theory = get_theoretical_analysis()
        viva = get_viva_reference()
        self.assertIn("TRIANGLE INEQUALITY", theory)
        self.assertIn("WHY CONSISTENCY PREVENTS A* FROM REOPENING NODES", theory)
        self.assertIn("UNIVERSITY VIVA VOCE EXAMINATION GUIDE", viva)
        self.assertIn("Q1: What is the triangle inequality", viva)


if __name__ == "__main__":
    unittest.main()
