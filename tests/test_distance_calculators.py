"""Unit test suite for Question 24: Manhattan and Euclidean Distance Calculators.

Tests cover:
- Exact syllabus sample input and output verification: (1, 1), (4, 5) -> Manhattan: 7, Euclidean: 5.00
- Core mathematical accuracy for Manhattan, Euclidean, Chebyshev, Octile, and Minkowski distances
- Metric space axioms verification (Non-negativity, Identity, Symmetry, Triangle Inequality)
- Edge cases: Identity points, negative coords, collinear horizontal/vertical, 45-degree diagonal, floating points
- Input validation & type coercion (tuples, lists, Point2D, invalid lengths, non-numeric, NaN, Inf)
- Batch comparison & ASCII table formatting
- String parsing utilities for points and multi-pair expressions
- Heuristic admissibility and movement topology evaluation
- ASCII grid rendering & Matplotlib visualizer
- Viva Voce reference validation
"""

import math
from pathlib import Path
import tempfile
import unittest
from typing import List, Tuple

from src.distance_calculators import (
    Point2D,
    DistanceResult,
    manhattan,
    euclidean,
    chebyshev,
    octile,
    minkowski,
    compute_distances,
    compare_point_pairs,
    format_comparison_table,
    parse_point,
    parse_point_pairs,
    verify_metric_space_axioms,
    compare_heuristics_for_grid_topology,
    render_ascii_distance_grid,
    plot_distance_metrics,
    get_distance_viva_reference,
    _coerce_point,
)


class TestDistanceCalculatorsCore(unittest.TestCase):
    """Test suite for core distance metrics and canonical sample input/output."""

    def test_canonical_sample_input_and_output(self) -> None:
        """Verify Task Requirements on sample input: (1, 1), (4, 5) -> Manhattan: 7, Euclidean: 5.00."""
        p1 = (1, 1)
        p2 = (4, 5)

        m = manhattan(p1, p2)
        e = euclidean(p1, p2)

        # Manhattan: |1-4| + |1-5| = 3 + 4 = 7
        self.assertEqual(m, 7)
        # Euclidean: sqrt(3^2 + 4^2) = sqrt(25) = 5.0
        self.assertAlmostEqual(e, 5.0, places=6)

        result = compute_distances(p1, p2)
        expected_output = "Manhattan: 7, Euclidean: 5.00"
        self.assertEqual(result.formatted(precision=2), expected_output)

    def test_fixed_decimal_formatting(self) -> None:
        """Verify fixed decimal output string: 'Manhattan: 7.00, Euclidean: 5.00'."""
        result = compute_distances((1, 1), (4, 5))
        self.assertEqual(result.formatted_fixed(precision=2), "Manhattan: 7.00, Euclidean: 5.00")

    def test_optional_challenge_chebyshev(self) -> None:
        """Verify Chebyshev distance (Optional Challenge): max(|dx|, |dy|)."""
        p1 = (1, 1)
        p2 = (4, 5)
        # dx = 3, dy = 4 -> Chebyshev = 4
        self.assertEqual(chebyshev(p1, p2), 4)

        result = compute_distances(p1, p2)
        self.assertEqual(
            result.formatted(precision=2, include_chebyshev=True),
            "Manhattan: 7, Euclidean: 5.00, Chebyshev: 4",
        )

    def test_octile_distance(self) -> None:
        """Verify Octile distance: D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)."""
        p1 = (1, 1)
        p2 = (4, 5)
        # dx=3, dy=4 -> 1 * 4 + (sqrt(2) - 1) * 3 = 4 + 3 * 0.41421356 = 5.24264
        expected = 1.0 * 4 + (math.sqrt(2.0) - 1.0) * 3
        self.assertAlmostEqual(octile(p1, p2), expected, places=5)

    def test_minkowski_distance_family(self) -> None:
        """Verify Minkowski distance generalisation connects Manhattan (p=1) and Euclidean (p=2)."""
        p1 = (2, 3)
        p2 = (5, 7)

        # For p=1, Minkowski equals Manhattan
        m_dist = float(manhattan(p1, p2))
        self.assertAlmostEqual(minkowski(p1, p2, p=1.0), m_dist, places=6)

        # For p=2, Minkowski equals Euclidean
        e_dist = euclidean(p1, p2)
        self.assertAlmostEqual(minkowski(p1, p2, p=2.0), e_dist, places=6)

        # For large p, approaches Chebyshev
        self.assertAlmostEqual(minkowski(p1, p2, p=50.0), float(chebyshev(p1, p2)), delta=0.1)

        # ValueError for p < 1
        with self.assertRaises(ValueError):
            minkowski(p1, p2, p=0.5)


class TestGeometricEdgeCases(unittest.TestCase):
    """Test geometric edge cases: identity, collinearity, diagonals, negative coordinates."""

    def test_identity_point_zero_distance(self) -> None:
        """Distance between identical points must be zero across all metrics."""
        origin = (0, 0)
        self.assertEqual(manhattan(origin, origin), 0)
        self.assertEqual(euclidean(origin, origin), 0.0)
        self.assertEqual(chebyshev(origin, origin), 0)
        self.assertEqual(octile(origin, origin), 0.0)

        arbitrary = (-4.5, 12.3)
        self.assertAlmostEqual(manhattan(arbitrary, arbitrary), 0.0, places=6)
        self.assertAlmostEqual(euclidean(arbitrary, arbitrary), 0.0, places=6)

    def test_collinear_horizontal(self) -> None:
        """On horizontal lines (dy=0), Manhattan, Euclidean, and Chebyshev must be identical."""
        p1 = (2, 5)
        p2 = (9, 5)
        # dx = 7, dy = 0
        self.assertEqual(manhattan(p1, p2), 7)
        self.assertEqual(euclidean(p1, p2), 7.0)
        self.assertEqual(chebyshev(p1, p2), 7)
        self.assertEqual(octile(p1, p2), 7.0)

    def test_collinear_vertical(self) -> None:
        """On vertical lines (dx=0), Manhattan, Euclidean, and Chebyshev must be identical."""
        p1 = (4, -3)
        p2 = (4, 8)
        # dx = 0, dy = 11
        self.assertEqual(manhattan(p1, p2), 11)
        self.assertEqual(euclidean(p1, p2), 11.0)
        self.assertEqual(chebyshev(p1, p2), 11)
        self.assertEqual(octile(p1, p2), 11.0)

    def test_diagonal_45_degree(self) -> None:
        """On 45-degree diagonals (dx=dy), Manhattan is dx+dy=2dx, Euclidean is dx*sqrt(2)."""
        p1 = (0, 0)
        p2 = (5, 5)
        self.assertEqual(manhattan(p1, p2), 10)
        self.assertAlmostEqual(euclidean(p1, p2), 5 * math.sqrt(2), places=6)
        self.assertEqual(chebyshev(p1, p2), 5)
        self.assertAlmostEqual(octile(p1, p2), 5 * math.sqrt(2), places=6)

    def test_negative_coordinates_across_quadrants(self) -> None:
        """Points across different quadrants with negative coordinates."""
        p1 = (-3, -4)
        p2 = (3, 4)
        # dx = 6, dy = 8 -> 6-8-10 triangle
        self.assertEqual(manhattan(p1, p2), 14)
        self.assertEqual(euclidean(p1, p2), 10.0)
        self.assertEqual(chebyshev(p1, p2), 8)

    def test_floating_point_coordinates(self) -> None:
        """Sub-pixel or floating point coordinates."""
        p1 = (1.5, 2.25)
        p2 = (4.5, 6.25)
        # dx = 3.0, dy = 4.0 -> M=7.0, E=5.0
        self.assertAlmostEqual(manhattan(p1, p2), 7.0, places=6)
        self.assertAlmostEqual(euclidean(p1, p2), 5.0, places=6)
        self.assertAlmostEqual(chebyshev(p1, p2), 4.0, places=6)


class TestMetricSpaceAxioms(unittest.TestCase):
    """Rigorous verification of the 4 metric space mathematical axioms."""

    def test_all_metrics_satisfy_metric_space_axioms(self) -> None:
        """Ensure all 4 metrics satisfy Non-negativity, Identity, Symmetry, and Triangle Inequality."""
        p1 = (1, 1)
        p2 = (4, 5)
        p3 = (2, 8)

        for name, fn in [
            ("Manhattan", manhattan),
            ("Euclidean", euclidean),
            ("Chebyshev", chebyshev),
            ("Octile", octile),
        ]:
            report = verify_metric_space_axioms(p1, p2, p3, fn)
            self.assertTrue(report["all_satisfied"], f"Metric {name} failed axioms: {report}")
            self.assertTrue(report["non_negativity"]["satisfied"])
            self.assertTrue(report["identity_of_indiscernibles"]["satisfied"])
            self.assertTrue(report["symmetry"]["satisfied"])
            self.assertTrue(report["triangle_inequality"]["satisfied"])

    def test_symmetry_property_explicitly(self) -> None:
        """Verify d(p1, p2) == d(p2, p1) for arbitrary points."""
        p1 = (-7, 13)
        p2 = (19, -4)
        self.assertEqual(manhattan(p1, p2), manhattan(p2, p1))
        self.assertEqual(euclidean(p1, p2), euclidean(p2, p1))
        self.assertEqual(chebyshev(p1, p2), chebyshev(p2, p1))
        self.assertEqual(octile(p1, p2), octile(p2, p1))


class TestInputValidationAndCoercion(unittest.TestCase):
    """Test robustness against diverse input structures and invalid data."""

    def test_valid_input_representations(self) -> None:
        """Test accepting tuples, lists, and Point2D instances."""
        self.assertEqual(manhattan((1, 2), [4, 6]), 7)
        self.assertEqual(manhattan(Point2D(1, 2), Point2D(4, 6)), 7)

    def test_invalid_point_lengths(self) -> None:
        """Tuples with fewer or more than 2 elements must raise ValueError."""
        with self.assertRaises(ValueError):
            _coerce_point((1,))  # 1D
        with self.assertRaises(ValueError):
            _coerce_point((1, 2, 3))  # 3D

    def test_invalid_types(self) -> None:
        """Strings, integers, or None passed as points must raise TypeError."""
        with self.assertRaises(TypeError):
            _coerce_point("1, 2")  # Raw string
        with self.assertRaises(TypeError):
            _coerce_point(42)  # Scalar
        with self.assertRaises(TypeError):
            _coerce_point(("a", "b"))  # Non-numeric elements

    def test_nan_and_inf_handling(self) -> None:
        """NaN and Inf coordinates must raise ValueError."""
        with self.assertRaises(ValueError):
            _coerce_point((float("nan"), 1.0))
        with self.assertRaises(ValueError):
            _coerce_point((1.0, float("inf")))


class TestParsingAndFormatting(unittest.TestCase):
    """Test CLI string parsing and ASCII table formatting."""

    def test_parse_point_strings(self) -> None:
        """Parse various valid point string formats."""
        self.assertEqual(parse_point("(1, 1)"), Point2D(1.0, 1.0))
        self.assertEqual(parse_point("4, 5"), Point2D(4.0, 5.0))
        self.assertEqual(parse_point(" -3.5 , 8.2 "), Point2D(-3.5, 8.2))
        self.assertEqual(parse_point("2 9"), Point2D(2.0, 9.0))

    def test_parse_point_invalid(self) -> None:
        """Invalid strings must raise ValueError."""
        with self.assertRaises(ValueError):
            parse_point("invalid")
        with self.assertRaises(ValueError):
            parse_point("1, 2, 3")

    def test_parse_point_pairs_expression(self) -> None:
        """Parse multi-pair expressions."""
        pairs_str = "(1, 1), (4, 5); (0, 0), (3, 4)"
        pairs = parse_point_pairs(pairs_str)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], (Point2D(1.0, 1.0), Point2D(4.0, 5.0)))
        self.assertEqual(pairs[1], (Point2D(0.0, 0.0), Point2D(3.0, 4.0)))

    def test_compare_point_pairs_and_table(self) -> None:
        """Process batch of pairs and generate formatted table."""
        pairs = [((1, 1), (4, 5)), ((0, 0), (3, 4))]
        results = compare_point_pairs(pairs)
        self.assertEqual(len(results), 2)

        table = format_comparison_table(results, include_chebyshev=True)
        self.assertIn("Pair #", table)
        self.assertIn("Manhattan", table)
        self.assertIn("Euclidean", table)
        self.assertIn("Chebyshev", table)
        self.assertIn("7.00", table)
        self.assertIn("5.00", table)


class TestHeuristicAdmissibilityAndTopology(unittest.TestCase):
    """Test heuristic admissibility evaluation across different movement topologies."""

    def test_orthogonal_topology_dominance(self) -> None:
        """Under orthogonal 4-way movement, Manhattan is exact h* and dominates Euclidean."""
        analysis = compare_heuristics_for_grid_topology((1, 1), (4, 5), topology="orthogonal")
        evals = analysis["evaluations"]

        self.assertTrue(evals["Manhattan"]["is_admissible"])
        self.assertTrue(evals["Manhattan"]["is_exact"])
        self.assertTrue(evals["Euclidean"]["is_admissible"])
        self.assertFalse(evals["Euclidean"]["is_exact"])
        # Manhattan cost >= Euclidean cost (dominance)
        self.assertGreater(evals["Manhattan"]["cost"], evals["Euclidean"]["cost"])

    def test_chebyshev_topology_inadmissibility(self) -> None:
        """Under 8-way uniform King moves, Manhattan overestimates and is INADMISSIBLE."""
        analysis = compare_heuristics_for_grid_topology((1, 1), (4, 5), topology="chebyshev")
        evals = analysis["evaluations"]

        # True cost is max(3, 4) = 4
        self.assertEqual(analysis["true_shortest_cost_without_obstacles"], 4.0)
        # Manhattan gives 7 > 4 -> Inadmissible!
        self.assertFalse(evals["Manhattan"]["is_admissible"])
        self.assertTrue(evals["Chebyshev"]["is_admissible"])
        self.assertTrue(evals["Chebyshev"]["is_exact"])


class TestVisualizationsAndVivaReference(unittest.TestCase):
    """Test ASCII rendering, Matplotlib figure generation, and Viva Voce reference."""

    def test_render_ascii_distance_grid(self) -> None:
        """Render ASCII grid containing Start 'S', Goal 'G', Manhattan '#', and Euclidean '*'."""
        grid_str = render_ascii_distance_grid((1, 1), (4, 5))
        self.assertIn("S", grid_str)
        self.assertIn("G", grid_str)
        self.assertIn("#", grid_str)
        self.assertIn("*", grid_str)
        self.assertIn("P1=(1, 1)", grid_str)
        self.assertIn("P2=(4, 5)", grid_str)

    def test_plot_distance_metrics_execution(self) -> None:
        """Test Matplotlib figure generation and saving to temporary image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_distance_plot.png"
            fig = plot_distance_metrics(
                [((1, 1), (4, 5)), ((0, 0), (3, 4))],
                save_path=out_file,
                show=False,
            )
            if fig is not None:
                self.assertTrue(out_file.exists())
                self.assertGreater(out_file.stat().st_size, 0)

    def test_viva_reference_contains_all_core_topics(self) -> None:
        """Verify viva reference dictionary contains detailed answers for all mandatory topics."""
        viva = get_distance_viva_reference()
        required_topics = [
            "overestimation_vs_underestimation",
            "when_to_use_manhattan_vs_euclidean",
            "admissible_heuristic_definition",
            "consistency_monotonicity",
            "heuristic_dominance",
            "metric_space_axioms",
        ]
        for topic in required_topics:
            self.assertIn(topic, viva)
            self.assertIn("question", viva[topic])
            self.assertIn("answer", viva[topic])
            self.assertGreater(len(viva[topic]["answer"]), 50)


if __name__ == "__main__":
    unittest.main()
