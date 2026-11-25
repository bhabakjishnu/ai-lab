"""Unit tests for Uniform Cost Search (UCS) vs A* Performance Comparison (Question 32).

Verifies grid construction, heuristic calculation, UCS and A* search correctness,
exact canonical expansion counts (UCS=380, A*=45), search optimality, edge cases,
JSON serialization, ASCII map rendering, and Matplotlib side-by-side visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.ucs_vs_astar_comparison import (
    CANONICAL_20X20_OBSTACLES,
    DEFAULT_20X20_HEIGHT,
    DEFAULT_20X20_WIDTH,
    DEFAULT_GOAL_COORD,
    DEFAULT_START_COORD,
    SYLLABUS_SAMPLE_OUTPUT,
    ComparisonResult,
    Coordinate,
    Grid2D,
    SearchResult,
    chebyshev_distance,
    compare_ucs_and_astar,
    create_canonical_20x20_grid,
    euclidean_distance,
    get_theoretical_analysis,
    get_viva_reference,
    manhattan_distance,
    solve_astar,
    solve_grid_pathfinding,
    solve_ucs,
    visualize_ucs_vs_astar,
    zero_heuristic,
)


class TestGrid2DEnvironment(unittest.TestCase):
    """Test suite validating 2D grid topological environment and constraints."""

    def test_default_grid_initialization(self) -> None:
        """Verify default 20x20 grid geometry and boundary conditions."""
        grid = Grid2D()
        self.assertEqual(grid.width, 20)
        self.assertEqual(grid.height, 20)
        self.assertEqual(grid.total_cells, 400)
        self.assertEqual(grid.free_cells_count, 400)
        self.assertEqual(grid.start, (0, 0))
        self.assertEqual(grid.goal, (19, 19))
        self.assertEqual(len(grid.obstacles), 0)

    def test_custom_grid_bounds_and_passability(self) -> None:
        """Verify in_bounds and is_passable logic with custom obstacles."""
        obstacles = {(1, 1), (2, 2)}
        grid = Grid2D(width=5, height=5, obstacles=obstacles, start=(0, 0), goal=(4, 4))
        self.assertTrue(grid.in_bounds((0, 0)))
        self.assertTrue(grid.in_bounds((4, 4)))
        self.assertFalse(grid.in_bounds((-1, 0)))
        self.assertFalse(grid.in_bounds((0, 5)))
        self.assertFalse(grid.in_bounds((5, 0)))

        self.assertTrue(grid.is_passable((0, 1)))
        self.assertFalse(grid.is_passable((1, 1)))
        self.assertFalse(grid.is_passable((2, 2)))
        self.assertFalse(grid.is_passable((5, 5)))

    def test_invalid_dimensions_raise_error(self) -> None:
        """Verify that zero or negative grid dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            Grid2D(width=0, height=10)
        with self.assertRaises(ValueError):
            Grid2D(width=10, height=-5)

    def test_out_of_bounds_start_or_goal_raises_error(self) -> None:
        """Verify that start or goal out of bounds raises ValueError."""
        with self.assertRaises(ValueError):
            Grid2D(width=5, height=5, start=(-1, 0))
        with self.assertRaises(ValueError):
            Grid2D(width=5, height=5, goal=(5, 5))

    def test_start_or_goal_on_obstacle_raises_error(self) -> None:
        """Verify that placing start or goal on an obstacle raises ValueError."""
        with self.assertRaises(ValueError):
            Grid2D(width=5, height=5, obstacles={(0, 0)}, start=(0, 0), goal=(4, 4))
        with self.assertRaises(ValueError):
            Grid2D(width=5, height=5, obstacles={(4, 4)}, start=(0, 0), goal=(4, 4))

    def test_get_neighbors_connectivity(self) -> None:
        """Verify 4-connected orthogonal neighbor discovery."""
        grid = Grid2D(width=5, height=5, obstacles={(0, 1)})
        # Corner cell at (0, 0): Right is blocked by (0, 1), Down is (1, 0)
        neighbors = grid.get_neighbors((0, 0))
        self.assertEqual(len(neighbors), 1)
        self.assertIn((1, 0), neighbors)

        # Center cell at (2, 2) without adjacent obstacles has 4 neighbors
        center_neighbors = grid.get_neighbors((2, 2))
        self.assertEqual(len(center_neighbors), 4)
        self.assertIn((2, 3), center_neighbors)
        self.assertIn((3, 2), center_neighbors)
        self.assertIn((2, 1), center_neighbors)
        self.assertIn((1, 2), center_neighbors)

    def test_matrix_roundtrip(self) -> None:
        """Verify conversion to and from binary matrix."""
        mat = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        grid = Grid2D.from_matrix(mat, start=(0, 0), goal=(2, 2))
        self.assertEqual(grid.width, 3)
        self.assertEqual(grid.height, 3)
        self.assertIn((0, 1), grid.obstacles)
        self.assertIn((1, 2), grid.obstacles)
        self.assertEqual(grid.to_matrix(), mat)

    def test_canonical_20x20_grid_properties(self) -> None:
        """Verify canonical 20x20 benchmark grid parameters."""
        grid = create_canonical_20x20_grid()
        self.assertEqual(grid.width, 20)
        self.assertEqual(grid.height, 20)
        self.assertEqual(grid.total_cells, 400)
        self.assertEqual(len(grid.obstacles), 19)
        self.assertEqual(grid.free_cells_count, 381)
        self.assertEqual(grid.start, (0, 0))
        self.assertEqual(grid.goal, (19, 19))
        self.assertEqual(sorted(list(grid.obstacles)), CANONICAL_20X20_OBSTACLES)

    def test_random_grid_generation(self) -> None:
        """Verify deterministic random grid generation using seed."""
        grid1 = Grid2D.generate_random(width=10, height=10, obstacle_density=0.15, seed=123)
        grid2 = Grid2D.generate_random(width=10, height=10, obstacle_density=0.15, seed=123)
        self.assertEqual(grid1.obstacles, grid2.obstacles)
        self.assertEqual(len(grid1.obstacles), 15)
        self.assertNotIn(grid1.start, grid1.obstacles)
        self.assertNotIn(grid1.goal, grid1.obstacles)


class TestHeuristicCalculations(unittest.TestCase):
    """Test suite validating distance metric calculation accuracy."""

    def test_manhattan_distance(self) -> None:
        """Verify L1 Manhattan distance calculation."""
        self.assertEqual(manhattan_distance((0, 0), (0, 0)), 0)
        self.assertEqual(manhattan_distance((0, 0), (19, 19)), 38)
        self.assertEqual(manhattan_distance((5, 12), (8, 3)), 3 + 9)
        self.assertEqual(manhattan_distance((10, 5), (2, 8)), manhattan_distance((2, 8), (10, 5)))

    def test_zero_heuristic(self) -> None:
        """Verify zero heuristic returns 0 identically for all coordinate pairs."""
        self.assertEqual(zero_heuristic((0, 0), (19, 19)), 0)
        self.assertEqual(zero_heuristic((5, 5), (10, 10)), 0)

    def test_euclidean_distance(self) -> None:
        """Verify L2 Euclidean distance calculation."""
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(euclidean_distance((0, 0), (0, 0)), 0.0)

    def test_chebyshev_distance(self) -> None:
        """Verify Chebyshev L-infinity distance calculation."""
        self.assertEqual(chebyshev_distance((0, 0), (3, 7)), 7)
        self.assertEqual(chebyshev_distance((4, 4), (4, 4)), 0)


class TestSearchAlgorithmsAndComparison(unittest.TestCase):
    """Test suite verifying search correctness, optimality, and comparison."""

    def test_trivial_start_equals_goal(self) -> None:
        """Verify edge case where start coordinate equals goal coordinate."""
        grid = Grid2D(width=5, height=5, start=(2, 2), goal=(2, 2))
        ucs_res = solve_ucs(grid)
        self.assertTrue(ucs_res.success)
        self.assertEqual(ucs_res.nodes_expanded, 0)
        self.assertEqual(ucs_res.path_cost, 0)
        self.assertEqual(ucs_res.path, [(2, 2)])

        astar_res = solve_astar(grid)
        self.assertTrue(astar_res.success)
        self.assertEqual(astar_res.nodes_expanded, 0)
        self.assertEqual(astar_res.path_cost, 0)
        self.assertEqual(astar_res.path, [(2, 2)])

    def test_blocked_goal_no_path(self) -> None:
        """Verify search termination when goal is completely unreachable."""
        # Enclose goal (2, 2) on a 3x3 grid with obstacles at (1, 2) and (2, 1)
        obstacles = {(1, 2), (2, 1)}
        grid = Grid2D(width=3, height=3, obstacles=obstacles, start=(0, 0), goal=(2, 2))
        ucs_res = solve_ucs(grid)
        self.assertFalse(ucs_res.success)
        self.assertEqual(ucs_res.path, [])

        astar_res = solve_astar(grid)
        self.assertFalse(astar_res.success)
        self.assertEqual(astar_res.path, [])

    def test_canonical_20x20_benchmark_exact_counts(self) -> None:
        """Verify that canonical 20x20 benchmark produces EXACT syllabus sample counts.

        Syllabus Sample Output:
            UCS expanded: 380 nodes. A* expanded: 45 nodes.
        """
        comp = compare_ucs_and_astar()

        # 1. Exact node expansion counts
        self.assertEqual(comp.ucs_result.nodes_expanded, 380)
        self.assertEqual(comp.astar_result.nodes_expanded, 45)

        # 2. Sample output formatting
        sample_str = comp.format_sample_output(use_literal_sample=False)
        self.assertEqual(sample_str, "UCS expanded: 380 nodes. A* expanded: 45 nodes.")
        self.assertEqual(sample_str, SYLLABUS_SAMPLE_OUTPUT)

        # 3. Path optimality: both algorithms discover strictly identical optimal path cost
        self.assertTrue(comp.ucs_result.success)
        self.assertTrue(comp.astar_result.success)
        self.assertEqual(comp.ucs_result.path_cost, 38)
        self.assertEqual(comp.astar_result.path_cost, 38)
        self.assertEqual(comp.ucs_result.path_length, 38)
        self.assertEqual(comp.astar_result.path_length, 38)
        self.assertEqual(len(comp.ucs_result.path), 39)
        self.assertEqual(len(comp.astar_result.path), 39)

        # 4. First and last path coordinates
        self.assertEqual(comp.ucs_result.path[0], (0, 0))
        self.assertEqual(comp.ucs_result.path[-1], (19, 19))
        self.assertEqual(comp.astar_result.path[0], (0, 0))
        self.assertEqual(comp.astar_result.path[-1], (19, 19))

        # 5. A* vast outperformance assertion
        self.assertEqual(comp.nodes_saved, 335)
        self.assertAlmostEqual(comp.expansion_reduction_percent, 88.15789, places=3)
        self.assertGreater(comp.expansion_ratio, 8.0)
        self.assertLess(comp.astar_result.nodes_expanded, comp.ucs_result.nodes_expanded / 5)

    def test_path_continuity_and_obstacle_avoidance(self) -> None:
        """Verify that returned paths are continuous and avoid all obstacles."""
        comp = compare_ucs_and_astar()
        grid = comp.grid

        for res in [comp.ucs_result, comp.astar_result]:
            path = res.path
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i + 1]
                # Each step must be exactly 1 orthogonal unit
                step_dist = abs(r1 - r2) + abs(c1 - c2)
                self.assertEqual(step_dist, 1)
                # No cell along path can be an obstacle
                self.assertNotIn((r1, c1), grid.obstacles)
                self.assertNotIn((r2, c2), grid.obstacles)

    def test_custom_obstacles_comparison(self) -> None:
        """Verify comparison engine on custom obstacle list."""
        custom_walls = [(1, 0), (1, 1), (1, 2)]
        comp = compare_ucs_and_astar(obstacles=custom_walls)
        self.assertTrue(comp.ucs_result.success)
        self.assertTrue(comp.astar_result.success)
        self.assertEqual(comp.ucs_result.path_cost, comp.astar_result.path_cost)
        self.assertLessEqual(comp.astar_result.nodes_expanded, comp.ucs_result.nodes_expanded)

    def test_comparison_result_formatting_and_dict(self) -> None:
        """Verify comparison table, ASCII side-by-side rendering, and JSON dictionary."""
        comp = compare_ucs_and_astar()

        # Comparison table
        tbl = comp.format_comparison_table()
        self.assertIn("UCS vs A* (MANHATTAN) PERFORMANCE COMPARISON", tbl)
        self.assertIn("States Evaluated (Expanded)", tbl)
        self.assertIn("380", tbl)
        self.assertIn("45", tbl)
        self.assertIn("88.16%", tbl)

        # ASCII maps
        ascii_map = comp.format_ascii_side_by_side()
        self.assertIn("UCS Map", ascii_map)
        self.assertIn("A* Map", ascii_map)
        self.assertIn("S", ascii_map)
        self.assertIn("G", ascii_map)
        self.assertIn("*", ascii_map)

        # JSON dictionary
        d = comp.to_dict()
        self.assertIn("grid", d)
        self.assertIn("ucs", d)
        self.assertIn("astar", d)
        self.assertIn("comparison", d)
        self.assertEqual(d["comparison"]["nodes_saved"], 335)
        self.assertEqual(d["comparison"]["sample_output"], "UCS expanded: 380 nodes. A* expanded: 45 nodes.")

        # Ensure JSON dumps cleanly
        json_str = json.dumps(d)
        self.assertTrue(len(json_str) > 0)


class TestMatplotlibVisualizationAndGuides(unittest.TestCase):
    """Test suite validating Matplotlib visualization creation and viva guides."""

    def test_matplotlib_visualization_generates_file(self) -> None:
        """Verify side-by-side Matplotlib visualization creates valid PNG artifact."""
        comp = compare_ucs_and_astar()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_comparison.png"
            fig = visualize_ucs_vs_astar(comparison=comp, save_path=out_file, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_theoretical_analysis_and_viva_reference(self) -> None:
        """Verify that theoretical analysis and viva voce guides are populated."""
        theory = get_theoretical_analysis()
        self.assertIn("Evaluation Function Decomposition", theory)
        self.assertIn("Heuristic Pull & Directionality", theory)
        self.assertIn("Admissibility & Consistency Guarantees", theory)

        viva = get_viva_reference()
        self.assertIn("special case of A*", viva)
        self.assertIn("directionality", viva)
        self.assertIn("expanded (popped)", viva)


if __name__ == "__main__":
    unittest.main()
