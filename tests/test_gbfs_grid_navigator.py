"""Comprehensive Unit Test Suite for Question 27: Greedy Best-First Search Grid Navigator.

Validates:
- Exact syllabus sample input & output ('Expanded: 25 nodes, Path Length: 22 (Suboptimal)').
- Dynamic heuristic calculation based on goal coordinates (Manhattan, Euclidean, Chebyshev).
- Verification that Greedy BFS explores fewer nodes than BFS (25 < 87).
- Verification that Greedy BFS finds a suboptimal path compared to BFS (22 > 18).
- A* search baseline finding optimal path (18) with informed pruning.
- Optional Challenge: Output the grid with the path marked by '*'.
- Edge cases: start == goal, adjacent start and goal, unsolvable maze, out-of-bounds, obstacle collision.
- Concave trap / local minimum benchmark.
- Open grid (zero obstacles) where path length equals Manhattan distance (18).
- Object-Oriented GBFSGridNavigator class interface.
- Serialization and dictionary export.
- Viva Voce Guide completeness (Greedy vs A*, obstacle influence).
"""

from __future__ import annotations

import json
import unittest

from src.gbfs_grid_navigator import (
    CONCAVE_TRAP_10X10_MAZE,
    DEFAULT_10X10_MAZE,
    DEFAULT_GOAL_COORD,
    DEFAULT_START_COORD,
    OPEN_10X10_MAZE,
    UNSOLVABLE_10X10_MAZE,
    GBFSGridComparisonResult,
    GBFSGridNavigator,
    GridSearchResult,
    GridStepRecord,
    a_star_search_grid,
    breadth_first_search_grid,
    chebyshev_distance,
    compare_gbfs_vs_bfs,
    euclidean_distance,
    get_gbfs_grid_viva_reference,
    greedy_best_first_search_grid,
    manhattan_distance,
    render_grid_with_path,
    render_side_by_side_grids,
)


class TestGBFSGridCore(unittest.TestCase):
    """Core algorithmic tests for Greedy Best-First Search on a 2D Grid."""

    def test_syllabus_sample_input_and_output(self) -> None:
        """Verify exact sample from Question 27 prompt: Expanded: 25 nodes, Path Length: 22 (Suboptimal)."""
        comp = compare_gbfs_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        res = comp.gbfs_result

        self.assertTrue(res.found)
        self.assertEqual(res.nodes_expanded, 25)
        self.assertEqual(res.path_length, 22)
        self.assertTrue(comp.is_gbfs_suboptimal)
        self.assertEqual(
            res.format_sample_output(),
            "Expanded: 25 nodes, Path Length: 22 (Suboptimal)",
        )

    def test_path_length_and_coordinates(self) -> None:
        """Verify path coordinates structure and length."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        self.assertTrue(res.found)
        # Path length = 22 means 23 coordinates from start to goal
        self.assertEqual(len(res.path), 23)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (9, 9))

        # Check consecutive steps are orthogonal adjacent neighbors
        for (r1, c1), (r2, c2) in zip(res.path[:-1], res.path[1:]):
            step_dist = abs(r1 - r2) + abs(c1 - c2)
            self.assertEqual(step_dist, 1, f"Step from ({r1},{c1}) to ({r2},{c2}) is not orthogonal adjacent!")
            self.assertEqual(DEFAULT_10X10_MAZE[r2][c2], 0, "Path stepped onto an obstacle!")

    def test_dynamic_heuristic_calculation(self) -> None:
        """Verify dynamic calculation of Manhattan, Euclidean, and Chebyshev heuristics."""
        goal1 = (9, 9)
        coord1 = (0, 0)
        self.assertEqual(manhattan_distance(coord1, goal1), 18)
        self.assertAlmostEqual(euclidean_distance(coord1, goal1), 12.727922, places=4)
        self.assertEqual(chebyshev_distance(coord1, goal1), 9)

        # Dynamic recalculation with an arbitrary goal
        goal2 = (3, 5)
        coord2 = (1, 2)
        self.assertEqual(manhattan_distance(coord2, goal2), abs(1 - 3) + abs(2 - 5))  # 2 + 3 = 5
        self.assertAlmostEqual(euclidean_distance(coord2, goal2), (2**2 + 3**2) ** 0.5, places=4)
        self.assertEqual(chebyshev_distance(coord2, goal2), 3)

    def test_fewer_nodes_than_bfs(self) -> None:
        """Demonstrate that Greedy search explores significantly fewer nodes than BFS."""
        comp = compare_gbfs_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        # GBFS: 25 nodes vs BFS: 87 nodes
        self.assertTrue(comp.is_gbfs_strictly_fewer_nodes)
        self.assertEqual(comp.gbfs_result.nodes_expanded, 25)
        self.assertEqual(comp.bfs_result.nodes_expanded, 87)
        self.assertEqual(comp.expansion_difference, 62)
        self.assertGreater(comp.expansion_reduction_pct, 70.0)

    def test_suboptimal_path_than_bfs(self) -> None:
        """Demonstrate that Greedy search might return a suboptimal path compared to BFS."""
        comp = compare_gbfs_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        # GBFS: 22 steps vs BFS: 18 steps
        self.assertTrue(comp.is_gbfs_suboptimal)
        self.assertEqual(comp.gbfs_result.path_length, 22)
        self.assertEqual(comp.bfs_result.path_length, 18)
        self.assertEqual(comp.path_length_difference, 4)
        self.assertAlmostEqual(comp.suboptimality_ratio, 22 / 18, places=2)

    def test_astar_finds_optimal_path(self) -> None:
        """Verify A* search finds the optimal 18-step path using f(n) = g(n) + h(n)."""
        astar_res = a_star_search_grid(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        self.assertTrue(astar_res.found)
        self.assertEqual(astar_res.path_length, 18)


class TestOptionalGridRenderingChallenge(unittest.TestCase):
    """Test Optional Challenge: Output the grid with the path marked by '*'."""

    def test_render_grid_with_path_default(self) -> None:
        """Verify rendering maze with path marked by '*'."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        grid_str = render_grid_with_path(
            DEFAULT_10X10_MAZE,
            res.path,
            start=DEFAULT_START_COORD,
            goal=DEFAULT_GOAL_COORD,
            show_coordinates=False,
        )
        lines = grid_str.strip().split("\n")
        self.assertEqual(len(lines), 10)
        for line in lines:
            symbols = line.split()
            self.assertEqual(len(symbols), 10)

        # Check that '*' appears in the rendered grid
        self.assertIn("*", grid_str)
        # Check that Start 'S' and Goal 'G' appear
        self.assertIn("S", grid_str)
        self.assertIn("G", grid_str)
        # Check that obstacles '#' appear
        self.assertIn("#", grid_str)

    def test_render_grid_endpoints_preserved(self) -> None:
        """Verify Start 'S' at (0,0) and Goal 'G' at (9,9)."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        grid_str = render_grid_with_path(
            DEFAULT_10X10_MAZE,
            res.path,
            start=DEFAULT_START_COORD,
            goal=DEFAULT_GOAL_COORD,
            preserve_endpoints=True,
            show_coordinates=False,
        )
        lines = grid_str.strip().split("\n")
        first_row_symbols = lines[0].split()
        last_row_symbols = lines[9].split()
        self.assertEqual(first_row_symbols[0], "S")
        self.assertEqual(last_row_symbols[9], "G")

    def test_side_by_side_grid_rendering(self) -> None:
        """Verify side-by-side comparative grid formatting."""
        comp = compare_gbfs_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_START_COORD, DEFAULT_GOAL_COORD)
        side_by_side = render_side_by_side_grids(DEFAULT_10X10_MAZE, comp.gbfs_result, comp.bfs_result)
        self.assertIn("Greedy Best-First Search", side_by_side)
        self.assertIn("Breadth-First Search", side_by_side)
        self.assertIn("*", side_by_side)
        self.assertIn("o", side_by_side)


class TestEdgeCasesAndValidation(unittest.TestCase):
    """Test boundary conditions, invalid inputs, and corner cases."""

    def test_start_equals_goal(self) -> None:
        """Verify immediate termination when start == goal."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (3, 3), (3, 3))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 0)
        self.assertEqual(res.path, [(3, 3)])
        self.assertEqual(res.nodes_expanded, 1)

    def test_start_adjacent_to_goal(self) -> None:
        """Verify 1-step navigation when start is adjacent to goal."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (0, 0), (0, 1))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 1)
        self.assertEqual(res.path, [(0, 0), (0, 1)])

    def test_unsolvable_maze(self) -> None:
        """Verify handling of severed maze where goal is unreachable."""
        res = greedy_best_first_search_grid(UNSOLVABLE_10X10_MAZE, (0, 0), (9, 9))
        self.assertFalse(res.found)
        self.assertEqual(res.path_length, -1)
        self.assertEqual(res.path, [])
        self.assertIn("Unreachable", res.format_sample_output())

    def test_start_on_obstacle_raises_error(self) -> None:
        """Verify ValueError when start coordinate is on a wall (1)."""
        with self.assertRaises(ValueError):
            greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (1, 0), (9, 9))

    def test_goal_on_obstacle_raises_error(self) -> None:
        """Verify ValueError when goal coordinate is on a wall (1)."""
        with self.assertRaises(ValueError):
            greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (0, 0), (1, 0))

    def test_out_of_bounds_coordinates_raise_error(self) -> None:
        """Verify ValueError when coordinates exceed grid bounds."""
        with self.assertRaises(ValueError):
            greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (-1, 0), (9, 9))
        with self.assertRaises(ValueError):
            greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (0, 0), (10, 10))

    def test_empty_maze_raises_error(self) -> None:
        """Verify ValueError on empty maze matrix."""
        with self.assertRaises(ValueError):
            greedy_best_first_search_grid([], (0, 0), (0, 0))


class TestBenchmarksAndLocalMinima(unittest.TestCase):
    """Test performance on Open grid and Concave Trap benchmark."""

    def test_open_grid_navigation(self) -> None:
        """On an open grid with zero obstacles, path length equals Manhattan distance (18)."""
        res = greedy_best_first_search_grid(OPEN_10X10_MAZE, (0, 0), (9, 9))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 18)
        # Optimal path with only 19 node expansions (straight to goal)
        self.assertEqual(res.nodes_expanded, 19)

    def test_concave_trap_maze_backtracking(self) -> None:
        """In a concave U-trap, Greedy search enters the trap and must backtrack out."""
        res = greedy_best_first_search_grid(CONCAVE_TRAP_10X10_MAZE, (0, 0), (9, 9), record_steps=True)
        self.assertTrue(res.found)
        self.assertGreater(len(res.steps), 0)


class TestOOPInterfaceAndSerialization(unittest.TestCase):
    """Test object-oriented GBFSGridNavigator class and JSON export."""

    def test_oop_navigator_methods(self) -> None:
        """Verify GBFSGridNavigator methods: solve_greedy, solve_bfs, solve_a_star, compare_with_bfs."""
        nav = GBFSGridNavigator()
        self.assertEqual(nav.rows, 10)
        self.assertEqual(nav.cols, 10)

        # Dynamic heuristic test
        h_val = nav.compute_heuristic((0, 0))
        self.assertEqual(h_val, 18)

        gbfs_res = nav.solve_greedy()
        self.assertEqual(gbfs_res.nodes_expanded, 25)
        self.assertEqual(gbfs_res.path_length, 22)

        bfs_res = nav.solve_bfs()
        self.assertEqual(bfs_res.path_length, 18)

        astar_res = nav.solve_a_star()
        self.assertEqual(astar_res.path_length, 18)

        comp = nav.compare_with_bfs()
        self.assertTrue(comp.is_gbfs_suboptimal)
        self.assertTrue(comp.is_gbfs_strictly_fewer_nodes)

        rendered = nav.render_path_grid()
        self.assertIn("*", rendered)

    def test_result_to_dict_and_json(self) -> None:
        """Verify serialization to dictionary and valid JSON string."""
        res = greedy_best_first_search_grid(DEFAULT_10X10_MAZE, (0, 0), (9, 9))
        res_dict = res.to_dict()
        self.assertEqual(res_dict["algorithm"], "Greedy Best-First Search")
        self.assertEqual(res_dict["path_length"], 22)
        self.assertEqual(res_dict["nodes_expanded"], 25)

        json_str = json.dumps(res_dict)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["nodes_expanded"], 25)


class TestVivaVoceGuide(unittest.TestCase):
    """Test completeness of the University Viva Voce guide data."""

    def test_viva_reference_data(self) -> None:
        """Verify viva guide covers Greedy vs A* and influence of obstacles."""
        viva = get_gbfs_grid_viva_reference()
        self.assertIn("title", viva)
        self.assertIn("core_topics", viva)
        self.assertIn("qa_pairs", viva)

        questions = [qa["question"] for qa in viva["qa_pairs"]]
        # Must cover Greedy vs A*
        self.assertTrue(any("Greedy" in q and "A*" in q for q in questions))
        # Must cover Influence of obstacles
        self.assertTrue(any("obstacles" in q.lower() for q in questions))


if __name__ == "__main__":
    unittest.main()
