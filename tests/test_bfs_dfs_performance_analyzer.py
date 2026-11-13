"""Unit tests for BFS vs DFS Performance Analyzer (Question 18).

Validates:
- BFS shortest-path optimality vs. DFS arbitrary pathfinding.
- Exact node expansion counter tracking across both algorithms.
- Preservation of identical neighbor ordering (Up, Down, Left, Right).
- Canonical 10x10 mazes (Default, DFS-favored, Corner, Open, Unsolvable).
- Edge cases (Start is Goal, wall collisions, out-of-bounds, invalid grids).
- Matplotlib dynamic expansion visualization saving.
- Viva Voce reference guide completeness.
"""

from pathlib import Path
import tempfile
import unittest

from src.bfs_dfs_performance_analyzer import (
    CORNER_10X10_MAZE,
    CORNER_GOAL_COORD,
    DEFAULT_10X10_MAZE,
    DEFAULT_DIRECTIONS,
    DEFAULT_GOAL_COORD,
    DEFAULT_START_COORD,
    DFS_FAVORED_10X10_MAZE,
    DFS_FAVORED_GOAL_COORD,
    OPEN_10X10_MAZE,
    SAMPLE_OUTPUT_TARGET_STRING,
    UNSOLVABLE_10X10_MAZE,
    calculate_manhattan_distance,
    compare_maze_solvers,
    get_successors,
    get_viva_reference,
    is_open_cell,
    is_valid_coordinate,
    parse_grid,
    plot_bfs_vs_dfs_expansion,
    solve_maze_bfs,
    solve_maze_dfs,
    validate_grid,
)


class TestBFSDmazePerformanceAnalyzer(unittest.TestCase):
    """Test suite for BFS vs DFS performance comparison on 10x10 mazes."""

    def test_default_10x10_maze_bfs_optimality(self) -> None:
        """Verify BFS finds the optimal 12-step path in exactly 45 node expansions."""
        res = solve_maze_bfs(DEFAULT_10X10_MAZE, start=DEFAULT_START_COORD, goal=DEFAULT_GOAL_COORD)
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 12)
        self.assertEqual(res.nodes_expanded, 45)
        self.assertEqual(len(res.path), 13)
        self.assertEqual(res.path[0], DEFAULT_START_COORD)
        self.assertEqual(res.path[-1], DEFAULT_GOAL_COORD)
        self.assertEqual(res.actions, ["D", "D", "D", "D", "D", "R", "R", "R", "R", "R", "R", "R"])

    def test_default_10x10_maze_dfs_arbitrary_suboptimality(self) -> None:
        """Verify DFS finds a suboptimal circuitous 22-step path (10 steps longer than BFS)."""
        res = solve_maze_dfs(DEFAULT_10X10_MAZE, start=DEFAULT_START_COORD, goal=DEFAULT_GOAL_COORD)
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 22)
        self.assertEqual(res.nodes_expanded, 40)
        self.assertEqual(len(res.path), 23)
        self.assertEqual(res.path[0], DEFAULT_START_COORD)
        self.assertEqual(res.path[-1], DEFAULT_GOAL_COORD)
        # DFS path is strictly longer than BFS optimal path
        self.assertGreater(res.path_length, 12)

    def test_compare_maze_solvers_default_metrics(self) -> None:
        """Verify comparative metrics: difference, ratio, and sample output formatting."""
        comp = compare_maze_solvers(DEFAULT_10X10_MAZE, start=DEFAULT_START_COORD, goal=DEFAULT_GOAL_COORD)
        self.assertTrue(comp.is_bfs_strictly_shorter)
        self.assertEqual(comp.path_length_difference, 10)  # 22 - 12
        self.assertEqual(comp.expanded_difference, 5)  # 45 - 40
        self.assertEqual(comp.bfs.path_length, 12)
        self.assertEqual(comp.dfs.path_length, 22)
        self.assertIn("BFS Nodes Expanded: 45, Path Length: 12", comp.sample_output_formatted)
        self.assertIn("DFS Nodes Expanded: 40, Path Length: 22", comp.sample_output_formatted)

    def test_dfs_favored_scenario_low_expansions(self) -> None:
        """Verify viva scenario where DFS plunges directly to the goal with only 14 expansions."""
        comp = compare_maze_solvers(
            DFS_FAVORED_10X10_MAZE, start=(0, 0), goal=DFS_FAVORED_GOAL_COORD
        )
        self.assertTrue(comp.bfs.found)
        self.assertTrue(comp.dfs.found)
        self.assertEqual(comp.bfs.nodes_expanded, 45)
        self.assertEqual(comp.bfs.path_length, 12)
        self.assertEqual(comp.dfs.nodes_expanded, 14)
        self.assertEqual(comp.dfs.path_length, 12)
        # In this scenario, DFS expands significantly fewer nodes than BFS (14 vs 45)
        self.assertLess(comp.dfs.nodes_expanded, comp.bfs.nodes_expanded)
        self.assertEqual(
            comp.sample_output_formatted,
            "BFS Nodes Expanded: 45, Path Length: 12 | DFS Nodes Expanded: 14, Path Length: 12",
        )

    def test_corner_to_corner_10x10_maze(self) -> None:
        """Verify both algorithms successfully solve the (0,0) to (9,9) corner-to-corner maze."""
        comp = compare_maze_solvers(CORNER_10X10_MAZE, start=(0, 0), goal=CORNER_GOAL_COORD)
        self.assertTrue(comp.bfs.found)
        self.assertTrue(comp.dfs.found)
        self.assertLessEqual(comp.bfs.path_length, comp.dfs.path_length)
        self.assertEqual(comp.bfs.start, (0, 0))
        self.assertEqual(comp.bfs.goal, (9, 9))

    def test_open_grid_manhattan_distance_equality(self) -> None:
        """On an open grid with zero obstacles, BFS path length equals Manhattan distance."""
        start = (0, 0)
        goal = (8, 6)
        manhattan = calculate_manhattan_distance(start, goal)
        res = solve_maze_bfs(OPEN_10X10_MAZE, start=start, goal=goal)
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, manhattan)
        self.assertEqual(res.path_length, 14)

    def test_unsolvable_maze_severed_by_wall_barrier(self) -> None:
        """Verify both BFS and DFS return found=False and steps=-1 when path is blocked."""
        comp = compare_maze_solvers(UNSOLVABLE_10X10_MAZE, start=(0, 0), goal=(9, 9))
        self.assertFalse(comp.bfs.found)
        self.assertFalse(comp.dfs.found)
        self.assertEqual(comp.bfs.path_length, -1)
        self.assertEqual(comp.dfs.path_length, -1)
        self.assertEqual(comp.bfs.path, [])
        self.assertEqual(comp.dfs.path, [])

    def test_start_is_goal_base_case(self) -> None:
        """When start equals goal, path length is 0 and nodes expanded is 1."""
        coord = (4, 4)  # Open cell in DEFAULT_10X10_MAZE
        res_bfs = solve_maze_bfs(DEFAULT_10X10_MAZE, start=coord, goal=coord)
        res_dfs = solve_maze_dfs(DEFAULT_10X10_MAZE, start=coord, goal=coord)
        self.assertTrue(res_bfs.found)
        self.assertTrue(res_dfs.found)
        self.assertEqual(res_bfs.path_length, 0)
        self.assertEqual(res_dfs.path_length, 0)
        self.assertEqual(res_bfs.nodes_expanded, 1)
        self.assertEqual(res_dfs.nodes_expanded, 1)
        self.assertEqual(res_bfs.path, [coord])
        self.assertEqual(res_dfs.path, [coord])

    def test_wall_collision_at_start_or_goal(self) -> None:
        """If start or goal is placed inside a wall (1), return found=False immediately."""
        wall_coord = (0, 1)  # cell is 1 in DEFAULT_10X10_MAZE
        res_start_wall = solve_maze_bfs(DEFAULT_10X10_MAZE, start=wall_coord, goal=(5, 7))
        res_goal_wall = solve_maze_dfs(DEFAULT_10X10_MAZE, start=(0, 0), goal=wall_coord)
        self.assertFalse(res_start_wall.found)
        self.assertFalse(res_goal_wall.found)
        self.assertEqual(res_start_wall.nodes_expanded, 0)
        self.assertEqual(res_goal_wall.nodes_expanded, 0)

    def test_out_of_bounds_coordinates_raise_value_error(self) -> None:
        """Out of bounds coordinates must raise ValueError."""
        with self.assertRaises(ValueError):
            solve_maze_bfs(DEFAULT_10X10_MAZE, start=(-1, 0), goal=(5, 7))
        with self.assertRaises(ValueError):
            solve_maze_dfs(DEFAULT_10X10_MAZE, start=(0, 0), goal=(10, 10))

    def test_invalid_grid_inputs(self) -> None:
        """Empty grids, non-rectangular grids, or invalid tokens must raise ValueError."""
        with self.assertRaises(ValueError):
            parse_grid([])
        with self.assertRaises(ValueError):
            parse_grid([[0, 1], [0]])
        with self.assertRaises(ValueError):
            parse_grid([[0, 2], [1, 0]])
        with self.assertRaises(ValueError):
            parse_grid(None)

    def test_string_and_json_grid_parsing(self) -> None:
        """Verify grid parsing from JSON strings and ASCII multiline strings."""
        json_grid = "[[0, 1], [0, 0]]"
        parsed = parse_grid(json_grid)
        self.assertEqual(parsed, [[0, 1], [0, 0]])

        ascii_grid = "0 1\n0 0"
        parsed_ascii = parse_grid(ascii_grid)
        self.assertEqual(parsed_ascii, [[0, 1], [0, 0]])

    def test_exact_neighbor_ordering_successors(self) -> None:
        """Verify successors are emitted in strict priority order: Up, Down, Left, Right."""
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        succs = get_successors((1, 1), grid, directions=DEFAULT_DIRECTIONS)
        actions = [act for act, _ in succs]
        coords = [coord for _, coord in succs]
        self.assertEqual(actions, ["U", "D", "L", "R"])
        self.assertEqual(coords, [(0, 1), (2, 1), (1, 0), (1, 2)])

    def test_ascii_path_rendering(self) -> None:
        """Verify ASCII rendering generates valid glyphs for start (S), goal (G), path (*), and walls (#)."""
        res = solve_maze_bfs(DEFAULT_10X10_MAZE, start=DEFAULT_START_COORD, goal=DEFAULT_GOAL_COORD)
        ascii_map = res.render_ascii_path()
        self.assertIn("S", ascii_map)
        self.assertIn("G", ascii_map)
        self.assertIn("*", ascii_map)
        self.assertIn("#", ascii_map)

    def test_summary_table_formatting(self) -> None:
        """Verify summary table contains expected metric labels and headings."""
        comp = compare_maze_solvers(DEFAULT_10X10_MAZE)
        table = comp.format_summary_table()
        self.assertIn("BFS (Breadth-First)", table)
        self.assertIn("DFS (Depth-First)", table)
        self.assertIn("Nodes Expanded (Popped)", table)
        self.assertIn("Path Length (Steps)", table)
        self.assertIn("Optimality Status", table)

    def test_viva_reference_structure(self) -> None:
        """Verify Viva Voce guide covers all required topics from the syllabus."""
        viva = get_viva_reference()
        self.assertIn("viva_topics", viva)
        topics = viva["viva_topics"]
        self.assertIn("algorithm_optimality", topics)
        self.assertIn("scenarios_where_dfs_outperforms_bfs", topics)
        self.assertIn("branching_factor_impact", topics)
        self.assertIn("complexity_matrix", viva)

    def test_matplotlib_plot_saving_headless(self) -> None:
        """Verify Optional Challenge: plot_bfs_vs_dfs_expansion saves image cleanly."""
        comp = compare_maze_solvers(DEFAULT_10X10_MAZE)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_expansion.png"
            fig = plot_bfs_vs_dfs_expansion(comp, save_path=out_path, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
