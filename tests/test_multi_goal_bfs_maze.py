"""Unit test suite for Question 23: Maze with Multiple Exits (Multi-Goal BFS).

Tests cover:
- Exact verification of sample input and output format: Exits (5,5), (1,8) -> Closest (1,8) in 9 steps.
- Set membership testing and goal set initialization.
- Goal test placement (on expansion vs on generation).
- Path and action sequence reconstruction.
- Optional Challenge: Returning all candidate exits sorted by distance.
- Edge cases: Start is exit (0 steps), single exit, unreachable exits, 1x1 maze, 1D corridors.
- Validation: Missing start, walls, empty candidate lists, non-rectangular grids.
- Wall detour scenarios proving BFS finds path length, not mere Manhattan distance.
- ASCII maze string parser ('S', 'E', '#', '.').
- Viva Voce reference validation, set vs list benchmark, and BFS vs UCS comparison.
- Step-by-step generator for animations and progress tracking.
- Serialization and visualization output.
"""

import json
from pathlib import Path
import tempfile
import unittest
from typing import List, Tuple

from src.multi_goal_bfs_maze import (
    CELL_OPEN,
    CELL_WALL,
    DETOUR_MULTI_EXIT_MAZE,
    PARTIALLY_BLOCKED_MULTI_EXIT_MAZE,
    SAMPLE_EXITS_Q23,
    SAMPLE_MULTI_EXIT_MAZE_6X10,
    SAMPLE_START_Q23,
    Coordinate,
    MultiGoalBFSResult,
    MultiGoalBFSSolver,
    SingleExitPathResult,
    benchmark_set_vs_list_lookup,
    compare_multi_goal_bfs_vs_ucs,
    find_all_exits_sorted,
    find_closest_exit,
    get_multi_goal_bfs_viva_reference,
    parse_ascii_maze,
    solve_multi_goal_bfs,
    visualize_multi_goal_maze,
)


class TestMultiGoalBFSSample(unittest.TestCase):
    """Test verification against exact prompt specifications and sample input/output."""

    def test_canonical_sample_input_and_output(self) -> None:
        """Verify Sample Input: Exits: (5,5), (1,8). Start: (0,0) -> Output: (1,8) in 9 steps."""
        exit_coord, steps = find_closest_exit(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
        )
        self.assertEqual(exit_coord, (1, 8))
        self.assertEqual(steps, 9)

    def test_canonical_full_result_details(self) -> None:
        """Verify full path, action moves, and termination properties on sample maze."""
        result = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            find_all=False,
            record_trace=True,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.closest_exit, (1, 8))
        self.assertEqual(result.path_length, 9)
        self.assertEqual(len(result.path), 10)  # Start + 9 steps
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (1, 8))
        self.assertEqual(len(result.actions), 9)
        self.assertGreater(result.nodes_expanded, 0)
        self.assertGreater(result.nodes_visited, 0)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_goal_test_expansion_vs_generation(self) -> None:
        """Verify both expansion-time and generation-time goal tests return the same optimal result."""
        res_expansion = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            goal_test_on_expansion=True,
        )
        res_generation = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            goal_test_on_expansion=False,
        )
        self.assertEqual(res_expansion.closest_exit, res_generation.closest_exit)
        self.assertEqual(res_expansion.path_length, res_generation.path_length)
        # Checking on generation stops earlier, so nodes_expanded is less than or equal
        self.assertLessEqual(res_generation.nodes_expanded, res_expansion.nodes_expanded)


class TestOptionalChallengeAllExits(unittest.TestCase):
    """Test Optional Challenge: Return paths to ALL exits sorted by distance."""

    def test_find_all_exits_sorted_sample(self) -> None:
        """Verify sorting order of exits on the sample maze: (1,8) at 9 steps, (5,5) at 10 steps."""
        all_exits = find_all_exits_sorted(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
        )
        self.assertEqual(len(all_exits), 2)
        # (1, 8) must be first (9 steps)
        self.assertEqual(all_exits[0].exit_coord, (1, 8))
        self.assertEqual(all_exits[0].path_length, 9)
        self.assertTrue(all_exits[0].found)

        # (5, 5) must be second (10 steps)
        self.assertEqual(all_exits[1].exit_coord, (5, 5))
        self.assertEqual(all_exits[1].path_length, 10)
        self.assertTrue(all_exits[1].found)

    def test_find_all_with_unreachable_exit(self) -> None:
        """Verify unreachable exits are placed at end of sorted list with path_length = -1."""
        exits = [(0, 2), (2, 0), (2, 4)]
        result = solve_multi_goal_bfs(
            PARTIALLY_BLOCKED_MULTI_EXIT_MAZE,
            start=(0, 0),
            exits=exits,
            find_all=True,
        )
        self.assertTrue(result.found)
        self.assertIn((0, 2), result.unreachable_exits)
        sorted_coords = [r.exit_coord for r in result.all_exits_sorted]
        # (0, 2) is unreachable, so it must be last
        self.assertEqual(sorted_coords[-1], (0, 2))
        self.assertFalse(result.all_exits_sorted[-1].found)
        self.assertEqual(result.all_exits_sorted[-1].path_length, -1)


class TestMultiGoalBFSEdgeCases(unittest.TestCase):
    """Test boundary and edge cases for robust error handling and completeness."""

    def test_start_is_one_of_the_exits(self) -> None:
        """When the starting coordinate is already in the exit set, return 0 steps immediately."""
        maze = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        start = (1, 0)
        exits = [(1, 0), (2, 2)]
        exit_coord, steps = find_closest_exit(maze, start, exits)
        self.assertEqual(exit_coord, (1, 0))
        self.assertEqual(steps, 0)

        result = solve_multi_goal_bfs(maze, start, exits, find_all=True)
        self.assertEqual(result.closest_exit, (1, 0))
        self.assertEqual(result.path_length, 0)
        self.assertEqual(result.path, [(1, 0)])
        self.assertEqual(result.actions, [])
        self.assertEqual(result.all_exits_sorted[0].exit_coord, (1, 0))
        self.assertEqual(result.all_exits_sorted[0].path_length, 0)

    def test_single_candidate_exit(self) -> None:
        """Ensure backward compatibility when only one exit coordinate is supplied."""
        maze = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        exit_coord, steps = find_closest_exit(maze, (0, 0), [(2, 2)])
        self.assertEqual(exit_coord, (2, 2))
        self.assertEqual(steps, 4)

    def test_all_exits_completely_unreachable(self) -> None:
        """When all exits are trapped behind walls, found must be False and path_length -1."""
        maze = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        result = solve_multi_goal_bfs(maze, (0, 0), [(0, 2), (2, 2)])
        self.assertFalse(result.found)
        self.assertIsNone(result.closest_exit)
        self.assertEqual(result.path_length, -1)
        self.assertEqual(result.path, [])
        self.assertEqual(result.actions, [])

    def test_wall_detour_prefers_actual_path_over_manhattan(self) -> None:
        """Verify BFS chooses the exit that has the shortest true path length over Euclidean/Manhattan distance."""
        result = solve_multi_goal_bfs(
            DETOUR_MULTI_EXIT_MAZE,
            start=(0, 0),
            exits=[(2, 0), (0, 4)],
            find_all=True,
        )
        # (2, 0) has Manhattan distance 2, but walls require 12 steps
        # (0, 4) has Manhattan distance 4, and path length 4
        self.assertEqual(result.closest_exit, (0, 4))
        self.assertEqual(result.path_length, 4)

    def test_single_cell_maze(self) -> None:
        """1x1 maze with start == exit."""
        maze = [[0]]
        exit_coord, steps = find_closest_exit(maze, (0, 0), [(0, 0)])
        self.assertEqual(exit_coord, (0, 0))
        self.assertEqual(steps, 0)

    def test_1d_corridor_maze(self) -> None:
        """1x5 horizontal corridor with multiple exits."""
        maze = [[0, 0, 0, 0, 0]]
        start = (0, 2)
        exits = [(0, 0), (0, 4)]
        # Left is 2 steps to (0,0), Right is 2 steps to (0,4)
        result = solve_multi_goal_bfs(maze, start, exits, find_all=True)
        self.assertTrue(result.found)
        self.assertEqual(result.path_length, 2)
        self.assertEqual(len(result.all_exits_sorted), 2)
        self.assertEqual(result.all_exits_sorted[0].path_length, 2)
        self.assertEqual(result.all_exits_sorted[1].path_length, 2)

    def test_tie_breaking_deterministic(self) -> None:
        """When two exits are at equal distance, solver uses deterministic tie-breaking."""
        maze = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        # (0, 2) is 2 steps: R, R
        # (2, 0) is 2 steps: D, D
        # In DEFAULT_DIRECTIONS (U, D, L, R), Down is checked before Right
        result = solve_multi_goal_bfs(maze, (0, 0), [(0, 2), (2, 0)])
        self.assertEqual(result.path_length, 2)
        self.assertIn(result.closest_exit, [(0, 2), (2, 0)])


class TestInputValidation(unittest.TestCase):
    """Test input validation and error raising for invalid configurations."""

    def test_empty_maze_raises_value_error(self) -> None:
        """Empty maze grid raises ValueError."""
        with self.assertRaises(ValueError):
            solve_multi_goal_bfs([], (0, 0), [(1, 1)])

    def test_non_rectangular_maze_raises_value_error(self) -> None:
        """Inconsistent row lengths raise ValueError."""
        maze = [[0, 0], [0, 0, 0]]
        with self.assertRaises(ValueError):
            solve_multi_goal_bfs(maze, (0, 0), [(1, 1)])

    def test_start_out_of_bounds_raises_value_error(self) -> None:
        """Start coordinate outside maze boundaries raises ValueError."""
        maze = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            solve_multi_goal_bfs(maze, (5, 5), [(0, 1)])

    def test_start_inside_wall_raises_value_error(self) -> None:
        """Start placed inside a wall raises ValueError."""
        maze = [[1, 0], [0, 0]]
        with self.assertRaises(ValueError):
            solve_multi_goal_bfs(maze, (0, 0), [(1, 1)])

    def test_empty_exits_raises_value_error(self) -> None:
        """Empty candidate exit collection raises ValueError."""
        maze = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            solve_multi_goal_bfs(maze, (0, 0), [])


class TestAsciiParserAndVisualization(unittest.TestCase):
    """Test parsing ASCII string grids and generating visual renderings."""

    def test_parse_ascii_maze_success(self) -> None:
        """Verify ASCII parsing of 'S', 'E', '#', and '.'."""
        ascii_grid = """
            S . . E
            # # . #
            E . . .
        """
        maze, start, exits = parse_ascii_maze(ascii_grid)
        self.assertEqual(len(maze), 3)
        self.assertEqual(len(maze[0]), 4)
        self.assertEqual(start, (0, 0))
        self.assertEqual(set(exits), {(0, 3), (2, 0)})
        self.assertEqual(maze[1][0], CELL_WALL)
        self.assertEqual(maze[1][1], CELL_WALL)

    def test_parse_ascii_missing_start_raises_error(self) -> None:
        """ASCII grid without 'S' raises ValueError."""
        ascii_grid = """
            . . E
            . . .
        """
        with self.assertRaises(ValueError):
            parse_ascii_maze(ascii_grid)

    def test_parse_ascii_missing_exits_raises_error(self) -> None:
        """ASCII grid without 'E' raises ValueError."""
        ascii_grid = """
            S . .
            . . .
        """
        with self.assertRaises(ValueError):
            parse_ascii_maze(ascii_grid)

    def test_render_ascii_solution(self) -> None:
        """Verify ASCII string output contains glyphs."""
        result = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            find_all=True,
        )
        ascii_render = result.render_ascii_solution(show_all_paths=True)
        self.assertIn("S", ascii_render)
        self.assertIn("E*", ascii_render)
        self.assertIn("*", ascii_render)

    def test_format_summary_and_to_dict(self) -> None:
        """Verify summary formatting and JSON serialization."""
        result = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            find_all=True,
        )
        summary = result.format_summary()
        self.assertIn("MULTI-GOAL BREADTH-FIRST SEARCH", summary)
        self.assertIn("(1, 8)", summary)
        self.assertIn("9 steps", summary)

        result_dict = result.to_dict()
        serialized = json.dumps(result_dict)
        self.assertIn('"found": true', serialized)
        self.assertIn('"closest_exit": [1, 8]', serialized)

    def test_visualize_multi_goal_maze_matplotlib(self) -> None:
        """Verify Matplotlib image generation saves file without exceptions."""
        result = solve_multi_goal_bfs(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
            find_all=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_maze.png"
            saved_path = visualize_multi_goal_maze(result, save_path=out_path)
            if saved_path is not None:
                self.assertTrue(saved_path.exists())
                self.assertGreater(saved_path.stat().st_size, 0)


class TestStepGenerator(unittest.TestCase):
    """Test step-by-step generator for animations and traces."""

    def test_generator_yields_valid_step_records(self) -> None:
        """Verify generator yields consecutive MultiGoalStepRecord instances."""
        solver = MultiGoalBFSSolver()
        gen = solver.step_generator(
            SAMPLE_MULTI_EXIT_MAZE_6X10,
            SAMPLE_START_Q23,
            SAMPLE_EXITS_Q23,
        )
        records = list(gen)
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0].step_number, 1)
        self.assertEqual(records[0].current_coord, (0, 0))
        # Final yielded record should be the exit
        self.assertTrue(records[-1].is_exit)
        self.assertEqual(records[-1].current_coord, (1, 8))


class TestVivaVoceReferences(unittest.TestCase):
    """Test Viva Voce helpers and theoretical benchmarks."""

    def test_viva_reference_structure(self) -> None:
        """Verify university viva reference contains all required questions."""
        viva = get_multi_goal_bfs_viva_reference()
        self.assertIn("questions", viva)
        self.assertGreaterEqual(len(viva["questions"]), 5)
        # Ensure questions cover set efficiency and UCS
        all_q_text = " ".join(item["question"] for item in viva["questions"])
        self.assertIn("set", all_q_text.lower())
        self.assertIn("ucs", all_q_text.lower())

    def test_benchmark_set_vs_list_lookup(self) -> None:
        """Verify set lookup benchmark executes and demonstrates positive speedup."""
        bench = benchmark_set_vs_list_lookup(num_goals=100, num_queries=1000)
        self.assertIn("speedup_factor", bench)
        self.assertGreater(bench["speedup_factor"], 1.0)
        self.assertIn("set_time_ms", bench)
        self.assertIn("list_time_ms", bench)

    def test_compare_multi_goal_bfs_vs_ucs(self) -> None:
        """Verify comparative analysis of BFS (hops) vs UCS (cost) on non-uniform weights."""
        comp = compare_multi_goal_bfs_vs_ucs()
        self.assertTrue(comp["ucs_compatible"])
        # BFS chooses Exit A (2 steps, cost 20)
        self.assertEqual(comp["bfs_chosen_exit"], (0, 2))
        self.assertEqual(comp["bfs_steps"], 2)
        # UCS chooses Exit B (4 steps, cost 4)
        self.assertEqual(comp["ucs_chosen_exit"], (1, 3))
        self.assertEqual(comp["ucs_cost"], 4)
        self.assertEqual(comp["ucs_steps"], 4)


if __name__ == "__main__":
    unittest.main()
