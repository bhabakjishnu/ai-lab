"""Comprehensive Unit Test Suite for Question 29: A* Pathfinding in 2D Maze.

Validates:
- 2D grid parsing from strings, 2D char lists, and 2D int matrices.
- f(n) = g(n) + manhattan(current, goal) evaluation.
- Tracking of expanded nodes, expansion order, and step records.
- Rendering of the maze with 'X' for obstacles, '.' for open, '*' for path, and '#' for explored.
- Exact syllabus sample input & output on 5x5 grid.
- Verification of A* optimality and reduction in explored set size relative to BFS.
- Optional Challenge: Live terminal animation generator and frame stepping.
- Comprehensive edge cases (start == goal, single cell, open grid, concave traps, serpentine corridors).
- Error handling (empty grids, non-rectangular matrices, out of bounds, start/goal on obstacles).
- Viva Voce theoretical coverage and dictionary serialization.
"""

from __future__ import annotations

import io
import unittest

from src.a_star_maze_solver import (
    CONCAVE_TRAP_MAZE,
    DEFAULT_5X5_GOAL,
    DEFAULT_5X5_MAZE,
    DEFAULT_5X5_START,
    DEFAULT_10X10_GOAL,
    DEFAULT_10X10_MAZE,
    DEFAULT_10X10_START,
    GLYPH_EXPLORED,
    GLYPH_OBSTACLE,
    GLYPH_OPEN,
    GLYPH_PATH,
    OPEN_5X5_MAZE,
    SERPENTINE_5X5_MAZE,
    SINGLE_CELL_MAZE,
    UNSOLVABLE_5X5_MAZE,
    VIVA_VOCE_GUIDE,
    AStarMazeComparisonResult,
    AStarMazeResult,
    AStarMazeStepRecord,
    a_star_maze_solver,
    animate_a_star_search,
    breadth_first_search_maze,
    chebyshev_distance,
    compare_astar_vs_bfs,
    euclidean_distance,
    generate_expansion_frames,
    greedy_best_first_search_maze,
    manhattan_distance,
    parse_grid,
    render_side_by_side_comparison,
    render_solved_maze,
)


class TestAStarMazeCore(unittest.TestCase):
    """Core algorithmic tests for A* Pathfinding on 2D Grid Mazes."""

    def test_canonical_5x5_sample_execution(self) -> None:
        """Verify A* search on the canonical 5x5 grid from syllabus sample input."""
        res = a_star_maze_solver(
            maze=DEFAULT_5X5_MAZE,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
            tie_breaker="fifo",
            record_steps=True,
        )

        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 8)
        self.assertEqual(len(res.path), 9)
        self.assertEqual(res.total_cost, 8)
        self.assertEqual(res.start, (0, 0))
        self.assertEqual(res.goal, (4, 4))
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (4, 4))
        self.assertEqual(res.nodes_expanded, 16)
        self.assertEqual(len(res.explored_set), 16)
        self.assertTrue(res.is_optimal)

        # Check step records
        self.assertEqual(len(res.steps), 16)
        first_step = res.steps[0]
        self.assertEqual(first_step.step_number, 1)
        self.assertEqual(first_step.current_coord, (0, 0))
        self.assertEqual(first_step.g_cost, 0)
        self.assertEqual(first_step.h_cost, 8)
        self.assertEqual(first_step.f_cost, 8)

        last_step = res.steps[-1]
        self.assertEqual(last_step.current_coord, (4, 4))
        self.assertEqual(last_step.action, "GOAL_REACHED")
        self.assertEqual(last_step.g_cost, 8)
        self.assertEqual(last_step.h_cost, 0)
        self.assertEqual(last_step.f_cost, 8)

    def test_canonical_5x5_ascii_rendering(self) -> None:
        """Verify exact ASCII visualization rendering matching task requirements.

        Render symbols:
        'X' for obstacles, '.' for open, '*' for path, and '#' for explored.
        """
        res = a_star_maze_solver(
            maze=DEFAULT_5X5_MAZE,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
            tie_breaker="fifo",
        )

        rendered = render_solved_maze(
            maze=DEFAULT_5X5_MAZE,
            path=res.path,
            explored=res.explored_set,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
            preserve_endpoints=False,
            show_coordinates=False,
        )

        expected_rendered = (
            "* # X . .\n"
            "* # X . .\n"
            "* # X . .\n"
            "* # # # #\n"
            "* * * * *"
        )
        self.assertEqual(rendered, expected_rendered)

    def test_render_with_preserved_endpoints(self) -> None:
        """Verify rendering with Start ('S') and Goal ('G') preserved."""
        res = a_star_maze_solver(
            maze=DEFAULT_5X5_MAZE,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
            tie_breaker="fifo",
        )

        rendered = render_solved_maze(
            maze=DEFAULT_5X5_MAZE,
            path=res.path,
            explored=res.explored_set,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
            preserve_endpoints=True,
            show_coordinates=False,
        )

        expected_rendered = (
            "S # X . .\n"
            "* # X . .\n"
            "* # X . .\n"
            "* # # # #\n"
            "* * * * G"
        )
        self.assertEqual(rendered, expected_rendered)

    def test_render_with_coordinates(self) -> None:
        """Verify rendering when show_coordinates=True includes axes."""
        res = a_star_maze_solver(DEFAULT_5X5_MAZE)
        rendered = render_solved_maze(
            DEFAULT_5X5_MAZE,
            res.path,
            explored=res.explored_set,
            show_coordinates=True,
        )
        lines = rendered.splitlines()
        self.assertIn("0 1 2 3 4", lines[0])
        self.assertIn("+----------", lines[1])
        self.assertTrue(lines[2].startswith("0 |"))

    def test_manhattan_distance_properties(self) -> None:
        """Verify mathematical properties of Manhattan distance (admissibility & consistency)."""
        goal = (4, 4)
        # Test basic calculations
        self.assertEqual(manhattan_distance((0, 0), goal), 8)
        self.assertEqual(manhattan_distance((4, 4), goal), 0)
        self.assertEqual(manhattan_distance((2, 3), goal), 3)

        # Monotonicity / consistency across all 4 orthogonal moves: |h(u) - h(v)| <= 1
        for r in range(5):
            for c in range(5):
                u = (r, c)
                h_u = manhattan_distance(u, goal)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5:
                        v = (nr, nc)
                        h_v = manhattan_distance(v, goal)
                        self.assertLessEqual(abs(h_u - h_v), 1)

    def test_alternative_heuristics(self) -> None:
        """Verify Euclidean and Chebyshev distance calculators."""
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertEqual(chebyshev_distance((0, 0), (3, 4)), 4)


class TestAStarGridParsing(unittest.TestCase):
    """Test suite for the 2D grid parser across formats and error boundaries."""

    def test_parse_multiline_string_spaced(self) -> None:
        """Verify parsing space-delimited multiline string."""
        grid_str = """
        . . X . .
        . . X . .
        . . X . .
        . . . . .
        . . . . .
        """
        maze, start, goal = parse_grid(grid_str)
        self.assertEqual(len(maze), 5)
        self.assertEqual(len(maze[0]), 5)
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (4, 4))
        self.assertEqual(maze[0][2], 1)
        self.assertEqual(maze[0][0], 0)

    def test_parse_multiline_string_dense(self) -> None:
        """Verify parsing contiguous character multiline string."""
        grid_str = "..X..\n..X..\n..X..\n.....\n....."
        maze, start, goal = parse_grid(grid_str)
        self.assertEqual(len(maze), 5)
        self.assertEqual(len(maze[0]), 5)
        self.assertEqual(maze[1][2], 1)

    def test_parse_with_embedded_start_and_goal(self) -> None:
        """Verify automatic extraction of 'S' and 'G' in grid string."""
        grid_str = """
        S . X
        . . .
        X . G
        """
        maze, start, goal = parse_grid(grid_str)
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (2, 2))
        self.assertEqual(maze[0][0], 0)
        self.assertEqual(maze[2][2], 0)
        self.assertEqual(maze[0][2], 1)
        self.assertEqual(maze[2][0], 1)

    def test_parse_list_of_strings(self) -> None:
        """Verify parsing sequence of row strings."""
        row_strings = [". . X", ". . .", ". X ."]
        maze, start, goal = parse_grid(row_strings)
        self.assertEqual(len(maze), 3)
        self.assertEqual(len(maze[0]), 3)
        self.assertEqual(maze[0][2], 1)
        self.assertEqual(maze[2][1], 1)

    def test_parse_2d_integer_matrix(self) -> None:
        """Verify parsing 2D list of integers (0 and 1)."""
        int_grid = [[0, 1], [0, 0]]
        maze, start, goal = parse_grid(int_grid)
        self.assertEqual(maze, [[0, 1], [0, 0]])
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (1, 1))

    def test_parse_validation_errors(self) -> None:
        """Verify proper exceptions raised for malformed grid inputs."""
        # Empty string
        with self.assertRaises(ValueError):
            parse_grid("")

        # Empty sequence
        with self.assertRaises(ValueError):
            parse_grid([])

        # Non-rectangular grid
        with self.assertRaises(ValueError):
            parse_grid(["..X", ".."])

        # Invalid character
        with self.assertRaises(ValueError):
            parse_grid(["..X", ".Z."])

        # Multiple Starts
        with self.assertRaises(ValueError):
            parse_grid(["S .", "S G"])

        # Multiple Goals
        with self.assertRaises(ValueError):
            parse_grid(["S G", ". G"])

        # Start on obstacle
        with self.assertRaises(ValueError):
            parse_grid(["X .", ". ."], default_start=(0, 0))

        # Goal on obstacle
        with self.assertRaises(ValueError):
            parse_grid([". .", ". X"], default_goal=(1, 1))

        # Out of bounds coordinates
        with self.assertRaises(ValueError):
            parse_grid([". .", ". ."], default_start=(5, 5))


class TestAStarExplorationAndOptimality(unittest.TestCase):
    """Test exploration efficiency and path optimality of A* relative to BFS."""

    def test_astar_vs_bfs_on_5x5_maze(self) -> None:
        """Verify A* finds strictly optimal path matching BFS with fewer or equal expansions."""
        comp = compare_astar_vs_bfs(DEFAULT_5X5_MAZE, DEFAULT_5X5_START, DEFAULT_5X5_GOAL)
        self.assertTrue(comp.is_astar_strictly_optimal)
        self.assertEqual(comp.path_length_difference_bfs, 0)
        self.assertEqual(comp.astar_result.path_length, comp.bfs_result.path_length)
        self.assertTrue(comp.is_astar_fewer_or_equal_nodes)
        self.assertLessEqual(comp.astar_result.nodes_expanded, comp.bfs_result.nodes_expanded)

    def test_astar_vs_bfs_on_10x10_maze(self) -> None:
        """Verify A* on 10x10 benchmark explores fewer nodes than BFS while remaining optimal."""
        comp = compare_astar_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_10X10_START, DEFAULT_10X10_GOAL)
        self.assertTrue(comp.is_astar_strictly_optimal)
        self.assertEqual(comp.astar_result.path_length, 18)
        self.assertEqual(comp.bfs_result.path_length, 18)
        # On 10x10 benchmark: A* expands 77 nodes vs BFS 87 nodes
        self.assertLess(comp.astar_result.nodes_expanded, comp.bfs_result.nodes_expanded)
        self.assertEqual(comp.astar_result.nodes_expanded, 77)
        self.assertEqual(comp.bfs_result.nodes_expanded, 87)
        self.assertAlmostEqual(comp.expansion_reduction_pct_bfs, (10 / 87) * 100, places=1)

    def test_astar_vs_greedy_on_10x10_maze(self) -> None:
        """Verify A* avoids the suboptimal detour taken by Greedy Best-First Search."""
        comp = compare_astar_vs_bfs(DEFAULT_10X10_MAZE, DEFAULT_10X10_START, DEFAULT_10X10_GOAL, include_gbfs=True)
        self.assertIsNotNone(comp.gbfs_result)
        # GBFS takes 22 steps (suboptimal) while A* takes 18 steps (optimal)
        self.assertEqual(comp.astar_result.path_length, 18)
        self.assertEqual(comp.gbfs_result.path_length, 22)
        self.assertLess(comp.astar_result.path_length, comp.gbfs_result.path_length)

    def test_tie_breaking_policies(self) -> None:
        """Verify behavior across FIFO, LIFO, and goal_directed tie-breakers."""
        res_fifo = a_star_maze_solver(DEFAULT_5X5_MAZE, tie_breaker="fifo")
        res_lifo = a_star_maze_solver(DEFAULT_5X5_MAZE, tie_breaker="lifo")
        res_goal = a_star_maze_solver(DEFAULT_5X5_MAZE, tie_breaker="goal_directed")

        # All policies must find the optimal path of length 8
        self.assertEqual(res_fifo.path_length, 8)
        self.assertEqual(res_lifo.path_length, 8)
        self.assertEqual(res_goal.path_length, 8)

        # Goal-directed tie-breaking aggressively reduces expansions
        self.assertLessEqual(res_goal.nodes_expanded, res_fifo.nodes_expanded)
        self.assertEqual(res_goal.nodes_expanded, 9)

    def test_open_obstacle_free_grid(self) -> None:
        """On an open grid, path length strictly equals Manhattan distance."""
        res = a_star_maze_solver(OPEN_5X5_MAZE, (0, 0), (4, 4))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 8)
        self.assertEqual(res.path_length, manhattan_distance((0, 0), (4, 4)))

    def test_serpentine_maze(self) -> None:
        """Verify A* successfully navigates winding serpentine passages."""
        res = a_star_maze_solver(SERPENTINE_5X5_MAZE, (0, 0), (4, 4))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 16)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (4, 4))

    def test_concave_trap_maze(self) -> None:
        """Verify A* successfully escapes a U-shaped concave obstacle trap."""
        res = a_star_maze_solver(CONCAVE_TRAP_MAZE, (0, 0), (9, 9))
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 18)


class TestAStarEdgeCasesAndErrors(unittest.TestCase):
    """Test boundary conditions, trivial base cases, and unreachable mazes."""

    def test_start_is_goal(self) -> None:
        """When start == goal, return immediate 0-step path."""
        res = a_star_maze_solver(DEFAULT_5X5_MAZE, start=(2, 0), goal=(2, 0), record_steps=True)
        self.assertTrue(res.found)
        self.assertEqual(res.path, [(2, 0)])
        self.assertEqual(res.path_length, 0)
        self.assertEqual(res.nodes_expanded, 1)
        self.assertEqual(len(res.steps), 1)
        self.assertEqual(res.steps[0].action, "GOAL_REACHED")

    def test_single_cell_maze(self) -> None:
        """Verify 1x1 grid where start is goal."""
        res = a_star_maze_solver(SINGLE_CELL_MAZE, start=(0, 0), goal=(0, 0))
        self.assertTrue(res.found)
        self.assertEqual(res.path, [(0, 0)])
        self.assertEqual(res.path_length, 0)

    def test_unsolvable_maze(self) -> None:
        """Verify behavior on a completely severed / unreachable maze."""
        res = a_star_maze_solver(UNSOLVABLE_5X5_MAZE, start=(0, 0), goal=(4, 4))
        self.assertFalse(res.found)
        self.assertEqual(res.path, [])
        self.assertEqual(res.path_length, -1)
        self.assertEqual(res.total_cost, -1)

    def test_invalid_arguments(self) -> None:
        """Verify validation exceptions for empty grids and illegal coordinates."""
        with self.assertRaises(ValueError):
            a_star_maze_solver([])

        with self.assertRaises(ValueError):
            a_star_maze_solver([[1, 0]], start=(0, 0))  # Start on obstacle

        with self.assertRaises(ValueError):
            a_star_maze_solver([[0, 1]], goal=(0, 1))  # Goal on obstacle

        with self.assertRaises(ValueError):
            a_star_maze_solver(DEFAULT_5X5_MAZE, start=(-1, 0))

        with self.assertRaises(ValueError):
            a_star_maze_solver(DEFAULT_5X5_MAZE, goal=(10, 10))


class TestOptionalChallengeAndViva(unittest.TestCase):
    """Test Optional Challenge animation generators and Viva Voce documentation."""

    def test_expansion_frames_generator(self) -> None:
        """Verify Optional Challenge generator yields sequential expansion frames."""
        frames = list(generate_expansion_frames(DEFAULT_5X5_MAZE, (0, 0), (4, 4)))
        self.assertGreater(len(frames), 0)
        self.assertEqual(frames[0]["frame_number"], 1)
        self.assertEqual(frames[0]["action"], "EXPAND")
        self.assertFalse(frames[0]["is_final"])

        last_frame = frames[-1]
        self.assertTrue(last_frame["is_final"])
        self.assertEqual(last_frame["action"], "GOAL_REACHED")
        self.assertEqual(last_frame["path_length"], 8)
        self.assertIn("*", last_frame["rendered_grid"])

    def test_animate_a_star_search(self) -> None:
        """Verify live animation function executes cleanly without errors (delay=0)."""
        buffer = io.StringIO()
        animate_a_star_search(DEFAULT_5X5_MAZE, (0, 0), (4, 4), delay=0, stream=buffer)
        output = buffer.getvalue()
        self.assertIn("A* 2D MAZE EXPANSION ANIMATION", output)
        self.assertIn("GOAL REACHED! Optimal Path Length: 8 steps", output)

    def test_side_by_side_rendering(self) -> None:
        """Verify side-by-side comparison string output."""
        comp = compare_astar_vs_bfs(DEFAULT_5X5_MAZE)
        side_by_side = render_side_by_side_comparison(comp)
        self.assertIn("A* Pathfinding", side_by_side)
        self.assertIn("Breadth-First Search", side_by_side)
        self.assertIn("Exploration Reduction:", side_by_side)

    def test_to_dict_serialization(self) -> None:
        """Verify dictionary serialization for result export."""
        res = a_star_maze_solver(DEFAULT_5X5_MAZE)
        data = res.to_dict()
        self.assertEqual(data["algorithm"], "A* Pathfinding")
        self.assertEqual(data["path_length"], 8)
        self.assertEqual(data["nodes_expanded"], 16)
        self.assertTrue(data["is_optimal"])

    def test_viva_voce_knowledge_base(self) -> None:
        """Verify completeness of Viva Voce guide content."""
        required_topics = [
            "A* in Video Games",
            "Explored Set Size Relative to BFS",
            "Manhattan Distance Admissibility & Consistency",
            "Tie-Breaking in A* Grid Search",
        ]
        for topic in required_topics:
            self.assertIn(topic, VIVA_VOCE_GUIDE)
            self.assertTrue(len(VIVA_VOCE_GUIDE[topic]["question"]) > 10)
            self.assertTrue(len(VIVA_VOCE_GUIDE[topic]["answer"]) > 50)


if __name__ == "__main__":
    unittest.main()
