"""Comprehensive unit test suite for BFS 2D Grid Maze Solver (Question 15).

Validates:
- Exact sample input and output from Question 15 prompt ([[0,1,0],[0,0,0],[1,1,0]] -> 4 steps).
- Optional Challenge: output string of moves ("D, R, R, D") and action sequence list.
- Exact coordinate path reconstruction [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)].
- Non-square rectangular grids (N != M, e.g. 3x5, 2x4, 4x2).
- Single-cell maze (1x1 grid, start == goal -> 0 steps, empty action sequence).
- Open obstacle-free grid: verifies BFS shortest path strictly equals Manhattan distance.
- Complex corridor detour: serpentine labyrinth where BFS steps > Manhattan distance.
- Unsolvable mazes: completely walled off destination returns steps = -1, found = False.
- Blocked endpoints: start cell or goal cell occupied by a wall (value 1).
- Custom start and goal coordinates: non-standard start and goal pairs.
- Grid parsing robustness: 2D lists of ints, lists of strings, JSON strings, multiline text, ASCII glyphs.
- Validation and error handling: non-rectangular grids, out-of-bounds coordinates, invalid cell tokens.
- Dynamic successor generation: boundary safety and obstacle collision avoidance.
- Object-Oriented BFSMazeSolver class interface and ASCII rendering.
- Headless graphical diagram generation via Matplotlib.
- Viva Voce theory reference and algorithm comparison matrix data completeness.
"""

from pathlib import Path
import tempfile
import unittest

from src.bfs_maze_solver import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BFSMazeResult,
    BFSMazeSolver,
    CELL_OPEN,
    CELL_WALL,
    DEFAULT_DIRECTIONS,
    DEFAULT_SAMPLE_MAZE,
    DOWN_RIGHT_DIRECTIONS,
    NON_SQUARE_RECTANGULAR_MAZE,
    OPEN_GRID_4X4_MAZE,
    SERPENTINE_MAZE,
    SINGLE_CELL_MAZE,
    UNSOLVABLE_BLOCKED_MAZE,
    calculate_manhattan_distance,
    get_maze_solver_viva_reference,
    get_successors,
    is_open_cell,
    is_valid_coordinate,
    parse_grid,
    solve_maze_bfs,
    validate_grid,
    visualize_maze_solution,
)


class TestBFSMazeSolverCore(unittest.TestCase):
    """Core algorithmic tests for BFS 2D Maze Solver."""

    def test_sample_input_and_expected_output(self) -> None:
        """Verify exact sample from Question 15 prompt: [[0,1,0],[0,0,0],[1,1,0]] -> steps: 4."""
        grid = [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
        result = solve_maze_bfs(grid)

        self.assertTrue(result.found)
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.start, (0, 0))
        self.assertEqual(result.goal, (2, 2))

    def test_optional_challenge_move_string(self) -> None:
        """Verify Optional Challenge: Output a string of moves: 'D, R, R, D'."""
        grid = [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
        result = solve_maze_bfs(grid)

        self.assertEqual(result.actions, ["D", "R", "R", "D"])
        self.assertEqual(result.actions_str, "D, R, R, D")

    def test_sample_path_coordinates_sequence(self) -> None:
        """Verify exact path coordinates traversal for the sample 3x3 maze."""
        grid = [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
        result = solve_maze_bfs(grid)

        expected_path = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
        self.assertEqual(result.path, expected_path)
        self.assertEqual(len(result.path) - 1, result.steps)

    def test_non_square_rectangular_grid_3x5(self) -> None:
        """Verify solver correctly handles non-square rectangular grids (3 rows x 5 columns)."""
        grid = NON_SQUARE_RECTANGULAR_MAZE
        result = solve_maze_bfs(grid)

        self.assertTrue(result.found)
        self.assertEqual(result.start, (0, 0))
        self.assertEqual(result.goal, (2, 4))
        self.assertEqual(result.steps, 6)
        self.assertEqual(result.actions_str, "R, D, D, R, R, R")

    def test_non_square_grid_2x4(self) -> None:
        """Verify solver on 2x4 non-square grid with walls."""
        grid = [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        result = solve_maze_bfs(grid)
        self.assertTrue(result.found)
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (1, 3))

    def test_single_cell_maze(self) -> None:
        """Verify base case: 1x1 grid where start is goal."""
        grid = SINGLE_CELL_MAZE
        result = solve_maze_bfs(grid)

        self.assertTrue(result.found)
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.path, [(0, 0)])
        self.assertEqual(result.actions, [])
        self.assertEqual(result.actions_str, "")

    def test_open_obstacle_free_grid_manhattan_equality(self) -> None:
        """Verify that on an obstacle-free grid, BFS steps strictly equals Manhattan distance."""
        grid = OPEN_GRID_4X4_MAZE
        result = solve_maze_bfs(grid)

        manhattan = calculate_manhattan_distance((0, 0), (3, 3))
        self.assertEqual(manhattan, 6)
        self.assertTrue(result.found)
        self.assertEqual(result.steps, manhattan)

    def test_serpentine_maze_forces_detour(self) -> None:
        """Verify that walls force detours where BFS steps > Manhattan distance."""
        grid = SERPENTINE_MAZE
        result = solve_maze_bfs(grid)

        manhattan = calculate_manhattan_distance((0, 0), (4, 4))
        self.assertEqual(manhattan, 8)
        self.assertTrue(result.found)
        self.assertEqual(result.steps, 16)
        self.assertGreater(result.steps, manhattan)
        self.assertEqual(len(result.actions), 16)

    def test_unsolvable_completely_blocked_maze(self) -> None:
        """Verify that when no open path exists to goal, steps=-1 and found=False."""
        grid = UNSOLVABLE_BLOCKED_MAZE
        result = solve_maze_bfs(grid)

        self.assertFalse(result.found)
        self.assertEqual(result.steps, -1)
        self.assertEqual(result.path, [])
        self.assertEqual(result.actions, [])
        self.assertEqual(result.actions_str, "")

    def test_blocked_start_cell_wall(self) -> None:
        """Verify handling when the start coordinate is a wall (value 1)."""
        grid = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        result = solve_maze_bfs(grid, start=(0, 0), goal=(2, 2))
        self.assertFalse(result.found)
        self.assertEqual(result.steps, -1)

    def test_blocked_goal_cell_wall(self) -> None:
        """Verify handling when the goal coordinate is a wall (value 1)."""
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]
        result = solve_maze_bfs(grid, start=(0, 0), goal=(2, 2))
        self.assertFalse(result.found)
        self.assertEqual(result.steps, -1)

    def test_custom_start_and_goal_coordinates(self) -> None:
        """Verify solving from non-default start and goal coordinates."""
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
        ]
        # Start at (1, 2) and navigate to (1, 0)
        result = solve_maze_bfs(grid, start=(1, 2), goal=(1, 0))
        self.assertTrue(result.found)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.path, [(1, 2), (1, 1), (1, 0)])
        self.assertEqual(result.actions, ["L", "L"])
        self.assertEqual(result.actions_str, "L, L")

    def test_custom_directional_priority_order(self) -> None:
        """Verify search with DOWN_RIGHT_DIRECTIONS custom move order."""
        grid = [[0, 0], [0, 0]]
        result = solve_maze_bfs(grid, directions=DOWN_RIGHT_DIRECTIONS)
        self.assertTrue(result.found)
        self.assertEqual(result.steps, 2)
        # DOWN then RIGHT
        self.assertEqual(result.actions, ["D", "R"])


class TestGridParsingAndValidation(unittest.TestCase):
    """Tests for grid parsing and input normalization."""

    def test_parse_grid_from_2d_list_ints(self) -> None:
        """Verify parsing 2D list of integers."""
        data = [[0, 1], [1, 0]]
        parsed = parse_grid(data)
        self.assertEqual(parsed, [[0, 1], [1, 0]])

    def test_parse_grid_from_json_string(self) -> None:
        """Verify parsing JSON string formatted grid."""
        raw_json = "[[0, 1, 0], [0, 0, 0], [1, 1, 0]]"
        parsed = parse_grid(raw_json)
        self.assertEqual(parsed, DEFAULT_SAMPLE_MAZE)

    def test_parse_grid_from_multiline_text(self) -> None:
        """Verify parsing multiline string with space/comma separation."""
        raw_text = """
        0 1 0
        0 0 0
        1 1 0
        """
        parsed = parse_grid(raw_text)
        self.assertEqual(parsed, DEFAULT_SAMPLE_MAZE)

    def test_parse_grid_from_ascii_glyphs(self) -> None:
        """Verify parsing ASCII glyphs ('.' for 0, '#' for 1)."""
        raw_ascii = """
        . # .
        . . .
        # # .
        """
        parsed = parse_grid(raw_ascii)
        self.assertEqual(parsed, DEFAULT_SAMPLE_MAZE)

    def test_parse_grid_from_booleans(self) -> None:
        """Verify parsing grid of booleans (False for open 0, True for wall 1)."""
        bool_grid = [[False, True], [False, False]]
        parsed = parse_grid(bool_grid)
        self.assertEqual(parsed, [[0, 1], [0, 0]])

    def test_parse_grid_rejects_empty_list(self) -> None:
        """Verify empty list raises ValueError."""
        with self.assertRaises(ValueError):
            parse_grid([])

    def test_parse_grid_rejects_none(self) -> None:
        """Verify None raises ValueError."""
        with self.assertRaises(ValueError):
            parse_grid(None)

    def test_parse_grid_rejects_non_rectangular(self) -> None:
        """Verify non-rectangular grid raises ValueError."""
        jagged = [[0, 1, 0], [0, 0]]
        with self.assertRaises(ValueError):
            parse_grid(jagged)

    def test_parse_grid_rejects_invalid_cell_values(self) -> None:
        """Verify values outside 0 and 1 raise ValueError."""
        invalid = [[0, 9], [0, 0]]
        with self.assertRaises(ValueError):
            parse_grid(invalid)

    def test_start_out_of_bounds_raises_value_error(self) -> None:
        """Verify out-of-bounds start coordinate raises ValueError."""
        with self.assertRaises(ValueError):
            solve_maze_bfs([[0, 0], [0, 0]], start=(-1, 0))

    def test_goal_out_of_bounds_raises_value_error(self) -> None:
        """Verify out-of-bounds goal coordinate raises ValueError."""
        with self.assertRaises(ValueError):
            solve_maze_bfs([[0, 0], [0, 0]], goal=(5, 5))


class TestSuccessorGenerationAndBoundaryChecks(unittest.TestCase):
    """Tests for dynamic successor generation and coordinate safety."""

    def test_corner_successors_boundary_safety(self) -> None:
        """Verify corner cell (0,0) only generates in-bounds orthogonal neighbors."""
        grid = [
            [0, 0],
            [0, 0],
        ]
        succs = get_successors((0, 0), grid)
        # Up (-1,0) and Left (0,-1) are out of bounds. Down (1,0) and Right (0,1) are valid.
        actions = [act for act, _ in succs]
        coords = [coord for _, coord in succs]
        self.assertIn("D", actions)
        self.assertIn("R", actions)
        self.assertNotIn("U", actions)
        self.assertNotIn("L", actions)
        self.assertEqual(coords, [(1, 0), (0, 1)])

    def test_wall_neighbors_excluded_from_successors(self) -> None:
        """Verify walls are excluded from dynamic successor list."""
        grid = [
            [0, 1],
            [0, 0],
        ]
        succs = get_successors((0, 0), grid)
        # (0,1) is wall, so only Down (1,0) should be generated
        self.assertEqual(succs, [("D", (1, 0))])

    def test_cell_surrounded_by_walls_has_zero_successors(self) -> None:
        """Verify enclosed cell has no valid successors."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        succs = get_successors((1, 1), grid)
        self.assertEqual(succs, [])

    def test_is_valid_coordinate(self) -> None:
        """Verify is_valid_coordinate boundary predicate."""
        self.assertTrue(is_valid_coordinate(0, 0, 3, 3))
        self.assertTrue(is_valid_coordinate(2, 2, 3, 3))
        self.assertFalse(is_valid_coordinate(-1, 0, 3, 3))
        self.assertFalse(is_valid_coordinate(0, -1, 3, 3))
        self.assertFalse(is_valid_coordinate(3, 0, 3, 3))
        self.assertFalse(is_valid_coordinate(0, 3, 3, 3))

    def test_is_open_cell(self) -> None:
        """Verify is_open_cell checks bounds and cell value."""
        grid = [[0, 1], [0, 0]]
        self.assertTrue(is_open_cell(0, 0, grid))
        self.assertFalse(is_open_cell(0, 1, grid))  # Wall
        self.assertFalse(is_open_cell(-1, 0, grid))  # Out of bounds


class TestBFSMazeSolverOOPInterface(unittest.TestCase):
    """Tests for Object-Oriented BFSMazeSolver wrapper."""

    def test_solver_oop_lifecycle(self) -> None:
        """Verify OOP solver methods: initialization, dimensions, solve, ascii rendering."""
        solver = BFSMazeSolver(DEFAULT_SAMPLE_MAZE)
        self.assertEqual(solver.get_dimensions(), (3, 3))

        result = solver.solve()
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.actions_str, "D, R, R, D")

        ascii_art = solver.render_ascii()
        self.assertIn("S", ascii_art)
        self.assertIn("G", ascii_art)
        self.assertIn("*", ascii_art)
        self.assertIn("#", ascii_art)

    def test_result_to_dict_and_format_summary(self) -> None:
        """Verify result serialization and formatted text summary."""
        result = solve_maze_bfs(DEFAULT_SAMPLE_MAZE)
        summary = result.format_summary()
        self.assertIn("Shortest Steps    : 4", summary)
        self.assertIn("Action Moves      : D, R, R, D", summary)

        d = result.to_dict()
        self.assertEqual(d["steps"], 4)
        self.assertEqual(d["found"], True)
        self.assertEqual(d["actions_str"], "D, R, R, D")


class TestVisualizationAndVivaVoce(unittest.TestCase):
    """Tests for headless diagram generation and Viva Voce reference completeness."""

    def test_visualize_maze_solution_headless(self) -> None:
        """Verify headless Matplotlib visualization produces valid image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_maze.png"
            saved_path = visualize_maze_solution(DEFAULT_SAMPLE_MAZE, output_path=out_file, show=False)

            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_get_maze_solver_viva_reference(self) -> None:
        """Verify viva reference contains required topics: implicit vs explicit, successors, Manhattan distance."""
        viva = get_maze_solver_viva_reference()
        self.assertEqual(viva["title"], "Question 15: BFS 2D Maze Solver - Viva Voce & Theory Guide")

        topic_titles = [t["topic"] for t in viva["core_topics"]]
        self.assertIn("Implicit vs. Explicit Graphs", topic_titles)
        self.assertIn("Valid Action Generation", topic_titles)
        self.assertIn("Manhattan Distance Relation", topic_titles)
        self.assertIn("FIFO Queue & collections.deque", topic_titles)
        self.assertIn("Time and Space Complexity", topic_titles)

        matrix = viva["comparison_matrix"]
        self.assertIn("BFS (Breadth-First)", [row[0] for row in matrix["rows"]])
        self.assertIn("DFS (Depth-First)", [row[0] for row in matrix["rows"]])
        self.assertIn("UCS (Uniform Cost)", [row[0] for row in matrix["rows"]])


if __name__ == "__main__":
    unittest.main()
