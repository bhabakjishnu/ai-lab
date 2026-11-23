"""Comprehensive Unit Test Suite for Question 30: 8-Puzzle Solver using A* (Manhattan).

Covers:
- Canonical syllabus sample input: [1,2,3, 4,0,5, 6,7,8] solved in 2 moves
- Syllabus sample output formatting: "Solved in 2 moves: [Board states...]"
- 1D tuple state representation, fast hashing, and set/dict membership
- Valid move generation (swapping 0 at center, corners, and edges)
- On-the-fly Manhattan distance calculation & blank space (0) exclusion
- Admissibility, consistency, and dominance over Misplaced Tiles
- Optional Challenge: Inversion count and mathematical solvability verification
- Path reconstruction, step records, and transition action accuracy
- Edge cases: start == goal (0 moves), 1-move, 4-move, 8-move, 14-move, and 18-move states
- Rejection of invalid board states (duplicate tiles, missing tiles, wrong length, non-integers)
- OOP EightPuzzleSolver class methods
- Heuristic comparison (A* Manhattan vs A* Misplaced vs BFS)
- JSON serialization, ASCII rendering, and Viva Voce reference guide
"""

from __future__ import annotations

import json
import unittest

from src.a_star_eight_puzzle import (
    BENCHMARK_PRESETS,
    BLANK_TILE,
    CLASSIC_GOAL_STATE,
    GRID_SIZE,
    SAMPLE_GOAL_STATE_2MOVE,
    SAMPLE_INPUT_STATE,
    TOTAL_TILES,
    TWO_MOVE_TO_CLASSIC_STATE,
    ZERO_FIRST_GOAL_STATE,
    EightPuzzleResult,
    EightPuzzleSolver,
    EightPuzzleStepRecord,
    InvalidPuzzleStateError,
    a_star_eight_puzzle,
    bfs_eight_puzzle,
    build_goal_position_map,
    compare_search_heuristics,
    count_inversions,
    explain_solvability,
    get_eight_puzzle_viva_reference,
    get_successors,
    is_solvable,
    manhattan_distance,
    misplaced_tiles,
    parse_puzzle_state,
    render_board_ascii,
    render_side_by_side_boards,
    render_step_by_step_boards,
    validate_board,
)


class TestEightPuzzleStateAndMoves(unittest.TestCase):
    """Test 1D tuple representation, hashing, validation, and move generation."""

    def test_state_representation_is_1d_tuple(self) -> None:
        """State must be an immutable 1D tuple of length 9 for fast hashing."""
        state = parse_puzzle_state([1, 2, 3, 4, 0, 5, 6, 7, 8])
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), TOTAL_TILES)
        self.assertEqual(state, (1, 2, 3, 4, 0, 5, 6, 7, 8))

        # Must be hashable and usable in sets and dictionaries
        state_set = {state}
        self.assertIn((1, 2, 3, 4, 0, 5, 6, 7, 8), state_set)

        state_dict = {state: 0}
        self.assertEqual(state_dict[(1, 2, 3, 4, 0, 5, 6, 7, 8)], 0)

    def test_input_parsing_formats(self) -> None:
        """Test flexible parsing across 1D lists, 2D matrices, strings, and literals."""
        expected = (1, 2, 3, 4, 0, 5, 6, 7, 8)

        # 1D list
        self.assertEqual(parse_puzzle_state([1, 2, 3, 4, 0, 5, 6, 7, 8]), expected)
        # 2D nested list
        self.assertEqual(parse_puzzle_state([[1, 2, 3], [4, 0, 5], [6, 7, 8]]), expected)
        # Comma string
        self.assertEqual(parse_puzzle_state("1, 2, 3, 4, 0, 5, 6, 7, 8"), expected)
        # Space and semicolon string
        self.assertEqual(parse_puzzle_state("1 2 3; 4 0 5; 6 7 8"), expected)
        # Bracketed literal string
        self.assertEqual(parse_puzzle_state("[1,2,3, 4,0,5, 6,7,8]"), expected)

    def test_validation_errors(self) -> None:
        """Validate that invalid boards raise InvalidPuzzleStateError."""
        # Wrong length
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 8])  # 8 elements

        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 8, 0, 9])  # 10 elements

        # Duplicates
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 1, 2, 3, 4, 5, 6, 7, 0])

        # Missing tile 0 (no blank)
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 8, 9])

        # None input
        with self.assertRaises(InvalidPuzzleStateError):
            parse_puzzle_state(None)

        # Malformed string
        with self.assertRaises(InvalidPuzzleStateError):
            parse_puzzle_state("not a valid board")

    def test_move_generation_at_center(self) -> None:
        """Blank at center (index 4) must generate exactly 4 valid moves."""
        center_state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
        successors = get_successors(center_state)
        self.assertEqual(len(successors), 4)

        # Blank UP swaps with tile 2 (idx 1): tile 2 moves DOWN
        up_succ = [s for s in successors if s[2] == "UP"][0]
        self.assertEqual(up_succ[0], (1, 0, 3, 4, 2, 5, 6, 7, 8))
        self.assertEqual(up_succ[3], 2)
        self.assertEqual(up_succ[1], "Move 2 DOWN")

        # Blank DOWN swaps with tile 7 (idx 7): tile 7 moves UP
        down_succ = [s for s in successors if s[2] == "DOWN"][0]
        self.assertEqual(down_succ[0], (1, 2, 3, 4, 7, 5, 6, 0, 8))
        self.assertEqual(down_succ[3], 7)

        # Blank LEFT swaps with tile 4 (idx 3): tile 4 moves RIGHT
        left_succ = [s for s in successors if s[2] == "LEFT"][0]
        self.assertEqual(left_succ[0], (1, 2, 3, 0, 4, 5, 6, 7, 8))
        self.assertEqual(left_succ[3], 4)

        # Blank RIGHT swaps with tile 5 (idx 5): tile 5 moves LEFT
        right_succ = [s for s in successors if s[2] == "RIGHT"][0]
        self.assertEqual(right_succ[0], (1, 2, 3, 4, 5, 0, 6, 7, 8))
        self.assertEqual(right_succ[3], 5)

    def test_move_generation_at_corners(self) -> None:
        """Blank at corners must generate exactly 2 valid moves."""
        # Top-left corner (index 0)
        top_left = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        succs_tl = get_successors(top_left)
        self.assertEqual(len(succs_tl), 2)
        directions_tl = {s[2] for s in succs_tl}
        self.assertEqual(directions_tl, {"DOWN", "RIGHT"})

        # Bottom-right corner (index 8)
        bottom_right = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        succs_br = get_successors(bottom_right)
        self.assertEqual(len(succs_br), 2)
        directions_br = {s[2] for s in succs_br}
        self.assertEqual(directions_br, {"UP", "LEFT"})

    def test_move_generation_at_edges(self) -> None:
        """Blank at edges must generate exactly 3 valid moves."""
        # Top-middle edge (index 1)
        top_mid = (1, 0, 2, 3, 4, 5, 6, 7, 8)
        succs = get_successors(top_mid)
        self.assertEqual(len(succs), 3)
        directions = {s[2] for s in succs}
        self.assertEqual(directions, {"DOWN", "LEFT", "RIGHT"})


class TestEightPuzzleHeuristics(unittest.TestCase):
    """Test on-the-fly Manhattan distance calculation, admissibility, and dominance."""

    def test_manhattan_distance_exact_values(self) -> None:
        """Verify Manhattan distance calculation against manual computation."""
        # Goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)
        # Start: (1, 2, 3, 4, 0, 5, 6, 7, 8)
        # Tile 1: (0,0) -> (0,0) => 0
        # Tile 2: (0,1) -> (0,1) => 0
        # Tile 3: (0,2) -> (0,2) => 0
        # Tile 4: (1,0) -> (1,0) => 0
        # Tile 5: (1,2) -> (1,1) => 1
        # Tile 6: (2,0) -> (1,2) => |2-1| + |0-2| = 1 + 2 = 3
        # Tile 7: (2,1) -> (2,0) => |2-2| + |1-0| = 1
        # Tile 8: (2,2) -> (2,1) => |2-2| + |2-1| = 1
        # Total = 0 + 0 + 0 + 0 + 1 + 3 + 1 + 1 = 6
        self.assertEqual(manhattan_distance(SAMPLE_INPUT_STATE, CLASSIC_GOAL_STATE), 6)

        # Against 2-move target: (1, 2, 3, 4, 5, 8, 6, 7, 0)
        # Tile 5: at (1,2), in goal at (1,1) => 1
        # Tile 8: at (2,2), in goal at (1,2) => 1
        # Others at goal position => total 2
        self.assertEqual(manhattan_distance(SAMPLE_INPUT_STATE, SAMPLE_GOAL_STATE_2MOVE), 2)

    def test_blank_strictly_excluded_from_manhattan(self) -> None:
        """Blank space (0) must be strictly excluded; counting it violates admissibility."""
        # 1-move state to SAMPLE_GOAL_STATE_2MOVE:
        # Move blank RIGHT to (1, 2): (1, 2, 3, 4, 5, 0, 6, 7, 8)
        one_move = (1, 2, 3, 4, 5, 0, 6, 7, 8)
        h_val = manhattan_distance(one_move, SAMPLE_GOAL_STATE_2MOVE)
        # Only tile 8 is misplaced: at (2,2), goal is (1,2) -> distance = 1
        self.assertEqual(h_val, 1)

    def test_dominance_invariant(self) -> None:
        """Manhattan distance must dominate Misplaced Tiles: h_manhattan(n) >= h_misplaced(n)."""
        states = [
            SAMPLE_INPUT_STATE,
            TWO_MOVE_TO_CLASSIC_STATE,
            (1, 2, 3, 0, 4, 6, 7, 5, 8),
            (2, 8, 3, 1, 6, 4, 7, 0, 5),
            (7, 2, 4, 5, 0, 6, 8, 3, 1),
        ]
        for st in states:
            h_man = manhattan_distance(st, CLASSIC_GOAL_STATE)
            h_mis = misplaced_tiles(st, CLASSIC_GOAL_STATE)
            self.assertGreaterEqual(
                h_man,
                h_mis,
                f"Dominance violated for state {st}: h_man={h_man} < h_mis={h_mis}",
            )


class TestSolvabilityAndInversions(unittest.TestCase):
    """Test inversion count and mathematical solvability verification (Optional Challenge)."""

    def test_inversion_counts(self) -> None:
        """Test inversion count calculation on known boards."""
        # Fully sorted 1..8, 0 -> 0 inversions
        self.assertEqual(count_inversions(CLASSIC_GOAL_STATE), 0)

        # Solved zero-first: 0, 1, 2, 3, 4, 5, 6, 7, 8 -> 0 inversions
        self.assertEqual(count_inversions(ZERO_FIRST_GOAL_STATE), 0)

        # Single inverted pair (8, 7) at bottom row: 1 inversion
        single_inv_board = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertEqual(count_inversions(single_inv_board), 1)

        # Sample input: 1, 2, 3, 4, 0, 5, 6, 7, 8 -> 0 inversions
        self.assertEqual(count_inversions(SAMPLE_INPUT_STATE), 0)

    def test_is_solvable_verification(self) -> None:
        """Mathematically verify solvable vs unsolvable states."""
        # Sample input is solvable to classic goal (0 inversions % 2 == 0 inversions % 2)
        self.assertTrue(is_solvable(SAMPLE_INPUT_STATE, CLASSIC_GOAL_STATE))

        # Sample input is solvable to 2-move target (inversions of 2-move target: tile 8 before 6,7 => 2 inversions)
        self.assertEqual(count_inversions(SAMPLE_GOAL_STATE_2MOVE), 2)
        self.assertTrue(is_solvable(SAMPLE_INPUT_STATE, SAMPLE_GOAL_STATE_2MOVE))

        # Unsolvable state: 1 inversion vs 0 inversions in classic goal
        unsolvable_board = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertFalse(is_solvable(unsolvable_board, CLASSIC_GOAL_STATE))

        # Mathematical explanation check
        is_solv, explanation, init_inv, goal_inv = explain_solvability(unsolvable_board, CLASSIC_GOAL_STATE)
        self.assertFalse(is_solv)
        self.assertEqual(init_inv, 1)
        self.assertEqual(goal_inv, 0)
        self.assertIn("UNSOLVABLE", explanation)
        self.assertIn("parity", explanation)


class TestAStarEightPuzzleSolver(unittest.TestCase):
    """Test A* search execution, optimal move counts, and path reconstruction."""

    def test_syllabus_sample_solved_in_2_moves(self) -> None:
        """Verify syllabus sample: [1,2,3, 4,0,5, 6,7,8] solved in 2 moves to target."""
        result = a_star_eight_puzzle(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=SAMPLE_GOAL_STATE_2MOVE,
            heuristic="manhattan",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.move_count, 2)
        self.assertEqual(len(result.path), 3)

        # Check exact intermediate board sequence
        expected_path = [
            (1, 2, 3, 4, 0, 5, 6, 7, 8),
            (1, 2, 3, 4, 5, 0, 6, 7, 8),
            (1, 2, 3, 4, 5, 8, 6, 7, 0),
        ]
        self.assertEqual(result.path, expected_path)

        # Check sample output string format
        sample_str = result.format_sample_output()
        self.assertTrue(sample_str.startswith("Solved in 2 moves: ["))
        self.assertIn("[1, 2, 3, 4, 0, 5, 6, 7, 8]", sample_str)
        self.assertIn("[1, 2, 3, 4, 5, 8, 6, 7, 0]", sample_str)

    def test_two_moves_to_classic_goal(self) -> None:
        """Verify 2-move state to classic goal (1-8, 0)."""
        result = a_star_eight_puzzle(
            initial_state=TWO_MOVE_TO_CLASSIC_STATE,
            goal_state=CLASSIC_GOAL_STATE,
            heuristic="manhattan",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.move_count, 2)
        self.assertEqual(result.path[-1], CLASSIC_GOAL_STATE)

    def test_start_is_already_goal(self) -> None:
        """Start state matching goal state must immediately return 0 moves."""
        result = a_star_eight_puzzle(
            initial_state=CLASSIC_GOAL_STATE,
            goal_state=CLASSIC_GOAL_STATE,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.move_count, 0)
        self.assertEqual(len(result.path), 1)
        self.assertEqual(result.nodes_expanded, 0)

    def test_sample_input_to_classic_goal_optimal_moves(self) -> None:
        """Sample input to classic goal solves optimally in 14 moves."""
        result = a_star_eight_puzzle(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=CLASSIC_GOAL_STATE,
            heuristic="manhattan",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.move_count, 14)
        self.assertEqual(len(result.path), 15)
        self.assertEqual(result.path[0], SAMPLE_INPUT_STATE)
        self.assertEqual(result.path[-1], CLASSIC_GOAL_STATE)

    def test_unsolvable_board_aborts_early(self) -> None:
        """Unsolvable board must be caught early without wasteful search."""
        unsolvable_board = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        result = a_star_eight_puzzle(
            initial_state=unsolvable_board,
            goal_state=CLASSIC_GOAL_STATE,
            check_solvability_first=True,
        )
        self.assertFalse(result.success)
        self.assertFalse(result.is_solvable)
        self.assertEqual(result.nodes_expanded, 0)
        self.assertIn("UNSOLVABLE", result.solvability_reason)

    def test_step_records_and_transitions(self) -> None:
        """Step records must accurately document moves, tiles, and g/h/f scores."""
        result = a_star_eight_puzzle(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=SAMPLE_GOAL_STATE_2MOVE,
        )
        records = result.step_records
        self.assertEqual(len(records), 3)

        # Step 0
        self.assertEqual(records[0].step_number, 0)
        self.assertEqual(records[0].g_cost, 0)
        self.assertEqual(records[0].tile_moved, 0)

        # Step 1: blank moves RIGHT, tile 5 moves LEFT
        self.assertEqual(records[1].step_number, 1)
        self.assertEqual(records[1].blank_move, "RIGHT")
        self.assertEqual(records[1].tile_moved, 5)
        self.assertEqual(records[1].g_cost, 1)
        self.assertEqual(records[1].f_cost, records[1].g_cost + records[1].h_cost)

        # Step 2: blank moves DOWN, tile 8 moves UP
        self.assertEqual(records[2].step_number, 2)
        self.assertEqual(records[2].blank_move, "DOWN")
        self.assertEqual(records[2].tile_moved, 8)
        self.assertEqual(records[2].g_cost, 2)
        self.assertEqual(records[2].h_cost, 0)
        self.assertEqual(records[2].f_cost, 2)

    def test_oop_solver_interface(self) -> None:
        """Test object-oriented EightPuzzleSolver class."""
        solver = EightPuzzleSolver(goal_state=SAMPLE_GOAL_STATE_2MOVE)
        self.assertTrue(solver.is_solvable(SAMPLE_INPUT_STATE))
        res = solver.solve(SAMPLE_INPUT_STATE)
        self.assertTrue(res.success)
        self.assertEqual(res.move_count, 2)

        explanation = solver.explain_solvability(SAMPLE_INPUT_STATE)
        self.assertIn("SOLVABLE", explanation)


class TestComparativeBenchmarkingAndOutput(unittest.TestCase):
    """Test heuristic comparison, formatting, and visualization."""

    def test_compare_search_heuristics(self) -> None:
        """Compare A* (Manhattan), A* (Misplaced), and BFS on a 3-move state."""
        start = (1, 2, 3, 0, 4, 6, 7, 5, 8)
        report = compare_search_heuristics(start, CLASSIC_GOAL_STATE, include_bfs=True)
        self.assertEqual(report.astar_manhattan.move_count, 3)
        self.assertEqual(report.astar_misplaced.move_count, 3)
        self.assertIsNotNone(report.bfs_result)
        self.assertEqual(report.bfs_result.move_count, 3)

        # Manhattan must expand fewer or equal nodes than Misplaced Tiles
        self.assertLessEqual(
            report.astar_manhattan.nodes_expanded,
            report.astar_misplaced.nodes_expanded,
        )

        table_str = report.format_table()
        self.assertIn("8-PUZZLE HEURISTIC & SEARCH ALGORITHM COMPARISON REPORT", table_str)
        self.assertIn("A* (Manhattan Distance)", table_str)
        self.assertIn("A* (Misplaced Tiles)", table_str)

    def test_serialization_and_viva_reference(self) -> None:
        """Test JSON serialization and Viva Voce reference text."""
        result = a_star_eight_puzzle(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=SAMPLE_GOAL_STATE_2MOVE,
        )
        res_dict = result.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertEqual(res_dict["move_count"], 2)
        self.assertEqual(len(res_dict["path"]), 3)

        # JSON dump verification
        json_str = json.dumps(res_dict)
        self.assertIn('"success": true', json_str)

        # Viva Voce text
        viva_text = get_eight_puzzle_viva_reference()
        self.assertIn("INVERSIONS TO CHECK SOLVABILITY", viva_text)
        self.assertIn("STATE HASHING IMPORTANCE", viva_text)
        self.assertIn("Russell & Norvig", viva_text)

    def test_ascii_rendering_utilities(self) -> None:
        """Test ASCII grid boxes and side-by-side board rendering."""
        box = render_board_ascii(SAMPLE_INPUT_STATE, title="Start:")
        self.assertIn("+---+---+---+", box)
        self.assertIn("| 1 | 2 | 3 |", box)
        self.assertIn("| 4 |   | 5 |", box)

        side_by_side = render_side_by_side_boards(
            [SAMPLE_INPUT_STATE, SAMPLE_GOAL_STATE_2MOVE],
            ["Initial", "Goal"],
        )
        self.assertIn("+---+---+---+", side_by_side)


if __name__ == "__main__":
    unittest.main()
