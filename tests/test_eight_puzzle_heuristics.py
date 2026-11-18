"""Comprehensive unit tests for Question 25: 8-Puzzle Heuristic Evaluator.

Covers:
- Canonical syllabus sample input/output (h1 = 8, h2 = 18)
- Goal state (h1 = 0, h2 = 0) and 1-move states (h1 = 1, h2 = 1)
- Blank space (0) exclusion guarantee & inadmissibility counterexample
- divmod coordinate math, flattening/unflattening, and O(N) position mapping
- Dominance invariant (0 <= h1(n) <= h2(n)) across canonical and random boards
- Solvability / inversion count parity (standard, classic, and custom odd-parity goals)
- Optional Challenge: random solvable board generator (seed determinism & solvability)
- Input parsing across matrix literals, semicolon rows, flat strings, and presets
- Strict validation for invalid dimensions, duplicates, missing/out-of-range tiles, and types
- Formatting, ASCII rendering, JSON serialization, OOP evaluator, and Viva reference
"""

import json
import unittest

from src.eight_puzzle_heuristics import (
    BLANK_TILE,
    CLASSIC_GOAL_STATE,
    EASY_STATE,
    HARD_STATE,
    ONE_MOVE_STATE,
    SAMPLE_CURRENT_STATE,
    SAMPLE_GOAL_STATE,
    UNSOLVABLE_STATE,
    EightPuzzleHeuristicEvaluator,
    analyze_dominance_over_samples,
    build_position_map,
    count_inversions,
    demonstrate_blank_space_inadmissibility,
    evaluate_heuristics,
    flatten_board,
    format_evaluation_report,
    format_tile_breakdown_table,
    generate_random_solvable_board,
    get_eight_puzzle_viva_reference,
    index_to_coords,
    is_solvable,
    manhattan_distance_heuristic,
    misplaced_tiles_heuristic,
    parse_board,
    render_board_ascii,
    render_side_by_side_boards,
    unflatten_board,
    validate_board,
)


class TestEightPuzzleCoreHeuristics(unittest.TestCase):
    """Test core h1 (Misplaced Tiles) and h2 (Manhattan Distance) calculations."""

    def test_canonical_syllabus_sample(self) -> None:
        """Verify exact syllabus sample: Current=[[7,2,4],[5,0,6],[8,3,1]], Goal=[[0,1,2],[3,4,5],[6,7,8]]."""
        current = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
        goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        self.assertEqual(misplaced_tiles_heuristic(current, goal), 8)
        self.assertEqual(manhattan_distance_heuristic(current, goal), 18)

        result = evaluate_heuristics(current, goal)
        self.assertEqual(result.h1_misplaced, 8)
        self.assertEqual(result.h2_manhattan, 18)
        self.assertEqual(
            result.format_sample_output(),
            "h1 (Misplaced): 8, h2 (Manhattan): 18",
        )
        self.assertFalse(result.is_goal)
        self.assertTrue(result.is_solvable)
        self.assertTrue(result.dominance_holds)
        self.assertEqual(result.dominance_gap, 10)

        # Verify per-tile Manhattan distances match manual calculation
        expected_tile_manhattan = {1: 3, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2}
        actual_tile_manhattan = {d.tile: d.manhattan_distance for d in result.tile_details}
        self.assertEqual(actual_tile_manhattan, expected_tile_manhattan)
        self.assertEqual(sum(actual_tile_manhattan.values()), 18)
        self.assertNotIn(BLANK_TILE, actual_tile_manhattan)

    def test_goal_state_returns_zero(self) -> None:
        """Goal state must yield h1 = 0, h2 = 0, is_goal = True."""
        result = evaluate_heuristics(SAMPLE_GOAL_STATE, SAMPLE_GOAL_STATE)
        self.assertEqual(result.h1_misplaced, 0)
        self.assertEqual(result.h2_manhattan, 0)
        self.assertTrue(result.is_goal)
        self.assertTrue(result.is_solvable)
        self.assertTrue(result.dominance_holds)
        self.assertEqual(result.dominance_gap, 0)
        for detail in result.tile_details:
            self.assertFalse(detail.is_misplaced)
            self.assertEqual(detail.manhattan_distance, 0)

    def test_blank_space_strictly_excluded_on_one_move_state(self) -> None:
        """1-move state must return h1 = 1, h2 = 1 (not 2!), proving blank 0 is ignored."""
        self.assertEqual(misplaced_tiles_heuristic(ONE_MOVE_STATE, SAMPLE_GOAL_STATE), 1)
        self.assertEqual(manhattan_distance_heuristic(ONE_MOVE_STATE, SAMPLE_GOAL_STATE), 1)

        demo = demonstrate_blank_space_inadmissibility()
        self.assertEqual(demo["true_optimal_cost"], 1)
        self.assertEqual(demo["h1_valid"], 1)
        self.assertEqual(demo["h2_valid"], 1)
        self.assertTrue(demo["h1_valid_admissible"])
        self.assertTrue(demo["h2_valid_admissible"])
        self.assertEqual(demo["h1_with_blank"], 2)
        self.assertEqual(demo["h2_with_blank"], 2)
        self.assertFalse(demo["h1_with_blank_admissible"])
        self.assertFalse(demo["h2_with_blank_admissible"])

    def test_classic_zero_last_goal_state(self) -> None:
        """Evaluate heuristics against alternative classic goal [[1,2,3],[4,5,6],[7,8,0]]."""
        # Swap only tile 8 and blank 0: 1 move away from CLASSIC_GOAL_STATE
        one_move_classic = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
        res = evaluate_heuristics(one_move_classic, CLASSIC_GOAL_STATE)
        self.assertEqual(res.h1_misplaced, 1)
        self.assertEqual(res.h2_manhattan, 1)
        self.assertTrue(res.is_solvable)

        # Reversed non-blank board vs classic goal
        rev_res = evaluate_heuristics(HARD_STATE, CLASSIC_GOAL_STATE)
        self.assertEqual(rev_res.h1_misplaced, 8)
        self.assertEqual(rev_res.h2_manhattan, 16)
        self.assertTrue(rev_res.dominance_holds)

    def test_custom_arbitrary_goal_state(self) -> None:
        """Heuristics must work correctly for arbitrary valid 3x3 goal states."""
        custom_goal = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]  # Spiral goal
        current = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
        # Tile 1: (1,0)->(0,0)=1; Tile 2: (0,0)->(0,1)=1; Tile 3: (0,2)->(0,2)=0;
        # Tile 4: (1,2)->(1,2)=0; Tile 5: (2,2)->(2,2)=0; Tile 6: (1,1)->(2,1)=1;
        # Tile 7: (2,0)->(2,0)=0; Tile 8: (0,1)->(1,0)=2 -> h1=4, h2=5
        self.assertEqual(misplaced_tiles_heuristic(current, custom_goal), 4)
        self.assertEqual(manhattan_distance_heuristic(current, custom_goal), 5)

    def test_easy_state_preset(self) -> None:
        """Verify EASY_STATE preset against SAMPLE_GOAL_STATE."""
        # EASY_STATE = ((1,2,5),(3,4,0),(6,7,8)) vs ((0,1,2),(3,4,5),(6,7,8))
        # Misplaced: 1 (dist 1), 2 (dist 1), 5 (dist 1) -> h1 = 3, h2 = 3
        res = evaluate_heuristics(EASY_STATE, SAMPLE_GOAL_STATE)
        self.assertEqual(res.h1_misplaced, 3)
        self.assertEqual(res.h2_manhattan, 3)
        self.assertTrue(res.is_solvable)


class TestIndexMathAndValidation(unittest.TestCase):
    """Test divmod coordinate conversion, flattening, parsing, and input validation."""

    def test_index_to_coords_divmod(self) -> None:
        """Verify divmod maps indices 0..8 to exact (row, col) coordinates."""
        expected = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        for idx, coords in enumerate(expected):
            self.assertEqual(index_to_coords(idx, size=3), coords)

        with self.assertRaises(ValueError):
            index_to_coords(-1, size=3)
        with self.assertRaises(ValueError):
            index_to_coords(9, size=3)
        with self.assertRaises(ValueError):
            index_to_coords(0, size=0)
        with self.assertRaises(TypeError):
            index_to_coords(True, size=3)  # type: ignore[arg-type]

    def test_flatten_unflatten_and_position_map(self) -> None:
        """Round-trip flatten/unflatten and verify O(N) position map."""
        flat = flatten_board(SAMPLE_CURRENT_STATE)
        self.assertEqual(flat, (7, 2, 4, 5, 0, 6, 8, 3, 1))
        self.assertEqual(unflatten_board(flat), SAMPLE_CURRENT_STATE)

        pos_map = build_position_map(SAMPLE_GOAL_STATE)
        self.assertEqual(len(pos_map), 9)
        self.assertEqual(pos_map[0], (0, 0))
        self.assertEqual(pos_map[4], (1, 1))
        self.assertEqual(pos_map[8], (2, 2))

    def test_validation_rejects_malformed_boards(self) -> None:
        """Ensure malformed dimensions, duplicates, out-of-range tiles, and invalid types fail."""
        with self.assertRaises(TypeError):
            validate_board("012345678")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_board([[0, 1, 2], [3, 4, 5]])  # Only 2 rows
        with self.assertRaises(ValueError):
            validate_board([[0, 1], [2, 3, 4], [5, 6, 7]])  # Ragged row
        with self.assertRaises(ValueError):
            validate_board([[0, 1, 2], [3, 4, 5], [6, 7, 7]])  # Duplicate tile 7
        with self.assertRaises(ValueError):
            validate_board([[0, 1, 2], [3, 4, 5], [6, 7, 9]])  # Out of range tile 9
        with self.assertRaises(ValueError):
            validate_board([[-1, 1, 2], [3, 4, 5], [6, 7, 8]])  # Negative tile
        with self.assertRaises(TypeError):
            validate_board([[True, 0, 2], [3, 4, 5], [6, 7, 8]])  # Boolean tile
        with self.assertRaises(TypeError):
            validate_board([[0, 1, "2"], [3, 4, 5], [6, 7, 8]])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            unflatten_board([0, 1, 2, 3])  # Wrong flat length

    def test_parse_board_formats(self) -> None:
        """Verify parse_board accepts matrix literals, semicolons, flat strings, and presets."""
        self.assertEqual(parse_board("[[7,2,4],[5,0,6],[8,3,1]]"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("7 2 4; 5 0 6; 8 3 1"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("7,2,4;5,0,6;8,3,1"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("7 2 4 5 0 6 8 3 1"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("7,2,4,5,0,6,8,3,1"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("sample"), SAMPLE_CURRENT_STATE)
        self.assertEqual(parse_board("standard"), SAMPLE_GOAL_STATE)
        self.assertEqual(parse_board("classic"), CLASSIC_GOAL_STATE)

        with self.assertRaises(ValueError):
            parse_board("")
        with self.assertRaises(ValueError):
            parse_board("a b c; d e f; g h i")
        with self.assertRaises(ValueError):
            parse_board("[[7,2,4")


class TestSolvabilityAndOptionalChallenge(unittest.TestCase):
    """Test inversion count parity, solvability, random board generation, and dominance."""

    def test_inversion_counts_and_solvability(self) -> None:
        """Verify exact inversion counts and solvability parity checks."""
        self.assertEqual(count_inversions(SAMPLE_GOAL_STATE), 0)
        self.assertEqual(count_inversions(SAMPLE_CURRENT_STATE), 16)
        self.assertEqual(count_inversions(UNSOLVABLE_STATE), 1)

        self.assertTrue(is_solvable(SAMPLE_CURRENT_STATE, SAMPLE_GOAL_STATE))
        self.assertTrue(is_solvable(SAMPLE_GOAL_STATE, SAMPLE_GOAL_STATE))
        self.assertFalse(is_solvable(UNSOLVABLE_STATE, SAMPLE_GOAL_STATE))

        # Relative to an odd-parity goal, UNSOLVABLE_STATE becomes solvable!
        odd_goal = ((0, 2, 1), (3, 4, 5), (6, 7, 8))
        self.assertTrue(is_solvable(UNSOLVABLE_STATE, odd_goal))
        self.assertFalse(is_solvable(SAMPLE_GOAL_STATE, odd_goal))

    def test_generate_random_solvable_board_optional_challenge(self) -> None:
        """Verify generated boards are valid, solvable, reproducible with seed, and satisfy h2 >= h1."""
        board_a = generate_random_solvable_board(goal=SAMPLE_GOAL_STATE, seed=123)
        board_b = generate_random_solvable_board(goal=SAMPLE_GOAL_STATE, seed=123)
        self.assertEqual(board_a, board_b)
        self.assertTrue(is_solvable(board_a, SAMPLE_GOAL_STATE))
        self.assertNotEqual(board_a, SAMPLE_GOAL_STATE)

        # Generate 50 boards across both standard and odd-parity goals
        odd_goal = ((1, 2, 3), (4, 5, 6), (8, 7, 0))
        for seed in range(25):
            b_std = generate_random_solvable_board(goal=SAMPLE_GOAL_STATE, seed=seed)
            self.assertTrue(is_solvable(b_std, SAMPLE_GOAL_STATE))
            res = evaluate_heuristics(b_std, SAMPLE_GOAL_STATE)
            self.assertGreaterEqual(res.h2_manhattan, res.h1_misplaced)

            b_odd = generate_random_solvable_board(goal=odd_goal, seed=seed)
            self.assertTrue(is_solvable(b_odd, odd_goal))

    def test_dominance_study(self) -> None:
        """Verify empirical dominance analysis confirms h2 >= h1 across 100% of samples."""
        summary = analyze_dominance_over_samples(num_samples=40, goal=SAMPLE_GOAL_STATE, seed=42)
        self.assertEqual(summary["num_samples"], 40)
        self.assertTrue(summary["all_solvable"])
        self.assertTrue(summary["all_h2_dominates_h1"])
        self.assertGreaterEqual(summary["mean_h2"], summary["mean_h1"])
        self.assertEqual(len(summary["results"]), 40)

        with self.assertRaises(ValueError):
            analyze_dominance_over_samples(num_samples=0)


class TestEvaluatorFormattingAndViva(unittest.TestCase):
    """Test OOP evaluator class, ASCII formatting, JSON serialization, and Viva guide."""

    def test_oop_evaluator_class(self) -> None:
        """Verify EightPuzzleHeuristicEvaluator matches standalone functions."""
        evaluator = EightPuzzleHeuristicEvaluator(SAMPLE_GOAL_STATE)
        self.assertEqual(evaluator.h1(SAMPLE_CURRENT_STATE), 8)
        self.assertEqual(evaluator.h2(SAMPLE_CURRENT_STATE), 18)
        self.assertTrue(evaluator.is_solvable(SAMPLE_CURRENT_STATE))
        self.assertFalse(evaluator.is_solvable(UNSOLVABLE_STATE))

        batch = evaluator.evaluate_batch([SAMPLE_GOAL_STATE, ONE_MOVE_STATE, SAMPLE_CURRENT_STATE])
        self.assertEqual([r.h1_misplaced for r in batch], [0, 1, 8])
        self.assertEqual([r.h2_manhattan for r in batch], [0, 1, 18])

        rand_board = evaluator.generate_random_solvable_board(seed=7)
        self.assertTrue(evaluator.is_solvable(rand_board))

    def test_formatting_ascii_and_json(self) -> None:
        """Verify ASCII rendering, breakdown table, full report, and JSON round-trip."""
        ascii_grid = render_board_ascii(SAMPLE_CURRENT_STATE, title="Current")
        self.assertIn("Current", ascii_grid)
        self.assertIn("| 7 | 2 | 4 |", ascii_grid)
        self.assertIn("| 5 | _ | 6 |", ascii_grid)

        side_by_side = render_side_by_side_boards(SAMPLE_CURRENT_STATE, SAMPLE_GOAL_STATE)
        self.assertIn("Current State", side_by_side)
        self.assertIn("Goal State", side_by_side)
        self.assertIn("--->", side_by_side)

        res = evaluate_heuristics(SAMPLE_CURRENT_STATE, SAMPLE_GOAL_STATE)
        table = format_tile_breakdown_table(res)
        self.assertIn("TOTAL (Blank '0' Excluded)", table)
        self.assertIn("18", table)

        report = format_evaluation_report(res)
        self.assertIn("h1 (Misplaced): 8, h2 (Manhattan): 18", report)
        self.assertIn("SOLVABLE", report)

        payload = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(payload["h1_misplaced"], 8)
        self.assertEqual(payload["h2_manhattan"], 18)
        self.assertEqual(len(payload["tile_details"]), 8)

    def test_viva_reference_completeness(self) -> None:
        """Verify all syllabus Viva topics are present and populated."""
        viva = get_eight_puzzle_viva_reference()
        required_keys = {
            "dominance_of_heuristics",
            "admissibility_of_h1_and_h2",
            "why_blank_space_is_not_counted",
            "solvability_and_inversion_parity",
            "complexity_and_divmod_math",
        }
        self.assertTrue(required_keys.issubset(viva.keys()))
        for entry in viva.values():
            self.assertIn("question", entry)
            self.assertIn("answer", entry)
            self.assertTrue(len(entry["answer"]) > 50)


if __name__ == "__main__":
    unittest.main()
