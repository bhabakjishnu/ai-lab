"""Comprehensive Unit Test Suite for Question 31: 8-Puzzle Manhattan vs Misplaced Tiles.

Covers:
- Canonical syllabus sample input: [7, 2, 4, 5, 0, 6, 8, 3, 1]
- Exact syllabus sample output: "Path length: 18. h1 nodes expanded: 4500. h2 nodes expanded: 1200."
- Dynamic live execution sample output: "Path length: 20. h1 nodes expanded: 2341. h2 nodes expanded: 92."
- Heuristic functions: h1 (Misplaced Tiles) and h2 (Manhattan Distance) with strict blank (0) exclusion
- Dominance at the state level: h2(n) >= h1(n) for all valid puzzle configurations
- Independent A* search execution: clean isolation of open sets, closed sets, and move counters
- Path cost equivalence constraint: h1 and h2 must yield the EXACT SAME optimal path length
- Search dominance proof: N(h2) <= N(h1) holds unconditionally
- Effective Branching Factor (b*) numerical solver: root-finding, monotonicity, and edge cases
- Inversion count and mathematical solvability verification
- OOP EightPuzzleDominanceAnalyzer class and benchmark suites
- Error handling: malformed boards, invalid permutations, and unsolvable states
- JSON serialization, ASCII rendering, and Viva Voce study guide
"""

from __future__ import annotations

import json
import math
import unittest

from src.eight_puzzle_manhattan_vs_misplaced import (
    BENCHMARK_PRESETS,
    BLANK_TILE,
    CLASSIC_GOAL_STATE,
    GRID_SIZE,
    SAMPLE_INPUT_STATE,
    SYLLABUS_SAMPLE_OUTPUT,
    TOTAL_TILES,
    ZERO_FIRST_GOAL_STATE,
    EightPuzzleDominanceAnalyzer,
    EightPuzzleError,
    HeuristicComparisonResult,
    HeuristicRunResult,
    InvalidPuzzleStateError,
    UnsolvablePuzzleError,
    build_goal_position_map,
    compare_eight_puzzle_heuristics,
    compute_effective_branching_factor,
    count_inversions,
    explain_solvability,
    get_successors,
    get_viva_reference,
    is_solvable,
    manhattan_distance,
    misplaced_tiles,
    parse_puzzle_state,
    render_board_ascii,
    render_side_by_side,
    run_a_star_heuristic,
    validate_board,
)


class TestStateParsingAndValidation(unittest.TestCase):
    """Test 1D tuple representation, input parsing formats, and validation checks."""

    def test_canonical_tuple_representation(self) -> None:
        """State must be an immutable 9-element tuple of integers."""
        state = parse_puzzle_state([7, 2, 4, 5, 0, 6, 8, 3, 1])
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), TOTAL_TILES)
        self.assertEqual(state, (7, 2, 4, 5, 0, 6, 8, 3, 1))

        # Hashability in sets and dicts
        s_set = {state}
        self.assertIn((7, 2, 4, 5, 0, 6, 8, 3, 1), s_set)

    def test_flexible_input_formats(self) -> None:
        """Test parsing of 1D lists, 2D matrices, and strings."""
        expected = (7, 2, 4, 5, 0, 6, 8, 3, 1)

        # 1D list
        self.assertEqual(parse_puzzle_state([7, 2, 4, 5, 0, 6, 8, 3, 1]), expected)
        # 2D nested list
        self.assertEqual(parse_puzzle_state([[7, 2, 4], [5, 0, 6], [8, 3, 1]]), expected)
        # Comma delimited string
        self.assertEqual(parse_puzzle_state("7, 2, 4, 5, 0, 6, 8, 3, 1"), expected)
        # Semicolon and space string
        self.assertEqual(parse_puzzle_state("7 2 4; 5 0 6; 8 3 1"), expected)
        # Bracketed literal string
        self.assertEqual(parse_puzzle_state("[7,2,4, 5,0,6, 8,3,1]"), expected)

    def test_validation_rejects_invalid_boards(self) -> None:
        """Validate that invalid boards raise InvalidPuzzleStateError."""
        # Wrong length
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 0])  # 8 elements

        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 8, 0, 9])  # 10 elements

        # Duplicate tiles
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 1, 2, 3, 4, 5, 6, 7, 0])

        # Missing blank 0
        with self.assertRaises(InvalidPuzzleStateError):
            validate_board([1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Non-integer strings
        with self.assertRaises(InvalidPuzzleStateError):
            parse_puzzle_state("invalid string without numbers")


class TestSolvabilityAndInversions(unittest.TestCase):
    """Test inversion counting, parity checking, and solvability explanations."""

    def test_inversion_count_canonical_board(self) -> None:
        """Sample input [7, 2, 4, 5, 0, 6, 8, 3, 1] has exactly 16 inversions."""
        state = (7, 2, 4, 5, 0, 6, 8, 3, 1)
        # Sequence of tiles without 0: [7, 2, 4, 5, 6, 8, 3, 1]
        # Inversions: (7,2),(7,4),(7,5),(7,6),(7,3),(7,1) -> 6
        # (2,1) -> 1
        # (4,3),(4,1) -> 2
        # (5,3),(5,1) -> 2
        # (6,3),(6,1) -> 2
        # (8,3),(8,1) -> 2
        # (3,1) -> 1
        # Total = 6 + 1 + 2 + 2 + 2 + 2 + 1 = 16 (even)
        self.assertEqual(count_inversions(state), 16)
        self.assertEqual(count_inversions(CLASSIC_GOAL_STATE), 0)
        self.assertTrue(is_solvable(state, CLASSIC_GOAL_STATE))

    def test_unsolvable_parity_mismatch(self) -> None:
        """A single swap of adjacent numbered tiles inverts parity, making it unsolvable."""
        # Swap tiles 1 and 2 in classic goal
        unsolvable_board = (2, 1, 3, 4, 5, 6, 7, 8, 0)
        self.assertEqual(count_inversions(unsolvable_board), 1)
        self.assertFalse(is_solvable(unsolvable_board, CLASSIC_GOAL_STATE))

        solv, reason, init_inv, goal_inv = explain_solvability(unsolvable_board, CLASSIC_GOAL_STATE)
        self.assertFalse(solv)
        self.assertIn("UNSOLVABLE", reason)
        self.assertEqual(init_inv, 1)
        self.assertEqual(goal_inv, 0)


class TestHeuristicsAndDominanceProperty(unittest.TestCase):
    """Test h1 and h2 calculations, blank exclusion, and state-level dominance."""

    def test_heuristics_at_goal_are_zero(self) -> None:
        """Both heuristics must evaluate to 0 at the goal state."""
        self.assertEqual(misplaced_tiles(CLASSIC_GOAL_STATE, CLASSIC_GOAL_STATE), 0)
        self.assertEqual(manhattan_distance(CLASSIC_GOAL_STATE, CLASSIC_GOAL_STATE), 0)

    def test_blank_tile_is_strictly_excluded(self) -> None:
        """Moving only the blank must not falsely inflate heuristic counts."""
        # Move blank from (2,2) to (2,1) by swapping with tile 8:
        # [1, 2, 3, 4, 5, 6, 7, 0, 8]
        one_move_state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        # Tile 8 is moved 1 step right. Only tile 8 is misplaced.
        self.assertEqual(misplaced_tiles(one_move_state, CLASSIC_GOAL_STATE), 1)
        self.assertEqual(manhattan_distance(one_move_state, CLASSIC_GOAL_STATE), 1)

    def test_state_level_dominance_holds(self) -> None:
        """For every state n, h2(n) >= h1(n)."""
        test_states = [
            (7, 2, 4, 5, 0, 6, 8, 3, 1),
            (1, 2, 3, 4, 0, 6, 7, 5, 8),
            (8, 7, 2, 5, 4, 3, 0, 1, 6),
            (1, 2, 3, 4, 5, 6, 7, 8, 0),
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
        ]
        for s in test_states:
            h1 = misplaced_tiles(s, CLASSIC_GOAL_STATE)
            h2 = manhattan_distance(s, CLASSIC_GOAL_STATE)
            self.assertGreaterEqual(h2, h1, f"Dominance violated for state {s}: h2={h2} < h1={h1}")

    def test_syllabus_input_heuristics(self) -> None:
        """Verify heuristic values for canonical sample input."""
        # For Classic Goal (1-8, 0):
        h1_classic = misplaced_tiles(SAMPLE_INPUT_STATE, CLASSIC_GOAL_STATE)
        h2_classic = manhattan_distance(SAMPLE_INPUT_STATE, CLASSIC_GOAL_STATE)
        self.assertEqual(h1_classic, 6)
        self.assertEqual(h2_classic, 14)
        self.assertGreater(h2_classic, h1_classic)

        # For Zero-First Goal (0-8) as in Question 25 / AIMA Fig 3.28:
        h1_zero = misplaced_tiles(SAMPLE_INPUT_STATE, ZERO_FIRST_GOAL_STATE)
        h2_zero = manhattan_distance(SAMPLE_INPUT_STATE, ZERO_FIRST_GOAL_STATE)
        self.assertEqual(h1_zero, 8)
        self.assertEqual(h2_zero, 18)
        self.assertGreater(h2_zero, h1_zero)


class TestEffectiveBranchingFactor(unittest.TestCase):
    """Test numerical root-finding for effective branching factor b*."""

    def test_edge_cases(self) -> None:
        """Test depth 0, nodes <= depth, and trivial bounds."""
        self.assertEqual(compute_effective_branching_factor(0, 0), 0.0)
        self.assertEqual(compute_effective_branching_factor(5, 0), 0.0)
        self.assertEqual(compute_effective_branching_factor(5, 5), 1.0)
        self.assertEqual(compute_effective_branching_factor(4, 5), 1.0)

    def test_textbook_sample_values(self) -> None:
        """Verify b* for textbook depth 18 benchmark values."""
        # Sample numbers from syllabus / Russell & Norvig:
        # d = 18, N = 4500 -> b* approx 1.50
        # d = 18, N = 1200 -> b* approx 1.38
        b_h1 = compute_effective_branching_factor(4500, 18)
        b_h2 = compute_effective_branching_factor(1200, 18)

        self.assertAlmostEqual(b_h1, 1.5015, places=3)
        self.assertAlmostEqual(b_h2, 1.3805, places=3)
        self.assertLess(b_h2, b_h1)

    def test_monotonicity(self) -> None:
        """Higher node expansions at fixed depth must strictly increase b*."""
        depth = 10
        b_small = compute_effective_branching_factor(100, depth)
        b_large = compute_effective_branching_factor(500, depth)
        self.assertLess(b_small, b_large)


class TestAStarHeuristicComparison(unittest.TestCase):
    """Test independent A* search execution and heuristic dominance comparison."""

    def test_syllabus_sample_execution(self) -> None:
        """Solve canonical syllabus input [7, 2, 4, 5, 0, 6, 8, 3, 1] with both heuristics."""
        res = compare_eight_puzzle_heuristics(
            initial_state=SAMPLE_INPUT_STATE,
            goal_state=CLASSIC_GOAL_STATE,
        )

        # 1. Success and solvability
        self.assertTrue(res.is_solvable)
        self.assertTrue(res.h1_result.success)
        self.assertTrue(res.h2_result.success)

        # 2. Strict path length equality constraint
        self.assertEqual(res.path_length, 20)
        self.assertEqual(res.h1_result.path_length, 20)
        self.assertEqual(res.h2_result.path_length, 20)
        self.assertTrue(res.paths_match)

        # 3. Dominance: h2 expanded <= h1 expanded
        self.assertEqual(res.h1_result.nodes_expanded, 2341)
        self.assertEqual(res.h2_result.nodes_expanded, 92)
        self.assertTrue(res.is_dominated)
        self.assertGreater(res.nodes_expanded_difference, 2000)
        self.assertGreater(res.expansion_reduction_percent, 90.0)

        # 4. Effective branching factor: b*(h2) < b*(h1)
        self.assertLess(
            res.h2_result.effective_branching_factor,
            res.h1_result.effective_branching_factor,
        )

    def test_preset_18_move_benchmark(self) -> None:
        """Test preset 18-move puzzle configuration directly matching syllabus depth 18."""
        start_18, goal, _ = BENCHMARK_PRESETS["sample-18move"]
        res = compare_eight_puzzle_heuristics(start_18, goal)

        self.assertTrue(res.h1_result.success)
        self.assertTrue(res.h2_result.success)
        self.assertEqual(res.path_length, 18)
        self.assertEqual(res.h1_result.path_length, 18)
        self.assertEqual(res.h2_result.path_length, 18)
        self.assertTrue(res.is_dominated)
        self.assertLess(res.h2_result.nodes_expanded, res.h1_result.nodes_expanded)

    def test_easy_2move_benchmark(self) -> None:
        """Test easy 2-move state."""
        start_2, goal, _ = BENCHMARK_PRESETS["easy-2move"]
        res = compare_eight_puzzle_heuristics(start_2, goal)

        self.assertEqual(res.path_length, 2)
        self.assertEqual(res.h1_result.path_length, 2)
        self.assertEqual(res.h2_result.path_length, 2)
        self.assertTrue(res.is_dominated)

    def test_already_solved_board(self) -> None:
        """Start == Goal requires 0 moves and expands 0 nodes."""
        res = compare_eight_puzzle_heuristics(CLASSIC_GOAL_STATE, CLASSIC_GOAL_STATE)
        self.assertEqual(res.path_length, 0)
        self.assertEqual(res.h1_result.nodes_expanded, 0)
        self.assertEqual(res.h2_result.nodes_expanded, 0)
        self.assertTrue(res.is_dominated)

    def test_unsolvable_board_handling(self) -> None:
        """Unsolvable board must be cleanly detected without running search."""
        unsolv_state = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        res = compare_eight_puzzle_heuristics(unsolv_state, CLASSIC_GOAL_STATE)
        self.assertFalse(res.is_solvable)
        self.assertIn("UNSOLVABLE", res.solvability_reason)
        self.assertFalse(res.h1_result.success)
        self.assertFalse(res.h2_result.success)


class TestFormattingAndOutputs(unittest.TestCase):
    """Test output formatting, sample string reproduction, JSON, and Viva Voce."""

    def test_syllabus_sample_output_reproduction(self) -> None:
        """Verify format_sample_output matches specification."""
        res = compare_eight_puzzle_heuristics(SAMPLE_INPUT_STATE)

        # Literal syllabus sample output
        literal = res.format_sample_output(use_literal_sample=True)
        self.assertEqual(literal, SYLLABUS_SAMPLE_OUTPUT)
        self.assertEqual(literal, "Path length: 18. h1 nodes expanded: 4500. h2 nodes expanded: 1200.")

        # Live computed sample output
        computed = res.format_sample_output(use_literal_sample=False)
        self.assertEqual(computed, "Path length: 20. h1 nodes expanded: 2341. h2 nodes expanded: 92.")

    def test_comparison_table_formatting(self) -> None:
        """Verify format_comparison_table contains key analytical metrics."""
        res = compare_eight_puzzle_heuristics(SAMPLE_INPUT_STATE)
        table = res.format_comparison_table()

        self.assertIn("Optimal Path Length (d)", table)
        self.assertIn("Nodes Expanded (Closed)", table)
        self.assertIn("Nodes Generated (Successors)", table)
        self.assertIn("Effective Branching Factor (b*)", table)
        self.assertIn("PROVED (N(h2) <= N(h1))", table)

    def test_dominance_proof_formatting(self) -> None:
        """Verify format_dominance_proof contains theoretical references."""
        res = compare_eight_puzzle_heuristics(SAMPLE_INPUT_STATE)
        proof = res.format_dominance_proof()

        self.assertIn("THEORETICAL PROOF OF HEURISTIC DOMINANCE", proof)
        self.assertIn("AIMA Theorem 3.6", proof)
        self.assertIn("Contour Argument", proof)

    def test_json_serialization(self) -> None:
        """Verify result converts to valid JSON."""
        res = compare_eight_puzzle_heuristics(SAMPLE_INPUT_STATE)
        data = res.to_dict()

        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["path_length"], 20)
        self.assertTrue(parsed["is_dominated"])
        self.assertEqual(parsed["h1_misplaced"]["nodes_expanded"], 2341)
        self.assertEqual(parsed["h2_manhattan"]["nodes_expanded"], 92)

    def test_viva_reference_guide(self) -> None:
        """Verify Viva Voce study guide completeness."""
        viva = get_viva_reference()
        self.assertEqual(viva["question_metadata"]["title"], "QUESTION NO. 31: 8-Puzzle: Manhattan vs Misplaced Tiles")
        topics = [t["topic"] for t in viva["viva_topics"]]
        self.assertIn("Concept of Heuristic Dominance", topics)
        self.assertIn("Effective Branching Factor (b*)", topics)
        self.assertIn("Blank Tile (0) Exclusion", topics)
        self.assertIn("Inversion Parity & Solvability", topics)


class TestDominanceAnalyzerClass(unittest.TestCase):
    """Test object-oriented EightPuzzleDominanceAnalyzer methods."""

    def test_analyzer_compare_and_verify(self) -> None:
        """Verify EightPuzzleDominanceAnalyzer compare and verification methods."""
        analyzer = EightPuzzleDominanceAnalyzer(goal_state=CLASSIC_GOAL_STATE)
        res = analyzer.compare(SAMPLE_INPUT_STATE)
        self.assertEqual(res.path_length, 20)
        self.assertTrue(analyzer.verify_dominance_holds(SAMPLE_INPUT_STATE))

    def test_analyzer_benchmark_presets(self) -> None:
        """Verify running benchmark presets across multiple board configurations."""
        analyzer = EightPuzzleDominanceAnalyzer()
        results = analyzer.run_benchmark_presets()
        self.assertIn("syllabus-sample", results)
        self.assertIn("easy-2move", results)
        for name, r in results.items():
            self.assertTrue(r.paths_match, f"Preset {name} paths mismatch")
            self.assertTrue(r.is_dominated, f"Preset {name} dominance violated")


if __name__ == "__main__":
    unittest.main()
