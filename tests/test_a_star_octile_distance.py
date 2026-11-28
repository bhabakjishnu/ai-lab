"""Comprehensive Unit Test Suite for Question 35: A* with Octile Distance.

Validates:
1. 8-directional successor generation (cardinal and diagonal transitions).
2. Octile distance heuristic mathematical precision, admissibility, and consistency.
3. Corner wall squeeze constraint: Cannot move diagonally through two corner walls.
4. Core A* search optimality and sample output: Start (0,0), Goal (2,2) -> Cost: 2.828 (Two diagonal moves).
5. Theoretical viva topics:
   - Why Manhattan distance is inadmissible for 8-way movement.
   - Euclidean vs. Octile distance heuristic comparison and dominance.
6. Optional Challenge: Terminal ASCII rendering, directional arrows, and animation frames.
7. Benchmark mazes (Open 3x3, Corner Pinch, Single Corner, 5x5 Obstacle, Concave Trap, Single Cell).
8. Edge cases and error handling (out of bounds, start/goal on obstacle, unsolvable mazes, empty grids).
9. Dictionary serialization and CLI script compatibility.
"""

from __future__ import annotations

import io
import math
import sys
import unittest

from src.a_star_octile_distance import (
    CONCAVE_TRAP_GOAL,
    CONCAVE_TRAP_MAZE,
    CONCAVE_TRAP_START,
    CORNER_PINCH_GOAL,
    CORNER_PINCH_MAZE,
    CORNER_PINCH_START,
    DEFAULT_5X5_GOAL,
    DEFAULT_5X5_MAZE,
    DEFAULT_5X5_START,
    DEFAULT_DIAGONAL_COST,
    DEFAULT_STRAIGHT_COST,
    DIRECTIONS_8,
    OPEN_3X3_GOAL,
    OPEN_3X3_MAZE,
    OPEN_3X3_START,
    SINGLE_CELL_MAZE,
    SINGLE_CORNER_GOAL,
    SINGLE_CORNER_MAZE,
    SINGLE_CORNER_START,
    VIVA_VOCE_GUIDE,
    AStarOctileComparisonResult,
    AStarOctileResult,
    AStarOctileStepRecord,
    CornerConstraintMode,
    a_star_octile_search,
    animate_octile_search,
    chebyshev_distance,
    compare_octile_vs_heuristics,
    euclidean_distance,
    generate_octile_expansion_frames,
    get_8way_successors,
    manhattan_distance,
    octile_distance,
    parse_grid,
    render_octile_maze,
    zero_heuristic,
)
from scripts.run_a_star_octile_distance import parse_coord, run_sample_output


class TestOctileHeuristicCalculations(unittest.TestCase):
    """Test suite validating Octile distance formula and comparative heuristics."""

    def test_octile_formula_exact_diagonal(self) -> None:
        """Verify pure diagonal Octile distance (0,0) to (2,2) equals 2.828."""
        dist = octile_distance((0, 0), (2, 2))
        self.assertAlmostEqual(dist, 2.828, places=3)
        self.assertEqual(round(dist, 3), 2.828)

    def test_octile_formula_pure_orthogonal(self) -> None:
        """Verify pure orthogonal Octile distance (0,0) to (0,4) equals 4.0."""
        dist = octile_distance((0, 0), (0, 4))
        self.assertEqual(dist, 4.0)

    def test_octile_formula_mixed_diagonal_and_straight(self) -> None:
        """Verify mixed Octile distance: 1 diagonal + 2 straight = 1.414 + 2.0 = 3.414."""
        # (0, 0) to (1, 3): dx = 1, dy = 3. min=1, max=3.
        # h = 1.0 * (3 - 1) + 1.414 * 1 = 2.0 + 1.414 = 3.414
        dist = octile_distance((0, 0), (1, 3))
        self.assertAlmostEqual(dist, 3.414, places=3)

    def test_octile_zero_distance_at_goal(self) -> None:
        """Verify Octile distance from a node to itself is zero."""
        self.assertEqual(octile_distance((3, 3), (3, 3)), 0.0)

    def test_manhattan_inadmissibility_for_8way_movement(self) -> None:
        """Verify Manhattan distance overestimates 8-way true cost (proving inadmissibility)."""
        # Between (0,0) and (2,2):
        # Manhattan assumes 4 steps of cost 1 = 4.0
        # True cost h* is 2 diagonal steps of 1.414 = 2.828
        h_man = manhattan_distance((0, 0), (2, 2))
        h_oct = octile_distance((0, 0), (2, 2))
        self.assertEqual(h_man, 4.0)
        self.assertAlmostEqual(h_oct, 2.828, places=3)
        # Manhattan strictly overestimates true cost!
        self.assertGreater(h_man, h_oct)

    def test_euclidean_underestimates_octile_dominance(self) -> None:
        """Verify Octile dominates Euclidean distance on discrete grid."""
        # For (0,0) to (1,2): dx=1, dy=2
        # Euclidean = sqrt(1+4) = sqrt(5) ~ 2.236
        # Octile = 1.0*(2-1) + 1.414*1 = 2.414
        h_euc = euclidean_distance((0, 0), (1, 2))
        h_oct = octile_distance((0, 0), (1, 2))
        self.assertAlmostEqual(h_euc, math.sqrt(5.0), places=5)
        self.assertAlmostEqual(h_oct, 2.414, places=3)
        # Octile strictly dominates Euclidean on discrete grid
        self.assertGreaterEqual(h_oct, h_euc)

    def test_chebyshev_distance(self) -> None:
        """Verify Chebyshev distance assumes uniform cost 1.0."""
        self.assertEqual(chebyshev_distance((0, 0), (2, 3)), 3.0)


class TestSuccessorGenerationAndCornerConstraint(unittest.TestCase):
    """Test suite validating 8-way successors and corner wall squeeze constraints."""

    def test_8way_successors_in_open_grid(self) -> None:
        """Verify center cell of 3x3 open grid generates all 8 directions."""
        succs = get_8way_successors(OPEN_3X3_MAZE, (1, 1))
        self.assertEqual(len(succs), 8)

        action_names = {act for _, act, _ in succs}
        expected_actions = {
            "Up",
            "Down",
            "Left",
            "Right",
            "Up-Left",
            "Up-Right",
            "Down-Left",
            "Down-Right",
        }
        self.assertEqual(action_names, expected_actions)

        # Verify step costs: 4 straight (1.0) and 4 diagonal (1.414)
        straight_costs = [cost for _, _, cost in succs if cost == DEFAULT_STRAIGHT_COST]
        diagonal_costs = [cost for _, _, cost in succs if cost == DEFAULT_DIAGONAL_COST]
        self.assertEqual(len(straight_costs), 4)
        self.assertEqual(len(diagonal_costs), 4)

    def test_corner_pinch_maze_blocks_diagonal_move(self) -> None:
        """Verify corner pinch: cannot move diagonally through two corner walls."""
        # In CORNER_PINCH_MAZE: (0,1)=1 and (1,0)=1
        # From (0,0), attempting to move diagonally to (1,1):
        # Both adjacent orthogonal neighbors (0,1) and (1,0) are walls!
        succs = get_8way_successors(
            CORNER_PINCH_MAZE,
            (0, 0),
            corner_mode=CornerConstraintMode.NO_DOUBLE_CORNER_WALLS,
        )
        destinations = {coord for coord, _, _ in succs}
        # Neither (0,1) nor (1,0) can be reached (obstacles)
        # And diagonal (1,1) is BLOCKED because of double corner walls!
        self.assertNotIn((1, 1), destinations)
        self.assertEqual(len(succs), 0)

    def test_single_corner_maze_allows_diagonal_move(self) -> None:
        """Verify single corner: moving past 1 corner wall is permitted in NO_DOUBLE_CORNER_WALLS."""
        # In SINGLE_CORNER_MAZE: (0,1)=1 and (1,0)=0 (open)
        # From (0,0), diagonal move to (1,1) has only ONE corner wall!
        succs = get_8way_successors(
            SINGLE_CORNER_MAZE,
            (0, 0),
            corner_mode=CornerConstraintMode.NO_DOUBLE_CORNER_WALLS,
        )
        destinations = {coord for coord, _, _ in succs}
        # (1,1) is reachable because only one corner wall is present
        self.assertIn((1, 1), destinations)
        # (1,0) is also open and reachable
        self.assertIn((1, 0), destinations)
        # (0,1) is an obstacle and not reachable
        self.assertNotIn((0, 1), destinations)

    def test_strict_no_corner_cutting_mode(self) -> None:
        """Verify STRICT_NO_CORNER_CUTTING mode blocks diagonal moves past even a single corner wall."""
        succs = get_8way_successors(
            SINGLE_CORNER_MAZE,
            (0, 0),
            corner_mode=CornerConstraintMode.STRICT_NO_CORNER_CUTTING,
        )
        destinations = {coord for coord, _, _ in succs}
        # (1,1) is BLOCKED in strict mode because (0,1) is a wall
        self.assertNotIn((1, 1), destinations)
        # (1,0) remains reachable
        self.assertIn((1, 0), destinations)

    def test_allow_all_mode(self) -> None:
        """Verify ALLOW_ALL mode permits diagonal moves regardless of corner walls."""
        succs = get_8way_successors(
            CORNER_PINCH_MAZE,
            (0, 0),
            corner_mode=CornerConstraintMode.ALLOW_ALL,
        )
        destinations = {coord for coord, _, _ in succs}
        self.assertIn((1, 1), destinations)


class TestAStarOctileSearchCore(unittest.TestCase):
    """Test suite validating A* Search with Octile distance on standard mazes."""

    def test_syllabus_sample_execution(self) -> None:
        """Verify exact syllabus sample: Start (0,0), Goal (2,2), open grid -> Cost: 2.828 (Two diagonal moves)."""
        res = a_star_octile_search(
            maze=OPEN_3X3_MAZE,
            start=OPEN_3X3_START,
            goal=OPEN_3X3_GOAL,
            record_steps=True,
        )
        self.assertTrue(res.found)
        self.assertEqual(res.path_length, 2)
        self.assertEqual(res.path, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(res.diagonal_moves_count, 2)
        self.assertEqual(res.straight_moves_count, 0)
        self.assertAlmostEqual(res.total_cost, 2.828, places=3)
        self.assertEqual(res.format_cost(3), "2.828")
        self.assertEqual(res.move_breakdown_summary(), "Two diagonal moves")

        # Verify exact syllabus string output format
        sample_out = run_sample_output(quiet=True)
        self.assertEqual(sample_out, "Cost: 2.828 (Two diagonal moves)")

        # Verify step records
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].current_coord, (0, 0))
        self.assertEqual(res.steps[-1].current_coord, (2, 2))

    def test_corner_pinch_maze_unsolvable(self) -> None:
        """Verify search correctly reports no path when trapped by corner pinch walls."""
        res = a_star_octile_search(
            maze=CORNER_PINCH_MAZE,
            start=CORNER_PINCH_START,
            goal=CORNER_PINCH_GOAL,
        )
        self.assertFalse(res.found)
        self.assertEqual(res.path, [])
        self.assertEqual(res.total_cost, 0.0)

    def test_single_corner_maze_navigates_around_obstacle(self) -> None:
        """Verify search successfully glides past single corner wall."""
        res = a_star_octile_search(
            maze=SINGLE_CORNER_MAZE,
            start=SINGLE_CORNER_START,
            goal=SINGLE_CORNER_GOAL,
        )
        self.assertTrue(res.found)
        self.assertEqual(res.path, [(0, 0), (1, 1), (2, 2)])
        self.assertAlmostEqual(res.total_cost, 2.828, places=3)

    def test_canonical_5x5_maze_with_obstacles(self) -> None:
        """Verify A* Octile search finds diagonal shortcuts around vertical obstacle wall."""
        res = a_star_octile_search(
            maze=DEFAULT_5X5_MAZE,
            start=DEFAULT_5X5_START,
            goal=DEFAULT_5X5_GOAL,
        )
        self.assertTrue(res.found)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (4, 4))
        # With 4-way movement, cost was 8.0.
        # With 8-way movement, diagonal shortcuts yield lower cost!
        self.assertLess(res.total_cost, 8.0)
        self.assertGreater(res.diagonal_moves_count, 0)

    def test_concave_trap_maze(self) -> None:
        """Verify A* with Octile distance escapes U-shaped concave trap."""
        res = a_star_octile_search(
            maze=CONCAVE_TRAP_MAZE,
            start=CONCAVE_TRAP_START,
            goal=CONCAVE_TRAP_GOAL,
        )
        self.assertTrue(res.found)
        self.assertEqual(res.path[0], CONCAVE_TRAP_START)
        self.assertEqual(res.path[-1], CONCAVE_TRAP_GOAL)

    def test_single_cell_start_is_goal(self) -> None:
        """Verify start == goal returns immediately with cost 0 and 0 moves."""
        res = a_star_octile_search(
            maze=SINGLE_CELL_MAZE,
            start=(0, 0),
            goal=(0, 0),
        )
        self.assertTrue(res.found)
        self.assertEqual(res.path, [(0, 0)])
        self.assertEqual(res.path_length, 0)
        self.assertEqual(res.total_cost, 0.0)
        self.assertEqual(res.move_breakdown_summary(), "Zero moves (Start is Goal)")


class TestComparativeAnalysisAndViva(unittest.TestCase):
    """Test suite validating multi-heuristic comparison and university viva coverage."""

    def test_multi_heuristic_comparison(self) -> None:
        """Verify comparison across Octile, Euclidean, Manhattan, and Zero/UCS."""
        comp = compare_octile_vs_heuristics(
            OPEN_3X3_MAZE,
            OPEN_3X3_START,
            OPEN_3X3_GOAL,
        )
        self.assertIsInstance(comp, AStarOctileComparisonResult)
        self.assertEqual(len(comp.entries), 4)

        octile_entry = comp.entries[0]
        self.assertEqual(octile_entry.heuristic_name, "Octile (Canonical)")
        self.assertTrue(octile_entry.is_admissible)
        self.assertTrue(octile_entry.is_consistent)

        manhattan_entry = comp.entries[2]
        self.assertEqual(manhattan_entry.heuristic_name, "Manhattan (INADMISSIBLE)")
        self.assertFalse(manhattan_entry.is_admissible)

        ucs_entry = comp.entries[3]
        # UCS expands more nodes than Octile on open grid
        self.assertGreater(ucs_entry.nodes_expanded, octile_entry.nodes_expanded)

        table_str = comp.to_table()
        self.assertIn("Octile (Canonical)", table_str)
        self.assertIn("INADMISSIBLE", table_str)

    def test_viva_guide_contents(self) -> None:
        """Verify presence of key viva topics."""
        self.assertIn("Q1_Manhattan_Inadmissible", VIVA_VOCE_GUIDE)
        self.assertIn("Q2_Euclidean_vs_Octile", VIVA_VOCE_GUIDE)
        self.assertIn("Q3_Corner_Wall_Constraint", VIVA_VOCE_GUIDE)

        ans1 = VIVA_VOCE_GUIDE["Q1_Manhattan_Inadmissible"]["answer"]
        self.assertIn("4.0", ans1)
        self.assertIn("2.828", ans1)

        ans2 = VIVA_VOCE_GUIDE["Q2_Euclidean_vs_Octile"]["answer"]
        self.assertIn("dominates", ans2)


class TestVisualizationAndSerialization(unittest.TestCase):
    """Test suite validating ASCII visualization, animation frames, and JSON serialization."""

    def test_render_octile_maze_glyphs(self) -> None:
        """Verify ASCII rendering contains expected glyphs."""
        res = a_star_octile_search(OPEN_3X3_MAZE, OPEN_3X3_START, OPEN_3X3_GOAL)
        rendered = render_octile_maze(
            OPEN_3X3_MAZE,
            path=res.path,
            explored_set=res.explored_set,
            start=OPEN_3X3_START,
            goal=OPEN_3X3_GOAL,
            use_directional_glyphs=True,
        )
        self.assertIn("S", rendered)
        self.assertIn("G", rendered)
        # Should contain directional arrow for diagonal move (↘)
        self.assertIn("↘", rendered)

    def test_expansion_frame_generator(self) -> None:
        """Verify animation frame generator produces valid step frames."""
        res = a_star_octile_search(
            OPEN_3X3_MAZE,
            OPEN_3X3_START,
            OPEN_3X3_GOAL,
            record_steps=True,
        )
        frames = list(generate_octile_expansion_frames(OPEN_3X3_MAZE, res))
        self.assertGreaterEqual(len(frames), 2)
        self.assertIn("Step 1", frames[0])
        self.assertIn("Goal Reached", frames[-1])

    def test_json_serialization(self) -> None:
        """Verify result dictionary serialization."""
        res = a_star_octile_search(OPEN_3X3_MAZE, OPEN_3X3_START, OPEN_3X3_GOAL)
        d = res.to_dict()
        self.assertEqual(d["start"], [0, 0])
        self.assertEqual(d["goal"], [2, 2])
        self.assertEqual(d["total_cost"], 2.828)
        self.assertEqual(d["diagonal_moves_count"], 2)


class TestGridParserAndErrorHandling(unittest.TestCase):
    """Test suite validating grid parser and error handling."""

    def test_parse_grid_from_string(self) -> None:
        """Verify grid parsing from multiline string."""
        grid_str = "S . .\n. X .\n. . G"
        maze, s, g = parse_grid(grid_str)
        self.assertEqual(s, (0, 0))
        self.assertEqual(g, (2, 2))
        self.assertEqual(maze[1][1], 1)

    def test_parse_coord_helper(self) -> None:
        """Verify parse_coord helper function."""
        self.assertEqual(parse_coord("1,2"), (1, 2))
        self.assertEqual(parse_coord("(3, 4)"), (3, 4))
        self.assertEqual(parse_coord("[0, 5]"), (0, 5))
        with self.assertRaises(ValueError):
            parse_coord("invalid")

    def test_out_of_bounds_error(self) -> None:
        """Verify error raised when start or goal is out of bounds."""
        with self.assertRaises(ValueError):
            a_star_octile_search(OPEN_3X3_MAZE, (-1, 0), (2, 2))
        with self.assertRaises(ValueError):
            a_star_octile_search(OPEN_3X3_MAZE, (0, 0), (5, 5))

    def test_start_or_goal_on_obstacle(self) -> None:
        """Verify error raised when start or goal is on an obstacle."""
        maze = [[1, 0], [0, 0]]
        with self.assertRaises(ValueError):
            a_star_octile_search(maze, (0, 0), (1, 1))

    def test_empty_grid_error(self) -> None:
        """Verify error raised for empty grid."""
        with self.assertRaises(ValueError):
            a_star_octile_search([], (0, 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
