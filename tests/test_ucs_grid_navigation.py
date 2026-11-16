"""Unit test suite for Question 22: Uniform Cost Search (UCS) Grid Navigation with Cost Modifiers.

Tests cover:
- Character grid parsing into coordinate cost dictionaries
- UCS execution using heapq min-heap priority queue
- 4-directional movement restriction (Up, Down, Left, Right)
- Verification of exact sample input and output format
- Variable step costs (Clear=1, Mountain=5, Water=10)
- Optimal path cost vs shortest hop count (detour verification)
- Edge cases: Start == Goal, 1x1 grid, 1D grids, impassable barriers, out-of-bounds
- Optional Challenge: Diagonal 8-directional movement with Euclidean multiplier
- Corner-cutting prevention
- Step-by-step generator for animations
- Comparative UCS vs BFS analysis
- University Viva Voce reference validation
"""

import math
import unittest
from typing import Dict, List, Set, Tuple

from src.ucs_grid_navigation import (
    DEFAULT_OBSTACLES,
    DEFAULT_TERRAIN_COSTS,
    ORTHOGONAL_DIRECTIONS,
    ALL_8_DIRECTIONS,
    Coordinate,
    TerrainType,
    TopologicalGrid,
    UCSGridResult,
    UCSGridSolver,
    compare_ucs_vs_bfs,
    format_path_coords,
    get_ucs_grid_viva_reference,
    parse_grid_to_cost_dict,
    render_grid_with_path,
    solve_ucs_grid,
)


class TestTopologicalGridParsing(unittest.TestCase):
    """Test parsing character grids into coordinate cost dictionaries and grid management."""

    def test_sample_grid_parsing(self) -> None:
        """Verify Task Requirement 1 on canonical sample grid: ['...', '.M.', '..G']."""
        grid = ["...", ".M.", "..G"]
        cost_dict = parse_grid_to_cost_dict(grid)

        # 3x3 = 9 cells
        self.assertEqual(len(cost_dict), 9)

        # Verify clear ground cells cost 1
        self.assertEqual(cost_dict[(0, 0)], 1)
        self.assertEqual(cost_dict[(0, 1)], 1)
        self.assertEqual(cost_dict[(0, 2)], 1)
        self.assertEqual(cost_dict[(1, 0)], 1)
        self.assertEqual(cost_dict[(1, 2)], 1)
        self.assertEqual(cost_dict[(2, 0)], 1)
        self.assertEqual(cost_dict[(2, 1)], 1)

        # Verify Mountain 'M' costs 5
        self.assertEqual(cost_dict[(1, 1)], 5)

        # Verify Goal 'G' defaults to cost 1
        self.assertEqual(cost_dict[(2, 2)], 1)

    def test_multiline_string_parsing(self) -> None:
        """Verify parsing multiline string with newlines."""
        raw_str = "...\n.M.\n..G"
        cost_dict = parse_grid_to_cost_dict(raw_str)
        self.assertEqual(len(cost_dict), 9)
        self.assertEqual(cost_dict[(1, 1)], 5)

    def test_custom_terrain_costs(self) -> None:
        """Verify custom terrain cost mapping."""
        grid = ["S.M", "W.G"]
        custom_costs = {".": 2, "M": 20, "W": 50, "S": 2, "G": 2}
        cost_dict = parse_grid_to_cost_dict(grid, terrain_costs=custom_costs)

        self.assertEqual(cost_dict[(0, 0)], 2)
        self.assertEqual(cost_dict[(0, 2)], 20)
        self.assertEqual(cost_dict[(1, 0)], 50)
        self.assertEqual(cost_dict[(1, 2)], 2)

    def test_obstacle_cells(self) -> None:
        """Verify obstacle characters '#' and 'X' are mapped to infinity."""
        grid = ["...", ".#.", ".X."]
        cost_dict = parse_grid_to_cost_dict(grid)
        self.assertTrue(math.isinf(cost_dict[(1, 1)]))
        self.assertTrue(math.isinf(cost_dict[(2, 1)]))

    def test_empty_grid_error(self) -> None:
        """Verify empty grid raises ValueError."""
        with self.assertRaises(ValueError):
            TopologicalGrid([])

    def test_detected_start_and_goal(self) -> None:
        """Verify automatic detection of 'S' and 'G' tokens."""
        grid = [".S.", "...", "..G"]
        topo = TopologicalGrid(grid)
        self.assertEqual(topo.detected_start, (0, 1))
        self.assertEqual(topo.detected_goal, (2, 2))


class TestUCSGridSolverSampleAndBasics(unittest.TestCase):
    """Test canonical sample execution and core UCS algorithm properties."""

    def setUp(self) -> None:
        self.sample_grid = ["...", ".M.", "..G"]

    def test_canonical_sample_output(self) -> None:
        """Verify exact sample input and output match problem statement requirements."""
        res = solve_ucs_grid(self.sample_grid, start=(0, 0), goal=(2, 2))

        self.assertTrue(res.success)
        self.assertEqual(res.cost, 4)
        expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        self.assertEqual(res.path, expected_path)
        self.assertEqual(res.formatted_output, "Cost: 4, Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)")
        self.assertEqual(res.path_arrow_string, "(0,0)->(1,0)->(2,0)->(2,1)->(2,2)")

    def test_inferred_start_and_goal(self) -> None:
        """Verify solver infers start at (0, 0) and goal at (2, 2) when None provided."""
        res = solve_ucs_grid(self.sample_grid)
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 4)
        self.assertEqual(res.start, (0, 0))
        self.assertEqual(res.goal, (2, 2))

    def test_start_equals_goal(self) -> None:
        """Verify cost is 0 and path has 1 coordinate when start == goal."""
        res = solve_ucs_grid(self.sample_grid, start=(1, 0), goal=(1, 0))
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 0)
        self.assertEqual(res.path, [(1, 0)])

    def test_movement_restricted_to_orthogonal_4_directions(self) -> None:
        """Verify default solver strictly moves Up, Down, Left, Right (no diagonals)."""
        res = solve_ucs_grid(self.sample_grid, start=(0, 0), goal=(2, 2), allow_diagonal=False)
        for i in range(len(res.path) - 1):
            r1, c1 = res.path[i]
            r2, c2 = res.path[i + 1]
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            # Manhattan step of exactly 1 confirms orthogonal movement
            self.assertEqual(dr + dc, 1)

    def test_1x1_grid(self) -> None:
        """Verify 1x1 grid behavior."""
        res = solve_ucs_grid(["G"], start=(0, 0), goal=(0, 0))
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 0)
        self.assertEqual(res.path, [(0, 0)])

    def test_1d_horizontal_grid(self) -> None:
        """Verify 1D horizontal corridor."""
        res = solve_ucs_grid(["...M.G"], start=(0, 0), goal=(0, 5))
        self.assertTrue(res.success)
        # Costs: (0,1): 1, (0,2): 1, (0,3): 5 (M), (0,4): 1, (0,5): 1 -> Total = 9
        self.assertEqual(res.cost, 9)
        self.assertEqual(len(res.path), 6)

    def test_1d_vertical_grid(self) -> None:
        """Verify 1D vertical corridor."""
        res = solve_ucs_grid([".", ".", "W", ".", "G"], start=(0, 0), goal=(4, 0))
        self.assertTrue(res.success)
        # Costs: (1,0): 1, (2,0): 10 (W), (3,0): 1, (4,0): 1 -> Total = 13
        self.assertEqual(res.cost, 13)


class TestUCSCostOptimalityAndDetours(unittest.TestCase):
    """Test that UCS chooses longer hop paths with lower total cost over direct high-cost paths."""

    def test_mountain_avoidance_detour(self) -> None:
        """Verify UCS takes a 4-hop clear path (cost 4) instead of climbing a mountain (cost 5)."""
        # Grid:
        # S . .
        # . M .
        # . . G
        grid = ["...", ".M.", "..."]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2))
        self.assertEqual(res.cost, 4)
        # Path must NOT step into mountain (1, 1)
        self.assertNotIn((1, 1), res.path)

    def test_water_avoidance_detour(self) -> None:
        """Verify UCS detours around water (cost 10) through clear land (cost 1)."""
        grid = [
            "S.W.G",
            "..W..",
            ".....",
        ]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(0, 4))
        self.assertTrue(res.success)
        # Direct path through water would cost 1 + 10 + 1 + 1 = 13
        # Detour around water costs 8 * 1 = 8
        self.assertEqual(res.cost, 8)
        self.assertNotIn((0, 2), res.path)
        self.assertNotIn((1, 2), res.path)

    def test_forced_mountain_climb_when_blocked(self) -> None:
        """Verify UCS will climb mountain if detour is completely walled off."""
        grid = [
            "S#G",
            ".#.",
            ".M.",
        ]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(0, 2))
        self.assertTrue(res.success)
        # Must go down to (2,0), through (2,1) which is Mountain (cost 5), then up to (0,2)
        self.assertIn((2, 1), res.path)
        # Step costs: (1,0)=1, (2,0)=1, (2,1)=5, (2,2)=1, (1,2)=1, (0,2)=1 -> Total = 10
        self.assertEqual(res.cost, 10)


class TestEdgeCasesAndFailureModes(unittest.TestCase):
    """Test boundary conditions, impassable walls, and exception handling."""

    def test_unreachable_goal_surrounded_by_walls(self) -> None:
        """Verify search fails gracefully when goal is completely enclosed."""
        grid = [
            "S..",
            "###",
            "..G",
        ]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2))
        self.assertFalse(res.success)
        self.assertTrue(math.isinf(res.cost))
        self.assertEqual(res.path, [])
        self.assertEqual(res.formatted_output, "Cost: inf, Path: None")

    def test_start_inside_obstacle(self) -> None:
        """Verify failure when start coordinate is on an obstacle."""
        grid = ["#.", ".."]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(1, 1))
        self.assertFalse(res.success)
        self.assertTrue(math.isinf(res.cost))

    def test_goal_inside_obstacle(self) -> None:
        """Verify failure when goal coordinate is on an obstacle."""
        grid = ["..", ".#"]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(1, 1))
        self.assertFalse(res.success)
        self.assertTrue(math.isinf(res.cost))

    def test_out_of_bounds_start_raises(self) -> None:
        """Verify out-of-bounds start raises ValueError."""
        with self.assertRaises(ValueError):
            solve_ucs_grid(["..."], start=(-1, 0), goal=(0, 2))

    def test_out_of_bounds_goal_raises(self) -> None:
        """Verify out-of-bounds goal raises ValueError."""
        with self.assertRaises(ValueError):
            solve_ucs_grid(["..."], start=(0, 0), goal=(5, 5))


class TestOptionalDiagonalChallenge(unittest.TestCase):
    """Test Optional Challenge: 8-directional diagonal movement with calculated Euclidean multiplier."""

    def test_diagonal_moves_enabled(self) -> None:
        """Verify diagonal moves are utilized when allowed."""
        # Open clear 3x3 grid
        grid = [
            "...",
            "...",
            "...",
        ]
        res_diag = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2), allow_diagonal=True)
        self.assertTrue(res_diag.success)
        # Direct diagonal path: (0,0) -> (1,1) -> (2,2)
        # Two diagonal steps of cost 1 * sqrt(2) ≈ 2.8284
        expected_cost = 2 * math.sqrt(2)
        self.assertAlmostEqual(res_diag.cost, expected_cost, places=4)
        self.assertEqual(res_diag.path, [(0, 0), (1, 1), (2, 2)])

    def test_diagonal_cost_modifier_avoids_mountain(self) -> None:
        """Verify diagonal solver does not blindly enter high-cost mountain cell."""
        # On sample grid: (1,1) is 'M' (cost 5). Diagonal move into (1,1) would cost 5 * sqrt(2) ≈ 7.07.
        # Detour with 1 orthogonal and 1 diagonal through clear ground is cheaper!
        grid = ["...", ".M.", "..G"]
        res_diag = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2), allow_diagonal=True)
        self.assertTrue(res_diag.success)
        # Cost should be 1 + sqrt(2) + 1 ≈ 3.4142 < 4
        self.assertLess(res_diag.cost, 4.0)
        self.assertNotIn((1, 1), res_diag.path)

    def test_corner_cutting_prevention(self) -> None:
        """Verify diagonal move cannot squeeze through touching diagonal walls."""
        # Grid:
        # S # .
        # # . .
        # . . G
        grid = [
            "S#.",
            "#..",
            "..G",
        ]
        solver = UCSGridSolver(allow_diagonal=True, prevent_corner_cutting=True)
        res = solver.search(grid, start=(0, 0), goal=(2, 2))
        # From (0,0), orthogonal to (0,1) is wall '#', orthogonal to (1,0) is wall '#'.
        # With corner-cutting prevention, diagonal to (1,1) is forbidden!
        # Thus (0,0) is trapped!
        self.assertFalse(res.success)


class TestStepByStepGeneratorAndVisualizers(unittest.TestCase):
    """Test step-by-step state space generator, metrics, and visualization utilities."""

    def test_step_by_step_generator_matches_search(self) -> None:
        """Verify step generator yields states and terminates with identical result."""
        grid = ["...", ".M.", "..G"]
        solver = UCSGridSolver()
        gen = solver.step_by_step_generator(grid, start=(0, 0), goal=(2, 2))

        steps = []
        final_res = None
        try:
            while True:
                step_dict = next(gen)
                steps.append(step_dict)
        except StopIteration as e:
            final_res = e.value

        self.assertIsNotNone(final_res)
        self.assertTrue(final_res.success)
        self.assertEqual(final_res.cost, 4)
        self.assertGreater(len(steps), 0)
        self.assertEqual(steps[-1]["current_node"], (2, 2))

    def test_ascii_rendering_contains_expected_tokens(self) -> None:
        """Verify ASCII renderer prints Start, Goal, and path symbols."""
        grid = ["...", ".M.", "..G"]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2))
        ascii_map = render_grid_with_path(grid, res.path, res.explored_set)
        self.assertIn("S", ascii_map)
        self.assertIn("G", ascii_map)
        self.assertIn("+", ascii_map)

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict returns complete and valid JSON dictionary."""
        grid = ["...", ".M.", "..G"]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2))
        d = res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["cost"], 4)
        self.assertEqual(d["start"], [0, 0])
        self.assertEqual(d["goal"], [2, 2])
        self.assertIn("nodes_expanded", d)
        self.assertIn("max_frontier_size", d)

    def test_summary_string(self) -> None:
        """Verify summary returns structured multi-line report."""
        grid = ["...", ".M.", "..G"]
        res = solve_ucs_grid(grid, start=(0, 0), goal=(2, 2))
        summary = res.summary()
        self.assertIn("UNIFORM COST SEARCH GRID RESULT", summary)
        self.assertIn("Optimal Path Found", summary)
        self.assertIn("Path Total Cost        : 4", summary)


class TestComparativeUCSvsBFS(unittest.TestCase):
    """Test comparative analysis demonstrating why UCS is required over BFS for variable costs."""

    def test_ucs_cheaper_than_bfs_on_water_barrier(self) -> None:
        """Verify UCS produces lower cost than BFS when water barrier is present."""
        grid = [
            "S.W.G",
            "..W..",
            ".....",
        ]
        comp = compare_ucs_vs_bfs(grid, start=(0, 0), goal=(0, 4))
        self.assertTrue(comp["is_ucs_cheaper"])
        self.assertEqual(comp["ucs"]["cost"], 8)
        self.assertEqual(comp["bfs"]["cost"], 13)
        self.assertEqual(comp["cost_savings"], 5)
        # BFS takes fewer hops (4) but higher cost (13)
        self.assertLess(comp["bfs"]["hops"], comp["ucs"]["hops"])

    def test_equal_cost_on_uniform_grid(self) -> None:
        """Verify UCS and BFS produce equal path cost on uniform cost grid."""
        grid = [
            "...",
            "...",
            "...",
        ]
        comp = compare_ucs_vs_bfs(grid, start=(0, 0), goal=(2, 2))
        self.assertEqual(comp["ucs"]["cost"], comp["bfs"]["cost"])


class TestVivaReferenceGuide(unittest.TestCase):
    """Test that the university viva voce questions and theory reference are complete."""

    def test_viva_reference_contents(self) -> None:
        """Verify viva guide covers priority queue popping, non-uniform costs, and completeness."""
        viva = get_ucs_grid_viva_reference()
        self.assertEqual(viva["syllabus_area"], "Uninformed Search Techniques")
        questions = viva["questions"]
        self.assertGreaterEqual(len(questions), 4)

        questions_text = " ".join(q["question"] for q in questions)
        self.assertIn("Priority Queue", questions_text)
        self.assertIn("non-uniform", questions_text)
        self.assertIn("Completeness", questions_text)
        self.assertIn("diagonal", questions_text)


if __name__ == "__main__":
    unittest.main()
