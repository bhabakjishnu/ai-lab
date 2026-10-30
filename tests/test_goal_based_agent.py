"""Unit tests for Goal-Based Navigation Agent and 5x5 Grid Environment (Question 4).

Verifies Manhattan and Euclidean distance calculations, grid boundary validation,
successor generation, explicit diagonal movement prevention (Optional Challenge),
strictly distance-reducing action selection, sample scenario path reproducibility,
boundary corner navigation, and visual grid rendering.
"""

import math
import unittest
from typing import List, Tuple

from src.goal_based_agent import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    DIAGONAL_DELTAS,
    ORTHOGONAL_DELTAS,
    GoalBasedNavigationAgent,
    GoalSimulationResult,
    GridEnvironment,
    NavigationStepRecord,
    euclidean_distance,
    generate_random_coordinate,
    get_successors,
    is_diagonal_movement,
    manhattan_distance,
    run_goal_based_simulation,
    validate_grid_boundary,
    validate_orthogonal_transition,
)


class TestDistanceMetrics(unittest.TestCase):
    """Test suite for Manhattan (L1) and Euclidean (L2) distance functions."""

    def test_manhattan_distance_sample_coordinates(self) -> None:
        """Verify Manhattan distance calculation on standard sample input (0,0) -> (3,4)."""
        dist = manhattan_distance((0, 0), (3, 4))
        self.assertEqual(dist, 7)

    def test_manhattan_distance_zero_when_identical(self) -> None:
        """Verify distance between identical coordinates is 0."""
        self.assertEqual(manhattan_distance((2, 2), (2, 2)), 0)

    def test_manhattan_distance_symmetry(self) -> None:
        """Verify distance metric symmetry d(A, B) == d(B, A)."""
        p1 = (1, 4)
        p2 = (4, 1)
        self.assertEqual(manhattan_distance(p1, p2), manhattan_distance(p2, p1))

    def test_manhattan_distance_triangle_inequality(self) -> None:
        """Verify triangle inequality d(A, C) <= d(A, B) + d(B, C)."""
        a = (0, 0)
        b = (2, 4)
        c = (4, 1)
        self.assertLessEqual(
            manhattan_distance(a, c),
            manhattan_distance(a, b) + manhattan_distance(b, c),
        )

    def test_euclidean_distance_sample_coordinates(self) -> None:
        """Verify Euclidean distance for 3-4-5 Pythagorean triangle from (0,0) to (3,4)."""
        dist = euclidean_distance((0, 0), (3, 4))
        self.assertAlmostEqual(dist, 5.0, places=5)

    def test_euclidean_less_than_or_equal_to_manhattan(self) -> None:
        """Verify mathematical theorem: d_E <= d_M holds across test coordinate pairs."""
        coords = [(0, 0), (1, 2), (3, 4), (4, 4), (0, 4), (4, 0)]
        for c1 in coords:
            for c2 in coords:
                d_m = manhattan_distance(c1, c2)
                d_e = euclidean_distance(c1, c2)
                self.assertLessEqual(d_e, d_m)

    def test_invalid_coordinate_types_raise_type_error(self) -> None:
        """Verify non-tuples or non-integers raise TypeError."""
        with self.assertRaises(TypeError):
            manhattan_distance([0, 0], (1, 1))  # type: ignore
        with self.assertRaises(TypeError):
            manhattan_distance((0, 0, 0), (1, 1))  # type: ignore
        with self.assertRaises(TypeError):
            manhattan_distance((0.5, 1.0), (1, 1))  # type: ignore
        with self.assertRaises(TypeError):
            manhattan_distance((True, 0), (1, 1))  # type: ignore


class TestGridBoundaryValidation(unittest.TestCase):
    """Test suite for coordinate boundary verification within grid limits."""

    def test_valid_coordinates_pass_validation(self) -> None:
        """Verify valid 5x5 coordinates pass validation without error."""
        valid_coords = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
        for coord in valid_coords:
            validate_grid_boundary(coord, (5, 5))

    def test_out_of_bounds_coordinates_raise_value_error(self) -> None:
        """Verify out-of-bounds coordinates raise ValueError."""
        invalid_coords = [(-1, 0), (0, -1), (5, 0), (0, 5), (5, 5), (10, 2)]
        for coord in invalid_coords:
            with self.assertRaises(ValueError):
                validate_grid_boundary(coord, (5, 5))

    def test_generate_random_coordinate_within_bounds(self) -> None:
        """Verify randomly generated coordinates lie strictly within 5x5 bounds."""
        for seed in range(20):
            coord = generate_random_coordinate(grid_dimensions=(5, 5), seed=seed)
            self.assertTrue(0 <= coord[0] < 5)
            self.assertTrue(0 <= coord[1] < 5)

    def test_generate_random_coordinate_respects_exclude_list(self) -> None:
        """Verify excluded coordinates are never generated."""
        exclude = [(0, 0), (0, 1), (0, 2)]
        for seed in range(30):
            coord = generate_random_coordinate(grid_dimensions=(5, 5), exclude=exclude, seed=seed)
            self.assertNotIn(coord, exclude)

    def test_generate_random_coordinate_fully_excluded_raises_error(self) -> None:
        """Verify ValueError raised when all grid positions are excluded."""
        all_cells = [(r, c) for r in range(2) for c in range(2)]
        with self.assertRaises(ValueError):
            generate_random_coordinate(grid_dimensions=(2, 2), exclude=all_cells)


class TestSuccessorFunctionAndOptionalChallenge(unittest.TestCase):
    """Test suite for successor generation and explicit diagonal movement rejection."""

    def test_is_diagonal_movement_detection(self) -> None:
        """Verify diagonal vectors return True while orthogonal vectors return False."""
        self.assertTrue(is_diagonal_movement((1, 1)))
        self.assertTrue(is_diagonal_movement((-1, -1)))
        self.assertTrue(is_diagonal_movement((1, -1)))
        self.assertTrue(is_diagonal_movement((-1, 1)))

        self.assertFalse(is_diagonal_movement((1, 0)))
        self.assertFalse(is_diagonal_movement((-1, 0)))
        self.assertFalse(is_diagonal_movement((0, 1)))
        self.assertFalse(is_diagonal_movement((0, -1)))
        self.assertFalse(is_diagonal_movement((0, 0)))

    def test_validate_orthogonal_transition_rejects_diagonals(self) -> None:
        """Optional Challenge: verify diagonal transitions raise ValueError."""
        diagonals = [
            ((2, 2), (1, 1)),
            ((2, 2), (1, 3)),
            ((2, 2), (3, 1)),
            ((2, 2), (3, 3)),
        ]
        for src, dst in diagonals:
            with self.assertRaises(ValueError) as ctx:
                validate_orthogonal_transition(src, dst)
            self.assertIn("Diagonal movement prohibited", str(ctx.exception))

    def test_validate_orthogonal_transition_rejects_step_size_greater_than_one(self) -> None:
        """Verify transitions jumping more than 1 unit raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_orthogonal_transition((0, 0), (2, 0))
        self.assertIn("exceeding single-step limit", str(ctx.exception))

    def test_get_successors_from_corner_zero_zero(self) -> None:
        """Verify successors from top-left corner (0, 0) are strictly Down and Right."""
        successors = get_successors((0, 0), grid_dimensions=(5, 5))
        actions = [act for act, _ in successors]
        coords = [coord for _, coord in successors]

        self.assertEqual(len(successors), 2)
        self.assertIn(ACTION_DOWN, actions)
        self.assertIn(ACTION_RIGHT, actions)
        self.assertIn((1, 0), coords)
        self.assertIn((0, 1), coords)

    def test_get_successors_from_corner_four_four(self) -> None:
        """Verify successors from bottom-right corner (4, 4) are strictly Up and Left."""
        successors = get_successors((4, 4), grid_dimensions=(5, 5))
        actions = [act for act, _ in successors]
        coords = [coord for _, coord in successors]

        self.assertEqual(len(successors), 2)
        self.assertIn(ACTION_UP, actions)
        self.assertIn(ACTION_LEFT, actions)
        self.assertIn((3, 4), coords)
        self.assertIn((4, 3), coords)

    def test_get_successors_from_center(self) -> None:
        """Verify successors from center (2, 2) produce 4 orthogonal neighbors."""
        successors = get_successors((2, 2), grid_dimensions=(5, 5))
        self.assertEqual(len(successors), 4)
        for _, coord in successors:
            self.assertEqual(manhattan_distance((2, 2), coord), 1)


class TestGoalBasedNavigationAgent(unittest.TestCase):
    """Test suite for GoalBasedNavigationAgent decision making and navigation."""

    def test_sample_scenario_exact_path_reproducibility(self) -> None:
        """Verify exact expected path for Sample Input: Start (0, 0), Goal (3, 4).

        Sample Output requirement:
        Path: [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2), (3,3), (3,4)]
        """
        agent = GoalBasedNavigationAgent(goal_pos=(3, 4), grid_dimensions=(5, 5))
        result = agent.navigate(start_pos=(0, 0))

        expected_path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
        self.assertEqual(result.path, expected_path)
        self.assertEqual(result.total_steps, 7)
        self.assertTrue(result.goal_reached)
        self.assertEqual(
            result.actions_taken,
            [ACTION_DOWN, ACTION_DOWN, ACTION_DOWN, ACTION_RIGHT, ACTION_RIGHT, ACTION_RIGHT, ACTION_RIGHT],
        )

    def test_each_step_strictly_reduces_manhattan_distance(self) -> None:
        """Verify every step taken strictly reduces the Manhattan distance to the goal."""
        agent = GoalBasedNavigationAgent(goal_pos=(3, 4), grid_dimensions=(5, 5))
        result = agent.navigate(start_pos=(0, 0))

        self.assertEqual(len(result.history), 7)
        for record in result.history:
            self.assertTrue(record.strictly_reduces_distance)
            self.assertEqual(record.distance_after, record.distance_before - 1)

    def test_path_length_equals_distance_plus_one(self) -> None:
        """Verify total coordinates in path equals Manhattan distance + 1."""
        test_pairs = [
            ((0, 0), (3, 4)),
            ((1, 1), (4, 2)),
            ((4, 4), (0, 0)),
            ((0, 4), (4, 0)),
        ]
        for start, goal in test_pairs:
            agent = GoalBasedNavigationAgent(goal_pos=goal, grid_dimensions=(5, 5))
            result = agent.navigate(start_pos=start)
            expected_dist = manhattan_distance(start, goal)
            self.assertEqual(result.total_steps, expected_dist)
            self.assertEqual(len(result.path), expected_dist + 1)
            self.assertEqual(result.path[0], start)
            self.assertEqual(result.path[-1], goal)

    def test_start_already_at_goal(self) -> None:
        """Verify immediate zero-step success when agent starts at goal."""
        agent = GoalBasedNavigationAgent(goal_pos=(2, 2), grid_dimensions=(5, 5))
        result = agent.navigate(start_pos=(2, 2))

        self.assertEqual(result.path, [(2, 2)])
        self.assertEqual(result.total_steps, 0)
        self.assertTrue(result.goal_reached)
        self.assertEqual(result.actions_taken, [])
        self.assertEqual(result.history, [])

    def test_reverse_navigation_up_and_left(self) -> None:
        """Verify navigation in negative direction from (4, 4) to (0, 0)."""
        agent = GoalBasedNavigationAgent(goal_pos=(0, 0), grid_dimensions=(5, 5))
        result = agent.navigate(start_pos=(4, 4))

        self.assertEqual(result.path[0], (4, 4))
        self.assertEqual(result.path[-1], (0, 0))
        self.assertEqual(result.total_steps, 8)
        self.assertTrue(result.goal_reached)
        # All actions should be either Up or Left
        self.assertTrue(all(a in (ACTION_UP, ACTION_LEFT) for a in result.actions_taken))

    def test_invalid_start_position_raises_value_error(self) -> None:
        """Verify invalid start position raises ValueError."""
        agent = GoalBasedNavigationAgent(goal_pos=(3, 4), grid_dimensions=(5, 5))
        with self.assertRaises(ValueError):
            agent.navigate(start_pos=(5, 0))
        with self.assertRaises(ValueError):
            agent.navigate(start_pos=(-1, 2))

    def test_invalid_goal_position_raises_value_error(self) -> None:
        """Verify invalid goal position during agent init raises ValueError."""
        with self.assertRaises(ValueError):
            GoalBasedNavigationAgent(goal_pos=(5, 5), grid_dimensions=(5, 5))


class TestGridEnvironment(unittest.TestCase):
    """Test suite for GridEnvironment matrix simulation."""

    def test_environment_initialization_with_fixed_goal(self) -> None:
        """Verify fixed goal placement in GridEnvironment."""
        env = GridEnvironment(dimensions=(5, 5), goal_pos=(3, 4), agent_pos=(0, 0))
        self.assertEqual(env.agent_pos, (0, 0))
        self.assertEqual(env.goal_pos, (3, 4))
        self.assertFalse(env.is_goal_reached)

    def test_environment_apply_action(self) -> None:
        """Verify action execution updates agent position correctly."""
        env = GridEnvironment(dimensions=(5, 5), goal_pos=(3, 4), agent_pos=(0, 0))
        new_pos = env.apply_action(ACTION_DOWN)
        self.assertEqual(new_pos, (1, 0))
        self.assertEqual(env.agent_pos, (1, 0))

    def test_environment_boundary_collision_raises_value_error(self) -> None:
        """Verify attempting to move outside boundaries raises ValueError."""
        env = GridEnvironment(dimensions=(5, 5), goal_pos=(3, 4), agent_pos=(0, 0))
        with self.assertRaises(ValueError):
            env.apply_action(ACTION_UP)
        with self.assertRaises(ValueError):
            env.apply_action(ACTION_LEFT)


class TestSimulationRunnerAndVisualization(unittest.TestCase):
    """Test high-level runner and visual rendering utilities."""

    def test_run_goal_based_simulation_default(self) -> None:
        """Verify high-level simulation runner produces expected sample output."""
        result = run_goal_based_simulation(start_pos=(0, 0), goal_pos=(3, 4))
        self.assertTrue(result.goal_reached)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (3, 4))

    def test_run_goal_based_simulation_random_goal_with_seed(self) -> None:
        """Verify deterministic random goal placement with seed."""
        res1 = run_goal_based_simulation(start_pos=(0, 0), random_goal=True, seed=42)
        res2 = run_goal_based_simulation(start_pos=(0, 0), random_goal=True, seed=42)
        self.assertEqual(res1.goal_pos, res2.goal_pos)
        self.assertEqual(res1.path, res2.path)

    def test_render_grid_contains_expected_tokens(self) -> None:
        """Verify rendered grid contains Start 'A', Goal 'G', path '*', and '.'."""
        result = run_goal_based_simulation(start_pos=(0, 0), goal_pos=(3, 4))
        grid_str = result.render_grid()
        self.assertIn("A", grid_str)
        self.assertIn("G", grid_str)
        self.assertIn("*", grid_str)
        self.assertIn(".", grid_str)

    def test_render_grid_coincident_start_and_goal(self) -> None:
        """Verify coincident start and goal rendered with 'S'."""
        result = run_goal_based_simulation(start_pos=(2, 2), goal_pos=(2, 2))
        grid_str = result.render_grid()
        self.assertIn("S", grid_str)

    def test_navigation_step_record_formatting(self) -> None:
        """Verify formatted trace log string for NavigationStepRecord."""
        record = NavigationStepRecord(
            step_number=1,
            position=(0, 0),
            action=ACTION_DOWN,
            next_position=(1, 0),
            distance_before=7,
            distance_after=6,
            path_so_far=((0, 0), (1, 0)),
        )
        trace = record.format_trace()
        self.assertIn("Step  1", trace)
        self.assertIn("Down", trace)
        self.assertIn("Decreased", trace)


if __name__ == "__main__":
    unittest.main()
