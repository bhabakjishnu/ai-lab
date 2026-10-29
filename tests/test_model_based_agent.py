"""Unit tests for Model-Based Reflex Agent and 2x2 Grid Environment (Question 3).

Verifies internal state maintenance, visited set updates, transition models,
decision rules, goal-directed termination, boundary collision physics,
sample scenario traces, all-clean / all-dirty variations, and optional sensor failure.
"""

import unittest
from typing import List, Tuple

from src.model_based_agent import (
    ACTION_CLEAN,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_SHUTDOWN,
    ACTION_UP,
    STATUS_CLEAN,
    STATUS_DIRTY,
    TERMINATION_MESSAGE,
    ModelBasedAgent,
    TwoByTwoGridEnvironment,
    normalize_status,
    run_model_based_simulation,
)


class TestModelBasedAgentInitialization(unittest.TestCase):
    """Test initialization, attribute defaults, and validation of ModelBasedAgent."""

    def test_default_initialization(self) -> None:
        """Verify default 2x2 grid dimensions and empty initial state."""
        agent = ModelBasedAgent()
        self.assertEqual(agent.grid_dimensions, (2, 2))
        self.assertEqual(agent.current_pos, (0, 0))
        self.assertEqual(agent.visited, set())
        self.assertEqual(agent.model, {})
        self.assertFalse(agent.is_shutdown)
        self.assertEqual(agent.total_rooms, 4)

    def test_custom_dimensions_and_initial_position(self) -> None:
        """Verify custom grid dimensions and starting position."""
        agent = ModelBasedAgent(grid_dimensions=(3, 3), initial_pos=(1, 2))
        self.assertEqual(agent.grid_dimensions, (3, 3))
        self.assertEqual(agent.current_pos, (1, 2))
        self.assertEqual(agent.total_rooms, 9)

    def test_invalid_dimensions_raise_value_error(self) -> None:
        """Verify that zero or negative dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            ModelBasedAgent(grid_dimensions=(0, 2))
        with self.assertRaises(ValueError):
            ModelBasedAgent(grid_dimensions=(2, -1))

    def test_out_of_bounds_initial_pos_raises_value_error(self) -> None:
        """Verify that an initial position outside dimensions raises ValueError."""
        with self.assertRaises(ValueError):
            ModelBasedAgent(grid_dimensions=(2, 2), initial_pos=(2, 0))
        with self.assertRaises(ValueError):
            ModelBasedAgent(grid_dimensions=(2, 2), initial_pos=(-1, 0))


class TestInternalStateAndModelUpdating(unittest.TestCase):
    """Test state update, visited set tracking, and internal model verification."""

    def test_update_state_records_visited_and_model(self) -> None:
        """Verify update_state adds coordinate to visited set and stores status in model."""
        agent = ModelBasedAgent()
        percept = ((0, 0), STATUS_DIRTY)
        agent.update_state(percept)

        self.assertIn((0, 0), agent.visited)
        self.assertEqual(agent.model.get((0, 0)), STATUS_DIRTY)
        self.assertEqual(agent.current_pos, (0, 0))

    def test_visited_set_uniqueness(self) -> None:
        """Verify multiple percepts from the same coordinate do not duplicate in visited set."""
        agent = ModelBasedAgent()
        agent.update_state(((0, 1), STATUS_DIRTY))
        agent.update_state(((0, 1), STATUS_CLEAN))

        self.assertEqual(len(agent.visited), 1)
        self.assertIn((0, 1), agent.visited)
        self.assertEqual(agent.model.get((0, 1)), STATUS_CLEAN)

    def test_transition_model_clean_action(self) -> None:
        """Verify that transition model updates room to Clean after Clean action."""
        agent = ModelBasedAgent()
        agent.update_state(((0, 0), STATUS_DIRTY))
        self.assertEqual(agent.model[(0, 0)], STATUS_DIRTY)

        # Apply transition model with Clean action
        agent.update_state(((0, 0), STATUS_CLEAN), action_executed=ACTION_CLEAN)
        self.assertEqual(agent.model[(0, 0)], STATUS_CLEAN)

    def test_transition_model_directional_movement(self) -> None:
        """Verify directional transition model predictions and boundary enforcement."""
        agent = ModelBasedAgent(grid_dimensions=(2, 2))

        # Valid transitions from (0, 0)
        self.assertEqual(agent.transition_model((0, 0), ACTION_RIGHT), (0, 1))
        self.assertEqual(agent.transition_model((0, 0), ACTION_DOWN), (1, 0))

        # Wall bumps (should stay in place)
        self.assertEqual(agent.transition_model((0, 0), ACTION_UP), (0, 0))
        self.assertEqual(agent.transition_model((0, 0), ACTION_LEFT), (0, 0))

    def test_is_all_clean_in_model_logic(self) -> None:
        """Verify conditions under which internal model confirms complete cleanliness."""
        agent = ModelBasedAgent(grid_dimensions=(2, 2))

        # Case 1: No rooms visited
        self.assertFalse(agent.is_all_clean_in_model())

        # Case 2: Only 3 of 4 rooms visited (even if all 3 are clean)
        agent.model[(0, 0)] = STATUS_CLEAN
        agent.visited.add((0, 0))
        agent.model[(0, 1)] = STATUS_CLEAN
        agent.visited.add((0, 1))
        agent.model[(1, 0)] = STATUS_CLEAN
        agent.visited.add((1, 0))
        self.assertFalse(agent.is_all_clean_in_model())

        # Case 3: All 4 rooms visited, but one is Dirty
        agent.model[(1, 1)] = STATUS_DIRTY
        agent.visited.add((1, 1))
        self.assertFalse(agent.is_all_clean_in_model())

        # Case 4: All 4 rooms visited and all 4 Clean
        agent.model[(1, 1)] = STATUS_CLEAN
        self.assertTrue(agent.is_all_clean_in_model())


class TestAgentDecideAction(unittest.TestCase):
    """Test reflex actions, model-based navigation, and termination decisions."""

    def test_dirty_percept_triggers_clean_action(self) -> None:
        """Verify that perceiving a dirty room triggers Clean action."""
        agent = ModelBasedAgent()
        action = agent.decide_action(((0, 0), STATUS_DIRTY))
        self.assertEqual(action, ACTION_CLEAN)

    def test_clean_percept_triggers_navigation_to_unvisited(self) -> None:
        """Verify agent moves towards unvisited rooms when current room is Clean."""
        agent = ModelBasedAgent()
        action = agent.decide_action(((0, 0), STATUS_CLEAN))
        # From (0, 0), unvisited rooms are (0, 1), (1, 0), (1, 1) -> must move Right or Down
        self.assertIn(action, (ACTION_RIGHT, ACTION_DOWN))

    def test_shutdown_action_when_model_confirms_clean(self) -> None:
        """Verify that agent triggers Shutdown once internal model confirms all clean."""
        agent = ModelBasedAgent(grid_dimensions=(2, 2))
        # Pre-populate all rooms as Clean in model and visited
        for r in range(2):
            for c in range(2):
                agent.visited.add((r, c))
                agent.model[(r, c)] = STATUS_CLEAN

        # Perception of clean at current pos should now trigger Shutdown
        action = agent.decide_action(((1, 1), STATUS_CLEAN))
        self.assertEqual(action, ACTION_SHUTDOWN)
        self.assertTrue(agent.is_shutdown)

        # Subsequent calls while shutdown return NoOp
        self.assertEqual(agent.decide_action(((1, 1), STATUS_CLEAN)), ACTION_NOOP)


class TestTwoByTwoGridEnvironment(unittest.TestCase):
    """Test 2x2 grid environment physics, percept generation, and actions."""

    def test_default_sample_grid_initialization(self) -> None:
        """Verify default sample grid [[Dirty, Clean], [Dirty, Dirty]]."""
        env = TwoByTwoGridEnvironment()
        self.assertEqual(env.dimensions, (2, 2))
        self.assertEqual(env.agent_pos, (0, 0))
        self.assertEqual(
            env.grid,
            [[STATUS_DIRTY, STATUS_CLEAN], [STATUS_DIRTY, STATUS_DIRTY]],
        )
        self.assertFalse(env.is_all_clean())

    def test_execute_clean_action(self) -> None:
        """Verify Clean action modifies actual physical environment state."""
        env = TwoByTwoGridEnvironment()
        self.assertEqual(env.grid[0][0], STATUS_DIRTY)

        env.execute_action(ACTION_CLEAN)
        self.assertEqual(env.grid[0][0], STATUS_CLEAN)
        self.assertEqual(env.clean_count, 1)

    def test_execute_movement_and_boundary_collision(self) -> None:
        """Verify movement updates coordinates and boundaries prevent out-of-bounds."""
        env = TwoByTwoGridEnvironment(start_pos=(0, 0))

        # Move Right to (0, 1)
        env.execute_action(ACTION_RIGHT)
        self.assertEqual(env.agent_pos, (0, 1))

        # Try to move Right again -> bump boundary, remain at (0, 1)
        env.execute_action(ACTION_RIGHT)
        self.assertEqual(env.agent_pos, (0, 1))

        # Move Down to (1, 1)
        env.execute_action(ACTION_DOWN)
        self.assertEqual(env.agent_pos, (1, 1))

    def test_invalid_action_raises_value_error(self) -> None:
        """Verify unrecognized action triggers ValueError."""
        env = TwoByTwoGridEnvironment()
        with self.assertRaises(ValueError):
            env.execute_action("Teleport")

    def test_render_ascii_output(self) -> None:
        """Verify ASCII rendering contains agent mark and cell statuses."""
        env = TwoByTwoGridEnvironment(start_pos=(0, 0))
        rendered = env.render()
        self.assertIn("[*D]", rendered)  # Agent at (0, 0) which is Dirty


class TestModelBasedSimulationTraces(unittest.TestCase):
    """Test complete end-to-end simulation runs on various grid configurations."""

    def test_sample_input_execution_and_termination(self) -> None:
        """Verify the exact sample scenario: [[Dirty, Clean], [Dirty, Dirty]]."""
        sample_grid = [
            [STATUS_DIRTY, STATUS_CLEAN],
            [STATUS_DIRTY, STATUS_DIRTY],
        ]
        env = TwoByTwoGridEnvironment(initial_grid=sample_grid, start_pos=(0, 0))
        agent = ModelBasedAgent(grid_dimensions=(2, 2), initial_pos=(0, 0))

        result = run_model_based_simulation(env, agent)

        # 1. Termination criteria
        self.assertTrue(result.stopped_by_model)
        self.assertEqual(result.termination_message, TERMINATION_MESSAGE)

        # 2. Complete coverage
        self.assertEqual(len(result.visited_rooms), 4)
        self.assertEqual(result.visited_rooms, {(0, 0), (0, 1), (1, 0), (1, 1)})

        # 3. Cleanliness verification
        self.assertTrue(result.is_all_clean_actual)
        self.assertTrue(result.is_all_clean_model)
        self.assertEqual(
            result.final_grid,
            [[STATUS_CLEAN, STATUS_CLEAN], [STATUS_CLEAN, STATUS_CLEAN]],
        )

        # 4. Clean counts: exactly 3 dirty rooms in sample grid
        self.assertEqual(result.total_cleans, 3)

    def test_all_dirty_grid(self) -> None:
        """Verify simulation cleans all 4 rooms when initially all Dirty."""
        all_dirty = [
            [STATUS_DIRTY, STATUS_DIRTY],
            [STATUS_DIRTY, STATUS_DIRTY],
        ]
        env = TwoByTwoGridEnvironment(initial_grid=all_dirty, start_pos=(0, 0))
        agent = ModelBasedAgent()

        result = run_model_based_simulation(env, agent)

        self.assertTrue(result.stopped_by_model)
        self.assertTrue(result.is_all_clean_actual)
        self.assertEqual(result.total_cleans, 4)
        self.assertEqual(len(result.visited_rooms), 4)

    def test_all_clean_grid(self) -> None:
        """Verify simulation visits all rooms and shuts down without cleaning."""
        all_clean = [
            [STATUS_CLEAN, STATUS_CLEAN],
            [STATUS_CLEAN, STATUS_CLEAN],
        ]
        env = TwoByTwoGridEnvironment(initial_grid=all_clean, start_pos=(0, 0))
        agent = ModelBasedAgent()

        result = run_model_based_simulation(env, agent)

        self.assertTrue(result.stopped_by_model)
        self.assertTrue(result.is_all_clean_actual)
        self.assertEqual(result.total_cleans, 0)
        self.assertEqual(len(result.visited_rooms), 4)

    def test_different_start_positions(self) -> None:
        """Verify agent successfully cleans and terminates from all 4 starting corners."""
        grid = [
            [STATUS_DIRTY, STATUS_CLEAN],
            [STATUS_DIRTY, STATUS_DIRTY],
        ]
        start_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for start_pos in start_positions:
            env = TwoByTwoGridEnvironment(initial_grid=grid, start_pos=start_pos)
            agent = ModelBasedAgent(initial_pos=start_pos)

            result = run_model_based_simulation(env, agent)

            self.assertTrue(result.stopped_by_model, f"Failed for start_pos {start_pos}")
            self.assertTrue(result.is_all_clean_actual, f"Not clean for {start_pos}")
            self.assertEqual(len(result.visited_rooms), 4, f"Not all visited for {start_pos}")


class TestSensorFailureChallenge(unittest.TestCase):
    """Test the optional challenge: sensor failure probability where Dirty is perceived as Clean."""

    def test_sensor_failure_percept_generation(self) -> None:
        """Verify 100% sensor failure causes Dirty room to be perceived as Clean."""
        grid = [[STATUS_DIRTY, STATUS_CLEAN], [STATUS_CLEAN, STATUS_CLEAN]]
        env = TwoByTwoGridEnvironment(
            initial_grid=grid,
            start_pos=(0, 0),
            sensor_failure_prob=1.0,
            seed=42,
        )

        pos, status = env.get_percept()
        self.assertEqual(pos, (0, 0))
        # Physically dirty, but sensor reports Clean due to failure
        self.assertEqual(status, STATUS_CLEAN)
        self.assertTrue(env.last_sensor_failed)

    def test_reproducibility_with_sensor_failure_seed(self) -> None:
        """Verify stochastic sensor failures are reproducible using seeds."""
        grid = [[STATUS_DIRTY, STATUS_DIRTY], [STATUS_DIRTY, STATUS_DIRTY]]
        env1 = TwoByTwoGridEnvironment(initial_grid=grid, sensor_failure_prob=0.5, seed=100)
        env2 = TwoByTwoGridEnvironment(initial_grid=grid, sensor_failure_prob=0.5, seed=100)

        percepts1 = [env1.get_percept()[1] for _ in range(10)]
        percepts2 = [env2.get_percept()[1] for _ in range(10)]

        self.assertEqual(percepts1, percepts2)


from scripts.run_model_based_agent import parse_grid_argument


class TestStatusNormalization(unittest.TestCase):
    """Test robust casing and whitespace handling for room statuses."""

    def test_valid_normalizations(self) -> None:
        """Verify clean/dirty string variants normalize to 'Clean' and 'Dirty'."""
        self.assertEqual(normalize_status("clean"), STATUS_CLEAN)
        self.assertEqual(normalize_status("DIRTY"), STATUS_DIRTY)
        self.assertEqual(normalize_status("  Dirty  "), STATUS_DIRTY)

    def test_invalid_status_raises_error(self) -> None:
        """Verify invalid statuses trigger ValueError."""
        with self.assertRaises(ValueError):
            normalize_status("Unknown")
        with self.assertRaises(ValueError):
            normalize_status("123")


class TestCliGridArgumentParsing(unittest.TestCase):
    """Test parsing and expansion of command-line grid strings."""

    def test_parse_default_comma_format(self) -> None:
        """Verify parsing 'Dirty Clean, Dirty Dirty'."""
        grid = parse_grid_argument("Dirty Clean, Dirty Dirty")
        self.assertEqual(
            grid,
            [[STATUS_DIRTY, STATUS_CLEAN], [STATUS_DIRTY, STATUS_DIRTY]],
        )

    def test_parse_shorthand_notation(self) -> None:
        """Verify parsing shorthand 'D C, D D'."""
        grid = parse_grid_argument("D C, D D")
        self.assertEqual(
            grid,
            [[STATUS_DIRTY, STATUS_CLEAN], [STATUS_DIRTY, STATUS_DIRTY]],
        )

    def test_parse_semicolon_format(self) -> None:
        """Verify parsing semicolon delimiter 'Dirty Clean; Clean Dirty'."""
        grid = parse_grid_argument("Dirty Clean; Clean Dirty")
        self.assertEqual(
            grid,
            [[STATUS_DIRTY, STATUS_CLEAN], [STATUS_CLEAN, STATUS_DIRTY]],
        )

    def test_parse_invalid_grid_size_raises_error(self) -> None:
        """Verify invalid grid sizes (e.g. 1x3, 3x3) raise ValueError."""
        with self.assertRaises(ValueError):
            parse_grid_argument("Dirty Clean Dirty")
        with self.assertRaises(ValueError):
            parse_grid_argument("Dirty Clean, Clean")


if __name__ == "__main__":
    unittest.main()

