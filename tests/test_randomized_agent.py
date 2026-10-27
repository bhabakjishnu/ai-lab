"""Unit tests for Randomized Reflex Agent and 1D Grid Environment.

Verifies condition-action reflex mapping, stochastic behavior on clean rooms,
partial observability constraints, grid state transitions, boundary collision
mechanics, reproducibility via random seeds, safety limit cutoffs, and edge cases.
"""

import unittest
from typing import List

from run_randomized_agent import parse_state_argument
from src.randomized_agent import (
    ACTION_CLEAN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ROOM_CLEAN,
    ROOM_DIRTY,
    AgentStepRecord,
    OneDimensionalGridEnvironment,
    RandomizedReflexAgent,
    SimulationResult,
    run_randomized_reflex_simulation,
)


class TestRandomizedReflexAgent(unittest.TestCase):
    """Test the condition-action mapping and stochastic action selection of the agent."""

    def test_dirty_room_percept_always_triggers_clean_action(self) -> None:
        """Verify that a dirty room (1) deterministically results in 'Clean' action."""
        agent = RandomizedReflexAgent(seed=42)
        for _ in range(20):
            action = agent.decide_action(ROOM_DIRTY)
            self.assertEqual(action, ACTION_CLEAN)

    def test_clean_room_percept_triggers_random_navigation(self) -> None:
        """Verify that a clean room (0) triggers stochastic choice between Left and Right."""
        agent = RandomizedReflexAgent(seed=123)
        actions = [agent.decide_action(ROOM_CLEAN) for _ in range(100)]

        # Must only pick Left or Right
        self.assertTrue(all(a in (ACTION_LEFT, ACTION_RIGHT) for a in actions))
        # Must pick both Left and Right across multiple trials (stochastic coverage)
        self.assertIn(ACTION_LEFT, actions)
        self.assertIn(ACTION_RIGHT, actions)

    def test_reproducibility_with_seed(self) -> None:
        """Verify that agents configured with identical seeds produce identical action streams."""
        agent1 = RandomizedReflexAgent(seed=99)
        agent2 = RandomizedReflexAgent(seed=99)

        stream1 = [agent1.decide_action(ROOM_CLEAN) for _ in range(30)]
        stream2 = [agent2.decide_action(ROOM_CLEAN) for _ in range(30)]

        self.assertEqual(stream1, stream2)

    def test_invalid_percept_raises_value_error(self) -> None:
        """Verify that invalid percept values raise ValueError."""
        agent = RandomizedReflexAgent()
        with self.assertRaises(ValueError) as ctx:
            agent.decide_action(2)
        self.assertIn("Invalid room percept", str(ctx.exception))

        with self.assertRaises(ValueError):
            agent.decide_action(-1)

    def test_agent_is_memoryless_and_has_no_global_state(self) -> None:
        """Verify the agent does not maintain memory of past percepts or global array state."""
        agent = RandomizedReflexAgent()
        self.assertFalse(hasattr(agent, "grid"))
        self.assertFalse(hasattr(agent, "history"))
        self.assertFalse(hasattr(agent, "memory"))
        self.assertFalse(hasattr(agent, "visited_rooms"))


class TestOneDimensionalGridEnvironment(unittest.TestCase):
    """Test 1D grid representation, physics, boundary bumps, and percept exposures."""

    def test_default_initialization(self) -> None:
        """Verify default environment setup with standard sample state."""
        env = OneDimensionalGridEnvironment()
        self.assertEqual(env.grid, [1, 0, 0, 1, 1])
        self.assertEqual(env.agent_index, 2)
        self.assertEqual(env.total_rooms, 5)
        self.assertEqual(env.step_count, 0)
        self.assertFalse(env.is_all_clean())

    def test_custom_valid_initialization(self) -> None:
        """Verify initialization with custom state array."""
        env = OneDimensionalGridEnvironment(initial_state=[0, 1, 0], start_index=1)
        self.assertEqual(env.grid, [0, 1, 0])
        self.assertEqual(env.agent_index, 1)

    def test_empty_grid_raises_value_error(self) -> None:
        """Verify that an empty grid sequence triggers ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OneDimensionalGridEnvironment(initial_state=[], start_index=0)
        self.assertIn("Grid state cannot be empty", str(ctx.exception))

    def test_invalid_room_state_values_raise_value_error(self) -> None:
        """Verify that non-binary room states trigger ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OneDimensionalGridEnvironment(initial_state=[1, 0, 2], start_index=0)
        self.assertIn("Rooms must be 0 (Clean) or 1 (Dirty)", str(ctx.exception))

    def test_out_of_bounds_start_index_raises_value_error(self) -> None:
        """Verify that out-of-bounds start index triggers ValueError."""
        with self.assertRaises(ValueError):
            OneDimensionalGridEnvironment(initial_state=[1, 0, 1], start_index=-1)

        with self.assertRaises(ValueError):
            OneDimensionalGridEnvironment(initial_state=[1, 0, 1], start_index=3)

    def test_grid_immutability_defensive_copy(self) -> None:
        """Verify external modification of retrieved grid property does not mutate environment."""
        env = OneDimensionalGridEnvironment(initial_state=[1, 0, 1], start_index=0)
        external_view = env.grid
        external_view[0] = 0
        self.assertEqual(env.grid[0], 1)

    def test_local_percept_isolation(self) -> None:
        """Verify get_percept returns only the current room cleanliness status."""
        env = OneDimensionalGridEnvironment(initial_state=[1, 0, 1], start_index=1)
        self.assertEqual(env.get_percept(), 0)

        # Move to index 0
        agent = RandomizedReflexAgent(seed=42)
        # Force a left move by calling execute_step directly
        env._agent_index = 0
        self.assertEqual(env.get_percept(), 1)

    def test_cleaning_action_transitions(self) -> None:
        """Verify executing 'Clean' action on a dirty room sets it to 0."""
        env = OneDimensionalGridEnvironment(initial_state=[1, 0], start_index=0)
        agent = RandomizedReflexAgent()
        record = env.execute_step(agent)

        self.assertEqual(record.action, ACTION_CLEAN)
        self.assertEqual(env.grid, [0, 0])
        self.assertEqual(env.clean_count, 1)
        self.assertTrue(env.is_all_clean())

    def test_left_boundary_bump_preserves_position(self) -> None:
        """Verify moving Left at Room 0 remains at Room 0 and flags boundary bump."""
        env = OneDimensionalGridEnvironment(initial_state=[0, 0], start_index=0)

        class MockLeftAgent(RandomizedReflexAgent):
            def decide_action(self, percept: int) -> str:
                return ACTION_LEFT

        agent = MockLeftAgent()
        record = env.execute_step(agent)

        self.assertEqual(env.agent_index, 0)
        self.assertTrue(record.is_boundary_bump)
        self.assertEqual(env.move_count, 1)

    def test_right_boundary_bump_preserves_position(self) -> None:
        """Verify moving Right at Room N-1 remains at Room N-1 and flags boundary bump."""
        env = OneDimensionalGridEnvironment(initial_state=[0, 0, 0], start_index=2)

        class MockRightAgent(RandomizedReflexAgent):
            def decide_action(self, percept: int) -> str:
                return ACTION_RIGHT

        agent = MockRightAgent()
        record = env.execute_step(agent)

        self.assertEqual(env.agent_index, 2)
        self.assertTrue(record.is_boundary_bump)
        self.assertEqual(env.move_count, 1)


class TestFullSimulationAndSampleRequirements(unittest.TestCase):
    """Test full simulation runs, sample input/output verification, and boundary conditions."""

    def test_sample_input_with_seed_58_completes_in_exact_12_steps(self) -> None:
        """Verify Sample Input: [1, 0, 0, 1, 1], Start Index: 2 yields 12 steps with seed 58."""
        result = run_randomized_reflex_simulation(
            grid=[1, 0, 0, 1, 1],
            start_index=2,
            seed=58,
        )

        self.assertTrue(result.is_all_clean)
        self.assertEqual(result.total_steps, 12)
        self.assertEqual(result.total_moves, 9)
        self.assertEqual(result.total_cleans, 3)
        self.assertEqual(result.final_grid, [0, 0, 0, 0, 0])
        self.assertFalse(result.stopped_by_limit)

    def test_already_clean_environment_terminates_immediately(self) -> None:
        """Verify that an environment starting with all clean rooms finishes in 0 steps."""
        result = run_randomized_reflex_simulation(grid=[0, 0, 0, 0, 0], start_index=2)
        self.assertTrue(result.is_all_clean)
        self.assertEqual(result.total_steps, 0)
        self.assertEqual(result.total_moves, 0)
        self.assertEqual(result.total_cleans, 0)

    def test_single_dirty_room_environment(self) -> None:
        """Verify single room grid cleaning."""
        result = run_randomized_reflex_simulation(grid=[1], start_index=0)
        self.assertTrue(result.is_all_clean)
        self.assertEqual(result.total_steps, 1)
        self.assertEqual(result.total_cleans, 1)
        self.assertEqual(result.total_moves, 0)

    def test_all_dirty_rooms_environment(self) -> None:
        """Verify all dirty rooms [1, 1, 1, 1, 1] are cleaned eventually."""
        result = run_randomized_reflex_simulation(grid=[1, 1, 1, 1, 1], start_index=0, seed=42)
        self.assertTrue(result.is_all_clean)
        self.assertEqual(result.total_cleans, 5)
        self.assertGreater(result.total_steps, 5)

    def test_max_steps_safety_ceiling(self) -> None:
        """Verify that simulation halts when max_steps limit is exceeded."""
        # Start in clean room with only 2 steps allowed
        result = run_randomized_reflex_simulation(
            grid=[1, 0, 1],
            start_index=1,
            seed=1,
            max_steps=2,
        )
        self.assertEqual(result.total_steps, 2)
        self.assertTrue(result.stopped_by_limit)
        self.assertFalse(result.is_all_clean)

    def test_trace_and_summary_formatting(self) -> None:
        """Verify step trace and summary string outputs."""
        result = run_randomized_reflex_simulation(grid=[1, 0, 1], start_index=1, seed=42)
        summary = result.summary()
        self.assertIn("All rooms clean in", summary)
        self.assertIn("Total Moves Executed:", summary)
        self.assertIn("Total Clean Operations:", summary)

        for record in result.history:
            trace = record.format_trace()
            self.assertIn(f"Step {record.step_number:2d}:", trace)
            self.assertIn("Grid:", trace)


class TestHelperFunctions(unittest.TestCase):
    """Test CLI argument parsing and helper utilities."""

    def test_parse_state_argument_comma_separated(self) -> None:
        """Verify parsing comma-separated room states."""
        parsed = parse_state_argument("1,0,1,1,0")
        self.assertEqual(parsed, [1, 0, 1, 1, 0])

    def test_parse_state_argument_space_separated(self) -> None:
        """Verify parsing space-separated room states."""
        parsed = parse_state_argument("1 0 0 1 1")
        self.assertEqual(parsed, [1, 0, 0, 1, 1])

    def test_parse_state_argument_invalid_tokens_raise_error(self) -> None:
        """Verify invalid tokens trigger ValueError."""
        with self.assertRaises(ValueError):
            parse_state_argument("1,2,0")

        with self.assertRaises(ValueError):
            parse_state_argument("clean dirty")

        with self.assertRaises(ValueError):
            parse_state_argument("   ")


if __name__ == "__main__":
    unittest.main()
