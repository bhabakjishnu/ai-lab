"""Unit tests for Simple Reflex Vacuum Cleaner Agent and Environment.

Tests condition-action mapping rules, memoryless property, environment state transitions,
discrete time-step simulation traces, boundary validation, and 3x3 grid extension.
"""

import unittest
from src.vacuum_agent import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_SUCK,
    ACTION_UP,
    CLEAN,
    DIRTY,
    LOCATION_A,
    LOCATION_B,
    GridVacuumEnvironment,
    SimpleReflexVacuumAgent,
    StepRecord,
    TwoSquareEnvironment,
)


class TestSimpleReflexAgentRules(unittest.TestCase):
    """Test the condition-action mapping rules of the Simple Reflex Agent."""

    def test_dirty_square_triggers_suck_action(self) -> None:
        """Verify that any dirty square immediately yields 'Suck' action."""
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule(LOCATION_A, DIRTY), ACTION_SUCK)
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule(LOCATION_B, DIRTY), ACTION_SUCK)

    def test_clean_square_navigation_rules(self) -> None:
        """Verify navigation rules when the current square is clean."""
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule(LOCATION_A, CLEAN), ACTION_RIGHT)
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule(LOCATION_B, CLEAN), ACTION_LEFT)

    def test_case_insensitive_input_handling(self) -> None:
        """Verify robust normalization of location and status casing."""
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule("a", "dirty"), ACTION_SUCK)
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule("b", "clean"), ACTION_LEFT)
        self.assertEqual(SimpleReflexVacuumAgent.reflex_rule("  A  ", " Clean "), ACTION_RIGHT)

    def test_invalid_status_raises_value_error(self) -> None:
        """Verify that unrecognized status triggers ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SimpleReflexVacuumAgent.reflex_rule(LOCATION_A, "UnknownStatus")
        self.assertIn("Invalid environment status", str(ctx.exception))

    def test_invalid_location_raises_value_error(self) -> None:
        """Verify that unrecognized location triggers ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SimpleReflexVacuumAgent.reflex_rule("C", CLEAN)
        self.assertIn("Invalid location", str(ctx.exception))

    def test_agent_is_strictly_memoryless(self) -> None:
        """Verify the agent does not retain internal state between invocations."""
        agent = SimpleReflexVacuumAgent()
        # Call with location A, Dirty multiple times
        action1 = agent.reflex_rule(LOCATION_A, DIRTY)
        action2 = agent.reflex_rule(LOCATION_A, DIRTY)
        self.assertEqual(action1, action2)

        # Agent has no instance variables tracking history
        self.assertFalse(hasattr(agent, "history"))
        self.assertFalse(hasattr(agent, "memory"))
        self.assertFalse(hasattr(agent, "past_percepts"))


class TestTwoSquareEnvironment(unittest.TestCase):
    """Test state dictionary representation, transitions, and perceptual interface."""

    def test_default_environment_initialization(self) -> None:
        """Verify default starting location and all-dirty initial state."""
        env = TwoSquareEnvironment()
        self.assertEqual(env.agent_location, LOCATION_A)
        self.assertEqual(env.state, {LOCATION_A: DIRTY, LOCATION_B: DIRTY})
        self.assertFalse(env.is_all_clean())

    def test_custom_environment_initialization(self) -> None:
        """Verify initialization with custom state dictionary and location."""
        custom_state = {LOCATION_A: DIRTY, LOCATION_B: CLEAN}
        env = TwoSquareEnvironment(initial_location=LOCATION_B, state=custom_state)
        self.assertEqual(env.agent_location, LOCATION_B)
        self.assertEqual(env.state, {LOCATION_A: DIRTY, LOCATION_B: CLEAN})

    def test_invalid_initial_location_raises_value_error(self) -> None:
        """Verify that an invalid starting location is rejected."""
        with self.assertRaises(ValueError):
            TwoSquareEnvironment(initial_location="Z")

    def test_missing_state_key_raises_value_error(self) -> None:
        """Verify that state dictionaries missing required keys are rejected."""
        with self.assertRaises(ValueError):
            TwoSquareEnvironment(state={LOCATION_A: DIRTY})

    def test_invalid_status_in_state_dict_raises_value_error(self) -> None:
        """Verify that invalid status values in state dictionary are rejected."""
        with self.assertRaises(ValueError):
            TwoSquareEnvironment(state={LOCATION_A: "Muddy", LOCATION_B: CLEAN})

    def test_state_encapsulation_defensive_copy(self) -> None:
        """Ensure external mutation of returned state copy does not alter internal state."""
        env = TwoSquareEnvironment()
        external_state = env.state
        external_state[LOCATION_A] = CLEAN
        self.assertEqual(env.state[LOCATION_A], DIRTY)

    def test_perceive_matches_agent_current_square(self) -> None:
        """Verify that perceive() returns current location and status."""
        env = TwoSquareEnvironment(initial_location=LOCATION_A, state={LOCATION_A: DIRTY, LOCATION_B: CLEAN})
        self.assertEqual(env.perceive(), (LOCATION_A, DIRTY))

        env.execute_action(ACTION_RIGHT)
        self.assertEqual(env.perceive(), (LOCATION_B, CLEAN))

    def test_actuator_transitions(self) -> None:
        """Verify state and position updates across all actuators."""
        env = TwoSquareEnvironment(initial_location=LOCATION_A, state={LOCATION_A: DIRTY, LOCATION_B: DIRTY})

        # Suck cleans location A
        env.execute_action(ACTION_SUCK)
        self.assertEqual(env.state[LOCATION_A], CLEAN)
        self.assertEqual(env.agent_location, LOCATION_A)

        # Move Right to B
        env.execute_action(ACTION_RIGHT)
        self.assertEqual(env.agent_location, LOCATION_B)

        # Suck cleans location B
        env.execute_action(ACTION_SUCK)
        self.assertEqual(env.state[LOCATION_B], CLEAN)
        self.assertTrue(env.is_all_clean())

        # Move Left back to A
        env.execute_action(ACTION_LEFT)
        self.assertEqual(env.agent_location, LOCATION_A)

        # NoOp leaves state intact
        env.execute_action(ACTION_NOOP)
        self.assertEqual(env.agent_location, LOCATION_A)
        self.assertTrue(env.is_all_clean())

    def test_unrecognized_action_raises_value_error(self) -> None:
        """Verify that invalid actuator actions trigger ValueError."""
        env = TwoSquareEnvironment()
        with self.assertRaises(ValueError):
            env.execute_action("FLY_AWAY")


class TestSimulationExecution(unittest.TestCase):
    """Test discrete time step execution and trace accuracy against problem requirements."""

    def test_sample_scenario_five_step_trace(self) -> None:
        """Verify the exact sample scenario from the problem statement:

        Initial Location: 'A', A: 'Dirty', B: 'Clean'
        Expected Output:
        Step 1: Location A is Dirty -> Action: Suck.
        Step 2: Location A is Clean -> Action: Right.
        Step 3: Location B is Clean -> Action: Left.
        Step 4: Location A is Clean -> Action: Right.
        Step 5: Location B is Clean -> Action: Left.
        """
        env = TwoSquareEnvironment(
            initial_location=LOCATION_A,
            state={LOCATION_A: DIRTY, LOCATION_B: CLEAN},
        )
        history = env.simulate(steps=5)

        self.assertEqual(len(history), 5)

        expected_traces = [
            "Step 1: Location A is Dirty -> Action: Suck.",
            "Step 2: Location A is Clean -> Action: Right.",
            "Step 3: Location B is Clean -> Action: Left.",
            "Step 4: Location A is Clean -> Action: Right.",
            "Step 5: Location B is Clean -> Action: Left.",
        ]

        for i, expected in enumerate(expected_traces):
            self.assertEqual(history[i].format_trace(), expected)

        # Final state check: both locations clean
        self.assertTrue(env.is_all_clean())
        self.assertEqual(env.state, {LOCATION_A: CLEAN, LOCATION_B: CLEAN})

    def test_simulation_both_dirty_starting_at_b(self) -> None:
        """Verify 5-step simulation when starting at B with both squares dirty."""
        env = TwoSquareEnvironment(
            initial_location=LOCATION_B,
            state={LOCATION_A: DIRTY, LOCATION_B: DIRTY},
        )
        history = env.simulate(steps=5)

        self.assertEqual(history[0].format_trace(), "Step 1: Location B is Dirty -> Action: Suck.")
        self.assertEqual(history[1].format_trace(), "Step 2: Location B is Clean -> Action: Left.")
        self.assertEqual(history[2].format_trace(), "Step 3: Location A is Dirty -> Action: Suck.")
        self.assertEqual(history[3].format_trace(), "Step 4: Location A is Clean -> Action: Right.")
        self.assertEqual(history[4].format_trace(), "Step 5: Location B is Clean -> Action: Left.")
        self.assertTrue(env.is_all_clean())

    def test_simulation_both_initially_clean(self) -> None:
        """Demonstrate the classic reflex infinite bounce when already clean."""
        env = TwoSquareEnvironment(
            initial_location=LOCATION_A,
            state={LOCATION_A: CLEAN, LOCATION_B: CLEAN},
        )
        history = env.simulate(steps=4)

        self.assertEqual(history[0].action, ACTION_RIGHT)
        self.assertEqual(history[1].action, ACTION_LEFT)
        self.assertEqual(history[2].action, ACTION_RIGHT)
        self.assertEqual(history[3].action, ACTION_LEFT)

    def test_invalid_steps_raises_value_error(self) -> None:
        """Verify that step counts less than 1 are rejected."""
        env = TwoSquareEnvironment()
        with self.assertRaises(ValueError):
            env.simulate(steps=0)
        with self.assertRaises(ValueError):
            env.simulate(steps=-3)


class TestGridVacuumChallenge(unittest.TestCase):
    """Test the optional 3x3 grid simulation challenge."""

    def test_grid_initialization_and_perceive(self) -> None:
        """Verify 3x3 grid setup and perceptual coordinates."""
        grid_env = GridVacuumEnvironment(agent_pos=(0, 0), dirt_probability=0.0)
        pos, status = grid_env.perceive()
        self.assertEqual(pos, (0, 0))
        self.assertEqual(status, CLEAN)
        self.assertEqual(len(grid_env.grid), 9)

    def test_grid_custom_dirt_and_cleaning(self) -> None:
        """Verify cleaning in 3x3 grid environment."""
        custom_grid = {(0, 0): DIRTY, (0, 1): CLEAN}
        grid_env = GridVacuumEnvironment(agent_pos=(0, 0), custom_grid=custom_grid)

        step_num, pos, status, action = grid_env.step(step_number=1)
        self.assertEqual(step_num, 1)
        self.assertEqual(pos, (0, 0))
        self.assertEqual(status, DIRTY)
        self.assertEqual(action, ACTION_SUCK)
        self.assertEqual(grid_env.grid[(0, 0)], CLEAN)

    def test_grid_invalid_coordinate_raises_value_error(self) -> None:
        """Verify out-of-bounds agent coordinate raises ValueError."""
        with self.assertRaises(ValueError):
            GridVacuumEnvironment(agent_pos=(3, 3))


if __name__ == "__main__":
    unittest.main()
