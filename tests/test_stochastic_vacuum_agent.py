"""Unit tests for Stochastic Environment Vacuum Agent Simulation (Question 7).

Verifies condition-action reflex mapping, probabilistic actuator transitions,
slip mechanics (stay in place and slip backward), Monte Carlo statistical averaging,
deterministic baseline comparison, performance penalty calculations, boundary walls,
and edge cases.
"""

from io import StringIO
import random
import statistics
import sys
import unittest
from unittest.mock import patch

from src.stochastic_vacuum_agent import (
    ACTION_CLEAN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_SUCK,
    OUTCOME_BLOCKED,
    OUTCOME_CLEANED,
    OUTCOME_SLIP_BACK,
    OUTCOME_SLIP_STAY,
    OUTCOME_SUCCESS,
    ROOM_CLEAN,
    ROOM_DIRTY,
    MonteCarloExperimentResult,
    SingleRunResult,
    StochasticStepRecord,
    StochasticVacuumAgent,
    StochasticVacuumEnvironment,
    calculate_theoretical_expected_steps,
    run_deterministic_simulation,
    run_monte_carlo_stochastic_simulation,
    run_single_stochastic_simulation,
)


class TestStochasticVacuumAgentPolicy(unittest.TestCase):
    """Test the condition-action rules and decision logic of the vacuum agent."""

    def setUp(self) -> None:
        self.agent = StochasticVacuumAgent(target_rooms=3)

    def test_dirty_percept_always_returns_clean(self) -> None:
        """Verify that any dirty room immediately triggers 'Clean' action."""
        for room_idx in range(3):
            action = self.agent.decide_action(percept=ROOM_DIRTY, current_room=room_idx)
            self.assertEqual(action, ACTION_CLEAN)

    def test_clean_percept_moves_right_when_not_at_boundary(self) -> None:
        """Verify that a clean room triggers 'Right' action to explore subsequent rooms."""
        self.assertEqual(self.agent.decide_action(ROOM_CLEAN, current_room=0), ACTION_RIGHT)
        self.assertEqual(self.agent.decide_action(ROOM_CLEAN, current_room=1), ACTION_RIGHT)

    def test_clean_percept_at_last_room_returns_noop(self) -> None:
        """Verify that when the final room is clean, agent issues 'NoOp'."""
        self.assertEqual(self.agent.decide_action(ROOM_CLEAN, current_room=2), ACTION_NOOP)

    def test_invalid_percept_raises_value_error(self) -> None:
        """Verify that invalid percept values raise ValueError."""
        with self.assertRaises(ValueError):
            self.agent.decide_action(percept=2, current_room=0)
        with self.assertRaises(ValueError):
            self.agent.decide_action(percept=-1, current_room=0)


class TestStochasticEnvironmentTransitions(unittest.TestCase):
    """Test actuator stochasticity, probability boundaries, and transition mechanics."""

    def test_clean_action_cleans_current_room_deterministically(self) -> None:
        """Verify that 'Clean' deterministically updates the current room to Clean (0)."""
        env = StochasticVacuumEnvironment(num_rooms=3, initial_grid=[1, 1, 1], initial_agent_room=0)
        record = env.execute_action(ACTION_CLEAN)

        self.assertEqual(record.outcome, OUTCOME_CLEANED)
        self.assertEqual(env.grid, [0, 1, 1])
        self.assertEqual(env.agent_room, 0)
        self.assertEqual(env.cleaning_steps_count, 1)

    def test_clean_alias_suck_works_identically(self) -> None:
        """Verify that 'Suck' action alias cleans the room correctly."""
        env = StochasticVacuumEnvironment(num_rooms=3, initial_grid=[1, 1, 1], initial_agent_room=1)
        record = env.execute_action(ACTION_SUCK)
        self.assertEqual(record.outcome, OUTCOME_CLEANED)
        self.assertEqual(env.grid, [1, 0, 1])

    @patch("random.random")
    def test_move_right_success_with_80_percent_prob(self, mock_random) -> None:
        """Verify that when random float < 0.8, the agent moves forward (room + 1)."""
        mock_random.return_value = 0.50  # < 0.8
        env = StochasticVacuumEnvironment(num_rooms=3, initial_grid=[0, 1, 1], initial_agent_room=0)
        record = env.execute_action(ACTION_RIGHT)

        self.assertEqual(record.outcome, OUTCOME_SUCCESS)
        self.assertEqual(env.agent_room, 1)
        self.assertEqual(env.successful_moves_count, 1)
        self.assertEqual(env.slips_stay_count, 0)

    @patch("random.random")
    def test_move_right_slip_stay_with_20_percent_prob(self, mock_random) -> None:
        """Verify that when random float >= 0.8, the agent slips and stays in place."""
        mock_random.return_value = 0.85  # >= 0.8
        env = StochasticVacuumEnvironment(num_rooms=3, initial_grid=[0, 1, 1], initial_agent_room=0)
        record = env.execute_action(ACTION_RIGHT)

        self.assertEqual(record.outcome, OUTCOME_SLIP_STAY)
        self.assertEqual(env.agent_room, 0)  # Stayed in room 0
        self.assertEqual(env.slips_stay_count, 1)
        self.assertEqual(env.successful_moves_count, 0)

    @patch("random.random")
    def test_move_right_boundary_wall_blocks_movement(self, mock_random) -> None:
        """Verify that at the boundary wall (room 2), moving Right is blocked."""
        mock_random.return_value = 0.10
        env = StochasticVacuumEnvironment(num_rooms=3, initial_grid=[0, 0, 0], initial_agent_room=2)
        record = env.execute_action(ACTION_RIGHT)

        self.assertEqual(record.outcome, OUTCOME_BLOCKED)
        self.assertEqual(env.agent_room, 2)

    @patch("random.random")
    def test_slip_back_challenge_mechanics(self, mock_random) -> None:
        """Verify optional challenge: slip-back moves agent backward (room - 1)."""
        # Config: success = 0.8, slip_stay = 0.1, slip_back = 0.1
        env = StochasticVacuumEnvironment(
            num_rooms=3,
            initial_grid=[0, 1, 0],
            initial_agent_room=1,
            move_success_prob=0.8,
            slip_stay_prob=0.1,
            slip_back_prob=0.1,
        )

        # Value >= 0.9 triggers slip back
        mock_random.return_value = 0.95
        record = env.execute_action(ACTION_RIGHT)

        self.assertEqual(record.outcome, OUTCOME_SLIP_BACK)
        self.assertEqual(env.agent_room, 0)
        self.assertEqual(env.slips_back_count, 1)

    def test_invalid_probability_distribution_raises_value_error(self) -> None:
        """Verify that probability configurations not summing to 1.0 raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StochasticVacuumEnvironment(move_success_prob=0.8, slip_stay_prob=0.3, slip_back_prob=0.0)
        self.assertIn("must sum to 1.0", str(ctx.exception))

        with self.assertRaises(ValueError):
            StochasticVacuumEnvironment(move_success_prob=-0.1, slip_stay_prob=1.1, slip_back_prob=0.0)

    def test_invalid_room_and_grid_inputs(self) -> None:
        """Verify error handling on malformed grid and initial positions."""
        with self.assertRaises(ValueError):
            StochasticVacuumEnvironment(num_rooms=0)
        with self.assertRaises(ValueError):
            StochasticVacuumEnvironment(num_rooms=3, initial_grid=[1, 1])  # length mismatch
        with self.assertRaises(ValueError):
            StochasticVacuumEnvironment(num_rooms=3, initial_grid=[1, 5, 0])  # invalid status
        with self.assertRaises(ValueError):
            StochasticVacuumEnvironment(num_rooms=3, initial_agent_room=5)  # out of bounds

    def test_invalid_action_raises_value_error(self) -> None:
        """Verify that unrecognized action strings raise ValueError."""
        env = StochasticVacuumEnvironment(num_rooms=3)
        with self.assertRaises(ValueError):
            env.execute_action("Teleport")


class TestDeterministicBaseline(unittest.TestCase):
    """Test deterministic baseline runs with 100% actuator reliability."""

    def test_three_room_deterministic_run_takes_exact_five_steps(self) -> None:
        """Verify that a 3-room all-dirty grid takes exactly 5 deterministic steps."""
        result = run_deterministic_simulation(num_rooms=3)

        self.assertEqual(result.steps_taken, 5)
        self.assertEqual(result.cleaning_steps, 3)
        self.assertEqual(result.successful_moves, 2)
        self.assertEqual(result.slips_stay_count, 0)
        self.assertEqual(result.slips_back_count, 0)
        self.assertTrue(result.is_all_clean)
        self.assertEqual(result.final_grid, [0, 0, 0])

        # Verify action sequence
        actions = [rec.action for rec in result.history]
        self.assertEqual(actions, [ACTION_CLEAN, ACTION_RIGHT, ACTION_CLEAN, ACTION_RIGHT, ACTION_CLEAN])

    def test_custom_grid_deterministic(self) -> None:
        """Verify deterministic run when some rooms are already clean."""
        # Room 0 is clean, Room 1 is dirty, Room 2 is clean
        result = run_deterministic_simulation(num_rooms=3, initial_grid=[0, 1, 0], initial_agent_room=0)
        # Sequence: Move Right (to room 1) -> Clean (room 1) -> All rooms clean!
        # Total steps = 2 steps (Right, Clean)
        self.assertEqual(result.steps_taken, 2)
        self.assertTrue(result.is_all_clean)


class TestTheoreticalCalculation(unittest.TestCase):
    """Test mathematical derivation of expected steps."""

    def test_theoretical_expectation_three_rooms_80_percent(self) -> None:
        """Verify 3 rooms with 80% success yields exact 5.5 expected steps."""
        expected = calculate_theoretical_expected_steps(num_rooms=3, move_success_prob=0.8)
        self.assertAlmostEqual(expected, 5.5, places=5)

    def test_theoretical_expectation_single_room(self) -> None:
        """Verify 1 room requires exactly 1 step."""
        expected = calculate_theoretical_expected_steps(num_rooms=1, move_success_prob=0.8)
        self.assertEqual(expected, 1.0)

    def test_theoretical_expectation_five_rooms(self) -> None:
        """Verify 5 rooms with 80% success: 5 clean + 4/0.8 moves = 5 + 5 = 10 steps."""
        expected = calculate_theoretical_expected_steps(num_rooms=5, move_success_prob=0.8)
        self.assertAlmostEqual(expected, 10.0, places=5)

    def test_theoretical_expectation_with_slip_back(self) -> None:
        """Verify theoretical calculation with slip back probability."""
        expected = calculate_theoretical_expected_steps(
            num_rooms=3, move_success_prob=0.8, slip_back_prob=0.1
        )
        # e0 = 2/0.8 + 0.1 / (0.8**2) = 2.5 + 0.1 / 0.64 = 2.5 + 0.15625 = 2.65625
        # Total expected = 3 + 2.65625 = 5.65625
        self.assertAlmostEqual(expected, 5.65625, places=5)

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Verify error handling on invalid calculation parameters."""
        with self.assertRaises(ValueError):
            calculate_theoretical_expected_steps(num_rooms=0)
        with self.assertRaises(ValueError):
            calculate_theoretical_expected_steps(move_success_prob=0.0)
        with self.assertRaises(ValueError):
            calculate_theoretical_expected_steps(slip_back_prob=-0.1)


class TestMonteCarloSimulation(unittest.TestCase):
    """Test Monte Carlo simulation execution, statistical metrics, and sample output."""

    def test_twenty_runs_records_each_step_count(self) -> None:
        """Verify 20 runs are executed and all step counts are recorded."""
        result = run_monte_carlo_stochastic_simulation(num_runs=20, num_rooms=3, seed=100)

        self.assertEqual(result.num_runs, 20)
        self.assertEqual(len(result.steps_per_run), 20)
        self.assertEqual(len(result.runs), 20)
        self.assertEqual(result.deterministic_steps, 5)

        # All runs must achieve clean grid
        self.assertTrue(all(r.is_all_clean for r in result.runs))

        # Each run must take at least 5 steps (minimum theoretical limit)
        self.assertTrue(all(steps >= 5 for steps in result.steps_per_run))

        # Mean must be mathematically plausible (between 5.0 and 8.0)
        self.assertGreaterEqual(result.mean_steps, 5.0)
        self.assertLessEqual(result.mean_steps, 8.0)

        # Performance penalty must be non-negative
        self.assertGreaterEqual(result.absolute_penalty, 0.0)
        self.assertGreaterEqual(result.percentage_penalty, 0.0)

    def test_reproducible_seed_produces_exact_sample_output_5_4(self) -> None:
        """Verify seed 28 produces exactly the sample output: 5.4 average steps."""
        result = run_monte_carlo_stochastic_simulation(num_runs=20, num_rooms=3, seed=28)

        self.assertAlmostEqual(result.mean_steps, 5.4, places=4)
        self.assertEqual(result.deterministic_steps, 5)
        self.assertAlmostEqual(result.absolute_penalty, 0.4, places=4)
        self.assertAlmostEqual(result.percentage_penalty, 8.0, places=4)

    def test_monte_carlo_with_slip_back_challenge(self) -> None:
        """Verify Monte Carlo simulation runs with optional slip back enabled."""
        result = run_monte_carlo_stochastic_simulation(
            num_runs=15,
            num_rooms=3,
            move_success_prob=0.8,
            slip_stay_prob=0.1,
            slip_back_prob=0.1,
            seed=42,
        )
        self.assertEqual(result.num_runs, 15)
        self.assertTrue(all(r.is_all_clean for r in result.runs))
        self.assertGreaterEqual(result.mean_steps, 5.0)

    def test_summary_table_formatting(self) -> None:
        """Verify summary table string includes all key metrics."""
        result = run_monte_carlo_stochastic_simulation(num_runs=5, num_rooms=3, seed=1)
        table = result.format_summary_table()

        self.assertIn("MONTE CARLO SIMULATION RESULTS", table)
        self.assertIn("Deterministic Baseline      : 5 steps", table)
        self.assertIn("Empirical Mean Steps (Avg)", table)
        self.assertIn("Performance Penalty", table)

    def test_step_trace_formatting(self) -> None:
        """Verify single step record formatting."""
        record = StochasticStepRecord(
            step_number=1,
            room_before=0,
            percept=ROOM_DIRTY,
            action=ACTION_CLEAN,
            outcome=OUTCOME_CLEANED,
            room_after=0,
            grid_state_after=[0, 1, 1],
            random_sample=None,
        )
        trace = record.format_trace()
        self.assertIn("Step  1:", trace)
        self.assertIn("Dirty", trace)
        self.assertIn("Clean", trace)
        self.assertIn("Cleaned Room", trace)


class TestEdgeCasesAndSafetyLimits(unittest.TestCase):
    """Test safety limits and boundary edge cases."""

    def test_all_clean_initial_grid_terminates_immediately(self) -> None:
        """Verify that an already clean grid takes 0 steps."""
        result = run_single_stochastic_simulation(num_rooms=3, initial_grid=[0, 0, 0])
        self.assertEqual(result.steps_taken, 0)
        self.assertTrue(result.is_all_clean)

    def test_max_steps_safety_ceiling(self) -> None:
        """Verify that runaway loops halt cleanly when max_steps is hit."""
        # Force 0% success, 100% slip stay
        result = run_single_stochastic_simulation(
            num_rooms=3,
            initial_grid=[0, 1, 1],
            move_success_prob=0.0001,
            slip_stay_prob=0.9999,
            max_steps=10,
        )
        self.assertLessEqual(result.steps_taken, 10)
        self.assertTrue(result.stopped_by_limit)


if __name__ == "__main__":
    unittest.main()
