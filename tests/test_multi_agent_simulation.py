"""Unit test suite for Multi-Agent Environment Simulation (Question 6).

Covers:
- BaseAgent, BuilderAgent, and DestroyerAgent initialization and parameter validation
- Percept-action rules for Builder (places blocks on 0) and Destroyer (removes blocks on 1)
- Shared environment reference mutation across agents
- Boundary checking and grid dimension constraints
- Sequential turn ordering (Agent A turn_index=1, Agent B turn_index=2)
- Discrete vs. Compound turn execution modes
- Optional Challenge: Active collision awareness and avoidance
- Movement strategies (Random vs. Greedy)
- Reproducibility with deterministic random seeds
- Output formatting (NumPy matrix display, ASCII board, summary report)
- Edge cases: corner starts, trapped agent fallback (NoOp), invalid inputs
"""

import unittest
from typing import List, Tuple

import numpy as np

from src.multi_agent_simulation import (
    ACTION_BUILD,
    ACTION_DESTROY,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_NOOP,
    CELL_BLOCK,
    CELL_EMPTY,
    ROLE_BUILDER,
    ROLE_DESTROYER,
    STRATEGY_GREEDY,
    STRATEGY_RANDOM,
    TURN_MODE_COMPOUND,
    TURN_MODE_DISCRETE,
    AgentTurnRecord,
    BaseAgent,
    BuilderAgent,
    DestroyerAgent,
    MultiAgentGridEnvironment,
    MultiAgentPercept,
    MultiAgentSimulationResult,
    manhattan_distance,
    run_multi_agent_simulation,
)


class TestAgentBasicsAndValidation(unittest.TestCase):
    """Tests for agent initialization, role enforcement, and coordinate validation."""

    def test_builder_initialization(self) -> None:
        """Verify Builder agent initializes with default attributes and role."""
        builder = BuilderAgent(name="Builder_Test", initial_position=(0, 0), seed=42)
        self.assertEqual(builder.name, "Builder_Test")
        self.assertEqual(builder.role, ROLE_BUILDER)
        self.assertEqual(builder.position, (0, 0))
        self.assertEqual(builder.actions_performed, 0)
        self.assertEqual(builder.moves_executed, 0)

    def test_destroyer_initialization(self) -> None:
        """Verify Destroyer agent initializes with default attributes and role."""
        destroyer = DestroyerAgent(name="Destroyer_Test", initial_position=(3, 3), seed=42)
        self.assertEqual(destroyer.name, "Destroyer_Test")
        self.assertEqual(destroyer.role, ROLE_DESTROYER)
        self.assertEqual(destroyer.position, (3, 3))

    def test_invalid_role_raises_value_error(self) -> None:
        """Verify subclassing or passing an invalid role raises ValueError."""
        class InvalidAgent(BaseAgent):
            def decide_discrete_action(self, percept, avoid_collisions=False):
                return ACTION_NOOP, None, False
            def decide_compound_action(self, percept, avoid_collisions=False):
                return ACTION_NOOP, None, None, False

        with self.assertRaises(ValueError):
            InvalidAgent(name="Bad", role="Spectator", initial_position=(0, 0))

    def test_invalid_strategy_raises_value_error(self) -> None:
        """Verify unrecognized movement strategy raises ValueError."""
        with self.assertRaises(ValueError):
            BuilderAgent(initial_position=(0, 0), strategy="quantum_leap")

    def test_malformed_initial_position_raises_type_error(self) -> None:
        """Verify non-tuple coordinates raise TypeError."""
        with self.assertRaises(TypeError):
            BuilderAgent(initial_position="top-left")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BuilderAgent(initial_position=(0, "one"))  # type: ignore[arg-type]

    def test_set_position_updates_coordinates_and_counter(self) -> None:
        """Verify set_position mutates coordinates and increments move counter."""
        agent = BuilderAgent(initial_position=(0, 0))
        agent.set_position((0, 1))
        self.assertEqual(agent.position, (0, 1))
        self.assertEqual(agent.moves_executed, 1)


class TestPerceptActionRules(unittest.TestCase):
    """Tests for Builder and Destroyer percept-action rules in both modes."""

    def test_builder_discrete_rule_empty_cell(self) -> None:
        """Verify Builder decides PlaceBlock when current cell is empty (0)."""
        builder = BuilderAgent(initial_position=(0, 0))
        percept = MultiAgentPercept(
            agent_name="Agent_A",
            position=(0, 0),
            cell_value=CELL_EMPTY,
            available_moves=((0, 1), (1, 0)),
        )
        action, target, avoided = builder.decide_discrete_action(percept)
        self.assertEqual(action, ACTION_BUILD)
        self.assertIsNone(target)
        self.assertFalse(avoided)
        self.assertEqual(builder.actions_performed, 1)

    def test_builder_discrete_rule_block_cell_moves(self) -> None:
        """Verify Builder decides to Move when current cell already has a block (1)."""
        builder = BuilderAgent(initial_position=(0, 0), seed=42)
        percept = MultiAgentPercept(
            agent_name="Agent_A",
            position=(0, 0),
            cell_value=CELL_BLOCK,
            available_moves=((0, 1), (1, 0)),
        )
        action, target, avoided = builder.decide_discrete_action(percept)
        self.assertIn(action, (ACTION_MOVE_RIGHT, ACTION_MOVE_DOWN))
        self.assertIn(target, ((0, 1), (1, 0)))

    def test_destroyer_discrete_rule_block_cell(self) -> None:
        """Verify Destroyer decides RemoveBlock when current cell has a block (1)."""
        destroyer = DestroyerAgent(initial_position=(3, 3))
        percept = MultiAgentPercept(
            agent_name="Agent_B",
            position=(3, 3),
            cell_value=CELL_BLOCK,
            available_moves=((2, 3), (3, 2)),
        )
        action, target, avoided = destroyer.decide_discrete_action(percept)
        self.assertEqual(action, ACTION_DESTROY)
        self.assertIsNone(target)
        self.assertEqual(destroyer.actions_performed, 1)

    def test_destroyer_discrete_rule_empty_cell_moves(self) -> None:
        """Verify Destroyer decides to Move when current cell is empty (0)."""
        destroyer = DestroyerAgent(initial_position=(3, 3), seed=42)
        percept = MultiAgentPercept(
            agent_name="Agent_B",
            position=(3, 3),
            cell_value=CELL_EMPTY,
            available_moves=((2, 3), (3, 2)),
        )
        action, target, avoided = destroyer.decide_discrete_action(percept)
        self.assertIn(action, (ACTION_MOVE_UP, ACTION_MOVE_LEFT))
        self.assertIn(target, ((2, 3), (3, 2)))

    def test_compound_mode_builder(self) -> None:
        """Verify Builder executes PlaceBlock AND moves to an adjacent cell."""
        builder = BuilderAgent(initial_position=(0, 0), seed=42)
        percept = MultiAgentPercept(
            agent_name="Agent_A",
            position=(0, 0),
            cell_value=CELL_EMPTY,
            available_moves=((0, 1), (1, 0)),
        )
        cell_action, move_action, target_pos, avoided = builder.decide_compound_action(percept)
        self.assertEqual(cell_action, ACTION_BUILD)
        self.assertIn(move_action, (ACTION_MOVE_RIGHT, ACTION_MOVE_DOWN))
        self.assertIn(target_pos, ((0, 1), (1, 0)))

    def test_compound_mode_destroyer(self) -> None:
        """Verify Destroyer executes RemoveBlock AND moves to an adjacent cell."""
        destroyer = DestroyerAgent(initial_position=(3, 3), seed=42)
        percept = MultiAgentPercept(
            agent_name="Agent_B",
            position=(3, 3),
            cell_value=CELL_BLOCK,
            available_moves=((2, 3), (3, 2)),
        )
        cell_action, move_action, target_pos, avoided = destroyer.decide_compound_action(percept)
        self.assertEqual(cell_action, ACTION_DESTROY)
        self.assertIn(move_action, (ACTION_MOVE_UP, ACTION_MOVE_LEFT))
        self.assertIn(target_pos, ((2, 3), (3, 2)))


class TestSharedEnvironmentAndStateModification(unittest.TestCase):
    """Tests for shared reference mechanics, grid bounds, and state mutation."""

    def test_environment_initial_state_empty_grid(self) -> None:
        """Verify environment initializes as 4x4 array of zeros."""
        env = MultiAgentGridEnvironment(rows=4, cols=4)
        np.testing.assert_array_equal(env.grid, np.zeros((4, 4), dtype=int))
        self.assertEqual(env.agent_a.position, (0, 0))
        self.assertEqual(env.agent_b.position, (3, 3))

    def test_invalid_dimensions_raise_value_error(self) -> None:
        """Verify negative or zero dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            MultiAgentGridEnvironment(rows=0, cols=4)
        with self.assertRaises(ValueError):
            MultiAgentGridEnvironment(rows=4, cols=-1)

    def test_out_of_bounds_agent_raises_index_error(self) -> None:
        """Verify placing an agent out of bounds raises IndexError."""
        bad_builder = BuilderAgent(initial_position=(4, 0))
        with self.assertRaises(IndexError):
            MultiAgentGridEnvironment(rows=4, cols=4, agent_a=bad_builder)

    def test_shared_reference_state_mutation(self) -> None:
        """Verify modifications by one agent immediately affect the shared grid."""
        env = MultiAgentGridEnvironment(rows=4, cols=4)
        # Agent A places block at (0, 0)
        env.set_cell((0, 0), CELL_BLOCK)
        self.assertEqual(env.get_cell((0, 0)), CELL_BLOCK)

        # Agent B perceives block at (0, 0)
        env.agent_b.set_position((0, 0))
        percept_b = env.generate_percept(env.agent_b)
        self.assertEqual(percept_b.cell_value, CELL_BLOCK)

        # Agent B removes block at (0, 0)
        env.set_cell((0, 0), CELL_EMPTY)
        self.assertEqual(env.get_cell((0, 0)), CELL_EMPTY)

    def test_set_cell_invalid_value_raises_value_error(self) -> None:
        """Verify setting a cell to a value other than 0 or 1 raises ValueError."""
        env = MultiAgentGridEnvironment(rows=4, cols=4)
        with self.assertRaises(ValueError):
            env.set_cell((0, 0), 2)

    def test_custom_initial_grid(self) -> None:
        """Verify custom 4x4 initial grid is respected."""
        custom_grid = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        env = MultiAgentGridEnvironment(rows=4, cols=4, initial_grid=custom_grid)
        np.testing.assert_array_equal(env.grid, custom_grid)

    def test_mismatched_initial_grid_shape_raises_value_error(self) -> None:
        """Verify dimension mismatch between initial_grid and (rows, cols) raises ValueError."""
        with self.assertRaises(ValueError):
            MultiAgentGridEnvironment(rows=4, cols=4, initial_grid=np.zeros((3, 3)))


class TestCollisionAvoidance(unittest.TestCase):
    """Tests for the Optional Challenge: Active collision awareness and avoidance."""

    def test_filter_safe_moves_removes_peer_coordinate(self) -> None:
        """Verify filter_safe_moves eliminates peer coordinate from candidates."""
        agent = BuilderAgent(initial_position=(1, 1))
        available = ((0, 1), (2, 1), (1, 0), (1, 2))
        peer_pos = (0, 1)

        safe_moves, avoided = agent.filter_safe_moves(available, peer_pos)
        self.assertTrue(avoided)
        self.assertNotIn(peer_pos, safe_moves)
        self.assertEqual(len(safe_moves), 3)

    def test_filter_safe_moves_without_peer(self) -> None:
        """Verify all moves retained when peer position is None."""
        agent = BuilderAgent(initial_position=(0, 0))
        available = ((0, 1), (1, 0))
        safe_moves, avoided = agent.filter_safe_moves(available, None)
        self.assertFalse(avoided)
        self.assertEqual(safe_moves, [(0, 1), (1, 0)])

    def test_cornered_agent_fallback_to_noop(self) -> None:
        """Verify agent falls back to NoOp when trapped with only peer coordinate adjacent."""
        builder = BuilderAgent(initial_position=(0, 0))
        # Suppose agent is at (0, 0) and the only neighbor is (0, 1) which peer occupies
        percept = MultiAgentPercept(
            agent_name="Agent_A",
            position=(0, 0),
            cell_value=CELL_BLOCK,
            available_moves=((0, 1),),
            other_agent_position=(0, 1),
        )
        action, target, avoided = builder.decide_discrete_action(percept, avoid_collisions=True)
        self.assertEqual(action, ACTION_NOOP)
        self.assertIsNone(target)
        self.assertTrue(avoided)

    def test_collision_avoidance_in_full_simulation(self) -> None:
        """Verify full simulation with collision avoidance reports zero collisions occurred."""
        res = run_multi_agent_simulation(steps=10, seed=42, avoid_collisions=True)
        self.assertTrue(res.collision_avoidance_active)
        self.assertEqual(res.collisions_occurred, 0)
        self.assertGreaterEqual(res.collisions_avoided_count, 1)


class TestMovementStrategies(unittest.TestCase):
    """Tests for Random and Greedy movement strategies."""

    def test_greedy_builder_seeks_empty_cell(self) -> None:
        """Verify Greedy Builder prefers moves that reduce distance to empty cells."""
        # Setup grid with only (0, 2) empty, others adjacent to (0, 0) have blocks
        grid_view = (
            (1, 1, 0, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        )
        builder = BuilderAgent(initial_position=(0, 0), strategy=STRATEGY_GREEDY, seed=42)
        # Neighbors of (0, 0) are (0, 1) and (1, 0).
        # Distance from (0, 1) to (0, 2) is 1. Distance from (1, 0) to (0, 2) is 3.
        chosen = builder.choose_move([(0, 1), (1, 0)], grid_view=grid_view)
        self.assertEqual(chosen, (0, 1))

    def test_greedy_destroyer_seeks_block_cell(self) -> None:
        """Verify Greedy Destroyer prefers moves that reduce distance to blocks."""
        # Setup grid with only (3, 1) having a block
        grid_view = (
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 1, 0, 0),
        )
        destroyer = DestroyerAgent(initial_position=(3, 3), strategy=STRATEGY_GREEDY, seed=42)
        # Neighbors of (3, 3) are (3, 2) and (2, 3).
        # Distance from (3, 2) to (3, 1) is 1. Distance from (2, 3) to (3, 1) is 3.
        chosen = destroyer.choose_move([(3, 2), (2, 3)], grid_view=grid_view)
        self.assertEqual(chosen, (3, 2))


class TestSimulationRunnerAndOutputs(unittest.TestCase):
    """Tests for the complete simulation loop and output formatting."""

    def test_run_simulation_ten_steps_default(self) -> None:
        """Verify default 10-step simulation runs and produces expected output formats."""
        res = run_multi_agent_simulation(steps=10, seed=42)
        self.assertEqual(res.total_steps, 10)
        self.assertEqual(len(res.history), 20)  # 10 steps * 2 agents
        self.assertEqual(res.final_grid.shape, (4, 4))
        self.assertTrue(np.all(np.isin(res.final_grid, [0, 1])))

        # Check sample output display matrix string
        matrix_str = res.display_matrix()
        self.assertIn("[[", matrix_str)
        self.assertIn("]]", matrix_str)

        # Check visual board string
        board_str = res.display_board()
        self.assertIn("R0 |", board_str)
        self.assertIn("Legend:", board_str)

        # Check summary string
        summary_str = res.summary()
        self.assertIn("MULTI-AGENT ENVIRONMENT SIMULATION SUMMARY", summary_str)
        self.assertIn("Builder placed", summary_str)
        self.assertIn("Destroyer removed", summary_str)

    def test_run_simulation_discrete_mode(self) -> None:
        """Verify simulation in discrete turn mode runs cleanly."""
        res = run_multi_agent_simulation(steps=10, step_mode=TURN_MODE_DISCRETE, seed=42)
        self.assertEqual(res.step_mode, TURN_MODE_DISCRETE)
        self.assertEqual(res.total_steps, 10)
        self.assertEqual(len(res.history), 20)

    def test_run_simulation_invalid_steps_raises_value_error(self) -> None:
        """Verify passing steps < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            run_multi_agent_simulation(steps=0)

    def test_run_simulation_invalid_step_mode_raises_value_error(self) -> None:
        """Verify invalid step mode raises ValueError."""
        with self.assertRaises(ValueError):
            run_multi_agent_simulation(step_mode="continuous")

    def test_deterministic_reproducibility(self) -> None:
        """Verify same seed produces bitwise identical simulation results."""
        res1 = run_multi_agent_simulation(steps=10, seed=99)
        res2 = run_multi_agent_simulation(steps=10, seed=99)
        np.testing.assert_array_equal(res1.final_grid, res2.final_grid)
        self.assertEqual(res1.blocks_placed, res2.blocks_placed)
        self.assertEqual(res1.blocks_destroyed, res2.blocks_destroyed)
        self.assertEqual(res1.agent_a_final, res2.agent_a_final)
        self.assertEqual(res1.agent_b_final, res2.agent_b_final)

    def test_manhattan_distance_calculation(self) -> None:
        """Verify Manhattan distance helper computation."""
        self.assertEqual(manhattan_distance((0, 0), (3, 3)), 6)
        self.assertEqual(manhattan_distance((1, 2), (1, 2)), 0)
        self.assertEqual(manhattan_distance((2, 1), (0, 3)), 4)


if __name__ == "__main__":
    unittest.main()
