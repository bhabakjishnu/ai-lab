"""Comprehensive unit test suite for State-Space Visualizer (Question 10).

Validates:
- Combinatorial state space generation (|S| = 2 * 2^2 = 8).
- Mathematical state size formula (|S| = n * k^n).
- Deterministic transition model Result(s, a).
- Identification of cyclic transitions and self-loops (12 out of 24).
- Goal state identification (highlighted in emerald green for Optional Challenge).
- NetworkX directed graph and MultiDiGraph construction.
- 8x8 adjacency matrix generation.
- Optimal shortest path search to goal states via BFS.
- Matplotlib visualization rendering (headless mode) and layout algorithms.
- Viva Voce reference theory integrity.
"""

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import networkx as nx

from src.state_space_visualizer import (
    ACTION_COLORS,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SUCK,
    CLEAN,
    COLOR_BORDER_GOAL,
    COLOR_BORDER_NON_GOAL,
    COLOR_GOAL,
    COLOR_NON_GOAL,
    DIRTY,
    LOCATION_A,
    LOCATION_B,
    VALID_ACTIONS,
    VALID_LOCATIONS,
    VALID_STATUSES,
    StateSpace,
    StateSpaceVisualizer,
    TransitionRecord,
    VacuumState,
    get_viva_reference,
)


class TestVacuumState(unittest.TestCase):
    """Test suite for VacuumState immutable data structure and properties."""

    def test_state_creation_and_attributes(self) -> None:
        """Verify normal creation of VacuumState with valid attributes."""
        state = VacuumState(
            agent_loc="A",
            room_a_status="Dirty",
            room_b_status="Clean",
            state_id=5,
        )
        self.assertEqual(state.agent_loc, LOCATION_A)
        self.assertEqual(state.room_a_status, DIRTY)
        self.assertEqual(state.room_b_status, CLEAN)
        self.assertEqual(state.state_id, 5)
        self.assertFalse(state.is_goal)
        self.assertTrue(state.is_dirty_at_current_loc)
        self.assertEqual(state.short_label, "S5: (A, D, C)")
        self.assertEqual(state.tuple, ("A", "Dirty", "Clean"))

    def test_goal_state_recognition(self) -> None:
        """Verify is_goal correctly identifies clean rooms."""
        goal_a = VacuumState("A", "Clean", "Clean", state_id=7)
        goal_b = VacuumState("B", "Clean", "Clean", state_id=8)
        non_goal = VacuumState("A", "Dirty", "Clean", state_id=5)

        self.assertTrue(goal_a.is_goal)
        self.assertTrue(goal_b.is_goal)
        self.assertFalse(non_goal.is_goal)

    def test_normalization_and_case_insensitivity(self) -> None:
        """Verify case and whitespace normalization on instantiation."""
        state = VacuumState(" a ", "  clean ", "dirty\t")
        self.assertEqual(state.agent_loc, LOCATION_A)
        self.assertEqual(state.room_a_status, CLEAN)
        self.assertEqual(state.room_b_status, DIRTY)

    def test_invalid_location_raises_value_error(self) -> None:
        """Verify invalid agent location raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            VacuumState("C", "Clean", "Dirty")
        self.assertIn("Invalid agent_loc 'C'", str(ctx.exception))

    def test_invalid_room_status_raises_value_error(self) -> None:
        """Verify invalid room cleanliness status raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            VacuumState("A", "Spotless", "Dirty")
        self.assertIn("Invalid room_a_status 'Spotless'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            VacuumState("A", "Clean", "Muddy")
        self.assertIn("Invalid room_b_status 'Muddy'", str(ctx.exception))

    def test_from_tuple_factory(self) -> None:
        """Verify factory method creates equivalent state from tuple."""
        t = ("B", "Dirty", "Dirty")
        state = VacuumState.from_tuple(t, state_id=2)
        self.assertEqual(state.agent_loc, "B")
        self.assertEqual(state.room_a_status, "Dirty")
        self.assertEqual(state.room_b_status, "Dirty")
        self.assertEqual(state.state_id, 2)

    def test_from_tuple_invalid_length(self) -> None:
        """Verify tuple with wrong element count raises ValueError."""
        with self.assertRaises(ValueError):
            VacuumState.from_tuple(("A", "Clean"))  # type: ignore

    def test_immutability_and_hashability(self) -> None:
        """Verify VacuumState is frozen and can be used in sets and dict keys."""
        s1 = VacuumState("A", "Clean", "Dirty", state_id=3)
        s2 = VacuumState("A", "Clean", "Dirty", state_id=3)
        self.assertEqual(s1, s2)
        self.assertEqual(hash(s1), hash(s2))

        # Check dictionary and set storage
        state_set = {s1, s2}
        self.assertEqual(len(state_set), 1)

        # Frozen check
        with self.assertRaises(AttributeError):
            s1.agent_loc = "B"  # type: ignore


class TestStateSpaceGeneration(unittest.TestCase):
    """Test suite for combinatorial state generation and mathematical scaling."""

    def setUp(self) -> None:
        self.space = StateSpace()

    def test_exact_eight_states_generated(self) -> None:
        """Verify exactly 8 states are generated combinatorially."""
        self.assertEqual(len(self.space.states), 8)
        # All states must be unique
        unique_states = set(self.space.states)
        self.assertEqual(len(unique_states), 8)

    def test_canonical_aima_state_order(self) -> None:
        """Verify canonical order matches Russell & Norvig textbook."""
        expected = [
            ("A", "Dirty", "Dirty"),
            ("B", "Dirty", "Dirty"),
            ("A", "Clean", "Dirty"),
            ("B", "Clean", "Dirty"),
            ("A", "Dirty", "Clean"),
            ("B", "Dirty", "Clean"),
            ("A", "Clean", "Clean"),
            ("B", "Clean", "Clean"),
        ]
        actual = [s.tuple for s in self.space.states]
        self.assertEqual(actual, expected)

        # State IDs must be 1 through 8
        self.assertEqual([s.state_id for s in self.space.states], list(range(1, 9)))

    def test_goal_states_subset(self) -> None:
        """Verify exactly 2 goal states exist (States 7 and 8)."""
        goals = self.space.goal_states
        self.assertEqual(len(goals), 2)
        self.assertEqual({g.state_id for g in goals}, {7, 8})
        for g in goals:
            self.assertTrue(g.is_goal)
            self.assertEqual(g.room_a_status, CLEAN)
            self.assertEqual(g.room_b_status, CLEAN)

    def test_state_space_size_formula(self) -> None:
        """Verify theoretical state space calculation |S| = n * k^n."""
        # 2 rooms, 2 statuses -> 2 * 2^2 = 8
        self.assertEqual(StateSpace.calculate_state_space_size(2, 2), 8)
        # 1 room, 2 statuses -> 1 * 2^1 = 2
        self.assertEqual(StateSpace.calculate_state_space_size(1, 2), 2)
        # 3 rooms, 2 statuses -> 3 * 2^3 = 24
        self.assertEqual(StateSpace.calculate_state_space_size(3, 2), 24)
        # 4 rooms, 2 statuses -> 4 * 2^4 = 64
        self.assertEqual(StateSpace.calculate_state_space_size(4, 2), 64)
        # 2 rooms, 3 statuses (Clean, Dirty, Wet) -> 2 * 3^2 = 18
        self.assertEqual(StateSpace.calculate_state_space_size(2, 3), 18)

    def test_state_space_size_formula_invalid_inputs(self) -> None:
        """Verify calculate_state_space_size validates positive integers."""
        with self.assertRaises(ValueError):
            StateSpace.calculate_state_space_size(0, 2)
        with self.assertRaises(ValueError):
            StateSpace.calculate_state_space_size(2, -1)


class TestTransitionDynamics(unittest.TestCase):
    """Test suite for transition function Result(s, a) and action consequences."""

    def setUp(self) -> None:
        self.space = StateSpace()

    def test_total_transitions_count(self) -> None:
        """Verify exactly 24 transitions exist (8 states * 3 actions)."""
        self.assertEqual(len(self.space.transitions), 24)

    def test_left_action_movement(self) -> None:
        """Verify action Left moves agent from B to A and stays at A."""
        # From B to A
        s2 = self.space.state_map[("B", "Dirty", "Dirty")]
        target_s1 = self.space.transition_function(s2, ACTION_LEFT)
        self.assertEqual(target_s1.agent_loc, LOCATION_A)
        self.assertEqual(target_s1.state_id, 1)

        # From A: cyclic self-loop
        s1 = self.space.state_map[("A", "Dirty", "Dirty")]
        target_self = self.space.transition_function(s1, ACTION_LEFT)
        self.assertEqual(target_self, s1)

    def test_right_action_movement(self) -> None:
        """Verify action Right moves agent from A to B and stays at B."""
        # From A to B
        s1 = self.space.state_map[("A", "Dirty", "Dirty")]
        target_s2 = self.space.transition_function(s1, ACTION_RIGHT)
        self.assertEqual(target_s2.agent_loc, LOCATION_B)
        self.assertEqual(target_s2.state_id, 2)

        # From B: cyclic self-loop
        s2 = self.space.state_map[("B", "Dirty", "Dirty")]
        target_self = self.space.transition_function(s2, ACTION_RIGHT)
        self.assertEqual(target_self, s2)

    def test_suck_action_cleaning(self) -> None:
        """Verify action Suck cleans current room and preserves other room."""
        # S1: (A, Dirty, Dirty) + Suck -> S3: (A, Clean, Dirty)
        s1 = self.space.state_map[("A", "Dirty", "Dirty")]
        succ_s3 = self.space.transition_function(s1, ACTION_SUCK)
        self.assertEqual(succ_s3.agent_loc, LOCATION_A)
        self.assertEqual(succ_s3.room_a_status, CLEAN)
        self.assertEqual(succ_s3.room_b_status, DIRTY)
        self.assertEqual(succ_s3.state_id, 3)

        # S2: (B, Dirty, Dirty) + Suck -> S6: (B, Dirty, Clean)
        s2 = self.space.state_map[("B", "Dirty", "Dirty")]
        succ_s6 = self.space.transition_function(s2, ACTION_SUCK)
        self.assertEqual(succ_s6.agent_loc, LOCATION_B)
        self.assertEqual(succ_s6.room_a_status, DIRTY)
        self.assertEqual(succ_s6.room_b_status, CLEAN)
        self.assertEqual(succ_s6.state_id, 6)

    def test_invalid_action_raises_value_error(self) -> None:
        """Verify unrecognized action string raises ValueError."""
        s1 = self.space.states[0]
        with self.assertRaises(ValueError) as ctx:
            self.space.transition_function(s1, "Jump")
        self.assertIn("Invalid action 'Jump'", str(ctx.exception))


class TestCyclicTransitionsAndCycles(unittest.TestCase):
    """Test suite verifying exact cycle identification and self-loop detection."""

    def setUp(self) -> None:
        self.space = StateSpace()

    def test_exact_self_loop_count_and_distribution(self) -> None:
        """Verify exactly 12 self-loops exist: 4 Left, 4 Right, 4 Suck."""
        cycle_info = self.space.identify_cycles()
        self.assertEqual(cycle_info["total_transitions"], 24)
        self.assertEqual(cycle_info["total_self_loops"], 12)
        self.assertEqual(cycle_info["state_changing_transitions"], 12)

        loops_by_act = cycle_info["self_loops_by_action"]
        self.assertEqual(loops_by_act[ACTION_LEFT], 4)
        self.assertEqual(loops_by_act[ACTION_RIGHT], 4)
        self.assertEqual(loops_by_act[ACTION_SUCK], 4)

    def test_left_self_loops_occur_at_location_a(self) -> None:
        """Verify Left produces self-loops only for states with agent at Location A."""
        self_loops = [t for t in self.space.transitions if t.is_cyclic and t.action == ACTION_LEFT]
        self.assertEqual(len(self_loops), 4)
        for t in self_loops:
            self.assertEqual(t.source.agent_loc, LOCATION_A)
            self.assertEqual(t.source, t.target)

    def test_right_self_loops_occur_at_location_b(self) -> None:
        """Verify Right produces self-loops only for states with agent at Location B."""
        self_loops = [t for t in self.space.transitions if t.is_cyclic and t.action == ACTION_RIGHT]
        self.assertEqual(len(self_loops), 4)
        for t in self_loops:
            self.assertEqual(t.source.agent_loc, LOCATION_B)
            self.assertEqual(t.source, t.target)

    def test_suck_self_loops_occur_when_current_room_is_clean(self) -> None:
        """Verify Suck produces self-loops only when current room is already clean."""
        self_loops = [t for t in self.space.transitions if t.is_cyclic and t.action == ACTION_SUCK]
        self.assertEqual(len(self_loops), 4)
        expected_state_ids = {3, 6, 7, 8}
        actual_state_ids = {t.source.state_id for t in self_loops}
        self.assertEqual(actual_state_ids, expected_state_ids)
        for t in self_loops:
            self.assertEqual(t.source, t.target)
            self.assertFalse(t.source.is_dirty_at_current_loc)

    def test_elementary_cycles_between_locations(self) -> None:
        """Verify elementary directed cycles exist between Location A and B."""
        cycle_info = self.space.identify_cycles()
        # Non-self-loop cycles (e.g. Left/Right oscillations)
        self.assertGreaterEqual(cycle_info["elementary_cycles_count"], 4)


class TestGraphModelAndAdjacencyMatrix(unittest.TestCase):
    """Test suite for NetworkX graph construction and matrix representations."""

    def setUp(self) -> None:
        self.space = StateSpace()

    def test_digraph_construction(self) -> None:
        """Verify NetworkX DiGraph contains 8 nodes with proper metadata."""
        g = self.space.to_networkx_graph(multi=False)
        self.assertIsInstance(g, nx.DiGraph)
        self.assertEqual(g.number_of_nodes(), 8)

        # Verify node attributes
        for s in g.nodes():
            self.assertIn("id", g.nodes[s])
            self.assertIn("label", g.nodes[s])
            self.assertIn("is_goal", g.nodes[s])
            self.assertIn("color", g.nodes[s])
            if s.is_goal:
                self.assertEqual(g.nodes[s]["color"], COLOR_GOAL)
                self.assertEqual(g.nodes[s]["border_color"], COLOR_BORDER_GOAL)
            else:
                self.assertEqual(g.nodes[s]["color"], COLOR_NON_GOAL)
                self.assertEqual(g.nodes[s]["border_color"], COLOR_BORDER_NON_GOAL)

    def test_multidigraph_construction(self) -> None:
        """Verify NetworkX MultiDiGraph contains all 24 individual directed edges."""
        mg = self.space.to_networkx_graph(multi=True)
        self.assertIsInstance(mg, nx.MultiDiGraph)
        self.assertEqual(mg.number_of_nodes(), 8)
        self.assertEqual(mg.number_of_edges(), 24)

    def test_adjacency_matrix_shape_and_content(self) -> None:
        """Verify 8x8 adjacency matrix dimensions and self-loop diagonal entries."""
        labels, matrix = self.space.to_adjacency_matrix()
        self.assertEqual(len(labels), 8)
        self.assertEqual(len(matrix), 8)
        for row in matrix:
            self.assertEqual(len(row), 8)
            # Row sum must equal 3 (each state has 3 outgoing actions)
            self.assertEqual(sum(row), 3)

        # Diagonal entries represent self-loops
        diagonal_self_loops = sum(matrix[i][i] for i in range(8))
        self.assertEqual(diagonal_self_loops, 12)


class TestShortestPathSearch(unittest.TestCase):
    """Test suite for BFS optimal path discovery to goal states."""

    def setUp(self) -> None:
        self.space = StateSpace()

    def test_shortest_path_from_both_dirty_at_a(self) -> None:
        """Verify optimal path from S1 (A, D, D) takes 3 steps to reach goal."""
        s1 = self.space.state_map[("A", "Dirty", "Dirty")]
        res = self.space.find_shortest_paths_to_goal(s1)
        self.assertEqual(res["distance"], 3)
        self.assertEqual(res["actions"], [ACTION_SUCK, ACTION_RIGHT, ACTION_SUCK])
        self.assertTrue(res["goal"].is_goal)
        self.assertEqual(res["goal"].state_id, 8)

    def test_shortest_path_from_both_dirty_at_b(self) -> None:
        """Verify optimal path from S2 (B, D, D) takes 3 steps to reach goal."""
        s2 = self.space.state_map[("B", "Dirty", "Dirty")]
        res = self.space.find_shortest_paths_to_goal(s2)
        self.assertEqual(res["distance"], 3)
        self.assertEqual(res["actions"], [ACTION_SUCK, ACTION_LEFT, ACTION_SUCK])
        self.assertTrue(res["goal"].is_goal)
        self.assertEqual(res["goal"].state_id, 7)

    def test_shortest_path_from_one_dirty_location(self) -> None:
        """Verify optimal path when only current room is dirty takes 1 step."""
        s4 = self.space.state_map[("B", "Clean", "Dirty")]
        res4 = self.space.find_shortest_paths_to_goal(s4)
        self.assertEqual(res4["distance"], 1)
        self.assertEqual(res4["actions"], [ACTION_SUCK])
        self.assertEqual(res4["goal"].state_id, 8)

        s5 = self.space.state_map[("A", "Dirty", "Clean")]
        res5 = self.space.find_shortest_paths_to_goal(s5)
        self.assertEqual(res5["distance"], 1)
        self.assertEqual(res5["actions"], [ACTION_SUCK])
        self.assertEqual(res5["goal"].state_id, 7)

    def test_shortest_path_from_goal_is_zero(self) -> None:
        """Verify distance from goal state to goal is 0 steps."""
        s7 = self.space.state_map[("A", "Clean", "Clean")]
        res7 = self.space.find_shortest_paths_to_goal(s7)
        self.assertEqual(res7["distance"], 0)
        self.assertEqual(res7["actions"], [])
        self.assertEqual(res7["goal"], s7)


class TestVisualizerAndLayout(unittest.TestCase):
    """Test suite for graph plotting, layouts, and ASCII rendering."""

    def setUp(self) -> None:
        self.space = StateSpace()
        self.viz = StateSpaceVisualizer(self.space)

    def test_textbook_layout_coordinates(self) -> None:
        """Verify 2-column textbook layout coordinates."""
        pos = self.viz.get_structured_layout()
        self.assertEqual(len(pos), 8)

        # Location A must have negative x coordinate (-1.6)
        # Location B must have positive x coordinate (1.6)
        for s, (x, y) in pos.items():
            if s.agent_loc == LOCATION_A:
                self.assertAlmostEqual(x, -1.6)
            else:
                self.assertAlmostEqual(x, 1.6)

            # Both clean (Goal) must have y == 0.0
            if s.is_goal:
                self.assertAlmostEqual(y, 0.0)

    def test_compute_positions_algorithms(self) -> None:
        """Verify various layout algorithms compute 8 node positions."""
        g = self.space.to_networkx_graph()
        for mode in ("textbook", "circular", "spring", "shell"):
            pos = self.viz.compute_positions(g, layout_type=mode)
            self.assertEqual(len(pos), 8)

    def test_plot_graph_headless(self) -> None:
        """Verify plot_graph renders Matplotlib figure without raising errors."""
        fig, ax = self.viz.plot_graph(layout_type="textbook", show=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_graph_save_to_file(self) -> None:
        """Verify plot_graph saves figure to specified file path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_graph.png"
            fig, ax = self.viz.plot_graph(
                layout_type="textbook",
                save_path=out_file,
                show=False,
            )
            plt.close(fig)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 0)

    def test_render_ascii_graph(self) -> None:
        """Verify ASCII transition table rendering contains all 8 states."""
        ascii_text = self.viz.render_ascii_graph()
        for i in range(1, 9):
            self.assertIn(f"S{i}", ascii_text)
        self.assertIn("[GOAL]", ascii_text)
        self.assertIn("Total States: 8", ascii_text)
        self.assertIn("Self-Loop Cycles (↺): 12", ascii_text)

    def test_viva_reference_content(self) -> None:
        """Verify viva Voce reference covers all syllabus requirements."""
        viva = get_viva_reference()
        self.assertIn("STATE SPACE SIZE CALCULATION", viva)
        self.assertIn("FORMULATION OF SEARCH PROBLEMS", viva)
        self.assertIn("CYCLIC TRANSITIONS & REVISION DYNAMICS", viva)
        self.assertIn("GRAPH VISUALIZATION TECHNIQUES", viva)
        self.assertIn("|S| = n * (k ^ n)", viva)


if __name__ == "__main__":
    unittest.main()
