"""Unit tests for Missionaries and Cannibals DFS Solver (Question 21).

Validates state-space formulation, the 5 canonical moves, constraint checking
during generation vs testing, loop avoidance, DFS traversal (iterative and recursive),
BFS optimality comparison, state space cardinality, formatting, and viva guide.
"""

import io
import unittest
from typing import List, Set, Tuple

from src.missionaries_cannibals_dfs import (
    BOAT_LEFT,
    BOAT_RIGHT,
    CANONICAL_5_MOVES,
    DEFAULT_BOAT_CAPACITY,
    DEFAULT_TOTAL_CANNIBALS,
    DEFAULT_TOTAL_MISSIONARIES,
    INITIAL_STATE,
    GOAL_STATE,
    MOVE_1C,
    MOVE_1M,
    MOVE_1M1C,
    MOVE_2C,
    MOVE_2M,
    OPTIMAL_DFS_MOVE_ORDER,
    VALID_BOAT_POSITIONS,
    BoatMove,
    MissionariesCannibalsResult,
    MissionariesCannibalsSolver,
    State,
    TransitionStep,
    animate_solution,
    apply_move,
    analyze_state_space,
    format_solution_trace,
    get_action_label,
    get_missionaries_cannibals_viva_reference,
    is_move_physically_possible,
    is_valid_state,
    render_ascii_river,
    solve_missionaries_cannibals_dfs,
    validate_state_detailed,
)


class TestStateRepresentationAndConstraints(unittest.TestCase):
    """Test validation of (M_left, C_left, Boat_pos) states and safety constraints."""

    def test_initial_and_goal_states_are_valid(self) -> None:
        """Initial state (3, 3, 'Left') and Goal state (0, 0, 'Right') must be safe."""
        self.assertTrue(is_valid_state(INITIAL_STATE))
        self.assertTrue(is_valid_state(GOAL_STATE))
        self.assertEqual(INITIAL_STATE, (3, 3, "Left"))
        self.assertEqual(GOAL_STATE, (0, 0, "Right"))

    def test_numerical_bounds_validation(self) -> None:
        """States with negative people or counts exceeding total must be rejected."""
        self.assertFalse(is_valid_state((-1, 3, BOAT_LEFT)))
        self.assertFalse(is_valid_state((3, -1, BOAT_LEFT)))
        self.assertFalse(is_valid_state((4, 3, BOAT_LEFT)))
        self.assertFalse(is_valid_state((3, 4, BOAT_LEFT)))
        self.assertFalse(is_valid_state((3, 3, "Middle")))

    def test_left_bank_cannibal_safety_constraint(self) -> None:
        """If M_left > 0, cannibals cannot outnumber missionaries on the left bank."""
        # 1M, 2C -> Cannibals outnumber missionaries on Left bank
        self.assertFalse(is_valid_state((1, 2, BOAT_LEFT)))
        # 1M, 3C -> Unsafe
        self.assertFalse(is_valid_state((1, 3, BOAT_LEFT)))
        # 2M, 3C -> Unsafe
        self.assertFalse(is_valid_state((2, 3, BOAT_LEFT)))
        # 2M, 2C -> Safe (M >= C)
        self.assertTrue(is_valid_state((2, 2, BOAT_LEFT)))
        # 1M, 1C -> Safe (M >= C)
        self.assertTrue(is_valid_state((1, 1, BOAT_LEFT)))

    def test_right_bank_cannibal_safety_constraint(self) -> None:
        """If M_right > 0, cannibals cannot outnumber missionaries on the right bank."""
        # Left has (2, 1) -> Right has (1M, 2C) -> Right bank unsafe!
        self.assertFalse(is_valid_state((2, 1, BOAT_LEFT)))
        # Left has (1, 0) -> Right has (2M, 3C) -> Right bank unsafe!
        self.assertFalse(is_valid_state((1, 0, BOAT_LEFT)))
        # Left has (2, 0) -> Right has (1M, 3C) -> Right bank unsafe!
        self.assertFalse(is_valid_state((2, 0, BOAT_LEFT)))

    def test_zero_missionary_bank_is_safe(self) -> None:
        """Banks with 0 missionaries are safe regardless of cannibal count (no one to eat)."""
        # Left has 0M, 3C -> Right has 3M, 0C -> Both safe!
        self.assertTrue(is_valid_state((0, 3, BOAT_LEFT)))
        # Left has 0M, 2C -> Right has 3M, 1C -> 3M >= 1C, safe!
        self.assertTrue(is_valid_state((0, 2, BOAT_LEFT)))
        # Left has 0M, 1C -> Right has 3M, 2C -> 3M >= 2C, safe!
        self.assertTrue(is_valid_state((0, 1, BOAT_LEFT)))
        # Left has 3M, 0C -> Right has 0M, 3C -> Both safe!
        self.assertTrue(is_valid_state((3, 0, BOAT_LEFT)))

    def test_detailed_validation_diagnostics(self) -> None:
        """Verify detailed error messages explain the exact constraint violation."""
        valid, msg = validate_state_detailed((1, 2, BOAT_LEFT))
        self.assertFalse(valid)
        self.assertIn("outnumber", msg)
        self.assertIn("Left bank", msg)

        valid_r, msg_r = validate_state_detailed((2, 1, BOAT_RIGHT))
        self.assertFalse(valid_r)
        self.assertIn("Right bank", msg_r)

        valid_ok, msg_ok = validate_state_detailed((3, 3, BOAT_LEFT))
        self.assertTrue(valid_ok)
        self.assertIn("Safe state", msg_ok)


class TestCanonical5Moves(unittest.TestCase):
    """Test validation and properties of the 5 canonical boat moves."""

    def test_canonical_moves_presence_and_properties(self) -> None:
        """The 5 canonical moves (1M, 2M, 1C, 2C, 1M1C) must have 1-2 passengers."""
        self.assertEqual(len(CANONICAL_5_MOVES), 5)
        labels = {m.label for m in CANONICAL_5_MOVES}
        self.assertEqual(labels, {"1M", "2M", "1C", "2C", "1M1C"})

        for move in CANONICAL_5_MOVES:
            self.assertGreaterEqual(move.total_passengers, 1)
            self.assertLessEqual(move.total_passengers, DEFAULT_BOAT_CAPACITY)

    def test_move_feasibility_on_departure_banks(self) -> None:
        """Moves require sufficient passengers on the departure bank."""
        # Initial state (3, 3, Left): Left has 3M, 3C. All 5 moves physically possible.
        for move in CANONICAL_5_MOVES:
            self.assertTrue(is_move_physically_possible(INITIAL_STATE, move))

        # State (0, 2, Left): Left has 0M, 2C. 1M, 2M, 1M1C cannot board from Left.
        st_no_m: State = (0, 2, BOAT_LEFT)
        self.assertFalse(is_move_physically_possible(st_no_m, MOVE_1M))
        self.assertFalse(is_move_physically_possible(st_no_m, MOVE_2M))
        self.assertFalse(is_move_physically_possible(st_no_m, MOVE_1M1C))
        self.assertTrue(is_move_physically_possible(st_no_m, MOVE_1C))
        self.assertTrue(is_move_physically_possible(st_no_m, MOVE_2C))

        # State (3, 1, Right): Right bank has (3 - 3 = 0M, 3 - 1 = 2C).
        st_right: State = (3, 1, BOAT_RIGHT)
        self.assertFalse(is_move_physically_possible(st_right, MOVE_1M))
        self.assertTrue(is_move_physically_possible(st_right, MOVE_1C))
        self.assertTrue(is_move_physically_possible(st_right, MOVE_2C))

    def test_directional_move_application(self) -> None:
        """Moving from Left subtracts counts and sets Boat to Right; vice versa."""
        # From (3, 3, Left) with 2C
        next_st = apply_move(INITIAL_STATE, MOVE_2C)
        self.assertEqual(next_st, (3, 1, BOAT_RIGHT))

        # From (3, 1, Right) with 1C returning to Left
        return_st = apply_move(next_st, MOVE_1C)
        self.assertEqual(return_st, (3, 2, BOAT_LEFT))


class TestSuccessorGenerationAndConstraintPruning(unittest.TestCase):
    """Test constraint checking in state generation vs testing."""

    def setUp(self) -> None:
        self.solver = MissionariesCannibalsSolver()

    def test_initial_state_valid_successors(self) -> None:
        """From (3, 3, Left), only 1C, 2C, and 1M1C lead to safe states."""
        succs, p_cap, p_safe = self.solver.generate_successors_during_generation(INITIAL_STATE)
        succ_states = [s[1] for s in succs]
        # 1M would lead to (2, 3, Right) -> unsafe (cannibals outnumber M on Left)
        # 2M would lead to (1, 3, Right) -> unsafe (cannibals outnumber M on Left)
        self.assertEqual(p_safe, 2)
        self.assertEqual(p_cap, 0)
        self.assertEqual(len(succs), 3)
        self.assertIn((3, 2, BOAT_RIGHT), succ_states)
        self.assertIn((3, 1, BOAT_RIGHT), succ_states)
        self.assertIn((2, 2, BOAT_RIGHT), succ_states)

    def test_generation_vs_testing_yields_identical_valid_successors(self) -> None:
        """Both validation strategies must produce identical sets of valid successor states."""
        test_states: List[State] = [
            (3, 3, BOAT_LEFT),
            (3, 1, BOAT_RIGHT),
            (3, 2, BOAT_LEFT),
            (0, 2, BOAT_RIGHT),
            (0, 3, BOAT_LEFT),
        ]
        for st in test_states:
            succ_gen, _, _ = self.solver.generate_successors_during_generation(st)
            succ_test, _, _ = self.solver.generate_successors_during_testing(st)
            states_gen = [s[1] for s in succ_gen]
            states_test = [s[1] for s in succ_test]
            self.assertEqual(set(states_gen), set(states_test))


class TestDFSSearchOptimalityAndCycleDetection(unittest.TestCase):
    """Test DFS execution, loop avoidance, and solution trace validation."""

    def setUp(self) -> None:
        self.solver = MissionariesCannibalsSolver()

    def test_dfs_finds_solution_to_goal(self) -> None:
        """DFS must find a path from (3, 3, Left) to (0, 0, Right)."""
        result = self.solver.solve_dfs(method="iterative")
        self.assertTrue(result.success)
        self.assertEqual(result.path[0], (3, 3, "Left"))
        self.assertEqual(result.path[-1], (0, 0, "Right"))
        self.assertEqual(result.total_trips, 11)

    def test_recursive_dfs_matches_solution(self) -> None:
        """Recursive DFS implementation must also find the solution safely."""
        result = self.solver.solve_dfs(method="recursive")
        self.assertTrue(result.success)
        self.assertEqual(result.path[0], (3, 3, "Left"))
        self.assertEqual(result.path[-1], (0, 0, "Right"))
        self.assertEqual(result.total_trips, 11)

    def test_every_state_in_solution_path_is_safe(self) -> None:
        """Every single state visited during the crossing sequence must satisfy constraints."""
        result = self.solver.solve_dfs()
        for i, st in enumerate(result.path):
            self.assertTrue(
                is_valid_state(st),
                f"State at index {i} violates safety constraint: {st}",
            )

    def test_consecutive_states_are_valid_transitions(self) -> None:
        """Each consecutive pair of states must correspond to a legal boat crossing."""
        result = self.solver.solve_dfs()
        for i in range(len(result.path) - 1):
            curr_st = result.path[i]
            next_st = result.path[i + 1]
            transition = result.transitions[i]

            self.assertEqual(transition.from_state, curr_st)
            self.assertEqual(transition.to_state, next_st)
            # Opposite boat positions
            self.assertNotEqual(curr_st[2], next_st[2])

            # Recomputed delta
            if curr_st[2] == BOAT_LEFT:
                dm = curr_st[0] - next_st[0]
                dc = curr_st[1] - next_st[1]
            else:
                dm = next_st[0] - curr_st[0]
                dc = next_st[1] - curr_st[1]

            self.assertEqual((dm, dc), transition.move.move_tuple)
            self.assertIn(transition.move.total_passengers, (1, 2))

    def test_sample_output_first_step_matches(self) -> None:
        """Step 1 must be '2C cross right' as specified in sample output."""
        result = self.solver.solve_dfs()
        first_step = result.transitions[0]
        self.assertEqual(first_step.step_number, 1)
        self.assertEqual(first_step.action, "2C cross right")
        self.assertEqual(first_step.to_state, (3, 1, "Right"))

    def test_cycle_detection_prunes_revisited_states(self) -> None:
        """Cycle detection counter must be strictly positive, proving loop prevention."""
        result = self.solver.solve_dfs(method="iterative")
        self.assertGreater(result.pruned_by_cycle, 0)


class TestStateSpaceAnalysisAndBFS(unittest.TestCase):
    """Test state space metrics, graph reachability, and BFS comparison."""

    def test_state_space_counts(self) -> None:
        """Verify 32 theoretical states, 20 safe states, and 16 reachable states."""
        analysis = analyze_state_space()
        self.assertEqual(analysis["total_theoretical_states"], 32)
        self.assertEqual(analysis["total_safe_states"], 20)
        self.assertEqual(analysis["total_reachable_safe_states"], 16)
        self.assertEqual(analysis["total_unreachable_safe_states"], 4)

    def test_bfs_optimality(self) -> None:
        """BFS must find the minimal path length of exactly 11 boat trips."""
        solver = MissionariesCannibalsSolver()
        bfs_res = solver.solve_bfs()
        self.assertTrue(bfs_res.success)
        self.assertEqual(bfs_res.total_trips, 11)
        self.assertEqual(bfs_res.path[-1], (0, 0, "Right"))

    def test_dfs_vs_bfs_comparison(self) -> None:
        """compare_dfs_vs_bfs must return valid metrics for both algorithms."""
        solver = MissionariesCannibalsSolver()
        cmp_data = solver.compare_dfs_vs_bfs()
        self.assertIn("dfs", cmp_data)
        self.assertIn("bfs", cmp_data)
        self.assertTrue(cmp_data["dfs"]["success"])
        self.assertTrue(cmp_data["bfs"]["success"])
        self.assertEqual(cmp_data["bfs"]["trips"], 11)


class TestFormattingAndAnimation(unittest.TestCase):
    """Test string formatters, ASCII rendering, and animation routine."""

    def setUp(self) -> None:
        self.solver = MissionariesCannibalsSolver()
        self.result = self.solver.solve_dfs()

    def test_format_traces(self) -> None:
        """All trace formats (standard, arrows, table, detailed) must render without error."""
        std_trace = format_solution_trace(self.result, format_type="standard")
        self.assertIn("Step 1: 2C cross right", std_trace)
        self.assertIn("Goal Reached:", std_trace)

        arrow_trace = format_solution_trace(self.result, format_type="arrows")
        self.assertIn("--[2C cross right]-->", arrow_trace)

        table_trace = format_solution_trace(self.result, format_type="table")
        self.assertIn("+------+", table_trace)
        self.assertIn("Left Bank", table_trace)

        detailed_trace = format_solution_trace(self.result, format_type="detailed")
        self.assertIn("MISSIONARIES & CANNIBALS", detailed_trace)

    def test_render_ascii_river(self) -> None:
        """render_ascii_river must format river and bank representations."""
        ascii_str = render_ascii_river(INITIAL_STATE)
        self.assertIn("Left Bank:", ascii_str)
        self.assertIn("[BOAT]", ascii_str)
        self.assertIn("Right Bank", ascii_str)

    def test_animation_execution_zero_delay(self) -> None:
        """animate_solution must write animated frames to stream without error."""
        buf = io.StringIO()
        animate_solution(self.result, delay=0.0, stream=buf)
        output = buf.getvalue()
        self.assertIn("MISSIONARIES & CANNIBALS RIVER CROSSING ANIMATION", output)
        self.assertIn("ALL 3 MISSIONARIES AND 3 CANNIBALS SAFELY CROSSED", output)

    def test_viva_reference_completeness(self) -> None:
        """Viva guide must cover all mandatory topics with comprehensive answers."""
        viva = get_missionaries_cannibals_viva_reference()
        expected_topics = [
            "Constraint Validation: Generation vs Testing",
            "State Space Size and Reachability",
            "Why BFS is Preferred Over DFS",
            "Generalization to N Missionaries and N Cannibals",
            "Cycle Detection in DFS",
        ]
        for topic in expected_topics:
            self.assertIn(topic, viva)
            self.assertGreater(len(viva[topic]["question"]), 10)
            self.assertGreater(len(viva[topic]["answer"]), 50)


if __name__ == "__main__":
    unittest.main()
