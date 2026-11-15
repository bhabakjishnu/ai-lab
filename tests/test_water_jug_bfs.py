"""Unit tests for Water Jug Problem BFS Solver (Question 20).

Validates state-space formulation, the 6 canonical production rules,
algebraic reachability checks (Bézout's identity and GCD), BFS search optimality,
cycle avoidance, edge cases, and serialization.
"""

import json
import math
import unittest
from typing import Set, Tuple

from src.water_jug_bfs import (
    COPRIME_WATER_JUG_PARAMS,
    DIE_HARD_WATER_JUG_PARAMS,
    EQUAL_CAPACITY_PARAMS,
    RULE_DEFINITIONS,
    SAMPLE_WATER_JUG_PARAMS,
    UNSOLVABLE_WATER_JUG_PARAMS,
    AlgebraicReachabilityResult,
    WaterJugAction,
    WaterJugSearchResult,
    WaterJugSolver,
    WaterJugState,
    check_reachability_algebraic,
    extended_gcd,
    format_state_arrow_path,
    get_water_jug_viva_reference,
    render_ascii_water_jugs,
    solve_water_jug_bfs,
)


class TestWaterJugSuccessorRules(unittest.TestCase):
    """Test verification of the 6 canonical production rules and precondition pruning."""

    def setUp(self) -> None:
        """Create solver with canonical capacities (4, 3)."""
        self.solver = WaterJugSolver(capacity_x=4, capacity_y=3, target_z=2, target_jug="1")

    def test_initial_state_successors(self) -> None:
        """From (0, 0), only Fill 1 and Fill 2 should be valid transitions."""
        successors = self.solver.get_successors((0, 0))
        states = [s[0] for s in successors]
        actions = [s[1] for s in successors]

        self.assertEqual(len(successors), 2)
        self.assertIn((4, 0), states)
        self.assertIn((0, 3), states)
        self.assertIn(WaterJugAction.FILL_1.value, actions)
        self.assertIn(WaterJugAction.FILL_2.value, actions)

    def test_fill_rules_pruning_when_full(self) -> None:
        """When Jug 1 is full, Fill 1 must be pruned. When Jug 2 is full, Fill 2 must be pruned."""
        succ_full_1 = self.solver.get_successors((4, 0))
        actions_1 = [s[1] for s in succ_full_1]
        self.assertNotIn(WaterJugAction.FILL_1.value, actions_1)

        succ_full_2 = self.solver.get_successors((0, 3))
        actions_2 = [s[1] for s in succ_full_2]
        self.assertNotIn(WaterJugAction.FILL_2.value, actions_2)

    def test_empty_rules_pruning_when_empty(self) -> None:
        """When a jug is empty, its empty rule must be pruned."""
        succ = self.solver.get_successors((0, 2))
        actions = [s[1] for s in succ]
        self.assertNotIn(WaterJugAction.EMPTY_1.value, actions)
        self.assertIn(WaterJugAction.EMPTY_2.value, actions)

    def test_pour_1_to_2_overflow_capping(self) -> None:
        """Pouring from (4, 0) into Jug 2 (capacity 3) caps at 3, leaving 1 in Jug 1."""
        succ = self.solver.get_successors((4, 0))
        pour_succ = [s for s in succ if s[1] == WaterJugAction.POUR_1_TO_2.value]
        self.assertEqual(len(pour_succ), 1)
        next_state, _, rule_num, desc = pour_succ[0]
        self.assertEqual(next_state, (1, 3))
        self.assertEqual(rule_num, 5)
        self.assertIn("Pour 3L", desc)

    def test_pour_1_to_2_partial(self) -> None:
        """Pouring from (1, 0) into Jug 2 (capacity 3) empties Jug 1 completely."""
        succ = self.solver.get_successors((1, 0))
        pour_succ = [s for s in succ if s[1] == WaterJugAction.POUR_1_TO_2.value]
        self.assertEqual(len(pour_succ), 1)
        next_state = pour_succ[0][0]
        self.assertEqual(next_state, (0, 1))

    def test_pour_2_to_1_overflow_capping(self) -> None:
        """Pouring from (3, 3) into Jug 1 (capacity 4) caps at 4, leaving 2 in Jug 2."""
        succ = self.solver.get_successors((3, 3))
        pour_succ = [s for s in succ if s[1] == WaterJugAction.POUR_2_TO_1.value]
        self.assertEqual(len(pour_succ), 1)
        next_state, _, rule_num, desc = pour_succ[0]
        self.assertEqual(next_state, (4, 2))
        self.assertEqual(rule_num, 6)
        self.assertIn("Pour 1L", desc)

    def test_rule_definitions_completeness(self) -> None:
        """Verify all 6 rules have complete metadata in RULE_DEFINITIONS."""
        self.assertEqual(len(RULE_DEFINITIONS), 6)
        for act in WaterJugAction:
            self.assertIn(act.value, RULE_DEFINITIONS)
            meta = RULE_DEFINITIONS[act.value]
            self.assertIn("rule_number", meta)
            self.assertIn("description", meta)
            self.assertIn("precondition", meta)
            self.assertIn("effect", meta)


class TestAlgebraicReachabilityAndGCD(unittest.TestCase):
    """Test algebraic reachability verification using Bézout's identity and GCD."""

    def test_extended_gcd_correctness(self) -> None:
        """Extended GCD must satisfy: a*x + b*y == gcd(a, b)."""
        test_pairs = [(4, 3), (5, 3), (7, 5), (6, 4), (12, 8), (17, 13)]
        for a, b in test_pairs:
            g, x, y = extended_gcd(a, b)
            self.assertEqual(g, math.gcd(a, b))
            self.assertEqual(a * x + b * y, g)

    def test_canonical_sample_algebraic_reachable(self) -> None:
        """Capacities 4, 3 with Target 2 is algebraically reachable."""
        res = check_reachability_algebraic(4, 3, 2, target_jug="1")
        self.assertTrue(res.is_reachable)
        self.assertEqual(res.gcd_value, 1)

    def test_die_hard_algebraic_reachable(self) -> None:
        """Capacities 5, 3 with Target 4 is algebraically reachable."""
        res = check_reachability_algebraic(5, 3, 4, target_jug="1")
        self.assertTrue(res.is_reachable)
        self.assertEqual(res.gcd_value, 1)

    def test_gcd_divisibility_violation(self) -> None:
        """Capacities 6, 4 with Target 5 is algebraically UNREACHABLE because gcd(6, 4) = 2 does not divide 5."""
        res = check_reachability_algebraic(6, 4, 5, target_jug="any")
        self.assertFalse(res.is_reachable)
        self.assertEqual(res.gcd_value, 2)
        self.assertIn("GCD divisibility violation", res.reason)

    def test_capacity_exceeded_unreachable(self) -> None:
        """Target exceeding maximum capacity is unreachable."""
        res = check_reachability_algebraic(3, 5, 7, target_jug="any")
        self.assertFalse(res.is_reachable)
        self.assertIn("exceeds maximum jug capacity", res.reason)

    def test_jug1_capacity_exceeded_unreachable(self) -> None:
        """Target exceeding Jug 1 capacity when target_jug=1 is unreachable."""
        res = check_reachability_algebraic(3, 7, 5, target_jug="1")
        self.assertFalse(res.is_reachable)
        self.assertIn("exceeds Jug 1 capacity", res.reason)

    def test_zero_target_reachable(self) -> None:
        """Target 0 is always reachable at (0, 0)."""
        res = check_reachability_algebraic(4, 3, 0, target_jug="1")
        self.assertTrue(res.is_reachable)
        self.assertIn("Target 0 is trivially reachable", res.reason)

    def test_negative_target_unreachable(self) -> None:
        """Negative target is invalid."""
        res = check_reachability_algebraic(4, 3, -1, target_jug="1")
        self.assertFalse(res.is_reachable)
        self.assertIn("cannot be negative", res.reason)

    def test_invalid_capacities_unreachable(self) -> None:
        """Zero or negative capacities are invalid."""
        res = check_reachability_algebraic(0, 3, 2)
        self.assertFalse(res.is_reachable)
        self.assertIn("strictly positive integers", res.reason)


class TestWaterJugBFSSolver(unittest.TestCase):
    """Test full BFS search execution, optimality, and syllabus conformance."""

    def test_canonical_sample_output(self) -> None:
        """Question 20 sample test: Capacities 4, 3; Target 2 in Jug 1.

        Expected Output:
        (0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)
        """
        res = solve_water_jug_bfs(
            capacity_x=SAMPLE_WATER_JUG_PARAMS["capacity_x"],
            capacity_y=SAMPLE_WATER_JUG_PARAMS["capacity_y"],
            target_z=SAMPLE_WATER_JUG_PARAMS["target_z"],
            target_jug="1",
        )

        expected_states: list[WaterJugState] = [
            (0, 0),
            (4, 0),
            (1, 3),
            (1, 0),
            (0, 1),
            (4, 1),
            (2, 3),
        ]
        expected_arrow = "(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)"

        self.assertTrue(res.is_solvable)
        self.assertEqual(res.path, expected_states)
        self.assertEqual(res.arrow_path_string, expected_arrow)
        self.assertEqual(res.step_count, 6)
        self.assertEqual(res.path[-1][0], 2)

    def test_target_in_any_jug_finds_shorter_path(self) -> None:
        """When measuring in either jug, (4, 2) is found in only 4 steps."""
        res = solve_water_jug_bfs(capacity_x=4, capacity_y=3, target_z=2, target_jug="any")
        self.assertTrue(res.is_solvable)
        self.assertEqual(res.step_count, 4)
        expected_states: list[WaterJugState] = [(0, 0), (0, 3), (3, 0), (3, 3), (4, 2)]
        self.assertEqual(res.path, expected_states)
        self.assertEqual(res.path[-1], (4, 2))

    def test_target_in_jug_2(self) -> None:
        """Test targeting Jug 2 explicitly."""
        res = solve_water_jug_bfs(capacity_x=4, capacity_y=3, target_z=2, target_jug="2")
        self.assertTrue(res.is_solvable)
        self.assertEqual(res.path[-1][1], 2)
        self.assertEqual(res.step_count, 4)

    def test_die_hard_problem(self) -> None:
        """Classic Die Hard 3 problem: Capacities 5, 3; Target 4."""
        res = solve_water_jug_bfs(
            capacity_x=DIE_HARD_WATER_JUG_PARAMS["capacity_x"],
            capacity_y=DIE_HARD_WATER_JUG_PARAMS["capacity_y"],
            target_z=DIE_HARD_WATER_JUG_PARAMS["target_z"],
            target_jug="1",
        )
        self.assertTrue(res.is_solvable)
        self.assertEqual(res.path[-1][0], 4)
        self.assertEqual(res.step_count, 6)
        expected_path: list[WaterJugState] = [
            (0, 0),
            (5, 0),
            (2, 3),
            (2, 0),
            (0, 2),
            (5, 2),
            (4, 3),
        ]
        self.assertEqual(res.path, expected_path)

    def test_unsolvable_case_returns_unreached(self) -> None:
        """Unsolvable configuration (6, 4, target 5) must terminate with is_solvable=False."""
        res = solve_water_jug_bfs(
            capacity_x=UNSOLVABLE_WATER_JUG_PARAMS["capacity_x"],
            capacity_y=UNSOLVABLE_WATER_JUG_PARAMS["capacity_y"],
            target_z=UNSOLVABLE_WATER_JUG_PARAMS["target_z"],
            target_jug="any",
        )
        self.assertFalse(res.is_solvable)
        self.assertEqual(len(res.path), 0)
        self.assertEqual(res.step_count, 0)
        self.assertIn("No solution", res.arrow_path_string)

    def test_target_zero_immediate_termination(self) -> None:
        """Target 0 is already satisfied at (0, 0) with 0 steps."""
        res = solve_water_jug_bfs(capacity_x=4, capacity_y=3, target_z=0, target_jug="1")
        self.assertTrue(res.is_solvable)
        self.assertEqual(res.step_count, 0)
        self.assertEqual(res.path, [(0, 0)])

    def test_custom_start_state(self) -> None:
        """Starting from an intermediate state (e.g., (4, 0)) correctly solves remainder."""
        solver = WaterJugSolver(capacity_x=4, capacity_y=3, target_z=2, target_jug="1")
        res = solver.solve_bfs(start_state=(4, 0))
        self.assertTrue(res.is_solvable)
        self.assertEqual(res.path[0], (4, 0))
        self.assertEqual(res.path[-1][0], 2)
        self.assertEqual(res.step_count, 5)

    def test_find_all_reachable_states(self) -> None:
        """For Capacities (4, 3), reachable states from (0,0) must not exceed (4+1)*(3+1) = 20."""
        solver = WaterJugSolver(capacity_x=4, capacity_y=3, target_z=2)
        all_states = solver.find_all_reachable_states()
        self.assertIsInstance(all_states, set)
        self.assertIn((0, 0), all_states)
        self.assertIn((4, 0), all_states)
        self.assertIn((2, 3), all_states)
        self.assertLessEqual(len(all_states), (4 + 1) * (3 + 1))
        # For gcd(4,3)=1, all perimeter states are reachable (14 states)
        self.assertEqual(len(all_states), 14)


class TestWaterJugFormattingAndOutput(unittest.TestCase):
    """Test tabular formatting, ASCII visualization, and serialization."""

    def setUp(self) -> None:
        """Run canonical search for formatting tests."""
        self.res = solve_water_jug_bfs(4, 3, 2, target_jug="1")

    def test_format_table_structure(self) -> None:
        """Check trace table format and column headers."""
        tbl = self.res.format_table()
        self.assertIn("Step", tbl)
        self.assertIn("Action", tbl)
        self.assertIn("Rule #", tbl)
        self.assertIn("State (J1,J2)", tbl)
        self.assertIn("Fill 1", tbl)
        self.assertIn("Pour 1->2", tbl)
        self.assertIn("Empty 2", tbl)

    def test_format_summary_fields(self) -> None:
        """Summary contains all critical problem metrics."""
        summary = self.res.format_summary()
        self.assertIn("Jug 1 Capacity (X)   : 4 L", summary)
        self.assertIn("Jug 2 Capacity (Y)   : 3 L", summary)
        self.assertIn("Target Water (Z)     : 2 L", summary)
        self.assertIn("Algebraic Reachable  : Yes", summary)
        self.assertIn("Shortest Path Length : 6 steps", summary)

    def test_ascii_visual_diagram(self) -> None:
        """Verify ASCII water jugs visual generation."""
        ascii_art = render_ascii_water_jugs(4, 3, 2, 3)
        self.assertIn("Jug 1 (2/4 L)", ascii_art)
        self.assertIn("Jug 2 (3/3 L)", ascii_art)
        self.assertIn("+-------------+", ascii_art)

        visual_path = self.res.format_ascii_visual_path()
        self.assertIn("Step 1: Fill 1", visual_path)
        self.assertIn("Step 6: Pour 1->2", visual_path)

    def test_json_serialization(self) -> None:
        """Test to_dict and to_json serialize and deserialize accurately."""
        d = self.res.to_dict()
        self.assertEqual(d["capacities"], [4, 3])
        self.assertEqual(d["target_z"], 2)
        self.assertEqual(d["step_count"], 6)

        json_str = self.res.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["capacities"], [4, 3])
        self.assertEqual(data["is_solvable"], True)

    def test_viva_reference_guide(self) -> None:
        """Verify viva guide contains comprehensive answers for required topics."""
        viva = get_water_jug_viva_reference()
        self.assertIn("1. State Space Representation", viva)
        self.assertIn("2. Action Rules Definition", viva)
        self.assertIn("3. Unsolvable States & GCD Rule", viva)
        self.assertIn("4. BFS Optimality and Completeness", viva)
        self.assertIn("5. Time and Space Complexity", viva)

        for topic, content in viva.items():
            self.assertIn("question", content)
            self.assertIn("answer", content)
            self.assertGreater(len(content["answer"]), 50)


class TestInputValidationAndEdgeCases(unittest.TestCase):
    """Test boundary conditions and error raising."""

    def test_invalid_capacity_raises_value_error(self) -> None:
        """Non-positive capacity raises ValueError."""
        with self.assertRaises(ValueError):
            WaterJugSolver(capacity_x=0, capacity_y=3, target_z=2)
        with self.assertRaises(ValueError):
            WaterJugSolver(capacity_x=4, capacity_y=-2, target_z=2)

    def test_negative_target_raises_value_error(self) -> None:
        """Negative target raises ValueError."""
        with self.assertRaises(ValueError):
            WaterJugSolver(capacity_x=4, capacity_y=3, target_z=-1)

    def test_invalid_start_state_raises_value_error(self) -> None:
        """Start state out of bounds raises ValueError."""
        solver = WaterJugSolver(capacity_x=4, capacity_y=3, target_z=2)
        with self.assertRaises(ValueError):
            solver.solve_bfs(start_state=(5, 0))
        with self.assertRaises(ValueError):
            solver.solve_bfs(start_state=(0, -1))

    def test_format_state_arrow_path_empty(self) -> None:
        """Empty path formats as ()"""
        self.assertEqual(format_state_arrow_path([]), "()")


if __name__ == "__main__":
    unittest.main()
