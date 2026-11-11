"""Unit tests for Depth-Limited Search (DLS) (Question 16).

Validates:
- Exact syllabus sample input: Limit: 2, Goal at depth 3 -> Result: Cutoff.
- Success state: Limit: 3, Goal at depth 3 -> Result: Path ['1', '2', '4', '6'].
- Failure state: Goal completely absent / unreachable -> Result: Failure.
- Cutoff vs Failure semantic distinction (essential for Iterative Deepening Search).
- Boundary conditions:
  - Start is Goal with Limit = 0 -> Success.
  - Start is not Goal with Limit = 0 -> Cutoff.
  - Negative limit (< 0) -> ValueError.
  - Missing start node -> ValueError.
  - Empty start/goal strings -> ValueError.
- Optional Challenge:
  - Tracking deepest node reached before cutoff.
  - Multiple deepest nodes at maximum depth.
  - Cutoff nodes tracking.
- Cyclic graphs and cycle prevention mechanics.
- Functional `dls(...)` return types (List[str] on success, 'Cutoff', 'Failure').
- Object-oriented `DepthLimitedSearch` class interface.
- Graph normalization and multi-format input parsing.
- Headless visualization generation.
- Viva Voce theoretical reference guide contents.
"""

from pathlib import Path
import tempfile
import unittest

from src.depth_limited_search import (
    BINARY_TREE_DLS_GRAPH,
    CYCLIC_DLS_GRAPH,
    DISCONNECTED_DLS_GRAPH,
    LINEAR_CHAIN_DLS_GRAPH,
    SAMPLE_DLS_GRAPH,
    CUTOFF,
    FAILURE,
    SUCCESS,
    DLSResult,
    DLSStepRecord,
    DepthLimitedSearch,
    depth_limited_search,
    dls,
    get_dls_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_dls,
)


class TestDepthLimitedSearchCore(unittest.TestCase):
    """Core algorithmic tests for Depth-Limited Search."""

    def test_sample_input_cutoff(self) -> None:
        """Verify exact sample input from Question 16 prompt: Limit: 2, Goal at depth 3 -> Cutoff."""
        result = depth_limited_search(
            graph=SAMPLE_DLS_GRAPH,
            start="1",
            goal="6",
            limit=2,
        )
        self.assertEqual(result.status, CUTOFF)
        self.assertTrue(result.is_cutoff())
        self.assertFalse(result.is_success())
        self.assertFalse(result.is_failure())
        self.assertIsNone(result.path)
        self.assertEqual(result.format_result(), "Result: Cutoff")

        # Functional convenience interface
        fn_output = dls(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=2)
        self.assertEqual(fn_output, "Cutoff")

    def test_sample_input_success_with_sufficient_limit(self) -> None:
        """Verify that increasing limit to 3 finds the goal at depth 3."""
        result = depth_limited_search(
            graph=SAMPLE_DLS_GRAPH,
            start="1",
            goal="6",
            limit=3,
        )
        self.assertEqual(result.status, SUCCESS)
        self.assertTrue(result.is_success())
        self.assertEqual(result.path, ["1", "2", "4", "6"])
        self.assertEqual(result.depth, 3)
        self.assertEqual(result.format_result(), "Result: Path: ['1', '2', '4', '6']")
        self.assertEqual(result.format_path(), "1 -> 2 -> 4 -> 6")

        # Functional convenience interface
        fn_output = dls(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=3)
        self.assertEqual(fn_output, ["1", "2", "4", "6"])

    def test_failure_state_when_goal_absent_and_no_cutoffs(self) -> None:
        """Verify Failure state when goal does not exist and no branches are cut off."""
        small_finite_graph = {
            "A": ["B", "C"],
            "B": [],
            "C": [],
        }
        # Tree depth is 1. Limit is 5. Goal 'Z' does not exist anywhere.
        result = depth_limited_search(
            graph=small_finite_graph,
            start="A",
            goal="Z",
            limit=5,
        )
        self.assertEqual(result.status, FAILURE)
        self.assertTrue(result.is_failure())
        self.assertFalse(result.is_cutoff())
        self.assertFalse(result.cutoff_occurred)
        self.assertEqual(result.format_result(), "Result: Failure")

        fn_output = dls(small_finite_graph, start="A", goal="Z", limit=5)
        self.assertEqual(fn_output, "Failure")

    def test_cutoff_vs_failure_distinction(self) -> None:
        """Verify the semantic difference between Cutoff and Failure on the same graph."""
        graph = {
            "R": ["N1"],
            "N1": ["N2"],
            "N2": [],
        }
        # 1. Goal is at depth 2 (N2), but limit is 1 -> CUTOFF
        res_cutoff = depth_limited_search(graph, start="R", goal="N2", limit=1)
        self.assertEqual(res_cutoff.status, CUTOFF)
        self.assertTrue(res_cutoff.cutoff_occurred)

        # 2. Goal is nonexistent (MISSING), but limit is 5 (> max depth 2) -> FAILURE
        res_failure = depth_limited_search(graph, start="R", goal="MISSING", limit=5)
        self.assertEqual(res_failure.status, FAILURE)
        self.assertFalse(res_failure.cutoff_occurred)

    def test_start_is_goal_with_limit_zero(self) -> None:
        """Boundary condition: start node is the goal with limit 0."""
        result = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="1", limit=0)
        self.assertEqual(result.status, SUCCESS)
        self.assertEqual(result.path, ["1"])
        self.assertEqual(result.depth, 0)

    def test_start_not_goal_with_limit_zero(self) -> None:
        """Boundary condition: start node is not the goal with limit 0 -> Cutoff."""
        result = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="2", limit=0)
        self.assertEqual(result.status, CUTOFF)
        self.assertIsNone(result.path)
        self.assertEqual(result.cutoff_nodes, ["1"])

    def test_negative_limit_raises_value_error(self) -> None:
        """Constraint verification: negative depth limit must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=-1)
        self.assertIn("non-negative", str(ctx.exception))

    def test_missing_start_node_raises_value_error(self) -> None:
        """Invalid start node must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            depth_limited_search(SAMPLE_DLS_GRAPH, start="NON_EXISTENT", goal="6", limit=2)
        self.assertIn("does not exist in the graph", str(ctx.exception))

    def test_empty_start_or_goal_raises_value_error(self) -> None:
        """Empty start or goal string must raise ValueError."""
        with self.assertRaises(ValueError):
            depth_limited_search(SAMPLE_DLS_GRAPH, start="", goal="6", limit=2)
        with self.assertRaises(ValueError):
            depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="", limit=2)


class TestOptionalChallengeDeepestNodeTracking(unittest.TestCase):
    """Tests verifying tracking of the deepest node reached before cutoff."""

    def test_linear_chain_deepest_node(self) -> None:
        """Verify deepest node reached in a linear chain graph before cutoff."""
        # Chain: N0 -> N1 -> N2 -> N3 -> N4 -> N5
        # Limit 3: N0 (depth 0), N1 (depth 1), N2 (depth 2), N3 (depth 3, limit 0 -> cutoff)
        result = depth_limited_search(
            graph=LINEAR_CHAIN_DLS_GRAPH,
            start="N0",
            goal="N5",
            limit=3,
        )
        self.assertEqual(result.status, CUTOFF)
        self.assertEqual(result.max_depth_reached, 3)
        self.assertEqual(result.deepest_nodes, ["N3"])
        self.assertEqual(result.deepest_node, "N3")
        self.assertEqual(result.cutoff_nodes, ["N3"])

    def test_binary_tree_multiple_deepest_nodes(self) -> None:
        """Verify multiple nodes at max depth are correctly tracked."""
        # BINARY_TREE_DLS_GRAPH with limit 2:
        # Depth 0: A
        # Depth 1: B, C
        # Depth 2: D, E, F, G
        result = depth_limited_search(
            graph=BINARY_TREE_DLS_GRAPH,
            start="A",
            goal="J",  # J is at depth 3
            limit=2,
        )
        self.assertEqual(result.status, CUTOFF)
        self.assertEqual(result.max_depth_reached, 2)
        self.assertEqual(sorted(result.deepest_nodes), ["D", "E", "F", "G"])
        self.assertEqual(sorted(result.cutoff_nodes), ["D", "E", "F", "G"])

    def test_step_records_include_deepest_and_cutoff_events(self) -> None:
        """Verify detailed step audit logs include actions, depth, and remaining limit."""
        result = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=2)
        self.assertTrue(len(result.steps) > 0)
        cutoff_steps = [s for s in result.steps if s.action == "CUTOFF_HIT"]
        self.assertEqual(len(cutoff_steps), 2)  # nodes 4 and 5
        for s in cutoff_steps:
            self.assertEqual(s.remaining_limit, 0)
            self.assertEqual(s.current_depth, 2)
            self.assertEqual(s.result_returned, CUTOFF)


class TestCyclicAndComplexGraphs(unittest.TestCase):
    """Tests evaluating DLS behavior on graphs with cycles and multiple paths."""

    def test_cyclic_graph_with_cycle_prevention(self) -> None:
        """Verify cyclic graph terminates and finds goal when branch cycle prevention is enabled."""
        result = depth_limited_search(
            graph=CYCLIC_DLS_GRAPH,
            start="1",
            goal="5",
            limit=3,
            cycle_prevention=True,
        )
        self.assertEqual(result.status, SUCCESS)
        self.assertEqual(result.path, ["1", "3", "5"])

    def test_cyclic_graph_without_cycle_prevention_terminates_due_to_limit(self) -> None:
        """Verify that even without cycle prevention, DLS terminates in finite time due to limit bound."""
        result = depth_limited_search(
            graph=CYCLIC_DLS_GRAPH,
            start="1",
            goal="999",  # Non-existent goal
            limit=3,
            cycle_prevention=False,
        )
        # Without cycle check, cycles hit the depth limit 3 and return Cutoff
        self.assertEqual(result.status, CUTOFF)
        self.assertEqual(result.max_depth_reached, 3)

    def test_disconnected_graph_failure(self) -> None:
        """Verify disconnected graph returns Failure when searching for a node in another component."""
        result = depth_limited_search(
            graph=DISCONNECTED_DLS_GRAPH,
            start="A",
            goal="Z",
            limit=5,
        )
        self.assertEqual(result.status, FAILURE)
        self.assertFalse(result.cutoff_occurred)


class TestDLSInterfaceAndUtilities(unittest.TestCase):
    """Tests for class wrappers, string parsing, formatting, and visualization."""

    def test_depth_limited_search_class_interface(self) -> None:
        """Verify object-oriented DepthLimitedSearch class behaves consistently."""
        solver = DepthLimitedSearch(SAMPLE_DLS_GRAPH)
        res_cutoff = solver.search(start="1", goal="6", limit=2)
        self.assertEqual(res_cutoff.status, CUTOFF)

        res_success = solver.search(start="1", goal="6", limit=3)
        self.assertEqual(res_success.status, SUCCESS)

    def test_result_formatting_methods(self) -> None:
        """Verify formatting helpers for summary, step table, and dictionary conversion."""
        result = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=2)
        summary = result.format_summary()
        self.assertIn("DEPTH-LIMITED SEARCH (DLS) EXECUTION SUMMARY", summary)
        self.assertIn("Depth Limit (l)          : 2", summary)
        self.assertIn("Final Outcome            : CUTOFF", summary)

        table = result.format_step_table()
        self.assertIn("CUTOFF_HIT", table)
        self.assertIn("Rem.Limit", table)

        data = result.to_dict()
        self.assertEqual(data["status"], CUTOFF)
        self.assertEqual(data["limit"], 2)
        self.assertEqual(data["max_depth_reached"], 2)

    def test_parse_graph_input_various_syntaxes(self) -> None:
        """Verify parse_graph_input handles multiple representation formats."""
        # 1. Dict direct
        g1 = parse_graph_input({"1": ["2"], "2": []})
        self.assertIn("1", g1)
        self.assertEqual(g1["1"], ["2"])

        # 2. JSON string
        g2 = parse_graph_input('{"1": ["2", "3"], "2": ["4"]}')
        self.assertEqual(g2["1"], ["2", "3"])
        self.assertEqual(g2["2"], ["4"])

        # 3. Compact notation
        g3 = parse_graph_input("1: 2, 3; 2: 4; 3: ; 4: ")
        self.assertEqual(g3["1"], ["2", "3"])
        self.assertEqual(g3["2"], ["4"])
        self.assertEqual(g3["3"], [])

        # 4. Edge list notation
        g4 = parse_graph_input("1->2, 1->3, 2->4")
        self.assertEqual(g4["1"], ["2", "3"])
        self.assertEqual(g4["2"], ["4"])

    def test_parse_graph_input_errors(self) -> None:
        """Verify parse_graph_input raises ValueError on invalid input."""
        with self.assertRaises(ValueError):
            parse_graph_input("")
        with self.assertRaises(ValueError):
            parse_graph_input("invalid non graph string without delimiters")

    def test_viva_reference_guide_structure(self) -> None:
        """Verify Viva Voce reference guide covers required topics."""
        viva = get_dls_viva_reference()
        self.assertIn("infinite_state_spaces", viva["viva_topics"])
        self.assertIn("drawback_of_standard_dfs", viva["viva_topics"])
        self.assertIn("cutoff_vs_failure", viva["viva_topics"])
        self.assertIn("complexity_comparison", viva)
        self.assertIn("DLS", viva["complexity_comparison"])
        self.assertEqual(viva["complexity_comparison"]["DLS"]["time"], "O(b^l)")
        self.assertEqual(viva["complexity_comparison"]["DLS"]["space"], "O(b * l)")

    def test_visualization_generation(self) -> None:
        """Verify visualize_dls generates an image file on disk."""
        result = depth_limited_search(SAMPLE_DLS_GRAPH, start="1", goal="6", limit=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img = Path(tmp_dir) / "test_dls.png"
            saved_path = visualize_dls(result, output_path=out_img)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
