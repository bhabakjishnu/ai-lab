"""Comprehensive unit test suite for Uniform Cost Search (UCS) Implementation (Question 14).

Validates:
- Exact sample input and output from Question 14 prompt (A->B:1, B->C:2, A->C:4 -> Path: A->B->C, Cost: 3).
- Direct expensive path vs cheaper multi-hop path (Diamond benchmark).
- Classical AIMA Romania Map benchmark (Arad to Bucharest -> cost 418).
- Positive cycles and self-loops handling without infinite loops.
- Goal test timing: ensures goal test is evaluated on expansion, not generation.
- Tie-breaking strategies: lexicographical (alphabetical), FIFO insertion order, and hop count.
- Edge cases: start == goal, unreachable target node, single node graph, linear chain.
- Input parsing robustness: arrow strings, JSON strings, adjacency syntax, Python dicts, edge tuples.
- Error handling: non-existent start node, non-numeric weights, empty graph input.
- Optional Challenge: detection of negative edge weights and negative cycle identification with Bellman-Ford.
- OOP UniformCostSearch class interface and all-pairs shortest paths query.
- Headless visual diagram generation using NetworkX and Matplotlib.
- Viva Voce study guide and comparison matrix data completeness.
"""

from pathlib import Path
import tempfile
import unittest

from src.uniform_cost_search import (
    CYCLIC_WEIGHTED_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DIAMOND_BENCHMARK_GRAPH,
    DISCONNECTED_WEIGHTED_GRAPH,
    NEGATIVE_CYCLE_GRAPH,
    NEGATIVE_EDGE_DAG,
    ROMANIA_MAP_SUBGRAPH,
    UCSStepRecord,
    UCSResult,
    UniformCostSearch,
    detect_negative_cycle,
    get_ucs_viva_reference,
    normalize_weighted_graph,
    parse_weighted_graph,
    uniform_cost_search,
    validate_strictly_positive_weights,
    visualize_ucs,
)


class TestUniformCostSearchCore(unittest.TestCase):
    """Core algorithmic tests for Uniform Cost Search."""

    def test_sample_input_and_expected_output(self) -> None:
        """Verify exact sample from Question 14 prompt: A->B:1, B->C:2, A->C:4. Start:A, Goal:C."""
        raw_input = "A->B:1, B->C:2, A->C:4"
        result = uniform_cost_search(raw_input, start="A", goal="C")

        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertEqual(result.total_cost, 3)
        self.assertEqual(result.format_path(), "Path: A->B->C, Total Cost: 3")
        self.assertEqual(result.start_node, "A")
        self.assertEqual(result.goal_node, "C")

    def test_direct_expensive_vs_cheap_multihop(self) -> None:
        """Verify UCS chooses the cheaper multi-hop path over the direct expensive path."""
        # Diamond graph:
        # S->A:1, A->C:2, C->G:1 (total cost 4)
        # S->B:5, B->G:2 (total cost 7)
        result = uniform_cost_search(DIAMOND_BENCHMARK_GRAPH, start="S", goal="G")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "A", "C", "G"])
        self.assertEqual(result.total_cost, 4)

    def test_romania_map_arad_to_bucharest(self) -> None:
        """Verify classic AIMA Romania benchmark from Arad to Bucharest."""
        # Arad -> Sibiu (140) -> Rimnicu Vilcea (80) -> Pitesti (97) -> Bucharest (101) = 418
        result = uniform_cost_search(ROMANIA_MAP_SUBGRAPH, start="Arad", goal="Bucharest")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"])
        self.assertEqual(result.total_cost, 418)

    def test_goal_test_on_expansion_not_generation(self) -> None:
        """Verify goal test executes upon expansion, preventing premature non-optimal termination.

        Scenario:
        Start -> Goal with cost 10 (generated first!)
        Start -> Mid with cost 1
        Mid -> Goal with cost 2 (total 3)
        If tested upon generation, Goal would terminate at cost 10.
        Tested upon expansion, Goal must terminate at optimal cost 3!
        """
        graph = {
            "Start": [("Goal", 10), ("Mid", 1)],
            "Mid": [("Goal", 2)],
            "Goal": [],
        }
        result = uniform_cost_search(graph, start="Start", goal="Goal")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["Start", "Mid", "Goal"])
        self.assertEqual(result.total_cost, 3)

        # Check that Mid was expanded before Goal was expanded
        self.assertIn("Mid", result.nodes_expanded)
        self.assertEqual(result.nodes_expanded[-1], "Goal")

    def test_positive_cycles_handling(self) -> None:
        """Verify directed graphs with positive cycles do not cause infinite loops."""
        result = uniform_cost_search(CYCLIC_WEIGHTED_GRAPH, start="A", goal="D")
        self.assertTrue(result.goal_found)
        # A->B:2, B->C:1, C->D:1 = 4 vs A->C:5, C->D:1 = 6
        self.assertEqual(result.path, ["A", "B", "C", "D"])
        self.assertEqual(result.total_cost, 4)

    def test_self_loops_handling(self) -> None:
        """Verify graph with self-loops does not enter infinite recursion."""
        graph = {
            "A": [("A", 1), ("B", 3)],
            "B": [("B", 2), ("C", 2)],
            "C": [("C", 5)],
        }
        result = uniform_cost_search(graph, start="A", goal="C")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertEqual(result.total_cost, 5)

    def test_start_equals_goal(self) -> None:
        """Verify edge case where start node is the destination goal."""
        result = uniform_cost_search(DEFAULT_SAMPLE_GRAPH, start="A", goal="A")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.format_path(), "Path: A, Total Cost: 0")

    def test_unreachable_goal_disconnected_graph(self) -> None:
        """Verify search gracefully reports failure when target is in a disconnected component."""
        result = uniform_cost_search(DISCONNECTED_WEIGHTED_GRAPH, start="A", goal="E")
        self.assertFalse(result.goal_found)
        self.assertIsNone(result.path)
        self.assertEqual(result.total_cost, float("inf"))
        self.assertIn("Unreachable", result.format_path())

    def test_single_node_graph(self) -> None:
        """Verify behavior on a single node graph with no outgoing edges."""
        graph = {"Isolated": []}
        res_self = uniform_cost_search(graph, start="Isolated", goal="Isolated")
        self.assertTrue(res_self.goal_found)
        self.assertEqual(res_self.path, ["Isolated"])
        self.assertEqual(res_self.total_cost, 0)

        res_missing = uniform_cost_search(graph, start="Isolated", goal="Other")
        self.assertFalse(res_missing.goal_found)
        self.assertIsNone(res_missing.path)

    def test_linear_chain(self) -> None:
        """Verify UCS on a simple unbranching linear chain."""
        graph = "N1->N2:3, N2->N3:4, N3->N4:5"
        result = uniform_cost_search(graph, start="N1", goal="N4")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["N1", "N2", "N3", "N4"])
        self.assertEqual(result.total_cost, 12)


class TestTieBreakingAndPriorityQueue(unittest.TestCase):
    """Test priority queue mechanisms, tuple comparison, and tie-breaking modes."""

    def test_lexicographical_tie_breaking(self) -> None:
        """Verify alphabetical tie-breaking when two successor nodes yield identical path costs."""
        # S->Z: 5, S->B: 5. Both have cost 5.
        # Alphabetically, B should be expanded before Z.
        graph = {
            "S": [("Z", 5), ("B", 5)],
            "B": [("G", 1)],
            "Z": [("G", 1)],
            "G": [],
        }
        result = uniform_cost_search(graph, start="S", goal="G", tie_breaker="lexicographical")
        self.assertTrue(result.goal_found)
        # B expands before Z, reaching G via B
        self.assertEqual(result.path, ["S", "B", "G"])
        self.assertEqual(result.total_cost, 6)

    def test_fifo_tie_breaking(self) -> None:
        """Verify FIFO sequence counter tie-breaking preserves discovery order."""
        graph = {
            "S": [("Z", 5), ("B", 5)],
            "B": [("G", 1)],
            "Z": [("G", 1)],
            "G": [],
        }
        # In FIFO mode, Z was pushed first, so Z is popped first
        result = uniform_cost_search(graph, start="S", goal="G", tie_breaker="fifo", sort_neighbors=False)
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "Z", "G"])
        self.assertEqual(result.total_cost, 6)

    def test_hops_tie_breaking(self) -> None:
        """Verify fewest hops tie-breaker when path costs are identical."""
        # S->G: 10 (1 hop, cost 10)
        # S->A: 5, A->G: 5 (2 hops, cost 10)
        graph = {
            "S": [("A", 5), ("G", 10)],
            "A": [("G", 5)],
            "G": [],
        }
        result = uniform_cost_search(graph, start="S", goal="G", tie_breaker="hops")
        self.assertTrue(result.goal_found)
        self.assertEqual(result.path, ["S", "G"])
        self.assertEqual(result.total_cost, 10)

    def test_step_records_and_table_formatting(self) -> None:
        """Verify step trace generation, field preservation, and table formatting."""
        result = uniform_cost_search(DEFAULT_SAMPLE_GRAPH, start="A", goal="C")
        self.assertGreater(len(result.steps), 0)
        first_step: UCSStepRecord = result.steps[0]
        self.assertEqual(first_step.current_node, "A")
        self.assertEqual(first_step.cumulative_cost, 0)
        self.assertEqual(first_step.action, "EXPAND")

        table_str = result.format_step_table()
        self.assertIn("Step", table_str)
        self.assertIn("g(n)", table_str)
        self.assertIn("Frontier Priority Queue", table_str)
        self.assertIn("A", table_str)
        self.assertIn("C", table_str)

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict returns a valid JSON-serializable dictionary."""
        result = uniform_cost_search(DEFAULT_SAMPLE_GRAPH, start="A", goal="C")
        data = result.to_dict()
        self.assertEqual(data["path"], ["A", "B", "C"])
        self.assertEqual(data["total_cost"], 3)
        self.assertTrue(data["goal_found"])
        self.assertIsInstance(data["nodes_expanded"], list)
        self.assertIsInstance(data["cost_to_nodes"], dict)


class TestGraphParsingAndValidation(unittest.TestCase):
    """Test flexible input syntax parsing and error handling."""

    def test_parse_arrow_string(self) -> None:
        """Verify arrow syntax parser on question prompt string."""
        raw = "A->B:1, B->C:2, A->C:4"
        parsed = parse_weighted_graph(raw)
        self.assertIn("A", parsed)
        self.assertIn("B", parsed)
        self.assertIn("C", parsed)
        self.assertEqual(parsed["A"], [("B", 1), ("C", 4)])
        self.assertEqual(parsed["B"], [("C", 2)])
        self.assertEqual(parsed["C"], [])

    def test_parse_arrow_with_whitespace_and_floats(self) -> None:
        """Verify parsing with spaced arrows and floating point weights."""
        raw = "Node1 -> Node2 : 2.5 , Node2 -> Node3 : 4.0"
        parsed = parse_weighted_graph(raw)
        self.assertEqual(parsed["Node1"], [("Node2", 2.5)])
        self.assertEqual(parsed["Node2"], [("Node3", 4)])

    def test_parse_semicolon_adjacency_format(self) -> None:
        """Verify parsing semicolon-separated adjacency list format."""
        raw = "A: B:1, C:4; B: C:2; C:"
        parsed = parse_weighted_graph(raw)
        self.assertEqual(parsed["A"], [("B", 1), ("C", 4)])
        self.assertEqual(parsed["B"], [("C", 2)])
        self.assertEqual(parsed["C"], [])

    def test_parse_json_string(self) -> None:
        """Verify JSON string parsing."""
        raw = '{"A": [["B", 1], ["C", 4]], "B": [["C", 2]], "C": []}'
        parsed = parse_weighted_graph(raw)
        self.assertEqual(parsed["A"], [("B", 1), ("C", 4)])

    def test_parse_edge_3tuples_list(self) -> None:
        """Verify list of 3-tuples (u, v, weight)."""
        raw = [("A", "B", 1), ("B", "C", 2), ("A", "C", 4)]
        parsed = parse_weighted_graph(raw)
        self.assertEqual(parsed["A"], [("B", 1), ("C", 4)])
        self.assertEqual(parsed["B"], [("C", 2)])
        self.assertEqual(parsed["C"], [])

    def test_normalize_dict_of_dicts(self) -> None:
        """Verify dictionary of nested dictionaries: {'A': {'B': 1, 'C': 4}}."""
        raw = {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}
        parsed = normalize_weighted_graph(raw)
        self.assertEqual(parsed["A"], [("B", 1), ("C", 4)])
        self.assertEqual(parsed["B"], [("C", 2)])
        self.assertEqual(parsed["C"], [])

    def test_error_on_missing_start_node(self) -> None:
        """Verify ValueError when specified start node is missing from graph."""
        with self.assertRaises(ValueError) as ctx:
            uniform_cost_search(DEFAULT_SAMPLE_GRAPH, start="NonExistent", goal="C")
        self.assertIn("does not exist in the graph", str(ctx.exception))

    def test_error_on_empty_string(self) -> None:
        """Verify ValueError on empty input string."""
        with self.assertRaises(ValueError):
            parse_weighted_graph("")

    def test_error_on_non_numeric_weight(self) -> None:
        """Verify ValueError when edge weight is non-numeric."""
        with self.assertRaises(ValueError):
            parse_weighted_graph("A->B:expensive")


class TestOptionalChallengeNegativeCycles(unittest.TestCase):
    """Test Optional Challenge: detection and handling of negative weights and negative cycles."""

    def test_validate_strictly_positive_weights_finds_violations(self) -> None:
        """Verify helper identifies negative or zero weights."""
        violations = validate_strictly_positive_weights(NEGATIVE_EDGE_DAG)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0], ("A", "C", -1))

    def test_detect_negative_cycle_on_clean_graph(self) -> None:
        """Verify Bellman-Ford cycle detector returns False on standard positive graphs."""
        has_cycle, cycle, _ = detect_negative_cycle(DEFAULT_SAMPLE_GRAPH, start_node="A")
        self.assertFalse(has_cycle)
        self.assertIsNone(cycle)

    def test_detect_negative_cycle_finds_and_traces_cycle(self) -> None:
        """Verify Bellman-Ford cycle detector finds and traces the negative cycle."""
        # Graph: A->B:1, B->C:2, C->D:-5, D->B:1 (Cycle: B->C->D->B with weight 2 - 5 + 1 = -2)
        has_cycle, cycle, _ = detect_negative_cycle(NEGATIVE_CYCLE_GRAPH, start_node="A")
        self.assertTrue(has_cycle)
        self.assertIsNotNone(cycle)
        # Cycle must start and end at the same node
        self.assertEqual(cycle[0], cycle[-1])
        # Nodes B, C, D must be in cycle
        for node in ["B", "C", "D"]:
            self.assertIn(node, cycle)

    def test_ucs_handles_negative_cycle_gracefully(self) -> None:
        """Verify UCS execution stops gracefully when a reachable negative cycle is detected."""
        result = uniform_cost_search(NEGATIVE_CYCLE_GRAPH, start="A", goal="Goal", detect_cycles=True)
        self.assertTrue(result.negative_cycle_detected)
        self.assertFalse(result.goal_found)
        self.assertEqual(result.total_cost, float("-inf"))
        self.assertIsNotNone(result.negative_cycle_path)
        self.assertIn("negative cycle detected", result.warning_message.lower())

    def test_negative_edges_without_cycles_warns(self) -> None:
        """Verify warning message when negative edges exist without negative cycles."""
        result = uniform_cost_search(NEGATIVE_EDGE_DAG, start="S", goal="G", detect_cycles=True)
        self.assertFalse(result.negative_cycle_detected)
        self.assertTrue(result.goal_found)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("non-positive edge", result.warning_message.lower())


class TestUniformCostSearchClassAPI(unittest.TestCase):
    """Test object-oriented UniformCostSearch wrapper and helper methods."""

    def test_class_search_method(self) -> None:
        """Verify instance search method executes properly."""
        ucs = UniformCostSearch(DEFAULT_SAMPLE_GRAPH)
        result = ucs.search(start="A", goal="C")
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertEqual(result.total_cost, 3)

    def test_find_all_shortest_paths(self) -> None:
        """Verify find_all_shortest_paths computes paths to all reachable destinations."""
        ucs = UniformCostSearch(DEFAULT_SAMPLE_GRAPH)
        all_paths = ucs.find_all_shortest_paths(start="A")
        self.assertEqual(all_paths["A"], (["A"], 0))
        self.assertEqual(all_paths["B"], (["A", "B"], 1))
        self.assertEqual(all_paths["C"], (["A", "B", "C"], 3))

    def test_check_negative_cycle_method(self) -> None:
        """Verify instance method for checking negative cycle."""
        ucs_clean = UniformCostSearch(DEFAULT_SAMPLE_GRAPH)
        has_cycle, _ = ucs_clean.check_negative_cycle("A")
        self.assertFalse(has_cycle)

        ucs_neg = UniformCostSearch(NEGATIVE_CYCLE_GRAPH)
        has_cycle_neg, cycle = ucs_neg.check_negative_cycle("A")
        self.assertTrue(has_cycle_neg)
        self.assertIsNotNone(cycle)


class TestVisualizationAndVivaVoce(unittest.TestCase):
    """Test NetworkX visualization generator and Viva Voce study guide."""

    def test_visualize_ucs_creates_png(self) -> None:
        """Verify visualize_ucs renders and saves a clean PNG image in headless mode."""
        result = uniform_cost_search(DEFAULT_SAMPLE_GRAPH, start="A", goal="C")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_png = Path(temp_dir) / "test_ucs.png"
            output_file = visualize_ucs(result, output_path=temp_png, show_plot=False)
            self.assertTrue(output_file.exists())
            self.assertGreater(output_file.stat().st_size, 1000)

    def test_viva_reference_completeness(self) -> None:
        """Verify viva voce reference contains all required questions and comparison matrix."""
        viva = get_ucs_viva_reference()
        self.assertEqual(viva["syllabus_area"], "Uninformed Search Techniques")
        self.assertGreaterEqual(len(viva["core_topics"]), 4)

        topics = [t["topic"] for t in viva["core_topics"]]
        self.assertTrue(any("UCS vs BFS" in t for t in topics))
        self.assertTrue(any("Priority Queue" in t for t in topics))
        self.assertTrue(any("negative edge weights" in t for t in topics))

        matrix = viva["comparison_matrix"]
        self.assertIn("headers", matrix)
        self.assertIn("rows", matrix)
        algos = [row[0] for row in matrix["rows"]]
        self.assertIn("UCS", algos)
        self.assertIn("BFS", algos)
        self.assertIn("DFS", algos)


if __name__ == "__main__":
    unittest.main()
