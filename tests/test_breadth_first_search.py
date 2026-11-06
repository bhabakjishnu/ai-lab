"""Comprehensive unit test suite for Breadth-First Search (BFS) on User-Defined Graph (Question 11).

Validates:
- Traversal order on sample graph ({'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []} -> ['A', 'B', 'C', 'D']).
- Deterministic alphabetical tie-breaking constraint.
- Cycle prevention in directed graphs with simple and complex cycles.
- Handling of self-loops and multi-parent convergent DAG paths.
- Level and depth tracking (Optional Challenge): total_levels, max_depth, and nodes_by_level.
- Single-node and linear chain edge cases.
- Disconnected graphs and reachability analysis.
- Minimum-hop shortest path reconstruction via parents mapping.
- Error handling (missing start node, empty graph, malformed types).
- Parsing user input from JSON, Python literal, compact adjacency, and edge-list formats.
- Object-oriented BreadthFirstSearch class API.
- Matplotlib/NetworkX visual generation in headless mode.
- Viva Voce reference guide completeness.
"""

from pathlib import Path
import tempfile
import unittest

from src.breadth_first_search import (
    BINARY_TREE_GRAPH,
    CYCLIC_BENCHMARK_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DISCONNECTED_GRAPH,
    BFSResult,
    BFSStepRecord,
    BreadthFirstSearch,
    breadth_first_search,
    get_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_bfs,
)


class TestBreadthFirstSearchCore(unittest.TestCase):
    """Core algorithmic tests for Breadth-First Search traversal."""

    def test_sample_input_traversal_order(self) -> None:
        """Verify exact sample output from Question 11 prompt."""
        graph = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
        result = breadth_first_search(graph, "A")
        self.assertEqual(result.traversal_order, ["A", "B", "C", "D"])
        self.assertEqual(result.format_traversal(), "A, B, C, D")

    def test_alphabetical_tie_breaking(self) -> None:
        """Verify that tied successor nodes are expanded in alphabetical order."""
        # Unsorted neighbor list: Z, M, B, K
        graph = {"A": ["Z", "M", "B", "K"], "B": [], "K": [], "M": [], "Z": []}
        result = breadth_first_search(graph, "A", sort_neighbors=True)
        self.assertEqual(result.traversal_order, ["A", "B", "K", "M", "Z"])

    def test_cycle_prevention(self) -> None:
        """Verify that visited set prevents infinite loops in cyclic directed graphs."""
        cyclic_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A", "D"],  # Back edge to A
            "D": ["B"],       # Back edge to B
        }
        result = breadth_first_search(cyclic_graph, "A")
        self.assertEqual(result.traversal_order, ["A", "B", "C", "D"])
        self.assertEqual(len(result.steps), 4)

    def test_self_loops(self) -> None:
        """Verify that self-loops do not cause duplicate processing or hangs."""
        graph_with_self_loop = {
            "A": ["A", "B"],
            "B": ["B", "C"],
            "C": ["C"],
        }
        result = breadth_first_search(graph_with_self_loop, "A")
        self.assertEqual(result.traversal_order, ["A", "B", "C"])

    def test_multiple_paths_convergence(self) -> None:
        """Verify DAG where multiple paths converge on the same node."""
        dag = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        result = breadth_first_search(dag, "A")
        # D should only be expanded once
        self.assertEqual(result.traversal_order, ["A", "B", "C", "D"])
        self.assertEqual(len([n for n in result.traversal_order if n == "D"]), 1)


class TestOptionalChallengeLevelsAndDepth(unittest.TestCase):
    """Tests for graph depth levels calculation (Optional Challenge)."""

    def test_sample_graph_levels(self) -> None:
        """Verify level partitioning on standard sample graph."""
        result = breadth_first_search(DEFAULT_SAMPLE_GRAPH, "A")
        # Level 0: A; Level 1: B, C; Level 2: D
        self.assertEqual(result.total_levels, 3)
        self.assertEqual(result.max_depth, 2)
        self.assertEqual(result.levels, {"A": 0, "B": 1, "C": 1, "D": 2})
        self.assertEqual(result.nodes_by_level[0], ["A"])
        self.assertEqual(result.nodes_by_level[1], ["B", "C"])
        self.assertEqual(result.nodes_by_level[2], ["D"])

    def test_single_node_graph(self) -> None:
        """Verify boundary condition: graph with a single node."""
        single = {"A": []}
        result = breadth_first_search(single, "A")
        self.assertEqual(result.traversal_order, ["A"])
        self.assertEqual(result.total_levels, 1)
        self.assertEqual(result.max_depth, 0)
        self.assertEqual(result.levels, {"A": 0})

    def test_linear_chain_levels(self) -> None:
        """Verify depth on a linear chain: A -> B -> C -> D."""
        chain = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
        result = breadth_first_search(chain, "A")
        self.assertEqual(result.traversal_order, ["A", "B", "C", "D"])
        self.assertEqual(result.total_levels, 4)
        self.assertEqual(result.max_depth, 3)
        self.assertEqual(result.levels, {"A": 0, "B": 1, "C": 2, "D": 3})

    def test_binary_tree_levels(self) -> None:
        """Verify level grouping on balanced binary tree."""
        result = breadth_first_search(BINARY_TREE_GRAPH, "A")
        self.assertEqual(result.total_levels, 3)
        self.assertEqual(result.max_depth, 2)
        self.assertEqual(result.nodes_by_level[0], ["A"])
        self.assertEqual(result.nodes_by_level[1], ["B", "C"])
        self.assertEqual(result.nodes_by_level[2], ["D", "E", "F", "G"])


class TestReachabilityAndShortestPaths(unittest.TestCase):
    """Tests for disconnected graphs and shortest path reconstruction."""

    def test_disconnected_components(self) -> None:
        """Verify reachable exploration and identification of unreachable components."""
        result = breadth_first_search(DISCONNECTED_GRAPH, "A")
        self.assertEqual(result.traversal_order, ["A", "B", "C"])
        self.assertIn("D", result.unreachable_nodes)
        self.assertIn("E", result.unreachable_nodes)
        self.assertEqual(sorted(result.unreachable_nodes), ["D", "E"])

    def test_shortest_path_reconstruction(self) -> None:
        """Verify BFS path reconstruction produces the minimum hop path."""
        # Diamond graph with shortcut
        graph = {
            "S": ["A", "B"],
            "A": ["C"],
            "B": ["D"],
            "C": ["G"],
            "D": ["G"],
            "G": [],
        }
        result = breadth_first_search(graph, "S")
        path_to_g = result.shortest_path_to("G")
        # In alphabetical tie-break: S -> A -> C -> G
        self.assertEqual(path_to_g, ["S", "A", "C", "G"])

        # Unreachable target returns None
        self.assertIsNone(result.shortest_path_to("NONEXISTENT"))

    def test_step_records_and_ascii_tree(self) -> None:
        """Verify step trace records and ASCII tree rendering."""
        result = breadth_first_search(DEFAULT_SAMPLE_GRAPH, "A")
        self.assertEqual(len(result.steps), 4)

        step1 = result.steps[0]
        self.assertEqual(step1.step_number, 1)
        self.assertEqual(step1.current_node, "A")
        self.assertEqual(step1.depth_level, 0)
        self.assertIsNone(step1.parent)
        self.assertEqual(step1.neighbors_enqueued, ["B", "C"])

        # ASCII tree output contains Root and hierarchy
        tree_str = result.format_ascii_tree()
        self.assertIn("A (Root / Level 0)", tree_str)
        self.assertIn("B (Level 1)", tree_str)
        self.assertIn("D (Level 2)", tree_str)

        # Step table formatted row
        table_str = result.format_step_table()
        self.assertIn("Step", table_str)
        self.assertIn("Frontier Queue", table_str)


class TestGraphInputParser(unittest.TestCase):
    """Tests for parsing various user input graph formats."""

    def test_parse_dict_literal(self) -> None:
        """Verify parsing Python dictionary literal string."""
        raw = "{'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}"
        parsed = parse_graph_input(raw)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])
        self.assertEqual(parsed["C"], [])
        self.assertEqual(parsed["D"], [])

    def test_parse_json_string(self) -> None:
        """Verify parsing standard JSON string."""
        raw = '{"A": ["B", "C"], "B": ["D"]}'
        parsed = parse_graph_input(raw)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])
        # Referenced C and D should be added automatically
        self.assertIn("C", parsed)
        self.assertIn("D", parsed)

    def test_parse_compact_adjacency(self) -> None:
        """Verify parsing compact adjacency format (e.g. 'A: B, C; B: D')."""
        raw = "A: B, C; B: D; C:; D:"
        parsed = parse_graph_input(raw)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])
        self.assertEqual(parsed["C"], [])
        self.assertEqual(parsed["D"], [])

    def test_parse_edge_list(self) -> None:
        """Verify parsing edge-list format (e.g. 'A->B, A->C, B->D')."""
        raw = "A->B, A->C, B->D"
        parsed = parse_graph_input(raw)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])
        self.assertIn("C", parsed)

    def test_parse_invalid_raises_error(self) -> None:
        """Verify invalid strings raise ValueError with helpful advice."""
        with self.assertRaises(ValueError):
            parse_graph_input("")
        with self.assertRaises(ValueError):
            parse_graph_input("this is not a valid graph representation")
        with self.assertRaises(TypeError):
            parse_graph_input(12345)  # type: ignore


class TestValidationAndErrorHandling(unittest.TestCase):
    """Tests for validation and boundary exceptions."""

    def test_missing_start_node_raises_key_error(self) -> None:
        """Verify missing start node raises KeyError."""
        with self.assertRaises(KeyError):
            breadth_first_search(DEFAULT_SAMPLE_GRAPH, "Z")

    def test_empty_graph_raises_value_error(self) -> None:
        """Verify empty graph raises ValueError."""
        with self.assertRaises(ValueError):
            normalize_graph({})

    def test_invalid_graph_type_raises_type_error(self) -> None:
        """Verify non-dict raises TypeError."""
        with self.assertRaises(TypeError):
            normalize_graph(["A", "B"])  # type: ignore


class TestOOPInterfaceAndVivaReference(unittest.TestCase):
    """Tests for BreadthFirstSearch class and viva voce documentation."""

    def test_oop_runner_interface(self) -> None:
        """Verify BreadthFirstSearch wrapper class methods."""
        runner = BreadthFirstSearch(DEFAULT_SAMPLE_GRAPH)
        self.assertEqual(runner.get_traversal_string("A"), "A, B, C, D")
        self.assertEqual(runner.get_level_count("A"), 3)
        res = runner.run("A")
        self.assertIsInstance(res, BFSResult)
        self.assertIs(runner.last_result, res)

    def test_viva_reference_structure(self) -> None:
        """Verify viva reference contents and answers."""
        viva = get_viva_reference()
        self.assertIn("title", viva)
        self.assertIn("qna", viva)
        self.assertGreaterEqual(len(viva["qna"]), 5)
        # Verify specific topics exist
        questions_lower = [item["question"].lower() for item in viva["qna"]]
        self.assertTrue(any("fifo" in q for q in questions_lower))
        self.assertTrue(any("complete" in q for q in questions_lower))
        self.assertTrue(any("time complexity" in q for q in questions_lower))
        self.assertTrue(any("space complexity" in q for q in questions_lower))

    def test_headless_visualization(self) -> None:
        """Verify visualize_bfs generates and saves figure without error."""
        result = breadth_first_search(DEFAULT_SAMPLE_GRAPH, "A")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "test_bfs.png"
            fig = visualize_bfs(result, save_path=out_img, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(out_img.exists())
            self.assertGreater(out_img.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
