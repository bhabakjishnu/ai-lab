"""Comprehensive unit test suite for Depth-First Search (DFS) for Graph Traversal (Question 12).

Validates:
- Traversal order on sample graph ({'1': ['2', '3'], '2': ['4'], '3': [], '4': []} -> ['1', '3', '2', '4']).
- LIFO stack behavior and order preservation nuances.
- Recursive DFS implementation and step execution.
- Optional Challenge: side-by-side comparison of iterative vs recursive DFS outputs.
- Cycle prevention in directed graphs with simple cycles, self-loops, and cross edges.
- Edge classification (Tree, Back, Forward, Cross) and cycle detection via back edges.
- Path reconstruction from root to targets via DFS spanning tree.
- Disconnected graph handling and unreachable node tracking.
- Single-node, linear chain, and binary tree edge cases.
- Input parsing across Python dicts, JSON, compact adjacency lists, and edge lists.
- Error handling (empty graphs, non-existent start nodes, invalid syntax).
- Object-oriented DepthFirstSearch class interface.
- Headless NetworkX/Matplotlib figure generation.
- Complete Viva Voce reference guide content.
"""

from pathlib import Path
import tempfile
import unittest

from src.depth_first_search import (
    BINARY_TREE_GRAPH,
    CYCLIC_BENCHMARK_GRAPH,
    DAG_GRAPH,
    DEFAULT_SAMPLE_GRAPH,
    DISCONNECTED_GRAPH,
    DFSResult,
    DFSStepRecord,
    DepthFirstSearch,
    compare_dfs_modes,
    depth_first_search,
    dfs_iterative,
    dfs_recursive,
    get_dfs_viva_reference,
    normalize_graph,
    parse_graph_input,
    visualize_dfs,
)


class TestDepthFirstSearchCore(unittest.TestCase):
    """Core algorithmic tests for Depth-First Search traversal and edge cases."""

    def test_sample_input_traversal_order_iterative(self) -> None:
        """Verify exact sample output from Question 12 prompt: 1, 3, 2, 4."""
        graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        result = dfs_iterative(graph, "1", order_preserving=False)
        self.assertEqual(result.traversal_order, ["1", "3", "2", "4"])
        self.assertEqual(result.format_traversal(), "1, 3, 2, 4")
        self.assertFalse(result.has_cycle())

    def test_sample_input_traversal_order_recursive_standard(self) -> None:
        """Verify standard recursive DFS traversal order (left-to-right branch exploration)."""
        graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        result = dfs_recursive(graph, "1", order_preserving=True)
        # Standard recursive visits '2' then '4', backtracks, then visits '3'
        self.assertEqual(result.traversal_order, ["1", "2", "4", "3"])
        self.assertEqual(result.format_traversal(), "1, 2, 4, 3")

    def test_sample_input_traversal_order_recursive_reversed(self) -> None:
        """Verify reversed recursive DFS matches iterative forward-push stack order."""
        graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        result = dfs_recursive(graph, "1", order_preserving=False)
        self.assertEqual(result.traversal_order, ["1", "3", "2", "4"])

    def test_iterative_order_preserving(self) -> None:
        """Verify iterative reverse-push matches standard recursive order."""
        graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        result = dfs_iterative(graph, "1", order_preserving=True)
        self.assertEqual(result.traversal_order, ["1", "2", "4", "3"])

    def test_unified_depth_first_search_entrypoint(self) -> None:
        """Verify unified depth_first_search() function dispatches correctly."""
        graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        iter_res = depth_first_search(graph, "1", mode="iterative")
        rec_res = depth_first_search(graph, "1", mode="recursive", order_preserving=True)

        self.assertEqual(iter_res.traversal_order, ["1", "3", "2", "4"])
        self.assertEqual(rec_res.traversal_order, ["1", "2", "4", "3"])
        self.assertEqual(iter_res.mode, "iterative")
        self.assertEqual(rec_res.mode, "recursive")

    def test_sorted_neighbors_option(self) -> None:
        """Verify that sort_neighbors deterministically sorts successor nodes."""
        graph = {"A": ["Z", "M", "B"], "B": [], "M": [], "Z": []}
        # With sort_neighbors=True and order_preserving=True:
        # sorted neighbors = ['B', 'M', 'Z'] -> visited order ['A', 'B', 'M', 'Z']
        res = dfs_recursive(graph, "A", order_preserving=True, sort_neighbors=True)
        self.assertEqual(res.traversal_order, ["A", "B", "M", "Z"])


class TestCyclesAndEdgeClassification(unittest.TestCase):
    """Test cycle prevention, back-edge detection, and CLRS edge classification."""

    def test_cycle_prevention_directed_cycle(self) -> None:
        """Verify visited set prevents infinite loops in a directed cyclic graph."""
        cyclic_graph = {
            "1": ["2"],
            "2": ["3"],
            "3": ["1"],
        }
        res_iter = dfs_iterative(cyclic_graph, "1")
        res_rec = dfs_recursive(cyclic_graph, "1")

        self.assertEqual(res_iter.traversal_order, ["1", "2", "3"])
        self.assertEqual(res_rec.traversal_order, ["1", "2", "3"])
        self.assertTrue(res_iter.has_cycle())
        self.assertTrue(res_rec.has_cycle())
        self.assertIn(("3", "1"), res_iter.cycles_detected)
        self.assertIn(("3", "1"), res_rec.cycles_detected)

    def test_self_loop_handling(self) -> None:
        """Verify self-loops are safely handled and classified as back-edges."""
        self_loop_graph = {
            "1": ["1", "2"],
            "2": [],
        }
        res = dfs_iterative(self_loop_graph, "1")
        self.assertEqual(set(res.traversal_order), {"1", "2"})
        self.assertTrue(res.has_cycle())
        self.assertIn(("1", "1"), res.cycles_detected)

    def test_complex_cyclic_benchmark(self) -> None:
        """Verify cyclic benchmark graph explores all reachable nodes and finds cycles."""
        res = dfs_iterative(CYCLIC_BENCHMARK_GRAPH, "1")
        # All 5 nodes reachable
        self.assertEqual(set(res.traversal_order), {"1", "2", "3", "4", "5"})
        self.assertTrue(res.has_cycle())
        self.assertEqual(len(res.unreachable_nodes), 0)

    def test_dag_edge_classification(self) -> None:
        """Verify CLRS edge classification distinguishes tree, forward, and cross edges in a DAG."""
        res = dfs_recursive(DAG_GRAPH, "1", order_preserving=True)
        self.assertFalse(res.has_cycle())
        # Tree edges should exist
        tree_edges = [e for e, t in res.edge_classification.items() if t == "TREE"]
        self.assertGreater(len(tree_edges), 0)
        # Check formatted edge classification string
        edge_table = res.format_edge_classification()
        self.assertIn("TREE", edge_table)
        self.assertIn("Significance", edge_table)


class TestOptionalChallengeComparison(unittest.TestCase):
    """Test suite validating the Optional Challenge comparison functionality."""

    def test_compare_dfs_modes_sample_graph(self) -> None:
        """Verify that compare_dfs_modes accurately identifies alignment between modes."""
        comp = compare_dfs_modes(DEFAULT_SAMPLE_GRAPH, "1")

        self.assertEqual(comp["start_node"], "1")
        self.assertEqual(comp["iterative_standard"]["order"], ["1", "3", "2", "4"])
        self.assertEqual(comp["recursive_standard"]["order"], ["1", "2", "4", "3"])
        self.assertEqual(comp["iterative_order_preserving"]["order"], ["1", "2", "4", "3"])
        self.assertEqual(comp["recursive_reversed"]["order"], ["1", "3", "2", "4"])

        self.assertTrue(comp["identical_standard_iter_vs_reversed_rec"])
        self.assertTrue(comp["identical_preserving_iter_vs_standard_rec"])
        self.assertIn("Pedagogical Analysis", comp["explanation"])


class TestTopologiesAndMetrics(unittest.TestCase):
    """Test diverse graph topologies, path reconstruction, and depth metrics."""

    def test_single_node_graph(self) -> None:
        """Verify traversal of a single isolated node."""
        graph = {"ROOT": []}
        res = dfs_iterative(graph, "ROOT")
        self.assertEqual(res.traversal_order, ["ROOT"])
        self.assertEqual(res.max_depth, 0)
        self.assertEqual(res.path_to("ROOT"), ["ROOT"])
        self.assertFalse(res.has_cycle())

    def test_linear_chain_graph(self) -> None:
        """Verify strict linear deep exploration."""
        chain = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
        res = dfs_iterative(chain, "A")
        self.assertEqual(res.traversal_order, ["A", "B", "C", "D"])
        self.assertEqual(res.max_depth, 3)
        self.assertEqual(res.path_to("D"), ["A", "B", "C", "D"])

    def test_binary_tree_graph(self) -> None:
        """Verify complete traversal of binary tree benchmark."""
        res = dfs_iterative(BINARY_TREE_GRAPH, "1", order_preserving=True)
        # Left branch explored completely before right branch
        self.assertEqual(res.traversal_order, ["1", "2", "4", "5", "3", "6", "7"])
        self.assertEqual(res.max_depth, 2)
        self.assertEqual(res.path_to("5"), ["1", "2", "5"])
        self.assertEqual(res.path_to("7"), ["1", "3", "7"])

    def test_disconnected_graph_unreachable_nodes(self) -> None:
        """Verify disconnected components are correctly tagged as unreachable."""
        res = dfs_iterative(DISCONNECTED_GRAPH, "1")
        self.assertEqual(res.traversal_order, ["1", "2", "3"])
        self.assertEqual(res.unreachable_nodes, ["4", "5"])
        self.assertIsNone(res.path_to("4"))
        self.assertIsNone(res.path_to("5"))

    def test_timestamps_discovery_and_finish(self) -> None:
        """Verify d[u] < f[u] timestamp relationship for all explored nodes."""
        res = dfs_recursive(DEFAULT_SAMPLE_GRAPH, "1")
        for node in res.traversal_order:
            d = res.discovery_time[node]
            f = res.finish_time[node]
            self.assertLess(d, f, f"Discovery time {d} must be strictly less than finish time {f} for {node}")


class TestInputParsingAndNormalization(unittest.TestCase):
    """Test flexible parsing of different user input formats."""

    def test_integer_keys_and_values_normalized_to_strings(self) -> None:
        """Verify integer dictionaries are normalized to string keys and values."""
        raw = {1: [2, 3], 2: [4], 3: [], 4: []}  # type: ignore[dict-item]
        norm = normalize_graph(raw)
        self.assertEqual(norm["1"], ["2", "3"])
        self.assertEqual(norm["2"], ["4"])
        self.assertEqual(norm["3"], [])
        self.assertEqual(norm["4"], [])

    def test_missing_leaf_node_keys_added(self) -> None:
        """Verify that neighbors without explicit keys are automatically created."""
        raw = {"1": ["2", "3"]}
        norm = normalize_graph(raw)
        self.assertIn("2", norm)
        self.assertIn("3", norm)
        self.assertEqual(norm["2"], [])
        self.assertEqual(norm["3"], [])

    def test_parse_json_string(self) -> None:
        """Verify JSON formatted string graph parsing."""
        raw_json = '{"1": ["2", "3"], "2": ["4"], "3": [], "4": []}'
        parsed = parse_graph_input(raw_json)
        self.assertEqual(parsed["1"], ["2", "3"])
        self.assertEqual(parsed["2"], ["4"])

    def test_parse_python_dict_literal(self) -> None:
        """Verify Python literal string dict parsing."""
        raw_literal = "{'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}"
        parsed = parse_graph_input(raw_literal)
        self.assertEqual(parsed["A"], ["B", "C"])
        self.assertEqual(parsed["B"], ["D"])

    def test_parse_compact_syntax(self) -> None:
        """Verify compact '1: 2, 3; 2: 4; 3: ; 4: ' syntax parsing."""
        compact = "1: 2, 3; 2: 4; 3:; 4:"
        parsed = parse_graph_input(compact)
        self.assertEqual(parsed["1"], ["2", "3"])
        self.assertEqual(parsed["2"], ["4"])
        self.assertEqual(parsed["3"], [])

    def test_parse_edge_list_syntax(self) -> None:
        """Verify edge list '1->2, 1->3, 2->4' syntax parsing."""
        edge_list = "1->2, 1->3, 2->4"
        parsed = parse_graph_input(edge_list)
        self.assertIn("2", parsed["1"])
        self.assertIn("3", parsed["1"])
        self.assertIn("4", parsed["2"])


class TestErrorHandling(unittest.TestCase):
    """Test validation and error handling for invalid graph structures."""

    def test_empty_graph_raises_value_error(self) -> None:
        """Verify empty graph dictionary raises ValueError."""
        with self.assertRaises(ValueError):
            normalize_graph({})

    def test_non_dict_graph_raises_value_error(self) -> None:
        """Verify non-dictionary input raises ValueError."""
        with self.assertRaises(ValueError):
            normalize_graph(["A", "B"])  # type: ignore[arg-type]

    def test_missing_start_node_raises_key_error(self) -> None:
        """Verify starting from an absent node raises KeyError."""
        with self.assertRaises(KeyError):
            dfs_iterative(DEFAULT_SAMPLE_GRAPH, "NON_EXISTENT")

    def test_invalid_string_syntax_raises_value_error(self) -> None:
        """Verify unparseable string raises ValueError."""
        with self.assertRaises(ValueError):
            parse_graph_input("%%%invalid syntax%%%")


class TestObjectOrientedEngine(unittest.TestCase):
    """Test the DepthFirstSearch class interface."""

    def test_engine_default_initialization(self) -> None:
        """Verify engine initializes with DEFAULT_SAMPLE_GRAPH."""
        engine = DepthFirstSearch()
        res = engine.search("1")
        self.assertEqual(res.traversal_order, ["1", "3", "2", "4"])

    def test_engine_compare_and_find_path(self) -> None:
        """Verify comparison and path finding methods on engine."""
        engine = DepthFirstSearch(DEFAULT_SAMPLE_GRAPH)
        comp = engine.compare("1")
        self.assertTrue(comp["identical_standard_iter_vs_reversed_rec"])

        path_to_4 = engine.find_path("1", "4")
        self.assertEqual(path_to_4, ["1", "2", "4"])

    def test_engine_detect_cycles_global(self) -> None:
        """Verify global cycle detection across entire graph."""
        engine = DepthFirstSearch(CYCLIC_BENCHMARK_GRAPH)
        cycles = engine.detect_cycles()
        self.assertGreater(len(cycles), 0)

    def test_engine_set_graph(self) -> None:
        """Verify updating graph on existing engine."""
        engine = DepthFirstSearch()
        engine.set_graph("A: B; B: C; C:")
        res = engine.search("A")
        self.assertEqual(res.traversal_order, ["A", "B", "C"])


class TestVisualizationAndFormatting(unittest.TestCase):
    """Test formatting tables, summaries, and Matplotlib generation."""

    def test_summary_and_step_table_output(self) -> None:
        """Verify formatting functions generate non-empty strings with key sections."""
        res = dfs_iterative(DEFAULT_SAMPLE_GRAPH, "1")
        summary = res.format_summary()
        step_table = res.format_step_table()

        self.assertIn("DEPTH-FIRST SEARCH (DFS) TRAVERSAL SUMMARY", summary)
        self.assertIn("DFS Spanning Tree Parent Connections", summary)
        self.assertIn("Step", step_table)
        self.assertIn("Stack (LIFO)", step_table)

    def test_visualize_dfs_file_output(self) -> None:
        """Verify visualize_dfs exports a PNG without errors."""
        res = dfs_iterative(DEFAULT_SAMPLE_GRAPH, "1")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_dfs.png"
            fig = visualize_dfs(res, save_path=out_file, show=False)
            self.assertIsNotNone(fig)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 0)


class TestVivaReferenceGuide(unittest.TestCase):
    """Test completeness of Viva Voce reference guide."""

    def test_viva_reference_structure_and_topics(self) -> None:
        """Verify all mandatory viva topics from prompt are covered."""
        viva = get_dfs_viva_reference()
        self.assertIn("title", viva)
        self.assertIn("core_topics", viva)
        self.assertIn("comparison_matrix", viva)

        topics_text = " ".join([t["topic"] for t in viva["core_topics"]]).lower()
        self.assertIn("stack vs queue", topics_text)
        self.assertIn("optimal", topics_text)
        self.assertIn("space complexity", topics_text)
        self.assertIn("cycle", topics_text)


if __name__ == "__main__":
    unittest.main()
