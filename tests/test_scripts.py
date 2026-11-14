"""Unit tests for script entry points and top-level CLI dispatcher.

Ensures that all organized scripts in scripts/ and module entry points
execute cleanly with expected return codes and outputs.
"""

import io
import sys
import unittest
from unittest.mock import patch

from scripts.smoke_test import main as smoke_main
from scripts.run_vacuum_agent import main as vacuum_main
from scripts.run_randomized_agent import main as randomized_main
from scripts.run_model_based_agent import main as model_main
from scripts.run_goal_based_agent import main as goal_main
from scripts.run_utility_based_agent import main as utility_main
from scripts.run_multi_agent_simulation import main as multi_agent_main
from scripts.run_stochastic_vacuum_agent import main as stochastic_main
from scripts.run_environment_classifier import main as env_classifier_main
from scripts.run_agent_performance_evaluator import main as agent_eval_main
from scripts.run_state_space_visualizer import main as state_space_main
from scripts.run_breadth_first_search import main as bfs_main
from scripts.run_depth_first_search import main as dfs_main
from scripts.run_uniform_cost_search import main as ucs_main
from scripts.run_bfs_maze_solver import main as maze_main
from scripts.run_depth_limited_search import main as dls_script_main
from scripts.run_iterative_deepening_search import main as ids_script_main
from scripts.run_bfs_dfs_performance_analyzer import main as bfs_dfs_script_main
from scripts.run_bidirectional_search import main as bds_script_main
from src.__main__ import main as cli_main


class TestScriptExecution(unittest.TestCase):
    """Test suite for validating CLI execution of all organized scripts."""

    def test_smoke_test_script_runs_successfully(self) -> None:
        """Verify smoke_test main function executes and produces expected trial summary."""
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            smoke_main()
        output = stdout_trap.getvalue()
        self.assertIn("Hello, World!", output)
        self.assertIn("Trial completed successfully", output)

    def test_vacuum_agent_script_default_run(self) -> None:
        """Verify run_vacuum_agent script executes standard 5-step simulation."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_vacuum_agent.py", "--steps", "2"]):
            with patch("sys.stdout", stdout_trap):
                vacuum_main()
        output = stdout_trap.getvalue()
        self.assertIn("SIMPLE REFLEX VACUUM CLEANER AGENT SIMULATION", output)
        self.assertIn("Step 1:", output)

    def test_randomized_agent_script_reproducible_seed(self) -> None:
        """Verify run_randomized_agent script completes reproducible 12-step trial."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_randomized_agent.py", "--seed", "58"]):
            with patch("sys.stdout", stdout_trap):
                randomized_main()
        output = stdout_trap.getvalue()
        self.assertIn("RANDOMIZED REFLEX AGENT SIMULATION", output)
        self.assertIn("All rooms clean in 12 steps", output)

    def test_model_based_agent_script_default_run(self) -> None:
        """Verify run_model_based_agent script completes default 2x2 grid trial."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_model_based_agent.py"]):
            with patch("sys.stdout", stdout_trap):
                model_main()
        output = stdout_trap.getvalue()
        self.assertIn("MODEL-BASED REFLEX AGENT SIMULATION", output)
        self.assertIn("Shutdown", output)

    def test_goal_based_agent_script_default_run(self) -> None:
        """Verify run_goal_based_agent script executes sample scenario and outputs path."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_goal_based_agent.py"]):
            with patch("sys.stdout", stdout_trap):
                goal_main()
        output = stdout_trap.getvalue()
        self.assertIn("GOAL-BASED NAVIGATION AGENT SIMULATION", output)
        self.assertIn("Path: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]", output)
        self.assertIn("SUCCESS (Goal Reached)", output)

    def test_goal_based_agent_script_challenge_mode(self) -> None:
        """Verify run_goal_based_agent script demonstrates diagonal prevention challenge."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_goal_based_agent.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                goal_main()
        output = stdout_trap.getvalue()
        self.assertIn("EXPLICIT DIAGONAL MOVEMENT PREVENTION", output)
        self.assertIn("Diagonal movement prohibited", output)

    def test_goal_based_agent_script_viva_notes(self) -> None:
        """Verify run_goal_based_agent script displays viva reference notes."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_goal_based_agent.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                goal_main()
        output = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE REFERENCE & THEORY GUIDE", output)
        self.assertIn("DIFFERENCE BETWEEN GOAL-BASED AND REFLEX AGENTS", output)
        self.assertIn("MANHATTAN VS. EUCLIDEAN DISTANCE", output)

    def test_utility_based_agent_script_default_run(self) -> None:
        """Verify run_utility_based_agent script executes 3 predefined paths and outputs optimal path."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_utility_based_agent.py"]):
            with patch("sys.stdout", stdout_trap):
                utility_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION 5: UTILITY-BASED AGENT SIMULATION", output)
        self.assertIn("Selected Path: A-C-D", output)
        self.assertIn("Utility:       -4.0", output)

    def test_utility_based_agent_script_challenge_mode(self) -> None:
        """Verify run_utility_based_agent script runs battery constraint challenge."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_utility_based_agent.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                utility_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: BATTERY-CONSTRAINED UTILITY AGENT", output)
        self.assertIn("Battery Capacity Limit: 5.0 units", output)
        self.assertIn("Best Feasible Path: A-C-D", output)

    def test_utility_based_agent_script_viva_notes(self) -> None:
        """Verify run_utility_based_agent script displays viva reference notes."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_utility_based_agent.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                utility_main()
        output = stdout_trap.getvalue()
        self.assertIn("Question 5: Utility-Based Agent Simulation", output)
        self.assertIn("UTILITY VS. GOAL", output)
        self.assertIn("EXPECTED UTILITY THEORY", output)

    def test_multi_agent_script_default_run(self) -> None:
        """Verify run_multi_agent_simulation script executes 10-step simulation and displays matrix."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_agent_simulation.py", "--steps", "10", "--seed", "42"]):
            with patch("sys.stdout", stdout_trap):
                multi_agent_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 6: MULTI-AGENT ENVIRONMENT SIMULATION", output)
        self.assertIn("Sample Output (NumPy Matrix displaying 1s and 0s):", output)
        self.assertIn("[[1 1 0 0]", output)
        self.assertIn("MULTI-AGENT ENVIRONMENT SIMULATION SUMMARY", output)

    def test_multi_agent_script_challenge_mode(self) -> None:
        """Verify run_multi_agent_simulation script demonstrates collision avoidance challenge."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_agent_simulation.py", "--challenge", "--seed", "42"]):
            with patch("sys.stdout", stdout_trap):
                multi_agent_main()
        output = stdout_trap.getvalue()
        self.assertIn("Collision Avoidance: ENABLED", output)
        self.assertIn("Collisions Occurred:        0", output)
        self.assertIn("Collisions Avoided:         2", output)

    def test_multi_agent_script_viva_notes(self) -> None:
        """Verify run_multi_agent_simulation script displays viva notes."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_agent_simulation.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                multi_agent_main()
        output = stdout_trap.getvalue()
        self.assertIn("Question 6: Multi-Agent Environment Simulation", output)
        self.assertIn("MULTI-AGENT COMPETITIVE ENVIRONMENTS", output)
        self.assertIn("EPISODIC VS. SEQUENTIAL ENVIRONMENTS", output)
        self.assertIn("STATIC VS. DYNAMIC ENVIRONMENTS", output)

    def test_stochastic_vacuum_script_reproducible_sample_output(self) -> None:
        """Verify run_stochastic_vacuum_agent reproduces sample output of 5.4 steps with seed 28."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_stochastic_vacuum_agent.py", "--runs", "20", "--seed", "28"]):
            with patch("sys.stdout", stdout_trap):
                stochastic_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 7: STOCHASTIC ENVIRONMENT VACUUM AGENT", output)
        self.assertIn("Average steps to clean 3 rooms: 5.4 steps", output)
        self.assertIn("Deterministic Baseline      : 5 steps", output)
        self.assertIn("Absolute Performance Penalty: +0.40 steps", output)

    def test_stochastic_vacuum_script_challenge_mode(self) -> None:
        """Verify run_stochastic_vacuum_agent script executes slip back challenge."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_stochastic_vacuum_agent.py", "--challenge", "--seed", "42"]):
            with patch("sys.stdout", stdout_trap):
                stochastic_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE ACTIVATED: Actuator 'Slip Back' Mechanics", output)
        self.assertIn("Average steps to clean 3 rooms:", output)

    def test_stochastic_vacuum_script_viva_notes(self) -> None:
        """Verify run_stochastic_vacuum_agent script outputs viva guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_stochastic_vacuum_agent.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                stochastic_main()
        output = stdout_trap.getvalue()
        self.assertIn("Question 7: Stochastic Environment Vacuum Agent", output)
        self.assertIn("DETERMINISTIC VS. STOCHASTIC ENVIRONMENTS & ACTUATORS", output)
        self.assertIn("EXPECTED PERFORMANCE & MONTE CARLO SIMULATION", output)
        self.assertIn("ACTUATORS IN ARTIFICIAL INTELLIGENCE (PEAS FRAMEWORK)", output)

    def test_environment_classifier_script_direct_input(self) -> None:
        """Verify run_environment_classifier script with sample input (Backgammon)."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_environment_classifier.py", "--input", "True,False,False,False,True"],
        ):
            with patch("sys.stdout", stdout_trap):
                env_classifier_main()
        output = stdout_trap.getvalue()
        self.assertIn("ENVIRONMENT TYPE CLASSIFIER REPORT", output)
        self.assertIn("Environment Match: Backgammon", output)

    def test_environment_classifier_script_individual_flags(self) -> None:
        """Verify run_environment_classifier script with individual property flags."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run_environment_classifier.py",
                "--observable",
                "yes",
                "--deterministic",
                "yes",
                "--episodic",
                "no",
                "--static",
                "no",
                "--discrete",
                "yes",
            ],
        ):
            with patch("sys.stdout", stdout_trap):
                env_classifier_main()
        output = stdout_trap.getvalue()
        self.assertIn("Environment Match: Chess (with clock)", output)

    def test_environment_classifier_script_demo_mode(self) -> None:
        """Verify run_environment_classifier script executes demo suite cleanly."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_environment_classifier.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                env_classifier_main()
        output = stdout_trap.getvalue()
        self.assertIn("STANDARD AI TEXTBOOK ENVIRONMENTS DEMONSTRATION SUITE", output)
        self.assertIn("Evaluating: Sample Case (Backgammon)", output)

    def test_environment_classifier_script_challenge_mode(self) -> None:
        """Verify run_environment_classifier script executes challenge distance metric."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_environment_classifier.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                env_classifier_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: NEAREST-NEIGHBOR DISTANCE METRIC", output)
        self.assertIn("Nearest Environment Matches", output)

    def test_environment_classifier_script_viva(self) -> None:
        """Verify run_environment_classifier script prints Question 8 viva guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_environment_classifier.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                env_classifier_main()
        output = stdout_trap.getvalue()
        self.assertIn("Question 8: Environment Type Classifier", output)
        self.assertIn("PEAS DESCRIPTION", output)
        self.assertIn("PROPERTIES OF CHESS VS. POKER", output)
        self.assertIn("CONTINUOUS VS. DISCRETE", output)

    def test_agent_performance_evaluator_script_default(self) -> None:
        """Verify run_agent_performance_evaluator script prints exact sample output."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_agent_performance_evaluator.py"]):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("Performance Score: 2 (10 - 3 moves - 5 bump)", output)

    def test_agent_performance_evaluator_script_events(self) -> None:
        """Verify run_agent_performance_evaluator script evaluates custom --events."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys, "argv", ["run_agent_performance_evaluator.py", "--events", "move,move,deliver"]
        ):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("Performance Score: 8 (10 - 2 moves)", output)

    def test_agent_performance_evaluator_script_detailed(self) -> None:
        """Verify run_agent_performance_evaluator script outputs detailed table."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_agent_performance_evaluator.py", "--detailed"]):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("AUTONOMOUS DELIVERY DRONE PERFORMANCE AUDIT", output)
        self.assertIn("Step-by-Step Flight Trace:", output)

    def test_agent_performance_evaluator_script_csv(self) -> None:
        """Verify run_agent_performance_evaluator script reads sample CSV."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_agent_performance_evaluator.py", "--csv", "data/raw/drone_actions_sample.csv"],
        ):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("Performance Score: 2 (10 - 3 moves - 5 bump)", output)

    def test_agent_performance_evaluator_script_demo(self) -> None:
        """Verify run_agent_performance_evaluator script executes demonstration suite."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_agent_performance_evaluator.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("AUTONOMOUS DRONE PERFORMANCE EVALUATION - BENCHMARK SUITE", output)
        self.assertIn("Scenario 1: Standard Sample Benchmark", output)

    def test_agent_performance_evaluator_script_viva(self) -> None:
        """Verify run_agent_performance_evaluator script prints viva notes."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_agent_performance_evaluator.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                agent_eval_main()
        output = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE THEORY & REFERENCE GUIDE", output)
        self.assertIn("DESIGNING PERFORMANCE MEASURES", output)
        self.assertIn("RATIONALITY DEFINITION", output)

    def test_state_space_visualizer_script_table(self) -> None:
        """Verify run_state_space_visualizer script prints 8-state transition table."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("STATE-SPACE TRANSITION TABLE: 2-ROOM REFLEX VACUUM AGENT", output)
        self.assertIn("Total States: 8", output)
        self.assertIn("Self-Loop Cycles (↺): 12", output)

    def test_state_space_visualizer_script_cycles(self) -> None:
        """Verify run_state_space_visualizer script prints cycle and self-loop audit."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--cycles"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("CYCLIC TRANSITIONS & SELF-LOOP AUDIT", output)
        self.assertIn("Self-Loop Cycles Result(s, a) == s: 12", output)
        self.assertIn("Action 'Left ': 4 self-loops", output)

    def test_state_space_visualizer_script_matrix(self) -> None:
        """Verify run_state_space_visualizer script prints adjacency matrix."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--matrix"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("8x8 ADJACENCY MATRIX OF STATE-SPACE GRAPH", output)

    def test_state_space_visualizer_script_search(self) -> None:
        """Verify run_state_space_visualizer script computes shortest paths."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--search"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("SHORTEST OPTIMAL ACTION PATHS TO GOAL", output)
        self.assertIn("Suck -> Right -> Suck", output)

    def test_state_space_visualizer_script_viva(self) -> None:
        """Verify run_state_space_visualizer script outputs viva guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE REFERENCE GUIDE: STATE-SPACE FORMULATION", output)
        self.assertIn("STATE SPACE SIZE CALCULATION", output)

    def test_breadth_first_search_script_default_run(self) -> None:
        """Verify run_breadth_first_search script runs Question 11 default sample graph."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_breadth_first_search.py"]):
            with patch("sys.stdout", stdout_trap):
                bfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("BFS TRAVERSAL EXECUTION (START NODE: 'A')", output)
        self.assertIn("BFS Traversal  : A, B, C, D", output)
        self.assertIn("Total Levels   : 3 level(s)", output)

    def test_breadth_first_search_script_custom_graph(self) -> None:
        """Verify run_breadth_first_search script parses custom graph via CLI."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_breadth_first_search.py", "--graph", "A: B, C; B: D; C:; D:", "--start", "A"],
        ):
            with patch("sys.stdout", stdout_trap):
                bfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("BFS Traversal  : A, B, C, D", output)

    def test_breadth_first_search_script_challenge(self) -> None:
        """Verify run_breadth_first_search script executes optional challenge level breakdown."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_breadth_first_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                bfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: GRAPH LEVELS & TREE DEPTH COMPUTATION", output)
        self.assertIn("Level 0 (hops = 0): ['A']", output)

    def test_breadth_first_search_script_demo(self) -> None:
        """Verify run_breadth_first_search script executes benchmarks in demo mode."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_breadth_first_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                bfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("DEMO: BREADTH-FIRST SEARCH ON BENCHMARK GRAPHS", output)
        self.assertIn("Cyclic Directed Graph", output)

    def test_breadth_first_search_script_viva(self) -> None:
        """Verify run_breadth_first_search script outputs viva voce study guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_breadth_first_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                bfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION 11: BREADTH-FIRST SEARCH (BFS) VIVA VOCE STUDY GUIDE", output)
        self.assertIn("FIFO Queue Mechanism", output)

    def test_depth_first_search_script_default_run(self) -> None:
        """Verify run_depth_first_search script executes default sample run matching Question 12 output."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_first_search.py"]):
            with patch("sys.stdout", stdout_trap):
                dfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("DEPTH-FIRST SEARCH TRAVERSAL (QUESTION 12)", output)
        self.assertIn("DFS Traversal            : 1, 3, 2, 4", output)
        self.assertIn("Step-by-Step Execution Trace Table", output)

    def test_depth_first_search_script_challenge(self) -> None:
        """Verify run_depth_first_search script executes Optional Challenge comparison."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_first_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                dfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: ITERATIVE VS. RECURSIVE DFS COMPARISON", output)
        self.assertIn("1. Iterative (Standard Forward-Push)", output)
        self.assertIn("2. Recursive (Standard Forward-Call)", output)

    def test_depth_first_search_script_demo(self) -> None:
        """Verify run_depth_first_search script executes benchmarks in demo mode."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_first_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                dfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("DEPTH-FIRST SEARCH AUTOMATED BENCHMARK SUITE", output)
        self.assertIn("Cyclic Benchmark Graph", output)

    def test_depth_first_search_script_viva(self) -> None:
        """Verify run_depth_first_search script outputs viva voce theory guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_first_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                dfs_main()
        output = stdout_trap.getvalue()
        self.assertIn("DEPTH-FIRST SEARCH (DFS) VIVA VOCE REFERENCE & THEORY GUIDE", output)
        self.assertIn("STACK VS QUEUE", output)
        self.assertIn("WHY DFS IS NOT OPTIMAL", output)

    def test_uniform_cost_search_script_default_run(self) -> None:
        """Verify run_uniform_cost_search executes default sample scenario."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_uniform_cost_search.py"]):
            with patch("sys.stdout", stdout_trap):
                ucs_main()
        output = stdout_trap.getvalue()
        self.assertIn("UNIFORM COST SEARCH: A -> C", output)
        self.assertIn("Path: A->B->C, Total Cost: 3", output)
        self.assertIn("Step-by-Step Priority Queue Trace Table", output)

    def test_uniform_cost_search_script_custom_args(self) -> None:
        """Verify run_uniform_cost_search accepts custom graph, start, and goal arguments."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_uniform_cost_search.py", "--graph", "X->Y:5, Y->Z:10, X->Z:20", "--start", "X", "--goal", "Z"],
        ):
            with patch("sys.stdout", stdout_trap):
                ucs_main()
        output = stdout_trap.getvalue()
        self.assertIn("UNIFORM COST SEARCH: X -> Z", output)
        self.assertIn("Path: X->Y->Z, Total Cost: 15", output)

    def test_uniform_cost_search_script_challenge(self) -> None:
        """Verify run_uniform_cost_search script executes challenge mode demonstration."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_uniform_cost_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                ucs_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: NEGATIVE EDGE WEIGHTS & NEGATIVE CYCLES IN UCS", output)
        self.assertIn("Reachable Negative Weight Cycle Detection via Bellman-Ford", output)
        self.assertIn("Negative Cycle Path", output)

    def test_uniform_cost_search_script_demo(self) -> None:
        """Verify run_uniform_cost_search script executes demo mode benchmarks."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_uniform_cost_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                ucs_main()
        output = stdout_trap.getvalue()
        self.assertIn("AUTOMATED BENCHMARKS: UNIFORM COST SEARCH TOPOLOGIES", output)
        self.assertIn("Diamond Graph", output)
        self.assertIn("Romania Map", output)

    def test_uniform_cost_search_script_viva(self) -> None:
        """Verify run_uniform_cost_search script outputs viva voce theory guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_uniform_cost_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                ucs_main()
        output = stdout_trap.getvalue()
        self.assertIn("UNIFORM COST SEARCH (UCS) VIVA VOCE REFERENCE & THEORY GUIDE", output)
        self.assertIn("UCS VS BFS", output)
        self.assertIn("PRIORITY QUEUE MECHANISMS & PYTHON HEAPQ", output)

    def test_bfs_maze_solver_script_default_run(self) -> None:
        """Verify run_bfs_maze_solver script executes with default sample grid and produces shortest steps 4."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_maze_solver.py"]):
            with patch("sys.stdout", stdout_trap):
                maze_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION 15: BFS 2D GRID MAZE SOLVER", output)
        self.assertIn("Shortest steps: 4", output)
        self.assertIn('Moves: "D, R, R, D"', output)
        self.assertIn("Step-by-Step FIFO Frontier & Dynamic Successor Trace", output)

    def test_bfs_maze_solver_script_custom_grid(self) -> None:
        """Verify run_bfs_maze_solver accepts custom grid string from CLI."""
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_bfs_maze_solver.py", "--grid", "[[0,0],[1,0]]", "--no-trace"],
        ):
            with patch("sys.stdout", stdout_trap):
                maze_main()
        output = stdout_trap.getvalue()
        self.assertIn("Shortest steps: 2", output)
        self.assertIn('Moves: "R, D"', output)

    def test_bfs_maze_solver_script_challenge(self) -> None:
        """Verify run_bfs_maze_solver script executes challenge mode demonstration."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_maze_solver.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                maze_main()
        output = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: DIRECTIONAL MOVE STRING ('D, R, R, D')", output)
        self.assertIn("Actual Computed Shortest Steps: 4", output)
        self.assertIn('Actual Computed Moves Sequence   : "D, R, R, D"', output)
        self.assertIn("Challenge passed with 100% precision", output)

    def test_bfs_maze_solver_script_demo(self) -> None:
        """Verify run_bfs_maze_solver script executes demo benchmarks across topologies."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_maze_solver.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                maze_main()
        output = stdout_trap.getvalue()
        self.assertIn("AUTOMATED BENCHMARKS: BFS 2D MAZE TOPOLOGIES", output)
        self.assertIn("Default Sample Maze", output)
        self.assertIn("Non-Square Rectangular Maze", output)
        self.assertIn("Winding Serpentine Corridor Maze", output)
        self.assertIn("Single-Cell Maze", output)

    def test_bfs_maze_solver_script_viva(self) -> None:
        """Verify run_bfs_maze_solver script outputs viva voce theory guide."""
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_maze_solver.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                maze_main()
        output = stdout_trap.getvalue()
        self.assertIn("QUESTION 15: BFS 2D MAZE SOLVER - VIVA VOCE & THEORY GUIDE", output)
        self.assertIn("Implicit vs. Explicit Graphs", output)
        self.assertIn("Valid Action Generation", output)
        self.assertIn("Manhattan Distance Relation", output)

    def test_cli_dispatcher_help_and_subcommands(self) -> None:
        """Verify top-level module CLI dispatcher provides help and forwards commands."""
        # Help text
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["--help"])
        self.assertIn("AI Lab: Intelligent Agent Simulations", stdout_trap.getvalue())

        # Subcommand forwarding for smoke-test
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["smoke-test"])
        self.assertIn("Hello, World!", stdout_trap.getvalue())

        # Subcommand forwarding for goal-based
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["goal-based"])
        self.assertIn("GOAL-BASED NAVIGATION AGENT SIMULATION", stdout_trap.getvalue())

        # Subcommand forwarding for utility-based
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["utility-based"])
        self.assertIn("QUESTION 5: UTILITY-BASED AGENT SIMULATION", stdout_trap.getvalue())

        # Subcommand forwarding for multi-agent
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["multi-agent", "--steps", "10", "--seed", "42"])
        self.assertIn("QUESTION NO. 6: MULTI-AGENT ENVIRONMENT SIMULATION", stdout_trap.getvalue())

        # Subcommand forwarding for stochastic
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["stochastic", "--runs", "20", "--seed", "28"])
        self.assertIn("Average steps to clean 3 rooms: 5.4 steps", stdout_trap.getvalue())

        # Subcommand forwarding for environment-classifier
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["environment-classifier", "--input", "True,False,False,False,True"])
        self.assertIn("Environment Match: Backgammon", stdout_trap.getvalue())

        # Subcommand forwarding for agent-evaluator
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["agent-evaluator", "--events", "move,move,bump,move,deliver"])
        self.assertIn("Performance Score: 2 (10 - 3 moves - 5 bump)", stdout_trap.getvalue())

        # Subcommand forwarding for state-space
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["state-space", "--table"])
        self.assertIn("STATE-SPACE TRANSITION TABLE: 2-ROOM REFLEX VACUUM AGENT", stdout_trap.getvalue())

        # Subcommand forwarding for bfs
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["bfs", "--start", "A"])
        self.assertIn("BFS Traversal  : A, B, C, D", stdout_trap.getvalue())

        # Subcommand forwarding for dfs
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["dfs", "--start", "1"])
        self.assertIn("DFS Traversal            : 1, 3, 2, 4", stdout_trap.getvalue())

        # Subcommand forwarding for ucs
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["ucs", "--start", "A", "--goal", "C"])
        self.assertIn("Path: A->B->C, Total Cost: 3", stdout_trap.getvalue())

        # Subcommand forwarding for maze
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["maze", "--no-trace"])
        self.assertIn("Shortest steps: 4", stdout_trap.getvalue())
        self.assertIn('Moves: "D, R, R, D"', stdout_trap.getvalue())

        # Subcommand forwarding for dls
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["dls", "--start", "1", "--goal", "6", "--limit", "2"])
        self.assertIn("Official Practical Output: Result: Cutoff", stdout_trap.getvalue())
        self.assertIn("Deepest Node Reached     : 4", stdout_trap.getvalue())

        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["dls", "--start", "1", "--goal", "6", "--limit", "3"])
        self.assertIn("Official Practical Output: Result: Path: ['1', '2', '4', '6']", stdout_trap.getvalue())

    def test_dls_script_default_and_flags(self) -> None:
        """Verify run_depth_limited_search script executes sample, challenge, demo, and viva modes."""
        # 1. Default run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_limited_search.py"]):
            with patch("sys.stdout", stdout_trap):
                dls_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("DEPTH-LIMITED SEARCH (DLS) EXECUTION RESULT", out)
        self.assertIn("Official Practical Output: Result: Cutoff", out)
        self.assertIn("Deepest Node Reached     : 4", out)

        # 2. Challenge mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_limited_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                dls_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: TRACKING DEEPEST NODE BEFORE CUTOFF", out)

        # 3. Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_limited_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                dls_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("DEPTH-LIMITED SEARCH (DLS) COMPREHENSIVE BENCHMARK SUITE", out)

        # 4. Viva mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_depth_limited_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                dls_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE REFERENCE & THEORY GUIDE", out)
        self.assertIn("Infinite State Spaces:", out)
        self.assertIn("Cutoff vs Failure:", out)

    def test_ids_cli_and_script_modes(self) -> None:
        """Verify IDS subcommand forwarding and script default, compare-bfs, challenge, demo, and viva modes."""
        # Subcommand forwarding for ids
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["ids", "--start", "1", "--goal", "4"])
        out = stdout_trap.getvalue()
        self.assertIn("Limit 0: Cutoff. Limit 1: Cutoff. Limit 2: Found ['1', '2', '4'].", out)
        self.assertIn("Total Node Expansions", out)

        # 1. Default run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_iterative_deepening_search.py"]):
            with patch("sys.stdout", stdout_trap):
                ids_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("ITERATIVE DEEPENING SEARCH (IDS) EXECUTION RESULT", out)
        self.assertIn("Nodes Expanded at Each Depth Iteration:", out)

        # 2. Compare BFS mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_iterative_deepening_search.py", "--compare-bfs"]):
            with patch("sys.stdout", stdout_trap):
                ids_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("IDS vs BFS COMPARISON", out)
        self.assertIn("Shortest Path Match?  : True", out)

        # 3. Challenge mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_iterative_deepening_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                ids_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: TOTAL NODE EXPANSIONS", out)

        # 4. Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_iterative_deepening_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                ids_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("ITERATIVE DEEPENING SEARCH (IDS) COMPREHENSIVE BENCHMARK SUITE", out)

        # 5. Viva mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_iterative_deepening_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                ids_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("ITERATIVE DEEPENING SEARCH (IDS) VIVA VOCE REFERENCE & THEORY GUIDE", out)
        self.assertIn("Redundant Node Generation:", out)

    def test_bfs_dfs_performance_analyzer_cli_and_script_modes(self) -> None:
        """Verify Question 18 BFS vs DFS CLI forwarding and script modes (default, single, dfs-favored, demo, viva)."""
        # 1. Subcommand forwarding via python -m src bfs-dfs
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["bfs-dfs", "--format", "single"])
        out = stdout_trap.getvalue()
        self.assertIn("BFS Nodes Expanded: 45, Path Length: 12", out)
        self.assertIn("DFS Nodes Expanded: 40, Path Length: 22", out)

        # 2. Default script run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_dfs_performance_analyzer.py"]):
            with patch("sys.stdout", stdout_trap):
                bfs_dfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 18: BFS vs DFS PERFORMANCE ANALYZER", out)
        self.assertIn("COMPARATIVE METRICS SUMMARY TABLE:", out)
        self.assertIn("BFS Path (Optimal Shortest Path: *):", out)
        self.assertIn("DFS Path (Arbitrary Path: *):", out)

        # 3. Single format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_dfs_performance_analyzer.py", "--format", "single"]):
            with patch("sys.stdout", stdout_trap):
                bfs_dfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BFS Nodes Expanded: 45, Path Length: 12", out)

        # 4. DFS favored maze mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_dfs_performance_analyzer.py", "--maze", "dfs-favored", "--format", "single"]):
            with patch("sys.stdout", stdout_trap):
                bfs_dfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BFS Nodes Expanded: 45, Path Length: 12 | DFS Nodes Expanded: 14, Path Length: 12", out)

        # 5. Viva mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_dfs_performance_analyzer.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                bfs_dfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BFS vs DFS PERFORMANCE ANALYZER VIVA VOCE THEORY GUIDE", out)
        self.assertIn("ALGORITHM OPTIMALITY:", out)
        self.assertIn("SCENARIOS WHERE DFS OUTPERFORMS BFS:", out)
        self.assertIn("BRANCHING FACTOR IMPACT:", out)

        # 6. Demo benchmark suite mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bfs_dfs_performance_analyzer.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                bfs_dfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BFS vs DFS 10x10 MAZE BENCHMARK SUITE", out)
        self.assertIn("Benchmark 1: Default 10x10 Benchmark Maze", out)

    def test_bidirectional_search_script_and_cli_modes(self) -> None:
        """Verify Question 19 Bidirectional Search CLI forwarding and script modes (default, single, demo, challenge, viva)."""
        # 1. Subcommand forwarding via python -m src bidirectional --format single
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["bidirectional", "--format", "single"])
        out = stdout_trap.getvalue()
        self.assertIn("Intersection at D. Path: A-B-D-F-G", out)

        # 2. Subcommand alias via python -m src bds --format single
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["bds", "--format", "single"])
        out = stdout_trap.getvalue()
        self.assertIn("Intersection at D. Path: A-B-D-F-G", out)

        # 3. Default script run (Question 19 full report & table)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bidirectional_search.py"]):
            with patch("sys.stdout", stdout_trap):
                bds_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 19: BIDIRECTIONAL SEARCH SIMULATION", out)
        self.assertIn("Intersection at D. Path: A-B-D-F-G", out)
        self.assertIn("STEP-BY-STEP ALTERNATING EXPANSION TRACE TABLE:", out)

        # 4. Single format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bidirectional_search.py", "--format", "single"]):
            with patch("sys.stdout", stdout_trap):
                bds_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "Intersection at D. Path: A-B-D-F-G")

        # 5. Optional Challenge: 2D Grid Maze mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bidirectional_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                bds_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: 2D GRID MAZE BIDIRECTIONAL SEARCH", out)
        self.assertIn("Intersection Coordinate: (4, 3)", out)
        self.assertIn("Shortest Path Length   : 12 steps", out)
        self.assertIn("MAZE VISUALIZATION:", out)

        # 6. Demo benchmark suite mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bidirectional_search.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                bds_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BIDIRECTIONAL SEARCH (BDS) BENCHMARK SUITE", out)
        self.assertIn("Benchmark 1: Canonical Sample Graph (Syllabus Default)", out)
        self.assertIn("Search Space Halving: Unidirectional BFS vs Bidirectional BFS", out)

        # 7. Viva Voce study guide mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_bidirectional_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                bds_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 19: BIDIRECTIONAL SEARCH VIVA VOCE THEORY GUIDE", out)
        self.assertIn("COMPLEXITY ADVANTAGE: O(b^(d/2)) vs O(b^d)", out)
        self.assertIn("DIFFICULTIES OF BACKWARD SEARCH IN DIRECTED GRAPHS", out)
        self.assertIn("PREDECESSOR vs SUCCESSOR GENERATION", out)


if __name__ == "__main__":
    unittest.main()
