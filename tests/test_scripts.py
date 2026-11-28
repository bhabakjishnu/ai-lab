"""Unit tests for script entry points and top-level CLI dispatcher.

Ensures that all organized scripts in scripts/ and module entry points
execute cleanly with expected return codes and outputs.
"""

import io
import json
import sys
import unittest
from unittest.mock import patch

from scripts.smoke_test import main as smoke_main
from scripts.run_vacuum_agent import main as vacuum_main
from scripts.run_randomized_agent import main as randomized_main
from scripts.run_model_based_agent import main as model_main
from scripts.run_goal_based_agent import main as goal_main
from scripts.run_utility_based_agent import main as utility_main
try:
    from scripts.run_multi_agent_simulation import main as multi_agent_main
except (ImportError, OSError):
    multi_agent_main = None
from scripts.run_stochastic_vacuum_agent import main as stochastic_main
from scripts.run_environment_classifier import main as env_classifier_main
from scripts.run_agent_performance_evaluator import main as agent_eval_main
try:
    from scripts.run_state_space_visualizer import main as state_space_main
except (ImportError, OSError):
    state_space_main = None
from scripts.run_breadth_first_search import main as bfs_main
from scripts.run_depth_first_search import main as dfs_main
from scripts.run_uniform_cost_search import main as ucs_main
from scripts.run_bfs_maze_solver import main as maze_main
from scripts.run_depth_limited_search import main as dls_script_main
from scripts.run_iterative_deepening_search import main as ids_script_main
from scripts.run_bfs_dfs_performance_analyzer import main as bfs_dfs_script_main
from scripts.run_bidirectional_search import main as bds_script_main
from scripts.run_water_jug_bfs import main as water_jug_script_main
from scripts.run_missionaries_cannibals_dfs import main as mc_script_main
from scripts.run_ucs_grid_navigation import main as ucs_grid_script_main
from scripts.run_multi_goal_bfs_maze import main as multi_goal_bfs_script_main
from scripts.run_distance_calculators import main as distance_script_main
from scripts.run_eight_puzzle_heuristics import main as eight_puzzle_script_main
from scripts.run_greedy_best_first_search import main as gbfs_script_main
from scripts.run_gbfs_grid_navigator import main as gbfs_grid_script_main
from scripts.run_a_star_search import main as astar_script_main
from scripts.run_a_star_maze_solver import main as astar_maze_script_main
from scripts.run_a_star_eight_puzzle import main as astar_puzzle_script_main
from scripts.run_eight_puzzle_manhattan_vs_misplaced import main as puzzle_dominance_script_main
from scripts.run_ucs_vs_astar_comparison import main as ucs_vs_astar_script_main
from scripts.run_admissible_heuristic_validator import main as admissible_validator_script_main
from scripts.run_consistent_heuristic_validator import main as consistent_validator_script_main
from scripts.run_a_star_octile_distance import main as astar_octile_script_main
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
        if multi_agent_main is None:
            self.skipTest("scripts.run_multi_agent_simulation unavailable")
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
        if multi_agent_main is None:
            self.skipTest("scripts.run_multi_agent_simulation unavailable")
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
        if multi_agent_main is None:
            self.skipTest("scripts.run_multi_agent_simulation unavailable")
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
        if state_space_main is None:
            self.skipTest("scripts.run_state_space_visualizer unavailable")
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
        if state_space_main is None:
            self.skipTest("scripts.run_state_space_visualizer unavailable")
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
        if state_space_main is None:
            self.skipTest("scripts.run_state_space_visualizer unavailable")
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--matrix"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("8x8 ADJACENCY MATRIX OF STATE-SPACE GRAPH", output)

    def test_state_space_visualizer_script_search(self) -> None:
        """Verify run_state_space_visualizer script computes shortest paths."""
        if state_space_main is None:
            self.skipTest("scripts.run_state_space_visualizer unavailable")
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_state_space_visualizer.py", "--search"]):
            with patch("sys.stdout", stdout_trap):
                state_space_main()
        output = stdout_trap.getvalue()
        self.assertIn("SHORTEST OPTIMAL ACTION PATHS TO GOAL", output)
        self.assertIn("Suck -> Right -> Suck", output)

    def test_state_space_visualizer_script_viva(self) -> None:
        """Verify run_state_space_visualizer script outputs viva guide."""
        if state_space_main is None:
            self.skipTest("scripts.run_state_space_visualizer unavailable")
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

    def test_water_jug_bfs_script_execution(self) -> None:
        """Verify Question 20 Water Jug Problem runner script across all CLI modes."""
        # 1. Module execution: python -m src water-jug --format single
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["water-jug", "--format", "single"])
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)")

        # 2. Module execution with alias: python -m src jug-bfs --format single
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["jug-bfs", "--format", "single"])
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)")

        # 3. Default script run (Question 20 full report, table & ASCII diagrams)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 20: WATER JUG PROBLEM USING BFS", out)
        self.assertIn("(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)", out)
        self.assertIn("STEP-BY-STEP TRANSITION TRACE TABLE:", out)
        self.assertIn("ASCII WATER LEVEL DIAGRAMS:", out)

        # 4. Single format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--format", "single"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)")

        # 5. Table format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--format", "table"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("| Step | Action       | Rule # | State (J1,J2)|", out)
        self.assertIn("Fill 1", out)
        self.assertIn("Pour 1->2", out)

        # 6. JSON format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--format", "json"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn('"arrow_path_string": "(0,0) -> (4,0) -> (1,3) -> (1,0) -> (0,1) -> (4,1) -> (2,3)"', out)

        # 7. Optional Challenge: Algebraic reachability analysis
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: ALGEBRAIC REACHABILITY & GCD RULE ANALYSIS", out)
        self.assertIn("Theorem (Bézout's Identity & Diophantine Solvability)", out)
        self.assertIn("[REACHABLE]", out)
        self.assertIn("[UNREACHABLE]", out)

        # 8. Demo benchmark suite mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("WATER JUG BFS BENCHMARK SUITE", out)
        self.assertIn("Benchmark 1: Canonical Sample (Syllabus Default)", out)
        self.assertIn("Benchmark 2: Die Hard 3 Puzzle", out)
        self.assertIn("Benchmark 5: Unsolvable", out)

        # 9. Viva Voce study guide mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_water_jug_bfs.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                water_jug_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 20: WATER JUG PROBLEM VIVA VOCE THEORY GUIDE", out)
        self.assertIn("State Space Representation", out)
        self.assertIn("Action Rules Definition", out)
        self.assertIn("Unsolvable States & GCD Rule", out)

    def test_missionaries_cannibals_dfs_script_executions(self) -> None:
        """Comprehensive verification of Question 21 runner script across all CLI modes."""
        # 1. Default standard run (iterative DFS, 3M 3C)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("MISSIONARIES & CANNIBALS PROBLEM (DFS ITERATIVE)", out)
        self.assertIn("Step 1: 2C cross right -> (3, 1, 'Right')", out)
        self.assertIn("Goal Reached: (0, 0, 'Right') in 11 boat trips!", out)

        # 2. Table format mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--format", "table"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("+------+--------------------+--------------------+", out)
        self.assertIn("Left Bank (M, C)", out)
        self.assertIn("2C cross right", out)

        # 3. Arrow trace mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--format", "arrows"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("(3,3,Left) --[2C cross right]--> (3,1,Right)", out)

        # 4. Detailed ASCII river mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--format", "detailed"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("MISSIONARIES & CANNIBALS RIVER CROSSING TRACE", out)
        self.assertIn("Left Bank:", out)
        self.assertIn("[BOAT]", out)

        # 5. Recursive DFS mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--method", "recursive"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("DFS RECURSIVE", out)
        self.assertIn("Goal Reached: (0, 0, 'Right') in 11 boat trips!", out)

        # 6. Compare validation strategies (generation vs testing)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--compare-validation"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("CONSTRAINT VALIDATION: GENERATION vs TESTING", out)
        self.assertIn("Forward Checking", out)
        self.assertIn("Generate-and-Test", out)

        # 7. Compare DFS vs BFS
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--compare-bfs"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("ALGORITHM COMPARISON: DEPTH-FIRST SEARCH (DFS) vs BREADTH-FIRST SEARCH (BFS)", out)
        self.assertIn("DFS (LIFO Stack)", out)
        self.assertIn("BFS (FIFO Queue)", out)

        # 8. State space combinatorial analysis
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--state-space"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("STATE SPACE COMBINATORIAL ANALYSIS", out)
        self.assertIn("Theoretical Unconstrained States: 32", out)
        self.assertIn("Valid (Safe) States: 20", out)
        self.assertIn("Reachable Safe States from (3, 3, 'Left'): 16", out)

        # 9. Animation mode with zero delay
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--animate", "--delay", "0.0"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("MISSIONARIES & CANNIBALS RIVER CROSSING ANIMATION", out)
        self.assertIn("ALL 3 MISSIONARIES AND 3 CANNIBALS SAFELY CROSSED", out)

        # 10. JSON mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn('"success": true', out)
        self.assertIn('"total_trips": 11', out)

        # 11. Viva Voce guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 21: MISSIONARIES & CANNIBALS VIVA VOCE THEORY GUIDE", out)
        self.assertIn("Constraint Validation: Generation vs Testing", out)
        self.assertIn("State Space Size and Reachability", out)
        self.assertIn("Why BFS is Preferred Over DFS", out)

        # 12. Full Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_missionaries_cannibals_dfs.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                mc_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 21: MISSIONARIES & CANNIBALS FULL LABORATORY DEMO", out)

        # 13. Top-level CLI invocation via 'python -m src mc'
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["mc"])
        out = stdout_trap.getvalue()
        self.assertIn("MISSIONARIES & CANNIBALS PROBLEM (DFS ITERATIVE)", out)
        self.assertIn("Step 1: 2C cross right -> (3, 1, 'Right')", out)

    def test_run_ucs_grid_navigation_script(self) -> None:
        """Comprehensive verification of Question 22 runner script across all CLI modes."""
        # 1. Default execution on sample grid
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Cost: 4, Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)", out)

        # 2. Sample flag
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Cost: 4, Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)", out)

        # 3. Custom grid with ASCII visualization
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--grid", "...", ".M.", "..G", "--ascii"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIFORM COST SEARCH GRID RESULT", out)
        self.assertIn("ASCII Map Visualization", out)

        # 4. Diagonal mode (Optional Challenge)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--diagonal"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Cost: 3.41", out)

        # 5. BFS Comparison mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--compare-bfs"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UCS VS BFS COMPARATIVE PERFORMANCE ANALYSIS", out)
        self.assertIn("BFS (Hop-Optimal) Path", out)
        self.assertIn("UCS (Cost-Optimal) Path", out)

        # 6. JSON output mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn('"success": true', out)
        self.assertIn('"cost": 4', out)

        # 7. Viva Voce guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 22: UCS GRID NAVIGATION VIVA VOCE THEORY GUIDE", out)
        self.assertIn("Priority Queue popping mechanism", out)
        self.assertIn("Completeness", out)

        # 8. Animation mode with zero delay
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--animate", "--delay", "0.0"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UCS WAVEFRONT EXPANSION ANIMATION", out)
        self.assertIn("GOAL REACHED! OPTIMAL PATH FOUND!", out)

        # 9. Full Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_grid_navigation.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                ucs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 22: UCS GRID NAVIGATION COMPREHENSIVE LABORATORY DEMO", out)
        self.assertIn("STANDARD SAMPLE EXECUTION", out)
        self.assertIn("OPTIONAL CHALLENGE: DIAGONAL MOVES", out)

        # 10. Top-level CLI invocation via 'python -m src ucs-grid'
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["ucs-grid"])
        out = stdout_trap.getvalue()
        self.assertIn("Cost: 4, Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)", out)

    def test_multi_goal_bfs_maze_script_execution(self) -> None:
        """Verify Question 23: Multi-Goal BFS maze script runs across all CLI modes."""
        # 1. Exact sample mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Closest exit found at (1,8) in 9 steps.", out)

        # 2. Default execution with ASCII
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION 23: MAZE WITH MULTIPLE EXITS (MULTI-GOAL BFS)", out)
        self.assertIn("Closest exit found at (1,8) in 9 steps.", out)
        self.assertIn("SUCCESS", out)
        self.assertIn("ASCII MAZE SOLUTION", out)

        # 3. Optional Challenge mode: All exits sorted by distance
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--all-exits"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: ALL EXITS SORTED BY DISTANCE", out)
        self.assertIn("Exit (1, 8): 9 steps [CLOSEST]", out)
        self.assertIn("Exit (5, 5): 10 steps", out)

        # 4. Viva Voce Topic 1 benchmark
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--benchmark-set"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE TOPIC 1: TESTING FOR MULTIPLE GOALS EFFICIENTLY", out)
        self.assertIn("Speedup Factor", out)

        # 5. Viva Voce Topic 2 UCS comparison
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--compare-ucs"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE TOPIC 2: DOES UCS STILL WORK WITH MULTIPLE GOALS?", out)
        self.assertIn("UCS Compatible with Multiple Goals? : YES", out)

        # 6. JSON output mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn('"found": true', out)
        self.assertIn('"path_length": 9', out)
        self.assertIn('"closest_exit": [', out)

        # 7. Viva Voce theory guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIVERSITY VIVA VOCE GUIDE", out)
        self.assertIn("Multi-goal search spaces", out)

        # 8. Animation mode with zero delay
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--animate", "--delay", "0.0"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("INTERACTIVE TERMINAL SEARCH ANIMATION", out)
        self.assertIn("Search concluded in", out)

        # 9. Full Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_multi_goal_bfs_maze.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                multi_goal_bfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("COMPREHENSIVE MULTI-GOAL BFS DEMONSTRATION", out)
        self.assertIn("Detour Demonstration", out)

        # 10. Top-level CLI invocation via 'python -m src multi-goal-bfs'
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["multi-goal-bfs", "--sample"])
        out = stdout_trap.getvalue()
        self.assertIn("Closest exit found at (1,8) in 9 steps.", out)

        # 11. Top-level CLI invocation via alias 'python -m src maze23'
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["maze23", "--sample"])
        out = stdout_trap.getvalue()
        self.assertIn("Closest exit found at (1,8) in 9 steps.", out)

    def test_run_distance_calculators_script_runs_successfully(self) -> None:
        """Verify Question 24 Distance Calculators CLI execution and all functional modes."""
        # 1. Default Sample Execution: (1, 1), (4, 5) -> Manhattan: 7, Euclidean: 5.00
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00", out)

        # 2. Custom Points via --p1 and --p2
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--p1", "0", "0", "--p2", "3", "4"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00", out)

        # 3. Optional Challenge: Chebyshev Distance
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--chebyshev"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00, Chebyshev: 4", out)

        # 4. Multi-Pair Expression
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--pairs", "(1,1),(4,5); (0,0),(3,4)"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Pair 1", out)
        self.assertIn("Pair 2", out)

        # 5. Benchmark Table Mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--benchmark"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("BENCHMARK COMPARISON TABLE", out)
        self.assertIn("Ratio (M/E)", out)

        # 6. ASCII Grid Visualization
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--ascii"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("2D ASCII TRAJECTORY VISUALIZATION", out)
        self.assertIn("Legend: S=Start", out)

        # 7. Metric Space Axiom Verification
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--axioms"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("METRIC SPACE AXIOM VERIFICATION", out)
        self.assertIn("Triangle Ineq", out)

        # 8. Movement Topology Admissibility Analysis
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--topology", "orthogonal"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("HEURISTIC ADMISSIBILITY & DOMINANCE ANALYSIS", out)
        self.assertIn("Optimal & Dominant", out)

        # 9. Viva Voce Reference Guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_distance_calculators.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                distance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("AI LAB VIVA VOCE STUDY GUIDE", out)
        self.assertIn("Overestimation Vs Underestimation", out)

        # 10. CLI Dispatcher: python -m src distance-calculators
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["distance-calculators"])
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00", out)

        # 11. CLI Dispatcher Alias: python -m src distances
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["distances"])
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00", out)

        # 12. CLI Dispatcher Alias: python -m src q24
        stdout_trap = io.StringIO()
        with patch("sys.stdout", stdout_trap):
            cli_main(["q24"])
        out = stdout_trap.getvalue()
        self.assertIn("Manhattan: 7, Euclidean: 5.00", out)

    def test_run_eight_puzzle_heuristics_script_modes(self) -> None:
        """Verify Question 25 8-Puzzle Heuristic Evaluator CLI and all script modes."""
        # 1. Default canonical sample execution
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 25: 8-PUZZLE HEURISTIC EVALUATOR", out)
        self.assertIn("h1 (Misplaced): 8, h2 (Manhattan): 18", out)
        self.assertIn("SOLVABLE", out)

        # 2. Exact sample output format
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--format", "sample"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "h1 (Misplaced): 8, h2 (Manhattan): 18")

        # 3. Custom matrices via --current and --goal
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run_eight_puzzle_heuristics.py",
                "--current",
                "[[1,0,2],[3,4,5],[6,7,8]]",
                "--goal",
                "[[0,1,2],[3,4,5],[6,7,8]]",
                "--format",
                "sample",
            ],
        ):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "h1 (Misplaced): 1, h2 (Manhattan): 1")

        # 4. Table format
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--format", "table"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("TOTAL (Blank '0' Excluded)", out)

        # 5. JSON format
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--format", "json"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn('"h1_misplaced": 8', out)
        self.assertIn('"h2_manhattan": 18', out)

        # 6. Optional Challenge: random solvable board generation
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_eight_puzzle_heuristics.py", "--random", "--seed", "42", "--count", "2"],
        ):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: RANDOM SOLVABLE 8-PUZZLE BOARD", out)
        self.assertIn("Random Solvable Board #1", out)
        self.assertIn("Random Solvable Board #2", out)
        self.assertIn("SOLVABLE", out)

        # 7. Dominance study
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_eight_puzzle_heuristics.py", "--dominance-study", "--samples", "20", "--seed", "7"],
        ):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("EMPIRICAL DOMINANCE ANALYSIS", out)
        self.assertIn("h2(n) >= h1(n) on 100% States? : True", out)

        # 8. Blank space inadmissibility demo
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--blank-demo"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("WHY THE BLANK SPACE (0) MUST NOT BE COUNTED", out)
        self.assertIn("Admissible: False", out)

        # 9. Full Demo mode
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--demo"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("FULL LABORATORY DEMO", out)
        self.assertIn("BENCHMARK STATE COMPARISON", out)

        # 10. Viva Voce study guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_heuristics.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                eight_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("VIVA VOCE STUDY GUIDE", out)
        self.assertIn("Dominance Of Heuristics", out)
        self.assertIn("Why Blank Space Is Not Counted", out)

        # 11. Top-level CLI dispatcher & aliases
        for cmd in ("eight-puzzle", "8-puzzle", "q25"):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--format", "sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), "h1 (Misplaced): 8, h2 (Manhattan): 18")

    def test_greedy_best_first_search_script_execution(self) -> None:
        """Verify Question 26 runner across all flags, benchmarks, and CLI dispatcher."""
        # 1. Default run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Greedy Best-First Search (GBFS) Execution Summary", out)
        self.assertIn("Path chosen based on minimum h(n): A -> B -> C", out)

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "Path chosen based on minimum h(n): A -> B -> C")

        # 3. Custom graph & heuristics via CLI
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run_greedy_best_first_search.py",
                "--graph",
                "X: Y; Y: Z; Z:",
                "--heuristics",
                "X: 10, Y: 4, Z: 0",
                "--start",
                "X",
                "--goal",
                "Z",
            ],
        ):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Path chosen based on minimum h(n): X -> Y -> Z", out)

        # 4. Optional Challenge suboptimality run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE", out)
        self.assertIn("YES (Optimal Failure Demonstrated)", out)
        self.assertIn("55.00x", out)

        # 5. AIMA Romania Map benchmark
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--romania"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Russell & Norvig Romania Map", out)
        self.assertIn("Arad -> Sibiu -> Fagaras -> Bucharest", out)
        self.assertIn("450", out)
        self.assertIn("418", out)

        # 6. Local Minima demo
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--local-minima"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Getting Stuck in Local Minima", out)
        self.assertIn("Dead_End", out)

        # 7. Viva Voce reference
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Viva Voce Guide", out)
        self.assertIn("Is Greedy Best-First Search optimal?", out)

        # 8. Step-by-step table trace
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_greedy_best_first_search.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("EXPAND", out)
        self.assertIn("GOAL_REACHED", out)

        # 9. Top-level CLI dispatcher & aliases
        for cmd in ("gbfs", "greedy", "greedy-best-first-search", "q26"):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), "Path chosen based on minimum h(n): A -> B -> C")

    def test_gbfs_grid_navigator_script_execution(self) -> None:
        """Verify Question 27 runner across all flags, benchmarks, and CLI dispatcher."""
        # 1. Default run
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Greedy Best-First Search (GBFS) Grid Navigator", out)
        self.assertIn("Expanded: 25 nodes, Path Length: 22 (Suboptimal)", out)

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "Expanded: 25 nodes, Path Length: 22 (Suboptimal)")

        # 3. Optional Challenge: Render grid with path marked by '*'
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--grid"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OPTIONAL CHALLENGE: 10x10 Maze Path Visualization ('*')", out)
        self.assertIn("*", out)

        # 4. Comparative analysis: GBFS vs BFS
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--compare"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Greedy Best-First Search vs. Breadth-First Search", out)
        self.assertIn("Fewer Nodes Explored?       : YES", out)
        self.assertIn("Suboptimal Path Chosen?     : YES", out)

        # 5. Viva Voce guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION 27 VIVA VOCE GUIDE", out)
        self.assertIn("Greedy Best-First Search vs A*", out)

        # 6. Step trace
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--trace"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Greedy Best-First Search: Step-by-Step Priority Queue Trace", out)

        # 7. Benchmarks
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_gbfs_grid_navigator.py", "--benchmarks"]):
            with patch("sys.stdout", stdout_trap):
                gbfs_grid_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("GREEDY BEST-FIRST SEARCH: MULTI-BENCHMARK EVALUATION", out)

        # 8. Top-level CLI dispatcher & aliases
        for cmd in ("gbfs-grid", "gbfs-grid-nav", "greedy-grid", "gbfs-grid-navigator", "q27"):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), "Expanded: 25 nodes, Path Length: 22 (Suboptimal)")

    def test_run_a_star_search(self) -> None:
        """Test Question 28: A* Search Algorithm CLI and dispatcher integration."""
        # 1. Default execution
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "Optimal Path: S -> C -> G, Total Cost: 9")

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), "Optimal Path: S -> C -> G, Total Cost: 9")

        # 3. Optional Challenge: Print frontier contents at each step
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--challenge"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("=== A* Frontier (Priority Queue) Contents at Each Step ===", out)
        self.assertIn("Optimal Path: S -> C -> G, Total Cost: 9", out)

        # 4. Detailed ASCII step table
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* SEARCH: STEP-BY-STEP EXECUTION TRACE TABLE", out)

        # 5. Romania benchmark
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--romania"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest", out)
        self.assertIn("Total Cost: 418", out)

        # 6. Reopen demo
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--reopen-demo"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("DEMO: Reopening Closed Nodes", out)
        self.assertIn("Closed Nodes Reopened: 1", out)

        # 7. Comparison with UCS and Greedy Best-First
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--compare"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("ALGORITHM COMPARISON: A* Search vs. Uniform Cost Search", out)

        # 8. Heuristic verification
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--check-heuristics"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("HEURISTIC VERIFICATION REPORT", out)
        self.assertIn("Admissibility  : YES", out)

        # 9. Viva Voce study guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_search.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* Search Algorithm University Viva Voce Reference Guide", out)

        # 10. Top-level CLI dispatcher & aliases
        for cmd in ("a-star", "astar", "a-star-search", "astar-search", "q28"):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), "Optimal Path: S -> C -> G, Total Cost: 9")

    def test_run_a_star_maze_solver(self) -> None:
        """Test Question 29: A* Pathfinding in 2D Maze CLI and dispatcher integration."""
        # 1. Default execution (formatted overview and ASCII visualization)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* PATHFINDING IN 2D MAZE: ASCII VISUALIZATION", out)
        self.assertIn("Optimal Steps   : 8 transitions", out)
        self.assertIn("Nodes Expanded  : 16 nodes", out)
        self.assertIn("* # X . .", out)

        # 2. Exact syllabus sample output (pure ASCII grid)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        expected_sample = (
            "* # X . .\n"
            "* # X . .\n"
            "* # X . .\n"
            "* # # # #\n"
            "* * * * *"
        )
        self.assertEqual(out.strip(), expected_sample)

        # 3. Optional Challenge: Live terminal animation
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--animate", "--delay", "0"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* 2D MAZE EXPANSION ANIMATION", out)
        self.assertIn("GOAL REACHED! Optimal Path Length: 8 steps", out)

        # 4. Comparative analysis: A* vs BFS vs Greedy
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--compare"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("SEARCH ALGORITHM COMPARISON: A* vs. BFS vs. GREEDY BEST-FIRST", out)
        self.assertIn("Nodes Explored Reduction vs BFS", out)

        # 5. Step-by-step priority queue trace table
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* PATHFINDING: STEP-BY-STEP PRIORITY QUEUE EXECUTION TRACE TABLE", out)
        self.assertIn("GOAL_REACHED", out)

        # 6. Benchmark suite
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--benchmarks"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* 2D MAZE PATHFINDING: MULTI-BENCHMARK SUITE", out)
        self.assertIn("Benchmark 10x10 Maze", out)

        # 7. Viva Voce guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_maze_solver.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 29: A* PATHFINDING IN 2D MAZE - VIVA VOCE REFERENCE GUIDE", out)
        self.assertIn("A* IN VIDEO GAMES", out)
        self.assertIn("EXPLORED SET SIZE RELATIVE TO BFS", out)

        # 8. Custom grid input via CLI
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_a_star_maze_solver.py", "--grid", ". . X\n. . .\n. X .", "--start", "0,0", "--goal", "2,2"],
        ):
            with patch("sys.stdout", stdout_trap):
                astar_maze_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Maze Dimensions : 3 rows x 3 columns", out)
        self.assertIn("Start / Goal    : (0, 0) -> (2, 2)", out)

        # 9. Top-level CLI dispatcher & aliases
        for cmd in ("a-star-maze", "astar-maze", "a-star-maze-solver", "astar-2d-maze", "q29"):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_run_a_star_eight_puzzle(self) -> None:
        """Test Question 30: 8-Puzzle Solver using A* (Manhattan) CLI and dispatcher."""
        expected_sample = "Solved in 2 moves: [[1, 2, 3, 4, 0, 5, 6, 7, 8], [1, 2, 3, 4, 5, 0, 6, 7, 8], [1, 2, 3, 4, 5, 8, 6, 7, 0]]"

        # 1. Default execution (formatted overview and ASCII step-by-step boards)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("8-PUZZLE SOLVER (A* WITH MANHATTAN HEURISTIC)", out)
        self.assertIn("Optimal Move Count: 2 moves", out)
        self.assertIn("Step 0 (Initial State):", out)
        self.assertIn("Step 1 (Move 5 LEFT | Blank RIGHT)", out)
        self.assertIn("Step 2 (Move 8 UP | Blank DOWN)", out)

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 3. Optional Challenge: Solvability check
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--check-solvability"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("8-PUZZLE MATHEMATICAL SOLVABILITY VERIFICATION REPORT", out)
        self.assertIn("Initial State Inversions: 0", out)
        self.assertIn("Solvability Status      : SOLVABLE", out)

        # 4. Comparative benchmarking
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--compare"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("8-PUZZLE HEURISTIC & SEARCH ALGORITHM COMPARISON REPORT", out)
        self.assertIn("A* (Manhattan Distance)", out)
        self.assertIn("A* (Misplaced Tiles)", out)

        # 5. Step-by-step trace table
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("8-PUZZLE OPTIMAL SOLUTION STEP-BY-STEP TRACE TABLE", out)
        self.assertIn("Move 5 LEFT", out)
        self.assertIn("Move 8 UP", out)

        # 6. Viva Voce reference guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 30: 8-PUZZLE SOLVER USING A* (MANHATTAN)", out)
        self.assertIn("INVERSIONS TO CHECK SOLVABILITY", out)
        self.assertIn("STATE HASHING IMPORTANCE", out)

        # 7. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_eight_puzzle.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertTrue(data["success"])
        self.assertEqual(data["move_count"], 2)

        # 8. Custom initial state via CLI
        stdout_trap = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["run_a_star_eight_puzzle.py", "--initial", "[1, 2, 3, 4, 0, 6, 7, 5, 8]", "--table"],
        ):
            with patch("sys.stdout", stdout_trap):
                astar_puzzle_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Total Moves: 2", out)

        # 9. Top-level CLI dispatcher & aliases
        for cmd in (
            "eight-puzzle-solver",
            "8-puzzle-solver",
            "astar-puzzle",
            "astar-8-puzzle",
            "eight-puzzle-astar",
            "q30",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_run_eight_puzzle_manhattan_vs_misplaced(self) -> None:
        """Test Question 31: 8-Puzzle Manhattan vs Misplaced Tiles Dominance Analysis CLI and dispatcher."""
        expected_sample = "Path length: 18. h1 nodes expanded: 4500. h2 nodes expanded: 1200."
        expected_computed = "Path length: 20. h1 nodes expanded: 2341. h2 nodes expanded: 92."

        # 1. Default execution
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("QUESTION NO. 31: 8-PUZZLE MANHATTAN VS MISPLACED TILES", out)
        self.assertIn("8-PUZZLE COMPARISON: h1 (MISPLACED TILES) vs h2 (MANHATTAN DISTANCE)", out)
        self.assertIn("Optimal Path Length (d)", out)
        self.assertIn("Effective Branching Factor (b*)", out)
        self.assertIn("[PROVED] Manhattan Distance (h2) dominates Misplaced Tiles (h1)", out)

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 3. Dynamic computed sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--sample-computed"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_computed)

        # 4. Solvability check
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--check-solvability"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Initial State Inversions : 16", out)
        self.assertIn("Solvability Status       : SOLVABLE", out)

        # 5. Table output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Metric", out)
        self.assertIn("h1 (Misplaced)", out)
        self.assertIn("h2 (Manhattan)", out)

        # 6. Dominance proof
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--proof"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("THEORETICAL PROOF OF HEURISTIC DOMINANCE", out)
        self.assertIn("AIMA Theorem 3.6", out)

        # 7. Viva guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Concept of Heuristic Dominance", out)
        self.assertIn("Effective Branching Factor (b*)", out)

        # 8. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertEqual(data["path_length"], 20)
        self.assertTrue(data["is_dominated"])
        self.assertEqual(data["h1_misplaced"]["nodes_expanded"], 2341)
        self.assertEqual(data["h2_manhattan"]["nodes_expanded"], 92)

        # 9. Benchmark preset (18-move)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_eight_puzzle_manhattan_vs_misplaced.py", "--preset", "sample-18move", "--table"]):
            with patch("sys.stdout", stdout_trap):
                puzzle_dominance_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Optimal Path Length (d)", out)
        self.assertIn("18", out)

        # 10. Top-level CLI dispatcher & aliases
        for cmd in (
            "eight-puzzle-dominance",
            "eight-puzzle-comparison",
            "manhattan-vs-misplaced",
            "puzzle-dominance",
            "q31",
            "question31",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_run_ucs_vs_astar_comparison(self) -> None:
        """Test Question 32: UCS vs A* Performance Comparison CLI and dispatcher."""
        expected_sample = "UCS expanded: 380 nodes. A* expanded: 45 nodes."

        # 1. Default execution
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UCS vs A* (MANHATTAN) PERFORMANCE COMPARISON", out)
        self.assertIn("380", out)
        self.assertIn("45", out)

        # 2. Exact syllabus sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 3. Dynamic computed sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--sample-computed"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 4. Table output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("States Evaluated (Expanded)", out)
        self.assertIn("380", out)
        self.assertIn("45", out)

        # 5. ASCII maps
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--ascii"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UCS Map", out)
        self.assertIn("A* Map", out)

        # 6. Theory output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--theory"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("THEORETICAL ANALYSIS", out)
        self.assertIn("Heuristic Pull & Directionality", out)

        # 7. Viva guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIVERSITY VIVA VOCE PREPARATION GUIDE", out)
        self.assertIn("special case of A*", out)

        # 8. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_ucs_vs_astar_comparison.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                ucs_vs_astar_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertEqual(data["ucs"]["nodes_expanded"], 380)
        self.assertEqual(data["astar"]["nodes_expanded"], 45)
        self.assertEqual(data["comparison"]["nodes_saved"], 335)

        # 9. Top-level CLI dispatcher & aliases
        for cmd in (
            "ucs-vs-astar",
            "ucs-astar",
            "ucs-astar-comparison",
            "ucs-vs-astar-comparison",
            "q32",
            "question32",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_run_admissible_heuristic_validator_script(self) -> None:
        """Verify Question 33 script execution across flags, JSON, and CLI aliases."""
        expected_sample = "False. Heuristic overestimates at node A (10 > 8)."

        # 1. Exact sample constant output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 2. Live computed sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--sample-computed"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 3. Table output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Comparative Heuristic vs. True Cost Table", out)
        self.assertIn("❌ Overestimates", out)
        self.assertIn("✅ Admissible", out)

        # 4. Repair output (Optional Challenge)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--repair"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("Automatically Lowered Heuristic Values", out)
        self.assertIn("Clamped from 10 -> 8", out)

        # 5. Theory output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--theory"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("THEORETICAL FOUNDATIONS", out)
        self.assertIn("Mathematical Definition of Admissibility", out)

        # 6. Viva guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIVERSITY VIVA VOCE PREPARATION GUIDE", out)
        self.assertIn("formal definition of an Admissible Heuristic", out)

        # 7. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_admissible_heuristic_validator.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                admissible_validator_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertFalse(data["is_admissible"])
        self.assertEqual(data["violations_count"], 1)
        self.assertEqual(data["violations"][0]["node"], "A")
        self.assertEqual(data["true_costs"]["A"], 8.0)
        self.assertEqual(data["repaired_heuristics"]["A"], 8.0)

        # 8. Top-level CLI dispatcher & aliases
        for cmd in (
            "admissible-heuristic",
            "admissible-validator",
            "heuristic-validator",
            "admissibility",
            "q33",
            "question33",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_run_consistent_heuristic_validator_script(self) -> None:
        """Verify Question 34 script execution across flags, JSON, and CLI aliases."""
        expected_sample = "False. Violation at edge A->B: 5 > 2 + 2."

        # 1. Exact sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 2. Consistent sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--sample-consistent"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("True", out)

        # 3. Table output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("CONSISTENCY VERIFICATION TABLE", out)
        self.assertIn("❌ Violation", out)
        self.assertIn("✅ Valid", out)

        # 4. Extended output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--extended"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("EXTENDED MULTI-NODE GRAPH VALIDATION DEMONSTRATION", out)

        # 5. Programmatic proof (Optional Challenge)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--prove"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("MATHEMATICAL INDUCTION PROOF", out)
        self.assertIn("Telescoping Sum", out)

        # 6. Theory output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--theory"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("THEORETICAL FOUNDATIONS", out)
        self.assertIn("TRIANGLE INEQUALITY", out)

        # 7. Viva guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIVERSITY VIVA VOCE EXAMINATION GUIDE", out)
        self.assertIn("triangle inequality", out)

        # 8. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_consistent_heuristic_validator.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                consistent_validator_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertFalse(data["is_consistent"])
        self.assertEqual(data["violations_count"], 1)
        self.assertEqual(data["primary_violation"]["u"], "A")
        self.assertEqual(data["primary_violation"]["v"], "B")

        # 9. Top-level CLI dispatcher & aliases
        for cmd in (
            "consistent-heuristic",
            "consistent-validator",
            "consistency-validator",
            "consistency",
            "q34",
            "question34",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)

    def test_a_star_octile_distance_script_execution(self) -> None:
        """Verify run_a_star_octile_distance script executes cleanly with all CLI flags."""
        expected_sample = "Cost: 2.828 (Two diagonal moves)"

        # 1. Exact sample output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--sample"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertEqual(out.strip(), expected_sample)

        # 2. Default run (Overview + ASCII visualization)
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("A* WITH OCTILE DISTANCE", out)
        self.assertIn("Total Cost            : 2.828", out)
        self.assertIn("ASCII Maze Visualization:", out)

        # 3. Multi-heuristic comparative table
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--compare"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("MULTI-HEURISTIC COMPARATIVE ANALYSIS", out)
        self.assertIn("Octile (Canonical)", out)
        self.assertIn("Manhattan (INADMISSIBLE)", out)

        # 4. Corner pinch constraint demonstration
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--corner-demo"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("CORNER WALL PINCH CONSTRAINT DEMONSTRATION", out)
        self.assertIn("Cannot move diagonally through two corner walls", out)
        self.assertIn("Path Found: False", out)
        self.assertIn("Path Found: True", out)

        # 5. Step-by-step priority queue trace table
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--table"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("STEP-BY-STEP PRIORITY QUEUE EXPANSION TRACE", out)
        self.assertIn("g(n)", out)
        self.assertIn("f(n)", out)

        # 6. Canonical benchmark suite
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--benchmarks"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("CANONICAL BENCHMARK SUITE", out)
        self.assertIn("Syllabus Sample Input", out)
        self.assertIn("Canonical 5x5 Maze", out)

        # 7. Theory output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--theory"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("OCTILE DISTANCE", out)
        self.assertIn("WHY MANHATTAN DISTANCE IS INADMISSIBLE", out)

        # 8. Viva guide
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--viva"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        self.assertIn("UNIVERSITY VIVA VOCE EXAMINATION GUIDE", out)
        self.assertIn("Manhattan", out)

        # 9. JSON output
        stdout_trap = io.StringIO()
        with patch.object(sys, "argv", ["run_a_star_octile_distance.py", "--json"]):
            with patch("sys.stdout", stdout_trap):
                astar_octile_script_main()
        out = stdout_trap.getvalue()
        data = json.loads(out)
        self.assertTrue(data["found"])
        self.assertEqual(data["total_cost"], 2.828)
        self.assertEqual(data["diagonal_moves_count"], 2)

        # 10. Top-level CLI dispatcher & aliases
        for cmd in (
            "astar-octile",
            "octile-astar",
            "a-star-octile",
            "octile-distance",
            "octile",
            "q35",
            "question35",
        ):
            stdout_trap = io.StringIO()
            with patch("sys.stdout", stdout_trap):
                cli_main([cmd, "--sample"])
            out = stdout_trap.getvalue()
            self.assertEqual(out.strip(), expected_sample)


if __name__ == "__main__":
    unittest.main()
