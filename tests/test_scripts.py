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


if __name__ == "__main__":
    unittest.main()
