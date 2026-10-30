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


if __name__ == "__main__":
    unittest.main()
