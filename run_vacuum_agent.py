"""Runner script for Simple Reflex Vacuum Cleaner Agent Simulation.

Simulates agent-environment interaction over discrete time steps with
detailed traces, support for custom initial states, and optional
extensions (3x3 grid and viva voce reference study notes).
"""

import argparse
import sys
from typing import Dict

from src.vacuum_agent import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SUCK,
    CLEAN,
    DIRTY,
    LOCATION_A,
    LOCATION_B,
    GridVacuumEnvironment,
    SimpleReflexVacuumAgent,
    TwoSquareEnvironment,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce preparation notes for Question 1."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
================================================================================

1. DEFINITION OF A RATIONAL AGENT:
   ---------------------------------
   * An agent is anything that perceives its environment through sensors and acts
     upon that environment through actuators (Russell & Norvig, AIMA).
   * A Rational Agent is an agent that acts so as to maximize its expected
     performance measure, given its percept sequence to date and its built-in
     prior knowledge.
   * Rationality is NOT omniscience:
     - Omniscience knows the actual outcome of its actions and acts with 100%
       clairvoyance (impossible in practice).
     - Rationality maximizes EXPECTED performance based on available information.
   * The PEAS Framework specifies:
     - P: Performance Measure (e.g., +10 for clean square, -1 for step/energy)
     - E: Environment (e.g., 2 squares A & B, dirt presence)
     - A: Actuators (e.g., Suck, Move Right, Move Left, NoOp)
     - S: Sensors (e.g., location sensor, dirt optical sensor)

2. FULLY OBSERVABLE VS. PARTIALLY OBSERVABLE ENVIRONMENTS:
   --------------------------------------------------------
   * Fully Observable:
     - The agent's sensors give it access to the complete state of the
       environment at each point in time.
     - Example: If the agent could perceive both square A and square B simultaneously.
   * Partially Observable:
     - The agent's sensors only provide local or noisy observations.
     - In this lab: The agent is situated in a partially observable setting;
       at any time step, it only senses its CURRENT square (its own location and
       whether that location is Clean or Dirty), with no visibility into the
       status of the other square without navigating there.

3. LIMITATIONS OF SIMPLE REFLEX AGENTS:
   --------------------------------------
   * No Internal State / Memory:
     - Simple reflex agents act on Condition-Action rules (if-then) based strictly
       on the current percept: f(Percept) -> Action.
     - They cannot maintain history of visited squares or track past events.
   * Infinite Loops:
     - If the environment is partially observable and deterministic, simple reflex
       agents can easily get trapped in infinite loops. For instance, when both
       squares are clean, the agent continuously bounces: A -> Right -> B -> Left -> A.
   * Inability to Handle Unseen Environmental Changes:
     - If dirt reappears in an unobserved square, the reflex agent has no memory or
       predictive model to know it needs cleaning unless it stumbles upon it.
   * Remedy:
     - Model-Based Reflex Agents (maintain an internal state/belief of unobserved world)
     - Goal-Based or Utility-Based Agents (evaluate sequences toward optimal goals).

================================================================================
"""
    print(notes)


def run_two_square_simulation(
    initial_location: str,
    status_a: str,
    status_b: str,
    steps: int,
) -> None:
    """Run and display the two-square simple reflex simulation.

    Args:
        initial_location: Starting square ('A' or 'B').
        status_a: Status of square A ('Clean' or 'Dirty').
        status_b: Status of square B ('Clean' or 'Dirty').
        steps: Total discrete time steps.
    """
    initial_state: Dict[str, str] = {LOCATION_A: status_a, LOCATION_B: status_b}

    env = TwoSquareEnvironment(
        initial_location=initial_location,
        state=initial_state,
    )
    agent = SimpleReflexVacuumAgent()

    print("\n" + "=" * 65)
    print("      SIMPLE REFLEX VACUUM CLEANER AGENT SIMULATION")
    print("=" * 65)
    print("Core AI Concept: Simple Reflex Agent (Condition-Action Mapping)")
    print("Memory Constraint: Strictly O(1) space, 0 past percept memory.")
    print("-" * 65)
    print(f"Initial Configuration:")
    print(f"  * Starting Location : Location {env.agent_location}")
    print(f"  * Environment State : Location A is {status_a}, Location B is {status_b}")
    print(f"  * Planned Steps     : {steps}")
    print("-" * 65)
    print("Step-by-Step Simulation Trace:")

    history = env.simulate(steps=steps, agent=agent)
    for record in history:
        print(record.format_trace())

    print("-" * 65)
    print(f"Final Environment State : {env.state}")
    print(f"All Squares Clean?      : {'Yes (Optimal cleanliness achieved)' if env.is_all_clean() else 'No'}")
    print("=" * 65 + "\n")


def run_grid_simulation(steps: int = 10, seed: int = 42) -> None:
    """Run and display the optional 3x3 grid challenge simulation.

    Args:
        steps: Number of time steps.
        seed: Random seed for dirt placement.
    """
    print("\n" + "=" * 65)
    print("   OPTIONAL CHALLENGE: 3x3 GRID VACUUM CLEANER SIMULATION")
    print("=" * 65)
    grid_env = GridVacuumEnvironment(agent_pos=(0, 0), dirt_probability=0.45, seed=seed)

    print("Initial 3x3 Grid State (D = Dirty, . = Clean):")
    for r in range(grid_env.GRID_SIZE):
        row_str = " ".join("D" if grid_env.grid[(r, c)] == DIRTY else "." for c in range(grid_env.GRID_SIZE))
        print(f"  Row {r}: [ {row_str} ]")
    print(f"Agent Starting Position: {grid_env.agent_pos}")
    print("-" * 65)
    print("Simulation Trace:")

    grid_history = grid_env.simulate(steps=steps)
    for step_num, pos, status, action in grid_history:
        print(f"Step {step_num:2d}: Pos {pos} is {status:5s} -> Action: {action}")

    print("-" * 65)
    dirty_count = sum(1 for v in grid_env.grid.values() if v == DIRTY)
    print(f"Remaining Dirty Squares : {dirty_count} / {grid_env.GRID_SIZE * grid_env.GRID_SIZE}")
    print("=" * 65 + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command line flags and options."""
    parser = argparse.ArgumentParser(
        description="Simulate Simple Reflex Vacuum Cleaner Agent in a 2-Square or 3x3 Grid Environment."
    )
    parser.add_argument(
        "--location",
        "-l",
        type=str,
        default=LOCATION_A,
        choices=[LOCATION_A, LOCATION_B],
        help="Initial location of the agent (default: 'A').",
    )
    parser.add_argument(
        "--status-a",
        type=str,
        default=DIRTY,
        choices=[CLEAN, DIRTY],
        help="Initial status of Location A (default: 'Dirty').",
    )
    parser.add_argument(
        "--status-b",
        type=str,
        default=CLEAN,
        choices=[CLEAN, DIRTY],
        help="Initial status of Location B (default: 'Clean').",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=5,
        help="Number of discrete time steps to simulate (default: 5).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt interactively for initial location and square states.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run the optional 3x3 grid simulation challenge.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce questions, answers, and theory.",
    )
    return parser.parse_args()


def main() -> None:
    """Main program entry point."""
    args = parse_arguments()

    if args.viva:
        display_viva_notes()

    if args.grid:
        run_grid_simulation(steps=args.steps)
        return

    initial_location = args.location
    status_a = args.status_a
    status_b = args.status_b
    steps = args.steps

    if args.interactive:
        print("\n--- Interactive Vacuum Agent Setup ---")
        try:
            user_loc = input("Enter starting location (A/B) [default: A]: ").strip().upper()
            if user_loc in (LOCATION_A, LOCATION_B):
                initial_location = user_loc

            user_sa = input("Enter status for Location A (Clean/Dirty) [default: Dirty]: ").strip().capitalize()
            if user_sa in (CLEAN, DIRTY):
                status_a = user_sa

            user_sb = input("Enter status for Location B (Clean/Dirty) [default: Clean]: ").strip().capitalize()
            if user_sb in (CLEAN, DIRTY):
                status_b = user_sb

            user_steps = input("Enter number of simulation steps [default: 5]: ").strip()
            if user_steps.isdigit() and int(user_steps) > 0:
                steps = int(user_steps)
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default configuration.")

    run_two_square_simulation(
        initial_location=initial_location,
        status_a=status_a,
        status_b=status_b,
        steps=steps,
    )


if __name__ == "__main__":
    main()
