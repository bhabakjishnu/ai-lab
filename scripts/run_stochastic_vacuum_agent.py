"""Runner script for Stochastic Environment Vacuum Agent Simulation (Question 7).

Demonstrates an intelligent vacuum agent operating in a stochastic environment
where executing 'Move Right' has an 80% chance of success and a 20% chance of
slipping and staying in place due to slippery floors.
Executes Monte Carlo batches (default: 20 runs), records step counts, measures
the empirical performance penalty compared to a deterministic baseline, supports
an optional slip-back challenge, and provides complete Viva Voce study notes.
"""

import argparse
from pathlib import Path
import random
import statistics
import sys
from typing import List, Optional

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.stochastic_vacuum_agent import (
    ACTION_CLEAN,
    ACTION_LEFT,
    ACTION_RIGHT,
    OUTCOME_CLEANED,
    OUTCOME_SLIP_BACK,
    OUTCOME_SLIP_STAY,
    OUTCOME_SUCCESS,
    ROOM_CLEAN,
    ROOM_DIRTY,
    MonteCarloExperimentResult,
    SingleRunResult,
    StochasticStepRecord,
    calculate_theoretical_expected_steps,
    run_deterministic_simulation,
    run_monte_carlo_stochastic_simulation,
    run_single_stochastic_simulation,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce theory guide for Question 7."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
             Question 7: Stochastic Environment Vacuum Agent
================================================================================

1. DETERMINISTIC VS. STOCHASTIC ENVIRONMENTS & ACTUATORS:
   -----------------------------------------------------
   * Deterministic Environment / Actuator:
     - The next state of the environment is completely and uniquely determined
       by the current state and the action executed by the agent:
           s' = f(s, a)
     - Certainty: Given state s and action a, P(s' | s, a) = 1.0 for exactly one s'.
     - Example: If the agent is in Room 0 and commands 'Right', it always arrives
       at Room 1 without failure.
   * Stochastic Environment / Actuator:
     - The outcome of an action is non-deterministic, governed by a probability
       distribution over potential successor states:
           P(s' | s, a)
     - Uncertainty: Real-world physical actuators suffer from mechanical slip,
       wheel friction variations, voltage dips, and environmental turbulence.
     - Slippery Floor Scenario:
         P(Room 1 | Room 0, 'Right') = 0.8  (Actuator successfully advances)
         P(Room 0 | Room 0, 'Right') = 0.2  (Actuator slips; remains in place)
     - Key Insight: The agent cannot assume an action succeeded simply because
       it issued the command. It must rely on sensors or probabilistic models.

2. EXPECTED PERFORMANCE & MONTE CARLO SIMULATION:
   ----------------------------------------------
   * Why Single Runs are Insufficient:
     - Because individual trials depend on stochastic outcomes (random sampling),
       a single trial might take 5 steps (pure luck) or 10 steps (unlucky slips).
     - To rigorously evaluate the agent, we employ Monte Carlo simulation:
       running M independent trials (e.g., M = 20) and calculating statistical
       averages (mean, variance, standard deviation).
   * Mathematical Expectation via Geometric Distribution:
     - A 3-room dirty grid requires 3 cleaning actions and 2 room transitions.
     - Cleaning each room is deterministic: 1 step each -> 3 steps total.
     - Each transition (Room 0 -> 1 and Room 1 -> 2) represents a Bernoulli
       trial with success probability p = 0.8.
     - The number of attempts until success follows a Geometric Distribution:
           E[Attempts per move] = 1 / p = 1 / 0.8 = 1.25 attempts.
     - Expected total steps:
           E[Total Steps] = 3 (Clean) + 2 * (1 / 0.8) = 3 + 2.5 = 5.5 steps.
   * Performance Penalty:
     - Deterministic baseline = 5 steps (Clean, Right, Clean, Right, Clean).
     - Stochastic expected = 5.5 steps.
     - Absolute Penalty = 5.5 - 5.0 = 0.5 steps (+10.0% time delay).

3. ACTUATORS IN ARTIFICIAL INTELLIGENCE (PEAS FRAMEWORK):
   ------------------------------------------------------
   * Actuators Defined:
     - Actuators are the physical or software mechanisms through which an agent
       asserts changes and exercises control over its environment.
     - Examples: Robotic vacuum wheels, suction motors, robotic gripper arms,
       steering servos, or software API dispatchers.
   * Actuator Failure Modes in AI:
     - Slippage / Incomplete Action: Wheels spin but robot does not move.
     - Drift / Noise: Steering commands deviate by random angular offsets.
     - Jamming: Motor stalls or suction intake becomes obstructed.
     - Slip-Back (Optional Challenge): Dynamic rebound or inclined slope causes
       reverse movement when forward torque is applied.
   * How Intelligent Agents Overcome Stochasticity:
     - Closed-loop feedback (continuous sensing after every action).
     - Markov Decision Processes (MDPs) & Policy Iteration / Q-Learning.
     - Probabilistic state estimation (Bayesian filtering, Kalman filters).

4. MARKOV PROPERTY & NON-DETERMINISTIC TRANSITIONS:
   -------------------------------------------------
   * The environment possesses the Markov Property if the future state depends
     only upon the current state and action, independent of past history:
         P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
   * This problem models a discrete Markov Decision Process (MDP) with
     stationary transition probabilities P(s' | s, a).

================================================================================
"""
    print(notes)


def parse_grid_argument(grid_str: str) -> List[int]:
    """Parse comma or space separated grid representation.

    Args:
        grid_str: String such as '1,1,1' or '1 0 1'.

    Returns:
        List of integers (0 or 1).

    Raises:
        ValueError: If tokens are invalid.
    """
    cleaned = grid_str.replace(",", " ").strip()
    tokens = cleaned.split()
    if not tokens:
        raise ValueError("Grid string cannot be empty.")
    result: List[int] = []
    for tok in tokens:
        if tok not in ("0", "1"):
            raise ValueError(f"Invalid room token '{tok}'; must be 0 (Clean) or 1 (Dirty)")
        result.append(int(tok))
    return result


def display_run_trace(result: SingleRunResult) -> None:
    """Print step-by-step trace of a single run.

    Args:
        result: SingleRunResult object.
    """
    print(f"\n--- Detailed Step Trace for Run #{result.run_id} ---")
    print(f"Initial Grid : {result.initial_grid}")
    for record in result.history:
        print(record.format_trace())
    print(f"Result: All Clean={result.is_all_clean} in {result.steps_taken} steps.\n")


def main() -> None:
    """Entry point dispatcher for stochastic vacuum agent runner."""
    parser = argparse.ArgumentParser(
        description="Question 7: Stochastic Environment Vacuum Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of Monte Carlo simulation runs (default: 20)",
    )
    parser.add_argument(
        "--rooms",
        type=int,
        default=3,
        help="Number of rooms in the linear grid (default: 3)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Optional initial grid state, e.g. '1,1,1' or '1,0,1' (default: all dirty)",
    )
    parser.add_argument(
        "--success-prob",
        type=float,
        default=0.8,
        help="Probability of 'Move Right' succeeding (default: 0.8)",
    )
    parser.add_argument(
        "--slip-stay-prob",
        type=float,
        default=0.2,
        help="Probability of 'Move Right' slipping and staying in place (default: 0.2)",
    )
    parser.add_argument(
        "--slip-back-prob",
        type=float,
        default=0.0,
        help="Probability of 'Move Right' slipping backward (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility (Seed 28 produces 5.4 steps sample output)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print detailed step trace of the first simulation run",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run optional challenge demonstrating slip-back probability (e.g. 10%% slip back)",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory notes and study guide",
    )

    args = parser.parse_args()

    if args.viva:
        display_viva_notes()
        return

    # Parse custom grid if provided
    initial_grid: Optional[List[int]] = None
    if args.grid is not None:
        initial_grid = parse_grid_argument(args.grid)
        num_rooms = len(initial_grid)
    else:
        num_rooms = args.rooms

    # Optional challenge override
    success_prob = args.success_prob
    slip_stay_prob = args.slip_stay_prob
    slip_back_prob = args.slip_back_prob

    if args.challenge:
        # Standard configuration for the slip back challenge:
        # 80% success, 10% stay, 10% slip back
        success_prob = 0.8
        slip_stay_prob = 0.1
        slip_back_prob = 0.1

    print("\n" + "=" * 74)
    print("      QUESTION NO. 7: STOCHASTIC ENVIRONMENT VACUUM AGENT")
    print("=" * 74)
    print(f"Environment Mode        : Stochastic 1D Grid")
    print(f"Number of Rooms         : {num_rooms} rooms")
    print(f"Initial State           : {initial_grid if initial_grid is not None else [1] * num_rooms}")
    print(f"Actuator Transitions   : 'Move Right' -> {success_prob * 100:.0f}% Success, "
          f"{slip_stay_prob * 100:.0f}% Slip-Stay" +
          (f", {slip_back_prob * 100:.0f}% Slip-Back" if slip_back_prob > 0 else ""))
    print(f"Monte Carlo Runs        : {args.runs} runs")
    print(f"Random Seed             : {args.seed if args.seed is not None else 'Dynamic (Entropy)'}")
    print("=" * 74)

    # 1. Deterministic baseline
    det_res = run_deterministic_simulation(num_rooms=num_rooms, initial_grid=initial_grid)
    print(f"\n[Baseline] Deterministic Environment Execution:")
    print(f"  Actions taken : {[rec.action for rec in det_res.history]}")
    print(f"  Total Steps   : {det_res.steps_taken} steps")

    # 2. Monte Carlo batch
    mc_result = run_monte_carlo_stochastic_simulation(
        num_runs=args.runs,
        num_rooms=num_rooms,
        initial_grid=initial_grid,
        move_success_prob=success_prob,
        slip_stay_prob=slip_stay_prob,
        slip_back_prob=slip_back_prob,
        seed=args.seed,
    )

    # If trace requested or run 1 trace
    if args.trace and mc_result.runs:
        display_run_trace(mc_result.runs[0])

    # 3. Print per-run step counts
    print("\n[Trial Log] Individual Monte Carlo Run Results:")
    print("-" * 74)
    print(" Run ID | Steps Taken | Clean Actions | Moves Attempted | Slips (Stay/Back)")
    print("-" * 74)
    for run in mc_result.runs:
        slips_str = f"{run.slips_stay_count} stay"
        if run.slips_back_count > 0:
            slips_str += f", {run.slips_back_count} back"
        print(
            f" Run {run.run_id:2d} |    {run.steps_taken:2d} steps  |   {run.cleaning_steps:2d} cleans   |"
            f"    {run.movement_attempts:2d} moves     | {slips_str}"
        )
    print("-" * 74)

    # 4. Standard Expected Output display
    print(f"\nAverage steps to clean {num_rooms} rooms: {mc_result.mean_steps:.1f} steps")

    # 5. Full Summary Report
    print("\n" + mc_result.format_summary_table())

    if args.challenge:
        print("\n>>> OPTIONAL CHALLENGE ACTIVATED: Actuator 'Slip Back' Mechanics <<<")
        print("When moving Right, agent had 10% chance to slip backward to previous room.")
        print(f"Theoretical Expected with Slip Back: {mc_result.theoretical_expected_steps:.3f} steps")
        print(f"Observed Monte Carlo Average Steps : {mc_result.mean_steps:.2f} steps\n")


if __name__ == "__main__":
    main()
