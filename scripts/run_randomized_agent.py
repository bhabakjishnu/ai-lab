"""Runner script for Randomized Reflex Agent Simulation (Question 2).

Demonstrates a Randomized Reflex Agent navigating a 1D grid of rooms (default: 5 rooms).
Simulates agent-environment interaction over discrete time steps, tracking moves,
clean operations, and boundary interactions, with interactive execution,
empirical Monte Carlo benchmarking, and comprehensive viva study notes.
"""

import argparse
import random
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import List, Optional

from src.randomized_agent import (
    ACTION_CLEAN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ROOM_CLEAN,
    ROOM_DIRTY,
    OneDimensionalGridEnvironment,
    RandomizedReflexAgent,
    SimulationResult,
    run_randomized_reflex_simulation,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce theory guide for Question 2."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
                    Question 2: Randomized Reflex Agent
================================================================================

1. DETERMINISTIC VS. STOCHASTIC (RANDOMIZED) BEHAVIOR:
   -----------------------------------------------------
   * Deterministic Agent:
     - Given the exact same percept (or percept sequence), the agent's action
       mapping is always identical: f(Percept) -> Action.
     - Advantage: Predictable, completely reproducible, easy to debug and verify.
     - Disadvantage: Highly vulnerable to getting trapped in infinite loops when
       operating in partially observable or unobservable environments.
   * Stochastic / Randomized Agent:
     - Uses random choices (e.g., via pseudo-random number generators) to pick
       among possible actions when percepts lack sufficient disambiguation.
     - Action choice follows a probability distribution: P(Action | Percept).
     - Advantage: Breaks symmetry, escapes deterministic cycles, guarantees
       eventual exploration of all connected states in finite spaces with
       probability 1 (absorbing Markov chain / random walk).

2. STATE LOOPS IN SIMPLE REFLEX AGENTS:
   -------------------------------------
   * The Partial Observability Problem:
     - A simple reflex agent only perceives the status of its current room.
     - It does NOT know its global position index, nor does it maintain a memory
       of which rooms were previously visited or cleaned.
   * Deterministic Infinite Boundary Loop:
     - Suppose an agent is at Room 0 (the leftmost room) and Room 0 is Clean.
     - If the deterministic rule says: "If Clean -> Move Left", the agent
       repeatedly attempts to move Left, bumps into the wall, remains at Room 0,
       and loops infinitely!
     - Similarly, if the rule alternates deterministically without state (e.g.
       Left then Right), it can oscillate between adjacent rooms forever without
       reaching distant dirty rooms.
   * How Randomization Solves It:
     - If the agent chooses Left or Right with probability 0.5 when clean:
       P(Escape Room 0 in step k) = 1 - (0.5)^k -> 1 as k -> inf.
       The probability of being stuck at a boundary room drops to zero.

3. RATIONALE FOR RANDOMIZATION IN UNKNOWN ENVIRONMENTS:
   -----------------------------------------------------
   * Russell & Norvig's Insight (AIMA Section 2.4.5):
     "Randomization is a simple and effective way for reflex agents to escape
      from infinite loops in partially observable environments."
   * Game Theory & Exploration:
     - Randomization prevents adversarial environments or unseen obstacles from
       exploiting fixed behavioral patterns.
   * Trade-offs:
     - While randomized reflex agents guarantee eventual completion, they are
       often inefficient compared to Model-Based Agents that maintain an internal
       state (map) of visited rooms.
     - Time Complexity: Expected number of steps on an N-room 1D random walk
       to cover all rooms is O(N^2) for general cover time and O(N) average moves
       per localized segment.

4. SAFETY MECHANISMS (MAXIMUM STEP LIMIT):
   ----------------------------------------
   * In stochastic simulations, execution must be bounded by a maximum step
     threshold (e.g., 1000 steps) to guarantee termination in real-time systems,
     detect pathological configurations, and prevent CPU starvation.

================================================================================
"""
    print(notes)


def run_single_simulation(
    grid: List[int],
    start_index: int,
    seed: Optional[int] = None,
    max_steps: int = 1000,
    verbose: bool = True,
) -> SimulationResult:
    """Execute a single randomized reflex agent simulation run.

    Args:
        grid: List array of 0s (Clean) and 1s (Dirty).
        start_index: Starting room position (0-indexed).
        seed: Random seed for reproducibility.
        max_steps: Maximum step ceiling.
        verbose: If True, prints detailed step traces and summaries.

    Returns:
        SimulationResult instance.
    """
    env = OneDimensionalGridEnvironment(initial_state=grid, start_index=start_index)
    agent = RandomizedReflexAgent(seed=seed)

    if verbose:
        print("\n" + "=" * 65)
        print("          RANDOMIZED REFLEX AGENT SIMULATION (1D GRID)")
        print("=" * 65)
        print("Core AI Concept: Randomized Agent Behavior in Partially Observable Space")
        print("Agent Access   : Local Room Percept Only (No Global Array Access)")
        print("-" * 65)
        print(f"Initial Grid State  : {grid} (Length: {len(grid)} rooms)")
        print(f"Agent Start Index   : Room [{start_index}]")
        print(f"Random Seed         : {seed if seed is not None else 'Stochastic (System Entropy)'}")
        print(f"Max Step Limit      : {max_steps}")
        print("-" * 65)
        print("Step-by-Step Simulation Trace:")

    while not env.is_all_clean():
        if env.step_count >= max_steps:
            break
        record = env.execute_step(agent)
        if verbose:
            print(record.format_trace())

    result = SimulationResult(
        initial_grid=list(grid),
        final_grid=env.grid,
        start_index=start_index,
        final_index=env.agent_index,
        total_steps=env.step_count,
        total_moves=env.move_count,
        total_cleans=env.clean_count,
        is_all_clean=env.is_all_clean(),
        stopped_by_limit=(not env.is_all_clean() and env.step_count >= max_steps),
        history=env.history,
    )

    if verbose:
        print("-" * 65)
        if result.is_all_clean:
            print(f"All rooms clean in {result.total_steps} steps.")
        else:
            print(f"Halted after {result.total_steps} steps (Safety limit reached).")
        print(f"Total Navigation Moves   : {result.total_moves}")
        print(f"Total Cleaning Actions   : {result.total_cleans}")
        print(f"Final Grid State         : {result.final_grid}")
        print(f"Final Agent Location     : Room [{result.final_index}]")
        print("=" * 65 + "\n")

    return result


def run_benchmark(
    grid: List[int],
    start_index: int,
    num_trials: int = 100,
    max_steps: int = 2000,
) -> None:
    """Run Monte Carlo simulation across multiple trials to analyze step distribution.

    Args:
        grid: Initial room state array.
        start_index: Starting agent index.
        num_trials: Number of stochastic trials.
        max_steps: Safety ceiling per trial.
    """
    print("\n" + "=" * 65)
    print(f"     MONTE CARLO EMPIRICAL BENCHMARK ({num_trials} TRIALS)")
    print("=" * 65)
    print(f"Environment Initial State : {grid}")
    print(f"Starting Position         : Room [{start_index}]")
    print(f"Total Trials              : {num_trials}")
    print("-" * 65)

    steps_list: List[int] = []
    moves_list: List[int] = []
    cleans_list: List[int] = []
    successes: int = 0

    for trial_idx in range(num_trials):
        # Unseeded for genuine stochastic observation
        res = run_single_simulation(
            grid=grid,
            start_index=start_index,
            seed=None,
            max_steps=max_steps,
            verbose=False,
        )
        steps_list.append(res.total_steps)
        moves_list.append(res.total_moves)
        cleans_list.append(res.total_cleans)
        if res.is_all_clean:
            successes += 1

    avg_steps = sum(steps_list) / len(steps_list)
    avg_moves = sum(moves_list) / len(moves_list)
    avg_cleans = sum(cleans_list) / len(cleans_list)
    min_steps = min(steps_list)
    max_steps_observed = max(steps_list)

    # Calculate standard deviation
    variance = sum((x - avg_steps) ** 2 for x in steps_list) / len(steps_list)
    std_dev = variance ** 0.5

    print(f"Success Rate              : {successes}/{num_trials} ({(successes / num_trials) * 100:.1f}%)")
    print(f"Average Total Steps       : {avg_steps:.2f} +/- {std_dev:.2f}")
    print(f"Average Navigation Moves  : {avg_moves:.2f}")
    print(f"Average Clean Operations  : {avg_cleans:.2f}")
    print(f"Minimum Steps Observed    : {min_steps}")
    print(f"Maximum Steps Observed    : {max_steps_observed}")
    print("=" * 65 + "\n")


def parse_state_argument(state_arg: str) -> List[int]:
    """Parse comma-separated or space-separated room state input.

    Args:
        state_arg: String representing room states, e.g. "1,0,0,1,1" or "1 0 0 1 1".

    Returns:
        List of integers (0 or 1).

    Raises:
        ValueError: If input format is invalid.
    """
    cleaned = state_arg.replace(",", " ").strip()
    tokens = cleaned.split()
    if not tokens:
        raise ValueError("State string cannot be empty.")

    rooms: List[int] = []
    for token in tokens:
        val = int(token)
        if val not in (ROOM_CLEAN, ROOM_DIRTY):
            raise ValueError(f"Invalid room value '{token}'. Must be 0 (Clean) or 1 (Dirty).")
        rooms.append(val)
    return rooms


def main() -> None:
    """CLI Entry point for Randomized Reflex Agent Simulation."""
    parser = argparse.ArgumentParser(
        description="Randomized Reflex Agent (1D Grid) - AI Lab Question 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default="1,0,0,1,1",
        help="Initial room state array (0=Clean, 1=Dirty). E.g. '1,0,0,1,1' or '1 0 1 1 0' (default: 1,0,0,1,1)",
    )
    parser.add_argument(
        "-i",
        "--start-index",
        type=int,
        default=2,
        help="Initial 0-based room index of the agent (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible stochastic choices (use --seed 58 for sample 12-step trace)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum step ceiling to prevent infinite execution (default: 1000)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run Monte Carlo simulation over 100 trials to observe step distribution",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials for benchmark mode (default: 100)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode prompting for room states and start index",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display curated Viva Voce theory notes for Question 2",
    )

    args = parser.parse_args()

    if args.viva:
        display_viva_notes()
        return

    if args.interactive:
        print("\n--- Interactive Setup: Randomized Reflex Agent ---")
        try:
            raw_state = input("Enter room states (0=Clean, 1=Dirty) separated by spaces [e.g. 1 0 0 1 1]: ").strip()
            state = parse_state_argument(raw_state if raw_state else "1 0 0 1 1")
            raw_idx = input(f"Enter starting index (0 to {len(state) - 1}) [default: 2]: ").strip()
            start_idx = int(raw_idx) if raw_idx else 2
            raw_seed = input("Enter random seed (or press Enter for purely stochastic): ").strip()
            seed = int(raw_seed) if raw_seed else None
        except (ValueError, KeyboardInterrupt) as exc:
            print(f"\n[ERROR] Invalid input: {exc}. Aborting.")
            sys.exit(1)

        run_single_simulation(grid=state, start_index=start_idx, seed=seed, max_steps=args.max_steps)
        return

    grid = parse_state_argument(args.state)

    if args.benchmark:
        run_benchmark(grid=grid, start_index=args.start_index, num_trials=args.trials, max_steps=args.max_steps)
    else:
        run_single_simulation(
            grid=grid,
            start_index=args.start_index,
            seed=args.seed,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
