"""Runner script for Multi-Agent Environment Simulation (Question 6).

Demonstrates two autonomous agents operating concurrently in a shared 4x4 grid:
- Agent A ('Builder') places physical blocks (1s) on empty cells (0s).
- Agent B ('Destroyer') removes physical blocks (1s) restoring them to empty cells (0s).
- Turn-based execution: Agents take turns sequentially.
- Starts at opposite corners (0, 0) and (3, 3).
- Optional Challenge: Collision awareness and proactive avoidance.
- Comprehensive Viva Voce theory reference.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.multi_agent_simulation import (
    CELL_BLOCK,
    CELL_EMPTY,
    ROLE_BUILDER,
    ROLE_DESTROYER,
    STRATEGY_GREEDY,
    STRATEGY_RANDOM,
    TURN_MODE_COMPOUND,
    TURN_MODE_DISCRETE,
    VALID_STRATEGIES,
    VALID_TURN_MODES,
    AgentTurnRecord,
    BuilderAgent,
    DestroyerAgent,
    MultiAgentGridEnvironment,
    MultiAgentSimulationResult,
    run_multi_agent_simulation,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce theory guide for Question 6."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
                Question 6: Multi-Agent Environment Simulation
================================================================================

1. MULTI-AGENT COMPETITIVE ENVIRONMENTS:
   -------------------------------------
   * What is a Multi-Agent System (MAS)?
     - A system composed of multiple autonomous decision-making agents interacting
       within a shared environment.
     - Each agent perceives local or global states, computes actions via its own
       percept-action function f: P* -> A, and modifies the shared environmental state.
   * Competitive vs. Cooperative vs. Mixed:
     - Purely Competitive (Zero-Sum): One agent's gain is strictly the other agent's
       loss. In our simulation, Builder's utility U_A = +1 * (blocks on grid), while
       Destroyer's utility U_B = -1 * (blocks on grid) = +1 * (empty cells). Their
       performance objectives are in direct opposition.
     - Cooperative: Agents share a joint utility function (e.g. two rescue robots
       maximizing survivors found).
     - Mixed-Motive (General-Sum): Agents have partially overlapping and partially
       competing goals (e.g. autonomous traffic merging, trading markets).
   * Key Challenges in Multi-Agent Systems:
     - Non-stationarity: From Agent A's perspective, the world dynamics change
       because Agent B is also learning or moving.
     - Coordination and Collision Avoidance: Ensuring agents do not occupy the same
       spatial coordinates or contend destructively for resources.
     - Imperfect Information: Agents may not have visibility into the opponent's
       internal goals, strategies, or sensor readings.

2. EPISODIC VS. SEQUENTIAL ENVIRONMENTS:
   -------------------------------------
   * Episodic Environments:
     - The agent's overall experience is divided into independent, atomic "episodes".
     - In each episode, the agent perceives the current input, performs a single
       action, and receives a reward/feedback.
     - Critical Property: The action taken in the current episode has NO effect
       whatsoever on future episodes.
     - Example: Handwritten digit classification (MNIST) or defect detection on
       an assembly line.
   * Sequential Environments:
     - The current decision influences all future environmental states, percepts,
       and available actions.
     - Short-term sacrifices may be necessary to achieve long-term optimal rewards.
     - Why is our Grid Simulation Sequential?
       When Agent A places a block at coordinate (0, 0) at Step 1, that block
       persists into Step 2, Step 3, etc., fundamentally altering the percept
       and action possibilities for both Agent A and Agent B in subsequent rounds.

3. STATIC VS. DYNAMIC ENVIRONMENTS:
   ---------------------------------
   * Static Environments:
     - The physical state of the environment does NOT change while the agent is
       deliberating its next move.
     - The passage of time during decision computation has zero impact on the world.
     - Example: Crossword puzzle solver, Sudoku solver, or offline path planning.
   * Dynamic Environments:
     - The environment continuously evolves or mutates while the agent is computing
       its action or between an agent's individual actions.
     - The agent must account for external changes or risk operating on stale data.
     - Example: Autonomous driving, stock trading.
     - In our Multi-Agent Grid World:
       Between Step t and Step t+1 for Agent A, Agent B enters the environment and
       removes a block or shifts position. Thus, from Agent A's viewpoint, the
       environment is dynamic because external forces modify the state!
   * Semidynamic Environments:
     - The physical state of the environment does not change with time, but the
       agent's performance score or reward decays as time elapses (e.g. blitz chess
       with a ticking clock).

4. SHARED REFERENCES & OBJECT-ORIENTED STATE MUTATION:
   --------------------------------------------------
   * In Python OOP, both Agent A and Agent B hold references to the same underlying
     `MultiAgentGridEnvironment` instance.
   * Modifying `grid[r, c] = 1` inside Agent A's turn immediately reflects in the
     matrix read by Agent B on its turn.
   * Defensive copies (`grid.copy()`) are critical when capturing historic snapshots
     or generating percepts to prevent unwanted side-channel mutations.

5. COLLISION AVOIDANCE MECHANISM (OPTIONAL CHALLENGE):
   ---------------------------------------------------
   * Percept Augmentation: The environment discloses peer position `(r_peer, c_peer)`.
   * Spatial Pruning: Candidate moves `M = {(r+dr, c+dc)}` are filtered:
     `M_safe = {m in M | m != (r_peer, c_peer)}`.
   * Fallback Safety: If `|M_safe| == 0` (cornered), the agent executes `NoOp`
     rather than causing a collision.
================================================================================
"""
    print(notes)


def print_step_trace(record: AgentTurnRecord) -> None:
    """Print formatted step trace for an agent's turn."""
    print(record.format_trace())


def run_interactive_simulation(
    steps: int = 10,
    avoid_collisions: bool = False,
    step_mode: str = TURN_MODE_COMPOUND,
    strategy: str = STRATEGY_RANDOM,
    seed: Optional[int] = 42,
) -> None:
    """Run simulation interactively step-by-step with terminal prompts."""
    print("\n" + "=" * 70)
    print("      INTERACTIVE MULTI-AGENT ENVIRONMENT SIMULATION")
    print("=" * 70)
    print(f"Grid: 4x4 | Steps: {steps} | Mode: {step_mode} | Strategy: {strategy}")
    print(f"Collision Avoidance: {'Enabled' if avoid_collisions else 'Disabled'}")
    print("Press [Enter] after each step to advance (or type 'q' to quit).\n")

    seed_a = seed if seed is not None else None
    seed_b = (seed + 1000) if seed is not None else None

    agent_a = BuilderAgent(
        name="Agent_A",
        initial_position=(0, 0),
        seed=seed_a,
        strategy=strategy,
    )
    agent_b = DestroyerAgent(
        name="Agent_B",
        initial_position=(3, 3),
        seed=seed_b,
        strategy=strategy,
    )
    env = MultiAgentGridEnvironment(
        rows=4,
        cols=4,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    print("Initial Grid State (Step 0):")
    print(env.grid)
    print()

    for s in range(1, steps + 1):
        try:
            user_in = input(f"Press Enter for Step {s}/{steps} (or 'q' to exit): ")
            if user_in.strip().lower() == "q":
                print("Interactive simulation aborted by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return

        rec_a, rec_b = env.step(
            step_number=s,
            avoid_collisions=avoid_collisions,
            step_mode=step_mode,
        )
        print(rec_a.format_trace())
        print(rec_b.format_trace())
        print(f"\nGrid Matrix after Step {s}:")
        print(env.grid)
        print()

    print("=" * 70)
    print("Final Matrix State (NumPy Array):")
    print(env.grid)
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for Multi-Agent Environment Simulation."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_multi_agent_simulation.py",
        description="Question 6: Multi-Agent Environment Simulation (Builder vs Destroyer on 4x4 Grid)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of interaction steps to simulate (default: 10)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of grid rows (default: 4)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of grid columns (default: 4)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_TURN_MODES,
        default=TURN_MODE_COMPOUND,
        help="Turn execution mode: 'compound' (act + move) or 'discrete' (act or move) (default: compound)",
    )
    parser.add_argument(
        "--strategy",
        choices=VALID_STRATEGIES,
        default=STRATEGY_RANDOM,
        help="Agent movement strategy: 'random' or 'greedy' (default: random)",
    )
    parser.add_argument(
        "--challenge",
        "--avoid-collisions",
        action="store_true",
        dest="avoid_collisions",
        help="Enable Optional Challenge: Agents actively avoid collisions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible simulation (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print per-turn trace log (default: True)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run simulation interactively step-by-step",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory reference guide",
    )

    args = parser.parse_args(argv)

    if args.viva:
        display_viva_notes()
        return

    if args.interactive:
        run_interactive_simulation(
            steps=args.steps,
            avoid_collisions=args.avoid_collisions,
            step_mode=args.mode,
            strategy=args.strategy,
            seed=args.seed,
        )
        return

    print("=" * 70)
    print("      QUESTION NO. 6: MULTI-AGENT ENVIRONMENT SIMULATION")
    print("=" * 70)
    print(f"Grid Dimensions:     {args.rows}x{args.cols}")
    print(f"Total Steps:         {args.steps}")
    print(f"Turn Mode:           {args.mode.capitalize()}")
    print(f"Movement Strategy:   {args.strategy.capitalize()}")
    print(f"Collision Avoidance: {'ENABLED (Optional Challenge)' if args.avoid_collisions else 'Disabled'}")
    print(f"Random Seed:         {args.seed}")
    print("=" * 70 + "\n")

    result: MultiAgentSimulationResult = run_multi_agent_simulation(
        steps=args.steps,
        grid_size=(args.rows, args.cols),
        avoid_collisions=args.avoid_collisions,
        movement_strategy=args.strategy,
        step_mode=args.mode,
        seed=args.seed,
    )

    if args.verbose:
        print("Chronological Turn Trace:")
        print("-" * 70)
        for record in result.history:
            print_step_trace(record)
        print("-" * 70 + "\n")

    print("Final Grid State (Visual Board):")
    print(result.display_board(show_agents=True))
    print()

    print("Sample Output (NumPy Matrix displaying 1s and 0s):")
    print(result.display_matrix())
    print()

    print(result.summary())


if __name__ == "__main__":
    main()
