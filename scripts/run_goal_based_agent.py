"""Runner script for Goal-Based Navigation Agent Simulation (Question 4).

Demonstrates a Goal-Based Agent tasked with finding a charging station in a 5x5
grid matrix. The agent maintains an explicit representation of the goal and employs
a greedy Manhattan distance-reducing heuristic to select orthogonal actions
(Up, Down, Left, Right).

Features:
- Default execution on sample input: Start (0, 0), Goal (3, 4)
- Custom Start and Goal coordinate inputs via CLI
- Randomized goal/start placement with seed reproducibility
- Interactive step-by-step simulation mode
- Optional challenge demonstration: Explicit prevention of diagonal movement
- Metric comparison: Manhattan distance (L1) vs. Euclidean distance (L2)
- Comprehensive Viva Voce theory guide covering all core practical topics
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.goal_based_agent import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    DIAGONAL_DELTAS,
    ORTHOGONAL_DELTAS,
    GoalBasedNavigationAgent,
    GoalSimulationResult,
    GridEnvironment,
    euclidean_distance,
    generate_random_coordinate,
    get_successors,
    is_diagonal_movement,
    manhattan_distance,
    run_goal_based_simulation,
    validate_orthogonal_transition,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce theory guide for Question 4."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
                    Question 4: Goal-Based Navigation Agent
================================================================================

1. DIFFERENCE BETWEEN GOAL-BASED AND REFLEX AGENTS:
   -------------------------------------------------
   * Simple Reflex Agent:
     - Operates strictly on condition-action rules: IF condition THEN action.
     - Has no memory of past states and NO knowledge of what it wants to achieve
       in the future.
     - Extremely rigid: If the goal location changes, the entire rule set must
       be rewritten.
   * Model-Based Reflex Agent:
     - Maintains an internal world model and history of visited states to handle
       partial observability.
     - Still acts primarily on reflex rules based on its current belief state.
   * Goal-Based Agent:
     - Combines environmental percepts with an explicit representation of a GOAL
       (a target state, such as reaching charging station at coordinate (3, 4)).
     - Evaluates candidate actions by considering their future outcomes:
       "Will taking this action bring me closer to achieving my goal?"
     - Highly adaptable: Changing the goal coordinates simply changes the goal
       test or distance metric without altering the underlying agent logic.

2. SEARCH VS. HEURISTICS:
   ----------------------
   * Search:
     - A systematic algorithmic exploration of the state space from an initial
       state through possible transitions to find a path to a goal state.
     - Uninformed Search (e.g., BFS, DFS): Explores blindly without knowing
       whether a branch is physically close to the goal. Guaranteed complete,
       but often expands unnecessary states (high time/space complexity).
     - Informed Search (e.g., A*, Greedy Best-First): Uses problem-specific
       knowledge to guide exploration efficiently.
   * Heuristic Function h(n):
     - An evaluation function estimating the cost or distance from current state
       n to the nearest goal state.
     - In our Goal-Based Agent, Manhattan distance h(n) = |x - gx| + |y - gy|
       acts as a heuristic function.
     - Greedy coordinate reduction selects the action that maximizes distance
       reduction (strictly decreases h(n)) at every step.

3. MANHATTAN VS. EUCLIDEAN DISTANCE:
   ---------------------------------
   * Manhattan Distance (L1 Norm / Taxicab Metric / City Block Distance):
       d_M((x1, y1), (x2, y2)) = |x1 - x2| + |y1 - y2|
     - Measures distance traveling strictly along orthogonal grid lines (at right
       angles: Up, Down, Left, Right).
     - Models real-world scenarios where movement is constrained to grid axes
       (e.g., Manhattan city streets, warehouse grid robots, 4-connected tiles).
     - In an obstacle-free 4-connected grid, Manhattan distance gives the EXACT
       minimum number of steps required to reach the goal.
   * Euclidean Distance (L2 Norm / Straight-Line Metric):
       d_E((x1, y1), (x2, y2)) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
     - Measures the direct geometric distance ("as the crow flies") in continuous
       Cartesian space.
   * Key Theoretical Relationship & Admissibility:
     - By the Minkowski inequality and triangle geometry:
         d_E(P1, P2) <= d_M(P1, P2)  for all grid coordinates.
     - In A* search on an orthogonal grid, Euclidean distance is ADMISSIBLE
       (never overestimates actual cost) but INADMISSIBLE if diagonal shortcuts
       are counted as unit cost when they are physically impossible.
     - Manhattan distance is both ADMISSIBLE and CONSISTENT (monotone) on an
       orthogonal 4-connected grid, making it the superior heuristic.

4. OPTIONAL CHALLENGE: DIAGONAL MOVEMENT PREVENTION:
   --------------------------------------------------
   * In an 8-connected grid (Chebyshev / L_infinity metric), diagonal movement
     is allowed with delta_x != 0 and delta_y != 0.
   * In a 4-connected orthogonal grid (Manhattan metric), diagonal moves
     (cutting corners) are strictly forbidden:
       - Manhattan step constraint: |dx| + |dy| == 1.
       - If |dx| != 0 and |dy| != 0, the transition is diagonal and explicitly
         rejected by the successor function.
================================================================================
"""
    print(notes)


def run_sample_scenario() -> None:
    """Execute standard sample scenario: Start (0, 0), Goal (3, 4)."""
    print("=" * 80)
    print("      QUESTION 4: GOAL-BASED NAVIGATION AGENT SIMULATION (SAMPLE INPUT)")
    print("=" * 80)
    print("Problem Statement: Navigate 5x5 grid from (0, 0) to Charging Station at (3, 4).")
    print("Technique        : Greedy coordinate reduction minimizing Manhattan distance.")
    print("-" * 80)

    result = run_goal_based_simulation(start_pos=(0, 0), goal_pos=(3, 4))

    print("\n--- Step-by-Step Navigation Trace ---")
    for record in result.history:
        print(record.format_trace())

    print("\n--- Final Simulation Summary ---")
    print(result.summary())

    print("\n--- Expected Output Path ---")
    print(f"Path: {result.path}")

    print("\n--- 5x5 Grid Matrix Visualization ---")
    print(result.render_grid())
    print("Legend: [A]=Start (0,0) | [G]=Charging Station Goal (3,4) | [*]=Path | [.]=Empty\n")


def run_optional_challenge_demo() -> None:
    """Demonstrate explicit prevention of diagonal movement in successor function."""
    print("=" * 80)
    print("       OPTIONAL CHALLENGE: EXPLICIT DIAGONAL MOVEMENT PREVENTION")
    print("=" * 80)
    print("Objective: Prevent diagonal movement explicitly in the successor function.")
    print("Rules    : 4-connected orthogonal movements only (|dx| + |dy| == 1).")
    print("           Diagonal movements (|dx| > 0 and |dy| > 0) must be rejected.\n")

    current = (2, 2)
    print(f"Current Position: {current}")
    print("\n1. Testing standard successor generation (allow_diagonal=False):")
    successors = get_successors(current, grid_dimensions=(5, 5), allow_diagonal=False)
    for action, next_pos in successors:
        dx = abs(next_pos[0] - current[0])
        dy = abs(next_pos[1] - current[1])
        print(f"   Action: {action:<5} -> Next: {next_pos} | dx={dx}, dy={dy} (Orthogonal: OK)")

    print("\n2. Testing explicit rejection of diagonal transitions:")
    test_diagonal_transitions = [
        ((2, 2), (1, 1), "Up-Left"),
        ((2, 2), (1, 3), "Up-Right"),
        ((2, 2), (3, 1), "Down-Left"),
        ((2, 2), (3, 3), "Down-Right"),
    ]

    for start_pt, diag_pt, label in test_diagonal_transitions:
        try:
            validate_orthogonal_transition(start_pt, diag_pt)
            print(f"   [FAIL] {label} transition {start_pt} -> {diag_pt} was not caught!")
        except ValueError as err:
            print(f"   [BLOCKED] {label} {start_pt} -> {diag_pt}: {err}")

    print("\nSummary: All diagonal movements are explicitly blocked by the successor guard.")
    print("=" * 80)


def run_metric_comparison(start: Tuple[int, int], goal: Tuple[int, int]) -> None:
    """Compare Manhattan and Euclidean metrics along the navigation path."""
    print("=" * 80)
    print("       DISTANCE METRIC COMPARISON: MANHATTAN (L1) VS. EUCLIDEAN (L2)")
    print("=" * 80)
    print(f"Start: {start} | Goal: {goal}\n")

    result = run_goal_based_simulation(start_pos=start, goal_pos=goal)

    print(f"{'Step':<6} | {'Coordinate':<12} | {'Manhattan (L1)':<16} | {'Euclidean (L2)':<16} | {'d_E <= d_M':<10}")
    print("-" * 70)

    for idx, coord in enumerate(result.path):
        d_m = manhattan_distance(coord, goal)
        d_e = euclidean_distance(coord, goal)
        admissible = d_e <= d_m
        print(f"{idx:<6} | {str(coord):<12} | {d_m:<16} | {d_e:<16.4f} | {str(admissible):<10}")

    print("-" * 70)
    print("Observation: At every single coordinate, Euclidean distance d_E <= Manhattan d_M.")
    print("In an orthogonal 4-connected grid, Manhattan distance reflects the true step count.\n")


def run_interactive_mode() -> None:
    """Run interactive CLI mode allowing custom inputs and step-by-step traversal."""
    print("=" * 80)
    print("              INTERACTIVE GOAL-BASED AGENT SIMULATION")
    print("=" * 80)

    def prompt_coord(label: str, default: Tuple[int, int]) -> Tuple[int, int]:
        while True:
            raw = input(f"Enter {label} coordinates as 'row col' [default: {default[0]} {default[1]}]: ").strip()
            if not raw:
                return default
            parts = raw.replace(",", " ").split()
            if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                r, c = int(parts[0]), int(parts[1])
                if 0 <= r < 5 and 0 <= c < 5:
                    return (r, c)
                print("Coordinates must be in range [0, 4] for a 5x5 grid.")
            else:
                print("Please enter two space-separated integers, e.g., '0 0'.")

    start = prompt_coord("Start", (0, 0))
    goal = prompt_coord("Goal (Charging Station)", (3, 4))

    result = run_goal_based_simulation(start_pos=start, goal_pos=goal)

    print("\n--- Simulation Result ---")
    print(result.summary())
    print("\n--- Path Taken ---")
    print(f"Path: {result.path}")
    print("\n--- 5x5 Grid Matrix ---")
    print(result.render_grid())


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI dispatcher for Question 4: Goal-Based Navigation Agent."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_goal_based_agent.py",
        description="Goal-Based Navigation Agent: 5x5 Grid Matrix Simulation (Question 4)",
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        default=[0, 0],
        help="Start coordinates (row col). Defaults to: 0 0.",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        default=[3, 4],
        help="Goal charging station coordinates (row col). Defaults to: 3 4.",
    )
    parser.add_argument(
        "--random-goal",
        action="store_true",
        help="Place the charging station randomly in the 5x5 grid.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Place the agent start position randomly in the 5x5 grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random placement.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal mode with custom coordinate prompts.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Demonstrate optional challenge: explicit prevention of diagonal movement.",
    )
    parser.add_argument(
        "--compare-metrics",
        action="store_true",
        help="Compare Manhattan vs Euclidean distances along the path.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display curated Viva Voce theory questions and answers.",
    )

    args = parser.parse_args(argv)

    if args.viva:
        display_viva_notes()
        return

    if args.challenge:
        run_optional_challenge_demo()
        return

    if args.interactive:
        run_interactive_mode()
        return

    start_pos = tuple(args.start)  # type: ignore[assignment]
    goal_pos = tuple(args.goal)  # type: ignore[assignment]

    if args.random_start:
        start_pos = generate_random_coordinate(grid_dimensions=(5, 5), seed=args.seed)

    if args.compare_metrics:
        run_metric_comparison(start=start_pos, goal=goal_pos)
        return

    # If neither interactive nor custom flags were given and start is (0,0) and goal is (3,4) without random:
    if not args.random_goal and start_pos == (0, 0) and goal_pos == (3, 4):
        run_sample_scenario()
        return

    print("=" * 80)
    print("      QUESTION 4: GOAL-BASED NAVIGATION AGENT SIMULATION")
    print("=" * 80)
    result = run_goal_based_simulation(
        start_pos=start_pos,
        goal_pos=None if args.random_goal else goal_pos,
        random_goal=args.random_goal,
        seed=args.seed,
    )

    print("\n--- Step-by-Step Navigation Trace ---")
    for record in result.history:
        print(record.format_trace())

    print("\n--- Final Simulation Summary ---")
    print(result.summary())

    print("\n--- Path Taken ---")
    print(f"Path: {result.path}")

    print("\n--- 5x5 Grid Matrix Visualization ---")
    print(result.render_grid())


if __name__ == "__main__":
    main()
