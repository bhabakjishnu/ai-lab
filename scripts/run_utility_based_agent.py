"""Runner script for Utility-Based Agent Simulation (Question 5).

Demonstrates a Utility-Based Agent tasked with evaluating candidate paths to a goal
in a weighted terrain graph. The agent formulates a utility function U(path) = -1 * total_cost
to select the path that maximizes utility (minimizes total cost).

Features:
- Default evaluation of 3 predefined paths on the university sample graph:
  * A-B-D (Cost 6.0, Utility -6.0)
  * A-C-D (Cost 4.0, Utility -4.0) [Selected Optimal Path]
  * A-E-D (Cost 8.0, Utility -8.0)
- Support for custom candidate paths via CLI flags
- Optional Challenge: Battery constraint filtering (rejects paths exceeding energy budget)
- Automatic path discovery between start and goal nodes using DFS
- Step-by-step interactive mode
- Comprehensive Viva Voce reference guide
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utility_based_agent import (
    DEFAULT_GRAPH,
    DEFAULT_PREDEFINED_PATHS,
    DEFAULT_TERRAIN_COSTS,
    EdgeDetail,
    GraphDict,
    PathEvaluation,
    UtilityBasedAgent,
    UtilitySimulationResult,
    calculate_path_cost,
    calculate_utility,
    find_all_paths,
    run_utility_based_simulation,
)


def display_viva_notes() -> None:
    """Print curated Viva Voce theory guide for Question 5."""
    notes = """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
                    Question 5: Utility-Based Agent Simulation
================================================================================

1. UTILITY VS. GOAL:
   -----------------
   * Goal-Based Agent:
     - Maintains a binary goal predicate: G(s) in {True, False}.
     - Differentiates only between goal states and non-goal states.
     - Cannot distinguish between different paths that achieve the same goal.
       For instance, whether reaching the goal costs $4 (Grass + Mud) or $6 (Water),
       a purely goal-based agent considers both equally successful.
   * Utility-Based Agent:
     - Uses a real-valued utility function U: S -> R that maps a state (or trajectory)
       to a continuous scalar of desirability / "happiness".
     - Enables trade-offs: When there are multiple ways to reach a goal, the agent
       selects the higher-utility path (e.g. lower cost, safer, less battery drain).
     - Resolves conflicts: When multiple conflicting objectives exist (e.g. speed vs.
       battery consumption vs. obstacle risk), a utility function provides a unified
       scoring metric.

2. EXPECTED UTILITY THEORY & MAXIMUM EXPECTED UTILITY (MEU):
   --------------------------------------------------------
   * In deterministic environments:
     - An agent's action directly leads to an outcome state s'.
     - Optimal decision rule: a* = argmax_a U(Result(a, s)).
   * In stochastic / uncertain environments:
     - Taking action 'a' can lead to multiple possible outcome states s', each with
       probability P(Result(a) = s' | a, Percept).
     - Expected Utility (EU) is the weighted expectation across outcomes:
         EU(a | e) = sum_{s'} P(Result(a) = s' | a, e) * U(s')
     - Principle of Maximum Expected Utility (MEU):
       A rational agent should select the action that maximizes its expected utility:
         action* = argmax_a EU(a | e)
     - Axioms of Utility (von Neumann & Morgenstern):
       Orderability, Transitivity, Continuity, Substitutability, Monotonicity, and
       Decomposability guarantee that preferences can be represented by a utility function.

3. ENVIRONMENT PERFORMANCE MEASURES & PEAS:
   ----------------------------------------
   * Part of Russell & Norvig's PEAS Framework:
     - Performance Measure: Path cost efficiency, energy conservation, survival.
     - Environment: Weighted graph representing terrain (Grass=1, Mud=3, Water=5).
     - Actuators: Waypoint path traversal / locomotion commands.
     - Sensors: Graph topography, terrain sensors, battery gauge.
   * Objective Formulation:
     - Performance measures must be formulated objectively from an external observer's
       perspective, evaluating real environmental impact (not internal agent perception).

4. UTILITY FUNCTION FORMULATION (U = -1 * total_cost):
   ---------------------------------------------------
   * Why negate total cost?
     - Traversal cost represents an expenditure/loss (smaller is better).
     - Utility functions represent desirability (larger is better).
     - Multiplying cost by -1 maps minimization directly to maximization:
         argmin_p Cost(p) == argmax_p (-Cost(p))
     - Example: Path A-C-D (Cost 4) -> Utility -4.
                Path A-B-D (Cost 6) -> Utility -6.
                Since -4 > -6, Path A-C-D provides higher utility!

5. COMPLEXITY ANALYSIS:
   --------------------
   * Time Complexity: O(P * L)
     - P is the number of candidate paths.
     - L is the path length (number of nodes).
     - Summing costs along an L-node path requires (L - 1) dictionary lookups in O(1).
     - Over P candidate paths, total runtime is O(P * L).
     - Finding the maximum utility among P paths takes O(P) comparisons.
   * Space Complexity: O(1) auxiliary space (evaluating paths iteratively without
     auxiliary allocations beyond the output result).
================================================================================
"""
    print(notes.strip())


def print_graph_summary(graph: GraphDict) -> None:
    """Print readable summary of graph topography and terrain edge costs."""
    print("\n--- Environment Graph Topography & Terrain Edge Costs ---")
    for node in sorted(graph.keys()):
        neighbors = graph[node]
        if not neighbors:
            print(f"  Node [{node}]: (Destination / Leaf Node)")
            continue
        edges_str = []
        for nbr, edge_info in neighbors.items():
            if isinstance(edge_info, dict):
                cost = edge_info.get("cost", 0.0)
                terrain = edge_info.get("terrain", "Direct")
                edges_str.append(f"{nbr} (Terrain: {terrain}, Cost: {cost})")
            elif isinstance(edge_info, (int, float)):
                edges_str.append(f"{nbr} (Cost: {edge_info})")
            else:
                edges_str.append(f"{nbr} ({edge_info})")
        print(f"  Node [{node}] -> {', '.join(edges_str)}")
    print("----------------------------------------------------------\n")


def print_evaluation_table(
    evaluations: Sequence[PathEvaluation],
    selected_path: Optional[PathEvaluation] = None,
    battery_limit: Optional[float] = None,
) -> None:
    """Print structured table of candidate paths, costs, utilities, and rankings."""
    print("Candidate Path Evaluations:")
    print("-" * 75)
    print(f"{'#':<3} {'Path':<16} {'Terrain Breakdown':<26} {'Cost':<7} {'Utility':<8} {'Status'}")
    print("-" * 75)

    for idx, ev in enumerate(evaluations, start=1):
        # Format breakdown string
        breakdown_parts = [
            f"{ed.terrain[0].upper()}:{int(ed.cost)}" for ed in ev.edge_details
        ]
        breakdown_str = " + ".join(breakdown_parts) if breakdown_parts else "N/A"

        # Determine status marker
        is_selected = selected_path is not None and ev.path == selected_path.path
        if is_selected:
            status = "SELECTED (Optimal)"
        elif not ev.is_valid:
            status = f"INVALID ({ev.rejection_reason})"
        elif not ev.is_feasible:
            status = f"REJECTED (>{battery_limit:.1f})"
        else:
            status = "FEASIBLE"

        print(
            f"{idx:<3} {ev.path_str:<16} {breakdown_str:<26} {ev.total_cost:<7.1f} {ev.utility:<8.1f} {status}"
        )
    print("-" * 75)


def run_default_demonstration() -> None:
    """Execute standard demonstration evaluating 3 predefined paths."""
    print("=" * 70)
    print("       QUESTION 5: UTILITY-BASED AGENT SIMULATION")
    print("=" * 70)
    print("Objective: Maximize Utility U(path) = -1 * total_cost across paths.")
    print("Terrains:  Grass=1.0, Sand=2.0, Mud=3.0, Rock=4.0, Water=5.0\n")

    print_graph_summary(DEFAULT_GRAPH)

    result = run_utility_based_simulation()
    print_evaluation_table(result.evaluated_paths, result.selected_path)

    if result.selected_path:
        print("\nOptimization Result:")
        print(f"  Selected Path: {result.selected_path.path_str}")
        print(f"  Total Cost:    {result.selected_path.total_cost}")
        print(f"  Utility:       {result.selected_path.utility}")
        print("\nDecision Rationale:")
        print(
            f"  The agent selected '{result.selected_path.path_str}' because its utility "
            f"({result.selected_path.utility}) is higher than all other candidate paths."
        )
        print(
            f"  Maximizing U = -Cost guarantees minimal energy expenditure (Cost = {result.selected_path.total_cost})."
        )
    print("=" * 70)


def run_challenge_demonstration(battery_limit: float = 5.0) -> None:
    """Execute Optional Challenge demonstrating battery constraint filtering."""
    print("=" * 70)
    print("   OPTIONAL CHALLENGE: BATTERY-CONSTRAINED UTILITY AGENT")
    print("=" * 70)
    print(f"Battery Capacity Limit: {battery_limit:.1f} units")
    print("Rule: Reject any path with total_cost > battery_limit regardless of utility.\n")

    result = run_utility_based_simulation(battery_limit=battery_limit)
    print_evaluation_table(
        result.evaluated_paths, result.selected_path, battery_limit=battery_limit
    )

    print("\nChallenge Analysis:")
    for ev in result.evaluated_paths:
        if ev.is_feasible:
            print(f"  [ACCEPTED] Path {ev.path_str:<10} Cost: {ev.total_cost} <= {battery_limit}")
        else:
            print(f"  [REJECTED] Path {ev.path_str:<10} Cost: {ev.total_cost} > {battery_limit} (Battery Depleted)")

    if result.selected_path:
        print(
            f"\nBest Feasible Path: {result.selected_path.path_str} "
            f"(Cost: {result.selected_path.total_cost}, Utility: {result.selected_path.utility})"
        )
    else:
        print("\nNo feasible path found: All paths exceeded the available battery capacity!")
    print("=" * 70)


def run_interactive_mode() -> None:
    """Interactive command-line mode allowing user to test custom paths."""
    print("=" * 70)
    print("          INTERACTIVE UTILITY-BASED AGENT MODE")
    print("=" * 70)
    print_graph_summary(DEFAULT_GRAPH)

    print("Enter candidate paths separated by hyphens or spaces (e.g., 'A-B-D' or 'A B D').")
    print("Enter a blank line when finished.\n")

    custom_paths: List[List[str]] = []
    path_idx = 1
    while True:
        try:
            line = input(f"Path #{path_idx} (or Enter to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborting interactive input.")
            return

        if not line:
            break

        # Parse path string: "A-B-D" or "A, B, D" or "A B D"
        delim = "-" if "-" in line else ("," if "," in line else None)
        if delim:
            nodes = [n.strip() for n in line.split(delim) if n.strip()]
        else:
            nodes = line.split()

        if len(nodes) < 2:
            print("  [!] A valid path must contain at least 2 nodes. Try again.")
            continue

        custom_paths.append(nodes)
        path_idx += 1

    if not custom_paths:
        print("No paths entered. Falling back to default predefined paths.")
        custom_paths = DEFAULT_PREDEFINED_PATHS

    battery_input = input("Enter battery limit (leave blank for unconstrained): ").strip()
    battery_limit: Optional[float] = None
    if battery_input:
        try:
            battery_limit = float(battery_input)
        except ValueError:
            print("  [!] Invalid number; proceeding without battery limit.")

    result = run_utility_based_simulation(paths=custom_paths, battery_limit=battery_limit)
    print_evaluation_table(
        result.evaluated_paths, result.selected_path, battery_limit=battery_limit
    )

    if result.selected_path:
        print(f"\nOptimal Path: {result.selected_path.path_str}")
        print(f"Utility:      {result.selected_path.utility}")
        print(f"Total Cost:   {result.selected_path.total_cost}")
    else:
        print("\nNo valid/feasible path found.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point dispatcher for Question 5 CLI."""
    parser = argparse.ArgumentParser(
        prog="run_utility_based_agent.py",
        description="Question 5: Utility-Based Agent Simulation & Path Cost Optimization",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Specify candidate paths to evaluate (e.g., --paths 'A-B-D' 'A-C-D' 'A-E-D').",
    )
    parser.add_argument(
        "--battery",
        type=float,
        default=None,
        help="Set maximum battery capacity threshold (rejects paths with cost > battery).",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run optional challenge demonstration with battery constraint (limit=5.0).",
    )
    parser.add_argument(
        "--all-paths",
        action="store_true",
        help="Automatically discover and evaluate all directed paths from 'A' to 'D'.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in step-by-step interactive mode.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory guide for Question 5.",
    )

    args = parser.parse_args(argv)

    if args.viva:
        display_viva_notes()
        return

    if args.interactive:
        run_interactive_mode()
        return

    if args.challenge:
        battery_limit = args.battery if args.battery is not None else 5.0
        run_challenge_demonstration(battery_limit=battery_limit)
        return

    # Check for auto-discovery of all paths
    candidate_paths = None
    if args.all_paths:
        discovered = find_all_paths(DEFAULT_GRAPH, "A", "D")
        print(f"\nAuto-discovered {len(discovered)} paths from A to D: {discovered}")
        candidate_paths = discovered
    elif args.paths:
        parsed_paths: List[List[str]] = []
        for p_str in args.paths:
            delim = "-" if "-" in p_str else ("," if "," in p_str else None)
            if delim:
                nodes = [n.strip() for n in p_str.split(delim) if n.strip()]
            else:
                nodes = p_str.split()
            parsed_paths.append(nodes)
        candidate_paths = parsed_paths

    if candidate_paths is not None or args.battery is not None:
        result = run_utility_based_simulation(
            paths=candidate_paths,
            battery_limit=args.battery,
        )
        print_graph_summary(DEFAULT_GRAPH)
        print_evaluation_table(
            result.evaluated_paths,
            result.selected_path,
            battery_limit=args.battery,
        )
        if result.selected_path:
            print(f"\nSelected Path: {result.selected_path.path_str}")
            print(f"Utility:       {result.selected_path.utility}")
            print(f"Total Cost:    {result.selected_path.total_cost}")
        else:
            print("\nNo valid/feasible path found.")
        return

    # Default run
    run_default_demonstration()


if __name__ == "__main__":
    main()
