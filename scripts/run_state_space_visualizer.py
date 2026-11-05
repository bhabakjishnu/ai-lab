"""Runner script for State-Space Visualizer (Question 10).

Models a 2-room vacuum cleaner agent's state space, generates all 8 states
mathematically, links them via valid actions (Left, Right, Suck), detects
cyclic transitions, and visualizes the state-space graph using matplotlib and networkx.

Usage Examples:
    # 1. Default Run (Display Matplotlib graph window and ASCII summary):
    python scripts/run_state_space_visualizer.py

    # 2. Display graph window explicitly:
    python scripts/run_state_space_visualizer.py --show

    # 3. Save graph visualization to an image file:
    python scripts/run_state_space_visualizer.py --save reports/state_space_graph.png

    # 4. Use different layouts (textbook, circular, spring):
    python scripts/run_state_space_visualizer.py --layout circular --save reports/state_space_circular.png

    # 5. Print comprehensive state-space transition table:
    python scripts/run_state_space_visualizer.py --table

    # 6. Display 8x8 adjacency matrix:
    python scripts/run_state_space_visualizer.py --matrix

    # 7. Perform cycle detection analysis:
    python scripts/run_state_space_visualizer.py --cycles

    # 8. Compute optimal shortest paths to goal states from all starting states:
    python scripts/run_state_space_visualizer.py --search

    # 9. Run complete automated demonstration benchmark:
    python scripts/run_state_space_visualizer.py --demo

    # 10. Display Viva Voce theory & reference study guide:
    python scripts/run_state_space_visualizer.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.state_space_visualizer import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SUCK,
    StateSpace,
    StateSpaceVisualizer,
    VacuumState,
    get_viva_reference,
)


def print_state_table(state_space: StateSpace) -> None:
    """Print the complete state-space transition table."""
    visualizer = StateSpaceVisualizer(state_space)
    print(visualizer.render_ascii_graph())


def print_adjacency_matrix(state_space: StateSpace) -> None:
    """Display the |S| x |S| adjacency matrix of the state-space graph."""
    labels, matrix = state_space.to_adjacency_matrix()
    divider = "=" * 76
    sub_divider = "-" * 76
    print(divider)
    print("           8x8 ADJACENCY MATRIX OF STATE-SPACE GRAPH (O(|S|^2))")
    print(divider)

    header = "State  " + "".join(f"{s[:2]:>7}" for s in labels)
    print(header)
    print(sub_divider)

    for i, row in enumerate(matrix):
        row_str = "".join(f"{val:>7}" for val in row)
        print(f"{labels[i][:6]:<7}{row_str}")

    print(sub_divider)
    print("  * Note: Non-zero diagonal entries indicate self-loop cyclic transitions.")
    print(divider)


def print_cycle_analysis(state_space: StateSpace) -> None:
    """Print detailed audit of cyclic transitions in the state space."""
    cycle_info = state_space.identify_cycles()
    divider = "=" * 76
    sub_divider = "-" * 76
    print(divider)
    print("                CYCLIC TRANSITIONS & SELF-LOOP AUDIT")
    print(divider)
    print(f"  Total Valid Transitions Evaluated : {cycle_info['total_transitions']}")
    print(f"  Self-Loop Cycles Result(s, a) == s: {cycle_info['total_self_loops']} (50% of total)")
    print(f"  State-Changing Transitions        : {cycle_info['state_changing_transitions']} (50% of total)")
    print(sub_divider)
    print("  Breakdown of Self-Loops by Action:")
    for action, count in cycle_info["self_loops_by_action"].items():
        print(f"    - Action '{action:<5}': {count} self-loops")
    print(sub_divider)
    print("  Detailed List of Identified Cyclic Transitions:")
    for record in cycle_info["self_loop_records"]:
        print(f"    [{record.source.short_label}] + '{record.action}' -> {record.description}")
    print(divider)


def print_shortest_paths(state_space: StateSpace) -> None:
    """Compute and display shortest paths to nearest goal state for all 8 states."""
    divider = "=" * 76
    sub_divider = "-" * 76
    print(divider)
    print("      SHORTEST OPTIMAL ACTION PATHS TO GOAL (BREADTH-FIRST SEARCH)")
    print(divider)
    print(f"{'Start State':<18}{'Steps':<8}{'Action Sequence':<30}{'Goal Reached'}")
    print(sub_divider)

    for s in state_space.states:
        res = state_space.find_shortest_paths_to_goal(s)
        actions_str = " -> ".join(res["actions"]) if res["actions"] else "(Already at Goal)"
        goal_label = res["goal"].short_label if res["goal"] else "N/A"
        print(f"{s.short_label:<18}{res['distance']:<8}{actions_str:<30}{goal_label}")

    print(divider)


def run_demonstration(state_space: StateSpace, save_path: Optional[str] = None) -> None:
    """Execute complete demonstration suite for Question 10."""
    print("\n" + "=" * 76)
    print("    QUESTION NO. 10: STATE-SPACE VISUALIZER FOR REFLEX VACUUM AGENT")
    print("    Syllabus Area: Intelligent Agents | Practical Type: Visualization")
    print("=" * 76)

    print("\n[Step 1: Combinatorial State Generation]")
    print(f"Generated {len(state_space.states)} states mathematically (|S| = 2 * 2^2 = 8).")
    for s in state_space.states:
        goal_tag = " -> [GOAL STATE: Both Rooms Clean]" if s.is_goal else ""
        print(f"  - State {s.state_id}: Loc = {s.agent_loc}, Room A = {s.room_a_status:<5}, Room B = {s.room_b_status:<5}{goal_tag}")

    print("\n[Step 2: State-Space Transition Table & Edge Mapping]")
    print_state_table(state_space)

    print("\n[Step 3: Cyclic Transitions & Self-Loop Detection]")
    print_cycle_analysis(state_space)

    print("\n[Step 4: Shortest Path Optimal Search]")
    print_shortest_paths(state_space)

    print("\n[Step 5: Directed Graph Visualization using NetworkX & Matplotlib]")
    viz = StateSpaceVisualizer(state_space)
    out_file = save_path or "reports/state_space_graph.png"
    fig, _ = viz.plot_graph(
        layout_type="textbook",
        highlight_goals=True,
        highlight_cycles=True,
        save_path=out_file,
        show=False,
    )
    print(f"State-space graph rendered and saved to: {out_file}")
    print("Goal states (S7, S8) highlighted in Emerald Green [Optional Challenge Completed].")
    print("=" * 76 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI dispatcher for the State-Space Visualizer."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_state_space_visualizer.py",
        description="State-Space Visualizer for 2-Room Reflex Vacuum Agent (Question 10)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive Matplotlib window displaying the 8-state directed graph.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILEPATH",
        help="Save state-space graph figure to file (e.g. reports/state_space_graph.png).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["textbook", "circular", "spring", "shell"],
        default="textbook",
        help="Graph layout algorithm (default: 'textbook' 2-column grid layout).",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print tabular ASCII state-space transition mapping.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Print the 8x8 state adjacency matrix.",
    )
    parser.add_argument(
        "--cycles",
        action="store_true",
        help="Run comprehensive cycle detection and self-loop audit.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Calculate optimal shortest path action sequences to goal states.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demonstration benchmark suite covering all requirements.",
    )
    parser.add_argument(
        "--no-highlight-goals",
        action="store_true",
        help="Disable emerald green node highlighting for goal states.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce theory and reference study guide.",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print(get_viva_reference())
        return

    state_space = StateSpace()
    viz = StateSpaceVisualizer(state_space)

    if args.demo:
        run_demonstration(state_space, save_path=args.save)
        if args.show:
            viz.plot_graph(
                layout_type=args.layout,
                highlight_goals=not args.no_highlight_goals,
                show=True,
            )
        return

    if args.table:
        print_state_table(state_space)
        return

    if args.matrix:
        print_adjacency_matrix(state_space)
        return

    if args.cycles:
        print_cycle_analysis(state_space)
        return

    if args.search:
        print_shortest_paths(state_space)
        return

    # Default action: render ASCII summary and generate/display the graph
    print_state_table(state_space)
    save_file = args.save or "reports/state_space_graph.png"
    fig, _ = viz.plot_graph(
        layout_type=args.layout,
        highlight_goals=not args.no_highlight_goals,
        save_path=save_file,
        show=args.show,
    )
    print(f"\n[Visual Output] State-space graph generated and saved to: {save_file}")
    if args.show:
        print("Interactive Matplotlib window displayed.")


if __name__ == "__main__":
    main()
