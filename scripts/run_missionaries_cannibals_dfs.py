"""Runner script for Missionaries and Cannibals Problem using Depth-First Search (Question 21).

Solves the classic Missionaries and Cannibals river-crossing problem using Depth-First
Search (DFS) with cycle detection over an implicitly generated state space.

Usage Examples:
    # 1. Default Standard Execution (DFS Iterative, 3M 3C):
    python scripts/run_missionaries_cannibals_dfs.py

    # 2. Table Output Format:
    python scripts/run_missionaries_cannibals_dfs.py --format table

    # 3. Compact Arrow Trace:
    python scripts/run_missionaries_cannibals_dfs.py --format arrows

    # 4. Detailed ASCII River View:
    python scripts/run_missionaries_cannibals_dfs.py --format detailed

    # 5. Recursive DFS Mode:
    python scripts/run_missionaries_cannibals_dfs.py --method recursive

    # 6. Compare Constraint Validation Strategies (Generation vs Testing):
    python scripts/run_missionaries_cannibals_dfs.py --compare-validation

    # 7. Compare DFS vs BFS (Optimality & Node Expansions):
    python scripts/run_missionaries_cannibals_dfs.py --compare-bfs

    # 8. State Space Combinatorial Analysis:
    python scripts/run_missionaries_cannibals_dfs.py --state-space

    # 9. Optional Challenge: Text Animation with Delays:
    python scripts/run_missionaries_cannibals_dfs.py --animate --delay 0.4

    # 10. Display University Viva Voce Theory Guide:
    python scripts/run_missionaries_cannibals_dfs.py --viva

    # 11. Run Comprehensive Automated Laboratory Demo:
    python scripts/run_missionaries_cannibals_dfs.py --demo
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.missionaries_cannibals_dfs import (
    DEFAULT_BOAT_CAPACITY,
    DEFAULT_TOTAL_CANNIBALS,
    DEFAULT_TOTAL_MISSIONARIES,
    INITIAL_STATE,
    GOAL_STATE,
    MissionariesCannibalsResult,
    MissionariesCannibalsSolver,
    animate_solution,
    analyze_state_space,
    format_solution_trace,
    get_missionaries_cannibals_viva_reference,
    solve_missionaries_cannibals_dfs,
)


def print_banner(title: str) -> None:
    """Print a styled laboratory section header."""
    width = 76
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_viva_guide() -> None:
    """Print the complete Viva Voce examination reference guide."""
    viva = get_missionaries_cannibals_viva_reference()
    print_banner("QUESTION NO. 21: MISSIONARIES & CANNIBALS VIVA VOCE THEORY GUIDE")

    for topic_title, content in viva.items():
        print(f"[{topic_title}]")
        print(f"Question: {content['question']}")
        print("Answer:")
        for line in content["answer"].split("\n"):
            cleaned = line.strip()
            if cleaned:
                print(f"  • {cleaned}")
        print("-" * 76)


def print_state_space_analysis() -> None:
    """Display exhaustive state-space metrics and reachability breakdown."""
    print_banner("STATE SPACE COMBINATORIAL ANALYSIS")
    analysis = analyze_state_space()

    print(
        f"1. Theoretical Unconstrained States: {analysis['total_theoretical_states']}\n"
        f"   Formula: (Total_M + 1) * (Total_C + 1) * 2 = 4 * 4 * 2 = 32 states\n\n"
        f"2. Valid (Safe) States: {analysis['total_safe_states']} / 32 states\n"
        f"   States where M == 0 or M >= C on both left and right riverbanks.\n\n"
        f"3. Reachable Safe States from (3, 3, 'Left'): {analysis['total_reachable_safe_states']}\n"
        f"   Reachable states connected via valid 1-person or 2-person boat transitions:\n"
    )

    for i, st in enumerate(analysis["reachable_safe_states"], 1):
        m_r = DEFAULT_TOTAL_MISSIONARIES - st[0]
        c_r = DEFAULT_TOTAL_CANNIBALS - st[1]
        print(f"   [{i:02d}] State: ({st[0]}, {st[1]}, '{st[2]}')  -->  Right Bank: ({m_r}M, {c_r}C)")

    print(f"\n4. Unreachable Safe States ({analysis['total_unreachable_safe_states']}):")
    for st in analysis["unreachable_safe_states"]:
        print(f"   • {st} (Violates boat crossing parity)")


def compare_dfs_and_bfs() -> None:
    """Run empirical comparison between DFS and BFS for Viva topic evaluation."""
    print_banner("ALGORITHM COMPARISON: DEPTH-FIRST SEARCH (DFS) vs BREADTH-FIRST SEARCH (BFS)")
    solver = MissionariesCannibalsSolver()
    cmp_data = solver.compare_dfs_vs_bfs()

    dfs = cmp_data["dfs"]
    bfs = cmp_data["bfs"]

    print(
        f"+------------------------------+--------------------+--------------------+\n"
        f"| Metric                       | DFS (LIFO Stack)   | BFS (FIFO Queue)   |\n"
        f"+------------------------------+--------------------+--------------------+\n"
        f"| Solution Found               | {str(dfs['success']):<18} | {str(bfs['success']):<18} |\n"
        f"| Solution Path Length (Trips) | {dfs['trips']:<18} | {bfs['trips']:<18} |\n"
        f"| Is Path Shortest (Optimal)?  | {str(dfs['optimal']):<18} | {str(bfs['optimal']):<18} |\n"
        f"| States Explored              | {dfs['states_explored']:<18} | {bfs['states_explored']:<18} |\n"
        f"| States Generated             | {dfs['states_generated']:<18} | {bfs['states_generated']:<18} |\n"
        f"| Execution Time (ms)          | {dfs['execution_time_ms']:<18.3f} | {bfs['execution_time_ms']:<18.3f} |\n"
        f"+------------------------------+--------------------+--------------------+\n"
    )
    print("Viva Examination Note:")
    print(f"  {cmp_data['optimality_notes']}")


def compare_validation_strategies() -> None:
    """Compare constraint validation during generation vs during testing."""
    print_banner("CONSTRAINT VALIDATION: GENERATION vs TESTING")
    solver = MissionariesCannibalsSolver()

    res_gen = solver.solve_dfs(validation_mode="generation")
    res_test = solver.solve_dfs(validation_mode="testing")

    print(
        f"+------------------------------+--------------------+--------------------+\n"
        f"| Metric                       | During Generation  | During Testing     |\n"
        f"+------------------------------+--------------------+--------------------+\n"
        f"| Strategy                     | Forward Checking   | Generate-and-Test  |\n"
        f"| Success                      | {str(res_gen.success):<18} | {str(res_test.success):<18} |\n"
        f"| Total Boat Trips             | {res_gen.total_trips:<18} | {res_test.total_trips:<18} |\n"
        f"| States Explored              | {res_gen.states_explored:<18} | {res_test.states_explored:<18} |\n"
        f"| Pruned by Capacity Bounds    | {res_gen.pruned_by_capacity:<18} | {res_test.pruned_by_capacity:<18} |\n"
        f"| Pruned by Safety Constraint  | {res_gen.pruned_by_safety:<18} | {res_test.pruned_by_safety:<18} |\n"
        f"| Pruned by Cycle Detection    | {res_gen.pruned_by_cycle:<18} | {res_test.pruned_by_cycle:<18} |\n"
        f"+------------------------------+--------------------+--------------------+\n"
    )
    print(
        "Analysis:\n"
        "  Evaluating safety constraints during state generation eliminates invalid states\n"
        "  before pushing them to the search stack, drastically pruning the search branching factor."
    )


def run_laboratory_demo() -> None:
    """Run full automated demonstration covering all features and theoretical topics."""
    print_banner("QUESTION NO. 21: MISSIONARIES & CANNIBALS FULL LABORATORY DEMO")

    print("--- 1. Standard DFS Trace ---")
    solver = MissionariesCannibalsSolver()
    result = solver.solve_dfs()
    print(format_solution_trace(result, format_type="standard"))

    print("\n--- 2. Formatted Table Trace ---")
    print(format_solution_trace(result, format_type="table"))

    print("\n--- 3. Compact Arrow Transition Trace ---")
    print(format_solution_trace(result, format_type="arrows"))

    compare_validation_strategies()
    compare_dfs_and_bfs()
    print_state_space_analysis()


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for Question 21 Missionaries and Cannibals runner."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_missionaries_cannibals_dfs.py",
        description="Missionaries and Cannibals Problem Solver using Depth-First Search (Question 21).",
    )
    parser.add_argument(
        "--format",
        choices=["standard", "table", "arrows", "detailed"],
        default="standard",
        help="Output display format: standard (default), table, arrows, or detailed.",
    )
    parser.add_argument(
        "--method",
        choices=["iterative", "recursive"],
        default="iterative",
        help="DFS implementation method: iterative (LIFO stack, default) or recursive.",
    )
    parser.add_argument(
        "--validation",
        choices=["generation", "testing"],
        default="generation",
        help="Constraint validation timing: generation (forward check) or testing (generate-and-test).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Run terminal text animation with delays (Optional Challenge).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay in seconds between animation frames (default: 0.35).",
    )
    parser.add_argument(
        "--compare-bfs",
        action="store_true",
        help="Compare DFS vs BFS performance, path optimality, and node expansions.",
    )
    parser.add_argument(
        "--compare-validation",
        action="store_true",
        help="Compare constraint validation during generation vs during testing.",
    )
    parser.add_argument(
        "--state-space",
        action="store_true",
        help="Display combinatorial state space size and reachability analysis.",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display the complete Viva Voce examination reference guide.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive automated laboratory demonstration suite.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in machine-readable JSON format.",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print_viva_guide()
        return 0

    if args.state_space:
        print_state_space_analysis()
        return 0

    if args.compare_bfs:
        compare_dfs_and_bfs()
        return 0

    if args.compare_validation:
        compare_validation_strategies()
        return 0

    if args.demo:
        run_laboratory_demo()
        return 0

    solver = MissionariesCannibalsSolver()
    result = solver.solve_dfs(method=args.method, validation_mode=args.validation)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.animate:
        animate_solution(result, delay=args.delay)
        return 0

    print_banner(
        f"QUESTION NO. 21: MISSIONARIES & CANNIBALS PROBLEM (DFS {args.method.upper()})"
    )
    print(format_solution_trace(result, format_type=args.format))
    print(f"\nExecution Metrics:")
    print(f"  • States Explored : {result.states_explored}")
    print(f"  • States Generated: {result.states_generated}")
    print(f"  • Cycles Pruned   : {result.pruned_by_cycle}")
    print(f"  • Execution Time  : {result.execution_time_ms:.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
