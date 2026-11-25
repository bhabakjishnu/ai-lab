"""Top-level command line interface for AI Lab simulations and utilities.

Allows running simulations directly via module syntax:
    python -m src vacuum --steps 5
    python -m src randomized --seed 58
    python -m src model-based
    python -m src goal-based
    python -m src utility-based
    python -m src multi-agent --steps 10
    python -m src stochastic --runs 20 --seed 28
    python -m src environment-classifier --input True,False,False,False,True
    python -m src agent-evaluator --events move,move,bump,move,deliver
    python -m src state-space --table
    python -m src bfs --start A
    python -m src ucs --start A --goal C
    python -m src maze
    python -m src smoke-test
"""

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point dispatcher for AI Lab simulations."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="AI Lab: Intelligent Agent Simulations & Experiment Runners",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Vacuum agent runner
    subparsers.add_parser("vacuum", help="Run Question 1: Simple Reflex Vacuum Cleaner Agent")

    # Randomized reflex agent runner
    subparsers.add_parser("randomized", help="Run Question 2: Randomized Reflex Agent (1D Grid)")

    # Model-based reflex agent runner
    subparsers.add_parser("model-based", help="Run Question 3: Model-Based Reflex Agent (2x2 Grid)")

    # Goal-based navigation agent runner
    subparsers.add_parser("goal-based", help="Run Question 4: Goal-Based Navigation Agent (5x5 Grid)")

    # Utility-based agent runner
    subparsers.add_parser("utility-based", help="Run Question 5: Utility-Based Agent Simulation")

    # Multi-agent simulation runner
    subparsers.add_parser("multi-agent", help="Run Question 6: Multi-Agent Environment Simulation (4x4 Grid)")

    # Stochastic environment vacuum agent runner
    subparsers.add_parser("stochastic", help="Run Question 7: Stochastic Environment Vacuum Agent Simulation")

    # Environment type classifier runner
    subparsers.add_parser(
        "environment-classifier",
        aliases=["env-classifier"],
        help="Run Question 8: Environment Type Classifier (PEAS & Knowledge Representation)",
    )

    # Agent performance measure evaluator runner
    subparsers.add_parser(
        "agent-evaluator",
        aliases=["drone-evaluator", "perf-evaluator"],
        help="Run Question 9: Agent Performance Measure Evaluator (PEAS & Rationality)",
    )

    # State-space visualizer runner
    subparsers.add_parser(
        "state-space",
        aliases=["state-visualizer", "state-space-visualizer"],
        help="Run Question 10: State-Space Visualizer for Reflex Agent (Graph & Cycles)",
    )

    # Breadth-first search runner
    subparsers.add_parser(
        "bfs",
        aliases=["breadth-first-search", "bfs-search"],
        help="Run Question 11: Breadth-First Search (BFS) on User-Defined Graph",
    )

    # Depth-first search runner
    subparsers.add_parser(
        "dfs",
        aliases=["depth-first-search", "dfs-search"],
        help="Run Question 12: Depth-First Search (DFS) for Graph Traversal",
    )

    # Uniform cost search runner
    subparsers.add_parser(
        "ucs",
        aliases=["uniform-cost-search", "ucs-search"],
        help="Run Question 14: Uniform Cost Search (UCS) on Weighted Directed Graph",
    )

    # BFS 2D Maze Solver runner
    subparsers.add_parser(
        "maze",
        aliases=["bfs-maze", "maze-solver", "bfs-maze-solver"],
        help="Run Question 15: Breadth-First Search (BFS) 2D Grid Maze Solver",
    )

    # Depth-limited search runner
    subparsers.add_parser(
        "dls",
        aliases=["depth-limited-search", "dls-search"],
        help="Run Question 16: Depth-Limited Search (DLS)",
    )

    # Iterative deepening search runner
    subparsers.add_parser(
        "ids",
        aliases=["iterative-deepening-search", "ids-search"],
        help="Run Question 17: Iterative Deepening Search (IDS)",
    )

    # BFS vs DFS Performance Analyzer runner
    subparsers.add_parser(
        "bfs-dfs",
        aliases=["bfs-vs-dfs", "performance-analyzer", "maze-analyzer", "maze-compare"],
        help="Run Question 18: BFS vs DFS Performance Analyzer on 10x10 Mazes",
    )

    # Bidirectional Search runner
    subparsers.add_parser(
        "bidirectional",
        aliases=["bidirectional-search", "bds", "bds-search"],
        help="Run Question 19: Bidirectional Search Simulation (Simultaneous BFS)",
    )

    # Water Jug Problem runner
    subparsers.add_parser(
        "water-jug",
        aliases=["water-jug-bfs", "jug-bfs", "waterjug"],
        help="Run Question 20: Water Jug Problem using BFS (Implicit Graph Search)",
    )

    # Missionaries and Cannibals problem runner
    subparsers.add_parser(
        "missionaries-cannibals",
        aliases=["missionaries-and-cannibals", "mc-dfs", "mc"],
        help="Run Question 21: Missionaries and Cannibals Problem using DFS",
    )

    # UCS Grid Navigation runner
    subparsers.add_parser(
        "ucs-grid",
        aliases=["ucs-grid-nav", "grid-ucs", "topological-ucs", "ucs22"],
        help="Run Question 22: Uniform Cost Search (UCS) Grid Navigation with Cost Modifiers",
    )

    # Multi-Goal BFS Maze runner
    subparsers.add_parser(
        "multi-goal-bfs",
        aliases=["maze-multiple-exits", "multi-exit-maze", "multi-goal-maze", "maze23"],
        help="Run Question 23: Maze with Multiple Exits (Multi-Goal BFS)",
    )

    # Manhattan and Euclidean Distance Calculators runner
    subparsers.add_parser(
        "distance-calculators",
        aliases=["distances", "manhattan-euclidean", "distance-metrics", "q24"],
        help="Run Question 24: Manhattan and Euclidean Distance Calculators (Informed Search & Heuristics)",
    )

    # 8-Puzzle Heuristic Evaluator runner
    subparsers.add_parser(
        "eight-puzzle",
        aliases=["8-puzzle", "eight-puzzle-heuristics", "puzzle-heuristics", "q25"],
        help="Run Question 25: 8-Puzzle Heuristic Evaluator (h1 Misplaced Tiles & h2 Manhattan Distance)",
    )

    # Greedy Best-First Search runner
    subparsers.add_parser(
        "gbfs",
        aliases=["greedy", "greedy-best-first-search", "best-first-search", "q26"],
        help="Run Question 26: Greedy Best-First Search on a Graph (Priority Queue & Pure Heuristics)",
    )

    # Greedy Best-First Search Grid Navigator runner
    subparsers.add_parser(
        "gbfs-grid",
        aliases=["gbfs-grid-nav", "greedy-grid", "gbfs-grid-navigator", "q27"],
        help="Run Question 27: Greedy Best-First Search Grid Navigator (10x10 Maze)",
    )

    # A* Search on a Graph runner
    subparsers.add_parser(
        "a-star",
        aliases=["astar", "a-star-search", "astar-search", "q28"],
        help="Run Question 28: A* Search Algorithm on a Weighted Graph (f(n) = g(n) + h(n))",
    )

    # A* Pathfinding in 2D Maze runner
    subparsers.add_parser(
        "a-star-maze",
        aliases=["astar-maze", "a-star-maze-solver", "astar-2d-maze", "q29"],
        help="Run Question 29: A* Pathfinding in 2D Maze Simulation (f(n) = g(n) + h(n))",
    )

    # 8-Puzzle A* Solver runner
    subparsers.add_parser(
        "eight-puzzle-solver",
        aliases=["8-puzzle-solver", "astar-puzzle", "astar-8-puzzle", "eight-puzzle-astar", "q30"],
        help="Run Question 30: 8-Puzzle Solver using A* and Manhattan Distance",
    )

    # 8-Puzzle Heuristic Dominance (Manhattan vs Misplaced Tiles) runner
    subparsers.add_parser(
        "eight-puzzle-dominance",
        aliases=[
            "eight-puzzle-comparison",
            "manhattan-vs-misplaced",
            "puzzle-dominance",
            "q31",
            "question31",
        ],
        help="Run Question 31: 8-Puzzle Manhattan vs Misplaced Tiles Dominance Analysis",
    )

    # UCS vs A* Performance Comparison runner
    subparsers.add_parser(
        "ucs-vs-astar",
        aliases=[
            "ucs-astar",
            "ucs-astar-comparison",
            "ucs-vs-astar-comparison",
            "q32",
            "question32",
        ],
        help="Run Question 32: UCS vs A* (Manhattan) Performance Comparison on 20x20 Grid",
    )

    # Starter smoke test runner
    subparsers.add_parser("smoke-test", help="Run starter experiment runner smoke test")

    if argv is None:
        argv = sys.argv[1:]

    # If first arg matches a subcommand, forward remaining args to that script
    allowed_subcommands = {
        "vacuum",
        "randomized",
        "model-based",
        "goal-based",
        "utility-based",
        "multi-agent",
        "stochastic",
        "environment-classifier",
        "env-classifier",
        "agent-evaluator",
        "drone-evaluator",
        "perf-evaluator",
        "state-space",
        "state-visualizer",
        "state-space-visualizer",
        "bfs",
        "breadth-first-search",
        "bfs-search",
        "dfs",
        "depth-first-search",
        "dfs-search",
        "ucs",
        "uniform-cost-search",
        "ucs-search",
        "maze",
        "bfs-maze",
        "maze-solver",
        "bfs-maze-solver",
        "dls",
        "depth-limited-search",
        "dls-search",
        "ids",
        "iterative-deepening-search",
        "ids-search",
        "bfs-dfs",
        "bfs-vs-dfs",
        "performance-analyzer",
        "maze-analyzer",
        "maze-compare",
        "bidirectional",
        "bidirectional-search",
        "bds",
        "bds-search",
        "water-jug",
        "water-jug-bfs",
        "jug-bfs",
        "waterjug",
        "missionaries-cannibals",
        "missionaries-and-cannibals",
        "mc-dfs",
        "mc",
        "ucs-grid",
        "ucs-grid-nav",
        "grid-ucs",
        "topological-ucs",
        "ucs22",
        "multi-goal-bfs",
        "maze-multiple-exits",
        "multi-exit-maze",
        "multi-goal-maze",
        "maze23",
        "distance-calculators",
        "distances",
        "manhattan-euclidean",
        "distance-metrics",
        "q24",
        "eight-puzzle",
        "8-puzzle",
        "eight-puzzle-heuristics",
        "puzzle-heuristics",
        "q25",
        "gbfs",
        "greedy",
        "greedy-best-first-search",
        "best-first-search",
        "q26",
        "gbfs-grid",
        "gbfs-grid-nav",
        "greedy-grid",
        "gbfs-grid-navigator",
        "q27",
        "a-star",
        "astar",
        "a-star-search",
        "astar-search",
        "q28",
        "a-star-maze",
        "astar-maze",
        "a-star-maze-solver",
        "astar-2d-maze",
        "q29",
        "eight-puzzle-solver",
        "8-puzzle-solver",
        "astar-puzzle",
        "astar-8-puzzle",
        "eight-puzzle-astar",
        "q30",
        "eight-puzzle-dominance",
        "eight-puzzle-comparison",
        "manhattan-vs-misplaced",
        "puzzle-dominance",
        "q31",
        "question31",
        "ucs-vs-astar",
        "ucs-astar",
        "ucs-astar-comparison",
        "ucs-vs-astar-comparison",
        "q32",
        "question32",
        "smoke-test",
    }
    if argv and argv[0] in allowed_subcommands:
        cmd = argv[0]
        remaining = argv[1:]
        if cmd == "vacuum":
            from scripts.run_vacuum_agent import main as vacuum_main
            sys.argv = [sys.argv[0]] + remaining
            vacuum_main()
        elif cmd == "randomized":
            from scripts.run_randomized_agent import main as randomized_main
            sys.argv = [sys.argv[0]] + remaining
            randomized_main()
        elif cmd == "model-based":
            from scripts.run_model_based_agent import main as model_main
            sys.argv = [sys.argv[0]] + remaining
            model_main()
        elif cmd == "goal-based":
            from scripts.run_goal_based_agent import main as goal_main
            sys.argv = [sys.argv[0]] + remaining
            goal_main()
        elif cmd == "utility-based":
            from scripts.run_utility_based_agent import main as utility_main
            sys.argv = [sys.argv[0]] + remaining
            utility_main()
        elif cmd == "multi-agent":
            from scripts.run_multi_agent_simulation import main as multi_agent_main
            sys.argv = [sys.argv[0]] + remaining
            multi_agent_main()
        elif cmd == "stochastic":
            from scripts.run_stochastic_vacuum_agent import main as stochastic_main
            sys.argv = [sys.argv[0]] + remaining
            stochastic_main()
        elif cmd in ("environment-classifier", "env-classifier"):
            from scripts.run_environment_classifier import main as env_main
            sys.argv = [sys.argv[0]] + remaining
            env_main()
        elif cmd in ("agent-evaluator", "drone-evaluator", "perf-evaluator"):
            from scripts.run_agent_performance_evaluator import main as agent_eval_main
            sys.argv = [sys.argv[0]] + remaining
            agent_eval_main()
        elif cmd in ("state-space", "state-visualizer", "state-space-visualizer"):
            from scripts.run_state_space_visualizer import main as state_space_main
            sys.argv = [sys.argv[0]] + remaining
            state_space_main()
        elif cmd in ("bfs", "breadth-first-search", "bfs-search"):
            from scripts.run_breadth_first_search import main as bfs_main
            sys.argv = [sys.argv[0]] + remaining
            bfs_main()
        elif cmd in ("dfs", "depth-first-search", "dfs-search"):
            from scripts.run_depth_first_search import main as dfs_main
            sys.argv = [sys.argv[0]] + remaining
            dfs_main()
        elif cmd in ("ucs", "uniform-cost-search", "ucs-search"):
            from scripts.run_uniform_cost_search import main as ucs_main
            sys.argv = [sys.argv[0]] + remaining
            ucs_main()
        elif cmd in ("maze", "bfs-maze", "maze-solver", "bfs-maze-solver"):
            from scripts.run_bfs_maze_solver import main as maze_main
            sys.argv = [sys.argv[0]] + remaining
            maze_main()
        elif cmd in ("dls", "depth-limited-search", "dls-search"):
            from scripts.run_depth_limited_search import main as dls_main
            sys.argv = [sys.argv[0]] + remaining
            dls_main()
        elif cmd in ("ids", "iterative-deepening-search", "ids-search"):
            from scripts.run_iterative_deepening_search import main as ids_main
            sys.argv = [sys.argv[0]] + remaining
            ids_main()
        elif cmd in ("bfs-dfs", "bfs-vs-dfs", "performance-analyzer", "maze-analyzer", "maze-compare"):
            from scripts.run_bfs_dfs_performance_analyzer import main as bfs_dfs_main
            sys.argv = [sys.argv[0]] + remaining
            bfs_dfs_main()
        elif cmd in ("bidirectional", "bidirectional-search", "bds", "bds-search"):
            from scripts.run_bidirectional_search import main as bds_main
            sys.argv = [sys.argv[0]] + remaining
            bds_main()
        elif cmd in ("water-jug", "water-jug-bfs", "jug-bfs", "waterjug"):
            from scripts.run_water_jug_bfs import main as water_jug_main
            sys.argv = [sys.argv[0]] + remaining
            water_jug_main()
        elif cmd in ("missionaries-cannibals", "missionaries-and-cannibals", "mc-dfs", "mc"):
            from scripts.run_missionaries_cannibals_dfs import main as mc_main
            sys.argv = [sys.argv[0]] + remaining
            mc_main()
        elif cmd in ("ucs-grid", "ucs-grid-nav", "grid-ucs", "topological-ucs", "ucs22"):
            from scripts.run_ucs_grid_navigation import main as ucs_grid_main
            sys.argv = [sys.argv[0]] + remaining
            ucs_grid_main()
        elif cmd in ("multi-goal-bfs", "maze-multiple-exits", "multi-exit-maze", "multi-goal-maze", "maze23"):
            from scripts.run_multi_goal_bfs_maze import main as multi_goal_bfs_main
            sys.argv = [sys.argv[0]] + remaining
            multi_goal_bfs_main()
        elif cmd in ("distance-calculators", "distances", "manhattan-euclidean", "distance-metrics", "q24"):
            from scripts.run_distance_calculators import main as dist_main
            sys.argv = [sys.argv[0]] + remaining
            dist_main()
        elif cmd in ("eight-puzzle", "8-puzzle", "eight-puzzle-heuristics", "puzzle-heuristics", "q25"):
            from scripts.run_eight_puzzle_heuristics import main as eight_puzzle_main
            sys.argv = [sys.argv[0]] + remaining
            eight_puzzle_main()
        elif cmd in ("gbfs", "greedy", "greedy-best-first-search", "best-first-search", "q26"):
            from scripts.run_greedy_best_first_search import main as gbfs_main
            sys.argv = [sys.argv[0]] + remaining
            gbfs_main()
        elif cmd in ("gbfs-grid", "gbfs-grid-nav", "greedy-grid", "gbfs-grid-navigator", "q27"):
            from scripts.run_gbfs_grid_navigator import main as gbfs_grid_main
            sys.argv = [sys.argv[0]] + remaining
            gbfs_grid_main()
        elif cmd in ("a-star", "astar", "a-star-search", "astar-search", "q28"):
            from scripts.run_a_star_search import main as astar_main
            sys.argv = [sys.argv[0]] + remaining
            astar_main()
        elif cmd in ("a-star-maze", "astar-maze", "a-star-maze-solver", "astar-2d-maze", "q29"):
            from scripts.run_a_star_maze_solver import main as astar_maze_main
            sys.argv = [sys.argv[0]] + remaining
            astar_maze_main()
        elif cmd in (
            "eight-puzzle-solver",
            "8-puzzle-solver",
            "astar-puzzle",
            "astar-8-puzzle",
            "eight-puzzle-astar",
            "q30",
        ):
            from scripts.run_a_star_eight_puzzle import main as astar_puzzle_main
            sys.argv = [sys.argv[0]] + remaining
            astar_puzzle_main()
        elif cmd in (
            "eight-puzzle-dominance",
            "eight-puzzle-comparison",
            "manhattan-vs-misplaced",
            "puzzle-dominance",
            "q31",
            "question31",
        ):
            from scripts.run_eight_puzzle_manhattan_vs_misplaced import main as dominance_main
            sys.argv = [sys.argv[0]] + remaining
            dominance_main()
        elif cmd in (
            "ucs-vs-astar",
            "ucs-astar",
            "ucs-astar-comparison",
            "ucs-vs-astar-comparison",
            "q32",
            "question32",
        ):
            from scripts.run_ucs_vs_astar_comparison import main as ucs_astar_main
            sys.argv = [sys.argv[0]] + remaining
            ucs_astar_main()
        elif cmd == "smoke-test":
            from scripts.smoke_test import main as smoke_main
            sys.argv = [sys.argv[0]] + remaining
            smoke_main()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
