# AI Lab

A structured workspace for prototyping, experimenting with AI ideas, scripts, and models.

[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Testing](https://img.shields.io/badge/Tests-unittest%20%7C%20pytest-2BAF2B?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Mermaid](https://img.shields.io/badge/Diagrams-Mermaid-FF3670?logo=mermaid&logoColor=white)](https://mermaid.js.org/)

## Tech Stack

| Layer | Tool |
| --- | --- |
| Language | Python 3 |
| Prototyping & Exploration | Jupyter Notebook (`.ipynb`) |
| Core Modules | `src/` (Clean, modular experiment runners) |
| Unit Testing | `unittest` / `pytest` in `tests/` |
| Data Layer | `data/raw/` and `data/processed/` |
| Diagrams | Mermaid in this README |

## Architecture

```mermaid
flowchart TD
  User([Developer]) --> Workflow{Workflow}
  Workflow -->|Interactive Experimentation| NB[notebooks/starter_experiment.ipynb]
  Workflow -->|Script Execution| Script[scripts/smoke_test.py]
  Workflow -->|Automated Validation| Tests[tests/test_experiment.py]
  
  NB --> Core[src/experiment.py]
  Script --> Core
  Tests --> Core
  
  Core --> Data[(data/raw & data/processed)]
```

## Project Layout

```text
ai-lab/
├── notebooks/
│   ├── goal_based_navigation_agent.ipynb# Question 4: Goal-Based Navigation Agent simulation & Viva guide
│   ├── model_based_reflex_agent.ipynb   # Question 3: Model-Based Reflex Agent simulation & Viva guide
│   ├── multi_agent_simulation.ipynb     # Question 6: Multi-Agent Environment Simulation (4x4 Grid)
│   ├── randomized_reflex_agent.ipynb    # Question 2: Randomized Reflex Agent simulation & Viva guide
│   ├── simple_reflex_vacuum_agent.ipynb # Question 1: Simple Reflex Vacuum Agent simulation & Viva guide
│   ├── starter_experiment.ipynb         # Interactive experimentation starter notebook
│   ├── environment_type_classifier.ipynb# Question 8: Environment Type Classifier & Knowledge Base
│   ├── stochastic_vacuum_agent.ipynb    # Question 7: Stochastic Vacuum Agent simulation & Viva guide
│   ├── agent_performance_evaluator.ipynb# Question 9: Agent Performance Measure Evaluator simulation & Viva guide
│   ├── state_space_visualizer.ipynb     # Question 10: State-Space Visualizer for Reflex Agent & Viva guide
│   ├── breadth_first_search.ipynb       # Question 11: Breadth-First Search (BFS) on User-Defined Graph & Viva guide
│   ├── depth_first_search.ipynb         # Question 12: Depth-First Search (DFS) for Graph Traversal & Viva guide
│   ├── bfs_shortest_path.ipynb          # Question 13: BFS Shortest Path Constructor & Viva guide
│   └── utility_based_agent.ipynb        # Question 5: Utility-Based Agent simulation & Viva guide
├── scripts/
│   ├── __init__.py                      # Scripts package marker
│   ├── run_goal_based_agent.py          # CLI & interactive runner for Question 4 (5x5 Grid Simulation)
│   ├── run_model_based_agent.py         # CLI & interactive runner for Question 3 (2x2 Grid Simulation)
│   ├── run_multi_agent_simulation.py    # CLI & interactive runner for Question 6 (Builder vs Destroyer)
│   ├── run_randomized_agent.py          # CLI & interactive runner for Question 2 (1D Grid Simulation)
│   ├── run_stochastic_vacuum_agent.py   # CLI & interactive runner for Question 7 (Monte Carlo Simulation)
│   ├── run_utility_based_agent.py       # CLI & interactive runner for Question 5 (Utility Optimization)
│   ├── run_environment_classifier.py    # CLI & interactive runner for Question 8 (Environment Classifier)
│   ├── run_agent_performance_evaluator.py# CLI & interactive runner for Question 9 (PEAS Performance Evaluator)
│   ├── run_state_space_visualizer.py    # CLI & interactive runner for Question 10 (State-Space Graph Visualizer)
│   ├── run_breadth_first_search.py      # CLI & interactive runner for Question 11 (Level-Order BFS & Graph Depth)
│   ├── run_depth_first_search.py        # CLI & interactive runner for Question 12 (LIFO DFS & Recursive Comparison)
│   ├── run_bfs_shortest_path.py         # CLI & interactive runner for Question 13 (Shortest Path & Goal Testing)
│   ├── run_vacuum_agent.py              # CLI & interactive runner for Question 1 (2-Square & 3x3 Grid)
│   └── smoke_test.py                    # Starter entry point and verification smoke test
├── src/
│   ├── __init__.py                      # Package initialization & public exports
│   ├── __main__.py                      # CLI dispatcher (python -m src {vacuum,randomized,model-based,goal-based,utility-based,multi-agent,stochastic,environment-classifier,agent-evaluator,state-space,bfs,dfs,bfs-shortest-path,smoke-test})
│   ├── experiment.py                    # Core experiment metrics and simulation runner
│   ├── goal_based_agent.py              # Question 4: Goal-Based Navigation Agent & 5x5 Grid Environment
│   ├── model_based_agent.py             # Question 3: Model-Based Reflex Agent & 2x2 Grid Environment
│   ├── multi_agent_simulation.py        # Question 6: Multi-Agent Environment Simulation & Shared 4x4 Grid
│   ├── randomized_agent.py              # Question 2: Randomized Reflex Agent & 1D Grid Environment
│   ├── stochastic_vacuum_agent.py       # Question 7: Stochastic Vacuum Agent & Non-deterministic Environment
│   ├── environment_classifier.py        # Question 8: Environment Type Classifier & Knowledge Base
│   ├── utility_based_agent.py           # Question 5: Utility-Based Agent & Terrain Graph Cost Optimization
│   ├── agent_performance_evaluator.py   # Question 9: Autonomous Delivery Drone PEAS Performance Evaluator
│   ├── state_space_visualizer.py        # Question 10: State-Space Visualizer, Combinatorial Model & Graph Renderer
│   ├── breadth_first_search.py          # Question 11: Breadth-First Search, FIFO Frontier, Cycle Avoidance & Levels
│   ├── depth_first_search.py            # Question 12: Depth-First Search, LIFO Stack, Recursion, Cycles & Back-Edges
│   ├── bfs_shortest_path.py             # Question 13: BFS Shortest Path Constructor, Parent Dict & Early Goal Testing
│   └── vacuum_agent.py                  # Question 1: Simple Reflex Vacuum Agent & Environments
├── tests/
│   ├── __init__.py                      # Test package marker
│   ├── test_experiment.py               # Unit test suite for experiment runner
│   ├── test_goal_based_agent.py         # Unit tests for Question 4 goal-based agent & 5x5 grid
│   ├── test_model_based_agent.py        # Unit tests for Question 3 model-based agent & 2x2 grid
│   ├── test_multi_agent_simulation.py   # Unit tests for Question 6 multi-agent simulation & rules
│   ├── test_randomized_agent.py         # Unit tests for Question 2 randomized agent & 1D grid
│   ├── test_scripts.py                  # Unit tests for script runners & CLI dispatcher
│   ├── test_stochastic_vacuum_agent.py  # Unit tests for Question 7 stochastic transitions & Monte Carlo
│   ├── test_utility_based_agent.py      # Unit tests for Question 5 utility-based agent & terrain costs
│   ├── test_agent_performance_evaluator.py# Unit tests for Question 9 linear evaluation, PEAS metric & CSV
│   ├── test_state_space_visualizer.py   # Unit tests for Question 10 combinatorial states, transitions & graph
│   ├── test_breadth_first_search.py     # Unit tests for Question 11 BFS traversal, tie-breaking, cycles & levels
│   ├── test_depth_first_search.py       # Unit tests for Question 12 DFS traversal, cycles, edge types & challenge
│   ├── test_bfs_shortest_path.py        # Unit tests for Question 13 shortest path, reconstruction & all-paths
│   └── test_vacuum_agent.py             # Unit tests for vacuum agent reflex rules & simulation traces
├── data/
│   ├── processed/                       # Processed artifacts & experiment outputs
│   └── raw/                             # Raw inputs & data snapshots
├── .gitignore                           # Comprehensive ignores (Python, Jupyter, venv, IDE)
├── pyproject.toml                       # Modern Python project configuration & packaging
├── README.md                            # Project documentation and practical guide
└── requirements.txt                     # Dependencies for notebooks and ML libraries
```

## Lab Practicals

### Question 1: Simple Reflex Vacuum Cleaner Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟢 Beginner
- **Core AI Concept**: Simple Reflex Agent, Agent-Environment Interaction, Condition-Action Mapping
- **Complexity**: Time Complexity: $O(1)$ per step | Space Complexity: $O(1)$

#### Condition-Action Reflex Rules
```mermaid
flowchart TD
    Percept["Percept: (Location, Status)"] --> Decision{"Status is Dirty?"}
    Decision -->|Yes| Suck["Action: Suck<br/>(Clean current square)"]
    Decision -->|No| CheckLoc{"Location?"}
    CheckLoc -->|Location A| Right["Action: Right<br/>(Move to B)"]
    CheckLoc -->|Location B| Left["Action: Left<br/>(Move to A)"]
```

#### Running the Vacuum Agent Simulation

1. **Default Sample Scenario** (Initial Location: `A`, Location A: `Dirty`, Location B: `Clean`, 5 Steps):
   ```bash
   python scripts/run_vacuum_agent.py
   ```

2. **Custom Location & Square States**:
   ```bash
   python scripts/run_vacuum_agent.py --location B --status-a Dirty --status-b Dirty --steps 5
   ```

3. **Interactive Mode**:
   ```bash
   python scripts/run_vacuum_agent.py --interactive
   ```

4. **Optional Challenge (3x3 Grid with Randomized Dirt Placement)**:
   ```bash
   python scripts/run_vacuum_agent.py --grid --steps 10
   ```

5. **Display Viva Voce Q&A and Theory Guide**:
   ```bash
   python scripts/run_vacuum_agent.py --viva
   ```

6. **Interactive Jupyter Notebook**:
   Open [`notebooks/simple_reflex_vacuum_agent.ipynb`](notebooks/simple_reflex_vacuum_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 2: Randomized Reflex Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟢 Beginner
- **Core AI Concept**: Randomized Agent Behavior, Handling Unobservable States, Escaping Boundary Loops
- **Complexity**: Time Complexity: $O(N)$ average | Space Complexity: $O(N)$

#### Condition-Action Stochastic Rules
```mermaid
flowchart TD
    Percept["Percept: Status of Current Room<br/>(0=Clean, 1=Dirty)"] --> CheckStatus{"Status is Dirty (1)?"}
    CheckStatus -->|Yes| Clean["Action: Clean<br/>(Set room status to 0)"]
    CheckStatus -->|No| RandomChoice{"Random Choice<br/>random.choice(['Left', 'Right'])"}
    RandomChoice -->|50% Probability| MoveLeft["Action: Left<br/>(Move left / bump boundary 0)"]
    RandomChoice -->|50% Probability| MoveRight["Action: Right<br/>(Move right / bump boundary N-1)"]
```

#### Running the Randomized Reflex Agent Simulation

1. **Sample Scenario with Exact 12-Step Reproducibility** (`[1, 0, 0, 1, 1]`, Start Index: `2`, Seed: `58`):
   ```bash
   python scripts/run_randomized_agent.py --seed 58
   ```
   *Output*:
   ```text
   All rooms clean in 12 steps.
   Total Navigation Moves   : 9
   Total Cleaning Actions   : 3
   Final Grid State         : [0, 0, 0, 0, 0]
   Final Agent Location     : Room [4]
   ```

2. **Custom Room State & Start Index**:
   ```bash
   python scripts/run_randomized_agent.py --state "1 0 1 1 0" --start-index 1
   ```

3. **Interactive Mode**:
   ```bash
   python scripts/run_randomized_agent.py --interactive
   ```

4. **Monte Carlo Empirical Benchmark (100 Trials)**:
   ```bash
   python scripts/run_randomized_agent.py --benchmark --trials 100
   ```

5. **Display Viva Voce Theory Guide**:
   ```bash
   python scripts/run_randomized_agent.py --viva
   ```

6. **Interactive Jupyter Notebook**:
   Open [`notebooks/randomized_reflex_agent.ipynb`](notebooks/randomized_reflex_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 3: Model-Based Reflex Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟡 Intermediate
- **Core AI Concept**: Model-Based Reflex Agent, Internal State Representation, Transition Models, Sensor Models
- **Complexity**: Time Complexity: $O(N)$ | Space Complexity: $O(N)$ where $N=4$ for 2x2 grid

#### Model-Based Architecture & Decision Flow
```mermaid
flowchart TD
    Percept["Percept: ((row, col), Status)"] --> UpdateState["Update Internal State<br/>• visited.add(pos)<br/>• model[pos] = status<br/>• Apply Transition Model"]
    UpdateState --> CheckDirty{"Is Current Room Dirty?"}
    CheckDirty -->|Yes| Clean["Action: Clean<br/>(model[pos] = 'Clean')"]
    CheckDirty -->|No| CheckAllClean{"Internal Model Confirms<br/>All Rooms Clean & Visited?"}
    CheckAllClean -->|Yes| Shutdown["Action: Shutdown<br/>'Agent visited all rooms. Cleaning complete. Shutting down.'"]
    CheckAllClean -->|No| PlanMove["Action: Plan Navigation<br/>(BFS to closest unvisited or dirty room)"]
```

#### Running the Model-Based Agent Simulation

1. **Default Sample Scenario** (2x2 Grid: `[[Dirty, Clean], [Dirty, Dirty]]`, Start at `(0, 0)`):
   ```bash
   python scripts/run_model_based_agent.py
   ```
   *Expected Output*:
   ```text
   Step  1: Pos (0, 0) | Percept: Dirty | Action: Clean    | Visited: {(0,0)}
   Step  2: Pos (0, 0) | Percept: Clean | Action: Down     | Visited: {(0,0)}
   Step  3: Pos (1, 0) | Percept: Dirty | Action: Clean    | Visited: {(0,0), (1,0)}
   Step  4: Pos (1, 0) | Percept: Clean | Action: Right    | Visited: {(0,0), (1,0)}
   Step  5: Pos (1, 1) | Percept: Dirty | Action: Clean    | Visited: {(0,0), (1,0), (1,1)}
   Step  6: Pos (1, 1) | Percept: Clean | Action: Up       | Visited: {(0,0), (1,0), (1,1)}
   Step  7: Pos (0, 1) | Percept: Clean | Action: Shutdown | Visited: {(0,0), (0,1), (1,0), (1,1)} -> Agent visited all rooms. Cleaning complete. Shutting down.
   ```

2. **Custom Grid & Start Position**:
   ```bash
   python scripts/run_model_based_agent.py --grid "Dirty Clean, Dirty Dirty" --start-row 0 --start-col 0
   ```

3. **Optional Challenge (Stochastic Sensor Failure)**:
   ```bash
   python scripts/run_model_based_agent.py --sensor-failure 0.2 --seed 42
   ```

4. **Interactive Walkthrough Mode**:
   ```bash
   python scripts/run_model_based_agent.py --interactive
   ```

5. **Display Viva Voce Reference & Theory Guide**:
   ```bash
   python scripts/run_model_based_agent.py --viva
   ```

6. **Interactive Jupyter Notebook**:
   Open [`notebooks/model_based_reflex_agent.ipynb`](notebooks/model_based_reflex_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 4: Goal-Based Navigation Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟡 Intermediate
- **Core AI Concept**: Goal-Based Agent, Distance Metrics (Manhattan & Euclidean), Greedy Coordinate Reduction
- **Complexity**: Time Complexity: $O(D)$ | Space Complexity: $O(D)$ where $D = |x_1 - x_2| + |y_1 - y_2|$ is Manhattan distance

#### Goal-Based Decision Architecture
```mermaid
flowchart TD
    Percept["Percept: Current Location (x, y)<br/>Goal Coordinate (gx, gy)"] --> CheckGoal{"At Charging Station?<br/>(x, y) == (gx, gy)"}
    CheckGoal -->|Yes| Halt["Goal Reached<br/>Halt & Output Path"]
    CheckGoal -->|No| Successors["Generate 4-Connected Successors<br/>(Up, Down, Left, Right)<br/>*Strictly Reject Diagonals*"]
    Successors --> EvalDist["Calculate Manhattan Distance<br/>d_M = |nx - gx| + |ny - gy|"]
    EvalDist --> Filter["Filter Actions Strictly Reducing Distance<br/>d_next < d_curr"]
    Filter --> Select["Select Best Orthogonal Action<br/>(Greedy Coordinate Reduction)"]
    Select --> Step["Execute Action & Record Path<br/>Update Agent Position"]
    Step --> Percept
```

#### Running the Goal-Based Agent Simulation

1. **Default Sample Scenario** (Start: `(0, 0)`, Goal: `(3, 4)`):
   ```bash
   python scripts/run_goal_based_agent.py
   ```
   *Expected Output*:
   ```text
   Step  1: Pos (0, 0) -> Action: Down  -> Next: (1, 0) | Dist: 7 -> 6 (Decreased)
   Step  2: Pos (1, 0) -> Action: Down  -> Next: (2, 0) | Dist: 6 -> 5 (Decreased)
   Step  3: Pos (2, 0) -> Action: Down  -> Next: (3, 0) | Dist: 5 -> 4 (Decreased)
   Step  4: Pos (3, 0) -> Action: Right -> Next: (3, 1) | Dist: 4 -> 3 (Decreased)
   Step  5: Pos (3, 1) -> Action: Right -> Next: (3, 2) | Dist: 3 -> 2 (Decreased)
   Step  6: Pos (3, 2) -> Action: Right -> Next: (3, 3) | Dist: 2 -> 1 (Decreased)
   Step  7: Pos (3, 3) -> Action: Right -> Next: (3, 4) | Dist: 1 -> 0 (Decreased)

   Path: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
   ```

2. **Custom Start & Goal Coordinates**:
   ```bash
   python scripts/run_goal_based_agent.py --start 1 1 --goal 4 2
   ```

3. **Random Goal Placement (with Seed Reproducibility)**:
   ```bash
   python scripts/run_goal_based_agent.py --random-goal --seed 42
   ```

4. **Optional Challenge (Explicit Diagonal Movement Prevention)**:
   ```bash
   python scripts/run_goal_based_agent.py --challenge
   ```

5. **Distance Metric Comparison (Manhattan $L_1$ vs. Euclidean $L_2$)**:
   ```bash
   python scripts/run_goal_based_agent.py --compare-metrics
   ```

6. **Interactive Terminal Mode**:
   ```bash
   python scripts/run_goal_based_agent.py --interactive
   ```

7. **Display Viva Voce Reference & Theory Guide**:
   ```bash
   python scripts/run_goal_based_agent.py --viva
   ```

8. **Interactive Jupyter Notebook**:
   Open [`notebooks/goal_based_navigation_agent.ipynb`](notebooks/goal_based_navigation_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 5: Utility-Based Agent Simulation

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟠 Advanced
- **Practical Type**: Optimization
- **Core AI Concept**: Utility-Based Agent, Cost Minimization vs. Utility Maximization, Expected Utility Theory
- **Complexity**: Time Complexity: $O(P \times L)$ | Space Complexity: $O(1)$ where $P$ is candidate paths and $L$ is path length

#### Decision & Optimization Flow
```mermaid
flowchart TD
    Input["Candidate Paths: [A-B-D, A-C-D, A-E-D]<br/>Environment Graph with Terrain Costs"] --> Cost["Path Cost Summation<br/>total_cost = sum(c(u, v))"]
    Cost --> Util["Formulate Utility Function<br/>U(path) = -1 * total_cost"]
    Util --> BatteryCheck{"Battery Constraint?<br/>cost <= battery_limit"}
    BatteryCheck -->|No| Reject["Reject Path<br/>(Battery Depleted)"]
    BatteryCheck -->|Yes| Feasible["Mark Path Feasible"]
    Feasible --> Sort["Custom Lambda Sorting<br/>sorted(feasible, key=lambda ev: ev.utility, reverse=True)"]
    Sort --> Maximize["Select Path with Highest Utility<br/>argmax_p U(p) == argmin_p Cost(p)"]
    Maximize --> Output["Selected Path: A-C-D | Utility: -4.0"]
```

#### Running the Utility-Based Agent Simulation

1. **Default Run (3 Predefined Paths on Sample Graph)**:
   ```bash
   python scripts/run_utility_based_agent.py
   ```
   *Expected Output*:
   ```text
   Candidate Path Evaluations:
   ---------------------------------------------------------------------------
   #   Path             Terrain Breakdown          Cost    Utility  Status
   ---------------------------------------------------------------------------
   1   A-B-D            G:1 + W:5                  6.0     -6.0     FEASIBLE
   2   A-C-D            G:1 + M:3                  4.0     -4.0     SELECTED (Optimal)
   3   A-E-D            M:3 + W:5                  8.0     -8.0     FEASIBLE
   ---------------------------------------------------------------------------

   Optimization Result:
     Selected Path: A-C-D
     Total Cost:    4.0
     Utility:       -4.0
   ```

2. **Custom Paths via CLI**:
   ```bash
   python scripts/run_utility_based_agent.py --paths "A-B-D" "A-C-D"
   ```

3. **Optional Challenge (Battery Capacity Constraint)**:
   ```bash
   python scripts/run_utility_based_agent.py --challenge
   ```
   *Output*:
   ```text
   Battery Capacity Limit: 5.0 units
   Rule: Reject any path with total_cost > battery_limit regardless of utility.

   Challenge Analysis:
     [REJECTED] Path A-B-D      Cost: 6.0 > 5.0 (Battery Depleted)
     [ACCEPTED] Path A-C-D      Cost: 4.0 <= 5.0
     [REJECTED] Path A-E-D      Cost: 8.0 > 5.0 (Battery Depleted)

   Best Feasible Path: A-C-D (Cost: 4.0, Utility: -4.0)
   ```

4. **Auto-Discovery of All Directed Paths (DFS)**:
   ```bash
   python scripts/run_utility_based_agent.py --all-paths
   ```

5. **Top-Level Module Dispatcher**:
   ```bash
   python -m src utility-based
   ```

6. **Interactive Terminal Mode**:
   ```bash
   python scripts/run_utility_based_agent.py --interactive
   ```

7. **Display Viva Voce Reference & Theory Guide**:
   ```bash
   python scripts/run_utility_based_agent.py --viva
   ```

8. **Interactive Jupyter Notebook**:
   Open [`notebooks/utility_based_agent.ipynb`](notebooks/utility_based_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 6: Multi-Agent Environment Simulation

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟠 Advanced
- **Practical Type**: Simulation
- **Core AI Concept**: Multi-Agent Systems, Competitive/Zero-Sum Environments, Sequential vs. Episodic, Dynamic vs. Static, Turn-Based Execution, Collision Avoidance
- **Complexity**: Time Complexity: $O(S)$ where $S$ is steps | Space Complexity: $O(\text{Grid Size}) = O(16) = O(1)$

#### Multi-Agent Architecture & Turn-Based Interaction Loop
```mermaid
flowchart TD
    Init["Initialize Shared 4x4 Grid (all 0s)<br/>• Agent A (Builder) at (0, 0)<br/>• Agent B (Destroyer) at (3, 3)"] --> TurnA["Round t: Agent A Turn (Builder)"]
    
    TurnA --> PerceptA["Perceive Shared Cell & Peers<br/>Cell = grid[pos_A]"]
    PerceptA --> RuleA{"Is Cell Empty (0)?"}
    RuleA -->|Yes| Place["Action: PlaceBlock<br/>grid[pos_A] = 1"]
    RuleA -->|No| MoveA["Action: Move to Safe Neighbor"]
    Place --> MoveACompound["Compound Step: Move to Safe Neighbor"]
    
    MoveACompound --> TurnB["Round t: Agent B Turn (Destroyer)"]
    MoveA --> TurnB
    
    TurnB --> PerceptB["Perceive Shared Cell & Peers<br/>Cell = grid[pos_B]"]
    PerceptB --> RuleB{"Is Cell Block (1)?"}
    RuleB -->|Yes| Destroy["Action: RemoveBlock<br/>grid[pos_B] = 0"]
    RuleB -->|No| MoveB["Action: Move to Safe Neighbor"]
    Destroy --> MoveBCompound["Compound Step: Move to Safe Neighbor"]
    
    MoveBCompound --> CheckStep{"Completed 10 Steps?"}
    MoveB --> CheckStep
    CheckStep -->|No| TurnA
    CheckStep -->|Yes| Final["Output Final 4x4 Grid Matrix (NumPy Array)<br/>Display 1s (Blocks) and 0s (Empty)"]
```

#### Running the Multi-Agent Simulation

1. **Default 10-Step Simulation** (4x4 Grid, Builder at `(0, 0)`, Destroyer at `(3, 3)`):
   ```bash
   python scripts/run_multi_agent_simulation.py --steps 10
   ```
   *Sample Output*:
   ```text
   Sample Output (NumPy Matrix displaying 1s and 0s):
   [[1 1 0 0]
    [1 1 1 0]
    [0 1 1 0]
    [0 0 0 0]]

   ======================================================================
                  MULTI-AGENT ENVIRONMENT SIMULATION SUMMARY             
   ======================================================================
   Execution Mode:             Compound Turns
   Total Rounds (Steps):       10
   Agent A:                    Agent_A (Builder) | Start: (0, 0) -> Final: (1, 3)
   Agent B:                    Agent_B (Destroyer) | Start: (3, 3) -> Final: (2, 0)
   Performance Metrics:        Builder placed 9 blocks | Destroyer removed 2 blocks | Net: +7
   Final Block Count in Grid:  7 / 16
   Collision Avoidance:        Disabled
   Collisions Avoided:         0
   Collisions Occurred:        1
   ----------------------------------------------------------------------
   Final Grid State Matrix (NumPy Array):
   [[1 1 0 0]
    [1 1 1 0]
    [0 1 1 0]
    [0 0 0 0]]
   ======================================================================
   ```

2. **Optional Challenge (Active Collision Awareness & Avoidance)**:
   ```bash
   python scripts/run_multi_agent_simulation.py --challenge
   ```
   *Output Highlights*:
   ```text
   Collision Avoidance: ENABLED (Optional Challenge)
   Collisions Avoided:  2
   Collisions Occurred: 0
   ```

3. **Discrete Turn Mode (Act OR Move)**:
   ```bash
   python scripts/run_multi_agent_simulation.py --mode discrete --steps 10
   ```

4. **Greedy Heuristic Movement Strategy**:
   ```bash
   python scripts/run_multi_agent_simulation.py --strategy greedy --steps 10
   ```

5. **Top-Level Module Dispatcher**:
   ```bash
   python -m src multi-agent --steps 10
   ```

6. **Interactive Step-by-Step Walkthrough**:
   ```bash
   python scripts/run_multi_agent_simulation.py --interactive
   ```

7. **Display Viva Voce Reference & Theory Guide**:
   ```bash
   python scripts/run_multi_agent_simulation.py --viva
   ```

8. **Interactive Jupyter Notebook**:
   Open [`notebooks/multi_agent_simulation.ipynb`](notebooks/multi_agent_simulation.ipynb) in Jupyter Lab or Antigravity IDE.

### Question 7: Stochastic Environment Vacuum Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟠 Advanced
- **Practical Type**: Simulation
- **Core AI Concept**: Stochastic Environments, Non-deterministic Actions, Actuator Slippage, Monte Carlo Estimation
- **Python Concepts Tested**: `random` module (`random.random()`), statistical averaging (`statistics`), loops, dataclasses
- **Complexity**: Time Complexity: $O(\text{Runs} \times \text{Steps})$ | Space Complexity: $O(\text{Runs})$

#### Probabilistic Actuator State Transition Model
```mermaid
stateDiagram-v2
    [*] --> Room_0: Agent starts in Room 0 (Dirty)
    Room_0 --> Room_0: Clean Room 0 (Deterministic, 100%)
    Room_0 --> Room_1: Action Move Right (80% Success)
    Room_0 --> Room_0: Action Move Right (20% Slip Stay)
    
    Room_1 --> Room_1: Clean Room 1 (Deterministic, 100%)
    Room_1 --> Room_2: Action Move Right (80% Success)
    Room_1 --> Room_1: Action Move Right (20% Slip Stay)
    Room_1 --> Room_0: Action Move Right (Optional Challenge: Slip Back)
    
    Room_2 --> Room_2: Clean Room 2 (Deterministic, 100%)
    Room_2 --> [*]: All 3 Rooms Clean (Task Goal Achieved)
```

#### Theoretical Expectation vs. Deterministic Baseline

| Metric | Deterministic Baseline | Stochastic Environment ($p=0.8$) |
| :--- | :--- | :--- |
| **Cleaning Steps** | 3 steps ($1$ step per room) | 3 steps ($1$ step per room) |
| **Room Transitions** | 2 steps ($1$ step per move) | $\mathbb{E}[\text{Moves}] = 2 \times \frac{1}{0.8} = 2.5$ steps |
| **Total Expected Steps** | **5.0 steps** | **5.5 steps** |
| **Performance Penalty** | Baseline (0.0 steps) | **+0.5 steps (+10.0% execution delay)** |

#### Running the Stochastic Vacuum Agent Simulation

1. **Standard Monte Carlo Batch (20 Runs)**:
   ```bash
   python scripts/run_stochastic_vacuum_agent.py --runs 20
   ```
   *Expected Output*:
   ```text
   Average steps to clean 3 rooms: 5.4 steps
   ```

2. **Reproducible Benchmark (Seed 28 reproduces sample output exactly)**:
   ```bash
   python scripts/run_stochastic_vacuum_agent.py --runs 20 --seed 28
   ```

3. **Detailed Step-by-Step Trajectory Trace**:
   ```bash
   python scripts/run_stochastic_vacuum_agent.py --runs 20 --seed 28 --trace
   ```

4. **Optional Challenge: Actuator 'Slip Back' Mechanics**:
   ```bash
   python scripts/run_stochastic_vacuum_agent.py --challenge --seed 42
   ```

5. **Display Viva Voce Theory & Reference Guide**:
   ```bash
   python scripts/run_stochastic_vacuum_agent.py --viva
   ```

6. **Top-Level Package Dispatcher**:
   ```bash
   python -m src stochastic --runs 20 --seed 28
   ```

7. **Interactive Jupyter Notebook**:
   Open [`notebooks/stochastic_vacuum_agent.ipynb`](notebooks/stochastic_vacuum_agent.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 8: Environment Type Classifier

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟡 Intermediate
- **Practical Type**: Knowledge Representation
- **Core AI Concept**: Environment Properties (PEAS), Environmental Taxonomy, Knowledge Dictionary Representation
- **Python Concepts Tested**: Tuples as dictionary keys, robust boolean input parsing, nearest-neighbor Hamming distance
- **Complexity**: Time Complexity: (1)$ dictionary lookup | Space Complexity: (K)$ for knowledge base ( \le 32$)

#### Knowledge Representation & Lookup Flowchart
`mermaid
flowchart TD
    User([User Prompt / CLI Input]) --> Parse[Parse Flexible Booleans]
    Parse --> Vector[5-Boolean Tuple Key: Obs, Det, Epi, Stat, Disc]
    Vector --> Lookup{O 1 Hash Key in Knowledge Base?}
    Lookup -->|Exact Key Found| Meta[Retrieve EnvironmentMetadata]
    Meta --> Success([Display Exact Match & PEAS Breakdown])
    Lookup -->|Unknown Combination| Challenge{Nearest-Neighbor Fallback?}
    Challenge -->|Enabled| Metric[Compute Hamming Distance D_H to all K entries]
    Metric --> Rank[Find Minimum Distance Tier]
    Rank --> Nearest([Display Nearest Environment & Property Differences])
    Challenge -->|Disabled| Graceful([Inform Unknown Combination Gracefully])
`

#### Canonical AI Environment Taxonomy (Russell & Norvig Table 2.1)

| Environment | Observable | Deterministic | Episodic | Static | Discrete | Agent Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Backgammon (with clock)** *(Sample)* | **True** | **False** (Dice) | **False** (Seq) | **False** (Clock) | **True** | Multi-Agent (Competitive) |
| **Backgammon (without clock)** | **True** | **False** (Dice) | **False** (Seq) | **True** | **True** | Multi-Agent (Competitive) |
| **Chess (with clock)** | **True** | **True** | **False** (Seq) | **False** (Clock) | **True** | Multi-Agent (Competitive) |
| **Chess (without clock) / Crossword** | **True** | **True** | **False** (Seq) | **True** | **True** | Multi / Single-Agent |
| **Poker (with clock)** | **False** (Hidden) | **False** (Deck) | **False** (Seq) | **False** (Clock) | **True** | Multi-Agent (Imperfect Info) |
| **Poker (without clock)** | **False** (Hidden) | **False** (Deck) | **False** (Seq) | **True** | **True** | Multi-Agent (Imperfect Info) |
| **Medical Diagnosis / Taxi Driving** | **False** | **False** | **False** (Seq) | **False** (Dynamic) | **False** (Cont) | Single / Multi-Agent |
| **Part-Picking Robot** | **False** | **False** | **True** (Episodic) | **False** (Belt) | **False** (Cont) | Single-Agent |
| **Image Analysis (Discrete)** | **True** | **True** | **True** (Episodic) | **True** | **True** | Single-Agent |
| **Battleship (Simplified)** | **False** | **True** | **False** (Seq) | **True** | **True** | Multi-Agent (Competitive) |

#### Running the Environment Type Classifier

1. **Interactive Prompt Mode** (Prompts user for 5 environmental properties):
   `ash
   python scripts/run_environment_classifier.py
   `

2. **Direct Tuple Input** (Sample execution matching Backgammon):
   `ash
   python scripts/run_environment_classifier.py --input True,False,False,False,True
   `
   *Expected Output*:
   `	ext
   Environment Match: Backgammon
   `

3. **Individual Environmental Dimension Flags**:
   `ash
   python scripts/run_environment_classifier.py --observable yes --deterministic yes --episodic no --static no --discrete yes
   `
   *Output*:
   `	ext
   Environment Match: Chess (with clock)
   `

4. **Standard Textbook Environments Demonstration Suite**:
   `ash
   python scripts/run_environment_classifier.py --demo
   `

5. **Optional Challenge: Nearest-Neighbor Distance Metric (Hamming Distance)**:
   `ash
   python scripts/run_environment_classifier.py --challenge
   `

6. **Display Viva Voce Theory & Reference Guide**:
   `ash
   python scripts/run_environment_classifier.py --viva
   `

7. **Top-Level Package Dispatcher**:
   `ash
   python -m src environment-classifier --input True,False,False,False,True
   `

8. **Interactive Jupyter Notebook**:
   Open [
otebooks/environment_type_classifier.ipynb](notebooks/environment_type_classifier.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 9: Agent Performance Measure Evaluator

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟡 Intermediate
- **Practical Type**: Algorithm Design
- **Core AI Concept**: Performance Measure, Rationality, Omniscience vs. Rationality, PEAS Evaluation
- **Python Concepts Tested**: List iteration, arithmetic updates, string matching, CSV stream processing
- **Complexity**: Time Complexity: $O(N)$ log length | Space Complexity: $O(1)$ auxiliary space

#### Autonomous Delivery Drone PEAS Specification

| PEAS Dimension | Drone Specification | Impact on Agent Rationality |
| :--- | :--- | :--- |
| **Performance Measure (P)** | $+10$ / delivery, $-1$ / move, $-5$ / bump | Objective utility function balancing mission completion, latency, and flight safety |
| **Environment (E)** | Urban airspace, buildings, power lines, wind, landing pads | Dynamic, partially observable, stochastic, continuous 3D world |
| **Actuators (A)** | Quadcopter/hexacopter rotors, pitch/yaw motors, parcel winch/gripper | Translates movement actions into physical flight trajectory |
| **Sensors (S)** | GPS receiver, 3D LiDAR scanner, ultrasonic sonars, downward camera, IMU | Supplies real-time percepts for obstacle detection and landing |

#### Performance Score Evaluation Model

```mermaid
flowchart TD
  Log["Simulated Action Event Stream: [move, move, bump, move, deliver]"] --> Seq[Linear Sequential Processing O(N)]
  Seq --> Match{Action Type}
  Match -->|move| Cost["Subtract Move Cost (-1 pt)"]
  Match -->|bump| Penalty["Subtract Collision Penalty (-5 pts)"]
  Match -->|deliver| Reward["Add Delivery Reward (+10 pts)"]
  Cost --> Running["Update Running Score (O(1) Memory)"]
  Penalty --> Running
  Reward --> Running
  Running --> Next{More Events?}
  Next -->|Yes| Match
  Next -->|No| Final["Output Final PEAS Performance Score: 2 (10 - 3 moves - 5 bump)"]
```

#### Running the Agent Performance Evaluator

1. **Standard Sample Benchmark Run**:
   ```bash
   python scripts/run_agent_performance_evaluator.py
   ```
   *Expected Output*:
   ```text
   Performance Score: 2 (10 - 3 moves - 5 bump)
   ```

2. **Custom Inline Event Sequence**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --events move,move,bump,move,deliver
   ```

3. **Detailed Step-by-Step Flight Audit Table**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --detailed
   ```

4. **Optional Challenge: Read Action Log from External CSV File**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --csv data/raw/drone_actions_sample.csv
   ```

5. **Automated Demonstration Benchmark Suite**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --demo
   ```

6. **Interactive Step-by-Step Terminal Prompt**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --interactive
   ```

7. **Display Viva Voce Theory & Reference Guide**:
   ```bash
   python scripts/run_agent_performance_evaluator.py --viva
   ```

8. **Top-Level Package Dispatcher**:
   ```bash
   python -m src agent-evaluator --events move,move,bump,move,deliver
   ```

9. **Interactive Jupyter Notebook**:
   Open [`notebooks/agent_performance_evaluator.ipynb`](notebooks/agent_performance_evaluator.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 10: State-Space Visualizer for Reflex Agent

- **Syllabus Area**: Intelligent Agents
- **Difficulty**: 🟠 Advanced
- **Practical Type**: Visualization & State-Space Formulation
- **Core AI Concept**: State-Space Formulation, Transition Model, Goal States, Cyclic Transitions
- **Python Concepts Tested**: Combinatorial state generation (`itertools`), NetworkX directed graphs (`DiGraph` & `MultiDiGraph`), Matplotlib publication-grade plotting
- **Complexity**: Time Complexity: $O(|S|^2)$ | Space Complexity: $O(|S|^2)$
- **Optional Challenge**: Highlight the goal states (both rooms clean) in emerald green.

#### State Space Size Formula & Combinatorial Model

For an environment with $n$ locations and $k$ cleanliness status levels:
$$|S| = n \times k^n$$

For the standard 2-room vacuum cleaner environment ($n = 2, k = 2$):
$$|S| = 2 \times 2^2 = 8 \text{ discrete states}$$

#### The 8 Canonical States (Russell & Norvig AIMA Convention)

| State ID | Agent Location | Room A Status | Room B Status | Goal State? |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | A | Dirty | Dirty | No |
| **S2** | B | Dirty | Dirty | No |
| **S3** | A | Clean | Dirty | No |
| **S4** | B | Clean | Dirty | No |
| **S5** | A | Dirty | Clean | No |
| **S6** | B | Dirty | Clean | No |
| **S7** | A | Clean | Clean | **YES (Emerald Green Goal State)** |
| **S8** | B | Clean | Clean | **YES (Emerald Green Goal State)** |

#### Action Transitions & Cyclic Self-Loop Dynamics

Valid actions are $A = \{\text{Left}, \text{Right}, \text{Suck}\}$. Total transitions across the state space:
$$8 \text{ states} \times 3 \text{ actions} = 24 \text{ directed transitions}$$

- **Cyclic Transitions (Self-Loops)**: Exactly 12 out of 24 transitions (50%) produce $\text{Result}(s, a) = s$:
  - 4 `Left` self-loops: Occur whenever agent is in Room A (States 1, 3, 5, 7).
  - 4 `Right` self-loops: Occur whenever agent is in Room B (States 2, 4, 6, 8).
  - 4 `Suck` self-loops: Occur whenever agent sucks in an already clean room (States 3, 6, 7, 8).
- **State-Changing Transitions**: Exactly 12 out of 24 transitions (50%) transition to a different state.

#### State Space Graph Architecture

```mermaid
flowchart TD
  subgraph Both_Dirty["Both Rooms Dirty (Row 3)"]
    S1["S1: (A, Dirty, Dirty)"]
    S2["S2: (B, Dirty, Dirty)"]
  end

  subgraph A_Clean["Room A Clean, Room B Dirty (Row 2)"]
    S3["S3: (A, Clean, Dirty)"]
    S4["S4: (B, Clean, Dirty)"]
  end

  subgraph B_Clean["Room A Dirty, Room B Clean (Row 1)"]
    S5["S5: (A, Dirty, Clean)"]
    S6["S6: (B, Dirty, Clean)"]
  end

  subgraph Both_Clean["Goal States: Both Rooms Clean (Row 0)"]
    S7["S7: (A, Clean, Clean) - GOAL"]
    S8["S8: (B, Clean, Clean) - GOAL"]
  end

  S1 -->|Right| S2
  S1 -->|Suck| S3
  S2 -->|Left| S1
  S2 -->|Suck| S6

  S3 -->|Right| S4
  S4 -->|Left| S3
  S4 -->|Suck| S8

  S5 -->|Right| S6
  S5 -->|Suck| S7
  S6 -->|Left| S5

  S7 -->|Right| S8
  S8 -->|Left| S7

  style S7 fill:#2ecc71,stroke:#196f3d,stroke-width:3px,color:#fff
  style S8 fill:#2ecc71,stroke:#196f3d,stroke-width:3px,color:#fff
```

#### Running the State-Space Visualizer

1. **Default Run (ASCII Table & Graph Generation)**:
   ```bash
   python scripts/run_state_space_visualizer.py
   ```

2. **Display Interactive Matplotlib Graph Window**:
   ```bash
   python scripts/run_state_space_visualizer.py --show
   ```

3. **Save High-Resolution Graph Image**:
   ```bash
   python scripts/run_state_space_visualizer.py --save reports/state_space_graph.png
   ```

4. **Alternative Graph Topologies (Circular, Spring)**:
   ```bash
   python scripts/run_state_space_visualizer.py --layout circular --save reports/state_space_circular.png
   ```

5. **Display Detailed 8-State Transition Table**:
   ```bash
   python scripts/run_state_space_visualizer.py --table
   ```

6. **Display 8x8 Adjacency Matrix ($O(|S|^2)$)**:
   ```bash
   python scripts/run_state_space_visualizer.py --matrix
   ```

7. **Cyclic Transitions & Self-Loop Audit**:
   ```bash
   python scripts/run_state_space_visualizer.py --cycles
   ```

8. **Optimal Shortest Paths to Goal (BFS Search)**:
   ```bash
   python scripts/run_state_space_visualizer.py --search
   ```

9. **Run Complete Demonstration Suite**:
   ```bash
   python scripts/run_state_space_visualizer.py --demo
   ```

10. **Display Viva Voce Theory & Reference Guide**:
    ```bash
    python scripts/run_state_space_visualizer.py --viva
    ```

11. **Top-Level Package Dispatcher**:
    ```bash
    python -m src state-space --table
    ```

12. **Interactive Jupyter Notebook**:
    Open [`notebooks/state_space_visualizer.ipynb`](notebooks/state_space_visualizer.ipynb) in Jupyter Lab or Antigravity IDE.

---

### Question 11: Breadth-First Search (BFS) on User-Defined Graph

- **Syllabus Area**: Uninformed Search Techniques
- **Difficulty**: 🟢 Beginner
- **Practical Type**: Implementation & Level-Order Traversal
- **Core AI Concept**: Breadth-First Search (BFS), FIFO Frontier Queue, Visited / Explored Set
- **Python Concepts Tested**: Dictionaries (`Dict` adjacency list), `collections.deque`, `while` loops, path reconstruction
- **Expected Algorithm / Technique**: Level-order graph traversal (FIFO queue)
- **Complexity Expectation**: Time Complexity: $O(V + E)$ | Space Complexity: $O(V)$
- **Constraints / Conditions**: Nodes must be visited in alphabetical order if tied.
- **University Practical Importance**: Very High
- **Optional Challenge**: Modify to return the number of levels (depth) of the graph.

#### Problem Statement & Task Requirements

Write a Python program to implement Breadth-First Search (BFS) on a directed graph. The program must traverse the graph starting from a user-defined source node, maintain a visited set to avoid cycles, and print the nodes in the exact order they are expanded.

1. **Graph Representation**: Represent the directed graph using a Python dictionary (adjacency list).
2. **FIFO Frontier Queue**: Use `collections.deque` for the search frontier open list.
3. **Visited Set**: Maintain a `set` to track discovered nodes and avoid infinite loops on cycles.
4. **Deterministic Tie-Breaking**: When multiple successor nodes are tied at the same parent, expand them in alphabetical order.
5. **Output**: Print the nodes in the exact order they are expanded.
6. **Optional Challenge**: Return the total number of depth levels and the maximum depth of the graph.

#### Sample Input & Output Verification

- **Sample Graph**: `{'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}`
- **Start Node**: `'A'`
- **Expected Traversal**: `BFS Traversal: A, B, C, D`
- **Optional Challenge Levels**:
  - Level 0 (depth 0): `['A']`
  - Level 1 (depth 1): `['B', 'C']`
  - Level 2 (depth 2): `['D']`
  - Total Levels Count: `3` | Maximum Depth: `2`

#### Algorithm Walkthrough & Architectural Diagram

```mermaid
flowchart TD
    Start([Start Node: 'A']) --> QInit["Frontier Queue: deque(['A'])<br/>Visited Set: {'A'}<br/>Levels: {'A': 0}"]
    QInit --> Loop{"Is Frontier Queue Empty?"}
    Loop -- No --> Dequeue["popleft() Node u from Frontier<br/>Append u to Traversal Order"]
    Dequeue --> Sort["Retrieve Neighbors Adj[u]<br/>Sort Alphabetically (Tie-Breaking)"]
    Sort --> ForEach["For each neighbor v in sorted Adj[u]"]
    ForEach --> VisCheck{"Is v in Visited Set?"}
    VisCheck -- No (Unvisited) --> Enqueue["Visited.add(v)<br/>Frontier.append(v)<br/>Levels[v] = Levels[u] + 1<br/>Parents[v] = u"]
    VisCheck -- Yes (Cycle / Re-visit) --> Skip["Skip neighbor v (Cycle Prevented)"]
    Enqueue --> ForEach
    Skip --> ForEach
    ForEach -- Finished all neighbors --> Loop
    Loop -- Yes (Frontier Empty) --> Finish([Return BFSResult & Traversal Order])
```

#### Chronological FIFO Frontier Trace Table (Sample Graph)

| Step | Node Expanded | Depth Level | Parent | Discovered & Enqueued | FIFO Frontier Queue Snapshot | Visited Set Snapshot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **A** | 0 | `ROOT` | `['B', 'C']` | `['B', 'C']` | `{'A', 'B', 'C'}` |
| **2** | **B** | 1 | `A` | `['D']` | `['C', 'D']` | `{'A', 'B', 'C', 'D'}` |
| **3** | **C** | 1 | `A` | `[]` (None) | `['D']` | `{'A', 'B', 'C', 'D'}` |
| **4** | **D** | 2 | `B` | `[]` (None) | `[]` (Empty) | `{'A', 'B', 'C', 'D'}` |

#### Performance & Asymptotic Complexity Analysis

- **Time Complexity: $O(V + E)$**
  - **Vertex Processing**: Every vertex $v \in V$ reachable from the start node is enqueued once and dequeued once from the `collections.deque` frontier. Because `deque.append()` and `deque.popleft()` operate in $O(1)$ amortized time, all queue manipulations take $O(V)$ time.
  - **Edge Processing**: When node $u$ is expanded, each of its outgoing directed edges $(u, v) \in E$ is scanned exactly once. Across the entire traversal, the sum of outgoing degrees is $\sum_{u \in V} \text{deg}^+(u) = O(E)$.
  - **Total Time**: $O(V + E)$ for adjacency list representation.
- **Space Complexity: $O(V)$**
  - **Visited Set**: Stores at most $V$ node identifiers, requiring $O(V)$ memory.
  - **Frontier Queue**: At any moment, the frontier contains vertices from at most two consecutive depth layers. In the worst case (e.g., a star graph or dense level), it stores $O(V)$ vertices.
  - **Parents & Levels Mappings**: Each maps at most $V$ vertices, requiring $O(V)$ memory.

#### Complete Execution Guide

1. **Default Run (Sample Graph from Question 11)**:
   ```bash
   python scripts/run_breadth_first_search.py
   ```

2. **Custom Graph via Command Line**:
   ```bash
   # Python dictionary literal
   python scripts/run_breadth_first_search.py --graph "{'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}" --start A

   # Compact adjacency format
   python scripts/run_breadth_first_search.py --graph "A: B, C; B: D; C:; D:" --start A

   # Directed edge-list format
   python scripts/run_breadth_first_search.py --graph "A->B, A->C, B->D" --start A
   ```

3. **Interactive Console Wizard**:
   ```bash
   python scripts/run_breadth_first_search.py --interactive
   ```

4. **Run Optional Challenge (Levels and Graph Depth Analysis)**:
   ```bash
   python scripts/run_breadth_first_search.py --challenge
   ```

5. **Run Benchmark Demonstration Suite (Cycles, Trees, Disconnected Graphs)**:
   ```bash
   python scripts/run_breadth_first_search.py --demo
   ```

6. **Generate and Save Graphical Visualization**:
   ```bash
   python scripts/run_breadth_first_search.py --save reports/bfs_traversal.png
   ```

7. **Display Viva Voce Theory & Reference Guide**:
   ```bash
   python scripts/run_breadth_first_search.py --viva
   ```

8. **Top-Level Package Dispatcher**:
   ```bash
   python -m src bfs --start A
   ```

9. **Interactive Jupyter Notebook**:
   Open [`notebooks/breadth_first_search.ipynb`](notebooks/breadth_first_search.ipynb) in Jupyter Lab or Antigravity IDE.

#### Viva Voce Questions & Authoritative Answers

##### 1. FIFO Queue Mechanism & `collections.deque` vs. `list`
- **Question**: Why does BFS require a FIFO queue, and why must `collections.deque` be used instead of a standard Python `list`?
- **Answer**:
  - BFS must explore states in exact order of their distance (depth) from the root. A First-In, First-Out (FIFO) queue guarantees that all nodes at level $k$ are dequeued and expanded before any node at level $k+1$ is expanded.
  - In CPython, `collections.deque` is implemented as a doubly-linked list of fixed-size contiguous memory blocks (chunks of 64 elements). Appending (`append()`) and popping from the left (`popleft()`) both execute in **$O(1)$ amortized time**.
  - In contrast, a Python `list` is a contiguous dynamic array of object references. While `list.append()` is $O(1)$ amortized, `list.pop(0)` is strictly **$O(N)$** because all remaining $N-1$ element pointers must be shifted in memory. Using `list.pop(0)` in BFS degrades overall search complexity from $O(V + E)$ to $O(V^2 + E)$.

##### 2. Completeness and Optimality of BFS
- **Question**: Is Breadth-First Search complete and optimal? Under what conditions?
- **Answer**:
  - **Completeness**: BFS is complete provided the branching factor $b$ is finite ($b < \infty$). In graph search with a `visited` set, it is guaranteed to terminate on finite graphs even if cyclic.
  - **Optimality**: BFS is optimal **if and only if all step costs are identical/uniform** (e.g., unit edge cost $c = 1$). Under uniform step costs, the shallowest goal state is guaranteed to be the least-cost path.
  - If step costs are non-uniform or arbitrary (e.g., varying road distances or terrain weights), BFS is **not optimal** because a deeper path with smaller weights might have lower total cost than a shallow path with heavy weights. In that case, **Uniform-Cost Search (UCS / Dijkstra's Algorithm)** must be used.

##### 3. Time Complexity Derivation
- **Question**: Derive the time complexity of BFS on both graph search and tree search formulations.
- **Answer**:
  - **Graph Search**: On a graph with $V$ vertices and $E$ edges using an adjacency list:
    $$\mathcal{O}(V + E)$$
    Each reachable vertex is pushed to the FIFO queue and visited set once ($O(V)$), and its outgoing adjacency list is traversed once during expansion ($\sum \text{deg}^+(u) = O(E)$).
  - **Tree Search (Branching Factor $b$, Goal Depth $d$)**:
    At depth 0: $1$ node.
    At depth 1: $b$ nodes.
    At depth 2: $b^2$ nodes.
    At depth $d$: $b^d$ nodes.
    Sum of geometric progression:
    $$T(b, d) = 1 + b + b^2 + \dots + b^d = \frac{b^{d+1} - 1}{b - 1} = \mathcal{O}(b^d)$$
    If the goal test is applied when a node is expanded rather than when generated, the algorithm explores nodes at level $d+1$, yielding $\mathcal{O}(b^{d+1})$.

##### 4. Space Complexity Bottleneck
- **Question**: Why is space complexity the primary practical limitation of BFS compared to DFS?
- **Answer**:
  - BFS has a space complexity of $\mathcal{O}(b^d)$ because every generated node must remain in memory (either in the frontier queue or the visited table). The frontier alone at level $d$ holds up to $b^d$ nodes.
  - In contrast, Depth-First Search (DFS) only needs to store the current active path from the root to the leaf plus unexpanded siblings, requiring only $\mathcal{O}(b \cdot d)$ linear memory.
  - For example, with branching factor $b = 10$ and depth $d = 8$, BFS must store $10^8$ nodes in memory ($\approx 10$ to $100\text{ GB}$ of RAM), exhausting system memory within minutes, whereas DFS requires storing only $10 \times 8 = 80$ nodes.

---

### Question 12: Depth-First Search (DFS) for Graph Traversal

- **Syllabus Area**: Uninformed Search Techniques
- **Difficulty**: 🟢 Beginner
- **Practical Type**: Implementation & Deep State-Space Exploration
- **Core AI Concept**: Depth-First Search (DFS), LIFO Frontier Stack, Backtracking, Visited Set, Cycle Prevention
- **Python Concepts Tested**: Recursion, explicit `list` stack manipulation (`append()` and `pop()`), dictionary adjacency lists
- **Expected Algorithm / Technique**: Depth-first traversal (LIFO frontier)
- **Complexity Expectation**: Time Complexity: $\mathcal{O}(V + E)$ | Space Complexity: $\mathcal{O}(V)$
- **Constraints / Conditions**: Graph may contain cycles (must prevent infinite loops).
- **University Practical Importance**: Very High
- **Optional Challenge**: Implement both iterative and recursive versions and compare output.

#### Problem Statement & Task Requirements

Implement Depth-First Search (DFS) in Python to traverse an unweighted graph. The program must explore as far as possible along each branch before backtracking. Output the exploration path.

1. **Stack or Recursion**: Implement DFS using an explicit stack (`list`) OR recursion.
2. **Cycle Avoidance**: Track explored nodes to prevent infinite loops in cyclic graphs.
3. **Neighbor Processing**: Process neighbor nodes systematically.
4. **Input Requirements**: Graph dictionary, Start node.
5. **Expected Output**: List of nodes in DFS traversal order.
6. **Optional Challenge**: Implement both iterative and recursive versions and compare output.

#### Sample Input & Output Verification

- **Sample Graph**: `{'1': ['2', '3'], '2': ['4'], '3': [], '4': []}`
- **Start Node**: `'1'`
- **Expected Traversal**: `DFS Traversal: 1, 3, 2, 4`

#### Traversal Order Nuance: Forward-Push Stack vs. Standard Recursion

A classic and insightful property of Depth-First Search is how successor node ordering impacts the visited sequence:

1. **Iterative DFS with Standard Forward-Push (Sample Output Match)**:
   - When node `'1'` expands its neighbors `['2', '3']` and pushes them onto the LIFO stack in order (`stack.append('2')`, then `stack.append('3')`), node `'3'` sits at the **top** of the stack.
   - The subsequent `stack.pop()` extracts `'3'` first, resulting in the exploration order:
     $$\mathbf{1 \to 3 \to 2 \to 4}$$
   - This matches the university problem statement's exact sample output!
2. **Standard Recursive DFS (Left-to-Right)**:
   - In standard recursive DFS, a loop `for neighbor in graph[node]` visits `'2'` before `'3'`. It explores deeply down `'2'` to `'4'`, backtracks to `'1'`, and then visits `'3'`:
     $$\mathbf{1 \to 2 \to 4 \to 3}$$
3. **Order-Preserving Iterative DFS (Reverse-Push)**:
   - If the iterative algorithm pushes neighbors in reversed order (`for neighbor in reversed(graph[node]): stack.append(neighbor)`), node `'2'` remains on top of the stack and is popped first, producing the identical sequence to standard recursion:
     $$\mathbf{1 \to 2 \to 4 \to 3}$$

Both ordering models are supported and compared side-by-side in this laboratory implementation.

#### Algorithm Walkthrough & Architectural Diagram

```mermaid
flowchart TD
    Start([Start Node: '1']) --> Init["Init LIFO Stack: [('1', None, 0)]<br/>Visited Set: set()<br/>Parents: {'1': None}"]
    Init --> Loop{"Is Stack Empty?"}
    Loop -- No --> Pop["pop() (curr_node, parent, depth)<br/>from LIFO Stack Top"]
    Pop --> CheckVis{"Is curr_node in Visited Set?"}
    CheckVis -- Yes --> Loop
    CheckVis -- No --> MarkVis["Visited.add(curr_node)<br/>Append curr_node to Traversal Order<br/>Record Discovery Timestamp d[u]"]
    MarkVis --> EvalNeighbors["Inspect Neighbors in Adj[curr_node]<br/>Identify Ancestors (Back-Edges / Cycles)"]
    EvalNeighbors --> PushLoop["For each unvisited neighbor v in Adj[curr_node]"]
    PushLoop --> PushStack["stack.append((v, curr_node, depth + 1))<br/>Parents[v] = curr_node"]
    PushStack --> PushLoop
    PushLoop -- All Neighbors Processed --> Loop
    Loop -- Yes (Stack Empty) --> Done([Return DFSResult & Complete Exploration Path])
```

#### Chronological LIFO Frontier Trace Table (Sample Graph)

| Step | Node Popped | Action | Depth | Parent | Unvisited Pushed to Stack | LIFO Stack Snapshot (Top on Right) | Visited Nodes Sequence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **1** | `EXPAND` | 0 | `ROOT` | `['2', '3']` | `['2', '3']` (Top: 3) | `['1']` |
| **2** | **3** | `EXPAND` | 1 | `1` | `[]` (None) | `['2']` (Top: 2) | `['1', '3']` |
| **3** | **2** | `EXPAND` | 1 | `1` | `['4']` | `['4']` (Top: 4) | `['1', '3', '2']` |
| **4** | **4** | `EXPAND` | 2 | `2` | `[]` (None) | `[]` (Empty) | `['1', '3', '2', '4']` |

#### Optional Challenge: Side-by-Side Comparison of Iterative vs. Recursive DFS

| Implementation Variant | Traversal Order | Step Count | Frontier Mechanism | Order Preserved? |
| :--- | :--- | :--- | :--- | :--- |
| **1. Iterative (Standard Forward-Push)** | `1, 3, 2, 4` | 4 | Explicit `list` stack (`pop()`) | No (LIFO reverses order) |
| **2. Recursive (Standard Forward-Call)** | `1, 2, 4, 3` | 8 | Implicit runtime call stack | Yes (Left-to-right) |
| **3. Iterative (Order-Preserving Reverse-Push)** | `1, 2, 4, 3` | 4 | Explicit `list` stack (`pop()`) | Yes (Matches standard recursion) |
| **4. Recursive (Reversed-Call)** | `1, 3, 2, 4` | 8 | Implicit runtime call stack | No (Matches forward-push stack) |

#### Performance & Asymptotic Complexity Analysis

- **Time Complexity: $\mathcal{O}(V + E)$**
  - **Vertex Processing**: Every vertex $v \in V$ reachable from the start node is pushed and popped from the stack at most once. Visited set operations (`add`, `in`) execute in $\mathcal{O}(1)$ average time. Total vertex time is $\mathcal{O}(V)$.
  - **Edge Processing**: When node $u$ is expanded, each of its outgoing directed edges $(u, v) \in E$ is evaluated once to check whether $v$ is an ancestor (back edge), unvisited neighbor (tree edge), or cross edge. The sum across all vertices is $\sum_{u \in V} \text{deg}^+(u) = \mathcal{O}(E)$.
  - **Total Time**: $\mathcal{O}(V + E)$ for adjacency list graph representation.
- **Space Complexity: $\mathcal{O}(V)$ (Graph Search) / $\mathcal{O}(b \cdot m)$ (Tree Search)**
  - **Graph Search**: The `visited` set and LIFO stack store at most $V$ node labels, yielding $\mathcal{O}(V)$ auxiliary memory.
  - **Tree Search Advantage**: Unlike BFS which must maintain the entire exponential frontier width $\mathcal{O}(b^d)$, DFS needs to retain only the single current branch from root to leaf, plus unexpanded sibling choices at each level, requiring strictly linear memory $\mathcal{O}(b \cdot m)$ where $b$ is the branching factor and $m$ is the maximum tree depth.

#### Complete Execution Guide

1. **Default Run (Sample Graph from Question 12: Output `1, 3, 2, 4`)**:
   ```bash
   python scripts/run_depth_first_search.py
   ```

2. **Custom Graph via Command Line**:
   ```bash
   # Python dictionary literal
   python scripts/run_depth_first_search.py --graph "{'1': ['2', '3'], '2': ['4'], '3': [], '4': []}" --start 1

   # Compact adjacency format
   python scripts/run_depth_first_search.py --graph "1: 2, 3; 2: 4; 3:; 4:" --start 1

   # Directed edge-list format
   python scripts/run_depth_first_search.py --graph "1->2, 1->3, 2->4" --start 1
   ```

3. **Optional Challenge (Iterative vs. Recursive Comparison)**:
   ```bash
   python scripts/run_depth_first_search.py --challenge
   ```

4. **Interactive Console Wizard**:
   ```bash
   python scripts/run_depth_first_search.py --interactive
   ```

5. **Run Benchmark Demonstration Suite (Cycles, Trees, Disconnected Graphs, DAGs)**:
   ```bash
   python scripts/run_depth_first_search.py --demo
   ```

6. **Generate and Save Graphical Visualization**:
   ```bash
   python scripts/run_depth_first_search.py --save reports/dfs_traversal.png
   ```

7. **Display Viva Voce Theory & Reference Guide**:
   ```bash
   python scripts/run_depth_first_search.py --viva
   ```

8. **Top-Level Package Dispatcher**:
   ```bash
   python -m src dfs --start 1
   ```

9. **Interactive Jupyter Notebook**:
   Open [`notebooks/depth_first_search.ipynb`](notebooks/depth_first_search.ipynb) in Jupyter Lab or Antigravity IDE.

#### Viva Voce Questions & Authoritative Answers

##### 1. Stack vs. Queue (LIFO vs. FIFO)
- **Question**: What is the fundamental difference between the frontier data structures of DFS and BFS, and how are they implemented in Python?
- **Answer**:
  - DFS uses a **Last-In, First-Out (LIFO) Stack**, whereas BFS uses a **First-In, First-Out (FIFO) Queue**.
  - In DFS, the most recently generated successor is expanded first, forcing search deep along a single branch before backtracking. In BFS, nodes are expanded in chronological discovery order, enforcing level-by-level exploration.
  - In Python, an iterative DFS stack is implemented directly using a built-in `list` with amortized $\mathcal{O}(1)$ `append()` and $\mathcal{O}(1)$ `pop()`. In contrast, BFS requires `collections.deque` because `list.pop(0)` is $\mathcal{O}(N)$ due to dynamic memory element shifting.

##### 2. Why DFS is Not Optimal
- **Question**: Why is Depth-First Search considered non-optimal, even when step costs are uniform?
- **Answer**:
  - An algorithm is optimal if it guarantees returning the lowest-cost solution path.
  - DFS is **not optimal** because it commits deeply down the first branch it encounters. If a goal node exists at depth 1 along branch B, but branch A plunges down 1,000 edges to another goal, DFS will explore all 1,000 edges along branch A and return that path without ever checking branch B.
  - Even under uniform unit edge costs ($c = 1$), BFS is optimal because it guarantees finding the shallowest goal, whereas DFS finds an arbitrary path dictated strictly by neighbor expansion order.

##### 3. Space Complexity Advantage of DFS over BFS
- **Question**: Why does DFS possess a decisive space complexity advantage over BFS in practical AI problems?
- **Answer**:
  - Space complexity is the primary practical bottleneck of BFS, and the greatest asset of DFS.
  - In a state space with branching factor $b = 10$ and goal depth $d = 8$:
    - **BFS** must store the entire current frontier layer: $\mathcal{O}(b^d) = 10^8$ nodes ($\approx 10\text{ to }100\text{ GB}$ of RAM), exhausting system memory within minutes.
    - **DFS** only retains the single active branch path plus unexpanded siblings at each level: $\mathcal{O}(b \cdot m) = 10 \times 8 = 80$ nodes, requiring mere kilobytes of memory.

##### 4. Completeness of DFS
- **Question**: Under what conditions is DFS complete or incomplete?
- **Answer**:
  - **Graph Search DFS (with a visited set)**: Complete on finite state spaces because the visited set prevents revisiting expanded vertices, guaranteeing termination even in cyclic graphs.
  - **Tree Search DFS (without a visited set)**: Incomplete on cyclic graphs (it gets trapped in infinite loops) and incomplete on infinite state spaces (it can plunge endlessly down infinite branches).
  - **Iterative Deepening Search (IDS)**: Solves this dilemma by pairing the $\mathcal{O}(b \cdot d)$ linear memory advantage of DFS with the completeness and optimality of BFS via progressive depth limits.

##### 5. Cycle Detection via Back-Edges (CLRS Classification)
- **Question**: How does Depth-First Search detect cycles in directed graphs?
- **Answer**:
  - In CLRS DFS edge classification, an edge $(u, v)$ is a **Back Edge** if vertex $v$ is an ancestor of vertex $u$ in the active DFS search tree (i.e., $v$ is currently on the active recursion stack / un-finished).
  - **Theorem**: A directed graph $G$ contains a directed cycle **if and only if** a DFS traversal of $G$ encounters at least one back edge.

---

### Question 13: BFS Shortest Path Constructor

- **Syllabus Area**: Uninformed Search Techniques
- **Difficulty**: 🟡 Intermediate
- **Practical Type**: Algorithm Design
- **Core AI Concept**: Path Reconstruction, Early Goal Testing vs Late Goal Testing, BFS Optimality
- **Python Concepts Tested**: Path tracking, dictionary parent mapping, `collections.deque`
- **Expected Algorithm / Technique**: Breadth-First Search (BFS) with path tracking
- **Complexity Expectation**: Time Complexity: $\mathcal{O}(V + E)$ (or $\mathcal{O}(b^d)$ with early goal test) | Space Complexity: $\mathcal{O}(V)$
- **Constraints / Conditions**: Graph is unweighted.
- **Sample Input**: Start: A, Goal: F
- **Sample Output**: Shortest Path: `['A', 'C', 'F']`
- **University Practical Importance**: Very High
- **Optional Challenge**: Print all possible shortest paths if multiple exist.

#### Problem Statement & Task Requirements

Extend the standard BFS algorithm to not just traverse a graph, but to find and return the exact shortest path (in terms of number of edges) between a given source and destination vertex. The program must reconstruct the path from the goal back to the start.

1. **Path Tracking**: The FIFO queue must store paths (lists of nodes) or maintain a parent dictionary.
2. **Immediate Termination**: Stop and return the path immediately when the goal is tested (Early Goal Testing at node generation).
3. **Robust Edge Handling**: Handle cases where no path exists (unreachable goal or disconnected components), start equals goal ($0$ edges), and graphs with cycles.
4. **Exact Shortest Path**: Guarantee the minimal edge count on unweighted graphs.
5. **Optional Challenge**: Discover and print all possible shortest paths if multiple minimal paths exist.

#### Theoretical Foundations

```mermaid
flowchart TD
  subgraph BFS_Shortest_Path_Workflow["BFS Shortest Path Constructor Workflow"]
    StartNode["Start Node v0"] --> Init["Initialize FIFO Queue with (start, 0)<br/>Initialize parents[start] = None<br/>Mark start visited"]
    Init --> QueueLoop{"Frontier Queue<br/>Empty?"}
    QueueLoop -->|Yes| NoPath["Goal Unreachable<br/>Return None / cost = -1"]
    QueueLoop -->|No| Pop["Dequeue current from Frontier"]
    Pop --> Expand["Expand unvisited neighbors in graph[current]"]
    Expand --> NeighborLoop{"For each neighbor"}
    NeighborLoop --> IsVisited{"Already<br/>Visited?"}
    IsVisited -->|Yes| Skip["Skip (Avoid Cycles)"]
    IsVisited -->|No| MarkVisited["Mark Visited<br/>parents[neighbor] = current"]
    MarkVisited --> EarlyGoalTest{"Early Goal Test:<br/>neighbor == Goal?"}
    EarlyGoalTest -->|Yes (Match)| ImmediateReturn["IMMEDIATE TERMINATION<br/>Reconstruct path: Goal back to Start<br/>Reverse list -> Return ['A', 'C', 'F']"]
    EarlyGoalTest -->|No| Enqueue["Enqueue (neighbor, depth + 1)"]
    Enqueue --> NeighborLoop
  end
```

##### 1. Path vs Traversal
- **Graph Traversal**: Systematically visits all reachable vertices in a graph to explore topology or enumerate nodes. It terminates only when the frontier becomes completely empty.
- **Path Construction**: Goal-directed navigation searching for a valid sequence of state transitions (edges) $P = \langle v_0, v_1, \dots, v_k \rangle$ connecting a specific start vertex $v_0$ to destination vertex $v_k$. Terminates immediately once the goal condition is satisfied.

##### 2. Early Goal Testing vs Late Goal Testing
- **Early Goal Testing (Node Generation Time)**: The goal test is applied immediately when a neighbor vertex is *generated* by expanding its parent, before placing it into the FIFO queue.
  - In unweighted graphs with uniform edge costs, the first time the goal node is generated is guaranteed to be at minimal depth $d^*$.
  - Immediate termination skips expanding all remaining nodes at depth $d^*$ and prevents enqueuing their children, saving up to $\mathcal{O}(b^{d+1})$ operations (an asymptotic factor of $b$ in time and memory!).
- **Late Goal Testing (Node Expansion Time)**: The goal test is deferred until the node is dequeued from the front of the frontier.
  - Necessary for non-uniform cost search (Dijkstra, Uniform-Cost Search, $A^*$) because generating a node does not guarantee the cheapest path.
  - In uniform BFS, late goal testing is strictly redundant and generates an entire extra level of vertices.

##### 3. Why BFS Finds the Shortest Path in Unweighted Graphs
- In unweighted graphs, every edge has uniform unit cost ($c = 1$). Path cost is strictly identical to path depth (number of edges).
- The FIFO queue guarantees monotonic non-decreasing exploration: all vertices at depth $d$ are expanded before any vertex at depth $d + 1$.
- **Inductive Optimality Proof**: Let $d^*$ be the length of the shortest path from start to goal. BFS explores all vertices at depth $< d^*$ before any vertex at depth $d^*$. Therefore, no path of length $< d^*$ exists (or goal would have been found earlier), and no path of length $> d^*$ can ever be returned first. Thus, the first discovered path to the goal is guaranteed to be optimal.

##### 4. Path Reconstruction: Parent Dictionary vs Queue-Stored Paths
| Dimension | Parent Dictionary Backtracking (Recommended) | Storing Full Paths in Queue |
| --- | --- | --- |
| **Data Structure** | `parents: Dict[str, Optional[str]]` mapping child $\to$ parent | `frontier: deque[List[str]]` storing full lists |
| **Auxiliary Memory** | $\mathcal{O}(V)$ strictly linear space | $\mathcal{O}(d \cdot b^d)$ due to path list replication |
| **Reconstruction Process** | Walk backward from goal to start: $v \leftarrow \text{parents}[v]$, then reverse list | Path is already stored in the dequeued item |
| **Academic / Industrial Standard** | Standard AIMA & CLRS graph search design | Educational demonstration / prototype shortcut |

#### Step-by-Step FIFO Frontier Execution Trace (Sample Input)

Graph: `{'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'], 'D': ['E'], 'E': ['F'], 'F': []}`  
Start Node: `A` | Goal Node: `F`

| Step | Action | Node | Depth | Candidate Neighbor | Goal Test | Frontier Queue Snapshot | Explanation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `START` | `A` | 0 | - | - | `[A]` | Initialized FIFO frontier with start node `A` |
| **2** | `EXPAND` | `A` | 0 | - | - | `[]` | Popped `A` from frontier; evaluating outgoing edges |
| **3** | `ENQUEUE` | `A` | 1 | `B` | `TEST` | `[B]` | Generated unvisited neighbor `B`; goal test False; enqueued |
| **4** | `ENQUEUE` | `A` | 1 | `C` | `TEST` | `[B, C]` | Generated unvisited neighbor `C`; goal test False; enqueued |
| **5** | `EXPAND` | `B` | 1 | - | - | `[C]` | Popped `B` from frontier; evaluating outgoing edges |
| **6** | `ENQUEUE` | `B` | 2 | `D` | `TEST` | `[C, D]` | Generated neighbor `D`; goal test False; enqueued |
| **7** | `ENQUEUE` | `B` | 2 | `E` | `TEST` | `[C, D, E]` | Generated neighbor `E`; goal test False; enqueued |
| **8** | `EXPAND` | `C` | 1 | - | - | `[D, E]` | Popped `C` from frontier; evaluating outgoing edges |
| **9** | `EARLY_GOAL_FOUND` | `C` | 2 | `F` | **`MATCH`** | `[D, E, F]` | Generated neighbor `F`; goal test **MATCH**! Immediate termination! |

**Backward Path Reconstruction**:  
- Start at Goal `F` $\to$ `parents['F'] = 'C'` $\to$ `parents['C'] = 'A'` $\to$ `parents['A'] = None (Start)`.  
- Backward sequence: `['F', 'C', 'A']`.  
- Reversed output: **`['A', 'C', 'F']`** (Cost: 2 edges).

#### Early Goal Testing vs Late Goal Testing Comparison

Running both strategies on the sample graph demonstrates the concrete efficiency advantage of Early Goal Testing:

| Metric | Early Goal Testing (Generation Time) | Late Goal Testing (Expansion Time) | Efficiency Savings |
| --- | :---: | :---: | :---: |
| **Shortest Path Discovered** | `['A', 'C', 'F']` | `['A', 'C', 'F']` | Exact same optimal path |
| **Path Cost (Edges)** | $2$ edges | $2$ edges | Guaranteed minimal hops |
| **Nodes Expanded** | **$3$ nodes** (`A`, `B`, `C`) | **$6$ nodes** (`A`, `B`, `C`, `D`, `E`, `F`) | **50% fewer expansions!** |
| **Total Step Records** | $9$ steps | $17$ steps | Terminated 8 steps earlier |

#### Optional Challenge: Discovering All Possible Shortest Paths

In symmetric graphs or multi-route topologies, multiple distinct paths share the identical minimum edge count.
- **Diamond Graph** (`A -> B -> D`, `A -> C -> D`):
  - Path #1: `A -> B -> D` (Length: 2 edges)
  - Path #2: `A -> C -> D` (Length: 2 edges)
- **Grid Graph** (`A -> B/C -> D/E/F -> G`):
  - Discovers all paths sharing minimal 3-edge length using multi-parent mapping.

#### Complete Execution Guide

1. **Default Run (Sample Graph from Question 13: Output `['A', 'C', 'F']`)**:
   ```bash
   python scripts/run_bfs_shortest_path.py
   ```

2. **Custom Start and Goal via CLI**:
   ```bash
   python scripts/run_bfs_shortest_path.py --start A --goal F
   ```

3. **Custom Graph via Command Line**:
   ```bash
   # Python dictionary literal
   python scripts/run_bfs_shortest_path.py --graph "{'A': ['B', 'C'], 'B': ['D'], 'C': ['F'], 'D': [], 'F': []}" --start A --goal F

   # Compact adjacency format
   python scripts/run_bfs_shortest_path.py --graph "A: B, C; B: D; C: F; D:; F:" --start A --goal F
   ```

4. **Run Optional Challenge (All Possible Shortest Paths)**:
   ```bash
   python scripts/run_bfs_shortest_path.py --all-paths
   ```

5. **Compare Early vs Late Goal Testing**:
   ```bash
   python scripts/run_bfs_shortest_path.py --compare-goals
   ```

6. **Compare Parent Dictionary vs Queue Paths Tracking**:
   ```bash
   python scripts/run_bfs_shortest_path.py --compare-modes
   ```

7. **Run Standard AI Benchmark Battery**:
   ```bash
   python scripts/run_bfs_shortest_path.py --demo
   ```

8. **Save Graphical Path Visualization**:
   ```bash
   python scripts/run_bfs_shortest_path.py --save reports/bfs_shortest_path.png
   ```

9. **Interactive Console Mode**:
   ```bash
   python scripts/run_bfs_shortest_path.py --interactive
   ```

10. **Module Dispatcher Syntax**:
    ```bash
    python -m src bfs-shortest-path --start A --goal F
    ```

11. **Display Viva Voce Theory Guide**:
    ```bash
    python scripts/run_bfs_shortest_path.py --viva
    ```

#### University Practical Viva Voce Preparation Guide

##### 1. Path vs Traversal
- **Question**: What is the key difference between a Graph Traversal and a Shortest Path Constructor?
- **Answer**:
  - A **Graph Traversal** (e.g. standard BFS or DFS) systematically explores all reachable vertices in an entire connected component to enumerate nodes or inspect topology. It terminates only when the frontier queue is empty.
  - A **Shortest Path Constructor** is a goal-directed search designed to find a specific sequence of edges connecting a designated start node $v_0$ to a destination goal $v_k$ with minimal step cost. It terminates immediately once the goal is tested.

##### 2. Early Goal Testing vs Late Goal Testing
- **Question**: What is the difference between Early Goal Testing and Late Goal Testing, and why does BFS use early testing?
- **Answer**:
  - **Early Goal Testing**: The goal condition is tested as soon as a successor node is generated by expanding its parent (before inserting into the FIFO queue). Because all edges in an unweighted graph have uniform cost ($c = 1$), the first time the goal is generated, it is guaranteed to be at minimal depth. Immediate termination prevents generating up to $b^{d+1}$ children and avoids expanding remaining nodes at depth $d$, achieving a $b$-factor reduction in time and space.
  - **Late Goal Testing**: The goal condition is tested only when the node is popped from the front of the queue. This is required in algorithms with non-uniform edge weights (Dijkstra / Uniform-Cost Search / $A^*$), but is wasteful in unweighted BFS.

##### 3. Why BFS Finds the Shortest Path in Unweighted Graphs
- **Question**: Why is BFS guaranteed to find the shortest path in an unweighted graph?
- **Answer**:
  - In an unweighted graph, all edge weights are equal ($c = 1$), so path cost equals edge count (depth $d$).
  - A FIFO queue guarantees that vertices are expanded in non-decreasing order of their distance from the start node: depth 0, then depth 1, then depth 2, ..., depth $d$.
  - By mathematical induction, BFS explores every vertex at depth $< d^*$ before any vertex at depth $d^*$. Therefore, the first path constructed to the goal must have minimal edge count.

##### 4. Path Reconstruction Mechanics
- **Question**: Compare Parent Dictionary Backtracking with Storing Full Paths in the Queue.
- **Answer**:
  - **Parent Dictionary Backtracking**: Maintains `parents[child] = parent`. Each node stores one pointer, requiring strictly $\mathcal{O}(V)$ auxiliary memory. When the goal is reached, the path is reconstructed by walking backward from goal to start in $\mathcal{O}(d)$ time.
  - **Queue-Stored Paths**: The queue stores full path lists (`[['A'], ['A', 'B'], ['A', 'C']]`). Intuitive, but duplicates path prefixes, consuming $\mathcal{O}(d \cdot b^d)$ memory.

##### 5. Handling Unreachable Goals
- **Question**: How does the algorithm detect and handle cases where no path exists?
- **Answer**:
  - If all reachable vertices in the start node's component are explored and the FIFO queue becomes empty without satisfying the goal test, the algorithm returns `None` with `goal_found=False` and `cost_edges=-1`.

---

---

## Getting Started

### 1. Requirements & Setup

Ensure Python 3 is installed. To install dependencies for notebooks and experimentation:

```bash
pip install -r requirements.txt
```

### 2. Run the Starter Script

From the project root:

```bash
python scripts/smoke_test.py
```

### 3. Run Automated Tests

Execute the comprehensive test suite (including ExperimentRunner and Agent tests):

```bash
python -m unittest discover -s tests
```

Or using pytest:

```bash
pytest
```

### 4. Interactive Jupyter Notebooks

Launch Jupyter to explore interactive experiments:

```bash
jupyter lab notebooks/multi_agent_simulation.ipynb
```
or open directly in your IDE (VS Code / Antigravity).
