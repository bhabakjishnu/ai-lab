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
  Workflow -->|Script Execution| Script[test.py]
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
│   ├── model_based_reflex_agent.ipynb   # Question 3: Model-Based Reflex Agent simulation & Viva guide
│   ├── randomized_reflex_agent.ipynb    # Question 2: Randomized Reflex Agent simulation & Viva guide
│   ├── simple_reflex_vacuum_agent.ipynb # Question 1: Simple Reflex Vacuum Agent simulation & Viva guide
│   └── starter_experiment.ipynb         # Interactive experimentation starter notebook
├── src/
│   ├── __init__.py                      # Package initialization & public exports
│   ├── experiment.py                    # Core experiment metrics and simulation runner
│   ├── model_based_agent.py             # Question 3: Model-Based Reflex Agent & 2x2 Grid Environment
│   ├── randomized_agent.py              # Question 2: Randomized Reflex Agent & 1D Grid Environment
│   └── vacuum_agent.py                  # Question 1: Simple Reflex Vacuum Agent & Environments
├── tests/
│   ├── __init__.py                      # Test package marker
│   ├── test_experiment.py               # Unit test suite for experiment runner
│   ├── test_model_based_agent.py        # Unit tests for Question 3 model-based agent & 2x2 grid
│   ├── test_randomized_agent.py         # Unit tests for Question 2 randomized agent & 1D grid
│   └── test_vacuum_agent.py             # Unit tests for vacuum agent reflex rules & simulation traces
├── .gitignore                           # Comprehensive ignores (Python, Jupyter, venv, IDE)
├── README.md                            # Project documentation and practical guide
├── requirements.txt                     # Dependencies for notebooks and ML libraries
├── run_model_based_agent.py             # CLI & interactive runner for Question 3 (2x2 Grid Simulation)
├── run_randomized_agent.py              # CLI & interactive runner for Question 2 (1D Grid Simulation)
├── run_vacuum_agent.py                  # CLI & interactive runner for Question 1 (2-Square & 3x3 Grid)
└── test.py                              # Starter entry point and verification script
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
   python run_vacuum_agent.py
   ```

2. **Custom Location & Square States**:
   ```bash
   python run_vacuum_agent.py --location B --status-a Dirty --status-b Dirty --steps 5
   ```

3. **Interactive Mode**:
   ```bash
   python run_vacuum_agent.py --interactive
   ```

4. **Optional Challenge (3x3 Grid with Randomized Dirt Placement)**:
   ```bash
   python run_vacuum_agent.py --grid --steps 10
   ```

5. **Display Viva Voce Q&A and Theory Guide**:
   ```bash
   python run_vacuum_agent.py --viva
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
   python run_randomized_agent.py --seed 58
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
   python run_randomized_agent.py --state "1 0 1 1 0" --start-index 1
   ```

3. **Interactive Mode**:
   ```bash
   python run_randomized_agent.py --interactive
   ```

4. **Monte Carlo Empirical Benchmark (100 Trials)**:
   ```bash
   python run_randomized_agent.py --benchmark --trials 100
   ```

5. **Display Viva Voce Theory Guide**:
   ```bash
   python run_randomized_agent.py --viva
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
   python run_model_based_agent.py
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
   python run_model_based_agent.py --grid "Dirty Clean, Dirty Dirty" --start-row 0 --start-col 0
   ```

3. **Optional Challenge (Stochastic Sensor Failure)**:
   ```bash
   python run_model_based_agent.py --sensor-failure 0.2 --seed 42
   ```

4. **Interactive Walkthrough Mode**:
   ```bash
   python run_model_based_agent.py --interactive
   ```

5. **Display Viva Voce Reference & Theory Guide**:
   ```bash
   python run_model_based_agent.py --viva
   ```

6. **Interactive Jupyter Notebook**:
   Open [`notebooks/model_based_reflex_agent.ipynb`](notebooks/model_based_reflex_agent.ipynb) in Jupyter Lab or Antigravity IDE.


## Getting Started

### 1. Requirements & Setup

Ensure Python 3 is installed. To install dependencies for notebooks and experimentation:

```bash
pip install -r requirements.txt
```

### 2. Run the Starter Script

From the project root:

```bash
python test.py
```

### 3. Run Automated Tests

Execute the comprehensive test suite (including ExperimentRunner and Vacuum Agent tests):

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
jupyter lab notebooks/simple_reflex_vacuum_agent.ipynb
```
or open directly in your IDE (VS Code / Antigravity).
