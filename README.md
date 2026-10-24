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
├── data/
│   ├── raw/                 # Original immutable datasets
│   └── processed/           # Cleaned & transformed datasets
├── notebooks/
│   └── starter_experiment.ipynb  # Interactive experimentation notebook
├── src/
│   ├── __init__.py          # Package initialization
│   └── experiment.py        # Core experiment metrics and simulation runner
├── tests/
│   ├── __init__.py          # Test package marker
│   └── test_experiment.py   # Unit test suite covering edge cases
├── .gitignore               # Comprehensive ignores (Python, Jupyter, venv, IDE)
├── README.md                # Project documentation and guide
├── requirements.txt         # Dependencies for notebooks and ML libraries
└── test.py                  # Starter entry point and verification script
```

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

Execute the test suite to verify module behavior:

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
jupyter lab notebooks/starter_experiment.ipynb
```
or open `notebooks/starter_experiment.ipynb` directly in your IDE (VS Code / Antigravity).
