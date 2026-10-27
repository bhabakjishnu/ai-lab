"""AI Lab core package.

Provides modular utilities, experiment runners, and evaluation tools
for artificial intelligence and machine learning experimentation.
"""

__version__ = "0.1.0"

from src.randomized_agent import (
    ACTION_CLEAN,
    ROOM_CLEAN,
    ROOM_DIRTY,
    AgentStepRecord,
    OneDimensionalGridEnvironment,
    RandomizedReflexAgent,
    SimulationResult,
    run_randomized_reflex_simulation,
)
from src.vacuum_agent import (
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_SUCK,
    CLEAN,
    DIRTY,
    LOCATION_A,
    LOCATION_B,
    GridVacuumEnvironment,
    SimpleReflexVacuumAgent,
    StepRecord,
    TwoSquareEnvironment,
)

__all__ = [
    # Question 1: Simple Reflex Agent
    "SimpleReflexVacuumAgent",
    "TwoSquareEnvironment",
    "StepRecord",
    "GridVacuumEnvironment",
    "CLEAN",
    "DIRTY",
    "ACTION_SUCK",
    "ACTION_RIGHT",
    "ACTION_LEFT",
    "ACTION_NOOP",
    "LOCATION_A",
    "LOCATION_B",
    # Question 2: Randomized Reflex Agent
    "RandomizedReflexAgent",
    "OneDimensionalGridEnvironment",
    "AgentStepRecord",
    "SimulationResult",
    "run_randomized_reflex_simulation",
    "ROOM_CLEAN",
    "ROOM_DIRTY",
    "ACTION_CLEAN",
]

