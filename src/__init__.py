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
from src.model_based_agent import (
    ACTION_SHUTDOWN,
    STATUS_CLEAN,
    STATUS_DIRTY,
    TERMINATION_MESSAGE,
    ModelAgentStepRecord,
    ModelBasedAgent,
    ModelSimulationResult,
    TwoByTwoGridEnvironment,
    run_model_based_simulation,
)
from src.goal_based_agent import (
    ACTION_DOWN,
    ACTION_UP,
    DIAGONAL_DELTAS,
    ORTHOGONAL_DELTAS,
    GoalBasedNavigationAgent,
    GoalSimulationResult,
    GridEnvironment,
    NavigationStepRecord,
    euclidean_distance,
    generate_random_coordinate,
    get_successors,
    is_diagonal_movement,
    manhattan_distance,
    run_goal_based_simulation,
    validate_orthogonal_transition,
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
    # Question 3: Model-Based Reflex Agent
    "ModelBasedAgent",
    "TwoByTwoGridEnvironment",
    "ModelAgentStepRecord",
    "ModelSimulationResult",
    "run_model_based_simulation",
    "STATUS_CLEAN",
    "STATUS_DIRTY",
    "ACTION_SHUTDOWN",
    "TERMINATION_MESSAGE",
    # Question 4: Goal-Based Navigation Agent
    "GoalBasedNavigationAgent",
    "GridEnvironment",
    "NavigationStepRecord",
    "GoalSimulationResult",
    "run_goal_based_simulation",
    "manhattan_distance",
    "euclidean_distance",
    "get_successors",
    "is_diagonal_movement",
    "validate_orthogonal_transition",
    "generate_random_coordinate",
    "ACTION_UP",
    "ACTION_DOWN",
    "ORTHOGONAL_DELTAS",
    "DIAGONAL_DELTAS",
]
