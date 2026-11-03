"""Environment Type Classifier (Question 8).

This module implements an expert knowledge-representation classifier that categorizes
AI problems and agent environments based on their foundational environmental
dimensions, as established in Stuart Russell & Peter Norvig's seminal textbook
'Artificial Intelligence: A Modern Approach' (AIMA, Chapter 2).

Core AI Concepts:
-----------------
An agent's task environment is formally specified using the PEAS framework:
  - Performance Measure: The metric evaluating the agent's degree of success.
  - Environment: The external context and world in which the agent operates.
  - Actuators: The mechanism through which the agent acts upon the environment.
  - Sensors: The perceptual devices through which the agent receives observations.

The characteristics of the environment dictate the complexity and architectural
requirements of the agent program (Simple Reflex, Model-Based, Goal-Based,
Utility-Based, or Learning Agent).

The 5 Core Environmental Dimensions:
-------------------------------------
1. Observability (Fully Observable vs. Partially Observable):
   - Fully Observable (True): The agent's sensors grant complete access to the
     entire state of the environment at each point in time (e.g., Chess, Backgammon).
   - Partially Observable (False): Sensor noise, occlusions, or hidden variables
     obscure aspects of the state (e.g., Poker card deck, Medical Diagnosis internal symptoms).

2. Determinism (Deterministic vs. Stochastic):
   - Deterministic (True): The next state is completely determined by the current
     state and the action executed by the agent (e.g., Chess without clock, Crossword).
   - Stochastic (False): Uncertainty or randomness governs transitions, or moves
     involve chance elements like dice rolls or market variations (e.g., Backgammon dice, Poker deck).

3. Episodic Nature (Episodic vs. Sequential):
   - Episodic (True): The agent's experience is divided into atomic episodes; the
     current action does not affect future episodes (e.g., Image Classification, Part-Picking).
   - Sequential (False): The current decision affects all future decisions and states;
     short-term actions have long-term consequences (e.g., Chess, Taxi Driving).

4. Statics (Static vs. Dynamic):
   - Static (True): The environment does not change while the agent is deliberating;
     the agent need not worry about the passage of wall-clock time (e.g., Crossword puzzle).
   - Dynamic (False): The environment changes continuously while the agent ponders;
     or time limits impose penalties (e.g., Taxi Driving, Games with a clock).

5. Discreteness (Discrete vs. Continuous):
   - Discrete (True): Finite or countably discrete states, distinct time steps,
     and discrete actions/percepts (e.g., Chess board positions, Tic-Tac-Toe).
   - Continuous (False): Smoothly varying, real-valued state variables, continuous
     time, and continuous actuator controls (e.g., Taxi speed/steering, Medical vitals).

Algorithm & Complexity:
-----------------------
- Technique: Hash-table / dictionary-based Knowledge Representation lookup.
- Tuple key: (Observability, Determinism, Episodic, Statics, Discreteness) -> 5-boolean tuple.
- Time Complexity: O(1) dictionary key lookup for exact match.
- Space Complexity: O(K) where K is the number of distinct environment entries stored.
- Nearest-Neighbor Fallback: O(K) Hamming distance metric when an exact match fails.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# Canonical Property Index Positions
INDEX_OBSERVABLE: int = 0
INDEX_DETERMINISTIC: int = 1
INDEX_EPISODIC: int = 2
INDEX_STATIC: int = 3
INDEX_DISCRETE: int = 4

# Property Dimension Names
PROPERTY_NAMES: Tuple[str, ...] = (
    "Observability",
    "Determinism",
    "Episodic",
    "Statics",
    "Discreteness",
)

# Human-Readable Labels for True/False States
PROPERTY_LABELS: Dict[int, Tuple[str, str]] = {
    INDEX_OBSERVABLE: ("Fully Observable", "Partially Observable"),
    INDEX_DETERMINISTIC: ("Deterministic", "Stochastic"),
    INDEX_EPISODIC: ("Episodic", "Sequential"),
    INDEX_STATIC: ("Static", "Dynamic"),
    INDEX_DISCRETE: ("Discrete", "Continuous"),
}

# Type alias for the 5-element boolean property tuple
EnvironmentTuple = Tuple[bool, bool, bool, bool, bool]


@dataclass(frozen=True)
class PEASDescription:
    """Structured representation of the PEAS elements for an environment."""

    performance_measure: str
    environment: str
    actuators: str
    sensors: str

    def to_dict(self) -> Dict[str, str]:
        """Convert the PEAS description to a standard dictionary."""
        return {
            "Performance Measure": self.performance_measure,
            "Environment": self.environment,
            "Actuators": self.actuators,
            "Sensors": self.sensors,
        }


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Comprehensive metadata for a classified task environment."""

    name: str
    properties: EnvironmentTuple
    category: str
    description: str
    peas: PEASDescription
    agent_type: str = "Single-Agent"

    @property
    def property_summary(self) -> str:
        """Return a formatted string of the 5 environmental dimension states."""
        labels = [
            PROPERTY_LABELS[i][0] if self.properties[i] else PROPERTY_LABELS[i][1]
            for i in range(5)
        ]
        return ", ".join(labels)


@dataclass
class ClassificationResult:
    """Result of classifying an environment property tuple."""

    input_tuple: EnvironmentTuple
    is_exact_match: bool
    matched_environment: Optional[str]
    metadata: Optional[EnvironmentMetadata] = None
    closest_matches: List[Tuple[EnvironmentMetadata, int, List[str]]] = field(
        default_factory=list
    )
    message: str = ""


# ==============================================================================
# Comprehensive Standard AI Textbook Knowledge Base (Russell & Norvig Table 2.1)
# ==============================================================================

KNOWLEDGE_BASE: Dict[EnvironmentTuple, EnvironmentMetadata] = {
    # 1. Backgammon: (Observable, Stochastic, Sequential, Dynamic/clock, Discrete)
    (True, False, False, False, True): EnvironmentMetadata(
        name="Backgammon",
        properties=(True, False, False, False, True),
        category="Board Game with Dice (Clock/Timer)",
        description=(
            "Fully observable board with complete view of all pieces and dice; "
            "stochastic due to probabilistic dice rolls; sequential moves influencing "
            "future board layouts; dynamic when played with tournament clocks; discrete points and turns."
        ),
        peas=PEASDescription(
            performance_measure="Win rate, margin of victory, points scored",
            environment="Backgammon board, dice, checkers, opponent",
            actuators="Move checkers along points, double the stakes",
            sensors="Visual recognition of board positions and dice roll numbers",
        ),
        agent_type="Multi-Agent (Competitive)",
    ),
    # 2. Backgammon (without clock / purely static deliberation)
    (True, False, False, True, True): EnvironmentMetadata(
        name="Backgammon (without clock)",
        properties=(True, False, False, True, True),
        category="Board Game with Dice (Untimed)",
        description=(
            "Fully observable board and dice; stochastic transitions governed by dice probabilities; "
            "sequential state trajectory; static since board state remains unchanged during deliberation; "
            "discrete board points and turns."
        ),
        peas=PEASDescription(
            performance_measure="Win percentage, pip count reduction",
            environment="Backgammon board, checkers, dice, human or AI opponent",
            actuators="Checker placement, dice rolling, doubling cube",
            sensors="Board state sensor, dice digit detector",
        ),
        agent_type="Multi-Agent (Competitive)",
    ),
    # 3. Chess with a clock (Semi-dynamic / Dynamic)
    (True, True, False, False, True): EnvironmentMetadata(
        name="Chess (with clock)",
        properties=(True, True, False, False, True),
        category="Perfect-Information Strategic Game (Timed)",
        description=(
            "Fully observable 8x8 grid; deterministic piece transitions; sequential tactical moves; "
            "dynamic because the clock continuously depletes the agent's remaining time; discrete squares and turns."
        ),
        peas=PEASDescription(
            performance_measure="Checkmate, material score, tournament rating, win/loss",
            environment="64-square chessboard, 32 pieces, clock, opponent",
            actuators="Legal piece movements, offer/accept draw, resign",
            sensors="Current board configuration, clock timer readings",
        ),
        agent_type="Multi-Agent (Competitive)",
    ),
    # 4. Chess without a clock / Crossword Puzzle
    (True, True, False, True, True): EnvironmentMetadata(
        name="Chess (without clock) / Crossword Puzzle",
        properties=(True, True, False, True, True),
        category="Perfect-Information Strategic / Word Puzzle (Untimed)",
        description=(
            "Fully observable board or grid; deterministic letter/move insertions; sequential problem solving; "
            "static environment that waits indefinitely for agent decisions; discrete character/board positions."
        ),
        peas=PEASDescription(
            performance_measure="Correct letters/words filled, minimal clues used, checkmate",
            environment="Crossword grid, word clues / 8x8 chessboard",
            actuators="Write letter/word into grid, move chess piece",
            sensors="Character array reader, board vision sensor",
        ),
        agent_type="Single-Agent (Crossword) / Multi-Agent (Chess)",
    ),
    # 5. Poker (with clock / dynamic timer) / Interactive English Tutor
    (False, False, False, False, True): EnvironmentMetadata(
        name="Poker / Interactive English Tutor",
        properties=(False, False, False, False, True),
        category="Imperfect-Information Game / Interactive Tutoring",
        description=(
            "Partially observable due to concealed opponent hole cards or hidden student cognitive states; "
            "stochastic card distribution and human reactions; sequential betting rounds or pedagogical progression; "
            "dynamic with countdown shot-clocks; discrete cards, bet values, or vocabulary units."
        ),
        peas=PEASDescription(
            performance_measure="Chips accumulated, expected value / student mastery rate and test score improvement",
            environment="Card table, deck of cards, opponents / Student learner, exercises, curriculum",
            actuators="Fold, Check, Call, Bet, Raise / Display lesson, present question, give feedback",
            sensors="Hole cards, community cards, pot size, timer / Student text answers, latency",
        ),
        agent_type="Multi-Agent (Competitive or Pedagogical)",
    ),
    # 6. Poker (without clock / untimed)
    (False, False, False, True, True): EnvironmentMetadata(
        name="Poker (without clock)",
        properties=(False, False, False, True, True),
        category="Imperfect-Information Card Game (Untimed)",
        description=(
            "Partially observable due to hidden cards; stochastic card distribution; sequential betting rounds; "
            "static while player considers pot odds without time penalties; discrete card ranks and chips."
        ),
        peas=PEASDescription(
            performance_measure="Long-term expected chip balance",
            environment="Poker table, hidden cards, community cards, opponents",
            actuators="Bet, Call, Raise, Fold",
            sensors="Visible cards, chip counts, action history",
        ),
        agent_type="Multi-Agent (Competitive)",
    ),
    # 7. Medical Diagnosis System / Taxi Driving / Refinery Controller
    (False, False, False, False, False): EnvironmentMetadata(
        name="Medical Diagnosis / Taxi Driving",
        properties=(False, False, False, False, False),
        category="Complex Real-World Cyber-Physical System",
        description=(
            "Partially observable (internal patient physiology, traffic occlusions, or pipe flows hidden); "
            "stochastic patient reactions, traffic dynamics, or thermal fluctuations; "
            "sequential treatment plans or driving paths; dynamic environment changing in real time; "
            "continuous physiological vitals or steering angle/velocity controls."
        ),
        peas=PEASDescription(
            performance_measure="Healthy patient recovery / Safe, fast, legal trip / Yield purity and safety",
            environment="Hospital ward, patient symptoms / Urban roads, pedestrians, traffic / Chemical plant",
            actuators="Prescribe medicine, order tests, surgery / Steering, braking, acceleration / Valve throttles",
            sensors="Lab test results, symptoms, ECG / Cameras, LIDAR, GPS / Thermocouples, pressure gauges",
        ),
        agent_type="Single-Agent (Diagnosis) / Multi-Agent (Taxi in traffic)",
    ),
    # 8. Automated Vehicle Simulator (Deterministic Physics)
    (False, True, False, False, False): EnvironmentMetadata(
        name="Automated Vehicle Simulator (Deterministic Physics)",
        properties=(False, True, False, False, False),
        category="Autonomous Robotics Simulation",
        description=(
            "Partially observable sensor range; deterministic physics simulation; sequential navigation trajectory; "
            "dynamic world with moving obstacles; continuous vehicle control inputs."
        ),
        peas=PEASDescription(
            performance_measure="Waypoint arrival time, collision avoidance, fuel efficiency",
            environment="Simulated road network, friction models, obstacle paths",
            actuators="Continuous steering torque, brake pressure, throttle",
            sensors="Simulated LIDAR point cloud, virtual camera feed",
        ),
        agent_type="Single-Agent / Multi-Agent",
    ),
    # 9. Part-Picking Robot (Episodic, Continuous, Dynamic)
    (False, False, True, False, False): EnvironmentMetadata(
        name="Part-Picking Robot",
        properties=(False, False, True, False, False),
        category="Industrial Robotics & Manipulation",
        description=(
            "Partially observable bin with occluded parts; stochastic gripper grasp success; episodic handling of "
            "independent parts; dynamic conveyer belt moving parts; continuous robotic arm trajectory."
        ),
        peas=PEASDescription(
            performance_measure="Parts correctly sorted per hour, zero part damage",
            environment="Conveyer belt, parts bin, sorting trays",
            actuators="Robotic arm joints, suction cup or gripper solenoid",
            sensors="3D depth camera, proximity infrared, joint encoders",
        ),
        agent_type="Single-Agent",
    ),
    # 10. Image Analysis / Part Inspection (Episodic, Discrete, Static)
    (True, True, True, True, True): EnvironmentMetadata(
        name="Image Analysis / Quality Inspection (Discrete)",
        properties=(True, True, True, True, True),
        category="Automated Computer Vision & Defect Detection",
        description=(
            "Fully observable image pixel matrix; deterministic processing pipeline; episodic classification of "
            "individual snapshots; static image buffer; discrete pixel values and categorical classification labels."
        ),
        peas=PEASDescription(
            performance_measure="Classification accuracy, F1-score, processing latency",
            environment="High-resolution digital image buffer, pixel array",
            actuators="Output classification label (Defective / Acceptable)",
            sensors="Digital CMOS sensor, uncompressed image file reader",
        ),
        agent_type="Single-Agent",
    ),
    # 11. Image Analysis (Continuous features / Pixel intensities)
    (True, True, True, True, False): EnvironmentMetadata(
        name="Satellite Image Radiometry (Continuous)",
        properties=(True, True, True, True, False),
        category="Remote Sensing & Earth Observation",
        description=(
            "Fully observable satellite scene; deterministic spectral reflectance; episodic frame analysis; "
            "static captured imagery; continuous spectral irradiance measurements."
        ),
        peas=PEASDescription(
            performance_measure="Mean squared error in canopy density, elevation accuracy",
            environment="Multispectral satellite tile dataset",
            actuators="Segment map generation, continuous vegetation index output",
            sensors="Hyperspectral radiometer, infrared sensor array",
        ),
        agent_type="Single-Agent",
    ),
    # 12. Battleship Game (Simplified)
    (False, True, False, True, True): EnvironmentMetadata(
        name="Battleship (Simplified)",
        properties=(False, True, False, True, True),
        category="Hidden Grid Deduction Game",
        description=(
            "Partially observable enemy ship placement; deterministic hit/miss mechanics; sequential torpedo strikes; "
            "static board during player turns; discrete grid coordinates (e.g. A1-J10)."
        ),
        peas=PEASDescription(
            performance_measure="Minimal turns to sink all enemy ships, win rate",
            environment="10x10 coordinate grid, hidden enemy ships, own fleet",
            actuators="Call target coordinate (x, y)",
            sensors="Hit, miss, or sunken ship feedback report",
        ),
        agent_type="Multi-Agent (Competitive)",
    ),
}


# ==============================================================================
# Helper Functions: Parsing and Validation
# ==============================================================================

def parse_boolean(value: Any) -> bool:
    """Parse a flexible user input string or value into a boolean.

    Accepts standard natural language responses:
      - Truthy: True, 'true', 't', 'yes', 'y', '1', 1, 'observable', 'deterministic',
                'episodic', 'static', 'discrete'
      - Falsy:  False, 'false', 'f', 'no', 'n', '0', 0, 'partially', 'stochastic',
                'sequential', 'dynamic', 'continuous'

    Args:
        value: Raw input from prompt or function argument.

    Returns:
        Canonical bool value.

    Raises:
        ValueError: If input cannot be unambiguously resolved to a boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"Numeric boolean input must be 0 or 1, got {value}.")

    if not isinstance(value, str):
        raise ValueError(f"Unsupported input type {type(value).__name__}: {value}")

    cleaned = value.strip().lower()

    truthy_tokens = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "observable",
        "fully observable",
        "deterministic",
        "episodic",
        "static",
        "discrete",
    }
    falsy_tokens = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "partially observable",
        "partially",
        "partial",
        "stochastic",
        "sequential",
        "dynamic",
        "continuous",
    }

    if cleaned in truthy_tokens:
        return True
    if cleaned in falsy_tokens:
        return False

    raise ValueError(
        f"Cannot parse '{value}' as boolean. "
        f"Valid affirmative: {sorted(list(truthy_tokens)[:5])}... "
        f"Valid negative: {sorted(list(falsy_tokens)[:5])}..."
    )


def parse_environment_tuple(
    inputs: Sequence[Any],
) -> EnvironmentTuple:
    """Validate and convert an arbitrary sequence of 5 inputs to an EnvironmentTuple.

    Args:
        inputs: Sequence of 5 boolean-convertible elements:
                [Observability, Determinism, Episodic, Statics, Discreteness].

    Returns:
        EnvironmentTuple (Tuple[bool, bool, bool, bool, bool]).

    Raises:
        ValueError: If length is not exactly 5 or any element fails boolean parsing.
    """
    if len(inputs) != 5:
        raise ValueError(
            f"Expected exactly 5 environment properties (got {len(inputs)}): "
            f"[Observability, Determinism, Episodic, Statics, Discreteness]."
        )

    parsed = tuple(parse_boolean(item) for item in inputs)
    return (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4])


def calculate_hamming_distance(
    tuple1: EnvironmentTuple,
    tuple2: EnvironmentTuple,
) -> int:
    """Compute the Hamming distance between two environment property tuples.

    The Hamming distance counts the number of positions at which the two
    boolean vectors differ:
        D_H(x, y) = sum_{i=0}^4 (x_i != y_i)

    Args:
        tuple1: First 5-boolean tuple.
        tuple2: Second 5-boolean tuple.

    Returns:
        Integer distance between 0 and 5.
    """
    return sum(1 for a, b in zip(tuple1, tuple2) if a != b)


def explain_property_differences(
    source: EnvironmentTuple,
    target: EnvironmentTuple,
) -> List[str]:
    """Generate human-readable explanations of property discrepancies.

    Args:
        source: User-provided input tuple.
        target: Candidate knowledge base tuple.

    Returns:
        List of formatted strings detailing each differing dimension.
    """
    differences: List[str] = []
    for i in range(5):
        if source[i] != target[i]:
            name = PROPERTY_NAMES[i]
            src_label = PROPERTY_LABELS[i][0] if source[i] else PROPERTY_LABELS[i][1]
            tgt_label = PROPERTY_LABELS[i][0] if target[i] else PROPERTY_LABELS[i][1]
            differences.append(
                f"{name}: Input is '{src_label}' but target is '{tgt_label}'"
            )
    return differences


# ==============================================================================
# Core Classifier Implementation
# ==============================================================================

class EnvironmentClassifier:
    """Knowledge Representation Classifier for AI Task Environments.

    Provides O(1) dictionary lookup for exact matching of environment property
    tuples to real-world AI domains, along with a nearest-neighbor distance metric
    (Hamming Distance) for graceful fallback when encountering unknown combinations.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[EnvironmentTuple, EnvironmentMetadata]] = None,
    ) -> None:
        """Initialize the classifier with a knowledge base.

        Args:
            knowledge_base: Mapping of 5-boolean tuples to EnvironmentMetadata.
                           Defaults to the standard Russell & Norvig KNOWLEDGE_BASE.
        """
        self._kb: Dict[EnvironmentTuple, EnvironmentMetadata] = (
            dict(knowledge_base) if knowledge_base is not None else dict(KNOWLEDGE_BASE)
        )

    @property
    def knowledge_base(self) -> Dict[EnvironmentTuple, EnvironmentMetadata]:
        """Return the internal knowledge base dictionary."""
        return self._kb

    def classify(
        self,
        prop_tuple: EnvironmentTuple,
        allow_nearest: bool = True,
    ) -> ClassificationResult:
        """Classify a given 5-property boolean tuple.

        Time Complexity: O(1) for exact dictionary lookup.
        When exact match fails and allow_nearest is True, performs O(K) Hamming
        distance scan across known environments.

        Args:
            prop_tuple: 5-tuple of booleans (Obs, Det, Epi, Stat, Disc).
            allow_nearest: If True, computes closest matches upon exact miss.

        Returns:
            ClassificationResult containing match details and PEAS metadata.
        """
        # Step 1: O(1) Exact Match Lookup
        if prop_tuple in self._kb:
            meta = self._kb[prop_tuple]
            return ClassificationResult(
                input_tuple=prop_tuple,
                is_exact_match=True,
                matched_environment=meta.name,
                metadata=meta,
                message=f"Exact Environment Match Found: {meta.name}",
            )

        # Step 2: Handle Unknown Combination Gracefully
        result = ClassificationResult(
            input_tuple=prop_tuple,
            is_exact_match=False,
            matched_environment=None,
            metadata=None,
            message="No exact environment match found in knowledge base.",
        )

        # Step 3: Optional Challenge - Distance Metric Nearest-Neighbor Search
        if allow_nearest and self._kb:
            closest = self.find_closest_environments(prop_tuple)
            result.closest_matches = closest
            if closest:
                best_dist = closest[0][1]
                names = ", ".join(m.name for m, _, _ in closest)
                result.message += (
                    f" Closest known match (Hamming distance {best_dist}/5): {names}"
                )

        return result

    def find_closest_environments(
        self,
        prop_tuple: EnvironmentTuple,
    ) -> List[Tuple[EnvironmentMetadata, int, List[str]]]:
        """Find the nearest environment(s) using Hamming distance metric.

        Args:
            prop_tuple: User input tuple.

        Returns:
            List of tuples (EnvironmentMetadata, distance, differences_list),
            sorted by ascending distance. Only the minimal distance tier is returned.
        """
        if not self._kb:
            return []

        scored: List[Tuple[EnvironmentMetadata, int, List[str]]] = []
        for kb_tuple, meta in self._kb.items():
            dist = calculate_hamming_distance(prop_tuple, kb_tuple)
            diffs = explain_property_differences(prop_tuple, kb_tuple)
            scored.append((meta, dist, diffs))

        # Sort by distance
        scored.sort(key=lambda item: item[1])
        min_dist = scored[0][1]

        # Return all environments sharing the minimum distance
        return [item for item in scored if item[1] == min_dist]


def classify_environment(
    properties: Sequence[Any],
    allow_nearest: bool = True,
    kb: Optional[Dict[EnvironmentTuple, EnvironmentMetadata]] = None,
) -> ClassificationResult:
    """Convenience functional wrapper to parse and classify an environment.

    Args:
        properties: Sequence of 5 boolean-convertible values.
        allow_nearest: Whether to search for nearest match if exact match fails.
        kb: Optional custom knowledge base.

    Returns:
        ClassificationResult object.
    """
    prop_tuple = parse_environment_tuple(properties)
    classifier = EnvironmentClassifier(knowledge_base=kb)
    return classifier.classify(prop_tuple, allow_nearest=allow_nearest)


def format_classification_report(result: ClassificationResult) -> str:
    """Format a ClassificationResult into a structured, readable report."""
    input_str = ", ".join(
        f"{PROPERTY_NAMES[i]}={'Yes' if result.input_tuple[i] else 'No'} "
        f"({PROPERTY_LABELS[i][0] if result.input_tuple[i] else PROPERTY_LABELS[i][1]})"
        for i in range(5)
    )

    lines: List[str] = [
        "=" * 78,
        "             ENVIRONMENT TYPE CLASSIFIER REPORT",
        "=" * 78,
        f"Input Tuple : {result.input_tuple}",
        f"Properties  : {input_str}",
        "-" * 78,
    ]

    if result.is_exact_match and result.metadata is not None:
        meta = result.metadata
        lines.extend([
            f"Classification Status : EXACT MATCH FOUND (O(1) Hash Lookup)",
            f"Environment Match     : {meta.name}",
            f"Category              : {meta.category}",
            f"Agent Type            : {meta.agent_type}",
            f"Description           : {meta.description}",
            "-" * 78,
            "PEAS Specification:",
            f"  * Performance Measure : {meta.peas.performance_measure}",
            f"  * Environment         : {meta.peas.environment}",
            f"  * Actuators           : {meta.peas.actuators}",
            f"  * Sensors             : {meta.peas.sensors}",
        ])
    else:
        lines.extend([
            "Classification Status : UNKNOWN COMBINATION (No Exact Match)",
            result.message,
        ])
        if result.closest_matches:
            lines.append("-" * 78)
            lines.append("Nearest Environment Matches (Hamming Distance Challenge):")
            for meta, dist, diffs in result.closest_matches:
                lines.append(
                    f"  * {meta.name} (Distance: {dist}/5 - {len(diffs)} difference(s))"
                )
                for diff in diffs:
                    lines.append(f"      - {diff}")
                lines.append(f"      PEAS Summary: {meta.peas.environment}")

    lines.append("=" * 78)
    return "\n".join(lines)


def get_viva_reference() -> str:
    """Return comprehensive Viva Voce theory notes for Question 8."""
    return """
================================================================================
                    VIVA VOCE REFERENCE & THEORY GUIDE
                 Question 8: Environment Type Classifier
================================================================================

1. PEAS DESCRIPTION:
   -----------------
   PEAS is an acronym defining the complete specification of an intelligent agent's
   task environment:
   * P - Performance Measure:
     The objective criterion used by an external observer to gauge the success of
     the agent's behavior. Must be formulated according to what one actually wants
     in the environment, rather than how the agent ought to behave.
     Example (Taxi): Safety, speed, legal compliance, passenger comfort, profits.
   * E - Environment:
     The external physical or virtual context within which the agent acts.
     Example (Taxi): Urban streets, highways, pedestrians, traffic, weather.
   * A - Actuators:
     The hardware or software mechanisms through which the agent performs actions.
     Example (Taxi): Steering wheel, brake, accelerator pedal, turn signals, horn.
   * S - Sensors:
     The devices and inputs through which the agent receives observations (percepts).
     Example (Taxi): Video cameras, LIDAR, radar, GPS, speedometer, accelerometer.

2. PROPERTIES OF CHESS VS. POKER ENVIRONMENTS:
   ------------------------------------------
   Comparison Table:
   +-------------------+----------------------------+----------------------------+
   | Property          | Chess (with Clock)         | Poker (Texas Hold'em)      |
   +-------------------+----------------------------+----------------------------+
   | Observability     | Fully Observable           | Partially Observable       |
   |                   | (All pieces visible)       | (Opponent cards hidden)    |
   +-------------------+----------------------------+----------------------------+
   | Determinism       | Deterministic              | Stochastic                 |
   |                   | (No chance moves / dice)   | (Deck shuffle, deals)      |
   +-------------------+----------------------------+----------------------------+
   | Episodic/Seq.     | Sequential                 | Sequential                 |
   |                   | (Moves affect end-game)    | (Betting affects stack)    |
   +-------------------+----------------------------+----------------------------+
   | Statics           | Dynamic (with clock) /     | Dynamic (with shot clock) /|
   |                   | Semi-dynamic               | Static (untimed)           |
   +-------------------+----------------------------+----------------------------+
   | Discreteness      | Discrete                   | Discrete                   |
   |                   | (64 squares, discrete moves| (Discrete cards & bets)    |
   +-------------------+----------------------------+----------------------------+
   | Agent Interaction | Multi-Agent (Competitive)  | Multi-Agent (Competitive,  |
   |                   |                            | Imperfect Information)     |
   +-------------------+----------------------------+----------------------------+

   Key Distinction:
   Chess has perfect information (no hidden state) and deterministic transitions,
   allowing deep minimax search with alpha-beta pruning. Poker features imperfect
   information and chance transitions, requiring game-theoretic equilibrium
   strategies (e.g., Nash equilibrium, Counterfactual Regret Minimization / CFR).

3. CONTINUOUS VS. DISCRETE ENVIRONMENTS:
   ------------------------------------
   * Discrete Environment:
     - The state space, time steps, percepts, and agent action space consist of a
       finite or countably distinct set of values.
     - Example: Chess board positions (64 discrete squares), Turn-based games,
       Grid navigation.
     - Advantage: Can be modeled cleanly as graph search, truth tables, or finite
       automata.
   * Continuous Environment:
     - The state variables, time progression, or action outputs take values from
       the set of real numbers (uncountably infinite).
     - Example: Automated Taxi driving (steering angle [-45.0, 45.0] degrees,
       velocity, acceleration), Medical vital monitoring (blood pressure, temperature).
     - Requirement: Requires differential equations, PID controllers, reinforcement
       learning with continuous action spaces, or fine discretization.

4. LOOKUP TABLE CLASSIFICATION & COMPLEXITY:
   -----------------------------------------
   - By utilizing a 5-tuple of booleans as a hash key in a Python dictionary:
     Time Complexity is O(1) for hash lookup.
     Space Complexity is O(K), where K <= 2^5 = 32 maximum possible combinations.
   - Hamming Distance Metric:
     D_H(u, v) = sum(u_i != v_i). Provides an optimal, mathematically rigorous
     nearest-neighbor fallback for unseen environment combinations.
================================================================================
"""
