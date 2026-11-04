"""Agent Performance Measure Evaluator (Question 9).

This module implements a linear evaluation engine to assess the performance of an
autonomous delivery drone operating in a simulated environment, based on Russell &
Norvig's foundational definitions of PEAS and Rationality in Artificial Intelligence:
A Modern Approach (AIMA, Chapters 2 & 3).

Core AI Concepts:
-----------------
1. PEAS Framework for Autonomous Delivery Drone:
   - P (Performance Measure):
     * Successful delivery reward: +10 points
     * Time step / movement cost:  -1 point per move
     * Obstacle collision penalty: -5 points per bump
     * Objective utility formula:
       U = (10 * N_deliver) - (1 * N_move) - (5 * N_bump)
   - E (Environment):
     * Urban airspace, buildings, power lines, trees, dynamic winds, landing pads.
   - A (Actuators):
     * Quadcopter/hexacopter rotors, pitch/roll/yaw attitude controllers, parcel
       gripper/release servos, transponder, warning beacon.
   - S (Sensors):
     * GPS receiver, LiDAR/sonar proximity sensors, downward visual landing camera,
       inertial measurement unit (IMU), battery/telemetry monitor.

2. Rationality Definition (Russell & Norvig):
   An agent is rational if, for each possible percept sequence, it selects an action
   that is expected to maximize its performance measure, given the evidence provided
   by the percept sequence and whatever built-in knowledge the agent possesses.

3. Rationality vs. Omniscience:
   - An omniscient agent knows the *actual* outcome of its actions and can act
     accordingly. In the real world, omniscience is physically impossible due to
     partial observability, stochasticity, and noisy sensors.
   - A rational agent maximizes *expected* performance based on the information it
     has received up to the current time step. Rationality does not guarantee
     infallibility or perfection; it guarantees mathematically sound decision-making
     under uncertainty.

4. Performance Measure Design Principles:
   - An agent's performance measure must evaluate *what* is desired in the environment
     (e.g., safe parcel delivery), not *how* the agent is designed to behave.
   - If an agent is rewarded for intermediate actions (e.g., points for taking steps),
     it may discover perverse reward loops (e.g., oscillating forever to accumulate
     step points without completing deliveries).
   - Penalties for time steps force efficiency and path optimization; collision
     penalties force safety and risk-averse navigation.

Algorithm & Complexity:
-----------------------
- Technique: Sequential linear evaluation over event stream.
- Time Complexity: O(N) where N is the number of events in the action log.
- Space Complexity: O(1) auxiliary working space (O(N) if preserving detailed audit trace).
- Constraints: The total score can be negative (e.g., excessive collisions or long paths).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# Canonical Event Identifiers
EVENT_MOVE: str = "move"
EVENT_BUMP: str = "bump"
EVENT_DELIVER: str = "deliver"

# Canonical Aliases / Synonyms for Flexible Parsing
EVENT_ALIASES: Dict[str, str] = {
    "move": EVENT_MOVE,
    "step": EVENT_MOVE,
    "forward": EVENT_MOVE,
    "fly": EVENT_MOVE,
    "bump": EVENT_BUMP,
    "collision": EVENT_BUMP,
    "hit": EVENT_BUMP,
    "crash": EVENT_BUMP,
    "deliver": EVENT_DELIVER,
    "delivery": EVENT_DELIVER,
    "dropoff": EVENT_DELIVER,
    "drop": EVENT_DELIVER,
}


@dataclass(frozen=True)
class ScoringMetric:
    """Configurable performance metric weights for the agent evaluation.

    Attributes:
        reward_delivery: Points awarded for each successful parcel delivery (+10).
        cost_move: Points deducted for each movement / time step taken (-1).
        penalty_bump: Points deducted for each obstacle collision (-5).
    """

    reward_delivery: int = 10
    cost_move: int = 1
    penalty_bump: int = 5

    def delta_for_event(self, normalized_event: str) -> int:
        """Calculate the arithmetic score adjustment for a normalized event.

        Args:
            normalized_event: Canonical event name ('move', 'bump', 'deliver').

        Returns:
            Integer delta (positive, negative, or zero).
        """
        if normalized_event == EVENT_DELIVER:
            return self.reward_delivery
        if normalized_event == EVENT_MOVE:
            return -self.cost_move
        if normalized_event == EVENT_BUMP:
            return -self.penalty_bump
        return 0


@dataclass
class StepEvaluation:
    """Detailed record of a single processed event in the action log.

    Attributes:
        step_number: 1-indexed sequential step counter.
        raw_event: The original string input from the log.
        normalized_event: The parsed canonical event identifier.
        score_delta: The arithmetic change applied to the score.
        running_score: The cumulative performance score after this step.
        description: Human-readable narrative of the event and its impact.
    """

    step_number: int
    raw_event: str
    normalized_event: str
    score_delta: int
    running_score: int
    description: str


@dataclass
class EvaluationResult:
    """Complete summary of the drone's evaluated action log.

    Attributes:
        final_score: The final integer performance score.
        total_steps: Number of events processed.
        move_count: Number of movement actions taken.
        bump_count: Number of obstacle collisions encountered.
        delivery_count: Number of successful deliveries completed.
        unknown_count: Number of unrecognized events ignored.
        step_records: List of sequential step records.
        metric: The scoring metric used for evaluation.
    """

    final_score: int
    total_steps: int
    move_count: int
    bump_count: int
    delivery_count: int
    unknown_count: int
    step_records: List[StepEvaluation] = field(default_factory=list)
    metric: ScoringMetric = field(default_factory=ScoringMetric)

    @property
    def formula_breakdown(self) -> str:
        """Return mathematical breakdown string matching textbook specification.

        Example:
            '10 - 3 moves - 5 bump'
        """
        parts: List[str] = []
        if self.delivery_count > 0:
            delivery_pts = self.delivery_count * self.metric.reward_delivery
            parts.append(str(delivery_pts))
        else:
            parts.append("0")

        if self.move_count > 0:
            move_pts = self.move_count * self.metric.cost_move
            parts.append(f"{move_pts} moves")

        if self.bump_count > 0:
            bump_pts = self.bump_count * self.metric.penalty_bump
            parts.append(f"{bump_pts} bump" if self.bump_count == 1 else f"{bump_pts} bumps")

        if len(parts) == 1:
            return f"{parts[0]}"
        return f"{parts[0]} - " + " - ".join(parts[1:])

    def format_summary(self) -> str:
        """Format a single-line standard output matching the sample requirement."""
        return f"Performance Score: {self.final_score} ({self.formula_breakdown})"

    def format_detailed_report(self) -> str:
        """Generate a comprehensive multi-line text report of the flight audit."""
        divider = "=" * 70
        sub_divider = "-" * 70
        lines = [
            divider,
            "         AUTONOMOUS DELIVERY DRONE PERFORMANCE AUDIT",
            divider,
            f"  Total Events Processed : {self.total_steps}",
            f"  Deliveries Completed   : {self.delivery_count} (+{self.delivery_count * self.metric.reward_delivery} pts)",
            f"  Movement Time Steps    : {self.move_count} (-{self.move_count * self.metric.cost_move} pts)",
            f"  Obstacle Collisions    : {self.bump_count} (-{self.bump_count * self.metric.penalty_bump} pts)",
            f"  Unknown/Ignored Events : {self.unknown_count} (0 pts)",
            sub_divider,
            f"  Calculation Formula    : {self.formula_breakdown} = {self.final_score}",
            f"  FINAL PERFORMANCE SCORE: {self.final_score}",
            divider,
            "",
            "Step-by-Step Flight Trace:",
            f"{'Step':<6}{'Event':<12}{'Delta':<10}{'Running Score':<16}{'Note'}",
            f"{'-'*4:<6}{'-'*10:<12}{'-'*8:<10}{'-'*14:<16}{'-'*20}",
        ]
        for step in self.step_records:
            sign = "+" if step.score_delta > 0 else ""
            delta_str = f"{sign}{step.score_delta}"
            lines.append(
                f"{step.step_number:<6}{step.raw_event:<12}{delta_str:<10}"
                f"{step.running_score:<16}{step.description}"
            )
        lines.append(divider)
        return "\n".join(lines)


class DronePerformanceEvaluator:
    """Evaluates the sequential performance of an autonomous delivery drone.

    Processes action event logs in linear time O(N) with O(1) auxiliary space,
    tracking running score transitions according to predefined PEAS metrics.
    """

    def __init__(self, metric: Optional[ScoringMetric] = None) -> None:
        """Initialize evaluator with an optional custom scoring metric.

        Args:
            metric: ScoringMetric instance defining rewards and penalties.
                    Defaults to Russell & Norvig standard drone configuration.
        """
        self.metric: ScoringMetric = metric if metric is not None else ScoringMetric()

    def normalize_event(self, raw_event: str) -> Optional[str]:
        """Normalize and sanitize raw event string to canonical form.

        Args:
            raw_event: Input string representing an action or event.

        Returns:
            Canonical event identifier ('move', 'bump', 'deliver') or None if unknown.
        """
        if not isinstance(raw_event, str):
            return None
        cleaned = raw_event.strip().lower()
        return EVENT_ALIASES.get(cleaned)

    def evaluate_log(
        self,
        events: Sequence[str],
        record_trace: bool = True,
    ) -> EvaluationResult:
        """Evaluate a sequence of simulated actions/events sequentially.

        Time Complexity: O(N) where N = len(events).
        Space Complexity: O(1) auxiliary if record_trace is False, O(N) if True.

        Args:
            events: Sequence of event strings, e.g. ['move', 'bump', 'deliver'].
            record_trace: Whether to preserve step-by-step history for auditing.

        Returns:
            EvaluationResult dataclass with score, tallies, and breakdown.
        """
        running_score: int = 0
        move_count: int = 0
        bump_count: int = 0
        delivery_count: int = 0
        unknown_count: int = 0
        step_records: List[StepEvaluation] = []

        for index, raw_item in enumerate(events, start=1):
            normalized = self.normalize_event(raw_item)
            delta = self.metric.delta_for_event(normalized) if normalized else 0
            running_score += delta

            desc: str
            if normalized == EVENT_DELIVER:
                delivery_count += 1
                desc = f"Successful delivery (+{self.metric.reward_delivery})"
            elif normalized == EVENT_MOVE:
                move_count += 1
                desc = f"Time step elapsed (-{self.metric.cost_move})"
            elif normalized == EVENT_BUMP:
                bump_count += 1
                desc = f"Obstacle collision penalty (-{self.metric.penalty_bump})"
            else:
                unknown_count += 1
                desc = f"Unrecognized event '{raw_item}' (no penalty)"

            if record_trace:
                step_records.append(
                    StepEvaluation(
                        step_number=index,
                        raw_event=str(raw_item),
                        normalized_event=normalized if normalized else "unknown",
                        score_delta=delta,
                        running_score=running_score,
                        description=desc,
                    )
                )

        return EvaluationResult(
            final_score=running_score,
            total_steps=len(events),
            move_count=move_count,
            bump_count=bump_count,
            delivery_count=delivery_count,
            unknown_count=unknown_count,
            step_records=step_records,
            metric=self.metric,
        )

    def evaluate_csv(
        self,
        filepath: Union[str, Path],
        event_column: Optional[str] = None,
        record_trace: bool = True,
    ) -> EvaluationResult:
        """Read action log from an external CSV file and evaluate performance.

        Supports CSV files with:
        - Named column header (e.g. 'action', 'event', 'operation')
        - Single column without header
        - Single row with comma-separated actions

        Args:
            filepath: Path to the external CSV file.
            event_column: Optional column name. If omitted, auto-detected.
            record_trace: Whether to record step trace.

        Returns:
            EvaluationResult containing the performance evaluation.
        """
        events = read_events_from_csv(filepath, column=event_column)
        return self.evaluate_log(events, record_trace=record_trace)


def evaluate_drone_performance(
    events: Sequence[str],
    metric: Optional[ScoringMetric] = None,
) -> int:
    """Evaluate an action log and return the final integer performance score.

    This function directly fulfills the core task specification:
        Input Requirements: List of string events: ['move', 'bump', 'move', 'deliver']
        Expected Output: Integer representing total performance score.

    Args:
        events: List of string events.
        metric: Optional custom scoring metric.

    Returns:
        Integer representing total PEAS performance score (can be negative).
    """
    evaluator = DronePerformanceEvaluator(metric=metric)
    result = evaluator.evaluate_log(events, record_trace=False)
    return result.final_score


def read_events_from_csv(
    filepath: Union[str, Path],
    column: Optional[str] = None,
) -> List[str]:
    """Parse a sequence of drone actions from an external CSV file.

    Handles diverse standard CSV layouts:
    - Tables with headers: 'action', 'event', 'operation', 'step'
    - Single-column logs with or without header
    - Single-line comma-delimited streams

    Args:
        filepath: Path to CSV file.
        column: Explicit column name to extract (case-insensitive).

    Returns:
        List of event strings extracted sequentially.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty or specified column is missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    events: List[str] = []
    with open(path, mode="r", encoding="utf-8", newline="") as csvfile:
        # Read lines to determine layout
        raw_text = csvfile.read().strip()
        if not raw_text:
            return []

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return []

    # Check if first line contains common column headers
    reader = csv.reader(lines)
    all_rows = list(reader)
    if not all_rows:
        return []

    header = [h.strip().lower() for h in all_rows[0]]
    candidate_headers = ["action", "event", "operation", "log", "actions", "events"]

    target_idx: Optional[int] = None
    data_start_row = 0

    if column is not None:
        clean_col = column.strip().lower()
        if clean_col in header:
            target_idx = header.index(clean_col)
            data_start_row = 1
        else:
            raise ValueError(f"Specified column '{column}' not found in CSV header: {header}")
    else:
        # Check if header matches known column name
        for candidate in candidate_headers:
            if candidate in header:
                target_idx = header.index(candidate)
                data_start_row = 1
                break

    # If no recognized header, check if row 0 itself is data or header
    if target_idx is None:
        first_token = header[0]
        if first_token in EVENT_ALIASES:
            # First row is actual data
            target_idx = 0
            data_start_row = 0
        else:
            # Assume row 0 is header, use first column
            target_idx = 0
            data_start_row = 1

    # Extract event strings
    for row_num, row in enumerate(all_rows[data_start_row:], start=data_start_row + 1):
        if not row:
            continue
        # If row has multiple items and target_idx is in range
        if target_idx < len(row):
            val = row[target_idx].strip()
            if val:
                events.append(val)
        elif len(row) == 1:
            val = row[0].strip()
            if val:
                events.append(val)

    # Edge case: If file was a single row with multiple comma-separated values
    if len(all_rows) == 1 and len(all_rows[0]) > 1 and data_start_row == 0:
        events = [item.strip() for item in all_rows[0] if item.strip()]

    return events


def get_viva_reference() -> str:
    """Return a comprehensive university Viva Voce study guide for Question 9.

    Covers Designing Performance Measures, Rationality Definition, Omniscience vs.
    Rationality, and PEAS formulation for autonomous delivery drones.
    """
    return """================================================================================
           QUESTION NO. 9: VIVA VOCE THEORY & REFERENCE GUIDE
               Core Subject: Intelligent Agents & PEAS Framework
================================================================================

1. DESIGNING PERFORMANCE MEASURES:
   -------------------------------
   * Definition:
     A performance measure embodies the criterion for success of an agent's
     behavior. It is an external standard designed by the evaluator or user to
     gauge how effectively the agent achieves desired goals in its environment.

   * The Golden Rule of Performance Measure Design (Russell & Norvig, AIMA):
     "As a general rule, it is better to design performance measures according
      to what one actually wants in the environment, rather than according to
      how one thinks the agent should behave."

   * Perverse Instantiation & Reward Gaming:
     - If we reward intermediate agent actions rather than real environmental outcomes,
       agents will exploit loopholes.
     - Example (Vacuum Agent): If rewarded +1 for every dirt patch sucked up, a
       rational agent might repeatedly drop dirt on the floor and suck it back up
       forever to accumulate infinite points.
     - Example (Delivery Drone): If rewarded for flight time or distance covered, the
       drone would fly in endless circles rather than completing deliveries.

   * Trade-offs in Drone Performance Scoring:
     U = (+10 * Deliveries) - (1 * Time Steps) - (5 * Obstacle Collisions)
     - (+10 Delivery): Incentivizes goal accomplishment.
     - (-1 Time Step): Forces speed and fuel efficiency; penalizes idle wandering.
     - (-5 Bump Penalty): Forces collision avoidance and safe flight trajectories.
     - If bump penalty is too low, the drone might deliberately crash through
       obstacles to save time. If bump penalty is too high, the drone might refuse
       to fly anywhere near tight corridors.

2. RATIONALITY DEFINITION:
   ----------------------
   * Formal AIMA Definition:
     "For each possible percept sequence, a rational agent should select an
      action that is expected to maximize its performance measure, given the
      evidence provided by the percept sequence and whatever built-in knowledge
      the agent has."

   * The 4 Pillars of Rationality:
     1. The Performance Measure that defines the criterion of success.
     2. The Agent's Prior Knowledge of the environment.
     3. The Actions that the agent is capable of performing.
     4. The Agent's Percept Sequence to date (historical sensory inputs).

   * Crucial Distinctions:
     - Rationality is NOT omniscience: It depends on expected, not actual, outcome.
     - Rationality is NOT perfection: Rationality maximizes expected success, while
       perfection maximizes actual outcome (which requires impossible foresight).
     - Rationality encourages Information Gathering: Doing actions to modify future
       percepts (e.g., drone camera panning to inspect an unknown radar blip).

3. OMNISCIENCE VS. RATIONALITY:
   ----------------------------
   +-------------------+----------------------------+----------------------------+
   | Characteristic    | Omniscient Agent           | Rational Agent             |
   +-------------------+----------------------------+----------------------------+
   | Definition        | Knows the *actual* outcome | Maximizes *expected*       |
   |                   | of its actions in advance  | outcome based on available |
   |                   | and acts accordingly.      | percepts and knowledge.    |
   +-------------------+----------------------------+----------------------------+
   | Feasibility       | Physically impossible in   | Fully achievable and       |
   |                   | real-world environments    | standard engineering       |
   |                   | (due to chaos/noise).      | objective.                 |
   +-------------------+----------------------------+----------------------------+
   | Epistemic State   | Complete certainty and     | Operates under uncertainty,|
   |                   | infallible foresight.      | partial observability, and |
   |                   |                            | stochastic noise.          |
   +-------------------+----------------------------+----------------------------+
   | Failure Handling  | Never fails.               | May experience failure if  |
   |                   |                            | an unexpected event occurs,|
   |                   |                            | yet remained fully rational|
   |                   |                            | at decision time.          |
   +-------------------+----------------------------+----------------------------+

   * Illustrative Drone Example:
     Suppose a delivery drone checks weather radar, verifies clear skies, and takes
     off. Midway through the flight, an unpredicted falling meteorite strikes the
     drone.
     - Was the drone omniscient? No (an omniscient drone would have stayed grounded).
     - Was the drone rational? YES. Based on all available percepts and meteorological
       knowledge, flying was the mathematically optimal decision to maximize its
       expected delivery score.

4. PEAS FRAMEWORK FOR AUTONOMOUS DELIVERY DRONE:
   --------------------------------------------
   - P (Performance Measure):
     High delivery throughput (+10/delivery), minimal delivery latency / flight
     duration (-1/time step), zero collisions (-5/bump), high battery efficiency,
     customer satisfaction.
   - E (Environment):
     Dynamic 3D urban airspace, high-rise buildings, trees, power lines, birds,
     gusty winds, rain, designated landing pads / customer doorsteps.
   - A (Actuators):
     Brushless motor ESCs, quad/hex-rotors, parcel latch/winch servo, navigation
     LEDs, emergency parachute release, audio warning siren.
   - S (Sensors):
     GNSS/GPS receiver, 3D LiDAR scanner, ultrasonic sonars, downward optical flow
     camera, IMU (gyro/accelerometers), barometer (altitude), battery voltage/current.

5. COMPUTATIONAL COMPLEXITY ANALYSIS:
   -----------------------------------
   - Time Complexity: O(N) where N is the length of the event log.
     Each event is processed in constant time O(1) through dictionary lookup and
     scalar arithmetic addition.
   - Space Complexity:
     * O(1) auxiliary space when computing running score cumulatively.
     * O(N) when maintaining detailed step records for audit logging or CSV output.
================================================================================
"""
