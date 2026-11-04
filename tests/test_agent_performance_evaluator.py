"""Unit test suite for Agent Performance Measure Evaluator (Question 9).

Validates sequential linear evaluation, arithmetic transitions, edge cases
(empty logs, negative scores, all bumps, all deliveries), custom metrics,
synonyms, external CSV loading, and performance under large scale inputs.
"""

from pathlib import Path
import tempfile
import time
import unittest

from src.agent_performance_evaluator import (
    EVENT_BUMP,
    EVENT_DELIVER,
    EVENT_MOVE,
    DronePerformanceEvaluator,
    EvaluationResult,
    ScoringMetric,
    evaluate_drone_performance,
    get_viva_reference,
    read_events_from_csv,
)


class TestAgentPerformanceEvaluator(unittest.TestCase):
    """Test suite for DronePerformanceEvaluator core logic and standalone functions."""

    def setUp(self) -> None:
        """Initialize standard evaluator for test cases."""
        self.evaluator = DronePerformanceEvaluator()

    def test_sample_problem_statement(self) -> None:
        """Verify the exact sample input and output from university problem statement.

        Sample Input: ['move', 'move', 'bump', 'move', 'deliver']
        Sample Output: Performance Score: 2 (10 - 3 moves - 5 bump)
        """
        sample_events = ["move", "move", "bump", "move", "deliver"]
        score = evaluate_drone_performance(sample_events)
        self.assertEqual(score, 2)

        result = self.evaluator.evaluate_log(sample_events)
        self.assertEqual(result.final_score, 2)
        self.assertEqual(result.move_count, 3)
        self.assertEqual(result.bump_count, 1)
        self.assertEqual(result.delivery_count, 1)
        self.assertEqual(result.formula_breakdown, "10 - 3 moves - 5 bump")
        self.assertEqual(
            result.format_summary(),
            "Performance Score: 2 (10 - 3 moves - 5 bump)",
        )

    def test_empty_event_log(self) -> None:
        """Verify empty event log produces 0 score and zero tallies."""
        result = self.evaluator.evaluate_log([])
        self.assertEqual(result.final_score, 0)
        self.assertEqual(result.total_steps, 0)
        self.assertEqual(result.move_count, 0)
        self.assertEqual(result.bump_count, 0)
        self.assertEqual(result.delivery_count, 0)
        self.assertEqual(result.formula_breakdown, "0")

    def test_all_moves_pure_cost(self) -> None:
        """Verify sequence of only movement actions results in negative score matching count."""
        moves = ["move", "move", "move", "move", "move"]
        score = evaluate_drone_performance(moves)
        self.assertEqual(score, -5)

    def test_all_bumps_pure_penalty(self) -> None:
        """Verify sequence of only obstacle collisions applies -5 per bump."""
        bumps = ["bump", "bump", "bump"]
        score = evaluate_drone_performance(bumps)
        self.assertEqual(score, -15)

    def test_all_deliveries_pure_reward(self) -> None:
        """Verify multiple deliveries without moves award +10 each."""
        deliveries = ["deliver", "deliver", "deliver"]
        score = evaluate_drone_performance(deliveries)
        self.assertEqual(score, 30)

    def test_negative_score_constraint(self) -> None:
        """Verify that scores can legitimately be negative without lower-bound clamping."""
        bad_flight = ["move", "bump", "bump", "move", "bump"]
        # 2 moves (-2) + 3 bumps (-15) = -17
        result = self.evaluator.evaluate_log(bad_flight)
        self.assertEqual(result.final_score, -17)
        self.assertLess(result.final_score, 0)
        self.assertEqual(result.formula_breakdown, "0 - 2 moves - 15 bumps")

    def test_custom_scoring_metric(self) -> None:
        """Verify that evaluator respects user-configured scoring parameters."""
        custom_metric = ScoringMetric(
            reward_delivery=25,
            cost_move=2,
            penalty_bump=10,
        )
        custom_evaluator = DronePerformanceEvaluator(metric=custom_metric)
        events = ["move", "bump", "deliver"]
        # -2 (move) - 10 (bump) + 25 (deliver) = 13
        result = custom_evaluator.evaluate_log(events)
        self.assertEqual(result.final_score, 13)

    def test_case_insensitivity_and_whitespace_trimming(self) -> None:
        """Verify robust string matching on uppercase, mixed case, and padded whitespace."""
        events = ["  MOVE ", "Bump\t", "DELIVER\n", "  MoVe  "]
        result = self.evaluator.evaluate_log(events)
        self.assertEqual(result.move_count, 2)
        self.assertEqual(result.bump_count, 1)
        self.assertEqual(result.delivery_count, 1)
        self.assertEqual(result.final_score, 3)  # 10 - 2 - 5 = 3

    def test_synonym_and_alias_recognition(self) -> None:
        """Verify common synonyms (step, collision, dropoff) map to canonical events."""
        events = ["step", "collision", "fly", "dropoff", "crash"]
        result = self.evaluator.evaluate_log(events)
        self.assertEqual(result.move_count, 2)  # step, fly
        self.assertEqual(result.bump_count, 2)  # collision, crash
        self.assertEqual(result.delivery_count, 1)  # dropoff
        self.assertEqual(result.final_score, -2)  # 10 - 2 - 10 = -2

    def test_unrecognized_events_ignored_gracefully(self) -> None:
        """Verify unexpected events are counted as unknown without disrupting running score."""
        events = ["move", "idle", "recharge", "deliver"]
        result = self.evaluator.evaluate_log(events)
        self.assertEqual(result.move_count, 1)
        self.assertEqual(result.delivery_count, 1)
        self.assertEqual(result.unknown_count, 2)
        self.assertEqual(result.final_score, 9)  # 10 - 1 = 9

    def test_sequential_step_records_and_audit_trace(self) -> None:
        """Verify step records track accurate running score and notes."""
        events = ["move", "bump", "deliver"]
        result = self.evaluator.evaluate_log(events, record_trace=True)
        self.assertEqual(len(result.step_records), 3)

        step1 = result.step_records[0]
        self.assertEqual(step1.step_number, 1)
        self.assertEqual(step1.score_delta, -1)
        self.assertEqual(step1.running_score, -1)

        step2 = result.step_records[1]
        self.assertEqual(step2.step_number, 2)
        self.assertEqual(step2.score_delta, -5)
        self.assertEqual(step2.running_score, -6)

        step3 = result.step_records[2]
        self.assertEqual(step3.step_number, 3)
        self.assertEqual(step3.score_delta, 10)
        self.assertEqual(step3.running_score, 4)

    def test_format_detailed_report(self) -> None:
        """Verify detailed report contains header, summary, and tabular trace."""
        events = ["move", "bump", "deliver"]
        result = self.evaluator.evaluate_log(events, record_trace=True)
        report = result.format_detailed_report()
        self.assertIn("AUTONOMOUS DELIVERY DRONE PERFORMANCE AUDIT", report)
        self.assertIn("Deliveries Completed   : 1", report)
        self.assertIn("FINAL PERFORMANCE SCORE: 4", report)
        self.assertIn("Step-by-Step Flight Trace:", report)

    def test_read_events_from_csv_with_standard_header(self) -> None:
        """Verify parsing CSV file with 'step,action' header."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("step,action\n1,move\n2,move\n3,bump\n4,move\n5,deliver\n")
            tmp_path = Path(tmp.name)

        try:
            events = read_events_from_csv(tmp_path)
            self.assertEqual(events, ["move", "move", "bump", "move", "deliver"])
            score = self.evaluator.evaluate_csv(tmp_path).final_score
            self.assertEqual(score, 2)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_read_events_from_csv_single_column_no_header(self) -> None:
        """Verify parsing single-column CSV where first row is data."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("move\nbump\ndeliver\n")
            tmp_path = Path(tmp.name)

        try:
            events = read_events_from_csv(tmp_path)
            self.assertEqual(events, ["move", "bump", "deliver"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_read_events_from_csv_comma_separated_single_line(self) -> None:
        """Verify parsing CSV with single row containing comma-separated tokens."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("move, move, bump, deliver\n")
            tmp_path = Path(tmp.name)

        try:
            events = read_events_from_csv(tmp_path)
            self.assertEqual(events, ["move", "move", "bump", "deliver"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_read_events_from_csv_explicit_column_selection(self) -> None:
        """Verify explicit column selection from multi-column CSV."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("id,timestamp,drone_action,battery\n1,10:00,move,99%\n2,10:01,deliver,95%\n")
            tmp_path = Path(tmp.name)

        try:
            events = read_events_from_csv(tmp_path, column="drone_action")
            self.assertEqual(events, ["move", "deliver"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_read_events_from_csv_errors(self) -> None:
        """Verify FileNotFoundError on missing file and ValueError on non-existent column."""
        with self.assertRaises(FileNotFoundError):
            read_events_from_csv("non_existent_file_path_12345.csv")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("action,status\nmove,ok\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError):
                read_events_from_csv(tmp_path, column="invalid_column_name")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_linear_time_complexity_large_event_log(self) -> None:
        """Verify linear time complexity O(N) by evaluating 100,000 events within 1 second."""
        large_events = (["move"] * 4 + ["bump"] + ["deliver"]) * 20000  # 120,000 events
        start_time = time.perf_counter()
        result = self.evaluator.evaluate_log(large_events, record_trace=False)
        elapsed = time.perf_counter() - start_time

        # Expected score per batch of 6: 10 - 4 - 5 = 1. For 20,000 batches: 20,000
        self.assertEqual(result.final_score, 20000)
        self.assertEqual(result.total_steps, 120000)
        self.assertLess(elapsed, 1.0, f"Evaluation took {elapsed:.4f}s, expected < 1.0s for O(N)")

    def test_viva_reference_contains_required_topics(self) -> None:
        """Verify viva reference documentation contains all core topics specified."""
        viva = get_viva_reference()
        self.assertIn("DESIGNING PERFORMANCE MEASURES", viva)
        self.assertIn("RATIONALITY DEFINITION", viva)
        self.assertIn("OMNISCIENCE VS. RATIONALITY", viva)
        self.assertIn("PEAS FRAMEWORK FOR AUTONOMOUS DELIVERY DRONE", viva)


if __name__ == "__main__":
    unittest.main()
