"""Unit tests for Environment Type Classifier (Question 8).

Validates knowledge dictionary lookup, boolean and tuple input parsing,
exact environment matching (Backgammon, Chess, Poker, Medical Diagnosis, etc.),
graceful handling of unknown combinations, and Hamming distance nearest-neighbor metric.
"""

import unittest
from typing import Dict, Tuple

from src.environment_classifier import (
    INDEX_DISCRETE,
    INDEX_DETERMINISTIC,
    INDEX_EPISODIC,
    INDEX_OBSERVABLE,
    INDEX_STATIC,
    KNOWLEDGE_BASE,
    PROPERTY_LABELS,
    PROPERTY_NAMES,
    ClassificationResult,
    EnvironmentClassifier,
    EnvironmentMetadata,
    EnvironmentTuple,
    PEASDescription,
    calculate_hamming_distance,
    classify_environment,
    explain_property_differences,
    format_classification_report,
    get_viva_reference,
    parse_boolean,
    parse_environment_tuple,
)


class TestBooleanParsing(unittest.TestCase):
    """Test suite for flexible boolean input parsing."""

    def test_parse_boolean_native_booleans(self) -> None:
        """Verify native bools pass through unchanged."""
        self.assertTrue(parse_boolean(True))
        self.assertFalse(parse_boolean(False))

    def test_parse_boolean_numeric(self) -> None:
        """Verify integers 1 and 0 convert correctly."""
        self.assertTrue(parse_boolean(1))
        self.assertFalse(parse_boolean(0))
        with self.assertRaises(ValueError):
            parse_boolean(2)
        with self.assertRaises(ValueError):
            parse_boolean(-1)

    def test_parse_boolean_string_variants(self) -> None:
        """Verify standard affirmative and negative string tokens."""
        affirmative = ["true", "True", "TRUE", "t", "T", "yes", "YES", "y", "Y", "1"]
        negative = ["false", "False", "FALSE", "f", "F", "no", "NO", "n", "N", "0"]

        for val in affirmative:
            self.assertTrue(parse_boolean(val), f"Failed to parse '{val}' as True")

        for val in negative:
            self.assertFalse(parse_boolean(val), f"Failed to parse '{val}' as False")

    def test_parse_boolean_domain_keywords(self) -> None:
        """Verify environment dimension keywords parse into appropriate booleans."""
        self.assertTrue(parse_boolean("observable"))
        self.assertTrue(parse_boolean("deterministic"))
        self.assertTrue(parse_boolean("episodic"))
        self.assertTrue(parse_boolean("static"))
        self.assertTrue(parse_boolean("discrete"))

        self.assertFalse(parse_boolean("partially"))
        self.assertFalse(parse_boolean("stochastic"))
        self.assertFalse(parse_boolean("sequential"))
        self.assertFalse(parse_boolean("dynamic"))
        self.assertFalse(parse_boolean("continuous"))

    def test_parse_boolean_invalid_inputs(self) -> None:
        """Verify invalid strings and incompatible types raise ValueError."""
        invalid_samples = ["maybe", "unknown", "none", "2", "foo", "bar"]
        for sample in invalid_samples:
            with self.assertRaises(ValueError):
                parse_boolean(sample)

        with self.assertRaises(ValueError):
            parse_boolean(None)
        with self.assertRaises(ValueError):
            parse_boolean(["yes"])


class TestTupleParsing(unittest.TestCase):
    """Test suite for parsing and validating 5-element environment property tuples."""

    def test_parse_environment_tuple_valid_strings(self) -> None:
        """Verify parsing sequence of string tokens into canonical 5-boolean tuple."""
        raw = ["yes", "no", "no", "no", "yes"]
        parsed = parse_environment_tuple(raw)
        self.assertEqual(parsed, (True, False, False, False, True))

    def test_parse_environment_tuple_mixed_types(self) -> None:
        """Verify parsing mixed sequence of ints, bools, and strings."""
        raw = [1, False, "sequential", 0, "discrete"]
        parsed = parse_environment_tuple(raw)
        self.assertEqual(parsed, (True, False, False, False, True))

    def test_parse_environment_tuple_incorrect_length(self) -> None:
        """Verify tuple parser rejects sequences with length != 5."""
        with self.assertRaises(ValueError) as ctx_short:
            parse_environment_tuple([True, False, True])
        self.assertIn("Expected exactly 5 environment properties", str(ctx_short.exception))

        with self.assertRaises(ValueError) as ctx_long:
            parse_environment_tuple([True, False, True, False, True, True])
        self.assertIn("Expected exactly 5 environment properties", str(ctx_long.exception))

    def test_parse_environment_tuple_invalid_element(self) -> None:
        """Verify invalid element inside tuple raises ValueError."""
        with self.assertRaises(ValueError):
            parse_environment_tuple([True, "bogus", False, True, True])


class TestExactEnvironmentClassification(unittest.TestCase):
    """Test suite for exact O(1) dictionary lookup across textbook AI domains."""

    def setUp(self) -> None:
        self.classifier = EnvironmentClassifier()

    def test_sample_input_backgammon(self) -> None:
        """Verify Question 8 prompt sample: (True, False, False, False, True) -> Backgammon."""
        sample_tuple = (True, False, False, False, True)
        result = self.classifier.classify(sample_tuple)

        self.assertTrue(result.is_exact_match)
        self.assertEqual(result.matched_environment, "Backgammon")
        self.assertIsNotNone(result.metadata)
        self.assertIn("Backgammon", result.metadata.name)
        self.assertIn("dice", result.metadata.description.lower())
        self.assertEqual(len(result.closest_matches), 0)

    def test_chess_with_clock(self) -> None:
        """Verify Chess with clock: (Observable, Deterministic, Sequential, Dynamic, Discrete)."""
        chess_timed_tuple = (True, True, False, False, True)
        result = self.classifier.classify(chess_timed_tuple)

        self.assertTrue(result.is_exact_match)
        self.assertEqual(result.matched_environment, "Chess (with clock)")
        self.assertEqual(result.metadata.agent_type, "Multi-Agent (Competitive)")

    def test_chess_without_clock(self) -> None:
        """Verify Chess without clock: (Observable, Deterministic, Sequential, Static, Discrete)."""
        chess_static_tuple = (True, True, False, True, True)
        result = self.classifier.classify(chess_static_tuple)

        self.assertTrue(result.is_exact_match)
        self.assertIn("Chess", result.matched_environment)
        self.assertIn("Crossword", result.matched_environment)

    def test_poker_classification(self) -> None:
        """Verify Poker environments (partially observable, stochastic, sequential, discrete)."""
        # Timed poker: (False, False, False, False, True)
        poker_timed = (False, False, False, False, True)
        res_timed = self.classifier.classify(poker_timed)
        self.assertTrue(res_timed.is_exact_match)
        self.assertIn("Poker", res_timed.matched_environment)

        # Untimed poker: (False, False, False, True, True)
        poker_untimed = (False, False, False, True, True)
        res_untimed = self.classifier.classify(poker_untimed)
        self.assertTrue(res_untimed.is_exact_match)
        self.assertIn("Poker", res_untimed.matched_environment)

    def test_medical_diagnosis_and_taxi_driving(self) -> None:
        """Verify Medical Diagnosis / Taxi Driving: (False, False, False, False, False)."""
        complex_tuple = (False, False, False, False, False)
        result = self.classifier.classify(complex_tuple)

        self.assertTrue(result.is_exact_match)
        self.assertIn("Medical Diagnosis", result.matched_environment)
        self.assertIn("Taxi Driving", result.matched_environment)

    def test_part_picking_robot(self) -> None:
        """Verify Part-Picking Robot: (Partially Obs, Stochastic, Episodic, Dynamic, Continuous)."""
        part_picker = (False, False, True, False, False)
        result = self.classifier.classify(part_picker)

        self.assertTrue(result.is_exact_match)
        self.assertEqual(result.matched_environment, "Part-Picking Robot")

    def test_image_analysis(self) -> None:
        """Verify Image Analysis: (Observable, Deterministic, Episodic, Static, Discrete)."""
        image_analysis = (True, True, True, True, True)
        result = self.classifier.classify(image_analysis)

        self.assertTrue(result.is_exact_match)
        self.assertIn("Image Analysis", result.matched_environment)

    def test_convenience_function_matches_class(self) -> None:
        """Verify functional helper classify_environment produces identical result."""
        res_func = classify_environment([True, False, False, False, True])
        self.assertTrue(res_func.is_exact_match)
        self.assertEqual(res_func.matched_environment, "Backgammon")


class TestUnknownCombinationAndDistanceMetric(unittest.TestCase):
    """Test suite for handling unknown environment property combinations and Hamming distance."""

    def setUp(self) -> None:
        self.classifier = EnvironmentClassifier()

    def test_unknown_combination_without_nearest_fallback(self) -> None:
        """Verify classifier handles unknown combination gracefully when nearest is disabled."""
        unknown_tuple = (True, False, True, False, True)
        result = self.classifier.classify(unknown_tuple, allow_nearest=False)

        self.assertFalse(result.is_exact_match)
        self.assertIsNone(result.matched_environment)
        self.assertIsNone(result.metadata)
        self.assertEqual(len(result.closest_matches), 0)
        self.assertIn("No exact environment match found", result.message)

    def test_unknown_combination_with_nearest_fallback(self) -> None:
        """Verify optional challenge: Hamming distance identifies closest environment."""
        # (True, False, True, False, True): differs from Backgammon (True, False, False, False, True) by 1 dimension (Episodic)
        unknown_tuple = (True, False, True, False, True)
        result = self.classifier.classify(unknown_tuple, allow_nearest=True)

        self.assertFalse(result.is_exact_match)
        self.assertGreater(len(result.closest_matches), 0)

        # The closest match must be Backgammon at Hamming distance 1
        best_match, distance, diffs = result.closest_matches[0]
        self.assertEqual(best_match.name, "Backgammon")
        self.assertEqual(distance, 1)
        self.assertEqual(len(diffs), 1)
        self.assertIn("Episodic", diffs[0])

    def test_hamming_distance_calculation(self) -> None:
        """Verify Hamming distance mathematical properties."""
        t1 = (True, True, True, True, True)
        t2 = (True, True, True, True, True)
        t3 = (False, False, False, False, False)
        t4 = (True, False, True, False, True)

        # Identity: D_H(x, x) = 0
        self.assertEqual(calculate_hamming_distance(t1, t2), 0)
        # Complement: D_H(all True, all False) = 5
        self.assertEqual(calculate_hamming_distance(t1, t3), 5)
        # Partial: 2 differences
        self.assertEqual(calculate_hamming_distance(t1, t4), 2)
        # Symmetry: D_H(x, y) == D_H(y, x)
        self.assertEqual(
            calculate_hamming_distance(t1, t4),
            calculate_hamming_distance(t4, t1),
        )

    def test_explain_property_differences(self) -> None:
        """Verify explain_property_differences highlights specific mismatching dimensions."""
        src = (True, False, True, True, False)
        tgt = (True, True, True, False, False)
        # Differences at index 1 (Determinism) and index 3 (Statics)
        diffs = explain_property_differences(src, tgt)
        self.assertEqual(len(diffs), 2)
        self.assertTrue(any("Determinism" in d for d in diffs))
        self.assertTrue(any("Statics" in d for d in diffs))


class TestPEASMetadataAndFormatting(unittest.TestCase):
    """Test suite for PEAS descriptions, metadata integrity, and reporting."""

    def test_all_knowledge_base_entries_valid(self) -> None:
        """Ensure all predefined knowledge base entries have complete, non-empty fields."""
        self.assertGreater(len(KNOWLEDGE_BASE), 5)
        for key, meta in KNOWLEDGE_BASE.items():
            self.assertEqual(len(key), 5)
            self.assertEqual(key, meta.properties)
            self.assertTrue(meta.name.strip())
            self.assertTrue(meta.category.strip())
            self.assertTrue(meta.description.strip())
            self.assertTrue(meta.peas.performance_measure.strip())
            self.assertTrue(meta.peas.environment.strip())
            self.assertTrue(meta.peas.actuators.strip())
            self.assertTrue(meta.peas.sensors.strip())

    def test_peas_to_dict(self) -> None:
        """Verify PEASDescription.to_dict generates all 4 canonical keys."""
        peas = PEASDescription(
            performance_measure="Score",
            environment="Board",
            actuators="Hands",
            sensors="Eyes",
        )
        d = peas.to_dict()
        self.assertIn("Performance Measure", d)
        self.assertIn("Environment", d)
        self.assertIn("Actuators", d)
        self.assertIn("Sensors", d)

    def test_metadata_property_summary(self) -> None:
        """Verify property summary string format."""
        meta = KNOWLEDGE_BASE[(True, False, False, False, True)]
        summary = meta.property_summary
        self.assertIn("Fully Observable", summary)
        self.assertIn("Stochastic", summary)
        self.assertIn("Sequential", summary)
        self.assertIn("Dynamic", summary)
        self.assertIn("Discrete", summary)

    def test_format_classification_report_exact_match(self) -> None:
        """Verify string report contains required headings and PEAS breakdown."""
        result = classify_environment([True, False, False, False, True])
        report = format_classification_report(result)
        self.assertIn("ENVIRONMENT TYPE CLASSIFIER REPORT", report)
        self.assertIn("EXACT MATCH FOUND", report)
        self.assertIn("Backgammon", report)
        self.assertIn("PEAS Specification:", report)

    def test_format_classification_report_unknown(self) -> None:
        """Verify string report for unknown input lists nearest neighbors."""
        result = classify_environment([True, False, True, False, True], allow_nearest=True)
        report = format_classification_report(result)
        self.assertIn("UNKNOWN COMBINATION", report)
        self.assertIn("Nearest Environment Matches", report)

    def test_get_viva_reference(self) -> None:
        """Verify Viva Voce study guide contains all required topics."""
        viva_text = get_viva_reference()
        self.assertIn("PEAS DESCRIPTION", viva_text)
        self.assertIn("PROPERTIES OF CHESS VS. POKER", viva_text)
        self.assertIn("CONTINUOUS VS. DISCRETE", viva_text)


if __name__ == "__main__":
    unittest.main()
