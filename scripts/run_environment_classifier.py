"""Runner script for Environment Type Classifier (Question 8).

Provides interactive user prompting, command-line parameter parsing, standard
AI benchmark demonstrations, nearest-neighbor Hamming distance matching for
unseen combinations, and comprehensive Viva Voce reference notes.

Usage Examples:
    # 1. Interactive terminal prompt:
    python scripts/run_environment_classifier.py

    # 2. Direct tuple input (Sample: Backgammon):
    python scripts/run_environment_classifier.py --input True,False,False,False,True

    # 3. Individual named CLI parameters:
    python scripts/run_environment_classifier.py --observable yes --deterministic no --episodic no --static no --discrete yes

    # 4. Run automated textbook benchmark demonstrations:
    python scripts/run_environment_classifier.py --demo

    # 5. Run unknown combination and nearest-neighbor distance metric challenge:
    python scripts/run_environment_classifier.py --challenge

    # 6. Display Viva Voce study guide:
    python scripts/run_environment_classifier.py --viva
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    EnvironmentTuple,
    classify_environment,
    format_classification_report,
    get_viva_reference,
    parse_boolean,
    parse_environment_tuple,
)


def prompt_user_for_property(dimension_index: int) -> bool:
    """Prompt the user for a single environmental property with validation and retries.

    Args:
        dimension_index: 0 to 4 corresponding to PROPERTY_NAMES.

    Returns:
        Validated boolean value.
    """
    dim_name = PROPERTY_NAMES[dimension_index]
    true_label, false_label = PROPERTY_LABELS[dimension_index]

    prompt_msg = (
        f"[{dimension_index + 1}/5] {dim_name}:\n"
        f"      Is the environment {true_label}? [yes/no | true/false | y/n]: "
    )

    while True:
        try:
            raw_input = input(prompt_msg)
            val = parse_boolean(raw_input)
            chosen_label = true_label if val else false_label
            print(f"      -> Registered: {val} ({chosen_label})\n")
            return val
        except ValueError as err:
            print(f"      [!] Invalid input: {err}. Please enter 'yes' or 'no'.\n")


def collect_interactive_inputs() -> EnvironmentTuple:
    """Interactively collect the 5 environment properties from the user.

    Returns:
        Validated 5-element EnvironmentTuple.
    """
    print("=" * 78)
    print("        AI ENVIRONMENT TYPE CLASSIFIER - INTERACTIVE PROMPT")
    print("=" * 78)
    print("Please answer the following 5 questions about the agent's task environment:")
    print("  (Type 'yes' / 'y' / 'true' for affirmative; 'no' / 'n' / 'false' for negative)\n")

    p1 = prompt_user_for_property(INDEX_OBSERVABLE)
    p2 = prompt_user_for_property(INDEX_DETERMINISTIC)
    p3 = prompt_user_for_property(INDEX_EPISODIC)
    p4 = prompt_user_for_property(INDEX_STATIC)
    p5 = prompt_user_for_property(INDEX_DISCRETE)

    return (p1, p2, p3, p4, p5)


def run_demonstration_suite() -> None:
    """Run an automated demonstration across standard AI textbook environments."""
    print("=" * 78)
    print("       STANDARD AI TEXTBOOK ENVIRONMENTS DEMONSTRATION SUITE")
    print("=" * 78)
    print("Evaluating canonical benchmarks from Russell & Norvig Table 2.1:\n")

    classifier = EnvironmentClassifier()
    demo_cases: List[Tuple[str, EnvironmentTuple]] = [
        ("Sample Case (Backgammon)", (True, False, False, False, True)),
        ("Chess with a Clock", (True, True, False, False, True)),
        ("Chess without a Clock / Crossword", (True, True, False, True, True)),
        ("Poker (Timed / Dynamic)", (False, False, False, False, True)),
        ("Poker (Untimed / Static)", (False, False, False, True, True)),
        ("Medical Diagnosis / Taxi Driving", (False, False, False, False, False)),
        ("Part-Picking Robot", (False, False, True, False, False)),
        ("Image Analysis (Discrete)", (True, True, True, True, True)),
    ]

    for label, prop_tuple in demo_cases:
        print(f"\n>>> Evaluating: {label}")
        result = classifier.classify(prop_tuple)
        print(f"    Input Vector: {prop_tuple}")
        if result.is_exact_match and result.metadata is not None:
            print(f"    Match       : {result.matched_environment}")
            print(f"    Category    : {result.metadata.category}")
            print(f"    PEAS Measure: {result.metadata.peas.performance_measure}")
        else:
            print(f"    Status      : {result.message}")
    print("\n" + "=" * 78)


def run_challenge_demonstration() -> None:
    """Demonstrate nearest-neighbor Hamming distance matching for unknown combinations."""
    print("=" * 78)
    print("       OPTIONAL CHALLENGE: NEAREST-NEIGHBOR DISTANCE METRIC")
    print("=" * 78)
    print("Simulating unknown combinations not directly in the knowledge base,")
    print("evaluating Hamming distance D_H(u, v) = sum(u_i != v_i) to find closest matches:\n")

    classifier = EnvironmentClassifier()
    unknown_test_cases: List[EnvironmentTuple] = [
        # (True, False, True, False, True): Observable, Stochastic, Episodic, Dynamic, Discrete
        (True, False, True, False, True),
        # (False, True, True, False, True): Partially Obs, Deterministic, Episodic, Dynamic, Discrete
        (False, True, True, False, True),
    ]

    for test_tuple in unknown_test_cases:
        print(f"\n--- Testing Unknown Combination: {test_tuple} ---")
        result = classifier.classify(test_tuple, allow_nearest=True)
        print(format_classification_report(result))


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution handler for Environment Type Classifier."""
    parser = argparse.ArgumentParser(
        description="Question 8: Environment Type Classifier for Intelligent Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_environment_classifier.py\n"
            "  python scripts/run_environment_classifier.py --input True,False,False,False,True\n"
            "  python scripts/run_environment_classifier.py --demo\n"
            "  python scripts/run_environment_classifier.py --viva\n"
        ),
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Comma-separated 5-boolean tuple (e.g. 'True,False,False,False,True' or '1,0,0,0,1')",
    )
    parser.add_argument(
        "--observable",
        type=str,
        help="Observability: Yes/No, True/False (Fully Observable vs. Partially Observable)",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        help="Determinism: Yes/No, True/False (Deterministic vs. Stochastic)",
    )
    parser.add_argument(
        "--episodic",
        type=str,
        help="Episodic Nature: Yes/No, True/False (Episodic vs. Sequential)",
    )
    parser.add_argument(
        "--static",
        type=str,
        help="Statics: Yes/No, True/False (Static vs. Dynamic)",
    )
    parser.add_argument(
        "--discrete",
        type=str,
        help="Discreteness: Yes/No, True/False (Discrete vs. Continuous)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run canonical standard AI textbook environment demonstrations",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Demonstrate distance metric nearest-neighbor classification for unknown inputs",
    )
    parser.add_argument(
        "--viva",
        action="store_true",
        help="Display comprehensive Viva Voce questions, answers, and study notes",
    )

    args = parser.parse_args(argv)

    if args.viva:
        print(get_viva_reference())
        return 0

    if args.demo:
        run_demonstration_suite()
        return 0

    if args.challenge:
        run_challenge_demonstration()
        return 0

    # Check for direct comma-separated tuple input
    if args.input:
        tokens = [t.strip() for t in args.input.split(",")]
        try:
            prop_tuple = parse_environment_tuple(tokens)
        except ValueError as err:
            print(f"Error parsing --input: {err}", file=sys.stderr)
            return 1
    # Check if individual flags were provided
    elif any(
        arg is not None
        for arg in [
            args.observable,
            args.deterministic,
            args.episodic,
            args.static,
            args.discrete,
        ]
    ):
        if not all(
            arg is not None
            for arg in [
                args.observable,
                args.deterministic,
                args.episodic,
                args.static,
                args.discrete,
            ]
        ):
            print(
                "Error: When using individual dimension flags, all 5 dimensions must be specified "
                "(--observable, --deterministic, --episodic, --static, --discrete).",
                file=sys.stderr,
            )
            return 1
        try:
            prop_tuple = parse_environment_tuple([
                args.observable,
                args.deterministic,
                args.episodic,
                args.static,
                args.discrete,
            ])
        except ValueError as err:
            print(f"Error parsing property flags: {err}", file=sys.stderr)
            return 1
    else:
        # Default: Interactive prompt mode
        prop_tuple = collect_interactive_inputs()

    # Perform Classification
    classifier = EnvironmentClassifier()
    result = classifier.classify(prop_tuple, allow_nearest=True)
    report = format_classification_report(result)
    print(report)

    # Print clean Sample Output format expected by examiners:
    if result.is_exact_match:
        print(f"\nEnvironment Match: {result.matched_environment}")
    else:
        print(f"\nEnvironment Match: None (Unknown Combination)")
        if result.closest_matches:
            closest_names = ", ".join(m.name for m, _, _ in result.closest_matches)
            print(f"Closest Match(es): {closest_names}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
