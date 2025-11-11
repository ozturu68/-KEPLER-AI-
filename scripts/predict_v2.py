#!/usr/bin/env python3
"""
v2 Final Model - Production Inference Script
=============================================

Author: sulegogh
Date: 2025-11-11
Version: 1.0

Description:
    Production-ready inference script for v2 Final CatBoost model.
    Command-line interface for batch predictions with optional confidence scores.
    Supports validation mode when target column is present.

Usage:
    Basic prediction:
        python scripts/predict_v2.py --input data.csv --output predictions.csv

    With confidence scores:
        python scripts/predict_v2.py --input data.csv --output predictions.csv --confidence

    Verbose mode:
        python scripts/predict_v2.py --input data.csv --output predictions.csv -v

    All options:
        python scripts/predict_v2.py \\
            --input data/selected/test_selected.csv \\
            --output predictions.csv \\
            --confidence \\
            --verbose

Examples:
    # Predict on test set
    python scripts/predict_v2.py \\
        --input data/selected/test_selected.csv \\
        --output predictions.csv

    # Predict with confidence scores
    python scripts/predict_v2.py \\
        --input data/selected/test_selected.csv \\
        --output predictions_with_confidence.csv \\
        --confidence

    # Validation mode (auto-detected when target column exists)
    python scripts/predict_v2.py \\
        --input data/selected/test_selected.csv \\
        --output predictions.csv \\
        --verbose

Output Format:
    Without confidence:
        - All input columns
        - prediction (numeric: 0, 1, 2)
        - prediction_label (string: CANDIDATE, CONFIRMED, FALSE POSITIVE)

    With confidence:
        - All input columns
        - prediction (numeric)
        - prediction_label (string)
        - confidence (overall confidence: 0-1)
        - confidence_candidate (class 0 probability)
        - confidence_confirmed (class 1 probability)
        - confidence_false_positive (class 2 probability)

Requirements:
    - pandas
    - numpy
    - catboost
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.catboost_model import CatBoostModel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = PROJECT_ROOT / "models/v2_final/catboost_v2_final.pkl"
CLASS_NAMES = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ============================================================================
# ARGUMENT PARSER
# ============================================================================


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="v2 Final Model - Production Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic prediction:
    python scripts/predict_v2.py --input data.csv --output predictions.csv

  With confidence scores:
    python scripts/predict_v2.py --input data.csv --output predictions.csv --confidence

  Verbose mode:
    python scripts/predict_v2.py --input data.csv --output predictions.csv -v

Output Columns:
  - prediction: Numeric class (0=CANDIDATE, 1=CONFIRMED, 2=FALSE POSITIVE)
  - prediction_label: Human-readable class name
  - confidence: Overall confidence score (if --confidence flag is used)
  - confidence_<class>: Per-class probabilities (if --confidence flag is used)

Author: sulegogh
Date: 2025-11-11
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        metavar="FILE",
        help="Input CSV file path (required)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        metavar="FILE",
        help="Output CSV file path (required)",
    )

    # Optional arguments
    parser.add_argument(
        "--confidence",
        "-c",
        action="store_true",
        help="Include confidence scores in output (optional)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed statistics (optional)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        metavar="FILE",
        help=f"Model file path (default: {MODEL_PATH.name})",
    )

    return parser.parse_args()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def print_header(text: str, color: str = Colors.HEADER):
    """Print formatted header."""
    print(f"\n{color}{Colors.BOLD}{'=' * 80}")
    print(text)
    print("=" * 80 + Colors.ENDC)


def print_section(text: str):
    """Print formatted section header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def load_data(input_path: str) -> tuple[pd.DataFrame, np.ndarray | None, bool]:
    """
    Load input data and check for target column.

    Args:
        input_path: Path to input CSV file

    Returns:
        Tuple of (features DataFrame, true labels if exist, has_target flag)
    """
    df = pd.read_csv(input_path)

    # Check if target column exists
    has_target = "koi_disposition" in df.columns

    if has_target:
        X = df.drop(columns=["koi_disposition"])
        y_true = df["koi_disposition"].values
    else:
        X = df.copy()
        y_true = None

    return df, X, y_true, has_target


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False):
    """
    Validate predictions against true labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        verbose: Print detailed statistics

    Returns:
        dict: Validation metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    metrics = {"accuracy": accuracy, "per_class": {}}

    for i, class_name in enumerate(CLASS_NAMES):
        metrics["per_class"][class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }

    if verbose:
        print_section("📊 Validation Metrics")
        print(f"   Overall Accuracy: {accuracy * 100:.2f}%")
        print("\n   Per-Class Metrics:")
        for class_name, class_metrics in metrics["per_class"].items():
            print(
                f"   {class_name:20s}: "
                f"Precision={class_metrics['precision']:.4f}, "
                f"Recall={class_metrics['recall']:.4f}, "
                f"F1={class_metrics['f1']:.4f} "
                f"(n={class_metrics['support']})"
            )

    return metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main inference function."""

    # Parse arguments
    args = parse_args()

    # Header
    print_header("🔮 v2 FINAL MODEL - PRODUCTION INFERENCE")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 User: sulegogh")
    print(f"📂 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"📊 Confidence: {'Yes' if args.confidence else 'No'}")
    print(f"📢 Verbose: {'Yes' if args.verbose else 'No'}")

    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    print_section("📦 Step 1/6: Loading model...")
    try:
        model = CatBoostModel.load(args.model)
        print_success(f"Model loaded: {Path(args.model).name}")

        if args.verbose:
            model_size = Path(args.model).stat().st_size / (1024 * 1024)
            print(f"   Model size: {model_size:.2f} MB")
    except FileNotFoundError:
        print_error(f"Model file not found: {args.model}")
        return 1
    except Exception as e:
        print_error(f"Error loading model: {e}")
        return 1

    # ========================================================================
    # 2. LOAD INPUT DATA
    # ========================================================================
    print_section("📊 Step 2/6: Loading input data...")
    try:
        df_original, X, y_true, has_target = load_data(args.input)
        print_success(f"Loaded {len(X)} samples")

        if args.verbose:
            print(f"   Columns: {df_original.shape[1]}")
            print(f"   Features: {X.shape[1]}")
            print(f"   Shape: {X.shape}")

        if has_target:
            print_info("Target column found - Validation mode enabled")
        else:
            print_info("No target column - Prediction mode")

    except FileNotFoundError:
        print_error(f"Input file not found: {args.input}")
        return 1
    except Exception as e:
        print_error(f"Error loading data: {e}")
        return 1

    # ========================================================================
    # 3. VALIDATE FEATURES
    # ========================================================================
    print_section("🔧 Step 3/6: Validating features...")
    try:
        # Check feature count
        expected_features = 50  # v2 model uses 50 features
        if X.shape[1] != expected_features:
            print_warning(f"Expected {expected_features} features, got {X.shape[1]}")
            print_warning("Model may produce unexpected results!")
        else:
            print_success(f"Feature validation passed ({X.shape[1]} features)")

        # Check for missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print_warning(f"Found {missing_count} missing values")
            print_info("Filling missing values with median...")
            X = X.fillna(X.median())
        else:
            print_success("No missing values found")

    except Exception as e:
        print_error(f"Feature validation error: {e}")
        return 1

    # ========================================================================
    # 4. GENERATE PREDICTIONS
    # ========================================================================
    print_section("🎯 Step 4/6: Generating predictions...")
    try:
        y_pred = model.predict(X)
        print_success("Predictions generated")

        if args.confidence:
            y_proba = model.predict_proba(X)
            y_proba_array = y_proba if isinstance(y_proba, np.ndarray) else y_proba.values
            print_success("Confidence scores calculated")

    except Exception as e:
        print_error(f"Prediction error: {e}")
        return 1

    # ========================================================================
    # 5. PREPARE OUTPUT
    # ========================================================================
    print_section("📝 Step 5/6: Preparing output...")

    # Create output DataFrame (copy original to preserve all columns)
    df_output = df_original.copy()

    # Add predictions
    df_output["prediction"] = y_pred
    df_output["prediction_label"] = [CLASS_NAMES[i] for i in y_pred]

    # Add confidence scores if requested
    if args.confidence:
        df_output["confidence"] = y_proba_array.max(axis=1)
        df_output["confidence_candidate"] = y_proba_array[:, 0]
        df_output["confidence_confirmed"] = y_proba_array[:, 1]
        df_output["confidence_false_positive"] = y_proba_array[:, 2]

        print_success("Added confidence scores")

        if args.verbose:
            avg_confidence = df_output["confidence"].mean()
            min_confidence = df_output["confidence"].min()
            max_confidence = df_output["confidence"].max()
            print(f"   Average confidence: {avg_confidence:.2%}")
            print(f"   Min confidence: {min_confidence:.2%}")
            print(f"   Max confidence: {max_confidence:.2%}")

    # ========================================================================
    # 6. VALIDATION (if target exists)
    # ========================================================================
    if has_target:
        print_section("📊 Step 6/6: Validating predictions...")
        metrics = validate_predictions(y_true, y_pred, verbose=args.verbose)
    else:
        print_section("⏭️  Step 6/6: Skipping validation (no target column)")

    # ========================================================================
    # 7. SAVE OUTPUT
    # ========================================================================
    print_section("💾 Saving results...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False)
        print_success(f"Saved: {output_path}")

        if args.verbose:
            output_size = output_path.stat().st_size / (1024 * 1024)
            print(f"   Output size: {output_size:.2f} MB")

    except Exception as e:
        print_error(f"Error saving output: {e}")
        return 1

    # ========================================================================
    # 8. SUMMARY
    # ========================================================================
    print_header("✅ INFERENCE COMPLETE!", Colors.OKGREEN)

    # Prediction distribution
    print("\n📊 Prediction Distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = (y_pred == i).sum()
        pct = count / len(y_pred) * 100
        print(f"   {class_name:20s}: {count:5d} ({pct:5.2f}%)")

    # Output info
    print(f"\n📁 Output File: {args.output}")
    print(f"📊 Total Predictions: {len(y_pred)}")

    if args.confidence:
        avg_confidence = df_output["confidence"].mean()
        print(f"📈 Average Confidence: {avg_confidence:.2%}")

    # Validation summary
    if has_target:
        print(f"\n🎯 Validation Accuracy: {metrics['accuracy'] * 100:.2f}%")

        # Highlight CANDIDATE recall (primary metric)
        candidate_recall = metrics["per_class"]["CANDIDATE"]["recall"]
        print(f"⭐ CANDIDATE Recall: {candidate_recall * 100:.2f}%")

    print()
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
