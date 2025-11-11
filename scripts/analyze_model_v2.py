#!/usr/bin/env python3
"""
v2 Final Model - Comprehensive Analysis Script
================================================

Author: sulegogh
Date: 2025-11-11
Version: 2.0

Description:
    Kapsamlı v2 Final model analizi yapar:
    - Confusion matrix visualization
    - Feature importance analysis (top 20)
    - Detailed classification report
    - Error analysis (class-wise breakdown)
    - Performance metrics visualization

    Tüm çıktılar logs/analysis_v2/ klasörüne kaydedilir.

Usage:
    python scripts/analyze_model_v2.py [--model MODEL_PATH]

    Options:
        --model: Model path (default: models/v2_final/catboost_v2_final)

    Examples:
        # Use default model
        python scripts/analyze_model_v2.py

        # Use custom model
        python scripts/analyze_model_v2.py --model models/v2_class_weights/4_manual_aggressive

Output Files:
    - logs/analysis_v2/confusion_matrix.png
    - logs/analysis_v2/feature_importance_top20.png
    - logs/analysis_v2/feature_importance_full.csv
    - logs/analysis_v2/class_distribution.png
    - logs/analysis_v2/error_analysis.png
    - logs/analysis_v2/analysis_report.txt

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - catboost
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.catboost_model import CatBoostModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths (can be overridden via command-line)
DEFAULT_MODEL_PATH = "models/v2_final/catboost_v2_final"
TEST_DATA_PATH = PROJECT_ROOT / "data/selected/test_selected.csv"
OUTPUT_DIR = PROJECT_ROOT / "logs/analysis_v2"

# Class names
CLASS_NAMES = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
CLASS_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# Plot style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ============================================================================
# ARGUMENT PARSER
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="v2 Final Model - Comprehensive Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model
  python scripts/analyze_model_v2.py

  # Use specific model
  python scripts/analyze_model_v2.py --model models/v2_class_weights/4_manual_aggressive

  # Use custom output directory
  python scripts/analyze_model_v2.py --output logs/custom_analysis
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH})",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix with annotations.

    Args:
        cm: Confusion matrix (numpy array)
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
        ax=ax,
        linewidths=2,
        linecolor="white",
        square=True,
    )

    # Title and labels
    ax.set_title(
        "Confusion Matrix - v2 Final Model\nCatBoost with Manual Aggressive Weights",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")

    # Add percentages
    total = cm.sum()
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            percentage = cm[i, j] / total * 100
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Plot top N feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        save_path: Output file path
    """
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_features["importance"].values, color=colors)

    # Y-axis labels
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=11)

    # Labels and title
    ax.set_xlabel("Importance (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=13, fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances - v2 Final Model", fontsize=16, fontweight="bold", pad=20)

    # Add value labels on bars
    for i, value in enumerate(top_features["importance"].values):
        ax.text(value + 0.2, i, f"{value:.2f}%", va="center", fontsize=9, fontweight="bold")

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Invert y-axis for top-to-bottom order
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_class_distribution(y_test, y_pred, save_path):
    """
    Plot true vs predicted class distribution.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # True distribution
    true_counts = pd.Series(y_test).value_counts().sort_index()
    ax1.bar(range(len(CLASS_NAMES)), true_counts.values, color=CLASS_COLORS, alpha=0.7)
    ax1.set_xticks(range(len(CLASS_NAMES)))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title("True Label Distribution", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add counts on bars
    for i, count in enumerate(true_counts.values):
        ax1.text(i, count + 10, str(count), ha="center", fontweight="bold")

    # Predicted distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    ax2.bar(range(len(CLASS_NAMES)), pred_counts.values, color=CLASS_COLORS, alpha=0.7)
    ax2.set_xticks(range(len(CLASS_NAMES)))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title("Predicted Label Distribution", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Add counts on bars
    for i, count in enumerate(pred_counts.values):
        ax2.text(i, count + 10, str(count), ha="center", fontweight="bold")

    plt.suptitle("Class Distribution Comparison - v2 Final Model", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_error_analysis(y_test, y_pred, save_path):
    """
    Plot error analysis by class.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate metrics for each class
    correct_counts = []
    error_counts = []

    for i in range(len(CLASS_NAMES)):
        mask = y_test == i
        correct = ((y_test == i) & (y_pred == i)).sum()
        errors = mask.sum() - correct
        correct_counts.append(correct)
        error_counts.append(errors)

    # Stacked bar chart
    x = np.arange(len(CLASS_NAMES))
    width = 0.6

    ax.bar(x, correct_counts, width, label="Correct", color="#2ECC71", alpha=0.8)
    ax.bar(x, error_counts, width, bottom=correct_counts, label="Errors", color="#E74C3C", alpha=0.8)

    # Labels and title
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=13, fontweight="bold")
    ax.set_title("Error Analysis by Class - v2 Final Model", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add percentages
    for i in range(len(CLASS_NAMES)):
        total = correct_counts[i] + error_counts[i]
        if total > 0:
            correct_pct = correct_counts[i] / total * 100
            error_pct = error_counts[i] / total * 100

            # Correct percentage
            if correct_counts[i] > 0:
                ax.text(
                    i,
                    correct_counts[i] / 2,
                    f"{correct_pct:.1f}%",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=11,
                    color="white",
                )

            # Error percentage
            if error_counts[i] > 0:
                ax.text(
                    i,
                    correct_counts[i] + error_counts[i] / 2,
                    f"{error_pct:.1f}%",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=11,
                    color="white",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {save_path}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_errors(y_test, y_pred, _X_test):
    """
    Perform detailed error analysis.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        _X_test: Test features (unused but kept for API consistency)

    Returns:
        dict: Error analysis results
    """
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]

    error_analysis = {
        "total_samples": len(y_test),
        "total_errors": len(error_indices),
        "error_rate": len(error_indices) / len(y_test) * 100,
        "correct_predictions": len(y_test) - len(error_indices),
        "accuracy": (len(y_test) - len(error_indices)) / len(y_test) * 100,
        "errors_by_class": {},
    }

    # Per-class error analysis
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        class_total = class_mask.sum()
        class_errors = (errors & class_mask).sum()
        class_correct = class_total - class_errors

        error_analysis["errors_by_class"][class_name] = {
            "total": int(class_total),
            "correct": int(class_correct),
            "errors": int(class_errors),
            "accuracy": float(class_correct / class_total * 100) if class_total > 0 else 0.0,
            "error_rate": float(class_errors / class_total * 100) if class_total > 0 else 0.0,
        }

    return error_analysis


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main analysis function."""

    # Parse arguments
    args = parse_args()

    # Setup paths
    model_path = PROJECT_ROOT / args.model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print("=" * 80)
    print("🔍 v2 FINAL MODEL - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 Author: sulegogh")
    print(f"📁 Model: {model_path}")
    print(f"📊 Output: {output_dir}")
    print()

    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    print("📦 Step 1/7: Loading v2 Final Model...")
    try:
        model = CatBoostModel.load(str(model_path))
        print("   ✓ Model loaded successfully")

        # Get model file size (.cbm file)
        cbm_path = Path(str(model_path) + ".cbm")
        if cbm_path.exists():
            model_size = cbm_path.stat().st_size / (1024 * 1024)
            print(f"   Model size: {model_size:.2f} MB")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return 1

    # ========================================================================
    # 2. LOAD TEST DATA
    # ========================================================================
    print("\n📊 Step 2/7: Loading test data...")
    try:
        df_test = pd.read_csv(TEST_DATA_PATH)

        # Separate features and target
        X_test = df_test.drop(columns=["koi_disposition"])
        y_test_raw = df_test["koi_disposition"]

        # Convert target to numeric if it's string type
        if y_test_raw.dtype == "object" or y_test_raw.dtype.name == "string":
            # String labels detected, convert to numeric
            label_mapping = {"CANDIDATE": 0, "CONFIRMED": 1, "FALSE POSITIVE": 2}

            y_test = y_test_raw.map(label_mapping).values

            # Verify conversion
            if np.isnan(y_test).any():
                unmapped = y_test_raw[pd.isna(y_test_raw.map(label_mapping))].unique()
                print(f"⚠️  Unmapped labels found: {unmapped}. " "These will be NaN in target array.")

            print(f"   ℹ️  Converted string labels to numeric (0=CANDIDATE, 1=CONFIRMED, 2=FALSE POSITIVE)")
        elif pd.api.types.is_numeric_dtype(y_test_raw):
            # Already numeric
            y_test = y_test_raw.values
            print(f"   ℹ️  Target already numeric")
        else:
            # Unknown type
            raise ValueError(f"Unsupported target dtype: {y_test_raw.dtype}. " "Expected 'object' (string) or numeric.")

        # Validate data
        if len(X_test) != len(y_test):
            raise ValueError(f"Feature-target length mismatch: X_test={len(X_test)}, y_test={len(y_test)}")

        # Check for missing values
        if np.isnan(y_test).any():
            n_missing = np.isnan(y_test).sum()
            raise ValueError(
                f"Target contains {n_missing} missing values after conversion. " "Check label_mapping completeness."
            )

        # Summary
        print(f"   ✓ Loaded {len(X_test)} test samples")
        print(f"   Features: {X_test.shape[1]}")
        print(f"   Target type: {y_test.dtype}")
        print(f"   Target range: [{int(y_test.min())}, {int(y_test.max())}]")
        print(f"   Unique classes: {np.unique(y_test).tolist()}")

        if args.verbose:
            # Class distribution
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"\n   Class Distribution:")
            for cls, count in zip(unique, counts):
                cls_name = CLASS_NAMES[int(cls)]
                percentage = count / len(y_test) * 100
                print(f"     {cls} ({cls_name:15s}): {count:4d} ({percentage:5.2f}%)")

    except FileNotFoundError:
        print(f"   ❌ Error: Test data file not found: {TEST_DATA_PATH}")
        return 1
    except KeyError as e:
        print(f"   ❌ Error: Required column not found: {e}")
        print(f"   Available columns: {df_test.columns.tolist()}")
        return 1
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"   ❌ Error loading test data: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # ========================================================================
    # 3. GENERATE PREDICTIONS
    # ========================================================================
    print("\n🎯 Step 3/7: Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print("   ✓ Predictions generated")

    if args.verbose:
        print(f"   Predictions shape: {y_pred.shape}")
        print(f"   Probabilities shape: {y_proba.shape if hasattr(y_proba, 'shape') else 'N/A'}")

    # ========================================================================
    # 4. CLASSIFICATION REPORT
    # ========================================================================
    print("\n📈 Step 4/7: Generating classification report...")
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")

    # ========================================================================
    # 5. CONFUSION MATRIX
    # ========================================================================
    print("\n📊 Step 5/7: Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    print("\nConfusion Matrix (Raw Counts):")
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in CLASS_NAMES],
        columns=[f"Pred {name}" for name in CLASS_NAMES],
    )
    print(cm_df.to_string())

    # ========================================================================
    # 6. FEATURE IMPORTANCE
    # ========================================================================
    print("\n🔍 Step 6/7: Analyzing feature importance...")
    importance_df = model.get_feature_importance()

    # Plot top 20
    plot_feature_importance(importance_df, top_n=20, save_path=output_dir / "feature_importance_top20.png")

    # Save full importance
    importance_df.to_csv(output_dir / "feature_importance_full.csv", index=False)
    print(f"✓ Saved: {output_dir / 'feature_importance_full.csv'}")

    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))

    # ========================================================================
    # 7. ERROR ANALYSIS & VISUALIZATIONS
    # ========================================================================
    print("\n⚠️  Step 7/7: Performing error analysis...")

    # Error analysis
    error_analysis = analyze_errors(y_test, y_pred, X_test)

    print("\nError Analysis Summary:")
    print(f"  Total Samples: {error_analysis['total_samples']}")
    print(f"  Correct Predictions: {error_analysis['correct_predictions']}")
    print(f"  Total Errors: {error_analysis['total_errors']}")
    print(f"  Overall Accuracy: {error_analysis['accuracy']:.2f}%")
    print(f"  Error Rate: {error_analysis['error_rate']:.2f}%")

    print("\nPer-Class Metrics:")
    for class_name, stats in error_analysis["errors_by_class"].items():
        print(
            f"  {class_name:20s}: "
            f"{stats['correct']:3d}/{stats['total']:3d} correct "
            f"({stats['accuracy']:5.2f}% accuracy, "
            f"{stats['errors']} errors)"
        )

    # Additional visualizations
    print("\n📊 Generating additional visualizations...")
    plot_class_distribution(y_test, y_pred, output_dir / "class_distribution.png")
    plot_error_analysis(y_test, y_pred, output_dir / "error_analysis.png")

    # ========================================================================
    # 8. SAVE COMPREHENSIVE REPORT
    # ========================================================================
    print("\n💾 Saving comprehensive report...")
    report_path = output_dir / "analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("v2 FINAL MODEL - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"Author: sulegogh\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Features: {X_test.shape[1]}\n\n")

        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(report + "\n\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Balanced Accuracy: {balanced_acc * 100:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n")
        f.write(cm_df.to_string() + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Samples: {error_analysis['total_samples']}\n")
        f.write(f"Correct Predictions: {error_analysis['correct_predictions']}\n")
        f.write(f"Total Errors: {error_analysis['total_errors']}\n")
        f.write(f"Overall Accuracy: {error_analysis['accuracy']:.2f}%\n")
        f.write(f"Error Rate: {error_analysis['error_rate']:.2f}%\n\n")

        f.write("Per-Class Metrics:\n")
        for class_name, stats in error_analysis["errors_by_class"].items():
            f.write(
                f"  {class_name:20s}: "
                f"{stats['correct']:3d}/{stats['total']:3d} correct "
                f"({stats['accuracy']:5.2f}% accuracy, "
                f"{stats['errors']} errors)\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 20 FEATURE IMPORTANCES\n")
        f.write("=" * 80 + "\n")
        f.write(importance_df.head(20).to_string(index=False) + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("GENERATED FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"1. {output_dir / 'confusion_matrix.png'}\n")
        f.write(f"2. {output_dir / 'feature_importance_top20.png'}\n")
        f.write(f"3. {output_dir / 'feature_importance_full.csv'}\n")
        f.write(f"4. {output_dir / 'class_distribution.png'}\n")
        f.write(f"5. {output_dir / 'error_analysis.png'}\n")
        f.write(f"6. {output_dir / 'analysis_report.txt'}\n")

    print(f"✓ Saved: {report_path}")

    # ========================================================================
    # 9. SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output Directory: {output_dir}")
    print("\nGenerated Files:")
    print("  1. confusion_matrix.png          (Confusion matrix heatmap)")
    print("  2. feature_importance_top20.png  (Top 20 features)")
    print("  3. feature_importance_full.csv   (All features)")
    print("  4. class_distribution.png        (True vs predicted)")
    print("  5. error_analysis.png            (Error breakdown)")
    print("  6. analysis_report.txt           (Comprehensive report)")
    print()

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
