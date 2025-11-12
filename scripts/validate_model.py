#!/usr/bin/env python3
"""
Model Validation Script
Quick sanity checks for model health

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

import sys

sys.path.insert(0, '.')

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.core.label_utils import encode_labels, validate_labels

# Import our utilities
from src.models.model_loader import ModelLoadError, load_catboost_model_v2


def validate_model(model_path: str, test_data_path: str) -> bool:
    """
    Run comprehensive model validation

    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 70)
    print("MODEL VALIDATION SCRIPT")
    print("=" * 70)

    all_passed = True

    # CHECK 1: Model loading
    print("\n📦 CHECK 1: Model Loading")
    try:
        model, metadata = load_catboost_model_v2(model_path)
        print("   ✅ Model loaded successfully")
        print(f"   ℹ️  Created: {metadata['created_at']}")
        print(f"   ℹ️  Features: {len(metadata['feature_names'])}")
    except Exception as e:
        print(f"   ❌ Model loading failed: {str(e)}")
        return False

    # CHECK 2: Model methods
    print("\n🔧 CHECK 2: Model Methods")
    has_predict = hasattr(model, 'predict')
    has_predict_proba = hasattr(model, 'predict_proba')

    if has_predict:
        print("   ✅ predict() method exists")
    else:
        print("   ❌ predict() method missing")
        all_passed = False

    if has_predict_proba:
        print("   ✅ predict_proba() method exists")
    else:
        print("   ⚠️  predict_proba() method missing")

    # CHECK 3: Test data loading
    print("\n📊 CHECK 3: Test Data Loading")
    try:
        df_test = pd.read_csv(test_data_path)
        print(f"   ✅ Test data loaded: {df_test.shape}")

        X_test = df_test.drop(columns=['koi_disposition'])
        y_test_raw = df_test['koi_disposition'].values

        print(f"   ℹ️  Features: {X_test.shape[1]}")
        print(f"   ℹ️  Samples: {len(X_test)}")

    except Exception as e:
        print(f"   ❌ Test data loading failed: {str(e)}")
        return False

    # CHECK 4: Feature count match
    print("\n🔢 CHECK 4: Feature Count Match")
    expected_features = len(metadata['feature_names'])
    actual_features = X_test.shape[1]

    if expected_features == actual_features:
        print(f"   ✅ Feature count matches: {expected_features}")
    else:
        print(f"   ❌ Feature mismatch: expected {expected_features}, got {actual_features}")
        all_passed = False

    # CHECK 5: Label format
    print("\n🏷️  CHECK 5: Label Format")
    print(f"   ℹ️  Label dtype: {y_test_raw.dtype}")
    print(f"   ℹ️  Unique labels: {np.unique(y_test_raw)}")

    try:
        validate_labels(y_test_raw)
        print("   ✅ Labels are valid")
    except ValueError as e:
        print(f"   ⚠️  Label validation warning: {str(e)}")

    # Normalize labels
    y_test = encode_labels(y_test_raw)
    print(f"   ✅ Labels normalized to numeric: {np.unique(y_test)}")

    # CHECK 6: Prediction test
    print("\n🔮 CHECK 6: Prediction Test")
    try:
        # Predict on small sample
        X_sample = X_test.head(10)
        predictions_raw = model.predict(X_sample)

        # Flatten if needed
        if len(predictions_raw.shape) > 1:
            predictions_raw = predictions_raw.flatten()

        print(f"   ✅ Predictions generated: {predictions_raw.shape}")
        print(f"   ℹ️  Prediction dtype: {predictions_raw.dtype}")
        print(f"   ℹ️  Unique predictions (raw): {np.unique(predictions_raw)}")

        # Normalize predictions
        predictions = encode_labels(predictions_raw)
        print(f"   ✅ Predictions normalized to numeric: {np.unique(predictions)}")
        print(f"   ℹ️  Predictions dtype after encoding: {predictions.dtype}")

    except Exception as e:
        print(f"   ❌ Prediction failed: {str(e)}")
        all_passed = False
        return all_passed

    # CHECK 7: Full test set prediction
    print("\n📈 CHECK 7: Full Test Set Prediction")
    try:
        predictions_raw = model.predict(X_test)

        # Flatten if needed
        if len(predictions_raw.shape) > 1:
            predictions_raw = predictions_raw.flatten()

        # Encode to numeric
        predictions = encode_labels(predictions_raw)

        # Debug info
        print(f"   ℹ️  y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
        print(f"   ℹ️  predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"   ℹ️  y_test sample: {y_test[:3]}")
        print(f"   ℹ️  predictions sample: {predictions[:3]}")

        accuracy = accuracy_score(y_test, predictions)

        print(f"   ✅ Full predictions successful")
        print(f"   ℹ️  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Accuracy threshold check
        if accuracy >= 0.80:
            print(f"   ✅ Accuracy meets threshold (≥0.80)")
        else:
            print(f"   ❌ Accuracy below threshold: {accuracy:.4f} < 0.80")
            all_passed = False

    except Exception as e:
        print(f"   ❌ Full prediction failed: {str(e)}")
        all_passed = False
        return all_passed

    # CHECK 8: CANDIDATE recall
    print("\n🎯 CHECK 8: CANDIDATE Recall")
    try:
        candidate_mask = y_test == 0
        candidate_predictions = predictions[candidate_mask]
        candidate_recall = (candidate_predictions == 0).mean()

        print(f"   ℹ️  CANDIDATE Recall: {candidate_recall:.4f} ({candidate_recall*100:.2f}%)")

        if candidate_recall >= 0.70:
            print(f"   ✅ CANDIDATE recall meets threshold (≥0.70)")
        else:
            print(f"   ❌ CANDIDATE recall below threshold: {candidate_recall:.4f} < 0.70")
            all_passed = False

    except Exception as e:
        print(f"   ❌ CANDIDATE recall check failed: {str(e)}")
        all_passed = False

    # CHECK 9: Confidence analysis
    print("\n📊 CHECK 9: Confidence Analysis")
    try:
        probabilities = model.predict_proba(X_test)
        max_probs = probabilities.max(axis=1)

        mean_conf = max_probs.mean()
        low_conf_pct = (max_probs < 0.6).mean() * 100

        print(f"   ℹ️  Mean confidence: {mean_conf:.4f}")
        print(f"   ℹ️  Low confidence (<0.6): {low_conf_pct:.2f}%")

        if mean_conf >= 0.75:
            print(f"   ✅ Mean confidence acceptable (≥0.75)")
        else:
            print(f"   ⚠️  Low mean confidence: {mean_conf:.4f}")

        if low_conf_pct < 15:
            print(f"   ✅ Low confidence rate acceptable (<15%)")
        else:
            print(f"   ⚠️  High low-confidence rate: {low_conf_pct:.2f}%")

    except Exception as e:
        print(f"   ⚠️  Confidence analysis failed: {str(e)}")

    # FINAL RESULT
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CRITICAL CHECKS PASSED")
        print("✅ Model is healthy and ready for use")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("❌ Model needs attention")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate CatBoost model")
    parser.add_argument('--model', default='models/v2_final/catboost_v2_final.pkl', help='Path to model file')
    parser.add_argument('--test-data', default='data/selected/test_selected.csv', help='Path to test data')

    args = parser.parse_args()

    success = validate_model(args.model, args.test_data)

    sys.exit(0 if success else 1)
