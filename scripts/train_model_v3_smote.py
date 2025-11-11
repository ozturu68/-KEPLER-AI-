#!/usr/bin/env python3
"""
CatBoost v3 - SMOTE Oversampling Training Script

Bu script, SMOTE (Synthetic Minority Over-sampling Technique) ile
CANDIDATE class'ını oversample eder ve model train eder.

SMOTE Stratejisi:
- CANDIDATE class için sentetik örnekler üretir
- K-Nearest Neighbors algoritması kullanır
- Data imbalance'ı azaltır

Test Edilecek SMOTE Varyantları:
1. Baseline (No SMOTE)           → Mevcut referans (v2)
2. SMOTE Default (k=5)           → Standard SMOTE
3. SMOTE Conservative (k=3)      → Daha az agresif
4. SMOTE Aggressive (k=7)        → Daha agresif
5. SMOTE + Class Weights         → Hybrid approach

Hedef: CANDIDATE recall 87.54% → 90%+

Usage:
    python scripts/train_model_v3_smote.py
    
Output:
    models/v3_smote/
    ├── 1_baseline_no_smote.pkl
    ├── 2_smote_default_k5.pkl
    ├── 3_smote_conservative_k3.pkl
    ├── 4_smote_aggressive_k7.pkl
    ├── 5_smote_with_weights.pkl
    ├── comparison_report.json
    └── comparison_summary.txt

Author: sulegogh
Date: 2025-11-11 19:33:00 UTC
Version: 3.0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core import TARGET_COLUMN
from src.models import CatBoostModel


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    log_file = (
        PROJECT_ROOT
        / "logs"
        / f"v3_smote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="DEBUG")


def load_data():
    """Train/val/test verilerini yükle."""
    logger.info("📂 Veri yükleniyor...")

    data_dir = PROJECT_ROOT / "data" / "selected"

    train_df = pd.read_csv(data_dir / "train_selected.csv")
    val_df = pd.read_csv(data_dir / "val_selected.csv")
    test_df = pd.read_csv(data_dir / "test_selected.csv")

    # NaN temizle
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.debug(f"   {name}: {nan_count} NaN → 0")
            df.fillna(0, inplace=True)

    logger.info(f"   Train: {len(train_df):,} samples")
    logger.info(f"   Val:   {len(val_df):,} samples")
    logger.info(f"   Test:  {len(test_df):,} samples")

    return train_df, val_df, test_df


def prepare_data(train_df, val_df, test_df):
    """Veriyi X, y olarak ayır."""
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_val, y_val, X_test, y_test


def show_class_distribution(y, label="Dataset"):
    """Class dağılımını göster."""
    from collections import Counter

    counts = Counter(y)
    total = len(y)

    logger.info(f"   {label} Class Distribution:")
    for cls in sorted(counts.keys()):
        count = counts[cls]
        pct = count / total * 100
        logger.info(f"      {cls:<15s}: {count:>5d} ({pct:>5.2f}%)")


def apply_smote(X_train, y_train, k_neighbors=5, sampling_strategy="auto"):
    """SMOTE uygula."""
    logger.info(f"   🔧 SMOTE uygulanıyor (k_neighbors={k_neighbors})...")

    # Önceki dağılım
    show_class_distribution(y_train, "Before SMOTE")

    # SMOTE
    smote = SMOTE(
        k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Sonraki dağılım
    show_class_distribution(y_train_resampled, "After SMOTE")

    logger.info(
        f"   ✓ SMOTE tamamlandı: {len(X_train):,} → {len(X_train_resampled):,} samples"
    )

    return X_train_resampled, y_train_resampled


def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name="Dataset"):
    """Detaylı metrikleri hesapla."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        )
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    class_wise = {}
    for cls in sorted(np.unique(y_true)):
        if cls in report:
            class_wise[cls] = {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1_score": report[cls]["f1-score"],
                "support": int(report[cls]["support"]),
            }

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "class_wise": class_wise,
    }


def train_strategy(
    strategy_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    use_smote=False,
    k_neighbors=5,
    use_class_weights=False,
):
    """Bir strateji ile model train et."""
    logger.info(f"\n🔧 Training: {strategy_name}")

    # SMOTE uygula
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(
            X_train, y_train, k_neighbors=k_neighbors
        )
    else:
        logger.info("   ⚠️  SMOTE kullanılmıyor (baseline)")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Model oluştur
    if use_class_weights:
        logger.info("   🎚️  Class weights: [3.0, 1.0, 0.5] (v2 best)")
        model = CatBoostModel(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            class_weights=[3.0, 1.0, 0.5],
            verbose=False,
        )
    else:
        logger.info("   🎚️  Class weights: None")
        model = CatBoostModel(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            verbose=False,
        )

    # Train
    logger.info("   Training başlıyor...")
    model.fit(X_train_resampled, y_train_resampled, X_val, y_val)
    logger.info(f"   ✓ Training tamamlandı ({model.training_time:.2f}s)")

    # Predictions (ORIGINAL test set üzerinde!)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # Calculate metrics
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "test")

    # Key metrics
    candidate_recall = test_metrics["class_wise"].get("CANDIDATE", {}).get("recall", 0)
    confirmed_recall = test_metrics["class_wise"].get("CONFIRMED", {}).get("recall", 0)
    fp_recall = (
        test_metrics["class_wise"].get("FALSE POSITIVE", {}).get("recall", 0)
    )

    logger.info("   📊 Test Results:")
    logger.info(
        f"      Accuracy:              {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)"
    )
    logger.info(f"      F1 Score:              {test_metrics['f1_score']:.4f}")
    logger.info(
        f"      CANDIDATE Recall:      {candidate_recall:.4f} ({candidate_recall*100:.2f}%)"
    )
    logger.info(
        f"      CONFIRMED Recall:      {confirmed_recall:.4f} ({confirmed_recall*100:.2f}%)"
    )
    logger.info(
        f"      FALSE POSITIVE Recall: {fp_recall:.4f} ({fp_recall*100:.2f}%)"
    )

    return {
        "strategy": strategy_name,
        "use_smote": use_smote,
        "k_neighbors": k_neighbors if use_smote else None,
        "use_class_weights": use_class_weights,
        "model": model,
        "test_metrics": test_metrics,
        "candidate_recall": candidate_recall,
        "confirmed_recall": confirmed_recall,
        "false_positive_recall": fp_recall,
        "training_time": model.training_time,
        "train_samples": len(X_train_resampled),
    }


def compare_results(results, output_dir):
    """Sonuçları karşılaştır."""
    logger.info("\n" + "=" * 110)
    logger.info("📊 STRATEGY COMPARISON")
    logger.info("=" * 110)

    header = f"\n{'Strategy':<32s} {'Samples':<9s} {'Accuracy':<10s} {'F1':<8s} {'CAN':<8s} {'CON':<8s} {'FP':<8s} {'Improvement':<12s}"
    logger.info(header)
    logger.info("-" * 110)

    baseline_recall = results[0]["candidate_recall"]

    for result in results:
        strategy = result["strategy"]
        samples = result["train_samples"]
        accuracy = result["test_metrics"]["accuracy"]
        f1 = result["test_metrics"]["f1_score"]
        can = result["candidate_recall"]
        con = result["confirmed_recall"]
        fp = result["false_positive_recall"]
        improvement = (can - baseline_recall) * 100

        row = (
            f"{strategy:<32s} "
            f"{samples:<9,} "
            f"{accuracy:<10.4f} "
            f"{f1:<8.4f} "
            f"{can:<8.4f} "
            f"{con:<8.4f} "
            f"{fp:<8.4f} "
            f"{improvement:+11.2f}%"
        )
        logger.info(row)

    # En iyi stratejiyi bul
    best_result = max(results[1:], key=lambda x: x["candidate_recall"])

    logger.info("\n" + "=" * 110)
    logger.info(f"🏆 BEST STRATEGY: {best_result['strategy']}")
    logger.info("=" * 110)
    logger.info(
        f"   CANDIDATE Recall:  {baseline_recall:.4f} → {best_result['candidate_recall']:.4f}"
    )
    logger.info(
        f"   Improvement:       {(best_result['candidate_recall'] - baseline_recall)*100:+.2f}%"
    )
    logger.info(f"   Test Accuracy:     {best_result['test_metrics']['accuracy']:.4f}")
    logger.info(f"   Training Time:     {best_result['training_time']:.2f}s")
    logger.info(f"   Train Samples:     {best_result['train_samples']:,}")

    # JSON rapor
    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "user": "sulegogh",
        "baseline_recall": float(baseline_recall),
        "best_strategy": best_result["strategy"],
        "best_recall": float(best_result["candidate_recall"]),
        "improvement_pct": float(
            (best_result["candidate_recall"] - baseline_recall) * 100
        ),
        "strategies": [],
    }

    for result in results:
        report["strategies"].append(
            {
                "name": result["strategy"],
                "use_smote": result["use_smote"],
                "k_neighbors": result["k_neighbors"],
                "use_class_weights": result["use_class_weights"],
                "train_samples": result["train_samples"],
                "test_accuracy": float(result["test_metrics"]["accuracy"]),
                "test_f1": float(result["test_metrics"]["f1_score"]),
                "candidate_recall": float(result["candidate_recall"]),
                "training_time": float(result["training_time"]),
            }
        )

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n💾 JSON rapor: {report_path.name}")

    # Text summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("CATBOOST v3 - SMOTE OVERSAMPLING COMPARISON\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"User: sulegogh\n\n")

        f.write(header + "\n")
        f.write("-" * 110 + "\n")

        for result in results:
            strategy = result["strategy"]
            samples = result["train_samples"]
            accuracy = result["test_metrics"]["accuracy"]
            f1 = result["test_metrics"]["f1_score"]
            can = result["candidate_recall"]
            con = result["confirmed_recall"]
            fp = result["false_positive_recall"]
            improvement = (can - baseline_recall) * 100

            row = (
                f"{strategy:<32s} "
                f"{samples:<9,} "
                f"{accuracy:<10.4f} "
                f"{f1:<8.4f} "
                f"{can:<8.4f} "
                f"{con:<8.4f} "
                f"{fp:<8.4f} "
                f"{improvement:+11.2f}%\n"
            )
            f.write(row)

        f.write("\n" + "=" * 110 + "\n")
        f.write(f"BEST: {best_result['strategy']}\n")
        f.write("=" * 110 + "\n")

    logger.info(f"💾 Text summary: {summary_path.name}")

    return best_result


def save_models(results, output_dir):
    """Model'leri kaydet."""
    logger.info("\n" + "=" * 110)
    logger.info("💾 SAVING MODELS")
    logger.info("=" * 110)

    for result in results:
        strategy = result["strategy"]
        model = result["model"]

        filename = (
            strategy.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
            + ".pkl"
        )
        filepath = output_dir / filename

        model.save(filepath)

        file_size = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"   ✓ {strategy:<32s} → {filepath.name} ({file_size:.2f} MB)")

    logger.info(f"\n📁 Tüm modeller: {output_dir}")


def main():
    """Ana fonksiyon."""
    setup_logger()

    logger.info("=" * 110)
    logger.info("🔬 CATBOOST v3 - SMOTE OVERSAMPLING TRAINING")
    logger.info("=" * 110)
    logger.info(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("👤 Kullanıcı: sulegogh")
    logger.info("🎯 Hedef: CANDIDATE recall 87.54% → 90%+")
    logger.info("📊 Test Stratejileri: 5 (Baseline + 4 SMOTE variants)")
    logger.info("=" * 110)

    # Output directory
    output_dir = PROJECT_ROOT / "models" / "v3_smote"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Veri yükle
    train_df, val_df, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_df, val_df, test_df
    )

    # Class distribution (original)
    logger.info("\n📊 Original Class Distribution:")
    show_class_distribution(y_train, "Train")

    # Stratejiler
    strategies = [
        ("1. Baseline (No SMOTE)", False, 5, False),
        ("2. SMOTE Default (k=5)", True, 5, False),
        ("3. SMOTE Conservative (k=3)", True, 3, False),
        ("4. SMOTE Aggressive (k=7)", True, 7, False),
        ("5. SMOTE + Class Weights", True, 5, True),
    ]

    # Her stratejiyi train et
    results = []

    for strategy_name, use_smote, k_neighbors, use_weights in strategies:
        logger.info("\n" + "=" * 110)

        try:
            result = train_strategy(
                strategy_name,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                use_smote=use_smote,
                k_neighbors=k_neighbors,
                use_class_weights=use_weights,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"❌ {strategy_name} failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    if not results:
        logger.error("❌ Hiçbir strateji başarılı olmadı!")
        return 1

    # Karşılaştır
    best_result = compare_results(results, output_dir)

    # Kaydet
    save_models(results, output_dir)

    # Özet
    logger.info("\n" + "=" * 110)
    logger.info("✅ v3 SMOTE TRAINING TAMAMLANDI!")
    logger.info("=" * 110)
    logger.info(
        f"📊 Baseline Recall:      {results[0]['candidate_recall']:.4f} ({results[0]['candidate_recall']*100:.2f}%)"
    )
    logger.info(
        f"📊 Best Recall:          {best_result['candidate_recall']:.4f} ({best_result['candidate_recall']*100:.2f}%)"
    )
    logger.info(
        f"📊 Improvement:          {(best_result['candidate_recall'] - results[0]['candidate_recall'])*100:+.2f}%"
    )
    logger.info(f"🏆 Best Strategy:        {best_result['strategy']}")
    logger.info(f"📁 Models:               {output_dir}")
    logger.info("=" * 110)

    return 0


if __name__ == "__main__":
    sys.exit(main())