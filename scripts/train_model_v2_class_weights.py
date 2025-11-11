#!/usr/bin/env python3
"""
CatBoost v2 - Class Weights Training Script

Bu script, CANDIDATE class'ının recall'ünü iyileştirmek için
farklı class weight stratejilerini test eder.

Test Edilen Stratejiler:
1. Baseline (no weights)      → Mevcut model referansı
2. Balanced (auto)             → CatBoost otomatik dengeler
3. Manual Conservative         → [2.5, 1.0, 0.7]
4. Manual Aggressive           → [3.0, 1.0, 0.5]
5. Inverse Frequency           → sklearn ile hesaplanır

Hedef: CANDIDATE recall 59.93% → 70%+

Usage:
    python scripts/train_model_v2_class_weights.py

Output:
    models/v2_class_weights/
    ├── 1_baseline_(no_weights).pkl
    ├── 2_balanced_(auto).pkl
    ├── 3_manual_conservative.pkl
    ├── 4_manual_aggressive.pkl
    ├── 5_inverse_frequency.pkl
    ├── comparison_report.json
    └── comparison_summary.txt

Author: sulegogh
Date: 2025-11-11 19:12:09 UTC
Version: 2.1 (Fixed class_wise metrics issue)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from src.core import TARGET_COLUMN
from src.models import CatBoostModel


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    log_file = PROJECT_ROOT / "logs" / f"v2_class_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
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


def compute_inverse_frequency_weights(y_train):
    """Inverse frequency weights hesapla."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    weight_dict = dict(zip(classes, weights))
    weight_list = [weight_dict[cls] for cls in sorted(classes)]

    logger.info(f"   Inverse frequency weights: {[f'{w:.4f}' for w in weight_list]}")

    return weight_list


def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name="Dataset"):
    """Detaylı metrikleri hesapla."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
    except Exception:
        roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report (class-wise metrics)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Class-wise metrics
    class_wise = {}
    for cls in sorted(np.unique(y_true)):
        if cls in report:
            class_wise[cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1_score': report[cls]['f1-score'],
                'support': int(report[cls]['support'])
            }

    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'class_wise': class_wise
    }


def train_strategy(strategy_name, class_weights, X_train, y_train, X_val, y_val, X_test, y_test):
    """Bir strateji ile model train et."""
    logger.info(f"\n🔧 Training: {strategy_name}")

    # Class weights'i logla
    if class_weights is None:
        logger.info(f"   Class weights: None (baseline)")
    elif class_weights == 'Balanced':
        logger.info(f"   Class weights: Auto-Balanced")
    else:
        logger.info(f"   Class weights: {[f'{w:.4f}' for w in class_weights]}")

    # Model oluştur
    if class_weights == 'Balanced':
        model = CatBoostModel(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            verbose=False
        )
    elif class_weights is None:
        model = CatBoostModel(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            verbose=False
        )
    else:
        model = CatBoostModel(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            class_weights=class_weights,
            verbose=False
        )

    # Train
    logger.info(f"   Training başlıyor...")
    model.fit(X_train, y_train, X_val, y_val)
    logger.info(f"   ✓ Training tamamlandı ({model.training_time:.2f}s)")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_proba = model.predict_proba(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "train")
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, "validation")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "test")

    # CANDIDATE recall (ana metrik)
    candidate_recall = test_metrics['class_wise'].get('CANDIDATE', {}).get('recall', 0)

    # Confirmed ve False Positive recall
    confirmed_recall = test_metrics['class_wise'].get('CONFIRMED', {}).get('recall', 0)
    false_positive_recall = test_metrics['class_wise'].get('FALSE POSITIVE', {}).get('recall', 0)

    logger.info(f"   📊 Test Results:")
    logger.info(f"      Accuracy:              {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"      F1 Score:              {test_metrics['f1_score']:.4f}")
    logger.info(f"      CANDIDATE Recall:      {candidate_recall:.4f} ({candidate_recall*100:.2f}%)")
    logger.info(f"      CONFIRMED Recall:      {confirmed_recall:.4f} ({confirmed_recall*100:.2f}%)")
    logger.info(f"      FALSE POSITIVE Recall: {false_positive_recall:.4f} ({false_positive_recall*100:.2f}%)")

    return {
        'strategy': strategy_name,
        'class_weights': str(class_weights) if class_weights != 'Balanced' else 'Balanced',
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'candidate_recall': candidate_recall,
        'confirmed_recall': confirmed_recall,
        'false_positive_recall': false_positive_recall,
        'training_time': model.training_time
    }


def compare_results(results, output_dir):
    """Sonuçları karşılaştır ve rapor oluştur."""
    logger.info("\n" + "="*100)
    logger.info("📊 STRATEGY COMPARISON")
    logger.info("="*100)

    # Tablo başlığı
    header = f"\n{'Strategy':<28s} {'Accuracy':<10s} {'F1':<8s} {'CAN Recall':<12s} {'CON Recall':<12s} {'FP Recall':<11s} {'Improvement':<12s}"
    logger.info(header)
    logger.info("-"*100)

    baseline_recall = results[0]['candidate_recall']

    for result in results:
        strategy = result['strategy']
        accuracy = result['test_metrics']['accuracy']
        f1 = result['test_metrics']['f1_score']
        can_recall = result['candidate_recall']
        con_recall = result['confirmed_recall']
        fp_recall = result['false_positive_recall']
        improvement = (can_recall - baseline_recall) * 100

        row = (
            f"{strategy:<28s} "
            f"{accuracy:<10.4f} "
            f"{f1:<8.4f} "
            f"{can_recall:<12.4f} "
            f"{con_recall:<12.4f} "
            f"{fp_recall:<11.4f} "
            f"{improvement:+11.2f}%"
        )
        logger.info(row)

    # En iyi stratejiyi bul
    best_result = max(results[1:], key=lambda x: x['candidate_recall'])

    logger.info("\n" + "="*100)
    logger.info(f"🏆 BEST STRATEGY: {best_result['strategy']}")
    logger.info("="*100)
    logger.info(f"   CANDIDATE Recall:  {baseline_recall:.4f} → {best_result['candidate_recall']:.4f}")
    logger.info(f"   Improvement:       {(best_result['candidate_recall'] - baseline_recall)*100:+.2f}%")
    logger.info(f"   Test Accuracy:     {best_result['test_metrics']['accuracy']:.4f}")
    logger.info(f"   Class Weights:     {best_result['class_weights']}")
    logger.info(f"   Training Time:     {best_result['training_time']:.2f}s")

    # JSON rapor
    report = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'user': 'sulegogh',
        'baseline_recall': float(baseline_recall),
        'best_strategy': best_result['strategy'],
        'best_recall': float(best_result['candidate_recall']),
        'improvement_pct': float((best_result['candidate_recall'] - baseline_recall) * 100),
        'strategies': []
    }

    for result in results:
        report['strategies'].append({
            'name': result['strategy'],
            'class_weights': result['class_weights'],
            'test_accuracy': float(result['test_metrics']['accuracy']),
            'test_f1': float(result['test_metrics']['f1_score']),
            'candidate_recall': float(result['candidate_recall']),
            'confirmed_recall': float(result['confirmed_recall']),
            'false_positive_recall': float(result['false_positive_recall']),
            'training_time': float(result['training_time']),
            'class_wise_metrics': result['test_metrics']['class_wise']
        })

    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n💾 JSON rapor kaydedildi: {report_path.name}")

    # Text summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CATBOOST v2 - CLASS WEIGHTS COMPARISON SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"User: sulegogh\n")
        f.write(f"Baseline Recall: {baseline_recall:.4f} ({baseline_recall*100:.2f}%)\n\n")

        f.write(header + "\n")
        f.write("-"*100 + "\n")

        for result in results:
            strategy = result['strategy']
            accuracy = result['test_metrics']['accuracy']
            f1 = result['test_metrics']['f1_score']
            can_recall = result['candidate_recall']
            con_recall = result['confirmed_recall']
            fp_recall = result['false_positive_recall']
            improvement = (can_recall - baseline_recall) * 100

            row = (
                f"{strategy:<28s} "
                f"{accuracy:<10.4f} "
                f"{f1:<8.4f} "
                f"{can_recall:<12.4f} "
                f"{con_recall:<12.4f} "
                f"{fp_recall:<11.4f} "
                f"{improvement:+11.2f}%\n"
            )
            f.write(row)

        f.write("\n" + "="*100 + "\n")
        f.write(f"BEST STRATEGY: {best_result['strategy']}\n")
        f.write("="*100 + "\n")
        f.write(f"CANDIDATE Recall: {baseline_recall:.4f} → {best_result['candidate_recall']:.4f}\n")
        f.write(f"Improvement: {(best_result['candidate_recall'] - baseline_recall)*100:+.2f}%\n")
        f.write(f"Test Accuracy: {best_result['test_metrics']['accuracy']:.4f}\n")
        f.write(f"Class Weights: {best_result['class_weights']}\n")

    logger.info(f"💾 Text summary kaydedildi: {summary_path.name}")

    return best_result


def save_models(results, output_dir):
    """Model'leri kaydet."""
    logger.info("\n" + "="*100)
    logger.info("💾 SAVING MODELS")
    logger.info("="*100)

    for result in results:
        strategy = result['strategy']
        model = result['model']

        # Dosya adı (temiz versiyon)
        filename = strategy.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') + '.pkl'
        filepath = output_dir / filename

        # Kaydet
        model.save(filepath)

        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   ✓ {strategy:<28s} → {filepath.name:<35s} ({file_size:.2f} MB)")

    logger.info(f"\n📁 Tüm modeller kaydedildi: {output_dir}")


def main():
    """Ana fonksiyon."""
    setup_logger()

    logger.info("="*100)
    logger.info("🔬 CATBOOST v2 - CLASS WEIGHTS TRAINING")
    logger.info("="*100)
    logger.info(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info(f"🎯 Hedef: CANDIDATE recall 59.93% → 70%+")
    logger.info(f"📊 Test Stratejileri: 5 (Baseline + 4 variants)")
    logger.info("="*100)

    # Output directory
    output_dir = PROJECT_ROOT / "models" / "v2_class_weights"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Veri yükle
    train_df, val_df, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)

    # Inverse frequency weights hesapla
    inverse_weights = compute_inverse_frequency_weights(y_train)

    # Stratejiler
    strategies = [
        ("1. Baseline (No Weights)", None),
        ("2. Balanced (Auto)", 'Balanced'),
        ("3. Manual Conservative", [2.5, 1.0, 0.7]),
        ("4. Manual Aggressive", [3.0, 1.0, 0.5]),
        ("5. Inverse Frequency", inverse_weights),
    ]

    # Her stratejiyi train et
    results = []

    for strategy_name, class_weights in strategies:
        logger.info("\n" + "="*100)

        try:
            result = train_strategy(
                strategy_name,
                class_weights,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )
            results.append(result)

        except Exception as e:
            logger.error(f"❌ {strategy_name} training failed: {e}")
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
    logger.info("\n" + "="*100)
    logger.info("✅ v2 CLASS WEIGHTS TRAINING TAMAMLANDI!")
    logger.info("="*100)
    logger.info(f"📊 Baseline Recall:      {results[0]['candidate_recall']:.4f} ({results[0]['candidate_recall']*100:.2f}%)")
    logger.info(f"📊 Best Recall:          {best_result['candidate_recall']:.4f} ({best_result['candidate_recall']*100:.2f}%)")
    logger.info(f"📊 Improvement:          {(best_result['candidate_recall'] - results[0]['candidate_recall'])*100:+.2f}%")
    logger.info(f"🏆 Best Strategy:        {best_result['strategy']}")
    logger.info(f"📁 Models saved:         {output_dir}")
    logger.info(f"📄 Reports:              comparison_report.json, comparison_summary.txt")
    logger.info("="*100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
