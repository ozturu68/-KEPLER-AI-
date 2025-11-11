#!/usr/bin/env python3
"""
Model analiz scripti - Detaylı performans ve feature analizi.

Bu script mevcut CatBoost modelini detaylı şekilde analiz eder:
- Feature importance (top 20)
- Confusion matrix
- Class-wise metrics
- Error analysis
- Confidence analysis
- Overfitting kontrolü

Usage:
    python scripts/analyze_model.py

Author: sulegogh
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.models import CatBoostModel


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def load_latest_model():
    """En son kaydedilen model'i yükle."""
    models_dir = Path("models")
    model_files = list(models_dir.glob("catboost_model_*.pkl"))

    if not model_files:
        raise FileNotFoundError("❌ Model dosyası bulunamadı!")

    # En yeni model'i seç
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"📂 Model yükleniyor: {latest_model.name}")

    model = CatBoostModel()
    model.load(latest_model)

    return model, latest_model


def load_data():
    """Test verilerini yükle."""
    logger.info("📂 Test verileri yükleniyor...")

    train_df = pd.read_csv("data/selected/train_selected.csv")
    val_df = pd.read_csv("data/selected/val_selected.csv")
    test_df = pd.read_csv("data/selected/test_selected.csv")

    # NaN temizle
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"   ⚠️  {name}: {nan_count} NaN bulundu, 0 ile doldruluyor...")
            df.fillna(0, inplace=True)
        else:
            logger.info(f"   ✓ {name}: NaN yok")

    return train_df, val_df, test_df


def analyze_feature_importance(model, top_n=20):
    """Feature importance detaylı analiz."""
    logger.info("\n" + "=" * 70)
    logger.info("📊 FEATURE IMPORTANCE ANALİZİ")
    logger.info("=" * 70)

    importance_df = model.get_feature_importance()

    # Top N features
    logger.info(f"\n🏆 Top {top_n} En Önemli Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"   {idx+1:2d}. {row['feature']:35s} → {row['importance']:.6f}")

    # İstatistikler
    logger.info(f"\n📈 Kümülatif Katkı:")
    logger.info(f"   Top 10: {importance_df.head(10)['importance'].sum():.2%}")
    logger.info(f"   Top 20: {importance_df.head(20)['importance'].sum():.2%}")
    logger.info(f"   Top 30: {importance_df.head(30)['importance'].sum():.2%}")

    return importance_df


def analyze_predictions(model, X, y, dataset_name="Dataset"):
    """Tahmin analizi - Confusion Matrix ve Class-wise Metrics."""
    logger.info(f"\n" + "=" * 70)
    logger.info(f"🔍 {dataset_name.upper()} TAHMİN ANALİZİ")
    logger.info("=" * 70)

    # Tahminler
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    labels = sorted(y.unique())

    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"\n{'':20s} " + " ".join([f"{lbl:>15s}" for lbl in labels]))
    logger.info("-" * 70)
    for i, true_label in enumerate(labels):
        row_str = f"{true_label:20s} "
        row_str += " ".join([f"{cm[i][j]:>15d}" for j in range(len(labels))])
        logger.info(row_str)

    # Class-wise metrics
    logger.info(f"\n📈 Class-wise Metrics:")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    for cls in labels:
        if cls in report:
            metrics = report[cls]
            logger.info(f"\n   📌 {cls}:")
            logger.info(f"      Precision: {metrics['precision']:.4f}")
            logger.info(f"      Recall:    {metrics['recall']:.4f}")
            logger.info(f"      F1-Score:  {metrics['f1-score']:.4f}")
            logger.info(f"      Support:   {int(metrics['support'])}")

    # Confidence Analizi
    logger.info(f"\n🎯 Confidence Analizi:")
    max_proba = y_pred_proba.max(axis=1)
    logger.info(f"   Ortalama confidence: {max_proba.mean():.4f}")
    logger.info(f"   Median confidence:   {np.median(max_proba):.4f}")
    logger.info(f"   Min confidence:      {max_proba.min():.4f}")
    logger.info(f"   Max confidence:      {max_proba.max():.4f}")

    # Düşük confidence tahminler
    low_conf_threshold = 0.6
    low_conf_count = (max_proba < low_conf_threshold).sum()
    logger.info(
        f"   Düşük confidence (<{low_conf_threshold}): {low_conf_count} ({low_conf_count/len(max_proba)*100:.2f}%)"
    )

    return y_pred, y_pred_proba, cm


def analyze_errors(y_true, y_pred, top_n=10):
    """
    Hata analizi - yanlış tahminleri detaylı incele.

    Args:
        y_true: Gerçek labels (pandas Series veya numpy array)
        y_pred: Tahmin edilen labels (numpy array)
        top_n: En sık hata yapılan kaç class çiftini göster

    Returns:
        None
    """
    logger.info(f"\n" + "=" * 70)
    logger.info(f"❌ HATA ANALİZİ")
    logger.info("=" * 70)

    # Type ve shape kontrolü
    # y_pred numpy array ise ve 2D ise flatten et
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim == 2:
            logger.debug(f"   y_pred 2D array tespit edildi: {y_pred.shape} → flatten")
            y_pred = y_pred.ravel()
        elif y_pred.ndim > 2:
            raise ValueError(f"y_pred çok fazla boyutlu: {y_pred.shape}")

    # y_true pandas Series ise numpy array'e çevir
    if hasattr(y_true, "values"):
        y_true_arr = y_true.values
    else:
        y_true_arr = np.array(y_true)

    # Eğer y_true_arr da 2D ise flatten et
    if y_true_arr.ndim == 2:
        logger.debug(f"   y_true 2D array tespit edildi: {y_true_arr.shape} → flatten")
        y_true_arr = y_true_arr.ravel()

    # Shape kontrolü
    if y_true_arr.shape != y_pred.shape:
        raise ValueError(
            f"y_true ve y_pred shape'leri uyuşmuyor! " f"y_true: {y_true_arr.shape}, y_pred: {y_pred.shape}"
        )

    logger.debug(f"   y_true shape: {y_true_arr.shape}")
    logger.debug(f"   y_pred shape: {y_pred.shape}")

    # Yanlış tahminler
    errors = y_true_arr != y_pred
    error_count = errors.sum()
    total_count = len(y_true_arr)

    logger.info(f"\n📉 Hata İstatistikleri:")
    logger.info(f"   Toplam örnek:  {total_count:,}")
    logger.info(f"   Doğru tahmin:  {total_count - error_count:,} ({(total_count-error_count)/total_count*100:.2f}%)")
    logger.info(f"   Yanlış tahmin: {error_count:,} ({error_count/total_count*100:.2f}%)")

    if error_count == 0:
        logger.info("\n   ✅ Hiç hata yok! (Mükemmel tahmin)")
        return

    # Hata dağılımı DataFrame
    error_indices = np.where(errors)[0]
    error_df = pd.DataFrame({"true": y_true_arr[errors], "pred": y_pred[errors], "index": error_indices})

    # En sık karıştırılan class çiftleri
    logger.info(f"\n📊 En Sık Karıştırılan Class Çiftleri (True → Predicted):")
    error_counts = error_df.groupby(["true", "pred"]).size().sort_values(ascending=False)

    logger.info(f"\n   {'#':<4s} {'True Label':<20s} {'→':^3s} {'Predicted Label':<20s} {'Count':>7s} {'Percent':>8s}")
    logger.info(f"   {'-'*70}")

    for i, ((true_cls, pred_cls), count) in enumerate(error_counts.head(top_n).items(), 1):
        pct = count / error_count * 100
        logger.info(f"   {i:<4d} {true_cls:<20s} {'→':^3s} {pred_cls:<20s} {count:>7,d} {pct:>7.1f}%")

    # Class-wise hata analizi
    logger.info(f"\n📈 Class-wise Hata Dağılımı:")
    unique_classes = np.unique(y_true_arr)

    logger.info(f"\n   {'Class':<20s} {'Total':>8s} {'Errors':>8s} {'Error Rate':>12s}")
    logger.info(f"   {'-'*55}")

    for cls in sorted(unique_classes):
        cls_mask = y_true_arr == cls
        cls_total = cls_mask.sum()
        cls_errors = (errors & cls_mask).sum()
        cls_error_rate = cls_errors / cls_total * 100 if cls_total > 0 else 0

        logger.info(f"   {cls:<20s} {cls_total:>8,d} {cls_errors:>8,d} {cls_error_rate:>11.2f}%")

    # En problemli örnekler (hata oranı en yüksek class)
    logger.info(f"\n🎯 En Problemli Class:")
    class_error_rates = []
    for cls in unique_classes:
        cls_mask = y_true_arr == cls
        cls_total = cls_mask.sum()
        cls_errors = (errors & cls_mask).sum()
        cls_error_rate = cls_errors / cls_total if cls_total > 0 else 0
        class_error_rates.append((cls, cls_error_rate, cls_errors, cls_total))

    # En yüksek hata oranına göre sırala
    class_error_rates.sort(key=lambda x: x[1], reverse=True)

    worst_class, worst_rate, worst_errors, worst_total = class_error_rates[0]
    logger.info(f"   Class: {worst_class}")
    logger.info(f"   Hata oranı: {worst_rate*100:.2f}%")
    logger.info(f"   Yanlış tahmin: {worst_errors}/{worst_total}")

    # Bu class için en sık karıştırılan hedef
    worst_class_errors = error_df[error_df["true"] == worst_class]
    if len(worst_class_errors) > 0:
        most_confused = worst_class_errors["pred"].value_counts().iloc[0]
        most_confused_class = worst_class_errors["pred"].value_counts().index[0]
        logger.info(f"   En çok karıştırılan: {most_confused_class} ({most_confused} kez)")


def compare_datasets(train_metrics, val_metrics, test_metrics):
    """Dataset'leri karşılaştır ve overfitting analizi."""
    logger.info(f"\n" + "=" * 70)
    logger.info(f"📊 DATASET KARŞILAŞTIRMASI")
    logger.info("=" * 70)

    # Tablo
    logger.info(f"\n{'Metric':<15s} {'Train':>10s} {'Val':>10s} {'Test':>10s}")
    logger.info("-" * 50)
    logger.info(
        f"{'Accuracy':<15s} {train_metrics['acc']:>10.4f} {val_metrics['acc']:>10.4f} {test_metrics['acc']:>10.4f}"
    )
    logger.info(
        f"{'F1 Score':<15s} {train_metrics['f1']:>10.4f} {val_metrics['f1']:>10.4f} {test_metrics['f1']:>10.4f}"
    )

    # Overfitting analizi
    train_val_diff = train_metrics["acc"] - val_metrics["acc"]
    val_test_diff = val_metrics["acc"] - test_metrics["acc"]

    logger.info(f"\n🔍 Overfitting Analizi:")
    logger.info(f"   Train-Val fark:  {train_val_diff:+.4f} ({train_val_diff*100:+.2f}%)")
    logger.info(f"   Val-Test fark:   {val_test_diff:+.4f} ({val_test_diff*100:+.2f}%)")

    if train_val_diff < 0.05:
        logger.info(f"   ✅ Model dengeli (fark < 5%)")
    elif train_val_diff < 0.10:
        logger.warning(f"   ⚠️  Hafif overfitting (fark 5-10%)")
    else:
        logger.error(f"   ❌ Ciddi overfitting (fark > 10%)")


def main():
    """Ana fonksiyon."""
    setup_logger()

    logger.info("=" * 70)
    logger.info("🔬 CATBOOST MODEL DETAYLI ANALİZ")
    logger.info("=" * 70)
    logger.info(f"📅 Tarih: 2025-11-11 18:19:08 UTC")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info("=" * 70)

    # 1. Model yükle
    model, model_path = load_latest_model()
    logger.info(f"✅ Model yüklendi: {model}")

    # 2. Veri yükle
    train_df, val_df, test_df = load_data()

    X_train = train_df.drop(columns=["koi_disposition"])
    y_train = train_df["koi_disposition"]

    X_val = val_df.drop(columns=["koi_disposition"])
    y_val = val_df["koi_disposition"]

    X_test = test_df.drop(columns=["koi_disposition"])
    y_test = test_df["koi_disposition"]

    logger.info(f"\n✅ Veri yüklendi:")
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Val:   {len(X_val):,} samples")
    logger.info(f"   Test:  {len(X_test):,} samples")

    # 3. Feature Importance Analizi
    importance_df = analyze_feature_importance(model, top_n=20)

    # 4. Test Set Detaylı Analizi
    y_test_pred, y_test_proba, test_cm = analyze_predictions(model, X_test, y_test, "Test Set")

    # 5. Validation Set Analizi
    y_val_pred, y_val_proba, val_cm = analyze_predictions(model, X_val, y_val, "Validation Set")

    # 6. Hata Analizi (Test Set)
    analyze_errors(y_test, y_test_pred, top_n=10)

    # 7. Dataset Karşılaştırması
    y_train_pred = model.predict(X_train)

    train_metrics = {
        "acc": accuracy_score(y_train, y_train_pred),
        "f1": f1_score(y_train, y_train_pred, average="weighted"),
    }
    val_metrics = {"acc": accuracy_score(y_val, y_val_pred), "f1": f1_score(y_val, y_val_pred, average="weighted")}
    test_metrics = {"acc": accuracy_score(y_test, y_test_pred), "f1": f1_score(y_test, y_test_pred, average="weighted")}

    compare_datasets(train_metrics, val_metrics, test_metrics)

    # 8. Final Özet
    logger.info(f"\n" + "=" * 70)
    logger.info(f"✅ ANALİZ TAMAMLANDI!")
    logger.info("=" * 70)
    logger.info(f"📊 Test Accuracy:  {test_metrics['acc']:.4f} ({test_metrics['acc']*100:.2f}%)")
    logger.info(f"📊 Test F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"📂 Model: {model_path.name}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
