#!/usr/bin/env python3
"""
Baseline Reference Creator - CatBoost Model v1.

Bu script mevcut CatBoost modelini "v1_baseline" olarak referans alır.
Tüm gelecek versiyonlar bu baseline ile karşılaştırılacak.

Features:
- Mevcut model'i baseline klasörüne kopyala
- Metrics'leri JSON formatında kaydet
- Feature importance'ı CSV olarak kaydet
- Detailed README oluştur
- Git commit için hazır hale getir

Usage:
    python scripts/create_baseline_reference.py

Output:
    models/v1_baseline/
    ├── catboost_model_baseline.pkl
    ├── metrics_baseline.json
    ├── feature_importance_baseline.csv
    ├── confusion_matrices_baseline.json
    └── README.md

Author: sulegogh
Date: 2025-11-11 18:36:24 UTC
Version: 1.0
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
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

from src.models import CatBoostModel


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Log file
    log_file = PROJECT_ROOT / "logs" / f"baseline_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="DEBUG")


def find_latest_model():
    """En son kaydedilen CatBoost modelini bul."""
    models_dir = PROJECT_ROOT / "models"
    model_files = list(models_dir.glob("catboost_model_*.pkl"))

    if not model_files:
        raise FileNotFoundError("❌ CatBoost model dosyası bulunamadı!")

    # En yeni model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"📂 En son model bulundu: {latest_model.name}")
    logger.info(f"   Tarih: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")
    logger.info(f"   Boyut: {latest_model.stat().st_size / (1024*1024):.2f} MB")

    return latest_model


def load_data():
    """Train/Val/Test verilerini yükle."""
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


def calculate_metrics(model, X, y, dataset_name="Dataset"):
    """Detaylı metrikleri hesapla."""
    logger.info(f"📊 {dataset_name} metrikleri hesaplanıyor...")

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    # ROC AUC (OvR - One vs Rest)
    try:
        roc_auc = roc_auc_score(y, y_pred_proba, average="weighted", multi_class="ovr")
    except Exception as e:
        logger.warning(f"   ROC AUC hesaplanamadı: {e}")
        roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Class-wise metrics
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Confidence stats
    max_proba = y_pred_proba.max(axis=1)

    metrics = {
        "dataset": dataset_name,
        "samples": len(y),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "class_wise": {},
        "confidence": {
            "mean": float(max_proba.mean()),
            "median": float(np.median(max_proba)),
            "min": float(max_proba.min()),
            "max": float(max_proba.max()),
            "std": float(max_proba.std()),
            "low_confidence_count": int((max_proba < 0.6).sum()),
            "low_confidence_pct": float((max_proba < 0.6).sum() / len(max_proba) * 100),
        },
    }

    # Class-wise metrics
    for cls in sorted(y.unique()):
        if cls in report:
            metrics["class_wise"][cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1_score": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

    logger.info(f"   ✓ {dataset_name}: Acc={accuracy:.4f}, F1={f1:.4f}")

    return metrics


def create_baseline_directory():
    """Baseline dizinini oluştur."""
    baseline_dir = PROJECT_ROOT / "models" / "v1_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Baseline dizini oluşturuldu: {baseline_dir}")

    return baseline_dir


def copy_model(source_path, baseline_dir):
    """Model'i baseline dizinine kopyala."""
    dest_path = baseline_dir / "catboost_model_baseline.pkl"

    shutil.copy2(source_path, dest_path)

    logger.info(f"📦 Model kopyalandı:")
    logger.info(f"   Kaynak: {source_path.name}")
    logger.info(f"   Hedef: {dest_path}")

    return dest_path


def save_metrics(metrics_dict, baseline_dir):
    """Metrics'leri JSON olarak kaydet."""
    metrics_path = baseline_dir / "metrics_baseline.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"💾 Metrics kaydedildi: {metrics_path.name}")

    return metrics_path


def save_feature_importance(model, baseline_dir):
    """Feature importance'ı CSV olarak kaydet."""
    importance_path = baseline_dir / "feature_importance_baseline.csv"

    importance_df = model.get_feature_importance()
    importance_df.to_csv(importance_path, index=False)

    logger.info(f"💾 Feature importance kaydedildi: {importance_path.name}")
    logger.info(f"   Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.2f})")

    return importance_path


def create_readme(metrics_dict, model_path, baseline_dir):
    """
    Detaylı README.md oluştur.

    Args:
        metrics_dict: Tüm metrikleri içeren dict
        model_path: Original model dosya yolu
        baseline_dir: Baseline dizini

    Returns:
        Path: README.md dosya yolu
    """
    readme_path = baseline_dir / "README.md"

    # Metrics'leri çıkar
    train_metrics = metrics_dict["train"]
    val_metrics = metrics_dict["validation"]
    test_metrics = metrics_dict["test"]

    # Overfitting analizi
    train_val_diff = train_metrics["accuracy"] - val_metrics["accuracy"]
    val_test_diff = val_metrics["accuracy"] - test_metrics["accuracy"]

    # CANDIDATE recall (önceden hesapla)
    candidate_recall = test_metrics["class_wise"].get("CANDIDATE", {}).get("recall", 0)

    # Confusion matrix'i string'e çevir
    cm_array = np.array(test_metrics["confusion_matrix"])
    cm_str = str(cm_array)

    # ============================================================
    # README İÇERİĞİ - f-string ile
    # ============================================================

    readme_content = f"""# CatBoost Model v1 - Baseline Reference

## 📋 Genel Bilgiler

**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Kullanıcı:** sulegogh
**Model Type:** CatBoost Classifier
**Version:** 1.0 (Baseline)
**Original Model:** `{model_path.name}`
**Training Time:** {metrics_dict.get('train', {}).get('training_time', 'N/A')}
**Features:** {len(test_metrics.get('feature_names', []))} features

---

## 📊 Performans Metrikleri

### Genel Karşılaştırma

| Dataset | Samples | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------|---------|----------|-----------|--------|----------|---------|
| **Train** | {train_metrics['samples']:,} | {train_metrics['accuracy']:.4f} | {train_metrics['precision']:.4f} | {train_metrics['recall']:.4f} | {train_metrics['f1_score']:.4f} | {train_metrics['roc_auc']:.4f} |
| **Val** | {val_metrics['samples']:,} | {val_metrics['accuracy']:.4f} | {val_metrics['precision']:.4f} | {val_metrics['recall']:.4f} | {val_metrics['f1_score']:.4f} | {val_metrics['roc_auc']:.4f} |
| **Test** | {test_metrics['samples']:,} | {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1_score']:.4f} | {test_metrics['roc_auc']:.4f} |

### Overfitting Analizi

- **Train-Val Gap:** {train_val_diff:+.4f} ({train_val_diff*100:+.2f}%)
- **Val-Test Gap:** {val_test_diff:+.4f} ({val_test_diff*100:+.2f}%)

**Status:** """

    # Overfitting durumu
    if train_val_diff < 0.05:
        readme_content += "✅ Model dengeli (< 5%)\n"
    elif train_val_diff < 0.10:
        readme_content += "⚠️ Hafif overfitting (5-10%)\n"
    else:
        readme_content += "❌ Ciddi overfitting (> 10%)\n"

    readme_content += "\n---\n\n"

    # Class-wise metrics
    readme_content += """### Class-wise Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""

    for cls in sorted(test_metrics["class_wise"].keys()):
        metrics = test_metrics["class_wise"][cls]
        readme_content += f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['support']:,} |\n"

    # Confusion Matrix
    readme_content += f"""
### Confusion Matrix (Test Set)

**Sıralama:** {sorted(test_metrics['class_wise'].keys())}

---

## 🎯 Confidence Analizi (Test Set)

- **Mean Confidence:** {test_metrics['confidence']['mean']:.4f}
- **Median Confidence:** {test_metrics['confidence']['median']:.4f}
- **Min Confidence:** {test_metrics['confidence']['min']:.4f}
- **Max Confidence:** {test_metrics['confidence']['max']:.4f}
- **Std Confidence:** {test_metrics['confidence']['std']:.4f}
- **Low Confidence (<0.6):** {test_metrics['confidence']['low_confidence_count']:,} ({test_metrics['confidence']['low_confidence_pct']:.2f}%)

**Değerlendirme:**
"""

    # Confidence değerlendirmesi
    if test_metrics["confidence"]["low_confidence_pct"] < 5:
        readme_content += "- ✅ Çok iyi (< 5% düşük confidence)\n"
    elif test_metrics["confidence"]["low_confidence_pct"] < 10:
        readme_content += "- ⚠️ Kabul edilebilir (5-10% düşük confidence)\n"
    else:
        readme_content += "- ❌ İyileştirme gerekli (> 10% düşük confidence)\n"

    readme_content += "\n---\n\n## 🔍 Tespit Edilen Sorunlar\n\n"

    # Sorunları tespit et
    issues_found = False

    # 1. CANDIDATE Class
    if candidate_recall < 0.65:
        issues_found = True
        readme_content += f"""### ⚠️ CANDIDATE Class Zayıf

- **Recall:** {candidate_recall:.4f} ({candidate_recall*100:.2f}%)
- **Durum:** KRITIK (< 65%)
- **Öneri:**
  - Class weights ayarla (`class_weights=[2.5, 1.0, 0.7]`)
  - SMOTE oversampling kullan
  - Threshold tuning uygula

"""

    # 2. Overfitting
    if train_val_diff > 0.07:
        issues_found = True
        readme_content += f"""### ⚠️ Overfitting Tespit Edildi

- **Train-Val Gap:** {train_val_diff*100:.2f}%
- **Durum:** HAFİF OVERFITTING (> 7%)
- **Öneri:**
  - L2 regularization artır (`l2_leaf_reg`: 3 → 5)
  - Tree depth azalt (`depth`: 6 → 5)
  - Learning rate düşür (`learning_rate`: 0.03 → 0.02)
  - Dropout ekle (`subsample=0.8`)

"""

    # 3. Low Confidence
    if test_metrics["confidence"]["low_confidence_pct"] > 10:
        issues_found = True
        readme_content += f"""### ⚠️ Yüksek Belirsizlik

- **Low Confidence:** {test_metrics['confidence']['low_confidence_pct']:.2f}%
- **Durum:** YÜKSEK (> 10%)
- **Öneri:**
  - Threshold tuning uygula
  - Daha fazla training data
  - Feature engineering v2

"""

    # 4. Class-specific issues
    for cls, metrics in test_metrics["class_wise"].items():
        if metrics["recall"] < 0.70 or metrics["precision"] < 0.70:
            if not issues_found:
                issues_found = True
            readme_content += f"""### ⚠️ {cls} Class Problemi

- **Precision:** {metrics['precision']:.4f}
- **Recall:** {metrics['recall']:.4f}
- **F1-Score:** {metrics['f1_score']:.4f}
- **Durum:** İyileştirme gerekli
- **Öneri:** Class-specific stratejiler uygula

"""

    if not issues_found:
        readme_content += "✅ **Kritik sorun tespit edilmedi!** Model performansı kabul edilebilir seviyede.\n\n"

    readme_content += """---

## 📁 Dosyalar

- `catboost_model_baseline.pkl` - Model binary (pickle format)
- `metrics_baseline.json` - Detaylı metrikler (JSON)
- `feature_importance_baseline.csv` - Feature importance sıralaması
- `README.md` - Bu dosya

---

## 🚀 Gelecek Versiyonlar İçin Hedefler

"""

    # Hedefleri dinamik oluştur
    readme_content += f"""1. **CANDIDATE Recall İyileştirme**
   - Mevcut: {candidate_recall:.2%}
   - Hedef: 75%+
   - Yöntem: Class weights + SMOTE

2. **Overfitting Azaltma**
   - Mevcut: {train_val_diff*100:.1f}%
   - Hedef: < 5%
   - Yöntem: Regularization + Hyperparameter tuning

3. **Low Confidence Azaltma**
   - Mevcut: {test_metrics['confidence']['low_confidence_pct']:.1f}%
   - Hedef: < 8%
   - Yöntem: Threshold tuning + Data augmentation

4. **Test Accuracy İyileştirme**
   - Mevcut: {test_metrics['accuracy']:.2%}
   - Hedef: 90%+
   - Yöntem: Ensemble models + Feature engineering v2

---

## 📊 Sonraki Versiyonlar

### v2: Class Weights
- `train_model_v2_class_weights.py`
- Hedef: CANDIDATE recall 70%+

### v3: SMOTE Oversampling
- `train_model_v3_smote.py`
- Hedef: CANDIDATE recall 75%+

### v4: Hyperparameter Tuning
- `train_model_v4_hypertuned.py`
- Hedef: Overfitting < 5%

### v5: Optuna AutoML
- `train_model_v5_optuna.py`
- Hedef: Optimal parameters

### v6: Threshold Tuning
- Optimal decision boundaries
- Hedef: +2% accuracy

### v7: Feature Engineering v2
- Yeni features
- koi_score bağımlılığını azalt

### v8: Cross-Validation
- 5-fold CV
- Model stability

---

## 📝 Notlar

- Bu baseline model tüm gelecek versiyonlar için **referans noktası**dır
- Her yeni versiyon bu metriklerle karşılaştırılacaktır
- Model production'a taşınmadan önce **v5 (Optuna)** sonuçları beklenmelidir

### Baseline Tarihi
**{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}**

### Geliştiren
**sulegogh**

---

## 🔗 İlgili Dosyalar

- Training Script: `scripts/train_model.py`
- Analysis Script: `scripts/analyze_model.py`
- Data: `data/selected/`
- Logs: `logs/`

---

*Generated by: `create_baseline_reference.py`*
"""

    # README'yi kaydet
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"📄 README oluşturuldu: {readme_path.name}")
    logger.info(f"   Boyut: {readme_path.stat().st_size / 1024:.2f} KB")

    return readme_path


def main():
    """Ana fonksiyon."""
    setup_logger()

    logger.info("=" * 70)
    logger.info("🏗️  BASELINE REFERENCE OLUŞTURMA")
    logger.info("=" * 70)
    logger.info(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info("=" * 70)

    try:
        # 1. En son model'i bul
        latest_model_path = find_latest_model()

        # 2. Model'i yükle
        logger.info("\n📦 Model yükleniyor...")
        model = CatBoostModel()
        model.load(latest_model_path)
        logger.info(f"✅ Model yüklendi: {model}")

        # 3. Veri yükle
        train_df, val_df, test_df = load_data()

        X_train = train_df.drop(columns=["koi_disposition"])
        y_train = train_df["koi_disposition"]

        X_val = val_df.drop(columns=["koi_disposition"])
        y_val = val_df["koi_disposition"]

        X_test = test_df.drop(columns=["koi_disposition"])
        y_test = test_df["koi_disposition"]

        # 4. Metrikleri hesapla
        logger.info("\n" + "=" * 70)
        logger.info("📊 METRİK HESAPLAMA")
        logger.info("=" * 70)

        train_metrics = calculate_metrics(model, X_train, y_train, "Train")
        val_metrics = calculate_metrics(model, X_val, y_val, "Validation")
        test_metrics = calculate_metrics(model, X_test, y_test, "Test")

        metrics_dict = {
            "version": "v1_baseline",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "user": "sulegogh",
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }

        # 5. Baseline dizini oluştur
        logger.info("\n" + "=" * 70)
        logger.info("📁 BASELINE DİZİNİ OLUŞTURMA")
        logger.info("=" * 70)

        baseline_dir = create_baseline_directory()

        # 6. Model'i kopyala
        baseline_model_path = copy_model(latest_model_path, baseline_dir)

        # 7. Metrics'leri kaydet
        metrics_path = save_metrics(metrics_dict, baseline_dir)

        # 8. Feature importance'ı kaydet
        importance_path = save_feature_importance(model, baseline_dir)

        # 9. README oluştur
        readme_path = create_readme(metrics_dict, latest_model_path, baseline_dir)

        # 10. Özet
        logger.info("\n" + "=" * 70)
        logger.info("✅ BASELINE REFERENCE OLUŞTURULDU!")
        logger.info("=" * 70)
        logger.info(f"📂 Dizin: {baseline_dir}")
        logger.info(f"📦 Model: {baseline_model_path.name}")
        logger.info(f"📊 Metrics: {metrics_path.name}")
        logger.info(f"📈 Importance: {importance_path.name}")
        logger.info(f"📄 README: {readme_path.name}")
        logger.info("=" * 70)
        logger.info(f"\n📊 Test Performansı:")
        logger.info(f"   Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        logger.info(f"   F1 Score:  {test_metrics['f1_score']:.4f}")
        logger.info(f"   ROC AUC:   {test_metrics['roc_auc']:.4f}")
        logger.info("=" * 70)
        logger.info("\n🎯 Sonraki Adım: train_model_v2_class_weights.py")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Hata oluştu: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
