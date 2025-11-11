#!/usr/bin/env python3
"""
Model training scripti.

Bu script, CatBoost model'ini train eder ve değerlendirir.

Kullanım:
    python scripts/train_model.py

    # Özel parametrelerle:
    python scripts/train_model.py --iterations 500 --learning-rate 0.05

    # Veya Makefile ile:
    make train-model
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.core import RANDOM_STATE, TARGET_COLUMN
from src.evaluation import compare_metrics, evaluate_model, print_metrics
from src.models import CatBoostModel


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()  # Default handler'ı kaldır
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def parse_args():
    """Komut satırı argümanlarını parse et."""
    parser = argparse.ArgumentParser(description="Model training scripti")

    parser.add_argument("--input-dir", type=str, default="data/selected", help="Input dizini (default: data/selected)")

    parser.add_argument("--output-dir", type=str, default="models", help="Output dizini (default: models)")

    # CatBoost parametreleri
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost iterations (default: 1000)")

    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate (default: 0.03)")

    parser.add_argument("--depth", type=int, default=6, help="Tree depth (default: 6)")

    parser.add_argument("--early-stopping", type=int, default=100, help="Early stopping rounds (default: 100)")

    return parser.parse_args()


def load_data(input_dir: Path) -> tuple:
    """
    Train/val/test verilerini yükle.

    Args:
        input_dir: Input dizini

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("=" * 70)
    logger.info("VERİ YÜKLEME")
    logger.info("=" * 70)

    train_path = input_dir / "train_selected.csv"
    val_path = input_dir / "val_selected.csv"
    test_path = input_dir / "test_selected.csv"

    # Dosya kontrolü
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    # Yükle
    logger.info(f"📂 Input: {input_dir}")

    train_df = pd.read_csv(train_path)
    logger.info(f"  ✓ train_selected.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")

    val_df = pd.read_csv(val_path)
    logger.info(f"  ✓ val_selected.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")

    test_df = pd.read_csv(test_path)
    logger.info(f"  ✓ test_selected.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")

    logger.info("=" * 70)

    return train_df, val_df, test_df


def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Veriyi X, y olarak ayır ve temizle.

    Args:
        train_df: Train DataFrame
        val_df: Val DataFrame
        test_df: Test DataFrame

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("\n" + "=" * 70)
    logger.info("VERİ HAZIRLAMA")
    logger.info("=" * 70)

    # Target kontrolü
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target sütunu bulunamadı: {TARGET_COLUMN}")

    # X, y split
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # NaN temizleme (varsa)
    logger.info("NaN değerler kontrol ediliyor...")

    for name, df in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"  ⚠️  {name}: {nan_count} NaN bulundu, 0 ile doldruluyor...")
            df.fillna(0, inplace=True)
        else:
            logger.info(f"  ✓ {name}: NaN yok")

    # Class distribution
    logger.info("\nClass dağılımı (Train):")
    class_dist = y_train.value_counts()
    for cls, count in class_dist.items():
        logger.info(f"  {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")

    logger.info("=" * 70)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_catboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, **params
) -> CatBoostModel:
    """
    CatBoost model'ini train et.

    Args:
        X_train: Train features
        y_train: Train target
        X_val: Val features
        y_val: Val target
        **params: CatBoost parametreleri

    Returns:
        CatBoostModel: Train edilmiş model
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING - CATBOOST")
    logger.info("=" * 70)

    # Model oluştur
    model = CatBoostModel(**params)

    # Train
    logger.info(f"Training başlıyor...")
    logger.info(f"  Iterations: {params.get('iterations', 1000)}")
    logger.info(f"  Learning rate: {params.get('learning_rate', 0.03)}")
    logger.info(f"  Depth: {params.get('depth', 6)}")

    model.fit(X_train, y_train, X_val, y_val)

    logger.info("=" * 70)

    return model


def evaluate_all(
    model: CatBoostModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Model'i tüm dataset'lerde değerlendir.

    Args:
        model: Train edilmiş model
        X_train, y_train: Train data
        X_val, y_val: Val data
        X_test, y_test: Test data

    Returns:
        dict: Tüm metrikler
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    all_metrics = {}

    # Train
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "Train")
    print_metrics(train_metrics)
    all_metrics["train"] = train_metrics

    # Val
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba, "Val")
    print_metrics(val_metrics)
    all_metrics["val"] = val_metrics

    # Test
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(test_metrics)
    all_metrics["test"] = test_metrics

    # Karşılaştır
    compare_metrics(train_metrics, val_metrics, test_metrics)

    return all_metrics


def save_model_and_results(model: CatBoostModel, metrics: dict, output_dir: Path):
    """
    Model ve sonuçları kaydet.

    Args:
        model: Train edilmiş model
        metrics: Metrikler
        output_dir: Output dizini
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL VE SONUÇLARI KAYDETME")
    logger.info("=" * 70)

    # Output dizini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"catboost_model_{timestamp}.pkl"
    model.save(model_path)
    logger.info(f"  ✓ Model: {model_path}")

    # Metrics kaydet (JSON)
    import json

    # Convert numpy types to Python types
    metrics_serializable = {}
    for dataset, dataset_metrics in metrics.items():
        metrics_serializable[dataset] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in dataset_metrics.items()
            if k != "confusion_matrix"  # CM'i ayrı kaydederiz
        }

    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"  ✓ Metrics: {metrics_path}")

    # Feature importance kaydet
    importance_df = model.get_feature_importance()
    importance_path = output_dir / f"feature_importance_{timestamp}.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  ✓ Feature importance: {importance_path}")

    logger.info("=" * 70)


def main():
    """Ana fonksiyon."""
    # Setup
    setup_logger()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("🔧 KEPLER EXOPLANET - MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info(f"🎯 Model: CatBoost")
    logger.info("=" * 70)

    # Veriyi yükle
    input_dir = Path(args.input_dir)
    train_df, val_df, test_df = load_data(input_dir)

    # Veriyi hazırla
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)

    # CatBoost parametreleri
    catboost_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "early_stopping_rounds": args.early_stopping,
    }

    # Model train et
    model = train_catboost(X_train, y_train, X_val, y_val, **catboost_params)

    # Değerlendir
    metrics = evaluate_all(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Kaydet
    output_dir = Path(args.output_dir)
    save_model_and_results(model, metrics, output_dir)

    # Final özet
    logger.info("\n" + "=" * 70)
    logger.info("🎉 MODEL TRAINING TAMAMLANDI!")
    logger.info("=" * 70)
    logger.info(f"✅ Model: CatBoost")
    logger.info(f"✅ Train Accuracy: {metrics['train']['accuracy']:.4f}")
    logger.info(f"✅ Val Accuracy:   {metrics['val']['accuracy']:.4f}")
    logger.info(f"✅ Test Accuracy:  {metrics['test']['accuracy']:.4f}")
    logger.info(f"📂 Output: {output_dir}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
