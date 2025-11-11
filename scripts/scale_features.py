#!/usr/bin/env python3
"""
Feature scaling scripti.

Bu script, preprocessed veriyi (train/val/test) scale eder ve
data/scaled/ dizinine kaydeder.

Kullanım:
    python scripts/scale_features.py

    # Farklı scaler ile:
    python scripts/scale_features.py --method standard

    # Veya Makefile ile:
    make scale-features
"""

import argparse
import sys
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.core import DATA_PROCESSED, TARGET_COLUMN
from src.features.scalers import FeatureScaler, scale_train_val_test


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
    parser = argparse.ArgumentParser(description="Feature scaling scripti")

    parser.add_argument(
        "--method",
        type=str,
        default="robust",
        choices=["standard", "robust", "minmax"],
        help="Scaling yöntemi (default: robust)",
    )

    parser.add_argument(
        "--input-dir", type=str, default=str(DATA_PROCESSED), help="Input dizini (default: data/processed)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED.parent / "scaled"),
        help="Output dizini (default: data/scaled)",
    )

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

    train_path = input_dir / "train.csv"
    val_path = input_dir / "val.csv"
    test_path = input_dir / "test.csv"

    # Dosya kontrolü
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    # Yükle
    logger.info(f"📂 Input: {input_dir}")

    train_df = pd.read_csv(train_path, low_memory=False)
    logger.info(f"  ✓ train.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")

    val_df = pd.read_csv(val_path, low_memory=False)
    logger.info(f"  ✓ val.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")

    test_df = pd.read_csv(test_path, low_memory=False)
    logger.info(f"  ✓ test.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")

    logger.info("=" * 70)

    return train_df, val_df, test_df


def show_scaling_comparison(original_df: pd.DataFrame, scaled_df: pd.DataFrame, sample_features: list):
    """
    Scaling öncesi ve sonrası karşılaştırma göster.

    Args:
        original_df: Orijinal DataFrame
        scaled_df: Scale edilmiş DataFrame
        sample_features: Örnek feature'lar
    """
    logger.info("\n" + "=" * 70)
    logger.info("SCALING KARŞILAŞTIRMA (İlk 3 Feature)")
    logger.info("=" * 70)

    comparison_data = []

    for feat in sample_features[:3]:
        if feat in original_df.columns and feat in scaled_df.columns:
            comparison_data.append(
                {
                    "Feature": feat,
                    "Original Min": f"{original_df[feat].min():.4f}",
                    "Original Max": f"{original_df[feat].max():.4f}",
                    "Original Mean": f"{original_df[feat].mean():.4f}",
                    "Scaled Min": f"{scaled_df[feat].min():.4f}",
                    "Scaled Max": f"{scaled_df[feat].max():.4f}",
                    "Scaled Mean": f"{scaled_df[feat].mean():.4f}",
                }
            )

    if comparison_data:
        import pandas as pd

        comparison_df = pd.DataFrame(comparison_data)

        # Güzel tablo formatı
        for _, row in comparison_df.iterrows():
            logger.info(f"\n📊 {row['Feature']}:")
            logger.info(
                f"   Original: min={row['Original Min']}, max={row['Original Max']}, mean={row['Original Mean']}"
            )
            logger.info(f"   Scaled:   min={row['Scaled Min']}, max={row['Scaled Max']}, mean={row['Scaled Mean']}")

    logger.info("\n" + "=" * 70)


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """
    Scaled veriyi kaydet.

    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output dizini
    """
    logger.info("\n" + "=" * 70)
    logger.info("VERİ KAYDETME")
    logger.info("=" * 70)

    # Output dizini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Output: {output_dir}")

    # Kaydet
    train_path = output_dir / "train_scaled.csv"
    train_df.to_csv(train_path, index=False)
    logger.info(f"  ✓ train_scaled.csv: {len(train_df):,} satır")

    val_path = output_dir / "val_scaled.csv"
    val_df.to_csv(val_path, index=False)
    logger.info(f"  ✓ val_scaled.csv: {len(val_df):,} satır")

    test_path = output_dir / "test_scaled.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"  ✓ test_scaled.csv: {len(test_df):,} satır")

    # Boyut kontrolü
    total_size = sum([p.stat().st_size for p in [train_path, val_path, test_path]])
    logger.info(f"  📏 Toplam boyut: {total_size / (1024*1024):.2f} MB")

    logger.info("=" * 70)


def main():
    """Ana fonksiyon."""
    # Setup
    setup_logger()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("🔧 KEPLER EXOPLANET - FEATURE SCALING")
    logger.info("=" * 70)
    logger.info(f"📅 Tarih: 2025-11-11 16:32:09 UTC")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info(f"🎯 Scaling Method: {args.method}")
    logger.info("=" * 70)

    # Veriyi yükle
    input_dir = Path(args.input_dir)
    train_df, val_df, test_df = load_data(input_dir)

    # Orijinal veriyi sakla (karşılaştırma için)
    train_original = train_df.copy()

    # Scale et
    logger.info("\n" + "=" * 70)
    logger.info(f"FEATURE SCALING (method={args.method})")
    logger.info("=" * 70)

    train_scaled, val_scaled, test_scaled, scaler = scale_train_val_test(
        train_df=train_df, val_df=val_df, test_df=test_df, method=args.method, exclude_cols=[TARGET_COLUMN]
    )

    # Karşılaştırma göster
    numerical_features = [col for col in train_df.select_dtypes(include=["number"]).columns if col != TARGET_COLUMN]
    show_scaling_comparison(train_original, train_scaled, numerical_features)

    # Kaydet
    output_dir = Path(args.output_dir)
    save_data(train_scaled, val_scaled, test_scaled, output_dir)

    # Özet
    logger.info("\n" + "=" * 70)
    logger.info("📊 ÖZET")
    logger.info("=" * 70)
    logger.info(f"✅ Scaling method: {args.method}")
    logger.info(f"✅ Scaled features: {len(scaler.numerical_features)}")
    logger.info(f"✅ Train: {len(train_scaled):,} satır")
    logger.info(f"✅ Val: {len(val_scaled):,} satır")
    logger.info(f"✅ Test: {len(test_scaled):,} satır")
    logger.info(f"📂 Output: {output_dir}")
    logger.info("=" * 70)
    logger.info("🎉 Feature scaling tamamlandı!")
    logger.info("\n🚀 Sonraki adım: Feature Engineering (polynomial features)")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
