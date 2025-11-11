#!/usr/bin/env python3
"""
Veri preprocessing scripti.

Bu script, ham veriyi temizler, preprocess eder ve
train/val/test olarak böler.

Kullanım:
    python scripts/preprocess_data.py

    # Veya Makefile ile:
    make preprocess-data
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.core import DATA_PROCESSED, DATA_RAW
from src.data import clean_data, preprocess_data, save_splits


def main():
    """Ana preprocessing fonksiyonu."""

    logger.info("=" * 70)
    logger.info("🔧 KEPLER EXOPLANET - VERİ PREPROCESSING")
    logger.info("=" * 70)

    # Veriyi yükle
    data_file = DATA_RAW / "kepler_koi.csv"
    logger.info(f"📂 Dosya: {data_file}")

    df = pd.read_csv(data_file, low_memory=False)
    logger.info(f"✅ Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun")

    # Veriyi temizle
    logger.info("\n" + "=" * 70)
    df_clean = clean_data(df)

    # Veriyi preprocess et
    logger.info("\n" + "=" * 70)
    result = preprocess_data(df_clean, handle_missing=True, split=True)

    # Split'leri kaydet
    logger.info("\n" + "=" * 70)
    save_splits(result["train"], result["val"], result["test"], output_dir=str(DATA_PROCESSED))

    # Özet
    logger.info("\n" + "=" * 70)
    logger.info("📊 ÖZET")
    logger.info("=" * 70)
    logger.info(f"✅ Temizlenmiş veri: {len(df_clean):,} satır")
    logger.info(f"✅ Train: {len(result['train']):,} satır")
    logger.info(f"✅ Val: {len(result['val']):,} satır")
    logger.info(f"✅ Test: {len(result['test']):,} satır")
    logger.info(f"📂 Çıktı: {DATA_PROCESSED}")
    logger.info("=" * 70)
    logger.info("🎉 Preprocessing tamamlandı!")
    logger.info("\n🚀 Sonraki adım: Feature Engineering")

    return 0


if __name__ == "__main__":
    sys.exit(main())
