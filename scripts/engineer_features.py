#!/usr/bin/env python3
"""
Feature engineering scripti.

Bu script, scaled veriyi alır, yeni feature'lar oluşturur ve
data/engineered/ dizinine kaydeder.

Kullanım:
    python scripts/engineer_features.py
    
    # Polynomial features ile:
    python scripts/engineer_features.py --polynomial
    
    # Veya Makefile ile:
    make engineer-features
"""

import sys
import argparse
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.core import TARGET_COLUMN
from src.features.engineering import ExoplanetFeatureEngineer, engineer_train_val_test


def setup_logger():
    """Logger'ı yapılandır."""
    logger.remove()  # Default handler'ı kaldır
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def parse_args():
    """Komut satırı argümanlarını parse et."""
    parser = argparse.ArgumentParser(description="Feature engineering scripti")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/scaled",
        help="Input dizini (default: data/scaled)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/engineered",
        help="Output dizini (default: data/engineered)"
    )
    
    parser.add_argument(
        "--polynomial",
        action="store_true",
        help="Polynomial features oluştur (çok fazla feature oluşturur!)"
    )
    
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree (default: 2)"
    )
    
    parser.add_argument(
        "--no-planetary",
        action="store_true",
        help="Planetary features oluşturma"
    )
    
    parser.add_argument(
        "--no-interactions",
        action="store_true",
        help="Interaction features oluşturma"
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
    logger.info("="*70)
    logger.info("VERİ YÜKLEME")
    logger.info("="*70)
    
    train_path = input_dir / "train_scaled.csv"
    val_path = input_dir / "val_scaled.csv"
    test_path = input_dir / "test_scaled.csv"
    
    # Dosya kontrolü
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    
    # Yükle
    logger.info(f"📂 Input: {input_dir}")
    
    train_df = pd.read_csv(train_path, low_memory=False)
    logger.info(f"  ✓ train_scaled.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")
    
    val_df = pd.read_csv(val_path, low_memory=False)
    logger.info(f"  ✓ val_scaled.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")
    
    test_df = pd.read_csv(test_path, low_memory=False)
    logger.info(f"  ✓ test_scaled.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")
    
    logger.info("="*70)
    
    return train_df, val_df, test_df


def show_new_features(original_cols: list, engineered_cols: list):
    """
    Yeni oluşturulan feature'ları göster.
    
    Args:
        original_cols: Orijinal sütunlar
        engineered_cols: Engineered sütunlar
    """
    new_features = set(engineered_cols) - set(original_cols)
    
    if new_features:
        logger.info("\n" + "="*70)
        logger.info(f"YENİ OLUŞTURULAN FEATURES ({len(new_features)} adet)")
        logger.info("="*70)
        
        # Kategorize et
        planetary = [f for f in new_features if any(key in f for key in ['planet', 'orbital', 'stellar', 'habitable', 'transit', 'teq', 'snr'])]
        interaction = [f for f in new_features if '_X_' in f]
        size_category = [f for f in new_features if f.startswith('is_')]
        polynomial = [f for f in new_features if '^' in f or ' ' in f]
        
        if planetary:
            logger.info(f"\n🪐 Planetary Features ({len(planetary)}):")
            for feat in sorted(planetary)[:10]:  # İlk 10'u göster
                logger.info(f"   - {feat}")
            if len(planetary) > 10:
                logger.info(f"   ... ve {len(planetary) - 10} tane daha")
        
        if size_category:
            logger.info(f"\n📏 Size Category Features ({len(size_category)}):")
            for feat in sorted(size_category):
                logger.info(f"   - {feat}")
        
        if interaction:
            logger.info(f"\n🔗 Interaction Features ({len(interaction)}):")
            for feat in sorted(interaction)[:10]:
                logger.info(f"   - {feat}")
            if len(interaction) > 10:
                logger.info(f"   ... ve {len(interaction) - 10} tane daha")
        
        if polynomial:
            logger.info(f"\n📊 Polynomial Features ({len(polynomial)}):")
            for feat in sorted(polynomial)[:5]:
                logger.info(f"   - {feat}")
            if len(polynomial) > 5:
                logger.info(f"   ... ve {len(polynomial) - 5} tane daha")
        
        logger.info("="*70)


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """
    Engineered veriyi kaydet.
    
    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output dizini
    """
    logger.info("\n" + "="*70)
    logger.info("VERİ KAYDETME")
    logger.info("="*70)
    
    # Output dizini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Output: {output_dir}")
    
    # Kaydet
    train_path = output_dir / "train_engineered.csv"
    train_df.to_csv(train_path, index=False)
    logger.info(f"  ✓ train_engineered.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")
    
    val_path = output_dir / "val_engineered.csv"
    val_df.to_csv(val_path, index=False)
    logger.info(f"  ✓ val_engineered.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")
    
    test_path = output_dir / "test_engineered.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"  ✓ test_engineered.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")
    
    # Boyut kontrolü
    total_size = sum([p.stat().st_size for p in [train_path, val_path, test_path]])
    logger.info(f"  📏 Toplam boyut: {total_size / (1024*1024):.2f} MB")
    
    logger.info("="*70)


def main():
    """Ana fonksiyon."""
    # Setup
    setup_logger()
    args = parse_args()
    
    logger.info("="*70)
    logger.info("🔧 KEPLER EXOPLANET - FEATURE ENGINEERING")
    logger.info("="*70)
    logger.info(f"📅 Tarih: 2025-11-11 16:37:44 UTC")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info(f"🎯 Planetary Features: {not args.no_planetary}")
    logger.info(f"🎯 Interaction Features: {not args.no_interactions}")
    logger.info(f"🎯 Polynomial Features: {args.polynomial}")
    if args.polynomial:
        logger.info(f"   └─ Degree: {args.poly_degree}")
    logger.info("="*70)
    
    # Veriyi yükle
    input_dir = Path(args.input_dir)
    train_df, val_df, test_df = load_data(input_dir)
    
    # Orijinal sütunları sakla
    original_cols = train_df.columns.tolist()
    
    # Feature engineering
    logger.info("\n" + "="*70)
    logger.info("FEATURE ENGINEERING BAŞLADI")
    logger.info("="*70)
    
    engineer = ExoplanetFeatureEngineer(
        poly_degree=args.poly_degree,
        interaction_only=False,
        include_bias=False
    )
    
    # Train
    logger.info("\n🔧 TRAIN SET")
    train_engineered = engineer.fit_transform(
        train_df,
        create_planetary=not args.no_planetary,
        create_polynomial=args.polynomial,
        create_interactions=not args.no_interactions,
        poly_feature_cols=None  # None = tüm numerical features
    )
    
    # Val
    logger.info("\n🔧 VAL SET")
    val_engineered = engineer.fit_transform(
        val_df,
        create_planetary=not args.no_planetary,
        create_polynomial=args.polynomial,
        create_interactions=not args.no_interactions,
        poly_feature_cols=None
    )
    
    # Test
    logger.info("\n🔧 TEST SET")
    test_engineered = engineer.fit_transform(
        test_df,
        create_planetary=not args.no_planetary,
        create_polynomial=args.polynomial,
        create_interactions=not args.no_interactions,
        poly_feature_cols=None
    )
    
    # Yeni feature'ları göster
    show_new_features(original_cols, train_engineered.columns.tolist())
    
    # Kaydet
    output_dir = Path(args.output_dir)
    save_data(train_engineered, val_engineered, test_engineered, output_dir)
    
    # Özet
    logger.info("\n" + "="*70)
    logger.info("📊 ÖZET")
    logger.info("="*70)
    logger.info(f"✅ Orijinal feature sayısı: {len(original_cols)}")
    logger.info(f"✅ Yeni feature sayısı: {len(train_engineered.columns)}")
    logger.info(f"✅ Oluşturulan feature: {len(train_engineered.columns) - len(original_cols)}")
    logger.info(f"✅ Train: {len(train_engineered):,} satır")
    logger.info(f"✅ Val: {len(val_engineered):,} satır")
    logger.info(f"✅ Test: {len(test_engineered):,} satır")
    logger.info(f"📂 Output: {output_dir}")
    logger.info("="*70)
    
    if args.polynomial:
        logger.warning("\n⚠️  DİKKAT: Polynomial features çok fazla feature oluşturdu!")
        logger.warning("    Sonraki adım: Feature Selection (şiddetle önerilir)")
    else:
        logger.info("\n💡 İPUCU: Polynomial features denemek için:")
        logger.info("    python scripts/engineer_features.py --polynomial")
    
    logger.info("\n🎉 Feature engineering tamamlandı!")
    logger.info("\n🚀 Sonraki adım: Feature Selection (en iyi 30-50 feature'ı seç)")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())