#!/usr/bin/env python3
"""
Feature selection scripti.

Bu script, engineered veriyi alır, en iyi feature'ları seçer ve
data/selected/ dizinine kaydeder.

Kullanım:
    python scripts/select_features.py
    
    # Farklı feature sayısı ile:
    python scripts/select_features.py --n-features 40
    
    # Veya Makefile ile:
    make select-features
"""

import sys
import argparse
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.core import TARGET_COLUMN
from src.features.selection import FeatureSelector, select_features_train_val_test


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
    parser = argparse.ArgumentParser(description="Feature selection scripti")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/engineered",
        help="Input dizini (default: data/engineered)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selected",
        help="Output dizini (default: data/selected)"
    )
    
    parser.add_argument(
        "--n-features",
        type=int,
        default=50,
        help="Seçilecek feature sayısı (default: 50)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "correlation", "importance", "hybrid"],
        help="Selection yöntemi (default: auto)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Feature importance grafiklerini kaydet"
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
    
    train_path = input_dir / "train_engineered.csv"
    val_path = input_dir / "val_engineered.csv"
    test_path = input_dir / "test_engineered.csv"
    
    # Dosya kontrolü
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    
    # Yükle
    logger.info(f"📂 Input: {input_dir}")
    
    train_df = pd.read_csv(train_path, low_memory=False)
    logger.info(f"  ✓ train_engineered.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")
    
    val_df = pd.read_csv(val_path, low_memory=False)
    logger.info(f"  ✓ val_engineered.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")
    
    test_df = pd.read_csv(test_path, low_memory=False)
    logger.info(f"  ✓ test_engineered.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")
    
    logger.info("="*70)
    
    return train_df, val_df, test_df


def show_selection_summary(selection_info: dict):
    """
    Feature selection özetini göster.
    
    Args:
        selection_info: Selection bilgileri
    """
    logger.info("\n" + "="*70)
    logger.info("SELECTION ÖZETİ")
    logger.info("="*70)
    
    logger.info(f"📊 İlk feature sayısı: {selection_info['initial_features']}")
    logger.info(f"📊 Filtreleme sonrası: {selection_info['after_filtering']}")
    logger.info(f"📊 Seçilen feature sayısı: {selection_info['selected_features']}")
    
    logger.info(f"\n🗑️  Kaldırılan feature'lar:")
    logger.info(f"   - Düşük varyans: {selection_info['dropped_low_variance']}")
    logger.info(f"   - Yüksek korelasyon: {selection_info['dropped_high_correlation']}")
    logger.info(f"   - Düşük importance: {selection_info['dropped_low_importance']}")
    logger.info(f"   - TOPLAM: {selection_info['initial_features'] - selection_info['selected_features']}")
    
    logger.info(f"\n🎯 Top 10 Features (Importance):")
    importance_df = selection_info['importance_df']
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"   {idx+1:2d}. {row['feature']:40s} → {row['importance']:.4f}")
    
    logger.info("="*70)


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path, top_n: int = 30):
    """
    Feature importance grafiği çiz ve kaydet.
    
    Args:
        importance_df: Feature importance DataFrame
        output_dir: Çıktı dizini
        top_n: Gösterilecek feature sayısı
    """
    logger.info(f"\n📊 Feature importance grafiği çiziliyor (top {top_n})...")
    
    # Top N al
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Horizontal bar plot
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # En önemli üstte
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Kaydet
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Grafik kaydedildi: {output_path}")


def show_dropped_features(selector: FeatureSelector):
    """
    Kaldırılan feature'ları göster.
    
    Args:
        selector: FeatureSelector objesi
    """
    logger.info("\n" + "="*70)
    logger.info("KALDIRILAN FEATURES")
    logger.info("="*70)
    
    if selector.dropped_features.get('low_variance'):
        logger.info(f"\n📉 Düşük Varyans ({len(selector.dropped_features['low_variance'])} adet):")
        for feat in selector.dropped_features['low_variance'][:10]:
            logger.info(f"   - {feat}")
        if len(selector.dropped_features['low_variance']) > 10:
            logger.info(f"   ... ve {len(selector.dropped_features['low_variance']) - 10} tane daha")
    
    if selector.dropped_features.get('high_correlation'):
        logger.info(f"\n🔗 Yüksek Korelasyon ({len(selector.dropped_features['high_correlation'])} adet):")
        for feat in selector.dropped_features['high_correlation'][:10]:
            logger.info(f"   - {feat}")
        if len(selector.dropped_features['high_correlation']) > 10:
            logger.info(f"   ... ve {len(selector.dropped_features['high_correlation']) - 10} tane daha")
    
    logger.info("="*70)


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
              output_dir: Path, selected_features: list):
    """
    Selected veriyi kaydet.
    
    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output dizini
        selected_features: Seçilen feature'lar
    """
    logger.info("\n" + "="*70)
    logger.info("VERİ KAYDETME")
    logger.info("="*70)
    
    # Output dizini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Output: {output_dir}")
    
    # Kaydet
    train_path = output_dir / "train_selected.csv"
    train_df.to_csv(train_path, index=False)
    logger.info(f"  ✓ train_selected.csv: {len(train_df):,} satır, {len(train_df.columns)} sütun")
    
    val_path = output_dir / "val_selected.csv"
    val_df.to_csv(val_path, index=False)
    logger.info(f"  ✓ val_selected.csv: {len(val_df):,} satır, {len(val_df.columns)} sütun")
    
    test_path = output_dir / "test_selected.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"  ✓ test_selected.csv: {len(test_df):,} satır, {len(test_df.columns)} sütun")
    
    # Feature listesini kaydet
    features_path = output_dir / "selected_features.txt"
    with open(features_path, 'w') as f:
        f.write(f"# Selected Features ({len(selected_features)} adet)\n")
        f.write(f"# Date: 2025-11-11 16:42:57 UTC\n")
        f.write(f"# User: sulegogh\n\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
    logger.info(f"  ✓ selected_features.txt: {len(selected_features)} feature")
    
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
    logger.info("🔧 KEPLER EXOPLANET - FEATURE SELECTION")
    logger.info("="*70)
    logger.info(f"📅 Tarih: 2025-11-11 16:42:57 UTC")
    logger.info(f"👤 Kullanıcı: sulegogh")
    logger.info(f"🎯 Target feature sayısı: {args.n_features}")
    logger.info(f"🎯 Selection method: {args.method}")
    logger.info("="*70)
    
    # Veriyi yükle
    input_dir = Path(args.input_dir)
    train_df, val_df, test_df = load_data(input_dir)
    
    # Feature selection
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION BAŞLADI")
    logger.info("="*70)
    
    train_selected, val_selected, test_selected, selector, selection_info = select_features_train_val_test(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_features=args.n_features,
        method=args.method
    )
    
    # Özet göster
    show_selection_summary(selection_info)
    show_dropped_features(selector)
    
    # Grafik çiz (opsiyonel)
    if args.save_plots:
        output_dir = Path(args.output_dir)
        plot_feature_importance(selection_info['importance_df'], output_dir, top_n=30)
    
    # Kaydet
    output_dir = Path(args.output_dir)
    save_data(train_selected, val_selected, test_selected, output_dir, selector.selected_features)
    
    # Final özet
    logger.info("\n" + "="*70)
    logger.info("📊 FİNAL ÖZET")
    logger.info("="*70)
    logger.info(f"✅ Orijinal feature sayısı: {selection_info['initial_features']}")
    logger.info(f"✅ Seçilen feature sayısı: {selection_info['selected_features']}")
    logger.info(f"✅ Reduction: {(1 - selection_info['selected_features']/selection_info['initial_features'])*100:.1f}%")
    logger.info(f"✅ Train: {len(train_selected):,} satır")
    logger.info(f"✅ Val: {len(val_selected):,} satır")
    logger.info(f"✅ Test: {len(test_selected):,} satır")
    logger.info(f"📂 Output: {output_dir}")
    logger.info("="*70)
    logger.info("🎉 Feature selection tamamlandı!")
    logger.info("\n🚀 Sonraki adım: Model Training (CatBoost, LightGBM, XGBoost)")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())