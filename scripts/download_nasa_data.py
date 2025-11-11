#!/usr/bin/env python3
"""
NASA Kepler KOI verisini indir.

Bu script, NASA Exoplanet Archive API'den Kepler Objects of Interest (KOI)
tablosunu indirir ve yerel 'data/raw/' dizinine kaydeder.

Kullanım:
    python scripts/download_nasa_data.py
    
    # Veya Makefile ile:
    make download-data

Ortam Değişkenleri:
    NASA_API_KEY: NASA API anahtarı (.env dosyasından okunur)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Proje kök dizinini Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Standart kütüphaneler
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Proje modülleri
from src.core import (
    DATA_RAW,
    NASA_API_BASE_URL,
    NASA_TABLE_NAME,
    NASA_OUTPUT_FORMAT,
    TARGET_COLUMN,
    DataDownloadError,
    DataValidationError,
)


# ============================================
# KONFİGÜRASYON
# ============================================

# .env dosyasını yükle
load_dotenv()

# NASA API ayarları
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
OUTPUT_FILE = DATA_RAW / "kepler_koi.csv"
TIMEOUT_SECONDS = 300  # 5 dakika


# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================

def print_header():
    """Script başlığını yazdır."""
    print("=" * 70)
    print("🪐 NASA KEPLER KOI VERİ İNDİRME")
    print("=" * 70)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔑 API Key: {NASA_API_KEY[:8]}..." if len(NASA_API_KEY) > 8 else "DEMO_KEY")
    print(f"📂 Hedef: {OUTPUT_FILE}")
    print("=" * 70)
    print()


def check_prerequisites():
    """Ön gereksinimleri kontrol et."""
    print("🔍 Ön kontroller yapılıyor...")
    
    # Data dizini var mı?
    if not DATA_RAW.exists():
        print(f"  ⚠️  {DATA_RAW} dizini bulunamadı, oluşturuluyor...")
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Dizin oluşturuldu: {DATA_RAW}")
    else:
        print(f"  ✓ Data dizini mevcut: {DATA_RAW}")
    
    # Eski dosya var mı?
    if OUTPUT_FILE.exists():
        file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)  # MB
        file_mtime = datetime.fromtimestamp(OUTPUT_FILE.stat().st_mtime)
        print(f"  ⚠️  Mevcut dosya bulundu:")
        print(f"     Boyut: {file_size:.2f} MB")
        print(f"     Tarih: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        response = input("  ❓ Üzerine yazmak istiyor musunuz? [y/N]: ")
        if response.lower() != 'y':
            print("  ℹ️  İndirme iptal edildi.")
            sys.exit(0)
        print("  ✓ Eski dosya silinecek")
    
    # API anahtarı kontrolü
    if NASA_API_KEY == "DEMO_KEY":
        print("  ⚠️  DEMO_KEY kullanılıyor (günde 30 request limiti)")
        print("     Gerçek API key için: https://api.nasa.gov/")
    else:
        print(f"  ✓ API Key yapılandırılmış")
    
    print()


def build_api_url() -> str:
    """
    NASA Exoplanet Archive API URL'ini oluştur.
    
    Returns:
        str: Tam API URL
    """
    params = {
        "table": NASA_TABLE_NAME,
        "format": NASA_OUTPUT_FORMAT,
        "select": "*",  # Tüm sütunları al
    }
    
    # URL parametrelerini oluştur
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{NASA_API_BASE_URL}?{param_str}"
    
    return url


def download_data(url: str) -> Optional[str]:
    """
    NASA API'den veri indir.
    
    Args:
        url: İndirme URL'i
        
    Returns:
        str: İndirilen CSV verisi (string)
        
    Raises:
        DataDownloadError: İndirme başarısız olursa
    """
    print("📥 Veri indiriliyor...")
    print(f"   URL: {url}")
    print()
    
    try:
        # Request gönder (stream=True ile progress bar için)
        response = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        
        # Total boyutu al (varsa)
        total_size = int(response.headers.get('content-length', 0))
        
        # Progress bar ile indir
        chunk_size = 8192  # 8KB chunks
        chunks = []
        
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc='  İndiriliyor',
            ncols=80
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    chunks.append(chunk)
                    pbar.update(len(chunk))
        
        # Tüm chunk'ları birleştir
        data = b''.join(chunks).decode('utf-8')
        
        print(f"  ✓ İndirme tamamlandı: {len(data) / (1024*1024):.2f} MB")
        print()
        
        return data
        
    except requests.exceptions.Timeout:
        raise DataDownloadError(
            f"İndirme zaman aşımına uğradı ({TIMEOUT_SECONDS}s)"
        )
    except requests.exceptions.RequestException as e:
        raise DataDownloadError(f"İndirme hatası: {str(e)}")
    except Exception as e:
        raise DataDownloadError(f"Beklenmeyen hata: {str(e)}")


def validate_data(df: pd.DataFrame):
    """
    İndirilen veriyi doğrula.
    
    Args:
        df: Pandas DataFrame
        
    Raises:
        DataValidationError: Doğrulama başarısız olursa
    """
    print("🔍 Veri doğrulaması yapılıyor...")
    
    # Boş mu?
    if df.empty:
        raise DataValidationError("DataFrame boş!")
    
    print(f"  ✓ Satır sayısı: {len(df):,}")
    print(f"  ✓ Sütun sayısı: {len(df.columns)}")
    
    # Target sütunu var mı?
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(
            f"Target sütunu '{TARGET_COLUMN}' bulunamadı!"
        )
    print(f"  ✓ Target sütunu mevcut: {TARGET_COLUMN}")
    
    # Target dağılımı
    target_dist = df[TARGET_COLUMN].value_counts()
    print(f"  ✓ Target dağılımı:")
    for value, count in target_dist.items():
        pct = (count / len(df)) * 100
        print(f"     {value}: {count:,} (%{pct:.1f})")
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    print(f"  ✓ Toplam eksik değer: {total_missing:,} (%{missing_pct:.1f})")
    
    # Memory kullanımı
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  ✓ Memory kullanımı: {memory_mb:.2f} MB")
    
    print()


def save_data(data: str, output_path: Path):
    """
    Veriyi dosyaya kaydet.
    
    Args:
        data: CSV verisi (string)
        output_path: Kayıt yolu
    """
    print(f"💾 Veri kaydediliyor: {output_path}")
    
    # Dizin yoksa oluştur
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dosyaya yaz
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(data)
    
    # Dosya boyutunu göster
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Kaydedildi: {file_size:.2f} MB")
    print()


def print_summary(df: pd.DataFrame, elapsed_time: float):
    """
    Özet bilgileri yazdır.
    
    Args:
        df: Pandas DataFrame
        elapsed_time: Geçen süre (saniye)
    """
    print("=" * 70)
    print("📊 ÖZET")
    print("=" * 70)
    print(f"✅ İndirme başarılı!")
    print(f"📂 Dosya: {OUTPUT_FILE}")
    print(f"📏 Boyut: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")
    print(f"📊 Satır: {len(df):,}")
    print(f"📊 Sütun: {len(df.columns)}")
    print(f"⏱️  Süre: {elapsed_time:.1f} saniye")
    print("=" * 70)
    print()
    print("🎉 Sonraki adım: Exploratory Data Analysis (EDA)")
    print("   make run-jupyter")
    print("   notebooks/01_exploratory_data_analysis.ipynb")
    print()


# ============================================
# ANA FONKSİYON
# ============================================

def main():
    """Ana indirme fonksiyonu."""
    from time import time
    
    start_time = time()
    
    try:
        # Başlık
        print_header()
        
        # Ön kontroller
        check_prerequisites()
        
        # API URL'i oluştur
        url = build_api_url()
        
        # Veriyi indir
        data = download_data(url)
        
        # CSV'yi pandas'a yükle
        print("📊 Veri parse ediliyor...")
        df = pd.read_csv(
            pd.io.common.StringIO(data),
            comment='#',  # Yorum satırlarını atla
            low_memory=False
        )
        print(f"  ✓ Parse tamamlandı")
        print()
        
        # Doğrulama
        validate_data(df)
        
        # Kaydet
        save_data(data, OUTPUT_FILE)
        
        # Özet
        elapsed_time = time() - start_time
        print_summary(df, elapsed_time)
        
        return 0
        
    except DataDownloadError as e:
        print(f"\n❌ İndirme hatası: {e}")
        return 1
        
    except DataValidationError as e:
        print(f"\n❌ Doğrulama hatası: {e}")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  İndirme kullanıcı tarafından iptal edildi.")
        return 130
        
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())