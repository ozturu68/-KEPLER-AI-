# 🪐 Exoplanet ML - NASA Kepler Dış Gezegen Tespiti

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Makine Öğrenmesi](https://img.shields.io/badge/ML-CatBoost%20%7C%20LightGBM%20%7C%20XGBoost-green)](https://github.com/ozturu68/kepler-new)

> NASA Kepler misyonu verilerini kullanarak dış gezegen (exoplanet) adaylarını tespit eden end-to-end makine öğrenmesi pipeline'ı.

---

## 📊 Proje Hakkında

Bu proje, NASA'nın **Kepler Uzay Teleskobu** tarafından toplanan Kepler Objects of Interest (KOI) veritabanını kullanarak, bir yıldızın etrafında gezegen olup olmadığını tahmin eden bir makine öğrenmesi sistemidir.

### 🎯 Hedefler

- ✅ NASA Kepler KOI veritabanını kullanarak exoplanet tespiti
- ✅ CatBoost, LightGBM ve XGBoost algoritmalarıyla model karşılaştırması
- ✅ SHAP ile model açıklanabilirliği (explainability)
- ✅ FastAPI ile production-ready REST API
- ✅ Streamlit ile interaktif web arayüzü
- ✅ Modern MLOps best practices (testler, CI/CD, monitoring)

### 🌟 Öne Çıkan Özellikler

- **Hibrit Depolama Stratejisi**: 1TB SSD ile güçlü yerel depolama, bulut deployment için Docker
- **Türkçe Dokümantasyon**: Tüm kod yorumları ve dökümanlar Türkçe
- **Modüler Mimari**: Temiz kod yapısı, kolay genişletilebilir
- **Kapsamlı Testler**: Unit, integration ve e2e testler
- **Code Quality**: Black, isort, mypy, pylint, bandit entegrasyonu

---

## 🚀 Hızlı Başlangıç

### Sistem Gereksinimleri

- **İşletim Sistemi**: Pop!_OS 22.04 (veya Ubuntu 20.04+)
- **Python**: 3.8 veya üzeri
- **RAM**: 16GB (önerilir)
- **Depolama**: 20GB boş alan (1TB SSD tercih edilir)
- **GPU**: NVIDIA GPU (opsiyonel, CUDA 11.8+ destekli)

### Kurulum

#### 1. Repository'yi Klonlayın

```bash
git clone https://github.com/ozturu68/kepler-new.git
cd kepler-new
```

#### 2. Otomatik Kurulum (Önerilen)

```bash
# Tek komutla tüm kurulumu tamamla
make setup
```

Bu komut:
- Virtual environment oluşturur
- Tüm bağımlılıkları kurar
- Pre-commit hooks'u yapılandırır

#### 3. Manuel Kurulum (Alternatif)

```bash
# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları kur
pip install -r requirements-dev.txt

# Pre-commit hooks'u kur
pre-commit install
```

#### 4. Environment Variables

```bash
# .env dosyası oluştur
cp .env.example .env

# .env dosyasını düzenle ve NASA API key'inizi ekleyin
nano .env
```

NASA API Key almak için: [https://api.nasa.gov/](https://api.nasa.gov/)

---

## 📁 Proje Yapısı

```
kepler-new/
├── config/                 # Konfigürasyon dosyaları (YAML)
│   ├── feature_config.yaml
│   ├── model_config.yaml
│   └── logging_config.yaml
│
├── data/                   # Veri dosyaları (GİTİGNORE'DA!)
│   ├── raw/               # Ham NASA verileri (~500MB-1GB)
│   ├── processed/         # İşlenmiş, temizlenmiş veri
│   ├── external/          # Harici kaynaklar
│   └── sample/            # Test için örnek veri
│
├── deployment/             # Deployment yapılandırmaları
│   ├── kubernetes/        # K8s manifests (opsiyonel)
│   ├── terraform/         # Infrastructure as Code (opsiyonel)
│   └── streamlit_cloud/   # Streamlit Cloud config
│
├── docs/                   # Dokümantasyon
│   ├── architecture.md    # Sistem mimarisi
│   ├── api_reference.md   # API dokümantasyonu
│   └── model_details.md   # Model detayları
│
├── models/                 # Model artifacts (GİTİGNORE'DA!)
│   ├── experiments/       # Deneme modelleri (~5-10GB)
│   ├── production/        # Production modeller
│   └── registry/          # Model versiyonları
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering_research.ipynb
│   └── 03_model_experiments.ipynb
│
├── results/                # Çıktılar (GİTİGNORE'DA!)
│   ├── figures/           # Grafikler ve görseller
│   ├── logs/              # Log dosyaları
│   └── reports/           # Raporlar (HTML, PDF)
│
├── scripts/                # Yardımcı scriptler
│   ├── download_nasa_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── batch_predict.py
│
├── src/                    # Kaynak kodlar
│   ├── api/               # FastAPI REST API
│   ├── cli/               # Command-line interface
│   ├── core/              # Core utilities ve constants
│   ├── data/              # Data processing pipeline
│   ├── evaluation/        # Model evaluation
│   ├── explainability/    # SHAP, feature importance
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── training/          # Training pipeline
│   ├── utils/             # Yardımcı fonksiyonlar
│   └── webapp/            # Streamlit web app
│
├── tests/                  # Testler
│   ├── unit/              # Birim testleri
│   ├── integration/       # Entegrasyon testleri
│   └── e2e/               # End-to-end testler
│
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore kuralları
├── .pre-commit-config.yaml # Pre-commit hooks
├── Dockerfile              # Docker image tanımı
├── docker-compose.yml      # Multi-container setup
├── Makefile                # Make komutları
├── pyproject.toml          # Python proje konfigürasyonu
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md               # Bu dosya
```

---

## 🛠️ Kullanım

### Makefile Komutları

Proje için tüm yaygın işlemler Makefile ile kolaylaştırılmıştır:

```bash
# Yardım menüsünü göster
make help

# Kurulum
make setup              # Otomatik kurulum
make install            # Sadece production dependencies
make install-dev        # Development dependencies + pre-commit

# Temizlik
make clean              # Cache dosyalarını temizle
make clean-all          # Her şeyi temizle (dikkatli!)

# Test
make test               # Tüm testler
make test-unit          # Sadece unit testler
make test-cov           # Coverage raporu ile

# Kod Kalitesi
make lint               # Tüm linter'lar
make format             # Kodu otomatik formatla
make type-check         # MyPy tip kontrolü
make security-check     # Bandit güvenlik taraması

# Servisler
make run-api            # FastAPI başlat (http://localhost:8000)
make run-webapp         # Streamlit başlat (http://localhost:8501)
make run-jupyter        # Jupyter Lab başlat

# ML İşlemleri
make download-data      # NASA verisini indir
make train              # Model eğit
make evaluate           # Model değerlendir
make predict            # Batch prediction

# Docker
make docker-build       # Image oluştur
make docker-run         # Container çalıştır

# CI/CD
make ci                 # CI pipeline (lint + test-cov)
make all                # Tam workflow (clean + install + lint + test)
```

---

## 📖 Detaylı Kullanım

### 1. Veri İndirme

```bash
# NASA Kepler KOI verisini indir
make download-data

# Veya manuel olarak:
python scripts/download_nasa_data.py
```

### 2. Exploratory Data Analysis (EDA)

```bash
# Jupyter Lab'i başlat
make run-jupyter

# notebooks/01_exploratory_data_analysis.ipynb'ı aç
```

### 3. Model Eğitimi

```bash
# Varsayılan konfigürasyon ile
make train

# Özel konfigürasyon ile
python scripts/train_model.py --config config/model_config.yaml

# Hiperparametre tuning ile
python scripts/train_model.py --tune --n-trials 100
```

### 4. Model Değerlendirme

```bash
make evaluate

# Veya belirli bir modeli değerlendir
python scripts/evaluate_model.py --model-path models/production/best_model.pkl
```

### 5. FastAPI Kullanımı

```bash
# API'yi başlat
make run-api

# API Docs: http://localhost:8000/docs
```

**Örnek API Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "koi_period": 10.5,
    "koi_depth": 100.0,
    "koi_duration": 3.5,
    "koi_prad": 2.0,
    "koi_teq": 300,
    "koi_steff": 5500
  }'
```

**Örnek Response:**

```json
{
  "prediction": "CONFIRMED",
  "probability": 0.87,
  "confidence": "high",
  "shap_values": {...}
}
```

### 6. Streamlit Web App Kullanımı

```bash
# Web app'i başlat
make run-webapp

# Browser'da aç: http://localhost:8501
```

Web arayüzünde:
- Tekli gezegen tahmini
- Toplu CSV upload
- SHAP açıklama grafikleri
- Model performans metrikleri

---

## 🧪 Testler

```bash
# Tüm testleri çalıştır
make test

# Sadece unit testler
make test-unit

# Sadece integration testler
make test-integration

# Coverage raporu ile
make test-cov
# Rapor: htmlcov/index.html
```

### Test Yapısı

- **Unit Tests**: `tests/unit/` - Modüllerin izole testleri
- **Integration Tests**: `tests/integration/` - Pipeline testleri
- **E2E Tests**: `tests/e2e/` - Tam workflow testleri

---

## 🐳 Docker Kullanımı

### Image Oluşturma

```bash
make docker-build
```

### Container Çalıştırma

```bash
make docker-run
```

### Docker Compose ile

```bash
# Tüm servisleri başlat (API + Streamlit)
docker-compose up -d

# Servisleri durdur
docker-compose down
```

---

## 💾 Hibrit Depolama Stratejisi

Bu proje **güçlü yerel depolama** stratejisi kullanır:

### Yerel Depolama (1TB SSD)
- ✅ **data/**: Ham ve işlenmiş veriler (~2GB)
- ✅ **models/**: Tüm model artifacts (~10GB)
- ✅ **results/**: Grafikler, loglar, raporlar (~500MB)

### Git Repository (Sadece Kod)
- ✅ Kaynak kodlar
- ✅ Konfigürasyon dosyaları
- ✅ Testler ve dokümantasyon
- ❌ Veri, modeller, sonuçlar (.gitignore'da)

### Neden DVC Yok?
- 🚀 **1TB SSD**: Yerel depolama bol ve hızlı
- 💰 **Maliyet**: Bulut storage gereksiz
- ⚡ **Performans**: Yerel erişim çok daha hızlı
- 🎯 **Basitlik**: Tek kişilik proje için yeterli

### Yedekleme (Opsiyonel)
```bash
# Manuel Google Drive yedekleme
# (Gelecekte eklenebilir)
```

---

## 📊 Model Performansı

### Mevcut Sonuçlar

| Model      | Accuracy | Precision | Recall | F1-Score | Training Time |
|------------|----------|-----------|--------|----------|---------------|
| CatBoost   | TBD      | TBD       | TBD    | TBD      | TBD           |
| LightGBM   | TBD      | TBD       | TBD    | TBD      | TBD           |
| XGBoost    | TBD      | TBD       | TBD    | TBD      | TBD           |

*Not: Model eğitimi tamamlandıkça güncellenecek.*

### Model Özellikleri

- **Algoritma**: Gradient Boosting (CatBoost, LightGBM, XGBoost)
- **Feature Engineering**: 50+ özellik
- **Imbalanced Data**: SMOTE kullanımı
- **Validation**: 5-fold cross-validation
- **Explainability**: SHAP values

---

## 🔧 Geliştirme

### Kod Kalitesi Standartları

```bash
# Kod formatla
make format

# Linter kontrolü
make lint

# Tip kontrolü
make type-check

# Güvenlik taraması
make security-check

# Tüm kontroller
make ci
```

### Pre-commit Hooks

Otomatik olarak her commit'te çalışır:
- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)
- Bandit (security)

### Yeni Özellik Ekleme

1. Yeni branch oluştur: `git checkout -b feature/yeni-ozellik`
2. Kod yaz ve test et: `make test`
3. Formatla: `make format`
4. Commit: `git commit -m "feat: yeni özellik açıklaması"`
5. Push: `git push origin feature/yeni-ozellik`

---

## 📚 Dokümantasyon

Detaylı dokümantasyon `docs/` klasöründe:

- [Mimari Dokümantasyon](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Model Detayları](docs/model_details.md)
- [Deployment Rehberi](docs/deployment.md)

---

## 🚀 Deployment

### Streamlit Cloud (Önerilen - Ücretsiz)

1. GitHub repository'yi public yap
2. [Streamlit Cloud](https://streamlit.io/cloud)'a git
3. Repository'yi bağla
4. `src/webapp/app.py` dosyasını seç
5. Deploy!

### Railway.app (API için)

1. [Railway.app](https://railway.app)'e git
2. GitHub repository'yi bağla
3. Environment variables ekle
4. Deploy!

### Docker (Self-hosted)

```bash
# Production image oluştur
make docker-build

# Container çalıştır
make docker-run
```

---

## 🤝 Katkıda Bulunma

Bu kişisel bir öğrenme projesidir. Öneriler ve geri bildirimler için:

- **Issues**: GitHub Issues'da bug/feature önerileri
- **Discussions**: Genel tartışmalar için

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

## 🙏 Teşekkürler

- **NASA Exoplanet Archive**: Kepler verileri için
- **Kepler Mission Team**: Bilimsel veriler için
- **Open Source Community**: Kullanılan kütüphaneler için

---

## 📧 İletişim

- **GitHub**: [@ozturu68](https://github.com/ozturu68)
- **Proje**: [kepler-new](https://github.com/ozturu68/kepler-new)

---

## 🎯 Proje Durumu

- [x] Proje yapısı oluşturuldu
- [x] Konfigürasyon dosyaları hazır
- [ ] NASA verisi indirildi
- [ ] EDA tamamlandı
- [ ] Feature engineering tamamlandı
- [ ] Model training tamamlandı
- [ ] API geliştirme tamamlandı
- [ ] Web app geliştirme tamamlandı
- [ ] Deployment yapıldı

**Mevcut Durum**: 🟡 Development aşamasında

---

## 💡 İpuçları

### Performans Optimizasyonu

```bash
# GPU kullanımını etkinleştir
export ENABLE_GPU=true

# Paralel processing
export TRAIN_BATCH_SIZE=64
export PYTEST_WORKERS=auto
```

### Debug Mode

```bash
# Debug logları için
export LOG_LEVEL=DEBUG

# Verbose mode
make test -v
```

### Hızlı Iterasyon

```bash
# Watch mode - dosya değişince otomatik test
make test-watch

# Jupyter auto-reload
%load_ext autoreload
%autoreload 2
```

---

**🌟 Projeyi beğendiyseniz GitHub'da yıldız vermeyi unutmayın!**

```bash
# Son güncelleme: 2024-11-09
# Versiyon: 0.1.0 (Alpha)
```