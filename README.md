# 🪐 Kepler Exoplanet ML Project

NASA Kepler uzay teleskobu verilerini kullanarak gezegen adaylarını (exoplanet) sınıflandıran machine learning projesi.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-359%20passed-success.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-66.17%25-yellow.svg)](htmlcov/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Test](#test)
- [Dokümantasyon](#dokümantasyon)
- [Katkıda Bulunma](#katkıda-bulunma)

## ✨ Özellikler

### ✅ Tamamlanan Modüller

- **🔧 Veri İşleme Pipeline**

  - Veri temizleme (97% test coverage)
  - Eksik değer yönetimi
  - Outlier tespiti ve işleme
  - Veri validasyonu

- **🎨 Feature Engineering**

  - Planetary feature oluşturma
  - Interaction features
  - Polynomial features
  - Feature scaling (Standard, MinMax, Robust)
  - Feature selection (84-93% coverage)

- **📊 Model Değerlendirme**

  - Comprehensive metrics (97% coverage)
  - Confusion matrix
  - Classification reports
  - ROC-AUC scoring
  - Cross-validation support

- **🔗 Integration Tests**
  - 19 end-to-end pipeline testi
  - 100% test coverage
  - Edge case scenarios

### 🚧 Geliştirme Aşamasında

- **🤖 Model Training Pipeline**

  - Base model class (refactor gerekli)
  - CatBoost implementation (test coverage düşük)
  - Hyperparameter tuning
  - Model registry & versioning

- **🌐 API & Serving** (Planlanan)

  - FastAPI REST endpoints
  - Prediction serving
  - Model explainability (SHAP)
  - Health checks

- **💻 CLI & Web Interface** (Planlanan)
  - Command-line interface
  - Streamlit dashboard
  - Interactive visualizations

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- pip
- virtualenv (önerilen)

### Hızlı Başlangıç

```bash
# Repository'yi klonla
git clone https://github.com/sulegogh/kepler-new.git
cd kepler-new

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# Pre-commit hook'ları kur
pre-commit install

# Testleri çalıştır
pytest tests/ -v
```

## 📊 Kullanım

### Veri İşleme

```python
from src.data.cleaners import clean_data
from src.data.preprocessors import MissingValueHandler

# Veriyi temizle
df_cleaned = clean_data(df, handle_outliers=True, method='clip')

# Eksik değerleri işle
handler = MissingValueHandler(numerical_strategy='median')
df_filled = handler.fit_transform(df_cleaned)
```

### Feature Engineering

```python
from src.features.engineering import ExoplanetFeatureEngineer
from src.features.scalers import FeatureScaler
from src.features.selection import FeatureSelector

# Yeni feature'lar oluştur
engineer = ExoplanetFeatureEngineer()
df_engineered = engineer.fit_transform(df)

# Scale features
scaler = FeatureScaler(method='standard')
df_scaled = scaler.fit_transform(df_engineered)

# Select best features
selector = FeatureSelector()
selected_features, info = selector.select_features(
    df_scaled,
    target_col='koi_disposition',
    n_features=50
)
```

### Model Değerlendirme

```python
from src.evaluation.metrics import evaluate_model, compare_metrics

# Modeli değerlendir
metrics = evaluate_model(y_true, y_pred, y_proba=y_proba, dataset_name='Test')

# Metrikleri karşılaştır
compare_metrics(train_metrics, val_metrics, test_metrics)
```

## 📁 Proje Yapısı

```
kepler-new/
├── src/
│   ├── core/              # Temel sabitler ve yardımcılar
│   ├── data/              # Veri işleme modülleri
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementasyonları
│   ├── evaluation/        # Değerlendirme metrikleri
│   ├── training/          # Eğitim pipeline (geliştiriliyor)
│   ├── api/               # REST API (planlanan)
│   ├── cli/               # CLI interface (planlanan)
│   └── webapp/            # Web dashboard (planlanan)
│
├── tests/
│   ├── test_core/         # 72 tests
│   ├── test_data/         # 78 tests
│   ├── test_features/     # 101 tests
│   ├── test_evaluation/   # 33 tests
│   ├── test_models/       # 56 tests
│   └── test_integrations/ # 19 tests
│
├── data/
│   ├── raw/              # Ham veri
│   └── processed/        # İşlenmiş veri
│
├── models/               # Eğitilmiş modeller
├── notebooks/            # Jupyter notebooks
├── docs/                 # Dokümantasyon
├── pytest.ini           # Pytest konfigürasyonu
├── .pre-commit-config.yaml
└── requirements.txt
```

## 🧪 Test

### Tüm Testleri Çalıştır

```bash
# Verbose mode
pytest tests/ -v

# Coverage report ile
pytest tests/ --cov=src --cov-report=html

# Hızlı özet
pytest tests/ -q

# Sadece belirli modül
pytest tests/test_features/ -v
```

### Test İstatistikleri

```
Total Tests:     359
Passed:          359 (100%)
Failed:          0
Coverage:        66.17%
Execution Time:  ~10 seconds
```

### Test Kategorileri

- **Unit Tests:** 340 tests (isolated component testing)
- **Integration Tests:** 19 tests (end-to-end pipeline testing)
- **Edge Cases:** Comprehensive edge case coverage

## 📚 Dokümantasyon

Detaylı dokümantasyon için:

- [API Dokümantasyonu](docs/API.md)
- [Geliştirici Kılavuzu](docs/DEVELOPMENT.md)
- [Veri Pipeline](docs/DATA_PIPELINE.md)
- [Feature Engineering](docs/FEATURES.md)
- [Model Training](docs/TRAINING.md) (yakında)

## 🎯 Proje Durumu

### ✅ Tamamlandı (Phase 1-7)

- [x] Core utilities (100% coverage)
- [x] Label encoding/decoding (97% coverage)
- [x] Data cleaning (97% coverage)
- [x] Data preprocessing (81% coverage)
- [x] Feature engineering (84% coverage)
- [x] Feature scaling (91% coverage)
- [x] Feature selection (93% coverage)
- [x] Evaluation metrics (97% coverage)
- [x] Integration tests (100% coverage)
- [x] Model loading infrastructure (91% coverage)

### 🚧 Geliştiriliyor (Phase 8)

- [ ] Model base class refactoring (12% → 80% target)
- [ ] CatBoost model tests (9% → 70% target)
- [ ] Model registry & versioning

### 📋 Planlanan (Phase 9-12)

- [ ] Training pipeline (trainer.py)
- [ ] Hyperparameter tuning (Optuna)
- [ ] REST API (FastAPI)
- [ ] CLI interface
- [ ] Web dashboard (Streamlit)
- [ ] SHAP explainability
- [ ] CI/CD pipeline

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

### Commit Mesaj Formatı

```
feat: Yeni özellik
fix: Bug düzeltme
docs: Dokümantasyon
test: Test ekleme/düzeltme
refactor: Kod refactoring
chore: Genel bakım
```

## 📊 Performans

- **Test Execution:** <10 seconds (359 tests)
- **Code Quality:** Black + isort + flake8 compliant
- **Pre-commit Hooks:** All passing
- **Coverage:** 66.17% (tested modules: ~90%)

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👤 Yazar

**sulegogh**

- GitHub: [@sulegogh](https://github.com/sulegogh)

## 🙏 Teşekkürler

- NASA Kepler Mission
- NASA Exoplanet Archive
- CatBoost Team
- scikit-learn Contributors

## 📈 Yol Haritası

### Q4 2024

- ✅ Phase 1-7: Core modules & comprehensive testing
- 🚧 Phase 8: Model refactoring & registry

### Q1 2025

- 📋 Phase 9: Training pipeline
- 📋 Phase 10: API development
- 📋 Phase 11: Explainability

### Q2 2025

- 📋 Phase 12: Web dashboard
- 📋 CI/CD integration
- 📋 Production deployment

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
