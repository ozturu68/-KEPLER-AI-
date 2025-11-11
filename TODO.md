# 🎯 Exoplanet ML - Proje Roadmap

> **Not**: Bu TODO listesi projenin gelişim yol haritasıdır. Her faz tamamlandıkça güncellenecektir.

---

## 📊 Genel İlerleme Durumu

```
█████████████████░░░░░░░░░░░░░░░ 55%

Tamamlanan Fazlar: 7/13
Devam Eden Faz: Faz 2 - Veri İndirme & EDA
```

| Faz      | Durum           | Tamamlanma | Son Güncelleme |
| -------- | --------------- | ---------- | -------------- |
| Faz 0    | ✅ Tamamlandı   | 100%       | 2024-11-09     |
| Faz 1    | ✅ Tamamlandı   | 100%       | 2024-11-09     |
| Faz 2    | 🔄 Devam Ediyor | 25%        | 2024-11-09     |
| Faz 3-13 | ⏳ Bekliyor     | 0%         | -              |

---

## ✅ FAZ 0: Sistem Kurulumu (TAMAMLANDI)

**Hedef**: Geliştirme ortamını hazırla  
**Durum**: ✅ %100 Tamamlandı  
**Süre**: 2024-11-07 → 2024-11-09

### Tamamlanan Görevler

- [x] Pop!\_OS sistem kurulumu
- [x] Python 3.8+ kurulumu
- [x] VS Codium kurulumu
- [x] Git yapılandırması
- [x] Virtual environment oluşturma
- [x] Dependencies kurulumu (requirements.txt + requirements-dev.txt)
- [x] Pre-commit hooks kurulumu

### Notlar

- 16GB RAM, i5 12. nesil, RTX 3050 4GB
- 1TB SSD - Hibrit yerel depolama stratejisi

---

## ✅ FAZ 1: Proje Yapılandırması (TAMAMLANDI)

**Hedef**: Proje iskeletini ve dokümantasyonu tamamla  
**Durum**: ✅ %100 Tamamlandı  
**Süre**: 2024-11-07 → 2024-11-09

### Tamamlanan Görevler

- [x] Proje klasör yapısı oluşturma
- [x] Git repository başlatma
- [x] .gitignore yapılandırması (veri/model dosyaları hariç)
- [x] .env.example ve .env dosyaları
- [x] requirements.txt (production)
- [x] requirements-dev.txt (development)
- [x] pyproject.toml (black, isort, mypy, pytest, coverage)
- [x] .flake8 (lint kuralları)
- [x] .pre-commit-config.yaml
- [x] Makefile (Türkçe, komple)
- [x] README.md (detaylı, hibrit strateji)
- [x] TODO.md (bu dosya)
- [x] LICENSE (MIT)

### Değişiklikler

**Hibrit Depolama Stratejisi:**

- ❌ DVC kaldırıldı (gereksiz, 1TB SSD yeterli)
- ✅ Yerel depolama (data/, models/, results/)
- ✅ Git sadece kod için
- ✅ .gitignore büyük dosyalar için

---

## 🔄 FAZ 2: Veri İndirme & EDA (DEVAM EDİYOR)

**Hedef**: NASA Kepler verisini indir ve analiz et  
**Durum**: 🔄 %25 Devam Ediyor  
**Başlangıç**: 2024-11-09  
**Tahmini Bitiş**: 2024-11-15

### 2.1 Veri İndirme

- [ ] `scripts/download_nasa_data.py` implementasyonu
- [ ] NASA Exoplanet Archive API entegrasyonu
- [ ] Kepler KOI tablosunu indir (~500MB-1GB)
- [ ] Veriyi `data/raw/kepler_koi.csv` olarak kaydet
- [ ] Veri indirme logları (`results/logs/download.log`)
- [ ] Veri validasyonu (satır/sütun sayısı, veri tipleri)

**Hedef Veriler:**

```
- Kayıt Sayısı: ~9,000-10,000 KOI
- Sütun Sayısı: ~50-100 feature
- Dosya Boyutu: ~500MB (compressed), ~1GB (uncompressed)
- Format: CSV
```

### 2.2 Exploratory Data Analysis (EDA)

- [ ] `notebooks/01_exploratory_data_analysis.ipynb` oluştur
- [ ] **Veri Genel Bakış**
  - Veri boyutu (shape)
  - Veri tipleri
  - Memory kullanımı
- [ ] **Target Distribution (koi_disposition)**
  - CONFIRMED: ~2,400 (%25)
  - FALSE POSITIVE: ~3,500 (%35)
  - CANDIDATE: ~3,100 (%30)
  - Class imbalance analizi
- [ ] **Missing Values Analizi**
  - Hangi sütunlarda eksik veri var?
  - Eksik veri oranları
  - Missing pattern analizi
- [ ] **Feature Distributions**
  - Numerical features (histogram, boxplot)
  - Categorical features (bar chart)
  - Outlier detection (IQR method)
- [ ] **Correlation Analizi**
  - Correlation matrix
  - Heatmap
  - Multicollinearity kontrolü (VIF)
- [ ] **Statistical Summary**
  - Describe() çıktıları
  - Skewness, kurtosis

### 2.3 Veri Kalitesi Kontrolü

- [ ] Duplicate kayıtları kontrol et ve temizle
- [ ] Data type validation (int, float, string)
- [ ] Range checks (fiziksel anlamlılık)
  - koi_period > 0
  - koi_depth > 0
  - koi_steff (stellar temperature) makul aralıkta (2500-8000K)
- [ ] Consistency checks

### 2.4 İlk İçgörüler

- [ ] En önemli feature'ları belirle (correlation ile)
- [ ] Target ile en çok ilişkili değişkenler
- [ ] Feature engineering fikirleri not al

### Çıktılar

```
results/reports/eda_report.html          # HTML rapor
results/figures/target_distribution.png  # Grafikler
results/figures/correlation_heatmap.png
results/figures/missing_values.png
results/logs/eda.log                     # Log dosyası
```

---

## ⏳ FAZ 3: Feature Engineering (BEKLİYOR)

**Hedef**: Feature'ları oluştur, temizle, seç  
**Durum**: ⏳ Bekliyor  
**Tahmini Başlangıç**: 2024-11-16  
**Tahmini Süre**: 1-2 hafta

### 3.1 Veri Temizleme

- [ ] `src/data/cleaners.py` implementasyonu
- [ ] **Missing Value Handling**
  - Numerical: median/mean imputation
  - Categorical: mode imputation veya yeni kategori
  - Drop eşik: %70+ missing → drop column
- [ ] **Outlier Handling**
  - IQR method ile outlier detection
  - Capping/Winsorization
  - Veya robust scaler kullan
- [ ] **Data Type Conversions**
  - Object → Category (memory optimization)
  - Float64 → Float32 (memory optimization)

### 3.2 Feature Engineering

- [ ] `src/features/engineering.py` implementasyonu
- [ ] **Domain-specific Features**
  - Transit ratios (duration/period)
  - Depth ratios
  - Planetary equilibrium temperature (koi_teq)
  - Stellar parameters combinations
  - Orbital resonance indicators
- [ ] **Polynomial Features** (degree=2, select features)
- [ ] **Interaction Features**
  - koi_period × koi_depth
  - koi_duration × koi_depth
  - koi_prad × koi_insol
- [ ] **Log Transformations** (skewed distributions için)
  - log(koi_period)
  - log(koi_insol)

### 3.3 Feature Scaling

- [ ] `src/features/scalers.py` implementasyonu
- [ ] StandardScaler (normal dağılım için)
- [ ] RobustScaler (outlier'lar için)
- [ ] MinMaxScaler (alternatif)

### 3.4 Feature Selection

- [ ] `src/features/selection.py` implementasyonu
- [ ] **Filter Methods**
  - Variance threshold (low variance → drop)
  - Correlation threshold (high correlation → drop one)
  - Mutual information
- [ ] **Embedded Methods**
  - Tree-based feature importance (CatBoost)
  - L1 regularization (Lasso)
- [ ] **Wrapper Methods**
  - Recursive Feature Elimination (RFE)
  - Forward/Backward selection

**Hedef**: 50-100 feature → 20-30 en iyi feature

### 3.5 Data Splitting

- [ ] `src/data/splitters.py` implementasyonu
- [ ] Train/Validation/Test split
  - Train: 70% (~6,300)
  - Validation: 15% (~1,350)
  - Test: 15% (~1,350)
- [ ] Stratified split (class balance korunsun)
- [ ] Time-based split (eğer timestamp varsa)

### Çıktılar

```
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/feature_names.json
results/figures/feature_importance.png
```

---

## ⏳ FAZ 4: Model Development (BEKLİYOR)

**Hedef**: Baseline ve advanced modeller oluştur  
**Durum**: ⏳ Bekliyor  
**Tahmini Başlangıç**: 2024-11-23  
**Tahmini Süre**: 2 hafta

### 4.1 Baseline Model

- [ ] Logistic Regression baseline
- [ ] Dummy Classifier (majority class)
- [ ] Baseline metrics kaydet (karşılaştırma için)

### 4.2 CatBoost Model

- [ ] `src/models/catboost_model.py` implementasyonu
- [ ] Categorical feature handling (automatic)
- [ ] Initial hyperparameters
- [ ] Training pipeline
- [ ] Model checkpointing

### 4.3 LightGBM ve XGBoost

- [ ] LightGBM implementation
- [ ] XGBoost implementation
- [ ] Model comparison tablosu

### 4.4 Hyperparameter Tuning

- [ ] `src/training/hyperparameter_tuner.py` implementasyonu
- [ ] Optuna entegrasyonu
- [ ] Hyperparameter search space
- [ ] Bayesian optimization
- [ ] Grid search (opsiyonel)

**Tuning Parameters:**

```python
{
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
}
```

### 4.5 Ensemble Methods

- [ ] `src/models/ensemble.py` implementasyonu
- [ ] Voting Classifier (soft voting)
- [ ] Stacking (meta-model)
- [ ] Blending

### 4.6 Training Pipeline

- [ ] `src/training/trainer.py` implementasyonu
- [ ] Training loop with logging
- [ ] Validation loop
- [ ] Early stopping
- [ ] Learning rate scheduling
- [ ] Callbacks (progress, metrics)

### Çıktılar

```
models/experiments/catboost_baseline.pkl
models/experiments/catboost_tuned.pkl
models/experiments/lightgbm_tuned.pkl
models/experiments/xgboost_tuned.pkl
models/production/best_model.pkl
results/reports/model_comparison.html
```

---

## ⏳ FAZ 5: Model Evaluation (BEKLİYOR)

**Hedef**: Modeli değerlendir ve raporla  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 5.1 Evaluation Metrics

- [ ] `src/evaluation/metrics.py` implementasyonu
- [ ] **Classification Metrics**
  - Accuracy
  - Precision (per class)
  - Recall (per class)
  - F1-Score (per class, macro, weighted)
  - ROC-AUC (OvR, OvO)
  - PR-AUC (imbalanced data için)
- [ ] **Confusion Matrix**
  - Heatmap visualization
  - Normalized confusion matrix
- [ ] **Classification Report**

### 5.2 Cross-Validation

- [ ] `src/evaluation/validators.py` implementasyonu
- [ ] K-fold cross-validation (k=5)
- [ ] Stratified k-fold
- [ ] CV scores (mean ± std)

### 5.3 Test Set Evaluation

- [ ] Final test set üzerinde evaluation
- [ ] Per-class analysis
- [ ] Error analysis (misclassified örnekler)

### 5.4 Evaluation Reports

- [ ] `src/evaluation/reports.py` implementasyonu
- [ ] HTML report generation
- [ ] Metrics comparison table
- [ ] Training curves (loss, accuracy)
- [ ] ROC curves
- [ ] PR curves

### Çıktılar

```
results/reports/evaluation_report.html
results/figures/confusion_matrix.png
results/figures/roc_curves.png
results/figures/training_curves.png
```

**Hedef Metrikler:**

- Accuracy: >85%
- F1-Score (CONFIRMED): >80%
- ROC-AUC: >0.90

---

## ⏳ FAZ 6: Model Explainability (BEKLİYOR)

**Hedef**: SHAP ile model açıklanabilirliği  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 6.1 SHAP Implementation

- [ ] `src/explainability/shap_explainer.py` implementasyonu
- [ ] TreeExplainer (CatBoost için)
- [ ] SHAP values hesaplama (sample=100-500)
- [ ] **SHAP Plots**
  - Summary plot (feature importance)
  - Dependence plots (feature interactions)
  - Force plots (individual predictions)
  - Waterfall plots

### 6.2 Feature Importance

- [ ] `src/explainability/feature_importance.py` implementasyonu
- [ ] Built-in feature importance (model.feature*importances*)
- [ ] Permutation importance
- [ ] Feature importance visualization

### 6.3 Visualizations

- [ ] `src/explainability/visualizers.py` implementasyonu
- [ ] Interactive plots (Plotly)
- [ ] Static plots (Matplotlib)
- [ ] Export plots (PNG, HTML)

### Çıktılar

```
results/figures/shap_summary.png
results/figures/feature_importance.png
results/figures/shap_dependence_*.png
```

---

## ⏳ FAZ 7: FastAPI Development (BEKLİYOR)

**Hedef**: Production-ready REST API  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 7.1 FastAPI Setup

- [ ] `src/api/main.py` implementasyonu
- [ ] App initialization
- [ ] CORS configuration
- [ ] Logging setup
- [ ] Error handling middleware
- [ ] Request/Response models (Pydantic)

### 7.2 API Endpoints

- [ ] **Health Check**: `GET /health`
- [ ] **Predict Single**: `POST /api/v1/predict`
- [ ] **Predict Batch**: `POST /api/v1/predict/batch`
- [ ] **Explain**: `POST /api/v1/explain` (SHAP)
- [ ] **Model Info**: `GET /api/v1/model/info`

### 7.3 Request/Response Schemas

- [ ] `src/api/schemas/request.py` implementasyonu
  - PredictionRequest
  - BatchPredictionRequest
- [ ] `src/api/schemas/response.py` implementasyonu
  - PredictionResponse
  - BatchPredictionResponse
  - ExplanationResponse

### 7.4 Dependencies

- [ ] `src/api/dependencies.py` implementasyonu
- [ ] Model loading (singleton pattern)
- [ ] Feature preprocessing
- [ ] Authentication (opsiyonel)

### 7.5 API Testing

- [ ] `tests/integration/test_api.py` implementasyonu
- [ ] Endpoint tests (pytest-asyncio)
- [ ] Error handling tests
- [ ] Load tests (locust - opsiyonel)

### Çıktılar

```
API Endpoints:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)
```

---

## ⏳ FAZ 8: Streamlit Web App (BEKLİYOR)

**Hedef**: İnteraktif web arayüzü  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 8.1 Streamlit App Structure

- [ ] `src/webapp/app.py` main app
- [ ] Multi-page app setup
- [ ] Session state management
- [ ] Custom CSS/styling

### 8.2 Pages

- [ ] **Home**: `src/webapp/pages/home.py`
  - Proje açıklaması
  - Model metrikleri özeti
- [ ] **Single Prediction**: `src/webapp/pages/single_analysis.py`
  - Form ile feature girişi
  - Prediction + confidence
  - SHAP explanation
- [ ] **Batch Prediction**: `src/webapp/pages/batch_analysis.py`
  - CSV upload widget
  - Batch prediction
  - Results download
- [ ] **System Status**: `src/webapp/pages/system_status.py`
  - Model info
  - System metrics
  - API health

### 8.3 Components

- [ ] **Upload Widget**: `src/webapp/components/upload_widget.py`
- [ ] **Prediction Card**: `src/webapp/components/prediction_card.py`
- [ ] **SHAP Visualizer**: `src/webapp/components/shap_visualizer.py`
- [ ] **Data Table**: `src/webapp/components/data_table.py`

### 8.4 Styling

- [ ] Custom theme (dark mode)
- [ ] Logo ve branding
- [ ] Responsive design

### Çıktılar

```
Streamlit App: http://localhost:8501
```

---

## ⏳ FAZ 9: Testing (BEKLİYOR)

**Hedef**: Kapsamlı test coverage  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 9.1 Unit Tests

- [ ] `tests/unit/test_data_loaders.py`
- [ ] `tests/unit/test_feature_engineering.py`
- [ ] `tests/unit/test_models.py`
- [ ] `tests/unit/test_validators.py`
- [ ] `tests/unit/test_explainability.py`

### 9.2 Integration Tests

- [ ] `tests/integration/test_api.py`
- [ ] `tests/integration/test_training_pipeline.py`
- [ ] `tests/integration/test_prediction_pipeline.py`

### 9.3 E2E Tests

- [ ] `tests/e2e/test_full_workflow.py`
  - Data download → training → prediction flow

### 9.4 Test Configuration

- [ ] `tests/conftest.py` - Pytest fixtures
- [ ] Mock data fixtures
- [ ] Test utilities

**Hedef**: >80% code coverage

---

## ⏳ FAZ 10: Docker & Deployment (BEKLİYOR)

**Hedef**: Production deployment  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 10.1 Docker

- [ ] `Dockerfile` optimize et
  - Multi-stage build
  - Slim base image
  - Layer caching
- [ ] `.dockerignore` güncellenmiş
- [ ] `docker-compose.yml` tamamla
  - API service
  - Streamlit service
  - Volume mapping
- [ ] Docker image test et

### 10.2 Streamlit Cloud

- [ ] `deployment/streamlit_cloud/config.toml` yapılandır
- [ ] Secrets setup (.env variables)
- [ ] Deploy
- [ ] Custom domain (opsiyonel)

### 10.3 Railway.app (API)

- [ ] Railway project oluştur
- [ ] GitHub repository bağla
- [ ] Environment variables setup
- [ ] Deploy
- [ ] Health check monitoring

### 10.4 CI/CD Pipeline

- [ ] `.github/workflows/ci-cd.yml` tamamla
  - Automated testing
  - Linting
  - Build and deploy
- [ ] GitHub Actions secrets setup

---

## ⏳ FAZ 11: MLOps & Monitoring (BEKLİYOR)

**Hedef**: Model monitoring ve tracking  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 11.1 Model Registry

- [ ] `src/models/registry.py` implementasyonu
- [ ] Model versioning (basit JSON)
- [ ] Model metadata (metrics, params, timestamp)
- [ ] Model comparison

### 11.2 Experiment Tracking

- [ ] `src/training/experiment_tracker.py` implementasyonu
- [ ] JSON logging (basit, tek kişi için yeterli)
- [ ] Metrics tracking (accuracy, f1, etc.)
- [ ] Hyperparameters tracking
- [ ] Training time tracking

**Format:**

```json
{
  "experiment_id": "exp_001",
  "timestamp": "2024-11-20T10:30:00",
  "model_type": "CatBoost",
  "hyperparameters": {...},
  "metrics": {...},
  "training_time": 120.5
}
```

### 11.3 Monitoring (Opsiyonel)

- [ ] Model performance monitoring
- [ ] Data drift detection (basit)
- [ ] Prediction latency monitoring
- [ ] Error rate tracking

---

## ⏳ FAZ 12: Documentation (BEKLİYOR)

**Hedef**: Detaylı dokümantasyon  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 12.1 Code Documentation

- [ ] Docstrings (numpy style) - tüm functions/classes
- [ ] Type hints - tüm public functions
- [ ] Inline comments (kritik yerler)

### 12.2 Project Documentation

- [ ] `docs/architecture.md` - Sistem mimarisi
  - Diagram'lar (mermaid)
  - Component açıklamaları
- [ ] `docs/api_reference.md` - API dokümantasyonu
  - Endpoint'ler
  - Request/Response örnekleri
  - Error codes
- [ ] `docs/model_details.md` - Model detayları
  - Feature'lar
  - Hyperparameters
  - Performance metrics
- [ ] `docs/deployment.md` - Deployment rehberi
  - Docker
  - Streamlit Cloud
  - Railway
- [ ] `docs/installation.md` - Detaylı kurulum
- [ ] `docs/quickstart.md` - Hızlı başlangıç

### 12.3 MkDocs (Opsiyonel)

- [ ] MkDocs setup
- [ ] Material theme
- [ ] Deploy to GitHub Pages

---

## ⏳ FAZ 13: Final Touches (BEKLİYOR)

**Hedef**: Son iyileştirmeler ve polish  
**Durum**: ⏳ Bekliyor  
**Tahmini Süre**: 1 hafta

### 13.1 Performance Optimization

- [ ] Profiling (cProfile)
- [ ] Bottleneck analizi
- [ ] Memory optimization
- [ ] Inference speed optimization

### 13.2 Code Quality

- [ ] Final lint check (100% pass)
- [ ] Type hints completion (mypy 100%)
- [ ] Test coverage >85%
- [ ] Security audit (bandit)

### 13.3 User Experience

- [ ] Error messages iyileştirme
- [ ] Loading indicators
- [ ] Help tooltips
- [ ] User feedback mechanisms

### 13.4 Release

- [ ] VERSION file oluştur
- [ ] CHANGELOG.md oluştur
- [ ] GitHub Release (v1.0.0)
- [ ] Tag oluştur: `git tag v1.0.0`

---

## 💡 Gelecek İyileştirmeler

### Veri & Features

- [ ] Ek veri kaynakları (TESS, K2)
- [ ] Deep learning features (CNN)
- [ ] Time-series analysis

### Modeller

- [ ] AutoML (H2O.ai, AutoGluon)
- [ ] Neural Networks (TabNet)
- [ ] Ensemble optimization

### MLOps

- [ ] MLflow entegrasyonu (ileride)
- [ ] A/B testing framework
- [ ] Active learning pipeline
- [ ] Model retraining automation

### Deployment

- [ ] Kubernetes production deployment
- [ ] Auto-scaling
- [ ] Blue-green deployment
- [ ] Canary releases

---

## 📝 Notlar

### Genel Kurallar

- Her faz tamamlandığında **git commit** yapılmalı
- Major feature'lar için **branch** oluşturulmalı
- Düzenli olarak **GitHub'a push** edilmeli
- Her hafta **TODO.md güncellemesi** yapılmalı

### Commit Message Kuralları

```
feat: yeni özellik ekle
fix: bug düzeltmesi
docs: dokümantasyon güncelle
style: kod formatı (black)
refactor: kod yeniden yapılandırma
test: test ekleme/düzeltme
chore: küçük değişiklikler (typo, config)
```

### Branch Stratejisi

```
main              # Production-ready kod
├── develop       # Development branch
├── feature/*     # Yeni özellikler
├── bugfix/*      # Bug düzeltmeleri
└── hotfix/*      # Acil düzeltmeler
```

---

## 📊 Milestone'lar

| Milestone                               | Hedef Tarih | Durum           |
| --------------------------------------- | ----------- | --------------- |
| **M1**: Veri Hazırlığı (Faz 2-3)        | 2024-11-30  | 🔄 Devam Ediyor |
| **M2**: Model Geliştirme (Faz 4-6)      | 2024-12-31  | ⏳ Bekliyor     |
| **M3**: Web Servisleri (Faz 7-8)        | 2025-01-15  | ⏳ Bekliyor     |
| **M4**: Testing & Deployment (Faz 9-10) | 2025-01-31  | ⏳ Bekliyor     |
| **M5**: Production Release (v1.0.0)     | 2025-02-15  | ⏳ Bekliyor     |

---

## 🐛 Bilinen Sorunlar

_(Şu an yok - geliştirme ilerledikçe listelenecek)_

---

## 📞 Destek & İletişim

Herhangi bir sorun veya öneri için:

- **GitHub Issues**: Bug raporları ve feature request'ler
- **GitHub Discussions**: Genel sorular ve tartışmalar

---

**Son Güncelleme**: 2024-11-09  
**Güncel Faz**: Faz 2 - Veri İndirme & EDA  
**Sonraki Adım**: NASA Kepler verisini indir (`scripts/download_nasa_data.py`)

---

_Bu roadmap canlı bir dokümandır ve proje ilerledikçe güncellenecektir._
