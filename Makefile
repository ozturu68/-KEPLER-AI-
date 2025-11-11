# ============================================
# EXOPLANET ML PROJECT - MAKEFILE (DVC'SİZ)
# ============================================
# Proje akışı için profesyonel Makefile
# Strateji: Güçlü Yerel (1TB SSD) + Bulut Dağıtımı

.PHONY: help setup install install-dev update clean clean-all \
        test test-unit test-integration test-e2e test-cov test-watch \
        lint lint-fix format format-check type-check security-check \
        pre-commit pre-commit-update \
        run-api run-webapp run-jupyter notebook-to-script \
        train evaluate predict download-data \
        docker-build docker-run docker-push docker-clean \
        git-setup git-status git-log \
        docs docs-serve \
        deps-tree deps-update deps-check \
        profile benchmark \
        ci all

# ============================================
# VARIABLES & CONFIGURATION
# ============================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Python
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

# Project paths
SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs
NOTEBOOKS_DIR := notebooks

# Docker
DOCKER_IMAGE := exoplanet-ml
DOCKER_TAG := latest
DOCKER_REGISTRY :=

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color

# ============================================
# HELP & DOCUMENTATION
# ============================================
help: ## Bu yardım menüsünü gösterir
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║        EXOPLANET ML PROJESİ - MAKEFILE KOMUTLARI        ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)📦 Kurulum (Setup & Installation):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /setup|install|update/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🧹 Temizlik (Cleaning):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /clean/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🧪 Test (Testing):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /^test/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🔍 Kod Kalitesi (Code Quality):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /lint|format|type|security|pre-commit/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🚀 Servisleri Çalıştırma (Running Services):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /^run-|notebook/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🤖 ML Operasyonları (ML Operations):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /train|evaluate|predict|download/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🐳 Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /^docker-/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)📚 Dokümantasyon (Documentation):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /^docs/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🔧 Yardımcı Araçlar (Utilities):$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /git-|deps-|profile|benchmark|ci|all/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================
# SETUP & INSTALLATION
# ============================================
setup: ## Proje kurulumunu tamamla (venv + install-dev)
	@echo "$(CYAN)🚀 Proje kurulumu başlıyor...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)📦 Sanal ortam (venv) oluşturuluyor...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)✓ Sanal ortam oluşturuldu.$(NC)"; \
	else \
		echo "$(GREEN)✓ Sanal ortam zaten mevcut.$(NC)"; \
	fi
	@echo "$(YELLOW)📥 Bağımlılıklar kuruluyor...$(NC)"
	@$(MAKE) install-dev
	@echo "$(GREEN)✓ Kurulum tamamlandı!$(NC)"
	@echo "$(BLUE)💡 Ortamı aktive etmek için 'source venv/bin/activate' komutunu çalıştırın.$(NC)"

install: ## Sadece üretim (production) bağımlılıklarını kur
	@echo "$(CYAN)📦 Üretim bağımlılıkları kuruluyor...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Üretim bağımlılıkları kuruldu.$(NC)"

install-dev: ## Geliştirme bağımlılıklarını kur (üretimi içerir)
	@echo "$(CYAN)📦 Geliştirme bağımlılıkları kuruluyor...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements-dev.txt
	@$(PIP) install -e .
	@$(VENV_BIN)/pre-commit install
	@echo "$(GREEN)✓ Geliştirme bağımlılıkları kuruldu.$(NC)"
	@echo "$(GREEN)✓ Pre-commit kancaları kuruldu.$(NC)"

update: ## Tüm bağımlılıkları son versiyonlara güncelle
	@echo "$(CYAN)🔄 Bağımlılıklar güncelleniyor...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install --upgrade -r requirements.txt
	@$(PIP) install --upgrade -r requirements-dev.txt
	@echo "$(GREEN)✓ Bağımlılıklar güncellendi.$(NC)"

# ============================================
# CLEANING
# ============================================
clean: ## Python önbellek dosyalarını ve build artıklarını temizle
	@echo "$(CYAN)🧹 Proje temizleniyor...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage 2>/dev/null || true
	@rm -rf build dist 2>/dev/null || true
	@echo "$(GREEN)✓ Proje temizlendi.$(NC)"

clean-all: clean ## Venv ve tüm cache'ler dahil her şeyi temizle
	@echo "$(RED)⚠️  UYARI: Bu komut venv, modeller ve logları silecek!$(NC)"
	@read -p "Emin misiniz? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(CYAN)🧹 Derin temizlik yapılıyor...$(NC)"; \
		rm -rf $(VENV) 2>/dev/null || true; \
		rm -rf models/experiments/* 2>/dev/null || true; \
		rm -rf results/logs/* results/figures/* 2>/dev/null || true; \
		echo "$(GREEN)✓ Derin temizlik tamamlandı.$(NC)"; \
	else \
		echo "$(YELLOW)İptal edildi.$(NC)"; \
	fi

# ============================================
# TESTING
# ============================================
test: ## Tüm testleri çalıştır (unit, integration, e2e)
	@echo "$(CYAN)🧪 Tüm testler çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pytest $(TEST_DIR)/ -v

test-unit: ## Sadece birim (unit) testlerini çalıştır
	@echo "$(CYAN)🧪 Birim testleri çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pytest $(TEST_DIR)/unit/ -v -m unit

test-integration: ## Sadece entegrasyon (integration) testlerini çalıştır
	@echo "$(CYAN)🧪 Entegrasyon testleri çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pytest $(TEST_DIR)/integration/ -v -m integration

test-e2e: ## Sadece uçtan uca (e2e) testleri çalıştır
	@echo "$(CYAN)🧪 E2E testleri çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pytest $(TEST_DIR)/e2e/ -v -m e2e

test-cov: ## Testleri kod kapsamı (coverage) raporu ile çalıştır
	@echo "$(CYAN)🧪 Testler ve kod kapsamı raporu çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pytest $(TEST_DIR)/ -v \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term \
		--cov-report=xml
	@echo "$(GREEN)✓ Kapsam raporu oluşturuldu: htmlcov/index.html$(NC)"

test-watch: ## Dosya değişimlerini izleyerek testleri yeniden çalıştır
	@echo "$(CYAN)🧪 Testler 'watch' modunda çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/ptw $(TEST_DIR)/ -- -v

# ============================================
# CODE QUALITY
# ============================================
lint: format-check type-check security-check ## Tüm denetleyicileri (lint) çalıştır
	@echo "$(CYAN)🔍 Tüm kod kalitesi denetleyicileri çalıştırılıyor...$(NC)"
	@echo "$(YELLOW)  → Flake8 (Stil Denetimi)...$(NC)"
	@$(VENV_BIN)/flake8 $(SRC_DIR)/ $(TEST_DIR)/ || true
	@echo "$(YELLOW)  → Pylint (Derin Analiz)...$(NC)"
	@$(VENV_BIN)/pylint $(SRC_DIR)/ || true
	@echo "$(GREEN)✓ Tüm denetimler tamamlandı.$(NC)"

lint-fix: format ## Lint sorunlarını otomatik düzelt (format alias'ı)

format: ## Kodu black ve isort ile otomatik formatla
	@echo "$(CYAN)🎨 Kod formatlanıyor (black, isort)...$(NC)"
	@$(VENV_BIN)/black $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/
	@$(VENV_BIN)/isort $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/
	@echo "$(GREEN)✓ Kod formatlandı.$(NC)"

format-check: ## Kod formatını (black, isort) kontrol et
	@echo "$(CYAN)🎨 Kod formatı kontrol ediliyor...$(NC)"
	@$(VENV_BIN)/black --check $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/
	@$(VENV_BIN)/isort --check $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/

type-check: ## MyPy ile statik tip kontrolü yap
	@echo "$(CYAN)🔍 Tip kontrolü (MyPy) yapılıyor...$(NC)"
	@$(VENV_BIN)/mypy $(SRC_DIR)/

security-check: ## Bandit ile güvenlik açığı tara (src ve scripts)
	@echo "$(CYAN)🔒 Güvenlik açığı taraması (Bandit) yapılıyor...$(NC)"
	@$(VENV_BIN)/bandit -r $(SRC_DIR)/ $(SCRIPTS_DIR)/ -c pyproject.toml

pre-commit: ## Tüm dosyalarda pre-commit kancalarını çalıştır
	@echo "$(CYAN)🪝 Pre-commit kancaları çalıştırılıyor...$(NC)"
	@$(VENV_BIN)/pre-commit run --all-files

pre-commit-update: ## Pre-commit kancalarını güncelle
	@echo "$(CYAN)🔄 Pre-commit kancaları güncelleniyor...$(NC)"
	@$(VENV_BIN)/pre-commit autoupdate

# ============================================
# RUNNING SERVICES
# ============================================
run-api: ## FastAPI sunucusunu 'development' modunda başlat
	@echo "$(CYAN)🚀 FastAPI sunucusu başlatılıyor...$(NC)"
	@echo "$(BLUE)   API: http://localhost:8000$(NC)"
	@echo "$(BLUE)   API Dokümanı: http://localhost:8000/docs$(NC)"
	@$(VENV_BIN)/uvicorn $(SRC_DIR).api.main:app --reload --host 0.0.0.0 --port 8000

run-webapp: ## Streamlit web uygulamasını başlat
	@echo "$(CYAN)🚀 Streamlit web uygulaması başlatılıyor...$(NC)"
	@echo "$(BLUE)   Uygulama: http://localhost:8501$(NC)"
	@$(VENV_BIN)/streamlit run $(SRC_DIR)/webapp/app.py

run-jupyter: ## Jupyter Lab'i başlat
	@echo "$(CYAN)🚀 Jupyter Lab başlatılıyor...$(NC)"
	@echo "$(BLUE)   Jupyter: http://localhost:8888$(NC)"
	@$(VENV_BIN)/jupyter lab --notebook-dir=$(NOTEBOOKS_DIR)

notebook-to-script: ## Jupyter notebook'larını Python script'lerine dönüştür
	@echo "$(CYAN)📓 Notebook'lar script'e dönüştürülüyor...$(NC)"
	@for notebook in $(NOTEBOOKS_DIR)/*.ipynb; do \
		echo "  Dönüştürülüyor: $$notebook"; \
		$(VENV_BIN)/jupyter nbconvert --to script "$$notebook"; \
	done
	@echo "$(GREEN)✓ Notebook'lar dönüştürüldü.$(NC)"

# ============================================
# ML OPERATIONS
# ============================================
train: ## Modeli eğit (scripts/train_model.py)
	@echo "$(CYAN)🤖 Model eğitiliyor...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/train_model.py

evaluate: ## Eğitilmiş modeli değerlendir (scripts/evaluate_model.py)
	@echo "$(CYAN)📊 Model değerlendiriliyor...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/evaluate_model.py

predict: ## Toplu tahmin (batch prediction) yap (scripts/batch_predict.py)
	@echo "$(CYAN)🔮 Tahminler çalıştırılıyor...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/batch_predict.py

download-data: ## NASA Kepler verisini indir (scripts/download_nasa_data.py)
	@echo "$(CYAN)📥 NASA verisi indiriliyor...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/download_nasa_data.py

# ============================================
# DOCKER
# ============================================
docker-build: ## Docker imajını build et
	@echo "$(CYAN)🐳 Docker imajı build ediliyor...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✓ Docker imajı build edildi: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-run: ## Docker container'ını çalıştır
	@echo "$(CYAN)🐳 Docker container çalıştırılıyor...$(NC)"
	@docker run -it --rm \
		-p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		--name exoplanet-ml-container \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-push: ## Docker imajını registry'ye push'la
	@echo "$(CYAN)🐳 Docker imajı registry'ye push'lanıyor...$(NC)"
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "$(RED)❌ DOCKER_REGISTRY değişkeni ayarlanmamış.$(NC)"; \
		exit 1; \
	fi
	@docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-clean: ## Docker container ve imajlarını temizle
	@echo "$(CYAN)🐳 Docker kaynakları temizleniyor...$(NC)"
	@docker container prune -f
	@docker image prune -f
	@echo "$(GREEN)✓ Docker kaynakları temizlendi.$(NC)"

# ============================================
# GIT UTILITIES
# ============================================
git-setup: ## Bu proje için git ayarlarını yapılandır
	@echo "$(CYAN)🔧 Git yapılandırılıyor...$(NC)"
	@git config --local core.editor "nano"
	@git config --local pull.rebase false
	@echo "$(GREEN)✓ Git yapılandırıldı.$(NC)"

git-status: ## Git durumunu özetle göster
	@echo "$(CYAN)📊 Git Durumu:$(NC)"
	@git status -sb

git-log: ## Son 10 commit'i formatlı göster
	@echo "$(CYAN)📜 Son commit'ler:$(NC)"
	@git log --oneline --graph --decorate -10

# ============================================
# DOCUMENTATION
# ============================================
docs: ## Dokümantasyonu (MkDocs) build et
	@echo "$(CYAN)📚 Dokümantasyon build ediliyor...$(NC)"
	@cd $(DOCS_DIR) && $(VENV_BIN)/mkdocs build
	@echo "$(GREEN)✓ Dokümantasyon build edildi: $(DOCS_DIR)/site/index.html$(NC)"

docs-serve: ## Dokümantasyonu yerel olarak sun
	@echo "$(CYAN)📚 Dokümantasyon sunuluyor...$(NC)"
	@echo "$(BLUE)   Doküman: http://localhost:8000$(NC)"
	@cd $(DOCS_DIR) && $(VENV_BIN)/mkdocs serve

# ============================================
# DEPENDENCY MANAGEMENT
# ============================================
deps-tree: ## Bağımlılık ağacını göster
	@echo "$(CYAN)🌳 Bağımlılık ağacı:$(NC)"
	@$(PIP) install pipdeptree 2>/dev/null || true
	@$(VENV_BIN)/pipdeptree

deps-update: ## Eski (outdated) bağımlılıkları kontrol et
	@echo "$(CYAN)🔄 Eski paketler kontrol ediliyor...$(NC)"
	@$(PIP) list --outdated

deps-check: ## 'safety' ile güvenlik açığı kontrolü yap
	@echo "$(CYAN)🔒 Bağımlılıklar güvenlik açığı için taranıyor...$(NC)"
	@$(PIP) install safety 2>/dev/null || true
	@$(VENV_BIN)/safety check

# ============================================
# PERFORMANCE & PROFILING
# ============================================
profile: ## cProfile ile kod performansını profille
	@echo "$(CYAN)⚡ Kod profili oluşturuluyor...$(NC)"
	@$(PYTHON_VENV) -m cProfile -o profile.stats $(SCRIPTS_DIR)/train_model.py
	@$(PYTHON_VENV) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

benchmark: ## 'pytest-benchmark' ile performans testi yap
	@echo "$(CYAN)⚡ Benchmark testleri çalıştırılıyor...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/ --benchmark-only -v

# ============================================
# CI/CD & AUTOMATION
# ============================================
ci: lint test-cov ## CI (Sürekli Entegrasyon) hattını çalıştır
	@echo "$(GREEN)✓ CI hattı başarıyla tamamlandı! (lint + test-cov)$(NC)"

all: clean install-dev lint test ## Tüm proje akışını (temizle, kur, denetle, test et) çalıştır
	@echo "$(GREEN)✓ Tüm akış başarıyla tamamlandı!$(NC)"

# ============================================
# SPECIAL TARGETS
# ============================================
.SILENT: help
