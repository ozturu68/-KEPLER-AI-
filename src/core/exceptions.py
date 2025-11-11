"""
Proje için özel exception (istisna) sınıfları.

Bu modül, projeye özel hata durumlarını yakalamak ve işlemek için
custom exception sınıfları içerir.
"""


class ExoplanetMLError(Exception):
    """Tüm proje exception'larının base sınıfı."""
    def __init__(self, message: str = "Exoplanet ML projesinde bir hata oluştu"):
        self.message = message
        super().__init__(self.message)


class DataError(ExoplanetMLError):
    """Veri işleme hatalarının base sınıfı."""
    pass


class DataNotFoundError(DataError):
    """Veri dosyası bulunamadı hatası."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Veri dosyası bulunamadı: {file_path}"
        super().__init__(message)


class DataValidationError(DataError):
    """Veri doğrulama (validation) hatası."""
    def __init__(self, validation_message: str):
        message = f"Veri doğrulama başarısız: {validation_message}"
        super().__init__(message)


class DataDownloadError(DataError):
    """Veri indirme hatası."""
    def __init__(self, reason: str):
        message = f"Veri indirme başarısız: {reason}"
        super().__init__(message)


class EmptyDataError(DataError):
    """Boş veri hatası."""
    def __init__(self, data_name: str = "Dataset"):
        message = f"{data_name} boş veya satır içermiyor"
        super().__init__(message)


class MissingColumnsError(DataError):
    """Eksik sütun hatası."""
    def __init__(self, missing_columns: list):
        self.missing_columns = missing_columns
        message = f"Gerekli sütunlar eksik: {', '.join(missing_columns)}"
        super().__init__(message)


class ModelError(ExoplanetMLError):
    """Model ile ilgili hatalarının base sınıfı."""
    pass


class ModelNotFoundError(ModelError):
    """Model dosyası bulunamadı hatası."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        message = f"Model dosyası bulunamadı: {model_path}"
        super().__init__(message)


class ModelTrainingError(ModelError):
    """Model eğitim hatası."""
    def __init__(self, reason: str):
        message = f"Model eğitimi başarısız: {reason}"
        super().__init__(message)


class ModelPredictionError(ModelError):
    """Model tahmin hatası."""
    def __init__(self, reason: str):
        message = f"Model tahmini başarısız: {reason}"
        super().__init__(message)


class ModelLoadError(ModelError):
    """Model yükleme hatası."""
    def __init__(self, model_path: str, reason: str):
        message = f"Model yüklenemedi ({model_path}): {reason}"
        super().__init__(message)


class ModelSaveError(ModelError):
    """Model kaydetme hatası."""
    def __init__(self, model_path: str, reason: str):
        message = f"Model kaydedilemedi ({model_path}): {reason}"
        super().__init__(message)


class FeatureEngineeringError(ExoplanetMLError):
    """Feature engineering hatalarının base sınıfı."""
    pass


class FeatureSelectionError(FeatureEngineeringError):
    """Feature seçim hatası."""
    def __init__(self, reason: str):
        message = f"Feature seçimi başarısız: {reason}"
        super().__init__(message)


class ScalingError(FeatureEngineeringError):
    """Scaling (ölçeklendirme) hatası."""
    def __init__(self, reason: str):
        message = f"Feature scaling başarısız: {reason}"
        super().__init__(message)


class APIError(ExoplanetMLError):
    """API ile ilgili hatalarının base sınıfı."""
    pass


class InvalidRequestError(APIError):
    """Geçersiz request hatası."""
    def __init__(self, reason: str):
        message = f"Geçersiz request: {reason}"
        super().__init__(message)


class PredictionServiceError(APIError):
    """Tahmin servisi hatası."""
    def __init__(self, reason: str):
        message = f"Tahmin servisi hatası: {reason}"
        super().__init__(message)


class ConfigError(ExoplanetMLError):
    """Konfigürasyon hatalarının base sınıfı."""
    pass


class ConfigNotFoundError(ConfigError):
    """Konfigürasyon dosyası bulunamadı hatası."""
    def __init__(self, config_path: str):
        message = f"Konfigürasyon dosyası bulunamadı: {config_path}"
        super().__init__(message)


class InvalidConfigError(ConfigError):
    """Geçersiz konfigürasyon hatası."""
    def __init__(self, reason: str):
        message = f"Geçersiz konfigürasyon: {reason}"
        super().__init__(message)


class ValidationError(ExoplanetMLError):
    """Doğrulama hatalarının base sınıfı."""
    pass


class SchemaValidationError(ValidationError):
    """Şema doğrulama hatası."""
    def __init__(self, field: str, reason: str):
        message = f"Şema doğrulama hatası ({field}): {reason}"
        super().__init__(message)


class RangeValidationError(ValidationError):
    """Değer aralığı doğrulama hatası."""
    def __init__(self, field: str, value: float, min_val: float, max_val: float):
        message = (
            f"Değer aralık dışı: {field}={value} "
            f"(geçerli aralık: {min_val} - {max_val})"
        )
        super().__init__(message)