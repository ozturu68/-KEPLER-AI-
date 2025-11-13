# 🌐 API Documentation

**Last Updated:** 2025-11-13  
**Status:** 🚧 In Development

## Overview

Kepler Exoplanet ML API REST endpoints for model inference, health checks, and explainability.

## Base URL

```
Development: http://localhost:8000
Production:  https://api.kepler-ml.example.com
```

## Authentication

Currently not implemented. Planned for Phase 10.

---

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-13T15:05:13Z"
}
```

**Status Codes:**

- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is down

---

### Predict Single

**POST** `/predict`

Make prediction for a single exoplanet candidate.

**Request Body:**

```json
{
  "features": {
    "koi_period": 3.52475,
    "koi_prad": 2.69,
    "koi_teq": 1517,
    "koi_insol": 141.28,
    "koi_steff": 5853,
    "koi_slogg": 4.467,
    "koi_srad": 0.927,
    "koi_smass": 0.94
  },
  "return_proba": true,
  "explain": false
}
```

**Response:**

```json
{
  "prediction": "CONFIRMED",
  "prediction_label": 0,
  "probabilities": {
    "CANDIDATE": 0.05,
    "CONFIRMED": 0.92,
    "FALSE POSITIVE": 0.03
  },
  "confidence": 0.92,
  "model_version": "catboost_v1.2.0",
  "timestamp": "2025-11-13T15:05:13Z"
}
```

**Status Codes:**

- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Model error

---

### Predict Batch

**POST** `/predict/batch`

Make predictions for multiple candidates.

**Request Body:**

```json
{
  "data": [
    {
      "koi_period": 3.52475,
      "koi_prad": 2.69,
      ...
    },
    {
      "koi_period": 10.15,
      "koi_prad": 1.2,
      ...
    }
  ],
  "return_proba": true
}
```

**Response:**

```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": "CONFIRMED",
      "probability": 0.92
    },
    {
      "index": 1,
      "prediction": "CANDIDATE",
      "probability": 0.65
    }
  ],
  "total": 2,
  "model_version": "catboost_v1.2.0",
  "timestamp": "2025-11-13T15:05:13Z"
}
```

---

### Explain Prediction

**POST** `/explain`

Get SHAP explanation for a prediction.

**Request Body:**

```json
{
  "features": {
    "koi_period": 3.52475,
    "koi_prad": 2.69,
    ...
  },
  "plot_type": "waterfall"
}
```

**Response:**

```json
{
  "prediction": "CONFIRMED",
  "shap_values": {
    "koi_period": 0.15,
    "koi_prad": 0.32,
    "koi_teq": -0.08,
    ...
  },
  "base_value": 0.33,
  "top_features": [
    {"name": "koi_prad", "contribution": 0.32},
    {"name": "koi_period", "contribution": 0.15}
  ],
  "plot_url": "/static/shap_plots/abc123.png"
}
```

---

### Model Info

**GET** `/model/info`

Get loaded model information.

**Response:**

```json
{
  "model_name": "CatBoostExoplanetClassifier",
  "version": "1.2.0",
  "created_at": "2025-11-01T10:30:00Z",
  "trained_on": "Kepler Dataset v7",
  "n_features": 50,
  "feature_names": ["koi_period", "koi_prad", ...],
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
  "metrics": {
    "train_accuracy": 0.94,
    "val_accuracy": 0.91,
    "test_accuracy": 0.90
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid feature values",
    "details": {
      "koi_period": "Must be positive"
    }
  },
  "timestamp": "2025-11-13T15:05:13Z"
}
```

### Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `MODEL_ERROR`: Model inference failed
- `NOT_FOUND`: Resource not found
- `INTERNAL_ERROR`: Internal server error

---

## Rate Limiting

- **Development:** No limits
- **Production:** 100 requests/minute per IP

---

## SDK Examples

### Python

```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Make prediction
data = {
    "features": {
        "koi_period": 3.52475,
        "koi_prad": 2.69,
        "koi_teq": 1517,
        "koi_insol": 141.28,
        "koi_steff": 5853,
        "koi_slogg": 4.467,
        "koi_srad": 0.927,
        "koi_smass": 0.94
    },
    "return_proba": True
}

response = requests.post(f"{base_url}/predict", json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "koi_period": 3.52475,
      "koi_prad": 2.69,
      "koi_teq": 1517,
      "koi_insol": 141.28,
      "koi_steff": 5853,
      "koi_slogg": 4.467,
      "koi_srad": 0.927,
      "koi_smass": 0.94
    }
  }'
```

---

## WebSocket API (Planned)

Real-time prediction streaming for Phase 11.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predict');

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Prediction:', result);
};

ws.send(JSON.stringify({
  features: { ... }
}));
```

---

## API Versioning

API version is included in response headers:

```
X-API-Version: 1.0.0
```

Breaking changes will use URL versioning:

```
/v1/predict
/v2/predict
```

---

## Status: 🚧 In Development

Current implementation status:

- [x] API design complete
- [ ] FastAPI implementation (Phase 10)
- [ ] Endpoint tests
- [ ] OpenAPI/Swagger docs
- [ ] Rate limiting
- [ ] Authentication
- [ ] WebSocket support
