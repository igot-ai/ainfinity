# Project Structure Overview

## 📁 New Structure

```
ainfinity/
│
├── core/                            # 🎯 Core Layer
│   ├── __init__.py
│   ├── config.py                   # Global configuration
│   ├── constants.py                # Application constants
│   ├── exceptions.py               # Custom exceptions
│   ├── logging.py                  # Logging utilities
│   └── utils.py                    # Utility functions
│
├── services/                        # 🔧 Service Layer (Business Logic)
│   ├── __init__.py
│   └── training_service.py         # SkyPilot training orchestration
│
├── api/                             # 🌐 API Layer
│   ├── __init__.py
│   ├── main.py                     # Main FastAPI application
│   ├── config.py                   # API configuration
│   ├── dependencies.py             # Dependency injection
│   ├── exceptions.py               # Exception handlers
│   │
│   └── v1/                         # API Version 1
│       ├── __init__.py             # V1 router
│       └── endpoints/              # V1 endpoints
│           ├── __init__.py
│           └── jobs.py            # Jobs CRUD endpoints
│
├── middleware/                      # 🛡️ Middleware Layer
│   ├── __init__.py
│   ├── auth.py                     # Authentication
│   ├── cors.py                     # CORS handling
│   ├── logging.py                  # Request logging
│   └── rate_limit.py               # Rate limiting
│
├── schema/                          # 📋 Data Models
│   ├── __init__.py
│   └── training_job.py             # Pydantic models
│
├── train/                           # 🎓 Training Scripts
│   ├── finetune.py
│   ├── dataset.py
│   └── ...
│
└── skylaunch/                       # ☁️ SkyPilot Templates
    └── deepspeed.yaml

examples/
└── training_client.py               # 📚 Usage Examples

docs/
└── ARCHITECTURE.md                  # 📖 Architecture Guide
```

## 🎯 Key Changes

### 1. **Service Layer** (`services/`)
- **Trước**: `serving/skypilot_service.py`
- **Sau**: `services/training_service.py`
- **Lý do**: Tách biệt business logic khỏi API layer

### 2. **API Versioning** (`serving/api/v1/`)
- **Endpoints**: `/api/v1/*`
- **Benefit**: Backwards compatibility, easy upgrades
- **Structure**:
  ```
  api/v1/endpoints/
    ├── jobs.py      # Current endpoints
    └── models.py    # Future endpoints
  ```

### 3. **Dependencies & Middleware**
- **dependencies.py**: Dependency injection pattern
- **middleware.py**: Request/response processing
- **exceptions.py**: Centralized error handling
- **config.py**: Environment-based configuration

## 🔄 Migration Changes

### Import Changes
```python
# ❌ Old
from ainfinity.serving.skypilot_service import SkyPilotService
from ainfinity.serving.api import app

# ✅ New
from ainfinity.services import SkyPilotService
from ainfinity.api import app
from ainfinity.middleware import LoggingMiddleware
```

### URL Changes
```python
# ❌ Old
POST /jobs
GET  /jobs/{name}

# ✅ New
POST /api/v1/jobs
GET  /api/v1/jobs/{name}
```
### 1. Start API Server
```bash
# Development mode with auto-reload
python -m ainfinity.api.main

# Or with environment variables
SERVING_PORT=8080 SERVING_RELOAD=true python -m ainfinity.api.main
```
# Or with environment variables
SERVING_PORT=8080 SERVING_RELOAD=true python -m ainfinity.serving.api
```

### 2. API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### 3. Example Usage
```python
from examples.training_client import TrainingClient

client = TrainingClient("http://localhost:8000")
result = client.launch_job({
    "job_name": "my-training",
    "resources": {"infra": "vast", "accelerators": "RTX3090:1"},
    "training": {"config_file": "finetuning"}
})
```

## 🏗️ Architecture Principles

### 1. **Separation of Concerns**
```
User Request → Middleware → API Layer → Service Layer → External Systems
              (Auth, Log)  (HTTP)      (Business)     (SkyPilot, DB)
```

### 2. **Dependency Injection**
```python
# Service không biết về API
class SkyPilotService:
    def launch_job(self, request): ...

# API inject service vào endpoints
def launch_job_endpoint(request):
    service = get_training_service()  # Injected
    return service.launch_job(request)
```

### 3. **Versioning Strategy**
- v1: Current stable API
- v2: Future enhancements (when needed)
- Backward compatibility maintained

## 📝 Configuration

### Environment Variables
```bash
# .env file
SERVING_HOST=0.0.0.0
SERVING_PORT=8000
SERVING_API_V1_PREFIX=/api/v1
SERVING_RELOAD=false
SERVING_LOG_LEVEL=INFO
SERVING_CORS_ORIGINS=["*"]
```

### Usage in Code
```python
from ainfinity.serving.config import serving_settings

print(serving_settings.PORT)  # 8000
print(serving_settings.API_V1_PREFIX)  # /api/v1
```

## 🧪 Testing

### Unit Tests (Service Layer)
```python
def test_launch_job():
    service = SkyPilotService()
    result = service.launch_job(request)
    assert result.status == "pending"
```

### Integration Tests (API Layer)
```python
from fastapi.testclient import TestClient

def test_api_launch_job():
    client = TestClient(app)
    response = client.post("/api/v1/jobs", json=...)
    assert response.status_code == 201
```

## 📚 Further Reading

- **API Documentation**: `ainfinity/serving/README.md`
- **Architecture Guide**: `docs/ARCHITECTURE.md`
- **Example Client**: `examples/training_client.py`

## 🎉 Benefits

### ✅ Scalability
- Easy to add new services
- Easy to add new API versions
- Modular architecture

### ✅ Maintainability
- Clear separation of concerns
- Single responsibility principle
- Type-safe with Pydantic

### ✅ Testability
- Services can be tested independently
- Mock dependencies easily
- Clear boundaries

### ✅ Extensibility
- Add v2 API without breaking v1
- Add new endpoints easily
- Add new middleware/handlers

## 🔜 Future Enhancements

1. **Authentication & Authorization**
   - JWT tokens
   - Role-based access control

2. **Database Integration**
   - SQLAlchemy models
   - Alembic migrations

3. **Async Operations**
   - Background tasks
   - WebSocket for real-time updates

4. **Monitoring & Logging**
   - Structured logging
   - Metrics collection
   - APM integration

5. **Additional Services**
   - Model service
   - Evaluation service
   - Dataset service
