# Architecture Documentation

## Cấu trúc dự án

```
ainfinity/
├── core/                         # Core utilities
│   ├── __init__.py
│   ├── config.py                # Global configuration
│   ├── constants.py             # Application constants
│   ├── exceptions.py            # Custom exceptions
│   ├── logging.py               # Logging utilities
│   └── utils.py                 # Utility functions
│
├── services/                     # Business logic layer
│   ├── __init__.py
│   └── training_service.py      # SkyPilot training service
│
├── api/                          # API layer
│   ├── __init__.py
│   ├── main.py                  # Main FastAPI app
│   ├── config.py                # API configuration
│   ├── dependencies.py          # Dependency injection
│   ├── exceptions.py            # Exception handlers
│   │
│   └── v1/                      # API version 1
│       ├── __init__.py          # V1 router
│       └── endpoints/           # V1 endpoints
│           ├── __init__.py
│           └── jobs.py          # Jobs endpoints
│
├── middleware/                   # Middleware layer
│   ├── __init__.py
│   ├── auth.py                  # Authentication
│   ├── cors.py                  # CORS handling
│   ├── logging.py               # Request logging
│   └── rate_limit.py            # Rate limiting
│
├── schema/                       # Data models
│   ├── __init__.py
│   └── training_job.py          # Training job schemas
│
├── train/                        # Training logic
│   ├── finetune.py
│   ├── dataset.py
│   └── ...
│
└── skylaunch/                    # SkyPilot templates
    └── deepspeed.yaml
```

## Kiến trúc Layered

### 1. **Core Layer** (`core/`)
- **Trách nhiệm**: Base utilities, configuration, exceptions
- **Components**:
  - `config.py`: Global settings
  - `constants.py`: Application constants
  - `exceptions.py`: Custom exception hierarchy
  - `logging.py`: Logging utilities
  - `utils.py`: Common utility functions
- **Nguyên tắc**:
  - Không depend vào layer nào khác
  - Reusable trong toàn bộ application

### 2. **Service Layer** (`services/`)
- **Trách nhiệm**: Business logic, external integrations
- **Ví dụ**:
  - `training_service.py`: Quản lý SkyPilot jobs
  - Future: `model_service.py`, `evaluation_service.py`
- **Nguyên tắc**:
  - Không depend vào API layer
  - Có thể test độc lập
  - Sử dụng exceptions từ core

### 3. **API Layer** (`api/`)
- **Trách nhiệm**: HTTP endpoints, request/response handling
- **Components**:
  - `main.py`: Main FastAPI application
  - `config.py`: API configuration settings
  - `dependencies.py`: Dependency injection (DI)
  - `exceptions.py`: Exception handlers
  - `v1/`: API version 1 routes

### 4. **Middleware Layer** (`middleware/`)
- **Trách nhiệm**: Request/response processing
- **Components**:
  - `auth.py`: Authentication
  - `cors.py`: CORS handling
  - `logging.py`: Request logging
  - `rate_limit.py`: Rate limiting

### 5. **API Versioning** (`api/v1/`)
- **Lý do versioning**:
  - Backwards compatibility
  - Gradual migration
  - Multiple API versions đồng thời
- **Cấu trúc**:
  - `/api/v1/jobs` - Current version
  - `/api/v2/jobs` - Future version (khi cần)

### 6. **Schema Layer** (`schema/`)
- **Trách nhiệm**: Data validation, serialization
- **Công nghệ**: Pydantic models
- **Sử dụng**: Request/Response validation

## Design Patterns

### 1. Dependency Injection
```python
# dependencies.py
@lru_cache()
def get_training_service() -> SkyPilotService:
    return SkyPilotService()

# endpoints/jobs.py
def launch_job(request: LaunchJobRequest):
    service = get_training_service()  # Injected
    return service.launch_job(request)
```

**Lợi ích**:
- Easy testing (mock dependencies)
- Singleton pattern (cached)
- Decoupling

### 2. Middleware Pattern
```python
# middleware.py
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response
```

**Use cases**:
- Logging
- Authentication
- Rate limiting
- CORS

### 3. Repository Pattern (in Service)
```python
class SkyPilotService:
    def __init__(self):
        self.jobs_db_path = ...  # Data storage

    def _load_jobs_db(self):
        # Data access logic
```

## API Versioning Strategy

### URL-based Versioning
```
/api/v1/jobs     → Version 1
/api/v2/jobs     → Version 2 (future)
```

### Adding V2
```python
# serving/api/v2/__init__.py
from fastapi import APIRouter
from ainfinity.serving.api.v2.endpoints import jobs

api_router = APIRouter()
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs-v2"])

# serving/api.py
from ainfinity.serving.api.v2 import api_router as api_v2_router
app.include_router(api_v2_router, prefix="/api/v2")
```

## Configuration Management

### Environment Variables
```bash
# .env
SERVING_HOST=0.0.0.0
SERVING_PORT=8000
SERVING_API_V1_PREFIX=/api/v1
SERVING_CORS_ORIGINS=["http://localhost:3000"]
SERVING_LOG_LEVEL=INFO
```

### Usage
```python
from ainfinity.serving.config import serving_settings

print(serving_settings.HOST)  # 0.0.0.0
print(serving_settings.PORT)  # 8000
```

## Error Handling

### Exception Hierarchy
```
BaseException
└── Exception
    ├── ValueError          → 400 Bad Request
    ├── HTTPException       → Custom status codes
    └── Exception           → 500 Internal Server Error
```

### Custom Handlers
```python
# exceptions.py
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )

# api.py
app.add_exception_handler(ValueError, value_error_handler)
```

## Testing Strategy

### Unit Tests
```python
# tests/services/test_training_service.py
def test_launch_job():
    service = SkyPilotService()
    # Test service logic
```

### Integration Tests
```python
# tests/api/test_jobs.py
from fastapi.testclient import TestClient

def test_launch_job_endpoint():
    client = TestClient(app)
    response = client.post("/api/v1/jobs", json=...)
    assert response.status_code == 201
```

## Best Practices

### 1. Separation of Concerns
- **Service layer**: Business logic
- **API layer**: HTTP handling
- **Schema layer**: Data validation

### 2. Single Responsibility
- Mỗi router handle 1 resource type
- Mỗi service handle 1 domain

### 3. Dependency Injection
- Services không biết về API
- API inject services vào endpoints

### 4. Configuration
- Environment-based config
- Type-safe với Pydantic

### 5. Error Handling
- Centralized exception handlers
- Consistent error responses

## Future Enhancements

### 1. Authentication & Authorization
```python
# dependencies.py
def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token
    return user

# endpoints/jobs.py
def launch_job(
    request: LaunchJobRequest,
    user = Depends(get_current_user)
):
    # Only authenticated users
```

### 2. Database Integration
```python
# services/training_service.py
from sqlalchemy import create_engine

class SkyPilotService:
    def __init__(self, db_session):
        self.db = db_session
```

### 3. Async Operations
```python
# services/training_service.py
async def launch_job(self, request):
    # Async SkyPilot operations
    result = await asyncio.create_subprocess_exec(...)
```

### 4. WebSocket for Real-time Updates
```python
# endpoints/jobs.py
@router.websocket("/ws/jobs/{job_name}")
async def job_updates(websocket: WebSocket, job_name: str):
    # Stream job status updates
```

### 5. Background Tasks
```python
from fastapi import BackgroundTasks

@router.post("/jobs")
async def launch_job(
    request: LaunchJobRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(monitor_job, request.job_name)
```

## Migration Guide

### From Old Structure
```
serving/
├── api.py              → serving/api.py (updated)
└── skypilot_service.py → services/training_service.py
```

### Update Imports
```python
# Old
from ainfinity.serving.skypilot_service import SkyPilotService

# New
from ainfinity.services import SkyPilotService
```

### Update URLs
```python
# Old
POST /jobs

# New
POST /api/v1/jobs
```
