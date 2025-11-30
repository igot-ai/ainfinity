# AIFininity Training Service - Architecture

## Cấu trúc thư mục

```
ainfinity/
├── services/                    # Business logic layer
│   ├── __init__.py
│   └── training_service.py      # SkyPilot training service
│
├── serving/                     # API/Presentation layer
│   ├── __init__.py
│   ├── api.py                   # Main FastAPI application
│   ├── config.py                # Serving configuration
│   ├── dependencies.py          # Dependency injection
│   ├── exceptions.py            # Exception handlers
│   │
│   ├── api/                     # API versioning
│   │   ├── __init__.py
│   │   └── v1/                  # API version 1
│   │       ├── __init__.py
│   │       └── endpoints/       # Route handlers
│   │           ├── __init__.py
│   │           └── jobs.py      # Jobs endpoints
│   │
│   └── middleware/              # HTTP middleware
│       ├── __init__.py
│       ├── auth.py              # Authentication
│       ├── cors.py              # CORS handling
│       ├── logging.py           # Request logging
│       └── rate_limit.py        # Rate limiting
│
└── schema/                      # Data models
    ├── __init__.py
    └── training_job.py          # Job schemas
```

## Kiến trúc phân tầng

### 1. **Schema Layer** (`schema/`)
- Định nghĩa data models với Pydantic
- Request/Response schemas
- Validation logic

### 2. **Service Layer** (`services/`)
- Business logic thuần túy
- Tương tác với SkyPilot CLI
- Quản lý job state
- Không phụ thuộc vào FastAPI

### 3. **Serving Layer** (`serving/`)
- API endpoints (HTTP handlers)
- Request validation
- Response formatting
- Middleware
- Exception handling

## Dependency Flow

```
api.py
  ├─> api/v1/endpoints/jobs.py
  │     └─> dependencies.get_training_service()
  │           └─> services/training_service.py
  │                 └─> schema/training_job.py
  │
  ├─> middleware/*
  ├─> exceptions.py
  └─> config.py
```

## API Versioning

Sử dụng URL-based versioning:
- `/api/v1/jobs` - Version 1 của API
- Future: `/api/v2/jobs` - Version 2

Mỗi version có router riêng trong `serving/api/v{version}/`

## Middleware Stack

Thứ tự xử lý request (từ ngoài vào trong):

1. **CORSMiddleware** - Xử lý CORS headers
2. **LoggingMiddleware** - Log requests/responses
3. **RateLimitMiddleware** (optional) - Giới hạn số request
4. **AuthenticationMiddleware** (optional) - Xác thực API key
5. **Exception Handlers** - Xử lý errors
6. **Router** - Xử lý endpoints

## Configuration

Configuration được quản lý qua:
- `serving/config.py` - Serving settings
- `ainfinity/config.py` - Global settings
- Environment variables với prefix `SERVING_`

## Best Practices

### 1. Separation of Concerns
- **Services**: Business logic, không biết về HTTP/FastAPI
- **Endpoints**: Chỉ xử lý HTTP, gọi services
- **Schemas**: Data validation và serialization

### 2. Dependency Injection
- Sử dụng `dependencies.py` để inject services
- Dễ dàng mock trong tests
- Singleton pattern cho services

### 3. Error Handling
- Custom exception handlers trong `exceptions.py`
- Consistent error response format
- Proper HTTP status codes

### 4. Extensibility
- Dễ dàng thêm endpoints mới trong `api/v1/endpoints/`
- Thêm middleware mới trong `middleware/`
- Thêm API version mới trong `api/v2/`

## Running the Application

```bash
# Development
python -m ainfinity.serving.api

# Production with uvicorn
uvicorn ainfinity.serving.api:app --host 0.0.0.0 --port 8000

# With environment variables
SERVING_API_TITLE="My Service" \
SERVING_PORT=8080 \
python -m ainfinity.serving.api
```

## Adding New Features

### Add new endpoint:
1. Create route handler in `serving/api/v1/endpoints/`
2. Register router in `serving/api/v1/__init__.py`
3. Add schemas if needed in `schema/`

### Add new middleware:
1. Create middleware class in `serving/middleware/`
2. Export in `serving/middleware/__init__.py`
3. Register in `serving/api.py`

### Add new service:
1. Create service class in `services/`
2. Export in `services/__init__.py`
3. Add dependency function in `serving/dependencies.py`
