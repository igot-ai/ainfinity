# Core Module Documentation

## Tổng quan

Module `core` chứa các components cơ bản, reusable được sử dụng trong toàn bộ application:

```
ainfinity/core/
├── __init__.py          # Exports
├── config.py            # Configuration settings
├── constants.py         # Application constants
├── exceptions.py        # Custom exceptions
├── logging.py           # Logging utilities
└── utils.py             # Utility functions
```

## Components

### 1. Configuration (`config.py`)

Global application configuration sử dụng Pydantic Settings.

```python
from ainfinity.core import settings

# Access configuration
print(settings.HF_TOKEN)
print(settings.WANDB_API_KEY)
print(settings.PROJECT)
```

**Environment Variables:**
- `HF_TOKEN` - HuggingFace token
- `WANDB_API_KEY` - Weights & Biases API key
- `PROJECT` - Project name
- `HUB_MODEL_ID` - Model ID for HuggingFace Hub
- `SEED` - Random seed

### 2. Constants (`constants.py`)

Application-wide constants.

```python
from ainfinity.core.constants import (
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    DEFAULT_INFRA,
    DEFAULT_ACCELERATOR,
)
```

**Available Constants:**
- Job statuses: `JOB_STATUS_*`
- Defaults: `DEFAULT_*`
- File paths: `*_DIR`, `*_FILENAME`

### 3. Exceptions (`exceptions.py`)

Custom exception hierarchy.

```python
from ainfinity.core import (
    JobNotFoundException,
    JobAlreadyExistsException,
    SkyPilotException,
    ValidationException,
)

# Raise exceptions
raise JobNotFoundException("Job 'my-job' not found")
raise JobAlreadyExistsException("Job already exists")
```

**Exception Hierarchy:**
```
AIFinityException (base)
├── ServiceException
│   ├── JobNotFoundException
│   ├── JobAlreadyExistsException
│   └── SkyPilotException
├── ValidationException
└── ConfigurationError
```

**Usage in Services:**
```python
# services/training_service.py
from ainfinity.core import JobNotFoundException

def get_job_status(self, job_name: str):
    if job_name not in jobs:
        raise JobNotFoundException(f"Job '{job_name}' not found")
```

**Automatic HTTP Mapping:**
- `JobNotFoundException` → 404 Not Found
- `JobAlreadyExistsException` → 409 Conflict
- `ValidationException` → 422 Unprocessable Entity
- `ServiceException` → 500 Internal Server Error

### 4. Logging (`logging.py`)

Logging configuration và utilities.

```python
from ainfinity.core import setup_logger, get_logger

# Setup logger
logger = setup_logger(
    name="training_service",
    level="INFO",
    log_file="logs/training.log"
)

logger.info("Job started")
logger.error("Job failed", exc_info=True)

# Get existing logger
logger = get_logger("training_service")
```

**Functions:**
- `setup_logger(name, level, log_file, format_string)` - Create configured logger
- `get_logger(name)` - Get existing logger

### 5. Utilities (`utils.py`)

Common utility functions.

```python
from ainfinity.core import (
    generate_job_id,
    ensure_dir,
    save_json,
    load_json,
    sanitize_name,
    format_duration,
)

# Generate unique job ID
job_id = generate_job_id("my-training-job")  # Returns: "a3f8c9d2e1b4"

# Ensure directory exists
path = ensure_dir(Path("./checkpoints/run1"))

# JSON operations
save_json({"status": "running"}, Path("job.json"))
data = load_json(Path("job.json"))

# Sanitize names
clean_name = sanitize_name("My Training Job!")  # Returns: "my-training-job"

# Format durations
duration = format_duration(3665)  # Returns: "1.0h"
duration = format_duration(125)   # Returns: "2.1m"
duration = format_duration(5.2)   # Returns: "5.2s"
```

## Usage Patterns

### Pattern 1: Service Layer

```python
# services/training_service.py
from ainfinity.core import (
    JobNotFoundException,
    JobAlreadyExistsException,
    load_json,
    save_json,
    get_logger,
)

class TrainingService:
    def __init__(self):
        self.logger = get_logger(__name__)

    def launch_job(self, request):
        jobs = load_json(self.jobs_db_path)

        if request.job_name in jobs:
            raise JobAlreadyExistsException(
                f"Job '{request.job_name}' already exists"
            )

        # ... create job ...

        jobs[request.job_name] = job_data
        save_json(jobs, self.jobs_db_path)

        self.logger.info(f"Job '{request.job_name}' launched")
```

### Pattern 2: API Layer

```python
# serving/api.py
from ainfinity.core import (
    JobNotFoundException,
    JobAlreadyExistsException,
    ValidationException,
)
from ainfinity.serving.exceptions import (
    job_not_found_handler,
    job_exists_handler,
)

app = FastAPI()

# Register exception handlers
app.add_exception_handler(JobNotFoundException, job_not_found_handler)
app.add_exception_handler(JobAlreadyExistsException, job_exists_handler)
```

### Pattern 3: Configuration

```python
# Any module
from ainfinity.core import settings

def train_model():
    wandb.login(key=settings.WANDB_API_KEY)
    model = load_model(settings.HUB_MODEL_ID)
```

## Design Principles

### 1. **Centralization**
- Tất cả exceptions ở một nơi
- Shared utilities dễ maintain
- Consistent error handling

### 2. **Reusability**
- Functions có thể dùng ở bất kỳ layer nào
- No dependencies on upper layers
- Pure functions khi có thể

### 3. **Type Safety**
- Type hints đầy đủ
- Pydantic cho validation
- IDE autocomplete support

### 4. **Testability**
- Mỗi utility function dễ test
- No side effects
- Clear input/output

## Best Practices

### ✅ DO

```python
# Import từ core package
from ainfinity.core import JobNotFoundException, save_json

# Raise specific exceptions
raise JobNotFoundException(f"Job '{name}' not found")

# Use utilities
ensure_dir(Path("./outputs"))
```

### ❌ DON'T

```python
# Không import trực tiếp từ submodules
from ainfinity.core.exceptions import JobNotFoundException  # OK but verbose

# Không raise generic exceptions trong services
raise ValueError("Job not found")  # Use JobNotFoundException instead

# Không duplicate logic
def my_save_json():  # Use core.save_json instead
    with open(...) as f:
        json.dump(...)
```

## Extension Guide

### Adding New Exception

1. Định nghĩa trong `core/exceptions.py`:
```python
class ModelNotFoundException(ServiceException):
    """Exception raised when a model is not found"""
    pass
```

2. Export trong `core/__init__.py`:
```python
from ainfinity.core.exceptions import ModelNotFoundException

__all__ = [..., "ModelNotFoundException"]
```

3. Tạo handler trong `serving/exceptions.py`:
```python
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "ModelNotFound", "message": str(exc)}
    )
```

4. Register trong `serving/api.py`:
```python
app.add_exception_handler(ModelNotFoundException, model_not_found_handler)
```

### Adding New Utility

1. Implement trong `core/utils.py`:
```python
def calculate_cost(hours: float, gpu_type: str) -> float:
    """Calculate training cost"""
    rates = {"RTX3090": 0.5, "A100": 2.0}
    return hours * rates.get(gpu_type, 1.0)
```

2. Export trong `core/__init__.py`:
```python
from ainfinity.core.utils import calculate_cost

__all__ = [..., "calculate_cost"]
```

3. Use anywhere:
```python
from ainfinity.core import calculate_cost

cost = calculate_cost(10.5, "A100")
```

## Testing

```python
# tests/core/test_utils.py
from ainfinity.core import generate_job_id, sanitize_name

def test_generate_job_id():
    job_id = generate_job_id("test")
    assert len(job_id) == 12
    assert job_id.isalnum()

def test_sanitize_name():
    assert sanitize_name("My Job!") == "my-job"
    assert sanitize_name("Test_123") == "test_123"
```

## Summary

Core module cung cấp:
- ✅ Centralized configuration
- ✅ Custom exception hierarchy
- ✅ Reusable utilities
- ✅ Logging infrastructure
- ✅ Application constants
- ✅ Type-safe operations
- ✅ Easy to test
- ✅ Clean imports
