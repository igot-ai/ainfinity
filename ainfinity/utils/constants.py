"""
Constants used throughout the application
"""

# Job statuses
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_STOPPED = "stopped"
JOB_STATUS_UNKNOWN = "unknown"

# Default values
DEFAULT_INFRA = "vast"
DEFAULT_ACCELERATOR = "RTX3090:1"
DEFAULT_DISK_SIZE = 100
DEFAULT_IMAGE = "docker:nvidia/cuda:12.1.1-devel-ubuntu20.04"

# API defaults
DEFAULT_API_V1_PREFIX = "/api/v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Rate limiting
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File paths
JOBS_DB_FILENAME = ".skypilot_jobs.json"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"

# SkyPilot
SKYPILOT_TEMPLATE_DIR = "ainfinity/skylaunch"
SKYPILOT_TEMPLATE_FILE = "deepspeed.yaml"
