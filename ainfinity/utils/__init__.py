"""
Core package - Base components and utilities
"""

from ainfinity.utils.config import settings
from ainfinity.utils.logging import Logger, setup_logger
from ainfinity.utils.utils import (
    ensure_dir,
    format_duration,
    generate_job_id,
    load_json,
    sanitize_name,
    save_json,
)

logger = Logger(name="ainfinity", level="INFO")

__all__ = [
    # Config
    "settings",
    # Logging
    "Logger",
    "setup_logger",
    # Utils
    "generate_job_id",
    "ensure_dir",
    "save_json",
    "load_json",
    "sanitize_name",
    "format_duration",
]
