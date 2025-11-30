"""
Logging utilities for the application
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str, level: str = "INFO", log_file: Optional[str] = None, format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and optional file handler

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
