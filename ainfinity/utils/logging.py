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


class Logger:
    """
    Logger class with convenient methods for different log levels
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
    ):
        """
        Initialize logger

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging
            format_string: Optional custom format string
        """
        self._logger = setup_logger(name, level, log_file, format_string)

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log a debug message

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log an info message

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log a warning message

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """
        Log an error message

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """
        Log a critical message

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """
        Log an exception with traceback

        Args:
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._logger.exception(message, *args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        """Get the underlying logger instance"""
        return self._logger
