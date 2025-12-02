"""
Base exceptions for the application
"""


class AIFinityException(Exception):
    """Base exception for all AIFinity errors"""


class ServiceException(AIFinityException):
    """Exception raised by service layer"""


class ConfigurationError(AIFinityException):
    """Exception raised for configuration errors"""


class JobNotFoundException(ServiceException):
    """Exception raised when a job is not found"""


class JobAlreadyExistsException(ServiceException):
    """Exception raised when trying to create a job that already exists"""


class SkyPilotException(ServiceException):
    """Exception raised for SkyPilot-related errors"""


class ValidationException(AIFinityException):
    """Exception raised for validation errors"""
