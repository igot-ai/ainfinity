"""
Base exceptions for the application
"""


class AIFinityException(Exception):
    """Base exception for all AIFinity errors"""
    pass


class ServiceException(AIFinityException):
    """Exception raised by service layer"""
    pass


class ConfigurationError(AIFinityException):
    """Exception raised for configuration errors"""
    pass


class JobNotFoundException(ServiceException):
    """Exception raised when a job is not found"""
    pass


class JobAlreadyExistsException(ServiceException):
    """Exception raised when trying to create a job that already exists"""
    pass


class SkyPilotException(ServiceException):
    """Exception raised for SkyPilot-related errors"""
    pass


class ValidationException(AIFinityException):
    """Exception raised for validation errors"""
    pass
