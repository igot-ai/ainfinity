"""
API package for AIFininity Training Service
"""

from ainfinity.app.api.dependencies import get_training_service
from ainfinity.utils.config import serving_settings

__all__ = ["serving_settings", "get_training_service"]
