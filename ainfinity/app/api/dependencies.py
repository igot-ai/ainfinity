"""
Dependencies for FastAPI routes
"""

from functools import lru_cache
from ainfinity.services import SkyPilotService


@lru_cache()
def get_training_service() -> SkyPilotService:
    """
    Get training service instance (cached singleton)

    Returns:
        SkyPilotService instance
    """
    return SkyPilotService()
