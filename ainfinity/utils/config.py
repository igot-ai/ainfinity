from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    PROJECT: str = "aifinity"
    HF_TOKEN: str
    HUB_MODEL_ID: Optional[str] = None
    WANDB_API_KEY: str
    SEED: int = 0

    model_config = SettingsConfigDict(case_sensitive=False)


class ServingSettings(BaseSettings):
    """Settings for serving API"""
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    API_TITLE: str = "AIFininity Training Service"
    API_DESCRIPTION: str = "API for managing training jobs with SkyPilot"
    API_VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]
    
    # Service Settings
    WORKSPACE_ROOT: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_prefix="SERVING_",
        case_sensitive=False
    )


settings = Settings()
serving_settings = ServingSettings()
