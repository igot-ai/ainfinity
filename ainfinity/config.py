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


settings = Settings()
