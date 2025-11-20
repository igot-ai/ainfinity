from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    HF_TOKEN: str
    WANDB_API_KEY: str

    model_config = SettingsConfigDict(case_sensitive=False)


settings = Settings()
