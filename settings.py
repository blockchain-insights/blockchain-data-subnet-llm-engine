from typing import Optional

from dotenv import load_dotenv
from pydantic import Extra
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    LLM_TYPE: str
    OPEN_AI_KEY: Optional[str]
    CORCEL_API_KEY: Optional[str]
    GRAPH_DB_URL: str
    GRAPH_DB_USER: str
    GRAPH_DB_PASSWORD: str
    DB_CONNECTION_STRING: str
    LLM_PROMPTS_URL: Optional[str] = None

    class Config:
        env_file = ".env"  # Path to the .env file
        extra = Extra.allow


settings = Settings()