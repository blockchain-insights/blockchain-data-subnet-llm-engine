from typing import Optional

from dotenv import load_dotenv
from pydantic import Extra
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL
import os

load_dotenv(override=True)


class Settings(BaseSettings):
    OPEN_AI_KEY: Optional[str] = os.environ.get('OPEN_AI_KEY')
    CORCEL_API_KEY: Optional[str] = os.environ.get('CORCEL_API_KEY')
    BITCOIN_GRAPH_DB_URL: str = os.environ.get('BITCOIN_GRAPH_DB_URL')
    BITCOIN_GRAPH_DB_USER: str = os.environ.get('BITCOIN_GRAPH_DB_USER')
    BITCOIN_GRAPH_DB_PASSWORD: str = os.environ.get('BITCOIN_GRAPH_DB_PASSWORD')
    LLM_PROMPTS_URL: Optional[str] = os.environ.get('LLM_PROMPTS_URL')
    
    db_url_obj: URL = URL.create(
            "postgresql+asyncpg",
            username=os.environ.get("BITCOIN_POSTGRES_USER"),
            password=os.environ.get("BITCOIN_POSTGRES_PASSWORD"),
            host=os.environ.get("BITCOIN_POSTGRES_HOST"),
            port=os.environ.get("BITCOIN_POSTGRES_PORT"),
            database=os.environ.get("BITCOIN_POSTGRES_DB")
        )
    model_config = SettingsConfigDict(env_file=".env", extra='allow')


settings = Settings()