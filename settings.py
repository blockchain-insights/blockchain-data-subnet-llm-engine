
from dotenv import load_dotenv
from pydantic import Extra
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    LLM_TYPE: str
    OPEN_AI_KEY: str
    GRAPH_DB_URL: str
    GRAPH_DB_USER: str
    GRAPH_DB_PASSWORD: str

    class Config:
        env_file = ".env"  # Path to the .env file
        extra = Extra.allow


settings = Settings()
print("Settings:")
print(settings.dict())
print(settings.LLM_TYPE)
print(settings.OPEN_AI_KEY)
print(settings.GRAPH_DB_URL)
print(settings.GRAPH_DB_USER)
print(settings.GRAPH_DB_PASSWORD)
