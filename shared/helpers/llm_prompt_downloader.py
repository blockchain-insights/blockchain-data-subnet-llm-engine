import requests
from loguru import logger
import os
import settings


def download_llm_prompt_content(file_name: str):
    try:
        llm_prompts_path = settings.Settings().LLM_PROMPTS_URL
        if llm_prompts_path is None:
            return None
        if llm_prompts_path is None:
            return None

        file_url = os.path.join(llm_prompts_path, file_name)
        response = requests.get(file_url, timeout=10)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        return content
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return None
