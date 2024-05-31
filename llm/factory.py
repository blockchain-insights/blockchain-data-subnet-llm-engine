from protocols.llm_engine import LLM_TYPE_CUSTOM, LLM_TYPE_OPENAI

from llm.base_llm import BaseLLM
from llm.custom import CustomLLM
from llm.openai import OpenAILLM
from settings import settings


class LLMFactory:
    @classmethod
    def create_llm(cls, llm_type: str) -> BaseLLM:
        llm_class = {
            LLM_TYPE_CUSTOM: CustomLLM,
            LLM_TYPE_OPENAI: OpenAILLM
            # Add other networks and their corresponding classes as needed
        }.get(llm_type)

        if llm_class is None:
            raise ValueError(f"Unsupported LLM Type: {llm_type}")

        return llm_class(settings=settings)