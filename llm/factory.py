from protocols.llm_engine import LLM_TYPE_CUSTOM, LLM_TYPE_OPENAI, LLM_TYPE_CORCEL
from llm.base_llm import BaseLLM
from llm.openai import OpenAILLM
from llm.corcel import CorcelLLM
from settings import settings


class LLMFactory:
    @classmethod
    def create_llm(cls, llm_type: str) -> BaseLLM:
        llm_class = {
            LLM_TYPE_OPENAI: OpenAILLM,
            LLM_TYPE_CORCEL: CorcelLLM,
        }.get(llm_type)

        if llm_class is None:
            raise ValueError(f"Unsupported LLM Type: {llm_type}")

        return llm_class(settings=settings)