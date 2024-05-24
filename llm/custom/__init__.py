from abc import ABC
from typing import List, Dict, Any

from llm.base_llm import BaseLLM
from protocol import LlmMessage, Query
from settings import Settings


class CustomLLM(BaseLLM, ABC):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def build_query_from_messages(self, llm_messages: List[LlmMessage]) -> Query:
        pass

    def interpret_result(self, llm_messages: List[LlmMessage], result: dict) -> str:
        pass

    def generate_llm_query_from_query(self, query: Query) -> str:
        pass

    def excute_generic_query(self, llm_message: str) -> str:
        pass


