from abc import ABC, abstractmethod
from typing import List, Dict, Any

from protocols.llm_engine import LlmMessage, Query

from settings import Settings


class BaseLLM(ABC):
    @abstractmethod
    def __init__(self, settings: Settings) -> None:
        """
        Initialize LLM
        """
    
    @abstractmethod
    def build_query_from_messages(self, llm_messages: List[LlmMessage]):
        """
        Build query synapse from natural language query
        Used by miner
        """

    @abstractmethod
    def build_cypher_query_from_messages(self, llm_messages: List[LlmMessage]):
        """
        Build query synapse from natural language query
        Used by miner
        """

    @abstractmethod
    def interpret_result(self, llm_messages: List[LlmMessage], result: list) -> str:
        """
        Interpret result into natural language based on user's query and structured result dict
        """

    @abstractmethod
    def generate_general_response(self, llm_messages: List[LlmMessage]) -> str:
        """
        Generate general response based on chat history
        """
    
    @abstractmethod
    def generate_llm_query_from_query(self, query: Query) -> str:
        """
        Generate natural language query from Query object
        Used by validator
        """

    @abstractmethod
    def excute_generic_query(self, llm_message: str) -> str:
        """
        Generate natural language response and intermediate result from query using MemgraphCypherQAChain
        Used by miner
        """