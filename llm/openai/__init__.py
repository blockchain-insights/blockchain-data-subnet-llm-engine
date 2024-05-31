import os
import json
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from protocols.blockchain import NETWORK_BITCOIN
from protocols.llm_engine import LLM_MESSAGE_TYPE_USER, LlmMessage, Query, LLM_ERROR_QUERY_BUILD_FAILED, \
    LLM_ERROR_INTERPRETION_FAILED, LLM_ERROR_NOT_APPLICAPLE_QUESTIONS, LLM_ERROR_GENERAL_RESPONSE_FAILED

from llm.base_llm import BaseLLM
from llm.openai.memgraph_chain import MemgraphCypherQAChain
from llm.prompts import query_schema, interpret_prompt, general_prompt
from loguru import logger
from langchain_community.graphs import MemgraphGraph

from settings import Settings


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat = ChatOpenAI(api_key=settings.OPEN_AI_KEY, model="gpt-4", temperature=0)

    def build_query_from_messages(self, llm_messages: List[LlmMessage]) -> Query:
        messages = [
            SystemMessage(
                content=query_schema
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))
        try:
            ai_message = self.chat.invoke(messages)
            query = json.loads(ai_message.content)
            return Query(
                network=NETWORK_BITCOIN,
                type=query["type"] if "type" in query else None,
                target=query["target"] if "target" in query else None,
                where=query["where"] if "where" in query else None,
                limit=query["limit"] if "limit" in query else None,
                skip=query["skip"] if "skip" in query else None,
            )
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def interpret_result(self, llm_messages: str, result: list) -> str:
        messages = [
            SystemMessage(
                content=interpret_prompt.format(result=result)
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))

        try:
            ai_message = self.chat.invoke(messages)
            return ai_message.content
        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)

    def generate_general_response(self, llm_messages: List[LlmMessage]) -> str:
        messages = [
            SystemMessage(
                content=general_prompt
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))

        try:
            ai_message = self.chat.invoke(messages)
            if ai_message == "not applicable questions":
                raise Exception(LLM_ERROR_NOT_APPLICAPLE_QUESTIONS)
            else:
                return ai_message.content
        except Exception as e:
            logger.error(f"LlmQuery general response error: {e}")
            raise Exception(LLM_ERROR_GENERAL_RESPONSE_FAILED)

    def generate_llm_query_from_query(self, query: Query) -> str:
        pass

    def excute_generic_query(self, llm_message: str) -> str:
        graph = MemgraphGraph(url=self.settings.GRAPH_DB_URL,
                              username=self.settings.GRAPH_DB_USER,
                              password=self.settings.GRAPH_DB_PASSWORD)

        # Note: Creating the GraphCypherQAChain
        chain = MemgraphCypherQAChain.from_llm(ChatOpenAI(temperature=0.7), graph=graph, return_intermediate_steps=True,
                                               verbose=True, model_name='gpt-4')
        # Note: Querying
        try:
            response = chain.run(llm_message)
        except:
            response = "Failed"
        return response
