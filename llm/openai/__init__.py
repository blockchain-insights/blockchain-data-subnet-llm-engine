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
from llm.prompts import query_schema, interpret_prompt, general_prompt, query_cypher_schema, balance_tracker_query_schema, balance_tracker_interpret_prompt, classification_prompt
from loguru import logger
from langchain_community.graphs import MemgraphGraph

from llm.utils import split_messages_into_chunks
from settings import Settings
from shared.helpers import blob_reader


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat_gpt4o = ChatOpenAI(api_key=settings.OPEN_AI_KEY, model="gpt-4o", temperature=0)
        self.MAX_TOKENS = 128000

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
            ai_message = self.chat_gpt4o.invoke(messages)
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

    def build_query_from_messages_balance_tracker(self, llm_messages: List[LlmMessage]) -> Query:

        blob_path = "bitcoin/balancer_tracker/query_prompt"
        prompt = download_blob_content(CONNECTION_STRING, CONTAINER_NAME, blob_path)
        if prompt:
            logger.info(f"Content of {blob_path}:\n{prompt}")
        else:
            # fallback: read from the local prompt
            prompt = balance_tracker_query_schema
            logger.error(f"Failed to read content of {blob_path}")

        messages = [
            SystemMessage(
                content=prompt
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))
        try:
            ai_message = self.chat_gpt4o.invoke(messages)
            logger.info(f'ai_message using GPT-4  : {ai_message}')

            # Log the entire response content
            logger.debug(f"AI message content: {ai_message.content}")

            # Check if the content is wrapped in triple backticks and contains SQL code
            if ai_message.content.startswith("```") and ai_message.content.endswith("```"):
                # Extract the SQL code from the response
                query = ai_message.content.strip("```sql\n").strip("```")
                return query
            else:
                logger.error("Received invalid format from AI response")
                raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def build_cypher_query_from_messages(self, llm_messages: List[LlmMessage]) -> str:

        blob_path = "bitcoin/funds_flow/query_prompt"
        prompt = download_blob_content(CONNECTION_STRING, CONTAINER_NAME, blob_path)
        if prompt:
            logger.info(f"Content of {blob_path}:\n{prompt}")
        else:
            # fallback: read from the local prompt
            prompt = query_cypher_schema
            logger.error(f"Failed to read content of {blob_path}")

        messages = [
            SystemMessage(
                content=prompt
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))
        try:
            ai_message = self.chat_gpt4o.invoke(messages)
            return ai_message.content
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def determine_model_type(self, llm_messages: List[LlmMessage]) -> str:

        blob_path = "classification/classification_prompt"
        prompt = download_blob_content(CONNECTION_STRING, CONTAINER_NAME, blob_path)
        if prompt:
            logger.info(f"Content of {blob_path}:\n{prompt}")
        else:
            # fallback: read from the local prompt
            prompt = classification_prompt
            logger.error(f"Failed to read content of {blob_path}")

        messages = [
            SystemMessage(
                content=prompt
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))

        try:
            ai_message = self.chat_gpt4o.invoke(messages)
            logger.info(f'ai_message using GPT-4: {ai_message}')

            # Log the entire response content
            logger.debug(f"AI message content: {ai_message.content}")

            # Extract the classification from the response
            if "Funds Flow" in ai_message.content:
                return "funds_flow"
            elif "Balance Tracking" in ai_message.content:
                return "balance_tracking"
            else:
                logger.error("Received invalid classification from AI response")
                raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")
        except Exception as e:
            logger.error(f"LlmQuery classification error: {e}")
            raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")

    def interpret_result(self, llm_messages: str, result: list) -> str:

        blob_path = "bitcoin/funds_flow/interpretation_prompt"
        prompt = download_blob_content(CONNECTION_STRING, CONTAINER_NAME, blob_path)
        if prompt:
            logger.info(f"Content of {blob_path}:\n{prompt}")
        else:
            # fallback: read from the local prompt
            prompt = interpret_prompt
            logger.error(f"Failed to read content of {blob_path}")

        messages = [
            SystemMessage(
                content=prompt.format(result=result)
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))

        try:
            message_chunks = split_messages_into_chunks(messages)
            ai_responses = []

            for chunk in message_chunks:
                ai_message = self.chat_gpt4o.invoke(chunk)
                ai_responses.append(ai_message.content)

            # Combine the responses
            combined_response = "\n".join(ai_responses)
            return combined_response

        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)

    def interpret_result_balance_tracker(self, llm_messages: List[LlmMessage], result: list) -> str:

        blob_path = "bitcoin/balancer_tracker/interpretation_prompt"
        prompt = download_blob_content(CONNECTION_STRING, CONTAINER_NAME, blob_path)
        if prompt:
            logger.info(f"Content of {blob_path}:\n{prompt}")
        else:
            #fallback: read from the local prompt
            prompt = balance_tracker_interpret_prompt
            logger.error(f"Failed to read content of {blob_path}")

        messages = [
            SystemMessage(
                content=prompt.format(result=result)
            ),
        ]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))

        try:
            ai_message = self.chat_gpt4o.invoke(messages)
            logger.info(f'ai_message using GPT-4  : {ai_message}')
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
            ai_message = self.chat_gpt4o.invoke(messages)
            logger.info(f'ai_message using GPT-4  : {ai_message}')
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
                                               verbose=True, model_name='gpt-4o')
        # Note: Querying
        try:
            response = chain.run(llm_message)
        except:
            response = "Failed"
        return response
