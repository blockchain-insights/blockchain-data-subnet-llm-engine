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
from llm.prompts import query_schema, interpret_prompt, general_prompt, query_cypher_schema
from loguru import logger
from langchain_community.graphs import MemgraphGraph

from llm.utils import split_messages_into_chunks
from settings import Settings


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
        balance_tracker_query_schema = """
            You are an assistant to help me query balance changes.
            I will ask you questions, and you will generate SQL queries to fetch the data.

            There are two database tables:

            1. `balance_changes` with the following columns:
               - address (string): The address involved in the balance change.
               - block (integer): The block in which the balance change occurred.
               - d_balance (big integer): The change in balance.
               - block_timestamp (timestamp): The timestamp of the block.

            2. `blocks` with the following columns:
               - block_height (integer): The height of the block.
               - timestamp (timestamp): The timestamp of the block.

            Relationships:
            - there are no relationships between the tables.

            You should be able to handle queries that span across these two tables. 
            The `balance_changes` table contains the balance changes over time for different addresses. 
            The `blocks` table contains information about the blocks and their timestamps.
            
            
            For example:
            "Return the address with the highest amount of BTC in December 2009." this question can be answered using the `balance_changes` table, take specific date and address with highest balance, do not SUM the balance.
            The data in this table is preprocessed and ready to be queried.
            
            "Return the block height of the block with the highest timestamp." this question can be answered using the `blocks` table.

            My question: {question}
            SQL query:
            """

        messages = [
            SystemMessage(
                content=balance_tracker_query_schema
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
        messages = [
            SystemMessage(
                content=query_cypher_schema
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
        balance_tracker_interpret_prompt = """
            You are an assistant to interpret the results of a query involving balance changes, current balances, and block information.
            Here is the result set:
            {result}

            The data comes from three tables:
            1. `balance_changes` which includes balance changes over time with columns:
               - address (string)
               - block (integer)
               - d_balance (big integer)
               - block_timestamp (timestamp)

            2. `current_balances` which includes the current balances for addresses with columns:
               - address (string)
               - balance (big integer)

            3. `blocks` which includes information about blocks and their timestamps with columns:
               - block_height (integer)
               - timestamp (timestamp)

            Relationships:
            - `balance_changes` is related to `current_balances` via the `address` field.
            - `blocks` is related to `balance_changes` via the `block_height` (in `blocks`) and `block` (in `balance_changes`) fields.

            Please summarize the results in a user-friendly way.
            """

        messages = [
            SystemMessage(
                content=balance_tracker_interpret_prompt.format(result=result)
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
