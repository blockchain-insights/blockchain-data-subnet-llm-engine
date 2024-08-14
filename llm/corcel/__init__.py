from typing import List
from settings import Settings
from protocols.llm_engine import LlmMessage, LLM_ERROR_QUERY_BUILD_FAILED, \
    LLM_ERROR_INTERPRETION_FAILED, LLM_ERROR_NOT_APPLICAPLE_QUESTIONS, LLM_ERROR_GENERAL_RESPONSE_FAILED
from llm.base_llm import BaseLLM
from loguru import logger
from llm.corcel.corcel_client import CorcelClient
from shared.helpers.llm_prompt_downloader import download_llm_prompt_content
from shared.helpers.llm_prompt_reader import read_local_file
import os

class CorcelLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.corcel_client = CorcelClient(settings.CORCEL_API_KEY)

    def build_query_from_messages_balance_tracker(self, llm_messages: List[LlmMessage], llm_type: str, network: str) -> str:
        return self._build_query_from_messages(llm_messages, llm_type, network, "balance_tracking")

    def build_cypher_query_from_messages(self, llm_messages: List[LlmMessage], llm_type: str, network: str) -> str:
        return self._build_query_from_messages(llm_messages, llm_type, network, "funds_flow")

    def _build_query_from_messages(self, llm_messages: List[LlmMessage], llm_type: str, network: str,
                                   subfolder: str) -> str:
        local_file_path = f"{llm_type}/{network}/{subfolder}/query_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            file_path = f"{llm_type}/{network}/{subfolder}/query_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if not prompt:
                raise Exception("Failed to read prompt content")

        question = "\n".join([message.content for message in llm_messages])

        try:
            ai_response, token_usage = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, question=question)
            logger.info(f'ai_response using GPT-4: {ai_response}')

            # Log the entire response content
            logger.debug(f"AI response content: {ai_response}")

            # Handle both cases: with and without triple backticks
            if ai_response.startswith("```") and ai_response.endswith("```"):
                # Extract the SQL code from the response
                query = ai_response.strip("```sql\n").strip("```")
            else:
                # Directly use the content as the query
                query = ai_response.strip()

            return query, token_usage
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def interpret_result_balance_tracker(self, llm_messages: List[LlmMessage], result: list, llm_type: str, network: str) -> str:
        return self._interpret_result(llm_messages, result, llm_type, network, "balance_tracking")

    def interpret_result_funds_flow(self, llm_messages: List[LlmMessage], result: list, llm_type: str, network: str) -> str:
        return self._interpret_result(llm_messages, result, llm_type, network, "funds_flow")

    def _interpret_result(self, llm_messages: List[LlmMessage], result: list, llm_type: str, network: str, subfolder: str) -> str:
        local_file_path = f"{llm_type}/{network}/{subfolder}/interpretation_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            file_path = f"{llm_type}/{network}/{subfolder}/interpretation_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if not prompt:
                raise Exception("Failed to read prompt content")

        prompt = prompt.format(result=result)
        question = "\n".join([message.content for message in llm_messages])

        try:
            ai_response, token_usage = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, result=result)
            ai_response = ai_response.strip('"')
            return ai_response, token_usage
        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)

    def determine_model_type(self, llm_messages: List[LlmMessage], llm_type: str, network: str) -> str:
        local_file_path = f"{llm_type}/{network}/classification/classification_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            file_path = f"{llm_type}/{network}/classification/classification_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if not prompt:
                raise Exception("Failed to read prompt content")

        question = "\n".join([message.content for message in llm_messages])
        logger.info(f"Formed question: {question}")

        try:
            ai_response, token_usage = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, question=question)

            if "Funds Flow" in ai_response:
                return "funds_flow", token_usage
            elif "Balance Tracking" in ai_response:
                return "balance_tracking", token_usage
            else:
                raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")
        except Exception as e:
            logger.error(f"LlmQuery classification error: {e}")
            raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")

    def generate_general_response(self, llm_messages: List[LlmMessage]) -> str:
        general_prompt = "Your general prompt here"
        question = "\n".join([message.content for message in llm_messages])

        try:
            ai_response, token_usage = self.corcel_client.send_prompt(model="gpt-4o", prompt=general_prompt, question=question)
            if ai_response == "not applicable questions":
                raise Exception(LLM_ERROR_NOT_APPLICAPLE_QUESTIONS)
            else:
                return ai_response, token_usage
        except Exception as e:
            logger.error(f"LlmQuery general response error: {e}")
            raise Exception(LLM_ERROR_GENERAL_RESPONSE_FAILED)

