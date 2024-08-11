from typing import List
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from protocols.llm_engine import LLM_MESSAGE_TYPE_USER, LlmMessage, Query, LLM_ERROR_QUERY_BUILD_FAILED, \
    LLM_ERROR_INTERPRETION_FAILED, LLM_ERROR_NOT_APPLICAPLE_QUESTIONS, LLM_ERROR_GENERAL_RESPONSE_FAILED
from llm.base_llm import BaseLLM
from llm.prompts import general_prompt
from loguru import logger
from llm.utils import split_messages_into_chunks
from settings import Settings
from shared.helpers.llm_prompt_downloader import download_llm_prompt_content
from shared.helpers.llm_prompt_reader import read_local_file
from string import Template


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat_gpt4o = ChatOpenAI(api_key=settings.OPEN_AI_KEY, model="gpt-4o", temperature=0)
        self.MAX_TOKENS = 128000

    def build_query_from_messages_balance_tracker(self, llm_messages: List[LlmMessage], llm_type: str, network: str):
        local_file_path = f"{llm_type}/{network}/balance_tracking/query_prompt.txt"
        prompt = read_local_file(local_file_path)
        if prompt:
            logger.info(f"Read content from local file {local_file_path}:\n{prompt}")
        else:
            logger.error(f"Failed to read content from local file {local_file_path}")
            # fallback: read from the remote file
            file_path = f"{llm_type}/{network}/balance_tracking/query_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if prompt:
                logger.info(f"Content of {file_path}:\n{prompt}")
            else:
                # If both methods fail, log an error and raise an exception
                logger.error(f"Failed to read content from both local file and remote {local_file_path} and {file_path}")
                raise Exception("Failed to read prompt content")

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

            token_usage = ai_message.response_metadata['token_usage']

            # Handle both cases: with and without triple backticks
            if ai_message.content.startswith("```") and ai_message.content.endswith("```"):
                # Extract the SQL code from the response
                query = ai_message.content.strip("```sql\n").strip("```")
            else:
                # Directly use the content as the query
                query = ai_message.content.strip()

            return query, token_usage
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def build_cypher_query_from_messages(self, llm_messages: List[LlmMessage], llm_type: str, network: str):

        local_file_path = f"{llm_type}/{network}/funds_flow/query_prompt.txt"
        prompt = read_local_file(local_file_path)
        if prompt:
            logger.info(f"Read content from local file {local_file_path}:\n{prompt}")
        else:
            logger.error(f"Failed to read content from local file {local_file_path}")
            # fallback: read from the remote file
            file_path = f"{llm_type}/{network}/funds_flow/query_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if prompt:
                logger.info(f"Content of {file_path}:\n{prompt}")
            else:
                # If both methods fail, log an error and raise an exception
                logger.error(f"Failed to read content from both local file and remote {local_file_path} and {file_path}")
                raise Exception("Failed to read prompt content")

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
            logger.info(f"AI-generated message: {ai_message.content}")
            token_usage = ai_message.response_metadata['token_usage']
            return ai_message.content, token_usage
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def determine_model_type(self, llm_messages: List[LlmMessage], llm_type: str, network: str):

        local_file_path = f"{llm_type}/{network}/classification/classification_prompt.txt"
        prompt = read_local_file(local_file_path)
        if prompt:
            logger.info(f"Read content from local file {local_file_path}:\n{prompt}")
        else:
            logger.error(f"Failed to read content from local file {local_file_path}")
            # fallback: read from the remote file
            file_path = f"{llm_type}/{network}/classification/classification_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if prompt:
                logger.info(f"Content of {file_path}:\n{prompt}")
            else:
                # If both methods fail, log an error and raise an exception
                logger.error(f"Failed to read content from both local file and remote {local_file_path} and {file_path}")
                raise Exception("Failed to read prompt content")

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
            
            # Count tokens for the output message
            token_usage = ai_message.response_metadata['token_usage']
            logger.info(f"Number of tokens: {token_usage}")
            
            # Extract the classification from the response
            if "Funds Flow" in ai_message.content:
                return "funds_flow", token_usage
            elif "Balance Tracking" in ai_message.content:
                return "balance_tracking", token_usage
            else:
                logger.error("Received invalid classification from AI response")
                raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")
        except Exception as e:
            logger.error(f"LlmQuery classification error: {e}")
            raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")

    def interpret_result_funds_flow(self, llm_messages: list, result: list, llm_type: str, network: str):
        local_file_path = f"{llm_type}/{network}/funds_flow/interpretation_prompt.txt"
        prompt = read_local_file(local_file_path)
        if prompt:
            logger.info(f"Read content from local file {local_file_path}:\n{prompt}")
        else:
            logger.error(f"Failed to read content from local file {local_file_path}")
            # fallback: read from the remote file
            file_path = f"{llm_type}/{network}/funds_flow/interpretation_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if prompt:
                logger.info(f"Content of {file_path}:\n{prompt}")
            else:
                # If both methods fail, log an error and raise an exception
                logger.error(f"Failed to read content from both local file and remote {local_file_path} and {file_path}")
                raise Exception("Failed to read prompt content")

        # Convert result to JSON string
        # Check if the result is empty
        if not result:
            logger.warning("The result is empty. Ensure the result data is correctly generated.")
            # Optionally handle empty results differently, e.g., set a default message
            result_str = "No data available to interpret. Result is empty."
        else:
            # Convert result to JSON string
            try:
                result_str = json.dumps(result, indent=2)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error during result formatting: {e}")
                raise Exception("Error formatting result as JSON") from e

        # Use string.format to safely substitute the result into the prompt
        try:
            full_prompt = prompt.format(result=result_str)
        except KeyError as e:
            logger.error(f"KeyError during prompt formatting: {e}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Result: {result_str}")
            raise Exception("Error formatting prompt with result") from e

        # Prepare the messages
        messages = [SystemMessage(content=full_prompt)]
        for llm_message in llm_messages:
            if llm_message.type == LLM_MESSAGE_TYPE_USER:
                messages.append(HumanMessage(content=llm_message.content))
            else:
                messages.append(AIMessage(content=llm_message.content))
        try:
            message_chunks = split_messages_into_chunks(messages)
            ai_responses = []
            total_token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

            for chunk in message_chunks:
                ai_message = self.chat_gpt4o.invoke(chunk)
                token_usage = ai_message.response_metadata['token_usage']
                total_token_usage['completion_tokens'] += token_usage['completion_tokens']
                total_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
                total_token_usage['total_tokens'] += token_usage['total_tokens']
                ai_responses.append(ai_message.content)

            # Combine the responses
            combined_response = "\n".join(ai_responses)
            return combined_response, total_token_usage

        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)

    def interpret_result_balance_tracker(self, llm_messages: List[LlmMessage], result: list, llm_type: str, network: str):

        local_file_path = f"{llm_type}/{network}/balance_tracking/interpretation_prompt.txt"
        prompt = read_local_file(local_file_path)
        if prompt:
            logger.info(f"Read content from local file {local_file_path}:\n{prompt}")
        else:
            logger.error(f"Failed to read content from local file {local_file_path}")
            # fallback: read from the remote file
            file_path = f"{llm_type}/{network}/balance_tracking/interpretation_prompt.txt"
            prompt = download_llm_prompt_content(file_path)
            if prompt:
                logger.info(f"Content of {file_path}:\n{prompt}")
            else:
                # If both methods fail, log an error and raise an exception
                logger.error(f"Failed to read content from both local file and remote {local_file_path} and {file_path}")
                raise Exception("Failed to read prompt content")

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
            token_usage = ai_message.response_metadata['token_usage']
            return ai_message.content, token_usage
        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)
    def generate_general_response(self, llm_messages: List[LlmMessage]):
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
            token_usage = ai_message.response_metadata['token_usage']
            if ai_message == "not applicable questions":
                raise Exception(LLM_ERROR_NOT_APPLICAPLE_QUESTIONS)
            else:
                return ai_message.content, token_usage
        except Exception as e:
            logger.error(f"LlmQuery general response error: {e}")
            raise Exception(LLM_ERROR_GENERAL_RESPONSE_FAILED)

    def generate_llm_query_from_query(self, query: Query):
        pass
