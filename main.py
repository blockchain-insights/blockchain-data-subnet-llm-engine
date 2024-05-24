import traceback
from typing import List

from loguru import logger
import sys
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel

from db.graph_search import GraphSearchFactory
from llm.factory import LLMFactory
from protocol import QueryOutput, LLM_ERROR_TYPE_NOT_SUPPORTED, LlmMessage, LLM_ERROR_MESSAGES

app = FastAPI()

logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def get_llm_factory() -> LLMFactory:
    return LLMFactory()


def get_graph_search_factory() -> GraphSearchFactory:
    return GraphSearchFactory()


# Add a request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"New request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Request completed: {response.status_code}")
    return response


# Define request body model
class LLMQueryRequest(BaseModel):
    llm_type: str
    network: str
    # messages: conversation history for llm agent to use as context
    messages: List[LlmMessage] = None

@app.post("/llm_query")
async def llm_query(request: LLMQueryRequest,
                    llm_factory: LLMFactory = Depends(get_llm_factory),
                    graph_search_factory: GraphSearchFactory = Depends(get_graph_search_factory)):

    logger.info(f"llm query received: {request.llm_type}, network: {request.network}")

    output = {}

    try:
        llm = llm_factory.create_llm(request.llm_type)
        query = llm.build_query_from_messages(request.messages)
        logger.info(f"extracted query: {query}")

        graph_search = graph_search_factory.create_graph_search(request.network)
        result = graph_search.execute_query(query=query)
        interpreted_result = llm.interpret_result(llm_messages=request.messages, result=result)

        output = QueryOutput(result=result, interpreted_result=interpreted_result)

    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0]
        if error_code == LLM_ERROR_TYPE_NOT_SUPPORTED:
            # handle unsupported query templates
            try:
                interpreted_result = llm.excute_generic_query(llm_message=request.messages[-1].content)
                if interpreted_result == "Failed":
                    interpreted_result = llm.generate_general_response(llm_messages=request.messages)
                    output = QueryOutput(error=error_code, interpreted_result=interpreted_result)
                else:
                    output = QueryOutput(error=error_code, interpreted_result=interpreted_result)
            except Exception as e:
                error_code = e.args[0]
                output = QueryOutput(error=error_code, interpreted_result=LLM_ERROR_MESSAGES[error_code])
        else:
            output = QueryOutput(error=error_code, interpreted_result=LLM_ERROR_MESSAGES[error_code])

    logger.info(f"Serving miner llm query output: {output}")

    return output
