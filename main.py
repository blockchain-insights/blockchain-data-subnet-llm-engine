import traceback
from typing import List

from loguru import logger
import sys
from fastapi import FastAPI, Request, Depends, Query, Path, Body, APIRouter
from pydantic import BaseModel, Field

import __init__
from db.graph_search import GraphSearchFactory
from llm.factory import LLMFactory
from protocol import QueryOutput, LLM_ERROR_TYPE_NOT_SUPPORTED, LlmMessage, LLM_ERROR_MESSAGES

app = FastAPI(
    title="Blockchain Insights LLM ENGINE",
    description="API designed to execute user prompts related to blockchain queries using LLM agents. It integrates with different LLMs and graph search functionalities to process and interpret blockchain data.",
    version=__init__.__version__,)

logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def get_llm_factory() -> LLMFactory:
    return LLMFactory()


def get_graph_search_factory() -> GraphSearchFactory:
    return GraphSearchFactory()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"New request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Request completed: {response.status_code}")
    return response


class LLMQueryRequestV1(BaseModel):
    llm_type: str = Field(default="openai", title="The type of the llm agent")
    network: str = Field(default="bitcoin", title="The network to query")
    messages: List[LlmMessage] = Field(default=[], title="The conversation history for llm agent to use as context")


v1_router = APIRouter()

@v1_router.post("/process_prompt", summary="Executes user prompt", description="Execute user prompt and return the result", tags=["v1"], response_model=QueryOutput)
async def llm_query_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [{"type": 0, "content": "What is the balance of address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory),
        graph_search_factory: GraphSearchFactory = Depends(get_graph_search_factory)):

    logger.info(f"llm query received: {request.llm_type}, network: {request.network}")

    output = QueryOutput()

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

app.include_router(v1_router, prefix="/v1")