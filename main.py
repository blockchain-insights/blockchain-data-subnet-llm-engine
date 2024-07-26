import asyncio
import json
import time
import traceback
from typing import List, Tuple
from pathlib import Path as FilePath

import protocols.blockchain
from loguru import logger
from fastapi import FastAPI, Request, Depends, Query, Body, APIRouter, HTTPException
from protocols.llm_engine import LlmMessage, QueryOutput, LLM_ERROR_TYPE_NOT_SUPPORTED, LLM_ERROR_MESSAGES, \
    LLM_UNKNOWN_ERROR, LLM_ERROR_INVALID_SEARCH_PROMPT, LLM_ERROR_MODIFICATION_NOT_ALLOWED,  MODEL_TYPE_BALANCE_TRACKING, MODEL_TYPE_FUNDS_FLOW
from pydantic import BaseModel, Field

import __init__
from data.bitcoin.balance_search import BalanceSearchFactory, BitcoinBalanceSearch
from data.bitcoin.graph_result_transformer import transform_result
from data.bitcoin.tabular_result_transformer import transform_result_set
from data.bitcoin.chart_result_transformer import is_chart_applicable, convert_funds_flow_to_chart, convert_balance_tracking_to_chart
from data.bitcoin.graph_search import GraphSearchFactory, get_graph_search
from data.bitcoin.query_builder import QueryBuilder
from llm.factory import LLMFactory
from settings import settings

from sqlalchemy import Column, Integer, BigInteger, String, TIMESTAMP, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

from typing import Dict, Union

app = FastAPI(
    title="Blockchain Insights LLM ENGINE",
    description="API designed to execute user prompts related to blockchain queries using LLM agents. It integrates with different LLMs and graph search functionalities to process and interpret blockchain data.",
    version=__init__.__version__, )

benchmark_funds_flow_restricted_keywords = ['CREATE', 'SET', 'DELETE', 'DETACH', 'REMOVE', 'MERGE', 'CREATE INDEX', 'DROP INDEX',
                                 'CREATE CONSTRAINT', 'DROP CONSTRAINT']

benchmark_balance_tracking_restricted_keywords = ['CREATE', 'SET', 'DELETE', 'DETACH', 'REMOVE', 'MERGE', 'CREATE INDEX', 'DROP INDEX',
                                 'CREATE CONSTRAINT', 'DROP CONSTRAINT']


def get_llm_factory() -> LLMFactory:
    return LLMFactory()


def get_graph_search_factory() -> GraphSearchFactory:
    return GraphSearchFactory()


def get_balance_search_factory() -> BalanceSearchFactory:
    return BalanceSearchFactory()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await asyncio.wait_for(call_next(request), timeout=3 * 60)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Request timeout: {request.method} {request.url}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()

    duration = end_time - start_time
    logger.info(f"Request completed: {request.method} {request.url} in {duration:.4f} seconds")

    return response


class LLMQueryRequestV1(BaseModel):
    llm_type: str = Field(default="openai", title="The type of the llm agent")
    network: str = Field(default="bitcoin", title="The network to query")
    messages: List[LlmMessage] = Field(default=[], title="The conversation history for llm agent to use as context")


class SwitchResponse(BaseModel):
    model: str = Field(default="funds_flow", title="The model to be prompted")


v1_router = APIRouter()
valid_networks = [protocols.blockchain.NETWORK_BITCOIN, protocols.blockchain.NETWORK_ETHEREUM]


@v1_router.get("/networks", summary="Get supported networks", description="Get the list of supported networks",
               tags=["v1"])
async def get_networks():
    return {"networks": valid_networks}


@v1_router.get("/schema/{network}", summary="Get schema for network",
               description="Get the schema for the specified network", tags=["v1"])
async def get_schema(network: str):
    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    file_path = FilePath(f"./schemas/{network}.json")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Schema file not found")

    try:
        with file_path.open("r") as file:
            content = json.load(file)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/discovery/{network}", summary="Get network discovery", description="Get the network discovery details",
               tags=["v1"])
async def discovery_v1(network: str,
                       balance_search_factory: BalanceSearchFactory = Depends(get_balance_search_factory)):
    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")
    if network == protocols.blockchain.NETWORK_BITCOIN:
        graph_search = get_graph_search(settings, network)
        funds_flow_model_start_block, funds_flow_model_last_block = graph_search.get_min_max_block_height_cache()
        graph_search.close()

        balance_search = balance_search_factory.create_balance_search(network)
        balance_model_last_block = balance_search.get_latest_block_number()
        balance_search.close()

        return {
            "network": network,
            "funds_flow_model_start_block": funds_flow_model_start_block,
            "funds_flow_model_ast_block": funds_flow_model_last_block,
            "balance_model_last_block": balance_model_last_block,
            "llm_engine_version": __init__.__version__
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid network")


@v1_router.get("/challenge/funds_flow/{network}",
               summary="Solve challenge",
               description="Solve the funds flow challenge", tags=["v1"])
async def challenge_funds_flow_v1(network: str,
                       in_total_amount: int = Query(None, description="Input total amount"),
                       out_total_amount: int = Query(None, description="Output total amount"),
                       tx_id_last_6_chars: str = Query(None, description="Transaction ID last 6 characters"),
                       checksum: str = Query(None, description="Checksum query parameter")):
    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    if network == protocols.blockchain.NETWORK_BITCOIN:
        if in_total_amount is None or out_total_amount is None or tx_id_last_6_chars is None:
            raise HTTPException(status_code=400,
                                detail="Missing required query parameters for Bitcoin network, required: in_total_amount, out_total_amount, tx_id_last_4_chars")
        graph_search = get_graph_search(settings, network)
        output = graph_search.solve_challenge(
            in_total_amount=in_total_amount,
            out_total_amount=out_total_amount,
            tx_id_last_6_chars=tx_id_last_6_chars
        )
        graph_search.close()
        return {
            "network": network,
            "output": output,
        }
    elif network == protocols.blockchain.NETWORK_ETHEREUM:
        if checksum is None:
            raise HTTPException(status_code=400,
                                detail="Missing required query parameters for EVM network, required: checksum")
        return {
            "network": network,
            "output": 0,
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid network")


@v1_router.get("/challenge/balance_tracking/{network}",
               summary="Solve balance tracking challenge",
               description="Solve the balance tracking challenge", tags=["v1"])
async def challenge_balance_tracking_v1(network: str,
                               block_height: int = Query(None, description="Block height query parameter")):
    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    if network == protocols.blockchain.NETWORK_BITCOIN:
        if block_height is None:
            raise HTTPException(status_code=400,
                                detail="Missing required query parameters for Bitcoin network, required: block_height")

        balance_tracking = BitcoinBalanceSearch()
        output = balance_tracking.execute_bitcoin_balance_challenge(block_height)

        return {
                "network": network,
                "output": output,
                }
    else:
        raise HTTPException(status_code=400, detail="Invalid network")


@v1_router.get("/benchmark/funds_flow/{network}", summary="Benchmark query", description="Benchmark the query", tags=["v1"])
async def benchmark_funds_flow_v1(network: str, query: str = Query(..., description="Query to benchmark")):
    if not is_query_only(benchmark_funds_flow_restricted_keywords, query):
        raise HTTPException(status_code=400, detail="Invalid query, restricted keywords found in query")

    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    graph_search = get_graph_search(settings, network)
    output = graph_search.execute_benchmark_query(query)
    graph_search.close()
    return {
        "network": network,
        "output": output[0],
    }

@v1_router.get("/benchmark/balance_tracking/{network}", summary="Benchmark query", description="Benchmark the query",
               tags=["v1"])
async def benchmark_balance_tracking_v1(network: str, query: str = Query(..., description="Query to benchmark")):
    if not is_query_only(benchmark_balance_tracking_restricted_keywords, query):
        raise HTTPException(status_code=400, detail="Invalid query, restricted keywords found in query")

    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    balance_search_factory = get_balance_search_factory()
    balance_search = balance_search_factory.create_balance_search(network)
    output = balance_search.execute_benchmark_query(query)
    balance_search.close()
    return {
        "network": network,
        "output": output,
    }


@v1_router.post("/process_prompt", summary="Executes user prompt",
                description="Execute user prompt and return the result", tags=["v1"],
                response_model=Union[List[QueryOutput], Dict, Tuple])
async def llm_query_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [
            {"type": 0,
             "content": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory),
        graph_search_factory: GraphSearchFactory = Depends(get_graph_search_factory),
        balance_search_factory: BalanceSearchFactory = Depends(get_balance_search_factory)):
    logger.info(f"llm query received: llm_type={request.llm_type}, network={request.network}, messages={request.messages}")
    output = None
    token_usage = None
    start_time = time.time()

    try:
        llm = llm_factory.create_llm(request.llm_type)
        logger.info(f"Created LLM: {llm}")

        # Determine the model type
        model_type, token_usage_classification  = llm.determine_model_type(request.messages, request.llm_type, request.network)
        logger.info(f"Determined model type: {model_type}")

        if model_type == 'funds_flow':
            output, token_usage_query_interpret = await handle_funds_flow_query(request, llm, graph_search_factory)
        elif model_type == 'balance_tracking':
            output, token_usage_query_interpret = await handle_balance_tracking_query(request, llm, balance_search_factory)
        else:
            output = {'error': 'Unsupported model type'}
        token_usage = {
            'completion_tokens': token_usage_classification['completion_tokens'] + token_usage_query_interpret['completion_tokens'], 
            'prompt_tokens': token_usage_classification['prompt_tokens'] + token_usage_query_interpret['prompt_tokens'], 
            'total_tokens': token_usage_classification['total_tokens'] + token_usage_query_interpret['total_tokens']
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
        output = [
            QueryOutput(type="error", error=error_code,
                        interpreted_result=LLM_ERROR_MESSAGES.get(error_code, 'An error occurred'))]

    logger.info(f"Serving miner llm query output: {output} (Total time taken: {time.time() - start_time} seconds)")

    return output, token_usage


@v1_router.post("/process_prompt_switch", summary="Determine proper model to be prompted", description="", tags=["v1"],
                response_model=SwitchResponse)
async def llm_query_switch_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [
            {"type": 0,
             "content": "Return me top 3 addresses that have the highest current balances plus return blocks and timestamps."}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory)):
    logger.info(f"Received request: {request}")
    llm = llm_factory.create_llm(request.llm_type)
    model_type, token_usage = llm.determine_model_type(request.messages, request.llm_type, request.network)
    logger.info(f"Determined model type: {model_type}")
    return SwitchResponse(model=model_type)

@v1_router.post("/process_prompt_funds_flow", summary="Executes user prompt",
                description="Execute user prompt and return the result", tags=["v1"],
                response_model=Union[List[QueryOutput], Dict])
async def llm_query_funds_flow_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [
            {"type": 0,
             "content": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory),
        graph_search_factory: GraphSearchFactory = Depends(get_graph_search_factory)):
    logger.info(f"llm query funds flow received: llm_type={request.llm_type}, network={request.network}, messages={request.messages}")
    output = None
    start_time = time.time()

    try:
        llm = llm_factory.create_llm(request.llm_type)
        logger.info(f"Created LLM: {llm}")

        output, token_usage = await handle_funds_flow_query(request, llm, graph_search_factory)

    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
        output = [
            QueryOutput(type="error", error=error_code,
                        interpreted_result=LLM_ERROR_MESSAGES.get(error_code, 'An error occurred'))]

    logger.info(f"Serving llm query funds flow output: {output} (Total time taken: {time.time() - start_time} seconds)")

    return output


@v1_router.post("/process_prompt_balance_tracking", summary="Executes user prompt for balance tracking",
                description="Execute user prompt for balance tracking and return the result", tags=["v1"],
                response_model=List[QueryOutput])
async def llm_query_balance_tracking_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [
            {"type": 0, "content": "Return me address who had highest amount of BTC in 2009-01"}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory),
        balance_search_factory: BalanceSearchFactory = Depends(get_balance_search_factory)):
    logger.info(f"llm query balance tracking received: llm_type={request.llm_type}, network={request.network}, messages={request.messages}")
    output = None
    start_time = time.time()

    try:
        llm = llm_factory.create_llm(request.llm_type)
        logger.info(f"Created LLM: {llm}")

        output, token_usage = await handle_balance_tracking_query(request, llm, balance_search_factory)

    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
        output = [
            QueryOutput(type="error", error=error_code,
                        interpreted_result=LLM_ERROR_MESSAGES.get(error_code, 'An error occurred'))]

    logger.info(f"Serving llm query balance tracking output: {output} (Total time taken: {time.time() - start_time} seconds)")

    return output


async def handle_funds_flow_query(request, llm, graph_search_factory):
    try:
        graph_search = graph_search_factory.create_graph_search(request.network)
        query_start_time = time.time()

        query, token_usage_query = llm.build_cypher_query_from_messages(request.messages, request.llm_type, request.network)
        query = query.strip('`')
        logger.info(f"Generated Cypher query: {query} (Time taken: {time.time() - query_start_time} seconds)")

        if query == 'modification_error':
            error_code = LLM_ERROR_MODIFICATION_NOT_ALLOWED
            error_message = LLM_ERROR_MESSAGES[error_code]
            logger.error(f"Error {error_code}: {error_message}")
            return [{'type': 'error', 'result': None, 'interpreted_result': error_message, 'error': error_code}], {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

        if query == 'invalid_prompt_error':
            error_code = LLM_ERROR_INVALID_SEARCH_PROMPT
            error_message = LLM_ERROR_MESSAGES[error_code]
            logger.error(f"Error {error_code}: {error_message}")
            return [{'type': 'error', 'result': None, 'interpreted_result': error_message, 'error': error_code}], {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

        execute_query_start_time = time.time()
        result = graph_search.execute_query(query)
        logger.info(f"Query execution time: {time.time() - execute_query_start_time} seconds")

        graph_search.close()
        graph_transformed_result = transform_result(result)

        chart_transformed_result = None
        if is_chart_applicable(result):
            chart_transformed_result= convert_funds_flow_to_chart(result)

        interpret_result_start_time = time.time()
        interpreted_result, token_usage_interpret = llm.interpret_result_funds_flow(llm_messages=request.messages, result=graph_transformed_result, llm_type=request.llm_type, network=request.network)
        logger.info(f"Result interpretation time: {time.time() - interpret_result_start_time} seconds")

        output = [
            QueryOutput(type="graph", result=graph_transformed_result, interpreted_result=interpreted_result),
            QueryOutput(type="text", interpreted_result="interpreted_result"),
            QueryOutput(type="table", interpreted_result="interpreted_result"),
            QueryOutput(type="chart", result=chart_transformed_result, interpreted_result="interpreted_result")
        ]

        token_usage = {
            'completion_tokens': token_usage_query.get('completion_tokens', 0) + token_usage_interpret.get('completion_tokens', 0),
            'prompt_tokens': token_usage_query.get('prompt_tokens', 0) + token_usage_interpret.get('prompt_tokens', 0),
            'total_tokens': token_usage_query.get('total_tokens', 0) + token_usage_interpret.get('total_tokens', 0)
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
        error_message = LLM_ERROR_MESSAGES.get(error_code, 'An unknown error occurred')
        output = [
            QueryOutput(type="error", error=error_code, interpreted_result=error_message)
        ]
        token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

    return output, token_usage



async def handle_balance_tracking_query(request, llm, balance_search_factory):
    try:
        query_start_time = time.time()
        query, token_usage_query = llm.build_query_from_messages_balance_tracker(request.messages, request.llm_type, request.network)
        logger.info(f"extracted query: {query} (Time taken: {time.time() - query_start_time} seconds)")

        if query in ['modification_error', 'invalid_prompt_error']:
            error_code = LLM_ERROR_MODIFICATION_NOT_ALLOWED if query == 'modification_error' else LLM_ERROR_INVALID_SEARCH_PROMPT
            error_message = LLM_ERROR_MESSAGES.get(error_code)
            logger.error(f"Error {error_code}: {error_message}")
            return [{'type': 'error', 'result': None, 'interpreted_result': error_message, 'error': error_code}], {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

        execute_query_start_time = time.time()
        balance_search = balance_search_factory.create_balance_search(request.network)
        result = balance_search.execute_query(query)
        balance_search.close()

        logger.info(f"Query execution time: {time.time() - execute_query_start_time} seconds")

        tabular_transformed_result = transform_result_set(result)

        chart_transformed_result = None
        if is_chart_applicable(result):
            chart_transformed_result = convert_balance_tracking_to_chart(result)

        interpret_result_start_time = time.time()
        interpreted_result, token_usage_interpret = llm.interpret_result_balance_tracker(
            llm_messages=request.messages,
            result=tabular_transformed_result,
            llm_type=request.llm_type,
            network=request.network
        )
        logger.info(f"Result interpretation time: {time.time() - interpret_result_start_time} seconds")

        output = [
            QueryOutput(type="graph", interpreted_result="interpreted_result"),
            QueryOutput(type="text", interpreted_result="interpreted_result"),
            QueryOutput(type="table", result=tabular_transformed_result, interpreted_result=interpreted_result),
            QueryOutput(type="chart", result=chart_transformed_result, interpreted_result=interpreted_result)
        ]

        token_usage = {
            'completion_tokens': token_usage_query.get('completion_tokens', 0) + token_usage_interpret.get('completion_tokens', 0),
            'prompt_tokens': token_usage_query.get('prompt_tokens', 0) + token_usage_interpret.get('prompt_tokens', 0),
            'total_tokens': token_usage_query.get('total_tokens', 0) + token_usage_interpret.get('total_tokens', 0)
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
        error_message = LLM_ERROR_MESSAGES.get(error_code, 'An error occurred')
        output = [
            QueryOutput(type="error", error=error_code, interpreted_result=error_message)
        ]
        token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

    return output, token_usage


def is_query_only(query_restricted_keywords, cypher_query):
    normalized_query = cypher_query.upper()
    for keyword in query_restricted_keywords:
        if keyword in normalized_query:
            return False
    return True


app.include_router(v1_router, prefix="/v1")
