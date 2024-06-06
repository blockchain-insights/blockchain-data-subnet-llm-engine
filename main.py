import asyncio
import json
import time
import traceback
from typing import List
from pathlib import Path as FilePath

import protocols.blockchain
from loguru import logger
from fastapi import FastAPI, Request, Depends, Query, Body, APIRouter, HTTPException
from protocols.llm_engine import LlmMessage, QueryOutput, LLM_ERROR_TYPE_NOT_SUPPORTED, LLM_ERROR_MESSAGES
from pydantic import BaseModel, Field

import __init__
from data.bitcoin.balance_search import get_balance_search
from data.bitcoin.graph_search import GraphSearchFactory, get_graph_search
from llm.factory import LLMFactory
from settings import settings

app = FastAPI(
    title="Blockchain Insights LLM ENGINE",
    description="API designed to execute user prompts related to blockchain queries using LLM agents. It integrates with different LLMs and graph search functionalities to process and interpret blockchain data.",
    version=__init__.__version__,)


benchmark_restricted_keywords = ['CREATE', 'SET', 'DELETE', 'DETACH', 'REMOVE', 'MERGE', 'CREATE INDEX', 'DROP INDEX', 'CREATE CONSTRAINT', 'DROP CONSTRAINT']

def get_llm_factory() -> LLMFactory:
    return LLMFactory()


def get_graph_search_factory() -> GraphSearchFactory:
    return GraphSearchFactory()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await asyncio.wait_for(call_next(request), timeout=3*60)
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


v1_router = APIRouter()
valid_networks = [protocols.blockchain.NETWORK_BITCOIN, protocols.blockchain.NETWORK_ETHEREUM]


@v1_router.get("/networks", summary="Get supported networks", description="Get the list of supported networks", tags=["v1"])
async def get_networks():
    return {"networks": valid_networks}



@v1_router.get("/schema/{network}", summary="Get schema for network", description="Get the schema for the specified network", tags=["v1"])
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


@v1_router.get("/discovery/{network}", summary="Get network discovery", description="Get the network discovery details", tags=["v1"])
async def discovery_v1(network: str):
    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")
    if network == protocols.blockchain.NETWORK_BITCOIN:
        graph_search = get_graph_search(settings, network)
        balance_search = get_balance_search(settings, network)
        funds_flow_model_start_block, funds_flow_model_last_block = graph_search.get_min_max_block_height_cache()
        balance_model_last_block = balance_search.get_latest_block_number()
        graph_search.close()
        return {
            "network": network,
            "funds_flow_model_start_block": funds_flow_model_start_block,
            "funds_flow_model_ast_block": funds_flow_model_last_block,
            "balance_model_last_block": balance_model_last_block
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid network")


@v1_router.get("/challenge/{network}",
               summary="Solve challenge",
               description="Solve the challenge", tags=["v1"])
async def challenge_v1(network: str,
                            in_total_amount: int = Query(None, description="Input total amount"),
                            out_total_amount: int = Query(None, description="Output total amount"),
                            tx_id_last_4_chars: str = Query(None, description="Transaction ID last 4 characters"),
                            checksum: str = Query(None, description="Checksum query parameter")):

    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")

    if network == protocols.blockchain.NETWORK_BITCOIN:
        if in_total_amount is None or out_total_amount is None or tx_id_last_4_chars is None:
            raise HTTPException(status_code=400, detail="Missing required query parameters for Bitcoin network, required: in_total_amount, out_total_amount, tx_id_last_4_chars")
        graph_search = get_graph_search(settings, network)
        output = graph_search.solve_challenge(
            in_total_amount=in_total_amount,
            out_total_amount=out_total_amount,
            tx_id_last_4_chars=tx_id_last_4_chars
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


@v1_router.get("/benchmark/{network}", summary="Benchmark query", description="Benchmark the query", tags=["v1"])
async def benchmark_v1(network: str, query: str = Query(..., description="Query to benchmark")):
    def is_query_only(query_restricted_keywords, cypher_query):
        normalized_query = cypher_query.upper()
        for keyword in query_restricted_keywords:
            if keyword in normalized_query:
                return False
        return True

    if not is_query_only(benchmark_restricted_keywords, query):
        raise HTTPException(status_code=400, detail="Invalid query, restricted keywords found in query")

    if network not in valid_networks:
        raise HTTPException(status_code=400, detail="Invalid network")
    if network == protocols.blockchain.NETWORK_BITCOIN:
        graph_search = get_graph_search(settings, network)
        output = graph_search.execute_benchmark_query(query)
        graph_search.close()
        return {
            "network": network,
            "output": output[0],
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid network")

@v1_router.post("/process_prompt", summary="Executes user prompt", description="Execute user prompt and return the result", tags=["v1"], response_model=List[QueryOutput])
async def llm_query_v1(
        request: LLMQueryRequestV1 = Body(..., example={"llm_type": "openai", "network": "bitcoin", "messages": [{"type": 0, "content": "Return 15 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"}]}),
        llm_factory: LLMFactory = Depends(get_llm_factory),
        graph_search_factory: GraphSearchFactory = Depends(get_graph_search_factory)):

    logger.info(f"llm query received: {request.llm_type}, network: {request.network}")

    output = None
    start_time = time.time()

    llm = llm_factory.create_llm(request.llm_type)

    try:
        query_start_time = time.time()
        query = llm.build_query_from_messages(request.messages)
        logger.info(f"extracted query: {query} (Time taken: {time.time() - query_start_time} seconds)")

        graph_search = graph_search_factory.create_graph_search(request.network)

        execute_query_start_time = time.time()
        result = graph_search.execute_query(query=query)
        logger.info(f"Query execution time: {time.time() - execute_query_start_time} seconds")

        interpret_result_start_time = time.time()
        interpreted_result = llm.interpret_result(llm_messages=request.messages, result=result)
        logger.info(f"Result interpretation time: {time.time() - interpret_result_start_time} seconds")

        graph_search.close()
        output = [
            QueryOutput(type="graph", result=result, interpreted_result=interpreted_result),
            QueryOutput(type="text", interpreted_result="interpreted_result"),
            QueryOutput(type="table", interpreted_result="interpreted_result")
        ]

    except Exception as e:
        logger.error(traceback.format_exc())
        error_code = e.args[0]
        if error_code == LLM_ERROR_TYPE_NOT_SUPPORTED:
            # handle unsupported query templates
            try:
                interpreted_result = llm.excute_generic_query(llm_message=request.messages[-1].content)
                if interpreted_result == "Failed":
                    interpreted_result = llm.generate_general_response(llm_messages=request.messages)
                    output = [QueryOutput(error=error_code, interpreted_result=interpreted_result)]
                else:
                    output = [QueryOutput(error=error_code, interpreted_result=interpreted_result)]
            except Exception as e:
                error_code = e.args[0]
                output = [QueryOutput(error=error_code, interpreted_result=LLM_ERROR_MESSAGES[error_code])]
        else:
            output = [QueryOutput(error=error_code, interpreted_result=LLM_ERROR_MESSAGES[error_code])]

    logger.info(f"Serving miner llm query output: {output} (Total time taken: {time.time() - start_time} seconds)")

    return output



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, BigInteger, String, TIMESTAMP, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

app = FastAPI()

# Define SQLAlchemy model
Base = declarative_base()


class BalanceChange(Base):
    __tablename__ = 'balance_changes'
    address = Column(String, primary_key=True)
    block = Column(Integer, primary_key=True)
    d_balance = Column(BigInteger)
    block_timestamp = Column(TIMESTAMP)


# Database connection setup
DATABASE_URL = os.getenv('DB_CONNECTION_STRING_MINER', "postgresql://postgres:changeit456$@localhost:5432/miner")
ENGINE = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

Base.metadata.create_all(bind=ENGINE)

# LangChain setup
llm = OpenAI(api_key=os.getenv('OPENAI_AI_KEY', 'sk-proj-cbnuinw4gHUgaK2BJdWsT3BlbkFJBMFkJZs0uYF5dtwrCtm2'))

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an assistant to help me query balance changes.
    I will ask you questions, and you will generate SQL queries to fetch the data.

    The database table is called `balance_changes` and has the following columns:
    - address (string)
    - block (integer)
    - d_balance (big integer)
    - block_timestamp (timestamp)

    For example:
    "Return the address with the highest amount of BTC in December 2009."

    My question: {question}
    SQL query:
    """
)

# Create an LLMChain with the prompt template and LLM
llm_chain = LLMChain(prompt=prompt_template, llm=llm)


# FastAPI request model
class QueryRequest(BaseModel):
    prompt: str


# Endpoint to handle queries
@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Translate the natural language prompt into an SQL query
        sql_query = llm_chain.run(question=request.prompt).strip()

        # Execute the SQL query
        with SessionLocal() as session:
            result = session.execute(text(sql_query))
            columns = result.keys()
            rows = result.fetchall()

        # Convert the result to a list of dictionaries
        results = [dict(zip(columns, row)) for row in rows]

        # Return the results
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(v1_router, prefix="/v1")