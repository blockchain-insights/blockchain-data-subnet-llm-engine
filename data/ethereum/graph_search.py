import os
import typing

from neo4j import GraphDatabase
from settings import Settings
from data.ethereum.utils import is_malicious
from loguru import logger

class EthereumGraphSearch:    
    def __init__(self, settings: Settings):
        self.driver = GraphDatabase.driver(
            settings.GRAPH_DB_URL,
            auth=(settings.GRAPH_DB_USER, settings.GRAPH_DB_PASSWORD),
            connection_timeout=60,
            max_connection_lifetime=60,
            max_connection_pool_size=128,
            encrypted=False,
        )
    def close(self):
        self.driver.close()
    
    def get_min_max_block_height(self) -> typing.Tuple[int, int]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction)
                RETURN MIN(t.block_number) AS min_block_height, MAX(t.block_number) AS max_block_height
                """
            )
            single_result = result.single()
            if not single_result:
                return 0, 0
            return single_result.get('min_block_height', 0), single_result.get('max_block_height', 0)

    def get_min_max_block_height_cache(self):
        with self.driver.session() as session:
            result_min = session.run(
                """
                MATCH (n:Cache {field: 'min_block_height'})
                RETURN n.value;
                """
            ).single()
            
            result_max = session.run(
                """
                MATCH (n:Cache {field: 'max_block_height'})
                RETURN n.value;
                """
            ).single()
            
            min_block_height = result_min[0] if result_min else 0
            max_block_height = result_max[0] if result_max else 0

            return min_block_height, max_block_height

    def execute_benchmark_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return result.single()
        
    def execute_query(self, query: str):
        if is_malicious(query):
            logger.warning(f"Potentially malicious query detected: {query}")
            return None

        with self.driver.session() as session:
            return session.run(query)
        
    def solve_challenge(self, checksum: str):
        with self.driver.session() as session:
            data_set = session.run(
                """
                MATCH (s: Checksum { checksum: $checksum })
                RETURN s.tx_hash
                """,
                checksum=checksum
            )
            single_result = data_set.single()

            if single_result[0] is None:
                return 0
            return single_result[0]