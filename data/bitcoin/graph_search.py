from neo4j import GraphDatabase
from loguru import logger
from protocols.llm_engine import Query
from data.utils.base_search import BaseGraphSearch

from data.bitcoin.query_builder import QueryBuilder

from settings import Settings

class BitcoinGraphSearch(BaseGraphSearch):
    def __init__(self, settings: Settings):
        logger.info(f'Here is loaded configs {settings.BITCOIN_GRAPH_DB_URL}')
        self.driver = GraphDatabase.driver(
            settings.BITCOIN_GRAPH_DB_URL,
            auth=(settings.BITCOIN_GRAPH_DB_USER, settings.BITCOIN_GRAPH_DB_PASSWORD),
            connection_timeout=60,
            max_connection_lifetime=60,
            max_connection_pool_size=128,
            encrypted=False,
        )

    def close(self):
        self.driver.close()

    def execute_predefined_query(self, query: Query):
        cypher_query = QueryBuilder.build_query(query)
        logger.info(f"Executing cypher query: {cypher_query}")
        result = self.execute_cypher_query(cypher_query)
        return result

    def execute_query(self, query: str):
        logger.info(f"Executing cypher query: {query}")
        result = self.execute_cypher_query(query)
        return result

    def execute_cypher_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            if not result:
                return None

            results_data = []
            for record in result:
                # Extract nodes and relationships from the record
                a1 = record['a1']
                t1 = record['t1']
                a2 = record['a2']
                s1 = record['s1']
                s2 = record['s2']

                results_data.append({
                    'a1': a1,
                    't1': t1,
                    'a2': a2,
                    's1': dict(s1),
                    's2': dict(s2)
                })

            return results_data

    def get_min_max_block_height_cache(self):
        with self.driver.session() as session:
            result_min = session.run(
                """
                MATCH (n:Cache {field: 'min_block_height'})
                RETURN n.value
                LIMIT 1;
                """
            ).single()

            result_max = session.run(
                """
                MATCH (n:Cache {field: 'max_block_height'})
                RETURN n.value
                LIMIT 1;
                """
            ).single()

            min_block_height = result_min[0] if result_min else 0
            max_block_height = result_max[0] if result_max else 0

            return min_block_height, max_block_height

    def solve_challenge(self, in_total_amount: int, out_total_amount: int, tx_id_last_6_chars: str) -> str:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction {out_total_amount: $out_total_amount})
                WHERE t.in_total_amount = $in_total_amount AND t.tx_id ENDS WITH $tx_id_last_6_chars
                RETURN t.tx_id
                LIMIT 1;
                """,
                in_total_amount=in_total_amount,
                out_total_amount=out_total_amount,
                tx_id_last_6_chars=tx_id_last_6_chars
            )
            single_result = result.single()
            if single_result is None or single_result[0] is None:
                return None
            return single_result[0]

    def execute_benchmark_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return result.single()
