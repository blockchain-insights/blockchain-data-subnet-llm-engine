from neo4j import GraphDatabase
from loguru import logger
from protocols.blockchain import NETWORK_BITCOIN
from protocols.llm_engine import Query

from data.bitcoin.query_builder import QueryBuilder
from settings import Settings, settings


class BaseGraphSearch:
    def execute_query(self, query: str):
        """Execute a query and return the result."""

    def execute_cypher_query(self, cypher_query: str):
        """Execute a cypher query and return the result."""

    def close(self):
        """Close the connection to the graph database."""


class GraphSearchFactory:
    @classmethod
    def create_graph_search(cls, network: str) -> BaseGraphSearch:
        graph_search_class = {
            NETWORK_BITCOIN: BitcoinGraphSearch,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if graph_search_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return graph_search_class(settings)


class BitcoinGraphSearch(BaseGraphSearch):
    def __init__(self, settings: Settings):
        logger.info(f'Here is loaded configs {settings.GRAPH_DB_URL}')
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
            return result.data()

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


def get_graph_search(settings, network):
    switch = {
        NETWORK_BITCOIN: lambda: BitcoinGraphSearch(settings),
    }
    return switch[network]()