from neo4j import GraphDatabase

from db.query_builder import QueryBuilder
from protocol import QueryOutput, Query, NETWORK_BITCOIN
from settings import Settings, settings


class BaseGraphSearch:
    def execute_query(self, query: Query) -> QueryOutput:
        """Execute a query and return the result."""

    def execute_cypher_query(self, cypher_query: str):
        """Execute a cypher query and return the result."""

    def close(self):
        """Close the connection to the graph database."""


class GraphSearchFactory:
    @classmethod
    def create_graph_search(cls, network: str) -> BaseGraphSearch:
        llm_class = {
            NETWORK_BITCOIN: BitcoinGraphSearch(settings),
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if llm_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return llm_class()


class BitcoinGraphSearch(BaseGraphSearch):
    def __init__(self, settings: Settings):
        self.driver = GraphDatabase.driver(
            settings.graph_db_url,
            auth=(settings.graph_db_user, settings.graph_db_password),
        )

    def close(self):
        self.driver.close()

    def execute_query(self, query: Query) -> QueryOutput:
        # build cypher query
        try:
            cypher_query = QueryBuilder.build_query(query)
        except Exception as e:
            raise Exception(f"query parse error: {e}")

        # execute cypher query
        try:
            result = self.execute_cypher_query(cypher_query)
            return result
        except Exception as e:
            raise Exception(f"cypher query execution error: {e}")

    def execute_cypher_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return result

