from neo4j import GraphDatabase

from db.query_builder import QueryBuilder
from protocol import QueryOutput, Query, NETWORK_BITCOIN
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
        self.driver = GraphDatabase.driver(
            settings.GRAPH_DB_URL,
            auth=(settings.GRAPH_DB_USER, settings.GRAPH_DB_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def execute_query(self, query: Query):
        # build cypher query
        cypher_query = QueryBuilder.build_query(query)
        # execute cypher query
        result = self.execute_cypher_query(cypher_query)
        return result

    def execute_cypher_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            if not result:
                return None
            return result.data()


