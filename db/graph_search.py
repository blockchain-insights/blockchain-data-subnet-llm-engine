import os
from neo4j import GraphDatabase

from db.query_builder import QueryBuilder
from protocol import QueryOutput, Query, NETWORK_BITCOIN


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
            NETWORK_BITCOIN: BitcoinGraphSearch,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if llm_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return llm_class()


class BitcoinGraphSearch(BaseGraphSearch):
    def __init__(
            self,
            graph_db_url: str = None,
            graph_db_user: str = None,
            graph_db_password: str = None,
    ):
        if graph_db_url is None:
            self.graph_db_url = (
                    os.environ.get("GRAPH_DB_URL") or "bolt://localhost:7687"
            )
        else:
            self.graph_db_url = graph_db_url

        if graph_db_user is None:
            self.graph_db_user = os.environ.get("GRAPH_DB_USER") or ""
        else:
            self.graph_db_user = graph_db_user

        if graph_db_password is None:
            self.graph_db_password = os.environ.get("GRAPH_DB_PASSWORD") or ""
        else:
            self.graph_db_password = graph_db_password

        self.driver = GraphDatabase.driver(
            self.graph_db_url,
            auth=(self.graph_db_user, self.graph_db_password),
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


