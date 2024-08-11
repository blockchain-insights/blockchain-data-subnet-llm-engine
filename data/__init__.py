from protocols.blockchain import NETWORK_BITCOIN
from data.utils.base_search import BaseGraphSearch, BaseBalanceSearch
from data.bitcoin.graph_search import BitcoinGraphSearch
from data.bitcoin.balance_search import BitcoinBalanceSearch
from data.bitcoin.chart_result_transformer import BitcoinChartTransformer
from data.bitcoin.graph_result_transformer import BitcoinGraphTransformer
from data.bitcoin.tabular_result_transformer import BitcoinTabularTransformer
from data.utils.base_transformer import BaseChartTransformer
from data.utils.base_transformer import BaseGraphTransformer
from data.utils.base_transformer import BaseTabularTransformer

from sqlalchemy.ext.asyncio import AsyncSession

from settings import settings

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

class BalanceSearchFactory:
    @classmethod
    def create_balance_search(cls, network: str, db: AsyncSession) -> BaseBalanceSearch:
        graph_search_class = {
            NETWORK_BITCOIN: BitcoinBalanceSearch,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if graph_search_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return graph_search_class(db)

class GraphTransformerFactory:
    @classmethod
    def create_graph_transformer(cls, network: str) -> BaseGraphTransformer:
        transformer_class = {
            NETWORK_BITCOIN: BitcoinGraphTransformer,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if transformer_class is None:
            raise ValueError(f"Unsupported network type: {network}")

        return transformer_class()

class ChartTransformerFactory:
    @classmethod
    def create_chart_transformer(cls, network: str) -> BaseChartTransformer:
        transformer_class = {
            NETWORK_BITCOIN: BitcoinChartTransformer,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if transformer_class is None:
            raise ValueError(f"Unsupported network type: {network}")

        return transformer_class()

class TabularTransformerFactory:
    @classmethod
    def create_tabular_transformer(cls, network: str) -> BaseTabularTransformer:
        transformer_class = {
            NETWORK_BITCOIN: BitcoinTabularTransformer,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if transformer_class is None:
            raise ValueError(f"Unsupported network type: {network}")

        return transformer_class()