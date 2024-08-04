from protocols.blockchain import NETWORK_BITCOIN, NETWORK_ETHEREUM
from data.utils.base_search import BaseGraphSearch, BaseBalanceSearch
from data.bitcoin.graph_search import BitcoinGraphSearch
from data.bitcoin.balance_search import BitcoinBalanceSearch
from data.ethereum.graph_search import EthereumGraphSearch

from settings import settings

class GraphSearchFactory:
    @classmethod
    def create_graph_search(cls, network: str) -> BaseGraphSearch:
        graph_search_class = {
            NETWORK_BITCOIN: BitcoinGraphSearch,
            NETWORK_ETHEREUM: EthereumGraphSearch,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if graph_search_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return graph_search_class(settings)

class BalanceSearchFactory:
    @classmethod
    def create_balance_search(cls, network: str) -> BaseBalanceSearch:
        graph_search_class = {
            NETWORK_BITCOIN: BitcoinBalanceSearch,
            # Add other networks and their corresponding classes as needed
        }.get(network)

        if graph_search_class is None:
            raise ValueError(f"Unsupported network Type: {network}")

        return graph_search_class(settings.DB_CONNECTION_STRING)
