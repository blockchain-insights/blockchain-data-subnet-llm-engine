import unittest
import os
from data.bitcoin.graph_search import BitcoinGraphSearch
from settings import settings


class TestGraphSearch(unittest.TestCase):
    def test_get_min_max_block_height(self):
        print("Running query for getting min max block height...")
        graph_search = BitcoinGraphSearch(settings=settings)
        min_block_height, max_block_height = graph_search.get_min_max_block_height()
        print((min_block_height, max_block_height))
        self.assertNotEqual(min_block_height, 0)
        self.assertNotEqual(max_block_height, 0)
        graph_search.close()

        
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    unittest.main()