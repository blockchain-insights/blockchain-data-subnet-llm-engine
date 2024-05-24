import unittest
import os

from db.query_builder import QueryBuilder
from protocol import QUERY_TYPE_SEARCH, Query


class TestGraphSearch(unittest.TestCase):
    def test_build_search_query(self):
        # test case 1
        query = Query(type=QUERY_TYPE_SEARCH, target='Transaction', limit=20)
        cypher_query = QueryBuilder.build_query(query)
        self.assertEqual(cypher_query, "MATCH (t:Transaction)\nRETURN t\nLIMIT 20;")
        
        # test case 2
        query = Query(type=QUERY_TYPE_SEARCH, target='Transaction', limit=20, where={
            "tx_id": "0123456789"
        })
        cypher_query = QueryBuilder.build_query(query)
        self.assertEqual(cypher_query, 'MATCH (t:Transaction{tx_id: "0123456789"})\nRETURN t\nLIMIT 20;')
        
        # test case 3
        query = Query(type=QUERY_TYPE_SEARCH, target='Transaction', limit=20, where={
            "from_address": "123",
            "to_address": "456",
        })
        cypher_query = QueryBuilder.build_query(query)
        self.assertEqual(cypher_query, 'MATCH (a1:Address{address: "123"})-[s1:SENT]->(t:Transaction)-[s2:SENT]->(a2:Address{address: "456"})\nRETURN t\nLIMIT 20;')
        
        # test case 4
        query = Query(type=QUERY_TYPE_SEARCH, target='Transaction', limit=20, where={
           "amount_range": {
               "from": 2000,
               "to": 3000,
           },
           "timestamp_range": {
               "from": 4000,
           },
        })
        cypher_query = QueryBuilder.build_query(query)
        self.assertEqual(cypher_query, 'MATCH (t:Transaction)\nWHERE t.out_total_amount >= 2000 AND t.out_total_amount <= 3000 AND t.timestamp >= 4000\nRETURN t\nLIMIT 20;')
        
if __name__ == '__main__':
    unittest.main()
