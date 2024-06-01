import unittest

from protocols.llm_engine import LlmMessage, LLM_MESSAGE_TYPE_USER, LLM_ERROR_TYPE_NOT_SUPPORTED

from data.bitcoin.graph_search import BitcoinGraphSearch
from llm.openai import OpenAILLM
from settings import settings


class TestLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = OpenAILLM(settings=settings)
        self.graph_search = BitcoinGraphSearch(settings=settings)
    
    def tearDown(self) -> None:
        self.graph_search.close()
    
    def test_build_query(self):
        # test case 1
        query_text = "Return 15 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
        query = self.llm.build_query_from_messages([
            LlmMessage(
                type=LLM_MESSAGE_TYPE_USER,
                content=query_text
            )
        ])
        expected_query = {
            "type": "search",
            "target": "Transaction",
            "where": {
                "from_address": "bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
            },
            "limit": 15,
            "skip": 0
        }
        self.assertEqual(query.type, expected_query["type"])
        self.assertEqual(query.target, expected_query["target"])
        self.assertDictEqual(query.where, expected_query["where"])
        self.assertEqual(query.limit, expected_query["limit"])
        self.assertEqual(query.skip, expected_query["skip"])

        # test case 2
        query_text = "I have sent more than 1.5 BTC to somewhere but I couldn't remember. Show me relevant transactions. My address is bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
        query = self.llm.build_query_from_messages([
            LlmMessage(
                type=LLM_MESSAGE_TYPE_USER,
                content=query_text
            )
        ])
        expected_query = {
            "type": "search",
            "target": "Transaction",
            "where": {
                "from_address": "bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r",
                "amount_range": {
                    "from": 1.5
                }
            },
            "limit": None,
            "skip": 0
        }
        self.assertEqual(query.type, expected_query["type"])
        self.assertEqual(query.target, expected_query["target"])
        self.assertDictEqual(query.where, expected_query["where"])
        self.assertEqual(query.limit, expected_query["limit"])
        self.assertEqual(query.skip, expected_query["skip"])
        
    def test_llm_query_handler(self):
        query_text = "Return 15 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
        llm_messages = [
            LlmMessage(
                type=LLM_MESSAGE_TYPE_USER,
                content=query_text
            )
        ]
        query = self.llm.build_query_from_messages(llm_messages)
        result = self.graph_search.execute_query(query=query)
        interpreted_result = self.llm.interpret_result(llm_messages=llm_messages, result=result)
        print("--- Interpreted result ---")
        print(interpreted_result)

    def test_edge_cases(self):
        with self.assertRaises(Exception) as context:
            query_text = "What is React.js?"
            llm_messages = [
                LlmMessage(
                    type=LLM_MESSAGE_TYPE_USER,
                    content=query_text
                )
            ]
            query = self.llm.build_query_from_messages(llm_messages)
            result = self.graph_search.execute_query(query=query)
            interpreted_result = self.llm.interpret_result(llm_messages=llm_messages, result=result)
        self.assertEqual(str(context.exception), str(LLM_ERROR_TYPE_NOT_SUPPORTED))


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    unittest.main()
