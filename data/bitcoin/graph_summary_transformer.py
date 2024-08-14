from data.utils.base_transformer import BaseGraphSummaryTransformer
from typing import List, Dict, Any

class BitcoinGraphSummaryTransformer(BaseGraphSummaryTransformer):
    def __init__(self):
        self.total_transactions = 0
        self.incoming_amount = 0.0
        self.outgoing_amount = 0.0
        self.processed_transactions = set()

    def transform_result(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.total_transactions = 0
        self.incoming_amount = 0.0
        self.outgoing_amount = 0.0
        self.processed_transactions = set()

        for entry in result:
            transaction_node = entry.get('t1')

            if transaction_node:
                tx_id = transaction_node.get('tx_id')

                # Process the transaction only if it hasn't been processed yet
                if tx_id and tx_id not in self.processed_transactions:
                    self.processed_transactions.add(tx_id)
                    self.total_transactions += 1

                    # Add the total amounts directly from the transaction node
                    in_total_amount = transaction_node.get('in_total_amount', 0)
                    out_total_amount = transaction_node.get('out_total_amount', 0)

                    self.incoming_amount += in_total_amount
                    self.outgoing_amount += out_total_amount

                    print(f"Transaction processed: {tx_id}")
                    print(f"Incoming amount: {in_total_amount} satoshi, Outgoing amount: {out_total_amount} satoshi")

        return {
            "total_transactions": self.total_transactions,
            "incoming_amount": self.satoshi_to_btc(self.incoming_amount),
            "outgoing_amount": self.satoshi_to_btc(self.outgoing_amount)
        }

    @staticmethod
    def satoshi_to_btc(satoshi: int) -> float:
        return satoshi / 1e8
