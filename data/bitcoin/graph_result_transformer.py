def transform_result(result):
    output_data = []
    transaction_ids = set()
    address_ids = set()

    for entry in result:
        process_transaction_entry(entry, output_data, transaction_ids, address_ids)
    return output_data

# Function to convert satoshi to BTC
def satoshi_to_btc(satoshi):
    return satoshi / 1e8

# Helper function to process a transaction entry
# Helper function to process a transaction entry
def process_transaction_entry(entry, output_data, transaction_ids, address_ids):
    for key, value in entry.items():
        if key.startswith('a'):
            address = value.get('address')
            if address and address not in address_ids:
                output_data.append({
                    "id": address,
                    "type": "node",
                    "label": "address"
                })
                address_ids.add(address)

        elif key.startswith('t'):
            transaction = value
            tx_id = transaction.get('tx_id')
            if tx_id and tx_id not in transaction_ids:
                output_data.append({
                    "id": tx_id,
                    "type": "node",
                    "label": "transaction"
                })
                transaction_ids.add(tx_id)

        elif key.startswith('s'):
            sent_transaction = value
            if isinstance(sent_transaction, tuple) and len(sent_transaction) == 3:
                sent_address = sent_transaction[0].get('address')
                tx_id = sent_transaction[2].get('tx_id')
                value_satoshi = sent_transaction[2].get('out_total_amount')

                if sent_address and sent_address not in address_ids:
                    output_data.append({
                        "id": sent_address,
                        "type": "node",
                        "label": "address"
                    })
                    address_ids.add(sent_address)

                if tx_id and tx_id not in transaction_ids:
                    output_data.append({
                        "id": tx_id,
                        "type": "node",
                        "label": "transaction"
                    })
                    transaction_ids.add(tx_id)

                if sent_address and tx_id and value_satoshi is not None:
                    output_data.append({
                        "id": f"{sent_address}-{tx_id}",
                        "type": "edge",
                        "label": f"{satoshi_to_btc(value_satoshi):.8f} BTC",
                        "from_id": sent_address,
                        "to_id": tx_id
                    })
