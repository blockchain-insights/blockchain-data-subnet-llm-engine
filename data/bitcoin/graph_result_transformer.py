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
def process_transaction_entry(entry, output_data, transaction_ids, address_ids):
    for key, value in entry.items():
        if key.startswith('a'):
            address = value['address']
            if address not in address_ids:
                output_data.append({
                    "id": address,
                    "type": "node",
                    "label": "address"
                })
                address_ids.add(address)

        elif key.startswith('t'):
            transaction = value
            tx_id = transaction['tx_id']
            if tx_id not in transaction_ids:
                output_data.append({
                    "id": tx_id,
                    "type": "node",
                    "label": "transaction"
                })
                transaction_ids.add(tx_id)

        elif key.startswith('s'):
            for s in value:
                if isinstance(s, dict) and 'address' in s:
                    sent_address = s['address']
                    if sent_address not in address_ids:
                        output_data.append({
                            "id": sent_address,
                            "type": "node",
                            "label": "address"
                        })
                        address_ids.add(sent_address)
                    tx_id = entry['t']['tx_id']
                    value_satoshi = entry['t']['out_total_amount']
                    output_data.append({
                        "id": f"{sent_address}-{tx_id}",
                        "type": "edge",
                        "label": f"{satoshi_to_btc(value_satoshi):.8f} BTC",
                        "from_id": sent_address,
                        "to_id": tx_id
                    })