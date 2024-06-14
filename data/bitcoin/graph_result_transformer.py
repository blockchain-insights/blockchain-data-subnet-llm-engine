def transform_result(result):
    output_data = []
    transaction_ids = set()
    address_ids = set()
    edge_ids = set()

    for entry in result:
        process_transaction_entry(entry, output_data, transaction_ids, address_ids, edge_ids)
    return output_data

# Function to convert satoshi to BTC
def satoshi_to_btc(satoshi):
    return satoshi / 1e8

# Helper function to process a transaction entry
def process_transaction_entry(entry, output_data, transaction_ids, address_ids, edge_ids):
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
            sent_transaction = value
            if isinstance(sent_transaction, tuple) and len(sent_transaction) == 3:
                process_sent_edge(sent_transaction, output_data, edge_ids)

# Helper function to process a SENT edge
def process_sent_edge(sent_transaction, output_data, edge_ids):
    from_node = sent_transaction[0]
    to_node = sent_transaction[2]

    if 'address' in from_node and 'tx_id' in to_node:
        sent_address = from_node.get('address')
        tx_id = to_node.get('tx_id')
        value_satoshi = to_node.get('out_total_amount')

        edge_id = f"{sent_address}-{tx_id}"
        if sent_address and tx_id and edge_id not in edge_ids:
            edge_label = f"{satoshi_to_btc(value_satoshi):.8f} BTC" if value_satoshi is not None else "SENT"
            output_data.append({
                "id": edge_id,
                "type": "edge",
                "label": edge_label,
                "from_id": sent_address,
                "to_id": tx_id
            })
            edge_ids.add(edge_id)

    elif 'tx_id' in from_node and 'address' in to_node:
        tx_id = from_node.get('tx_id')
        sent_address = to_node.get('address')
        value_satoshi = from_node.get('out_total_amount')

        edge_id = f"{tx_id}-{sent_address}"
        if sent_address and tx_id and edge_id not in edge_ids:
            edge_label = f"{satoshi_to_btc(value_satoshi):.8f} BTC" if value_satoshi is not None else "SENT"
            output_data.append({
                "id": edge_id,
                "type": "edge",
                "label": edge_label,
                "from_id": tx_id,
                "to_id": sent_address
            })
            edge_ids.add(edge_id)