def transform_result(result):
    print(result)  # Add this line to inspect the result structure
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
    address = entry.get('address') or entry.get('a1', {}).get('address')
    recipient = entry.get('recipient') or entry.get('a2', {}).get('address')
    transaction = entry.get('t1', {})

    # Ensure all necessary fields are in the transaction dictionary
    if 'balance' in entry or 'timestamp' in entry or 'block_height' in entry:
        transaction = {
            'tx_id': transaction.get('tx_id') or f"{address}_{recipient}_{entry.get('timestamp')}",
            'out_total_amount': transaction.get('out_total_amount') or entry.get('balance'),
            'timestamp': transaction.get('timestamp') or entry.get('timestamp'),
            'block_height': transaction.get('block_height') or entry.get('block_height')
        }

    if address and address not in address_ids:
        output_data.append({
            "id": address,
            "type": "node",
            "label": "address"
        })
        address_ids.add(address)

    if recipient and recipient not in address_ids:
        output_data.append({
            "id": recipient,
            "type": "node",
            "label": "address"
        })
        address_ids.add(recipient)

    tx_id = transaction.get('tx_id')
    if tx_id and tx_id not in transaction_ids:
        output_data.append({
            "id": tx_id,
            "type": "node",
            "label": "transaction",
            "balance": transaction.get('out_total_amount'),
            "timestamp": transaction.get('timestamp'),
            "block_height": transaction.get('block_height')
        })
        transaction_ids.add(tx_id)

    # Process edges
    sent_transaction = entry.get('s1', (address, 'SENT', transaction))
    process_sent_edge(sent_transaction, output_data, edge_ids)
    sent_transaction = entry.get('s2', (transaction, 'SENT', recipient))
    process_sent_edge(sent_transaction, output_data, edge_ids)


# Helper function to process a SENT edge
def process_sent_edge(sent_transaction, output_data, edge_ids):
    from_node, _, to_node = sent_transaction

    if isinstance(from_node, dict) and 'address' in from_node and isinstance(to_node, dict) and 'tx_id' in to_node:
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

    elif isinstance(from_node, dict) and 'tx_id' in from_node and isinstance(to_node, dict) and 'address' in to_node:
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
