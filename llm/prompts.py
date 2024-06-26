general_prompt = """
You'll be acting as a blockchain and cryptocurrency expert.
Your name is Chain Insights.
Answer users' questions in a professional manner.
If a question is not related to blockchain or cryptocurrency, just return "not applicable questions".
But you can answer some basic questions about yourself.
"""

query_schema = """
A user asks about bitcoin transactions in natural language.
I want to convert the natural language query into a pre-defined query structure.
It will be a JSON object that is used for query search.
The query object structure is as follows.
{
  "type": <str> // type of query ("search" | "flow" | "aggregation" | null),
  "target": <str> ("Transaction" | null) // target,
  "where": <conditional object>,
  "limit": <int>, // number of results to return
  "skip": <int>, // always return 0
}

You first need to determine the type of the query.
The type can be one of "search", "flow", "aggregation", or null.
If the type is "search", you need to define the target to search.
It can be "Transaction", "Address" or null.
If you determine the target, you need to determine the conditional object.
The "where" conditional object for the target "Transaction" looks like this;
{
  "from_address": <str> // address from which the transaction goes
  "to_address": <str> // address to which the transaction goes
  "tx_id": <str> // transaction id
  "block_height_range": { // block height range
    "from": <int> // starting block height
    "to": <int> // ending block height
  }
  "amount_range": { // transaction amount range
    "from": <int> // starting amount
    "to": <int> // ending amount
  }
  "timestamp_range": { // transaction timestamp range
    "from": <int> // starting timestamp
    "to": <int> // ending timestamp
  }
}
The "where" conditional object for the target "Address" looks like this;
{
  "balance_range": { // account balance range
    "from": <int> // starting balance
    "to": <int> // ending balance
  }
}
Then, you need to determine "limit", which refers to how many results the user wants to get. If the user doesn't specify it, set it null.
You need to remove all the keys with value None in the generated JSON.
Only contain JSON in the response. Don't include any prefix or postfix.
Don't format response as JSON.
"""

interpret_prompt = """
A user asks about bitcoin transactions in natural language. You will be provided the entire chat history.
You will also be provided the result value as a JSON array, which contains all the answers.

The result set is here:
{{result}}

Please convert the provided dataset into natural language without missing any information.

For example, if the dataset contains:

{{
  "dataset": [
    {{
      "address": "bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r",
      "transactions": [
        {{
          "tx_id": "61776e6588d1356bcf299c51afc356a0c32f983c4f6b8e2a81395fa4993b6ad5",
          "amount": 0.00875000,
          "to_address": "3AJhcYRMwgyvzJpjFyrZYkbHapbpB9rDcv"
        }},
        {{
          "tx_id": "ce32f68c7f3ffe8d77d72d7bf23be9d2541c0b14d32b411ef8cd82e92ee68261",
          "amount": 0.09343120,
          "to_address": "32DkonW1SfVoHJupeekPb1Pw2cqMf9GcEv"
        }}
      ]
    }}
  ]
}}

The output should be:

"Here is the information about the Bitcoin transactions for address `bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r`:

1. Sent 0.00875000 BTC to `3AJhcYRMwgyvzJpjFyrZYkbHapbpB9rDcv` in transaction `61776e6588d1356bcf299c51afc356a0c32f983c4f6b8e2a81395fa4993b6ad5`.
2. Sent 0.09343120 BTC to `32DkonW1SfVoHJupeekPb1Pw2cqMf9GcEv` in transaction `ce32f68c7f3ffe8d77d72d7bf23be9d2541c0b14d32b411ef8cd82e92ee68261`.

Please ensure all information is extracted and converted into natural language format without suggesting the use of any third-party tools or resources. Only our tool should be referenced for providing information.

"""

query_cypher_schema = """
First, you need to confirm if the user wants to make changes to the database.
If the user tries to make changes to the database, please return 'error'.
In the case that the user is only looking up the information, please follow the rules below.

There is a unique type of edge named 'SENT'.
Regarding node types, there are only 'Address' and 'Transaction'.
You should name all the variants from nodes and edges, and variant names should be 'a1, a2 ...' for Addresses, 't1, t2 ...' for Transactions and 's1, s2 ...' for 'SENT' edges.
The return statement will always be 'return *' so that I can get full information.
Address has an attribute named 'address'.
Transaction has several attributes and those are 'in_total_amount', 'out_total_amount', 'timestamp', 'block_height', 'tx_id' and 'is_coinbase'.
Any time variables should be written as timestamps.
Any ranges should be defined as unwinds, instead of using operators like '<,<=,>=,>'.

Provide the Cypher query as raw text so that it can be directly executed by the graph database. Do not add any prefix or postfix.

For queries involving transactions and associated details, include all related nodes and edges, such as:
- Transactions sent from the specified address.
- Addresses receiving funds from those transactions.

Ensure to include the 'SENT' edges in both directions (from the specified address to the transaction and from the transaction to other addresses).

For example, if the user asks for the last transaction, include the address and all related transactions and addresses.

Example valid Cypher query:
MATCH (a1:Address {address: 'bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r'})-[s1:SENT]->(t1:Transaction)-[s2:SENT]->(a2:Address)
RETURN *
ORDER BY t1.timestamp DESC
LIMIT 1

In the case of a query like "what is my last transaction?", ensure the query returns not only the transaction but also all related addresses and SENT edges.

For queries involving block height ranges, use UNWIND to define the range instead of operators like <, <=, >=, > for better performance.

Example query for block height range:
WITH range(0, 8000) AS block_range
UNWIND block_range AS block_height
MATCH (a1:Address {address: 'bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r'})-[s1:SENT]->(t1:Transaction {block_height: block_height})-[s2:SENT]->(a2:Address)
RETURN *

Any Cypher query that attempts to modify the database (e.g., using CREATE, DELETE, SET, or MERGE) should immediately return 'error' to avoid any unauthorized changes.

Please write the Cypher query in raw text without any additional formatting or comments, ensuring it adheres to the above rules.
"""

balance_tracker_query_schema = """
           You are an assistant to help me query balance changes.
           I will ask you questions, and you will generate SQL queries to fetch the data.

           There are two database tables:

           1. `balance_changes` with the following columns:
              - address (string): The address involved in the balance change.
              - block (integer): The block in which the balance change occurred.
              - d_balance (big integer): The change in balance.
              - block_timestamp (timestamp): The timestamp of the block.

           2. `blocks` with the following columns:
              - block_height (integer): The height of the block.
              - timestamp (timestamp): The timestamp of the block.

           Relationships:
           - there are no relationships between the tables.

           You should be able to handle queries that span across these two tables. 
           The `balance_changes` table contains the balance changes over time for different addresses. 
           The `blocks` table contains information about the blocks and their timestamps.


           For example:
           "Return the address with the highest amount of BTC in December 2009." this question can be answered using the `balance_changes` table, take specific date and address with highest balance, do not SUM the balance.
           The data in this table is preprocessed and ready to be queried.

           "Return the block height of the block with the highest timestamp." this question can be answered using the `blocks` table.

           My question: {question}
           SQL query:
           """

balance_tracker_interpret_prompt = """
            You are an assistant to interpret the results of a query involving balance changes, current balances, and block information.
            Here is the result set:
            {result}

            The data comes from three tables:
            1. `balance_changes` which includes balance changes over time with columns:
               - address (string)
               - block (integer)
               - d_balance (big integer)
               - block_timestamp (timestamp)

            2. `current_balances` which includes the current balances for addresses with columns:
               - address (string)
               - balance (big integer)

            3. `blocks` which includes information about blocks and their timestamps with columns:
               - block_height (integer)
               - timestamp (timestamp)

            Relationships:
            - `balance_changes` is related to `current_balances` via the `address` field.
            - `blocks` is related to `balance_changes` via the `block_height` (in `blocks`) and `block` (in `balance_changes`) fields.

            Please summarize the results in a user-friendly way.
            """

classification_prompt = """
        You are an assistant that classifies prompts into two categories: "Funds Flow" and "Balance Tracking". Your task is to identify the type of each prompt given to you. Here are the definitions and examples for each category:

        **Funds Flow**:
        Questions related to specific transactions, including outgoing and incoming transactions, transactions related to a particular address in a specific block, and tracing where funds were transferred from an address.

        Examples:
        1. Return 15 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r.
        2. Show me 20 transactions incoming to my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r.
        3. I have sent more than 1.5 BTC to somewhere but I couldn't remember. Show me 30 relevant transactions.
        4. My address is bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r. Show 10 transactions related to my address in the block 402913.
        5. Show where funds were transferred from address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r.

        **Balance Tracking**:
        Questions related to the current or historical balance of addresses, including identifying addresses with the highest balances and retrieving mined blocks and their timestamps.

        Examples:
        1. Return me top 3 addresses that have the highest current balances plus return blocks and timestamps.
        2. Return me the address that has the highest current balance over time.
        3. Return me the address who had the highest amount of BTC in 2009-01.
        4. Return me all mined blocks and their timestamps.
        5. Show me the top 3 blocks that have the highest balance.

        Given a prompt, classify it as either "Funds Flow" or "Balance Tracking". For example:

        1. "Show me 20 transactions incoming to my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r."
        - Classification: Funds Flow

        2. "Return me the address that has the highest current balance over time."
        - Classification: Balance Tracking

        Classify the following prompt:
        {prompt}
        """

