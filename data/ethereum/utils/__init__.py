from decimal import Decimal
import re


# doesn't check all possible cases since Memgraph also runs some checks (comment checks etc)
def generate_patterns_for_terms(terms):
    patterns = []
    for term in terms:
        lower_term = term.lower()
        # Escape term for regex pattern
        escaped_term = re.escape(lower_term)
        # Basic term
        patterns.append(escaped_term)

        # With spaces between each character
        patterns.append(r'\s*'.join(escaped_term))

        # Unicode escape sequences (basic example)
        unicode_pattern = ''.join([f'\\u{ord(char):04x}' for char in lower_term])
        patterns.append(unicode_pattern)

        # Mixed case variations to catch case obfuscation
        mixed_case_variations = [f'[{char.lower()}{char.upper()}]' for char in lower_term]
        patterns.append(''.join(mixed_case_variations))

        # Detecting comments that might hide portions of malicious queries
        # This is a simplistic approach and might need refinement
        patterns.append(f'/{escaped_term}|{escaped_term}/')

    return patterns


def is_malicious(query, terms):
    # Normalize the query by lowercasing (maintaining original for comment checks)
    normalized_query = query.lower()

    # Generate patterns for the given terms, including obfuscation-resistant versions
    write_patterns = generate_patterns_for_terms(terms)

    # Compile regex patterns to detect any of the write operations
    pattern = re.compile('|'.join(write_patterns), re.IGNORECASE)

    # Check if the normalized query matches any of the patterns
    if pattern.search(normalized_query):
        return True  # Query is potentially malicious or not read-only

    return False  # Query passed the check

def block_base_reward_for_number(block_number: int) -> int:
    if block_number >= 12965000:  # London hard fork
        base_reward = 2 * 10 ** 18
    elif block_number >= 9200000:  # Muir Glacier
        base_reward = 2 * 10 ** 18
    elif block_number >= 7280000:  # Constantinople
        base_reward = 2 * 10 ** 18
    elif block_number >= 4370000:  # Byzantium
        base_reward = 3 * 10 ** 18
    else:
        base_reward = 5 * 10 ** 18

    return base_reward


def calculate_block_reward(block_number: int, block_data: dict) -> dict:
    base_reward = block_base_reward_for_number(block_number)

    uncle_count = int(block_data.get('ommerCount', '0x0'), 16)

    # Calculate uncle rewards
    uncles = block_data.get('ommers', [])
    uncle_rewards = 0
    for uncle in uncles:
        uncle_number = int(uncle['number'], 16)
        uncle_rewards += calculate_uncle_reward(block_number, uncle_number, base_reward)

    # If no uncles are provided, estimate based on count
    if not uncles and uncle_count > 0:
        uncle_rewards = uncle_count * (base_reward // 32)

    # Calculate total transaction fees
    total_tx_fees = sum(
        int(tx['gasUsed'], 16) * int(tx['gasPrice'], 16)
        for tx in block_data['transactions']
    )

    # Handle EIP-1559 transactions
    if block_data.get('baseFeePerGas') is not None:
        base_fee_per_gas = int(block_data['baseFeePerGas'], 16) if block_data['baseFeePerGas'] else 0
        total_gas_used = int(block_data['gasUsed'], 16)
        burnt_fees = base_fee_per_gas * total_gas_used
        total_tx_fees -= burnt_fees

    # Calculate miner reward (base reward + uncle inclusion rewards)
    miner_reward = base_reward + (uncle_count * (base_reward // 32))

    total_reward = miner_reward + uncle_rewards + total_tx_fees

    return {
        'base_reward': str(base_reward),
        'uncle_rewards': str(uncle_rewards),
        'transaction_fees': str(total_tx_fees),
        'miner_reward': str(miner_reward),
        'total_reward': str(total_reward),
    }


def calculate_uncle_reward(block_number: int, uncle_number: int, base_reward: int) -> int:
    if block_number < 7280000:  # Before Constantinople
        return (base_reward * (8 + uncle_number - block_number)) // 8
    else:  # After Constantinople
        return base_reward // 32


def calculate_burnt_fees(block_data: dict) -> str:
    if 'baseFeePerGas' in block_data and block_data['baseFeePerGas'] is not None:
        base_fee_per_gas = int(block_data['baseFeePerGas'], 16)
        total_gas_used = int(block_data['gasUsed'], 16)
        return str(base_fee_per_gas * total_gas_used)
    return "0"
