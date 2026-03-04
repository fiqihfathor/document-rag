from collections import Counter
import re


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_sparse_vector(text: str) -> dict:
    tokens = tokenize(text=text)
    tf = Counter(tokens)

    seen = {}
    for token, count in tf.items():
        idx = hash(token) % 2**20
        if idx not in seen:
            seen[idx] = float(count)

    return {
        "indices": list(seen.keys()),
        "values": list(seen.values()),
    }