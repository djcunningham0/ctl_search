from typing import Literal

import numpy as np


def precision_at_k(
        retrieved: list,
        relevant: set | dict,
        k: int,
) -> float:
    if len(retrieved) == 0:
        return 0.0
    retrieved = retrieved[:k]
    return len([x for x in retrieved if x in relevant]) / len(retrieved)


def recall_at_k(
        retrieved: list,
        relevant: set | dict,
        k: int,
) -> float:
    if len(relevant) == 0:
        return 0.0
    retrieved = retrieved[:k]
    return len([x for x in retrieved if x in relevant]) / len(relevant)


def dcg_at_k(
        retrieved: list,
        relevant: set | dict,  # {item: relevance_score}
        k: int,
        dcg_type: Literal["linear", "exponential"] = "linear",
) -> float:
    if len(retrieved) == 0:
        return 0.0

    relevant = _validate_relevant_dict(relevant)

    retrieved = retrieved[:k]
    if dcg_type == "linear":
        # linear discounting: rel_i / log2(i+2)
        return sum(relevant.get(x, 0) / np.log2(i + 2) for i, x in enumerate(retrieved))
    else:
        # exponential discounting: (2^rel_i - 1) / log2(i+2)
        return sum((2 ** relevant.get(x, 0) - 1) / np.log2(i + 2) for i, x in enumerate(retrieved))


def ndcg_at_k(
        retrieved: list,
        relevant: set | dict,  # {item: relevance_score}
        k: int,
        dcg_type: Literal["linear", "exponential"] = "linear",
) -> float:
    relevant = _validate_relevant_dict(relevant)
    dcg = dcg_at_k(retrieved, relevant, k, dcg_type)
    ideal_order = sorted(relevant, key=lambda x: relevant[x], reverse=True)
    idcg = dcg_at_k(ideal_order, relevant, k, dcg_type)
    return dcg / idcg if idcg > 0 else 0.0


def _validate_relevant_dict(relevant: set | dict):
    if isinstance(relevant, dict):
        return relevant
    elif isinstance(relevant, set):
        return {x: 1 for x in relevant}
    else:
        raise ValueError(f"invalid type: {type(relevant)}")
