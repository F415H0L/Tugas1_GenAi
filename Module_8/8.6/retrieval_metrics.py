# 8.6 Evaluating Retrieval Quality - Precision@k, Recall@k, MRR
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(store, eval_cases: list[RetrievalEvalCase], k: int = 5) -> dict:
    """Run all evaluation cases and return mean precision, recall, and MRR."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not eval_cases:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, "MRR": 0.0}

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [result.document.id for result in results]
        precision_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        recall_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        reciprocal_ranks.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(precision_scores)), 4),
        f"recall@{k}": round(float(np.mean(recall_scores)), 4),
        "MRR": round(float(np.mean(reciprocal_ranks)), 4),
    }


EVAL_CASES = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
]


def load_semantic_search_module():
    """Load the VectorStore and corpus from Module 8.4."""
    module_path = Path(__file__).resolve().parents[1] / "8.4" / "1.Semantic Search.py"
    spec = spec_from_file_location("semantic_search_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load semantic search module at {module_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    semantic_search = load_semantic_search_module()
    store = semantic_search.VectorStore()
    store.add_documents(semantic_search.CORPUS)
    metrics = evaluate_retrieval(store, EVAL_CASES, k=3)
    print(metrics)
