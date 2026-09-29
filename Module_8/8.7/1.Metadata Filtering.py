from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)

@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

class FilteredVectorStore:
    def __init__(self, embed_model: str = MODEL_NAME):
        self._model = SentenceTransformer(embed_model)
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        if not docs:
            return

        embeddings = embed_texts([doc.text for doc in docs], self._model)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0, 1, norms)
        for doc, emb in zip(docs, normed):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        if k <= 0:
            return []

        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        query_vector = embed_texts([query], self._model)[0]
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_normed = query_vector / q_norm

        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_normed
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(candidates[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    store = FilteredVectorStore()
    docs = [
        FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
        FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
        FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
        FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
        FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
    ]
    store.add(docs)

    query = "Which model is good at coding?"
    print("=== All docs ===")
    results = store.search(query, k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.id}: {doc.text}")

    print("\n=== Anthropic only ===")
    results = store.search(
        query,
        k=3,
        filter_fn=lambda doc: doc.metadata["category"] == "anthropic",
    )
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.id}: {doc.text}")
