# 8.4 In-Memory VectorStore with Semantic Search
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_batch(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed texts locally and return an (n, dim) float32 array."""
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

class VectorStore:
    """Small in-memory semantic index backed by a local sentence transformer."""

    def __init__(self, embed_model: str = MODEL_NAME):
        self._model = SentenceTransformer(embed_model)
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self) -> None:
        if not self._documents:
            self._matrix = None
            return
        self._matrix = np.array([document.embedding for document in self._documents], dtype=np.float32)

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        vectors = embed_batch([document.text for document in documents], self._model)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        self._rebuild_matrix()

    def delete(self, doc_id: str) -> Document:
        """Delete a document by ID and rebuild the aligned embedding matrix."""
        for index, document in enumerate(self._documents):
            if document.id == doc_id:
                deleted = self._documents.pop(index)
                self._rebuild_matrix()
                return deleted
        raise KeyError(f"Document not found: {doc_id}")

    def update(self, doc_id: str, new_text: str) -> Document:
        """Update a document's text and embedding without changing its position."""
        for index, document in enumerate(self._documents):
            if document.id == doc_id:
                vector = embed_batch([new_text], self._model)[0]
                norm = np.linalg.norm(vector)
                normalized = (vector / norm if norm else vector).astype(np.float32)
                document.text = new_text
                document.embedding = normalized
                if self._matrix is not None:
                    self._matrix[index] = normalized
                return document
        raise KeyError(f"Document not found: {doc_id}")

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("No documents indexed yet.")
        if k <= 0:
            return []

        query_vector = embed_batch([query], self._model)[0]
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_normed = (query_vector / q_norm).astype(np.float32)

        scores = self._matrix @ q_normed
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(document=self._documents[int(i)], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_idx)
        ]


CORPUS = [
    Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation to answer questions using external knowledge."),
    Document("d02", "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search using algorithms like HNSW and IVF."),
    Document("d03", "Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a curated dataset with task-specific examples."),
    Document("d04", "Prompt engineering involves designing and optimising input prompts to guide language models toward producing the desired output."),
    Document("d05", "LangChain is a Python framework that provides abstractions for building applications with large language models, including chains, agents, and memory."),
    Document("d06", "Cosine similarity measures the angle between two vectors and is the standard metric for comparing text embeddings in semantic search."),
    Document("d07", "RLHF (Reinforcement Learning from Human Feedback) aligns language models with human preferences by training a reward model on human rankings."),
    Document("d08", "Chunking strategies for RAG include fixed-size chunks, sentence-aware splits, and recursive character splitting with configurable overlap."),
    Document("d09", "The transformer architecture uses self-attention mechanisms to model relationships between all tokens in a sequence simultaneously."),
    Document("d10", "Agents use language models as a reasoning engine, enabling them to plan multi-step tasks, call tools, and take actions based on observations."),
]

QUERIES = [
    "How does RAG work?",
    "What algorithms do vector databases use?",
    "How do I split documents for embedding?",
]


if __name__ == "__main__":
    store = VectorStore()
    store.add_documents(CORPUS)

    for query in QUERIES:
        print(f"\nQuery: {query!r}")
        results = store.search(query, k=3)
        for result in results:
            print(f"  [{result.rank}] score={result.score:.4f} | {result.document.text[:80]}...")
