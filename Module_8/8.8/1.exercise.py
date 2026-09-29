# 8.8 Retrieval Exercises: duplicates, hybrid search, CRUD, and cache
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_API_MODEL = "text-embedding-3-small"


@dataclass
class Document:
	id: str
	text: str
	embedding: Optional[np.ndarray] = None


class DuplicateDetector:
	def __init__(
		self,
		documents: list[Document],
		threshold: float = 0.95,
		model: Optional[SentenceTransformer] = None,
	):
		if not -1.0 <= threshold <= 1.0:
			raise ValueError("threshold must be between -1 and 1.")
		self.documents = documents
		self.threshold = threshold
		self._model = model or SentenceTransformer(MODEL_NAME)
		vectors = self._model.encode(
			[document.text for document in documents],
			convert_to_numpy=True,
			normalize_embeddings=True,
		)
		self._matrix = np.asarray(vectors, dtype=np.float32)
		for document, vector in zip(self.documents, self._matrix):
			document.embedding = vector

	def find_duplicates(self) -> list[tuple[Document, Document, float]]:
		pairs = []
		similarities = self._matrix @ self._matrix.T
		for left_index in range(len(self.documents)):
			for right_index in range(left_index + 1, len(self.documents)):
				score = float(similarities[left_index, right_index])
				if score >= self.threshold:
					pairs.append((self.documents[left_index], self.documents[right_index], score))
		return pairs


def _tokenize(text: str) -> list[str]:
	return re.findall(r"\b[\w'-]+\b", text.lower())


class HybridSearch:
	"""Blend normalized semantic cosine and BM25-style keyword scores."""

	def __init__(
		self,
		documents: list[Document],
		model: Optional[SentenceTransformer] = None,
		k1: float = 1.5,
		b: float = 0.75,
	):
		if k1 <= 0 or not 0 <= b <= 1:
			raise ValueError("k1 must be positive and b must be between 0 and 1.")
		self.documents = documents
		self._model = model or SentenceTransformer(MODEL_NAME)
		vectors = self._model.encode(
			[document.text for document in documents],
			convert_to_numpy=True,
			normalize_embeddings=True,
		)
		self._matrix = np.asarray(vectors, dtype=np.float32)
		for document, vector in zip(self.documents, self._matrix):
			document.embedding = vector

		self.k1 = k1
		self.b = b
		self._tokens = [_tokenize(document.text) for document in documents]
		self._lengths = [len(tokens) for tokens in self._tokens]
		self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
		self._document_frequency = Counter(
			token for tokens in self._tokens for token in set(tokens)
		)

	def _keyword_scores(self, query: str) -> np.ndarray:
		query_terms = set(_tokenize(query))
		document_count = len(self.documents)
		scores = np.zeros(document_count, dtype=np.float32)
		if not query_terms or document_count == 0 or self._average_length == 0:
			return scores

		for index, tokens in enumerate(self._tokens):
			term_counts = Counter(tokens)
			document_length = self._lengths[index]
			for term in query_terms:
				frequency = term_counts[term]
				if not frequency:
					continue
				doc_frequency = self._document_frequency[term]
				inverse_frequency = math.log(
					1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
				)
				denominator = frequency + self.k1 * (
					1 - self.b + self.b * document_length / self._average_length
				)
				scores[index] += inverse_frequency * frequency * (self.k1 + 1) / denominator

		highest = float(scores.max(initial=0.0))
		if highest > 0:
			scores /= highest
		return scores

	def search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[tuple[Document, float]]:
		if not 0.0 <= alpha <= 1.0:
			raise ValueError("alpha must be between 0 and 1.")
		if k <= 0 or not self.documents:
			return []

		query_vector = self._model.encode(
			[query], convert_to_numpy=True, normalize_embeddings=True
		)[0].astype(np.float32)
		semantic_scores = self._matrix @ query_vector
		keyword_scores = self._keyword_scores(query)
		final_scores = alpha * semantic_scores + (1.0 - alpha) * keyword_scores
		top_indices = np.argsort(final_scores)[::-1][: min(k, len(self.documents))]
		return [
			(self.documents[int(index)], float(final_scores[index]))
			for index in top_indices
		]


def _cache_key(text: str, model: str) -> str:
	return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def embed_with_cache(
	texts: list[str],
	model: str = EMBEDDING_API_MODEL,
	db_path: str | Path = "embeddings_cache.sqlite3",
	client=None,
) -> np.ndarray:
	"""Embed via OpenAI and cache float32 vectors in SQLite by model and text hash."""
	if not texts:
		return np.empty((0, 0), dtype=np.float32)

	if client is None:
		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise RuntimeError("Set OPENAI_API_KEY to create uncached OpenAI embeddings.")
		client = OpenAI(api_key=api_key)

	cache_path = Path(db_path)
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	vectors_by_key: dict[str, np.ndarray] = {}
	missing: dict[str, str] = {}

	with sqlite3.connect(cache_path) as connection:
		connection.execute(
			"CREATE TABLE IF NOT EXISTS embeddings ("
			"cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)"
		)

		for text in texts:
			key = _cache_key(text, model)
			if key in vectors_by_key or key in missing:
				continue
			row = connection.execute(
				"SELECT vector FROM embeddings WHERE cache_key = ?", (key,)
			).fetchone()
			if row is None:
				missing[key] = text
			else:
				vectors_by_key[key] = np.frombuffer(row[0], dtype=np.float32).copy()

		if missing:
			response = client.embeddings.create(input=list(missing.values()), model=model)
			ordered_data = sorted(response.data, key=lambda item: item.index)
			with connection:
				for (key, _text), item in zip(missing.items(), ordered_data):
					vector = np.asarray(item.embedding, dtype=np.float32)
					vectors_by_key[key] = vector
					connection.execute(
						"INSERT OR REPLACE INTO embeddings (cache_key, model, vector) VALUES (?, ?, ?)",
						(key, model, vector.tobytes()),
					)

	return np.stack([vectors_by_key[_cache_key(text, model)] for text in texts]).astype(np.float32)


def make_fifty_document_corpus() -> list[Document]:
	topics = [
		"RAG retrieves relevant documents and uses them to ground a language model's answer.",
		"Vector databases index embeddings and support fast nearest-neighbour search.",
		"Fine-tuning adapts a pretrained model using examples for a target task.",
		"Prompt engineering shapes model behavior through carefully written instructions.",
		"Transformers use self-attention to model relationships between tokens.",
		"Chunking divides long documents into smaller passages for retrieval.",
		"Agents use tools and observations to complete multi-step tasks.",
		"Embeddings represent text as numeric vectors for similarity comparisons.",
		"Evaluation measures retrieval quality with precision, recall, and ranking metrics.",
		"Metadata filters restrict search results by attributes such as date or category.",
	]
	return [
		Document(id=f"doc-{topic_index:02d}-copy-{copy_index}", text=topic)
		for topic_index, topic in enumerate(topics, start=1)
		for copy_index in range(1, 6)
	]


def test_embedding_cache() -> None:
	class FakeEmbeddings:
		def __init__(self):
			self.create = Mock(side_effect=self._create)

		@staticmethod
		def _create(input: list[str], model: str):
			data = [
				SimpleNamespace(index=index, embedding=[float(len(text)), float(index + 1)])
				for index, text in enumerate(input)
			]
			return SimpleNamespace(data=data)

	fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
	with tempfile.TemporaryDirectory() as directory:
		database = Path(directory) / "cache.sqlite3"
		first = embed_with_cache(["same text", "other text"], db_path=database, client=fake_client)
		second = embed_with_cache(["other text", "same text"], db_path=database, client=fake_client)
		assert first.shape == (2, 2)
		assert np.array_equal(second, first[::-1])
		assert fake_client.embeddings.create.call_count == 1


def load_vector_store_module():
	"""Load the VectorStore class from Module 8.4."""
	import importlib.util
	import sys

	module_path = Path(__file__).resolve().parents[1] / "8.4" / "1.Semantic Search.py"
	spec = importlib.util.spec_from_file_location("semantic_search_for_exercises", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load VectorStore from {module_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def main() -> None:
	model = SentenceTransformer(MODEL_NAME)
	corpus = make_fifty_document_corpus()
	duplicate_detector = DuplicateDetector(corpus, threshold=0.95, model=model)
	duplicates = duplicate_detector.find_duplicates()
	print(f"Duplicate pairs in 50-document corpus: {len(duplicates)}")
	for first, second, score in duplicates[:5]:
		print(f"  {first.id} <-> {second.id}: {score:.4f}")

	hybrid_search = HybridSearch(corpus, model=model)
	print("\nHybrid search results:")
	for document, score in hybrid_search.search("How can I retrieve relevant documents for RAG?", k=3, alpha=0.65):
		print(f"  {score:.4f} | {document.id}: {document.text}")

	semantic_module = load_vector_store_module()
	store = semantic_module.VectorStore()
	store.add_documents(semantic_module.CORPUS[:3])
	store.update("d01", "Updated RAG document about retrieving external knowledge.")
	store.delete("d02")
	assert store._matrix is not None and store._matrix.shape[0] == len(store._documents) == 2
	print("\nVectorStore update/delete matrix check passed.")

	test_embedding_cache()
	print("SQLite OpenAI embedding cache test passed (one API call for repeated texts).")


if __name__ == "__main__":
	main()
