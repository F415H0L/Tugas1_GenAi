# 8.2 Generating Embeddings Locally
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts_local(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed texts locally and return a float32 array with shape (n, dim)."""
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

if __name__ == "__main__":
    embeddings = embed_texts_local(texts)
    print(f"Shape: {embeddings.shape}")
    print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")
