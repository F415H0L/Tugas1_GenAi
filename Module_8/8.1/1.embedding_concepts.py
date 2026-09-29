# 8.1 What Are Embeddings?
"""
An embedding is a fixed-length vector of floating-point numbers
representing semantic information of text.
Semantically similar texts are geometrically close in vector space.

Key properties:
- Dimension: 768 to 3072 depending on model
- Magnitude: Typically L2-normalized (length = 1.0)
- Metric: Cosine similarity or inner product
"""

def print_embedding_summary():
    print("Embeddings power semantic search, RAG retrieval, and clustering.")

if __name__ == "__main__":
    print_embedding_summary()
