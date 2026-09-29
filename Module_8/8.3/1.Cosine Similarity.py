# 8.3 Cosine Similarity and Pairwise Matrices
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix of shape (n, n)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms
    similarities = (normed @ normed.T).astype(np.float32)
    np.fill_diagonal(similarities, 1.0)
    return similarities

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sample_vecs = rng.standard_normal((4, 8)).astype(np.float32)
    sim_matrix = pairwise_similarity(sample_vecs)
    print("Pairwise similarities:")
    for i in range(4):
        for j in range(i + 1, 4):
            print(f"  vec[{i}] vs vec[{j}]: {sim_matrix[i, j]:.4f}")
    print(f"Diagonal: {np.diag(sim_matrix)}")
