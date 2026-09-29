# 8.2 Generating Embeddings with Voyage AI (Anthropic recommended)
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def embed_texts_voyage(texts: list[str], model: str = "voyage-3") -> np.ndarray:
    try:
        import voyageai
        vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", "mock-key"))
        result = vo.embed(texts, model=model, input_type="document")
        return np.array(result.embeddings, dtype=np.float32)
    except ImportError:
        print("Install voyageai via: pip install voyageai")
        return np.zeros((len(texts), 1024), dtype=np.float32)

if __name__ == "__main__":
    print("Voyage AI embedding helper ready.")
