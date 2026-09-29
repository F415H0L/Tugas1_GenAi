# 6.6 Token Management and Cost Estimation
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "qwen/qwen3.8-max:free"
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

PRICING = {
    MODEL: {"input": 0.00, "output": 0.00},  # Free model on xKiro; per 1M tokens
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  # per 1M tokens
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

CONTEXT_LIMITS = {
    MODEL: 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD based on per-1M-token pricing."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit

if __name__ == "__main__":
    response = client.messages.count_tokens(
        model=MODEL,
        system="You are a concise assistant.",
        messages=[{"role": "user", "content": "Explain the transformer architecture."}],
    )
    print(f"Estimated input tokens: {response.input_tokens}")

    cost = estimate_cost(MODEL, input_tokens=500, output_tokens=300)
    print(f"Estimated cost: ${cost:.6f}")
    print(f"Fits in {MODEL} (50k tokens):", fits_in_context(MODEL, 50_000))
