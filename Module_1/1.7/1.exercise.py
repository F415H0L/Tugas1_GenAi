import time
from functools import wraps


def token_cost(tokens: int, model: str) -> float:
    """Return the estimated cost based on the model price per 1K tokens."""
    costs_per_1k = {
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.0015,
        "claude-3": 0.015,
    }

    if model not in costs_per_1k:
        raise ValueError(f"Unknown model: {model}")

    return (tokens / 1000) * costs_per_1k[model]


def retry(n: int, delay: float):
    """Retry a function up to n times after an exception."""
    if n < 0:
        raise ValueError("n must be non-negative")

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            for attempt in range(n + 1):
                try:
                    return function(*args, **kwargs)
                except Exception:
                    if attempt == n:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


attempts = 0


@retry(n=2, delay=0.01)
def fails_twice():
    """Fail twice, then return successfully."""
    global attempts
    attempts += 1
    if attempts <= 2:
        raise RuntimeError("temporary failure")
    return "succeeded"


def temperature_label(t: float) -> str:
    """Map a temperature from 0.0 to 1.0 to one of three labels."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if t < 0.3:
        return "precise"
    if t < 0.7:
        return "balanced"
    return "creative"


def parse_token_cost(text: str) -> tuple[int, float]:
    """Extract token count and cost using string methods only."""
    token_part, cost_part = text.split(",", 1)
    tokens = int(token_part.strip().split()[0])
    cost = float(cost_part.strip().split()[0])
    return tokens, cost


if __name__ == "__main__":
    print(f"1. Token cost: {token_cost(1500, 'gpt-4')}")
    print(f"2. Retry result: {fails_twice()}")
    print(f"3. Temperature label: {temperature_label(0.5)}")
    print(f"4. Parsed token info: {parse_token_cost('128000 tokens, 0.005 USD per 1K')}")