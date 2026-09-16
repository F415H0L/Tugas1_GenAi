import time
from functools import wraps


# 1. Token cost calculator

def token_cost(tokens: int, model: str) -> float:
    """Return estimated cost for a model based on price per 1K tokens."""
    rates = {
        "gpt-4o-mini": 0.15,
        "gpt-4o": 5.00,
        "claude-3-haiku": 0.25,
        "claude-3-sonnet": 3.00,
    }

    if model not in rates:
        raise ValueError(f"Unknown model: {model}")

    price_per_1k = rates[model]
    return (tokens / 1000) * price_per_1k


# 2. Retry decorator

def retry(max_retries: int = 3, delay: float = 0.5, verbose: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    if attempt == max_retries:
                        raise
                    if verbose:
                        print(f"Attempt {attempt} failed: {exc}. Retrying in {delay}s...")
                    time.sleep(delay)
            raise last_error

        return wrapper

    return decorator


counter = 0


@retry(max_retries=3, delay=0.1, verbose=False)
def flaky_function():
    global counter
    counter += 1
    if counter < 3:
        raise ValueError("Temporary failure")
    return "Success after retries"


# 3. Temperature label

def temperature_label(t: float) -> str:
    """Map a temperature value to a descriptive label."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if 0.0 <= t < 0.3:
        return "precise"
    if 0.3 <= t < 0.7:
        return "balanced"
    return "creative"


# 4. Parse token count and cost from a string

def parse_token_info(text: str):
    """Extract token count and cost from a string without regex."""
    token_part = text.split("tokens")[0].strip()
    cost_part = text.split("USD")[0].split(",")[-1].strip()

    token_count = int(token_part)
    cost_value = float(cost_part)
    return token_count, cost_value


# Demo runs
print("Task 1:", token_cost(1500, "gpt-4o-mini"))
print("Task 2:", flaky_function())
print("Task 3:", temperature_label(0.5))
print("Task 4:", parse_token_info("128000 tokens, 0.005 USD per 1K"))
