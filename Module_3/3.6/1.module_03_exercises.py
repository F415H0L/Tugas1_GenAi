# 3.6 Module 03 Exercises
import sys
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from string import Formatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 1. RateLimiter
class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 5):
        if max_calls_per_minute <= 0:
            raise ValueError("max_calls_per_minute must be positive")
        self.max_calls_per_minute = max_calls_per_minute
        self.min_interval = 60.0 / max_calls_per_minute
        self.last_call = 0.0

    def check_and_wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# 2. PromptTemplate with validation
@dataclass
class PromptTemplate:
    template: str

    def render(self, **kwargs) -> str:
        fields = {
            name
            for _, name, _, _ in Formatter().parse(self.template)
            if name is not None
        }
        missing = fields - set(kwargs)
        if missing:
            raise ValueError(f"Missing placeholders: {sorted(missing)}")
        return self.template.format_map(kwargs)


# 3. Retry decorator with syntax @retry(max_attempts=3, delay=0.1)
def retry(max_attempts: int = 3, delay: float = 0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


# Example function that fails the first two calls
attempt_counter = {"count": 0}


@retry(max_attempts=3, delay=0.1)
def flaky_function():
    attempt_counter["count"] += 1
    if attempt_counter["count"] < 3:
        raise RuntimeError("Temporary failure")
    return "Success on third try"


# 4. Package structure: ConversationHistory + LLMConfig
from llm import ConversationHistory, LLMConfig


if __name__ == "__main__":
    # 1. Test rate limiter
    limiter = RateLimiter(max_calls_per_minute=2)
    for i in range(5):
        limiter.check_and_wait()
        print(f"RateLimiter call {i + 1}: {time.time():.2f}")

    # 2. Prompt template demo
    template = PromptTemplate("Hello {name}, you are a {role}.")
    print(template.render(name="Alice", role="developer"))

    # 3. Retry decorator demo
    print(flaky_function())

    # 4. Package demo
    history = ConversationHistory(max_turns=2)
    history.add("user", "Tell me about embeddings")
    history.add("assistant", "Embeddings convert text into vectors.")
    print(history)

    config = LLMConfig(model="claude-sonnet-4-5", temperature=0.3)
    print(config.as_dict)
