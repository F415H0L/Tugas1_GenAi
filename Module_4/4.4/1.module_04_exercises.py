# 4.4 Module 04 Exercises
import asyncio
import csv
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


# 1. Save and load conversation history to JSON

def save_conversation(history: list[dict], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_conversation(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


# 2. Async compare endpoints concurrently

async def fetch_endpoint(client: httpx.AsyncClient, url: str):
    start = time.perf_counter()
    try:
        response = await client.get(url, timeout=10.0)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return (url, response.status_code, elapsed_ms)
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return (url, 0, elapsed_ms)


async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, int]]:
    async with httpx.AsyncClient() as client:
        tasks = [fetch_endpoint(client, url) for url in urls]
        return await asyncio.gather(*tasks)


# 3. JSON config loader with environment overrides

def _coerce_env_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_config(config_path: str | Path, env_prefix: str = "APP_") -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        return {}

    config = json.loads(config_file.read_text(encoding="utf-8"))

    for key, value in list(config.items()):
        env_key = f"{env_prefix}{key.upper()}"
        if env_key in os.environ:
            config[key] = _coerce_env_value(os.environ[env_key])

    return config


# 4. Thread-safe CSV logger for LLM calls

class CSVLogger:
    def __init__(self, filename: str | Path):
        self.file_path = Path(filename)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

        if not self.file_path.exists():
            with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "timestamp",
                        "model",
                        "input_tokens",
                        "output_tokens",
                        "latency_ms",
                    ]
                )

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        with self.lock:
            with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        timestamp,
                        model,
                        input_tokens,
                        output_tokens,
                        latency_ms,
                    ]
                )


# Demo usage
if __name__ == "__main__":
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    save_conversation(history, "conversation.json")
    print(load_conversation("conversation.json"))

    async def demo_async():
        urls = [
            "https://jsonplaceholder.typicode.com/posts/1",
            "https://jsonplaceholder.typicode.com/posts/2",
        ]
        print(await compare_endpoints(urls))

    asyncio.run(demo_async())

    config = {
        "model": "claude-sonnet-4-5",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    config_path = Path("config.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    os.environ["APP_MODEL"] = "gpt-4o-mini"
    print(load_config(config_path, env_prefix="APP_"))

    logger = CSVLogger("llm_calls.csv")
    logger.log_call("gpt-4o-mini", 120, 80, 250.5)
    logger.log_call("claude-3-sonnet", 200, 140, 410.2)
    print("CSV log written.")
