# 6.8 Exercises: retries, token budgets, model comparison, and streaming
import asyncio
import os
import time
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock, call, patch

import anthropic
import httpx
import openai
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

QWEN_MODEL = "qwen/qwen3.8-max:free"
XKIRO_BASE_URL = os.environ.get("XKIRO_BASE_URL", "https://api.xkiro.com/v1")
RATE_LIMIT_ERRORS = (anthropic.RateLimitError, openai.RateLimitError)


class BudgetExceeded(Exception):
	"""Raised when adding usage would exceed a token budget."""


class TokenBudgetManager:
	def __init__(self, limit: int):
		if limit < 0:
			raise ValueError("Token budget cannot be negative.")
		self.limit = limit
		self.total_tokens = 0

	def add_usage(self, input_tokens: int, output_tokens: int) -> int:
		if input_tokens < 0 or output_tokens < 0:
			raise ValueError("Token counts cannot be negative.")

		new_total = self.total_tokens + input_tokens + output_tokens
		if new_total > self.limit:
			raise BudgetExceeded(
				f"Token budget exceeded: {new_total} requested, limit is {self.limit}."
			)

		self.total_tokens = new_total
		return self.total_tokens


def retry_on_rate_limit(
	client,
	messages: list[dict],
	max_retries: int = 5,
	*,
	model: str = QWEN_MODEL,
	max_tokens: int = 1024,
):
	"""Call an Anthropic or OpenAI-compatible client with exponential backoff."""
	if max_retries < 0:
		raise ValueError("max_retries cannot be negative.")

	for attempt in range(max_retries + 1):
		try:
			if hasattr(client, "messages"):
				return client.messages.create(
					model=model,
					max_tokens=max_tokens,
					messages=messages,
				)
			if hasattr(client, "chat"):
				return client.chat.completions.create(
					model=model,
					max_tokens=max_tokens,
					messages=messages,
				)
			raise TypeError("client must be an Anthropic or OpenAI-compatible client.")
		except RATE_LIMIT_ERRORS:
			if attempt == max_retries:
				raise
			time.sleep(2**attempt)


def _mock_rate_limit_error(error_type):
	request = httpx.Request("POST", "https://mock.xkiro.test/v1/messages")
	response = httpx.Response(429, request=request)
	return error_type("Mocked rate limit", response=response, body={"error": "rate_limit"})


def test_retry_on_rate_limit() -> None:
	messages = [{"role": "user", "content": "test"}]
	cases = [
		(Mock(spec=Anthropic), anthropic.RateLimitError, "messages"),
		(Mock(spec=openai.OpenAI), openai.RateLimitError, "chat"),
	]

	with patch("time.sleep") as mocked_sleep:
		for client, error_type, provider in cases:
			if provider == "messages":
				request = client.messages.create
			else:
				request = client.chat.completions.create

			expected = object()
			request.side_effect = [_mock_rate_limit_error(error_type), expected]
			result = retry_on_rate_limit(client, messages, max_retries=1)

			assert result is expected
			assert request.call_count == 2

		assert mocked_sleep.call_args_list == [call(1), call(1)]


async def _compare_one_model(client: AsyncOpenAI, prompt: str, model: str) -> dict:
	started = perf_counter()
	try:
		response = await client.chat.completions.create(
			model=model,
			max_tokens=1024,
			messages=[{"role": "user", "content": prompt}],
		)
		usage = response.usage
		return {
			"model": model,
			"response_text": response.choices[0].message.content or "",
			"input_tokens": usage.prompt_tokens if usage else 0,
			"output_tokens": usage.completion_tokens if usage else 0,
			"latency_ms": (perf_counter() - started) * 1000,
		}
	except Exception as exc:
		return {
			"model": model,
			"response_text": f"Error: {exc}",
			"input_tokens": 0,
			"output_tokens": 0,
			"latency_ms": (perf_counter() - started) * 1000,
		}


async def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
	"""Call xKiro-compatible model IDs concurrently and return their results."""
	api_key = os.environ.get("XKIRO_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		raise RuntimeError("Set XKIRO_API_KEY in .env before comparing models.")

	async with AsyncOpenAI(api_key=api_key, base_url=XKIRO_BASE_URL) as client:
		rows = await asyncio.gather(
			*(_compare_one_model(client, prompt, model) for model in models)
		)

	columns = ["model", "response_text", "input_tokens", "output_tokens", "latency_ms"]
	return pd.DataFrame(rows, columns=columns)


def stream_to_file(prompt: str, output_path: str) -> None:
	"""Stream a response from xKiro's Anthropic-compatible API into a file."""
	api_key = os.environ.get("XKIRO_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		raise RuntimeError("Set XKIRO_API_KEY in .env before streaming.")

	output_file_path = Path(output_path)
	output_file_path.parent.mkdir(parents=True, exist_ok=True)
	client = Anthropic(api_key=api_key, base_url="https://api.xkiro.com")

	with output_file_path.open("w", encoding="utf-8") as output_file:
		with client.messages.stream(
			model=QWEN_MODEL,
			max_tokens=1024,
			messages=[{"role": "user", "content": prompt}],
		) as stream:
			for text in stream.text_stream:
				output_file.write(text)
				output_file.flush()


def main() -> None:
	test_retry_on_rate_limit()
	print("Mocked Anthropic/OpenAI rate-limit retry tests passed.")

	budget = TokenBudgetManager(limit=1000)
	budget.add_usage(input_tokens=100, output_tokens=50)
	print(f"Token budget used: {budget.total_tokens}/{budget.limit}")
	try:
		budget.add_usage(input_tokens=850, output_tokens=1)
	except BudgetExceeded:
		print("BudgetExceeded test passed.")


if __name__ == "__main__":
	main()

