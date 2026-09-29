# 7.7 Systematic Prompt Evaluation
import json
import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
	api_key=os.environ["ANTHROPIC_API_KEY"],
	base_url="https://api.xkiro.com",
)

MODEL = "qwen/qwen3.8-max:free"


@dataclass
class EvalCase:
	input_text: str
	expected_keywords: list[str]
	must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
	"""Run a prompt against test cases and return pass rate and details."""
	results = []
	for case in cases:
		response = client.messages.create(
			model=MODEL,
			max_tokens=256,
			system=system,
			messages=[{"role": "user", "content": case.input_text}],
		)
		text = response.content[0].text.strip()

		keyword_hit = any(keyword.lower() in text.lower() for keyword in case.expected_keywords)
		json_valid = True
		if case.must_be_json:
			try:
				json.loads(text)
			except json.JSONDecodeError:
				json_valid = False

		results.append(
			{
				"input": case.input_text[:60],
				"passed": keyword_hit and json_valid,
				"response_preview": text[:80],
			}
		)

	pass_rate = sum(result["passed"] for result in results) / len(results) if results else 0.0
	return {"pass_rate": pass_rate, "results": results}


CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

TEST_CASES = [
	EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
	EvalCase("Write a product description for headphones.", ["GENERATION"]),
	EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
	EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
	EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]


if __name__ == "__main__":
	report = evaluate_prompt(CLASSIFY_SYSTEM, TEST_CASES)
	print(f"Pass rate: {report['pass_rate']:.0%}")
	for result in report["results"]:
		status = "PASS" if result["passed"] else "FAIL"
		print(f"[{status}] {result['input']!r} -> {result['response_preview']!r}")
