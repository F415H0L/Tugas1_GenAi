# 7.3 Few-Shot Prompting for Structured Extraction
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

MODEL = "qwen/qwen3.8-max:free"

FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string,
extract: model name, task, and score as a JSON object.

Examples:
Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation."""

test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]

if __name__ == "__main__":
    for text in test_inputs:
        response = client.messages.create(
            model=MODEL,
            max_tokens=128,
            system=FEW_SHOT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        print(f"Input: {text}")
        print(f"Output: {response.content[0].text}\n")
