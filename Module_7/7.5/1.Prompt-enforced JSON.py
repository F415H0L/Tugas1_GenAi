# 7.5 Approach 1: Prompt-Enforced JSON Output
import os
import json

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
  api_key=os.environ["ANTHROPIC_API_KEY"],
  base_url="https://api.xkiro.com",
)

MODEL = "qwen/qwen3.8-max:free"

SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

texts = [
  "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
  "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)

if __name__ == "__main__":
  for text in texts:
    response = client.messages.create(
      model=MODEL,
      max_tokens=256,
      system=SYSTEM,
      messages=[{"role": "user", "content": text}],
    )
    info = clean_json_response(response.content[0].text)
    print(json.dumps(info, indent=2))
    print()
