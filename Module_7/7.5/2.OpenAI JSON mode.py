# 7.5 Approach 2: OpenAI JSON Mode
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

SAMPLE_TEXT = "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."

def extract_entities_json(user_text: str) -> dict:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-max:free",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 'Extract entities. Return JSON with this schema: {"people": [string], "organizations": [string], "locations": [string]}. Keep a city and its state or country together as one location string.',
            },
            {"role": "user", "content": user_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    result = extract_entities_json(SAMPLE_TEXT)
    print(result)


