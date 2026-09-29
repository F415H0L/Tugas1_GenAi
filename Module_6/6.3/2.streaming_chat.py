# 6.3 The OpenAI SDK - Streaming with OpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

stream = client.chat.completions.create(
    model="qwen/qwen3.8-max:free",
    max_tokens=512,
    stream=True,
    messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}],
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
