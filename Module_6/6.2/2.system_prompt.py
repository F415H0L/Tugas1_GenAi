# 6.2 The Anthropic SDK - System Prompt
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

message = client.messages.create(
    model="qwen/qwen3.8-max:free",
    max_tokens=512,
    system="You are a concise technical writer. Answer in plain English, no jargon.",
    messages=[{"role": "user", "content": "Explain what a vector database does."}],
)
print(message.content[0].text)
   