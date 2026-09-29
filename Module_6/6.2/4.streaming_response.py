# 6.2 The Anthropic SDK - Streaming Responses
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)



with client.messages.stream(
        model="qwen/qwen3.8-max:free",
        max_tokens=512,
        messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        final = stream.get_final_message()
        print(f"\nTotal tokens: {final.usage.input_tokens + final.usage.output_tokens}")


