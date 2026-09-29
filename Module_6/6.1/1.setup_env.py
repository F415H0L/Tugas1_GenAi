# 6.1 Setting Up Clients and Loading API Keys
"""
Never put API keys directly in source code.
Store them in a .env file and load them via python-dotenv.

Sample .env file:
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
"""
import os
from dotenv import load_dotenv

load_dotenv()

anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

print("Anthropic API key configured:", bool(anthropic_key))
print("OpenAI API key configured:", bool(openai_key))
