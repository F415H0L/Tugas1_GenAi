# 6.4 Tool Calling (Function Calling) with Anthropic
import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

# 1. Define the tool schema
tools = [
    {
        "name": "get_model_info",
        "description": "Returns context window size and cost per 1K tokens for a given LLM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'.",
                }
            },
            "required": ["model_name"],
        },
    }
]


# 2. The actual function the tool will call
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o": {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro": {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name, {"error": f"Unknown model: {model_name}"})


user_question = "How large is the context window of claude-sonnet-4-5?"

# 3. First API call - model may return a tool_use block
response = client.messages.create(
    model="qwen/qwen3.8-max:free",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": user_question}],
)

# 4. Check if model wants to use a tool
if response.stop_reason == "tool_use":
    tool_block = next(block for block in response.content if block.type == "tool_use")
    tool_name = tool_block.name
    tool_input = tool_block.input
    tool_use_id = tool_block.id

    # 5. Execute the function
    result = get_model_info(**tool_input)
    print(f"Tool called: {tool_name}({tool_input})")
    print(f"Tool result: {result}")

    # 6. Send tool result back to the model
    final = client.messages.create(
        model="qwen/qwen3.8-max:free",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result),
                    }
                ],
            },
        ],
    )
    print("\nFinal answer:")
    print(final.content[0].text)
