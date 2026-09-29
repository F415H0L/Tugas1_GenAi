# 6.4 Tool Calling with OpenAI
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"],
            },
        },
    }
]

def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})

def run_openai_tool_flow():
    messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]
    response = client.chat.completions.create(
        model="qwen/qwen3.8-max:free",
        tools=tools,
        tool_choice="required",
        messages=messages,
    )
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        tool_call = choice.message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = get_model_info(**args)

        messages.append(choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        final = client.chat.completions.create(
            model="qwen/qwen3.8-max:free",
            messages=messages,
        )
        print("Final output:", final.choices[0].message.content)

if __name__ == "__main__":
    try:
        run_openai_tool_flow()
    except Exception as e:
        print(f"Demo run: {e}")
