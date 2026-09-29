# 7.4 Structured CoT with XML Tags
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

SYSTEM_XML = """Solve the problem and use this exact format:
<thinking>
Give a brief, user-facing calculation summary.
</thinking>
<answer>
Give the final answer only.
</answer>"""

def parse_xml_sections(response_text: str) -> tuple[str, str]:
    thinking = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    calculation = thinking.group(1).strip() if thinking else "not found"
    ans = answer.group(1).strip() if answer else "not found"
    return calculation, ans

if __name__ == "__main__":
    response = client.messages.create(
        model="qwen/qwen3.8-max:free",
        max_tokens=512,
        system=SYSTEM_XML,
        messages=[
            {
                "role": "user",
                "content": (
                    "A RAG pipeline retrieves 5 documents, each 400 tokens. "
                    "The query is 50 tokens. The model has a 4096-token limit "
                    "for context. How many tokens remain for the response?"
                ),
            }
        ],
    )
    calculation, final_ans = parse_xml_sections(response.content[0].text)
    print("Reasoning:", calculation)
    print("Answer:", final_ans)
