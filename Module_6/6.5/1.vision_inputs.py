# 6.5 Vision - Images as Input (URL & Base64)
import anthropic
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

# Option A: Image via URL
def describe_image_url(url: str) -> str:
    response = client.messages.create(
        model="qwen/qwen3.8-max:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": url}},
                {"type": "text", "text": "Describe what you see in this image."}
            ]
        }]
    )
    return response.content[0].text

# Option B: Image via Base64 for local files
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"

    response = client.messages.create(
        model="qwen/qwen3.8-max:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": "What is in this image?"}
            ]
        }]
    )
    return response.content[0].text

# Usage:
image_url = "https://png.pngtree.com/background/20250602/original/pngtree-fluffy-white-clouds-on-sunny-blue-sky-picture-image_16610581.jpg"
text = describe_image_url(image_url)
print(text)
