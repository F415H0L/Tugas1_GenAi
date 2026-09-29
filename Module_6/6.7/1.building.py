# 6.7 Building a Provider-Agnostic Client Abstraction
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChatMessage:
    role: str  # "user", "assistant", or "system"
    content: str

@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        pass

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = ""):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

    def chat(self, messages: list[ChatMessage], system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> ChatResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

    def chat(self, messages: list[ChatMessage], system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )

if __name__ == "__main__":
    print("Provider-agnostic BaseLLMClient interface compiled.")
