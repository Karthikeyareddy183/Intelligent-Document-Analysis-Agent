"""LLM service — provider-agnostic abstraction with streaming support."""

from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import Settings
from core.logger import setup_logger
from core.exceptions import LLMError

logger = setup_logger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_sync(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Stream tokens from a chat messages list. Yields content strings."""
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider with vision and streaming support."""

    def __init__(self, config: Settings):
        try:
            from openai import OpenAI, AsyncOpenAI

            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.LLM_MODEL
            logger.info("OpenAI provider initialized", extra={"model": self.model})
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI provider: {e}")

    def _build_messages(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> list[dict]:
        content = []
        content.append({"type": "text", "text": prompt})

        if images:
            for img in images:
                b64 = b64encode(img).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        return [{"role": "user", "content": content}]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            messages = self._build_messages(prompt, images)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:
            logger.error("OpenAI API call failed", extra={"error": str(e)})
            raise LLMError(f"OpenAI generation failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_sync(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            messages = self._build_messages(prompt, images)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:
            logger.error("OpenAI sync call failed", extra={"error": str(e)})
            raise LLMError(f"OpenAI sync generation failed: {e}")

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("OpenAI stream failed", extra={"error": str(e)})
            raise LLMError(f"OpenAI streaming failed: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with vision and streaming support."""

    def __init__(self, config: Settings):
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.async_client = anthropic.AsyncAnthropic(
                api_key=config.ANTHROPIC_API_KEY
            )
            self.model = config.LLM_MODEL
            logger.info("Anthropic provider initialized", extra={"model": self.model})
        except Exception as e:
            raise LLMError(f"Failed to initialize Anthropic provider: {e}")

    def _build_content(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> list[dict]:
        content = []
        if images:
            for img in images:
                b64 = b64encode(img).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
        content.append({"type": "text", "text": prompt})
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            content = self._build_content(prompt, images)
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
            )
        except Exception as e:
            logger.error("Anthropic API call failed", extra={"error": str(e)})
            raise LLMError(f"Anthropic generation failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_sync(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            content = self._build_content(prompt, images)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
            )
        except Exception as e:
            logger.error("Anthropic sync call failed", extra={"error": str(e)})
            raise LLMError(f"Anthropic sync generation failed: {e}")

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        try:
            # Extract system message if present
            system = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    user_messages.append(msg)

            kwargs = {
                "model": self.model,
                "max_tokens": 2048,
                "messages": user_messages,
            }
            if system:
                kwargs["system"] = system

            async with self.async_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error("Anthropic stream failed", extra={"error": str(e)})
            raise LLMError(f"Anthropic streaming failed: {e}")


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider — OpenAI-compatible API with streaming support."""

    def __init__(self, config: Settings):
        try:
            from openai import OpenAI, AsyncOpenAI

            self.client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
            )
            self.async_client = AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
            )
            self.model = config.LLM_MODEL
            logger.info(
                "OpenRouter provider initialized",
                extra={"model": self.model, "base_url": config.OPENROUTER_BASE_URL},
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenRouter provider: {e}")

    def _build_messages(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> list[dict]:
        content = []
        content.append({"type": "text", "text": prompt})

        if images:
            for img in images:
                b64 = b64encode(img).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        return [{"role": "user", "content": content}]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            messages = self._build_messages(prompt, images)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model or self.model,
                usage=usage,
            )
        except Exception as e:
            logger.error("OpenRouter API call failed", extra={"error": str(e)})
            raise LLMError(f"OpenRouter generation failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_sync(
        self, prompt: str, images: Optional[list[bytes]] = None
    ) -> LLMResponse:
        try:
            messages = self._build_messages(prompt, images)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model or self.model,
                usage=usage,
            )
        except Exception as e:
            logger.error("OpenRouter sync call failed", extra={"error": str(e)})
            raise LLMError(f"OpenRouter sync generation failed: {e}")

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("OpenRouter stream failed", extra={"error": str(e)})
            raise LLMError(f"OpenRouter streaming failed: {e}")


def get_llm_provider(config: Settings) -> BaseLLMProvider:
    """Factory — returns the configured LLM provider."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }
    provider_name = config.LLM_PROVIDER.lower()
    if provider_name not in providers:
        raise LLMError(
            f"Unknown LLM provider: {provider_name}. Choose from: {list(providers.keys())}"
        )
    return providers[provider_name](config)
