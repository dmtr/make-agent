"""OpenAI SDK provider implementation.

Messages are already in OpenAI format throughout the codebase, so this
provider passes them through with minimal translation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import openai

from .base import (
    ContextExceededChunk,
    StreamChunk,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    UsageDelta,
    is_context_exceeded,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF_CAP = 60


def _parse_retry_after(exc: openai.RateLimitError) -> float | None:
    """Return wait time in seconds from Retry-After headers, or None."""
    try:
        headers = exc.response.headers if exc.response is not None else {}
    except Exception:
        return None
    if ms := headers.get("retry-after-ms"):
        return float(ms) / 1000
    if sec := headers.get("retry-after"):
        return float(sec)
    return None


def _normalize_model(model: str) -> str:
    """Strip the ``openai/`` provider prefix if present."""
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


class OpenAIProvider:
    """Streaming provider backed by the native OpenAI SDK.

    Messages are already in OpenAI format so they are forwarded as-is.
    Normalises the stream to :data:`~make_agent.provider.base.StreamChunk`
    objects.
    """

    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI()

    async def astream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        use_prompt_cache: bool = False,
        reasoning_effort: str = "",
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict[str, Any] = dict(
            model=_normalize_model(model),
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = tools
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            stream = await self._create_with_retry(kwargs)
            async for chunk in stream:
                if chunk.usage is not None:
                    yield UsageDelta(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                    continue

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    yield TextDelta(text=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            yield ToolCallStart(
                                index=tc.index,
                                id=tc.id,
                                name=tc.function.name,
                            )
                        if tc.function and tc.function.arguments:
                            yield ToolCallDelta(
                                index=tc.index,
                                args_delta=tc.function.arguments,
                            )
        except Exception as e:
            if is_context_exceeded(e):
                yield ContextExceededChunk()
            else:
                raise

    async def _create_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call ``client.chat.completions.create`` with rate-limit retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = _parse_retry_after(e) or min(2**attempt, BACKOFF_CAP)
                print(
                    f"Rate limited, retrying in {wait:.0f}s"
                    f" (attempt {attempt + 1}/{MAX_RETRIES})...",
                    flush=True,
                )
                await asyncio.sleep(wait)
