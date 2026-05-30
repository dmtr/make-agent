"""Anthropic SDK provider implementation.

Translates OpenAI-style message dicts to Anthropic wire format at the call
boundary.  The rest of the codebase uses OpenAI-style dicts throughout.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import anthropic

from .base import StreamChunk, TextDelta, ToolCallDelta, ToolCallStart, UsageDelta

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF_CAP = 60


def _parse_retry_after(exc: anthropic.RateLimitError) -> float | None:
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


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool schemas to Anthropic format."""
    result = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def _openai_messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to ``(system_text, anthropic_messages)``.

    Extracts system messages into a single string.  Converts tool_calls
    assistant messages and tool result messages to Anthropic content blocks.
    Consecutive tool result messages are merged into a single user message.
    """
    system_parts: list[str] = []
    result: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block["text"])
            continue

        if role == "user":
            result.append({"role": "user", "content": content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": args,
                        }
                    )
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": content,
            }
            # Group consecutive tool results into a single user message.
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append(tool_result_block)
            else:
                result.append({"role": "user", "content": [tool_result_block]})

    return "\n\n".join(system_parts), result


def _normalize_model(model: str) -> str:
    """Strip the ``anthropic/`` provider prefix if present."""
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


class AnthropicProvider:
    """Streaming provider backed by the native Anthropic SDK.

    Translates OpenAI-style message dicts to Anthropic wire format and
    normalises the stream back to :data:`~make_agent.provider.base.StreamChunk`
    objects.
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def astream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        use_prompt_cache: bool = False,
        reasoning_effort: str = "",
    ) -> AsyncIterator[StreamChunk]:
        system_text, anthropic_messages = _openai_messages_to_anthropic(messages)

        system_content: str | list[dict]
        if use_prompt_cache and system_text:
            system_content = [
                {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_content = system_text

        anthropic_tools = _openai_tools_to_anthropic(tools)

        kwargs: dict[str, Any] = dict(
            model=_normalize_model(model),
            max_tokens=max_tokens,
            messages=anthropic_messages,
            stream=True,
        )
        if system_content:
            kwargs["system"] = system_content
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        stream = await self._create_with_retry(kwargs)
        async for event in stream:
            if event.type == "message_start":
                usage = event.message.usage
                yield UsageDelta(input_tokens=usage.input_tokens, output_tokens=0)
            elif event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    yield ToolCallStart(index=event.index, id=block.id, name=block.name)
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextDelta(text=delta.text)
                elif delta.type == "input_json_delta":
                    yield ToolCallDelta(index=event.index, args_delta=delta.partial_json)
            elif event.type == "message_delta":
                yield UsageDelta(input_tokens=0, output_tokens=event.usage.output_tokens)

    async def _create_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call ``client.messages.create`` with rate-limit retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = _parse_retry_after(e) or min(2**attempt, BACKOFF_CAP)
                print(
                    f"Rate limited, retrying in {wait:.0f}s"
                    f" (attempt {attempt + 1}/{MAX_RETRIES})...",
                    flush=True,
                )
                await asyncio.sleep(wait)


