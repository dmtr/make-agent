"""Tests for the middleware system."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from make_agent.agent_core import (
    AgentManager,
    DoneEvent,
    MessageCallback,
    SessionMiddleware,
    TokenCallback,
    TokenEvent,
    UsageCallback,
)
from make_agent.agent_core.middleware import MiddlewareBase, Request, Response


# ── Request / Response ────────────────────────────────────────────────────────


class TestRequestResponse:
    def test_request_fields(self):
        req = Request(session_id="s1", message="hello")
        assert req.session_id == "s1"
        assert req.message == "hello"

    def test_response_fields(self):
        resp = Response(
            session_id="s1",
            content="reply",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4",
        )
        assert resp.session_id == "s1"
        assert resp.content == "reply"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "gpt-4"


# ── MiddlewareBase ────────────────────────────────────────────────────────────


class TestMiddlewareBase:
    async def test_passthrough_yields_all_events(self):
        events = [TokenEvent(text="a"), DoneEvent(content="done")]

        async def core(req):
            for e in events:
                yield e

        mw = MiddlewareBase()
        req = Request(session_id="s", message="hi")
        collected = [e async for e in mw(req, core)]
        assert collected == events

    async def test_after_response_is_noop(self):
        mw = MiddlewareBase()
        req = Request(session_id="s", message="hi")
        resp = Response(session_id="s", content="ok", input_tokens=0, output_tokens=0, model="")
        # Should not raise
        await mw.after_response(req, resp)


# ── SessionMiddleware ─────────────────────────────────────────────────────────


class TestSessionMiddleware:
    def _make_memory(self):
        mem = MagicMock()
        mem.store = MagicMock()
        mem.record_token_usage = MagicMock()
        return mem

    async def test_passthrough_yields_all_events(self):
        events = [TokenEvent(text="hi"), DoneEvent(content="done")]

        async def core(req):
            for e in events:
                yield e

        mw = SessionMiddleware(self._make_memory())
        req = Request(session_id="s", message="hi")
        collected = [e async for e in mw(req, core)]
        assert collected == events

    async def test_after_response_stores_user_and_agent(self):
        mem = self._make_memory()
        mw = SessionMiddleware(mem)
        req = Request(session_id="s1", message="hello")
        resp = Response(session_id="s1", content="reply", input_tokens=10, output_tokens=5, model="gpt-4")
        await mw.after_response(req, resp)
        mem.store.assert_any_call("user", "hello")
        mem.store.assert_any_call("agent", "reply")

    async def test_after_response_records_token_usage_when_model_present(self):
        mem = self._make_memory()
        mw = SessionMiddleware(mem)
        req = Request(session_id="s1", message="hi")
        resp = Response(session_id="s1", content="reply", input_tokens=10, output_tokens=5, model="gpt-4")
        await mw.after_response(req, resp)
        mem.record_token_usage.assert_called_once_with("s1", "gpt-4", 10, 5)

    async def test_after_response_skips_token_usage_when_no_model(self):
        mem = self._make_memory()
        mw = SessionMiddleware(mem)
        req = Request(session_id="s1", message="hi")
        resp = Response(session_id="s1", content="reply", input_tokens=0, output_tokens=0, model="")
        await mw.after_response(req, resp)
        mem.record_token_usage.assert_not_called()


# ── AgentManager middleware chain ─────────────────────────────────────────────


def _make_tool_handler():
    th = MagicMock()
    th.is_skill_trusted = MagicMock(return_value=True)
    th.execute = AsyncMock(return_value=MagicMock(output="ok", is_error=False))
    return th


def _mock_loop_with_cbs(manager, session_id, cbs):
    """Inject a mock AgenticLoop that yields given callbacks."""
    from make_agent.agent_core import AgenticLoop
    from unittest.mock import MagicMock

    async def _fake_astream(msg):
        for cb in cbs:
            yield cb

    loop = MagicMock(spec=AgenticLoop)
    loop.astream = _fake_astream
    loop._max_tool_output = 0
    manager._sessions[session_id] = loop


class TestAgentManagerMiddlewareChain:
    async def test_no_middlewares_streams_events(self):
        th = _make_tool_handler()
        manager = AgentManager(th)
        session_id = manager.get_session_id()
        cbs = [TokenCallback("hello "), MessageCallback("hello world")]
        _mock_loop_with_cbs(manager, session_id, cbs)

        events = [e async for e in manager.astream_events(session_id, "hi")]
        assert [type(e).__name__ for e in events] == ["TokenEvent", "DoneEvent"]

    async def test_after_response_called_after_stream_exhausted(self):
        th = _make_tool_handler()
        calls: list[str] = []

        class TrackingMiddleware(MiddlewareBase):
            def __call__(self, request, call_next):
                async def _gen():
                    async for event in call_next(request):
                        calls.append(f"event:{type(event).__name__}")
                        yield event
                return _gen()

            async def after_response(self, request, response):
                calls.append(f"after:{response.content}")

        manager = AgentManager(th, middlewares=[TrackingMiddleware()])
        session_id = manager.get_session_id()
        cbs = [MessageCallback("final")]
        _mock_loop_with_cbs(manager, session_id, cbs)

        [e async for e in manager.astream_events(session_id, "hi")]

        assert calls == ["event:DoneEvent", "after:final"]

    async def test_chain_order_innermost_first(self):
        """middlewares=[A, B, C]: request order C→B→A→core, after_response order A→B→C."""
        th = _make_tool_handler()
        order: list[str] = []

        def make_mw(name: str) -> MiddlewareBase:
            class _Mw(MiddlewareBase):
                def __call__(self, request, call_next):
                    async def _gen():
                        order.append(f"enter:{name}")
                        async for event in call_next(request):
                            yield event
                        order.append(f"exit:{name}")
                    return _gen()

                async def after_response(self, request, response):
                    order.append(f"after:{name}")
            return _Mw()

        a, b, c = make_mw("A"), make_mw("B"), make_mw("C")
        manager = AgentManager(th, middlewares=[a, b, c])
        session_id = manager.get_session_id()
        _mock_loop_with_cbs(manager, session_id, [MessageCallback("done")])

        [e async for e in manager.astream_events(session_id, "go")]

        # entry: C outermost first, then B, then A
        assert order.index("enter:C") < order.index("enter:B") < order.index("enter:A")
        # after_response: A innermost first, then B, then C
        assert order.index("after:A") < order.index("after:B") < order.index("after:C")

    async def test_response_accumulates_usage_from_multiple_turns(self):
        th = _make_tool_handler()
        received: list[Response] = []

        class CaptureMiddleware(MiddlewareBase):
            async def after_response(self, request, response):
                received.append(response)

        manager = AgentManager(th, middlewares=[CaptureMiddleware()])
        session_id = manager.get_session_id()
        cbs = [
            UsageCallback(model="gpt-4", input_tokens=10, output_tokens=5),
            UsageCallback(model="gpt-4", input_tokens=20, output_tokens=8),
            MessageCallback("result"),
        ]
        _mock_loop_with_cbs(manager, session_id, cbs)

        [e async for e in manager.astream_events(session_id, "go")]

        assert len(received) == 1
        resp = received[0]
        assert resp.input_tokens == 30
        assert resp.output_tokens == 13
        assert resp.content == "result"
        assert resp.model == "gpt-4"

    def test_get_token_stats_delegates_to_session_middleware(self, tmp_path):
        from make_agent.memory import Memory

        mem = Memory(tmp_path / "test.db")
        mem.record_token_usage("sess-1", "gpt-4", 100, 50)
        th = _make_tool_handler()
        manager = AgentManager(th, middlewares=[SessionMiddleware(mem)])

        stats = manager.get_token_stats("sess-1")
        assert stats["input_tokens"] == 100
        assert stats["output_tokens"] == 50
        mem.close()

    def test_get_token_stats_returns_empty_without_session_middleware(self):
        th = _make_tool_handler()
        manager = AgentManager(th)
        assert manager.get_token_stats("any") == {}


# ── AgentManager compact path ─────────────────────────────────────────────────


def _make_agent_config():
    from make_agent.agent_core.loop import AgentConfig

    return AgentConfig(system_prompt="sys", model="gpt-4")


class TestAgentManagerCompact:
    def _turns(self, n: int) -> list[dict]:
        """Build *n* user+assistant turn pairs."""
        msgs = []
        for i in range(1, n + 1):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({"role": "assistant", "content": f"answer {i}"})
        return msgs

    async def test_compact_event_emitted_when_tokens_exceed_threshold(self):
        from make_agent.agent_core import CompactEvent
        from unittest.mock import AsyncMock, patch

        th = _make_tool_handler()
        manager = AgentManager(th, compact_threshold=100)
        session_id = manager.get_session_id()
        cbs = [UsageCallback(model="gpt-4", input_tokens=200, output_tokens=10), MessageCallback("done")]
        _mock_loop_with_cbs(manager, session_id, cbs)

        # Seed 6 turns so _compact_session actually removes messages.
        loop = manager._sessions[session_id]
        loop._config = _make_agent_config()
        loop.messages = [{"role": "system", "content": "sys"}] + self._turns(6)

        with patch("make_agent.agent_core.agent._summarize_messages", AsyncMock(return_value="summary")):
            events = [e async for e in manager.astream_events(session_id, "go")]
        types = [type(e).__name__ for e in events]
        assert "CompactEvent" in types
        compact_event = next(e for e in events if isinstance(e, CompactEvent))
        assert compact_event.messages_removed > 0

    async def test_compact_event_not_emitted_when_threshold_zero(self):
        from make_agent.agent_core import CompactEvent

        th = _make_tool_handler()
        manager = AgentManager(th, compact_threshold=0)
        session_id = manager.get_session_id()
        cbs = [UsageCallback(model="gpt-4", input_tokens=99999, output_tokens=10), MessageCallback("done")]
        _mock_loop_with_cbs(manager, session_id, cbs)

        events = [e async for e in manager.astream_events(session_id, "go")]
        assert not any(isinstance(e, CompactEvent) for e in events)

    async def test_compact_event_not_emitted_when_tokens_below_threshold(self):
        from make_agent.agent_core import CompactEvent

        th = _make_tool_handler()
        manager = AgentManager(th, compact_threshold=1000)
        session_id = manager.get_session_id()
        cbs = [UsageCallback(model="gpt-4", input_tokens=500, output_tokens=10), MessageCallback("done")]
        _mock_loop_with_cbs(manager, session_id, cbs)

        events = [e async for e in manager.astream_events(session_id, "go")]
        assert not any(isinstance(e, CompactEvent) for e in events)

    async def test_compact_session_prunes_in_place_and_returns_count(self):
        from make_agent.agent_core.loop import AgentConfig, AgenticLoop
        from unittest.mock import AsyncMock, patch

        th = _make_tool_handler()
        manager = AgentManager(th, compact_threshold=100)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)

        loop = manager._sessions[session_id]
        # 6 turns → compact removes most messages, replacing with a summary
        loop._messages = [{"role": "system", "content": "sys"}] + self._turns(6)
        original_len = len(loop._messages)

        with patch("make_agent.agent_core.agent._summarize_messages", AsyncMock(return_value="summary")):
            removed = await manager._compact_session(session_id)
        assert removed > 0
        assert len(loop.messages) == original_len - removed
        assert manager._sessions[session_id] is loop  # same object, pruned in-place

    async def test_compact_session_returns_zero_when_nothing_pruned(self):
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th, compact_threshold=100)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)

        loop = manager._sessions[session_id]
        # Only 1 non-system message → too few to summarise, nothing removed
        loop._messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        removed = await manager._compact_session(session_id)
        assert removed == 0

    async def test_compact_triggered_on_context_exceeded_error(self):
        """When the API raises a context-window error, compact_fn prunes messages and the turn retries."""
        import litellm
        import pytest
        from unittest.mock import AsyncMock, patch
        from make_agent.agent_core import CompactEvent
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)

        # Seed 6 past turns so compact_fn has something to remove.
        loop = manager._sessions[session_id]
        loop._messages.extend(self._turns(6))

        call_count = 0

        async def _fake_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise litellm.BadRequestError(
                    message="request (50000 tokens) exceeds the available context size (40000 tokens)",
                    model="claude",
                    llm_provider="anthropic",
                )

            async def _stream():
                from tests.test_agent import _make_text_stream
                async for chunk in _make_text_stream("ok"):
                    yield chunk

            return _stream()

        with (
            patch("make_agent.agent_core.loop.acompletion_with_retry", _fake_completion),
            patch("make_agent.agent_core.agent._summarize_messages", AsyncMock(return_value="summary")),
        ):
            events = [e async for e in manager.astream_events(session_id, "do something")]

        types = [type(e).__name__ for e in events]
        assert "CompactEvent" in types
        assert "DoneEvent" in types
        assert call_count == 2  # first failed, second succeeded after compact

    async def test_compact_triggered_during_stream_iteration(self):
        """Context-exceeded raised during stream iteration (not at call time) is also handled."""
        import litellm
        from unittest.mock import AsyncMock, patch
        from make_agent.agent_core import CompactEvent
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)

        loop = manager._sessions[session_id]
        loop._messages.extend(self._turns(6))

        call_count = 0

        async def _fake_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            async def _error_stream():
                raise litellm.BadRequestError(
                    message="request (43980 tokens) exceeds the available context size (42752 tokens)",
                    model="claude",
                    llm_provider="anthropic",
                )
                yield  # make it an async generator

            async def _ok_stream():
                from tests.test_agent import _make_text_stream
                async for chunk in _make_text_stream("ok"):
                    yield chunk

            return _error_stream() if call_count == 1 else _ok_stream()

        with (
            patch("make_agent.agent_core.loop.acompletion_with_retry", _fake_completion),
            patch("make_agent.agent_core.agent._summarize_messages", AsyncMock(return_value="summary")),
        ):
            events = [e async for e in manager.astream_events(session_id, "do something")]

        types = [type(e).__name__ for e in events]
        assert "CompactEvent" in types
        assert "DoneEvent" in types
        assert call_count == 2

    async def test_context_exceeded_reraises_when_nothing_to_prune(self):
        """If nothing can be pruned, re-raises the original error."""
        import litellm
        import pytest
        from unittest.mock import patch
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)
        # No purgeable messages — compact_fn returns the same list.

        async def _fake_completion(*args, **kwargs):
            raise litellm.BadRequestError(
                message="request exceeds the available context size",
                model="claude",
                llm_provider="anthropic",
            )

        with patch("make_agent.agent_core.loop.acompletion_with_retry", _fake_completion):
            with pytest.raises(litellm.BadRequestError):
                [e async for e in manager.astream_events(session_id, "hi")]

    async def test_compact_triggered_on_custom_provider_error(self):
        """A non-litellm exception with status_code=400 from a custom provider triggers compact."""
        import pytest
        from unittest.mock import AsyncMock, patch
        from make_agent.agent_core import CompactEvent
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th)
        config = AgentConfig(system_prompt="sys", model="custom/my-model")
        session_id = manager.create_session(config)

        loop = manager._sessions[session_id]
        loop._messages.extend(self._turns(6))

        call_count = 0

        class CustomBadRequestError(Exception):
            status_code = 400

        async def _fake_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise CustomBadRequestError(
                    "request (11313 tokens) exceeds the available context size (8192 tokens)"
                )

            async def _stream():
                from tests.test_agent import _make_text_stream
                async for chunk in _make_text_stream("ok"):
                    yield chunk

            return _stream()

        with (
            patch("make_agent.agent_core.loop.acompletion_with_retry", _fake_completion),
            patch("make_agent.agent_core.agent._summarize_messages", AsyncMock(return_value="summary")),
        ):
            events = [e async for e in manager.astream_events(session_id, "do something")]

        types = [type(e).__name__ for e in events]
        assert "CompactEvent" in types
        assert "DoneEvent" in types
        assert call_count == 2

    async def test_multi_round_compaction(self):
        """If context is still exceeded after one compact round, a second round fires.

        This tests that the removal of the 'compacted' flag allows reactive compaction
        to retry when the first compact round wasn't enough to fit within the window.
        Uses a custom compact_fn that removes one pair per call to demonstrate multi-round.
        """
        import litellm
        from unittest.mock import patch
        from make_agent.agent_core import CompactEvent
        from make_agent.agent_core.loop import AgentConfig

        th = _make_tool_handler()
        manager = AgentManager(th)
        config = AgentConfig(system_prompt="sys", model="gpt-4")
        session_id = manager.create_session(config)

        loop = manager._sessions[session_id]
        # Seed 8 past turns
        loop._messages.extend(self._turns(8))

        # Custom compact that removes only the oldest turn per call
        async def _one_at_a_time(messages):
            system = [m for m in messages if m.get("role") == "system"]
            rest = [m for m in messages if m.get("role") != "system"]
            # Find first user message and remove its turn
            for i, msg in enumerate(rest):
                if msg.get("role") == "user":
                    # Find where the next user message is
                    end = next((j for j in range(i + 1, len(rest)) if rest[j].get("role") == "user"), len(rest))
                    return system + rest[:i] + rest[end:]
            return messages

        loop._compact_fn = _one_at_a_time

        call_count = 0

        async def _fake_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise litellm.BadRequestError(
                    message="request exceeds the available context size",
                    model="gpt-4",
                    llm_provider="openai",
                )

            async def _stream():
                from tests.test_agent import _make_text_stream
                async for chunk in _make_text_stream("ok"):
                    yield chunk

            return _stream()

        with patch("make_agent.agent_core.loop.acompletion_with_retry", _fake_completion):
            events = [e async for e in manager.astream_events(session_id, "do something")]

        compact_events = [e for e in events if isinstance(e, CompactEvent)]
        assert len(compact_events) == 2  # two rounds of compaction
        assert call_count == 3  # failed twice, succeeded third time


# ── is_context_exceeded ──────────────────────────────────────────────────────


class TestIsContextExceeded:
    def test_context_window_exceeded_error(self):
        import litellm
        from make_agent.provider import is_context_exceeded

        exc = litellm.ContextWindowExceededError(
            message="context window exceeded", model="claude", llm_provider="anthropic"
        )
        assert is_context_exceeded(exc)

    def test_bad_request_with_context_message(self):
        import litellm
        from make_agent.provider import is_context_exceeded

        exc = litellm.BadRequestError(
            message="request (44403 tokens) exceeds the available context size (42752 tokens)",
            model="claude",
            llm_provider="anthropic",
        )
        assert is_context_exceeded(exc)

    def test_bad_request_unrelated_message(self):
        import litellm
        from make_agent.provider import is_context_exceeded

        exc = litellm.BadRequestError(
            message="invalid parameter: max_tokens must be positive",
            model="claude",
            llm_provider="anthropic",
        )
        assert not is_context_exceeded(exc)

    def test_generic_exception_returns_false(self):
        from make_agent.provider import is_context_exceeded

        assert not is_context_exceeded(ValueError("something went wrong"))

    def test_non_litellm_bad_request_with_context_message(self):
        """A non-litellm exception with status_code=400 and a context-overflow message is detected."""
        from make_agent.provider import is_context_exceeded

        class CustomBadRequestError(Exception):
            status_code = 400

        exc = CustomBadRequestError(
            "request (11313 tokens) exceeds the available context size (8192 tokens)"
        )
        assert is_context_exceeded(exc)

    def test_non_litellm_bad_request_unrelated_message(self):
        """A non-litellm exception with status_code=400 but unrelated message is not detected."""
        from make_agent.provider import is_context_exceeded

        class CustomBadRequestError(Exception):
            status_code = 400

        exc = CustomBadRequestError("invalid parameter: stop_sequences")
        assert not is_context_exceeded(exc)
