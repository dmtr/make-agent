"""Tests for the middleware system."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
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
        resp = Response(
            session_id="s", content="ok", input_tokens=0, output_tokens=0, model=""
        )
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
        resp = Response(
            session_id="s1",
            content="reply",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4",
        )
        await mw.after_response(req, resp)
        mem.store.assert_any_call("user", "hello")
        mem.store.assert_any_call("agent", "reply")

    async def test_after_response_records_token_usage_when_model_present(self):
        mem = self._make_memory()
        mw = SessionMiddleware(mem)
        req = Request(session_id="s1", message="hi")
        resp = Response(
            session_id="s1",
            content="reply",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4",
        )
        await mw.after_response(req, resp)
        mem.record_token_usage.assert_called_once_with("s1", "gpt-4", 10, 5)

    async def test_after_response_skips_token_usage_when_no_model(self):
        mem = self._make_memory()
        mw = SessionMiddleware(mem)
        req = Request(session_id="s1", message="hi")
        resp = Response(
            session_id="s1", content="reply", input_tokens=0, output_tokens=0, model=""
        )
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
    from unittest.mock import MagicMock

    from make_agent.agent_core import AgenticLoop

    async def _fake_astream(msg):
        for cb in cbs:
            yield cb

    loop = MagicMock(spec=AgenticLoop)
    loop.astream = _fake_astream
    loop._max_tool_output = 0
    loop._messages = []
    loop.compact_history = MagicMock(return_value=(0, 0))
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

    async def test_all_middlewares_wired_into_chain(self):
        """Each middleware in [A,B,C,D] must actually execute — not just the outermost."""
        th = _make_tool_handler()
        executed: list[str] = []

        def make_mw(name: str) -> MiddlewareBase:
            class _Mw(MiddlewareBase):
                def __call__(self, request, call_next):
                    async def _gen():
                        executed.append(f"start:{name}")
                        async for event in call_next(request):
                            yield event
                        executed.append(f"end:{name}")

                    return _gen()

                async def after_response(self, request, response):
                    executed.append(f"after:{name}")

            return _Mw()

        a, b, c, d = make_mw("A"), make_mw("B"), make_mw("C"), make_mw("D")
        manager = AgentManager(th, middlewares=[a, b, c, d])
        session_id = manager.get_session_id()
        _mock_loop_with_cbs(manager, session_id, [MessageCallback("done")])

        [e async for e in manager.astream_events(session_id, "go")]

        # Verify all 4 middlewares executed (not just D)
        start_events = [e for e in executed if e.startswith("start:")]
        assert len(start_events) == 4, (
            f"Expected 4 start events, got {len(start_events)}: {start_events}"
        )
        # Verify correct nesting order: D→C→B→A enter, A→B→C→D exit
        assert start_events[0] == "start:D"
        assert start_events[1] == "start:C"
        assert start_events[2] == "start:B"
        assert start_events[3] == "start:A"
        # Verify after_response order: A→B→C→D
        after_events = [e for e in executed if e.startswith("after:")]
        assert after_events == ["after:A", "after:B", "after:C", "after:D"]

    async def test_middleware_after_response_runs_even_if_previous_fails(self):
        """Cleanup (after_response) must not short-circuit on errors."""
        th = _make_tool_handler()
        after_calls: list[str] = []

        class FailingMiddleware(MiddlewareBase):
            async def after_response(self, request, response):
                after_calls.append("failing")
                raise RuntimeError("cleanup error")

        class HealthyMiddleware(MiddlewareBase):
            async def after_response(self, request, response):
                after_calls.append("healthy")

        manager = AgentManager(
            th, middlewares=[HealthyMiddleware(), FailingMiddleware()]
        )
        session_id = manager.get_session_id()
        _mock_loop_with_cbs(manager, session_id, [MessageCallback("done")])

        with pytest.raises(RuntimeError):
            [e async for e in manager.astream_events(session_id, "go")]

        # Both should have been called — failing one first, then healthy one
        # (order may vary; the key is that neither is skipped)
        assert "failing" in after_calls
        assert "healthy" in after_calls

    async def test_single_middleware_runs_correctly(self):
        """Edge case: a single middleware should still work properly."""
        th = _make_tool_handler()
        calls: list[str] = []

        class SingleMw(MiddlewareBase):
            def __call__(self, request, call_next):
                async def _gen():
                    calls.append("enter")
                    async for event in call_next(request):
                        yield event
                    calls.append("exit")

                return _gen()

            async def after_response(self, request, response):
                calls.append(f"after:{response.content}")

        manager = AgentManager(th, middlewares=[SingleMw()])
        session_id = manager.get_session_id()
        _mock_loop_with_cbs(manager, session_id, [MessageCallback("final")])

        [e async for e in manager.astream_events(session_id, "hi")]

        assert calls == ["enter", "exit", "after:final"]

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


class TestIsContextExceeded:
    def test_context_window_exceeded_error(self):
        import anthropic
        from make_agent.provider import is_context_exceeded

        exc = anthropic.BadRequestError(
            message="request (50000 tokens) exceeds the available context size (40000 tokens)",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert is_context_exceeded(exc)

    def test_bad_request_with_context_message(self):
        import anthropic
        from make_agent.provider import is_context_exceeded

        exc = anthropic.BadRequestError(
            message="request (44403 tokens) exceeds the available context size (42752 tokens)",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert is_context_exceeded(exc)

    def test_bad_request_unrelated_message(self):
        import anthropic
        from make_agent.provider import is_context_exceeded

        exc = anthropic.BadRequestError(
            message="invalid parameter: max_tokens must be positive",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert not is_context_exceeded(exc)

    def test_generic_exception_returns_false(self):
        from make_agent.provider import is_context_exceeded

        assert not is_context_exceeded(ValueError("something went wrong"))

    def test_non_anthropic_bad_request_with_context_message(self):
        """A non-anthropic exception with status_code=400 and a context-overflow message is detected."""
        from make_agent.provider import is_context_exceeded

        class CustomBadRequestError(Exception):
            status_code = 400

        exc = CustomBadRequestError(
            "request (11313 tokens) exceeds the available context size (8192 tokens)"
        )
        assert is_context_exceeded(exc)

    def test_non_anthropic_bad_request_unrelated_message(self):
        """A non-anthropic exception with status_code=400 but unrelated message is not detected."""
        from make_agent.provider import is_context_exceeded

        class CustomBadRequestError(Exception):
            status_code = 400

        exc = CustomBadRequestError("invalid parameter: stop_sequences")
        assert not is_context_exceeded(exc)


class TestIsCorruptMessageHistory:
    def test_anthropic_failed_to_parse_tool_call(self):
        import anthropic
        from make_agent.provider import is_corrupt_message_history

        exc = anthropic.BadRequestError(
            message="AnthropicException - Failed to parse tool call arguments for tool 'execute_skill' (Anthropic tool invoke). Error: Unterminated string start",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert is_corrupt_message_history(exc)

    def test_unrelated_bad_request_returns_false(self):
        import anthropic
        from make_agent.provider import is_corrupt_message_history

        exc = anthropic.BadRequestError(
            message="invalid parameter: max_tokens must be positive",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert not is_corrupt_message_history(exc)

    def test_context_exceeded_error_returns_false(self):
        import anthropic
        from make_agent.provider import is_corrupt_message_history

        exc = anthropic.BadRequestError(
            message="request (50000 tokens) exceeds the available context size (40000 tokens)",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert not is_corrupt_message_history(exc)

    def test_non_bad_request_returns_false(self):
        from make_agent.provider import is_corrupt_message_history

        assert not is_corrupt_message_history(ValueError("something else"))
