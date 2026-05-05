"""Tests for HTML conversation export helpers."""

from make_agent.commands import _render_html


def test_render_html_escapes_model_name():
    html = _render_html([], '<img src=x onerror=alert(1)>')
    assert '<img src=x onerror=alert(1)>' not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
