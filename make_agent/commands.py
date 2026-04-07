import html as _html
from datetime import datetime
from pathlib import Path
from string import Template

_CSS = """
body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;background:#f5f5f5;color:#222}
h1{font-size:1rem;color:#666;border-bottom:1px solid #ddd;padding-bottom:.5rem;margin-bottom:1.5rem}
.msg{margin:1rem 0}
.label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
.bubble{border-radius:10px;padding:.7rem 1rem;white-space:pre-wrap;word-break:break-word;line-height:1.5}
.system .label{color:#888}
.system .bubble{background:#ececec;font-size:.85rem}
.user .label{color:#1a73e8;text-align:right}
.user .bubble{background:#1a73e8;color:#fff;margin-left:4rem}
.assistant .label{color:#188038}
.assistant .bubble{background:#fff;border:1px solid #ddd;margin-right:4rem}
details.tc{margin:.4rem 0;border:1px solid #e8d5a3;border-radius:8px;background:#fffbf0}
details.tc summary{cursor:pointer;padding:.45rem .8rem;font-size:.82rem;font-weight:600;color:#856404;user-select:none}
.tc-body{padding:.4rem .8rem .6rem}
.tc-body pre{background:#f5f5f5;border-radius:5px;padding:.5rem;font-size:.78rem;overflow-x:auto;margin:.25rem 0 0}
.tc-label{font-size:.75rem;font-weight:600;color:#555;margin-top:.4rem}
"""

_BODY_TEMPLATE = Template(
    """<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Conversation — ${ts}</title>
        <style>${CSS}</style>
        </head>
        <body>
        <h1>Conversation &middot; ${ts} &middot; ${model}</h1>
        ${body}
        </body>
        </html>"""
)


def _render_html(messages: list[dict], model: str) -> str:
    """Render *messages* as a self-contained HTML conversation page."""

    esc = _html.escape

    # Pre-index tool results by tool_call_id for O(1) lookup.
    tool_results: dict[str, str] = {msg["tool_call_id"]: msg.get("content", "") for msg in messages if msg.get("role") == "tool"}

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = esc(msg.get("content", ""))
            parts.append(f'<div class="msg system">' f'<div class="label">System prompt</div>' f'<div class="bubble">{content}</div></div>')

        elif role == "user":
            content = esc(msg.get("content", ""))
            parts.append(f'<div class="msg user">' f'<div class="label">You</div>' f'<div class="bubble">{content}</div></div>')

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            content = esc(msg.get("content") or "")
            inner: list[str] = []
            for tc in tool_calls:
                name = esc(tc["function"]["name"])
                args = esc(tc["function"].get("arguments", ""))
                result = esc(tool_results.get(tc["id"], "(no result)"))
                inner.append(
                    f'<details class="tc">'
                    f"<summary>🔧 {name}</summary>"
                    f'<div class="tc-body">'
                    f'<div class="tc-label">Arguments</div><pre>{args}</pre>'
                    f'<div class="tc-label">Result</div><pre>{result}</pre>'
                    f"</div></details>"
                )
            if content:
                inner.append(f'<div class="bubble">{content}</div>')
            if inner:
                parts.append('<div class="msg assistant">' '<div class="label">Assistant</div>' + "".join(inner) + "</div>")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(parts)
    return _BODY_TEMPLATE.substitute(ts=ts, model=model, CSS=_CSS, body=body)


def export_conversation(messages: list[dict], model: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = Path(f"conversation-{ts}.html")
    path.write_text(_render_html(messages, model), encoding="utf-8")
    return path
