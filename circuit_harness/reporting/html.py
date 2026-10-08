# ruff: noqa: E501
"""Escaped HTML rendering primitives for saved circuit episodes."""

import html
import json
import re
from typing import Any


def esc(value: Any) -> str:
    """Escape anything for HTML text/attribute context. Never returns None."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = (
            json.dumps(value, ensure_ascii=False, indent=2)
            if isinstance(value, (dict, list))
            else str(value)
        )
    return html.escape(value, quote=True)


def content_to_text(content: Any) -> str:
    """Chat content may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(
                    part.get("text") or part.get("content") or json.dumps(part, ensure_ascii=False)
                )
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, indent=2)


def block(
    kind: str, label: str, body_html: str, *, meta: str = "", open_by_default: bool = True
) -> str:
    """A titled, collapsible block. Long bodies scroll inside their own box."""
    op = " open" if open_by_default else ""
    meta_html = f'<span class="meta">{esc(meta)}</span>' if meta else ""
    return (
        f'<details class="blk {kind}"{op}>'
        f'<summary><span class="tag">{esc(label)}</span>{meta_html}</summary>'
        f'<div class="body">{body_html}</div>'
        f"</details>"
    )


def pre(text: str) -> str:
    return f'<pre class="txt">{esc(text)}</pre>'


INLINE_TAG_RE = re.compile(
    r"<(?P<tag>think|thinking|tool_call|python_code|code|tool_response|tool_result|answer)>"
    r"(?P<inner>.*?)"
    r"</(?P=tag)>",
    re.S | re.I,
)

_TAG_STYLE = {
    "think": ("think", "reasoning"),
    "thinking": ("think", "reasoning"),
    "tool_call": ("toolcall", "tool call"),
    "python_code": ("toolcall", "python code (inline tool call)"),
    "code": ("toolcall", "code (inline tool call)"),
    "tool_response": ("tres", "tool response"),
    "tool_result": ("tres", "tool result"),
    "answer": ("final", "answer"),
}


def render_inline_text(text: str) -> str:
    """Render assistant text, promoting inline special tokens into blocks.

    Everything is escaped; tags are recognised structurally before escaping so a
    literal `<answer>` in the output can never swallow the rest of the page.
    """
    if not text:
        return '<p class="empty">(no text)</p>'
    if not INLINE_TAG_RE.search(text):
        return pre(text)
    out: list[str] = []
    pos = 0
    for m in INLINE_TAG_RE.finditer(text):
        lead = text[pos : m.start()]
        if lead.strip():
            out.append(pre(lead.strip()))
        tag = m.group("tag").lower()
        kind, label = _TAG_STYLE.get(tag, ("usr", tag))
        inner = m.group("inner").strip()
        # tool_call payloads are usually JSON - pretty-print when they are
        if kind == "toolcall":
            try:
                inner = json.dumps(json.loads(inner), ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        looks_err = kind == "tres" and re.search(r"traceback|error|exception", inner, re.I)
        out.append(
            block(
                "raw" if looks_err else kind,
                label + (" (error)" if looks_err else ""),
                pre(inner),
                meta=f"&lt;{tag}&gt;".replace("&lt;", "<").replace("&gt;", ">"),
                open_by_default=kind != "think",
            )
        )
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        out.append(pre(tail.strip()))
    return "".join(out)


def render_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out.append(pre(str(tc)))
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name") or "?"
        args = fn.get("arguments") if "arguments" in fn else tc.get("arguments")
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        out.append(
            f'<div class="tcall"><div class="tname">&#9654; {esc(name)}</div>{pre(content_to_text(args) if not isinstance(args, str) else args)}</div>'
        )
    return block("toolcall", f"tool calls ({len(tool_calls)})", "".join(out))


def render_messages(messages: list[Any], start: int) -> str:
    """Render messages from index `start` on (the delta new to this turn)."""
    out = []
    for msg in messages[start:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        text = content_to_text(msg.get("content"))
        kind = {"system": "sys", "user": "usr", "tool": "tres", "assistant": "asst"}.get(
            role, "usr"
        )
        label = {"tool": "tool result"}.get(role, role)
        meta = ""
        if role == "tool":
            meta = str(msg.get("name") or msg.get("tool_call_id") or "")
        # tool calls carried on an assistant message in history
        extra = render_tool_calls(msg.get("tool_calls"))
        # Inline special tokens can appear on any role: python mode returns the
        # tool observation as a *user* message carrying <tool_response>.
        body = (
            render_inline_text(text)
            if role != "system"
            else (pre(text) if text else '<p class="empty">(no text)</p>')
        )
        out.append(block(kind, label, body + extra, meta=meta, open_by_default=role != "system"))
    return "".join(out)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0f1115;color:#d7dce3;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header.top{position:sticky;top:0;z-index:20;background:#161a21;border-bottom:1px solid #2a303a;padding:14px 20px}
header.top h1{margin:0 0 4px;font-size:17px;color:#fff;letter-spacing:.3px}
header.top .sub{color:#8b95a5;font-size:12px}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
.findings{background:#1b1f27;border:1px solid #3a4150;border-left:3px solid #e0a33e;padding:14px 16px;margin:18px 0;border-radius:6px}
.findings h3{margin:0 0 8px;font-size:13px;color:#e0a33e;text-transform:uppercase;letter-spacing:.6px}
.findings li{margin:6px 0;color:#c3cad4}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
nav.toc a{background:#1b1f27;border:1px solid #2f3641;color:#9fb0c6;padding:6px 10px;border-radius:5px;text-decoration:none;font-size:12px}
nav.toc a:hover{border-color:#5b667a;color:#fff}
.cell{background:#141821;border:1px solid #2a303a;border-radius:8px;padding:16px;margin:22px 0}
.cell h2{margin:0 0 2px;font-size:16px;color:#fff;display:flex;align-items:center;gap:9px}
.cell .mode{font-size:11px;background:#26303f;color:#8fb4e0;padding:2px 8px;border-radius:10px}
.cell .path{color:#69737f;font-size:11px;margin-bottom:10px;word-break:break-all}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.ok{background:#4ec27a}.dot.bad{background:#e0554e}.dot.unk{background:#7a838f}
.cellsum{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.cellkv{background:#1b202a;border:1px solid #2b323d;border-radius:5px;padding:5px 9px;font-size:11px}
.cellkv span{color:#7d8896;margin-right:6px}
.cellkv b{color:#e6ebf2;font-weight:600}
.seq{margin:12px 0;padding:9px 11px;background:#11151c;border:1px solid #262c36;border-radius:6px;font-size:11px;color:#5e6773;line-height:2.1}
.pill{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11px;margin:0 3px;background:#2b323d;color:#cbd3dd}
.pill sub{font-size:9px;opacity:.75}
.s-problem{background:#2f3a4a;color:#9dc0e8}.s-propose{background:#2c4034;color:#86d3a2}
.s-verify{background:#453529;color:#e5b478}.s-revise{background:#40304a;color:#c39fe0}
.s-final{background:#1f3f45;color:#83cfd8}
.step{border:1px solid #262c36;border-radius:7px;margin:14px 0;overflow:hidden}
.stephdr{background:#1a1f28;padding:8px 11px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:11px;color:#78828f}
.stephdr .iter{color:#a9b4c2}
.stephdr .tid{color:#5b6470;word-break:break-all}
.stepbody{padding:11px}
.turn{border-left:2px solid #2b323d;padding-left:11px;margin:11px 0}
.turnhdr{color:#707a87;font-size:11px;margin-bottom:7px;text-transform:uppercase;letter-spacing:.7px}
.toolbar{background:#161b23;border:1px solid #272e38;border-radius:5px;padding:6px 10px;margin:7px 0;font-size:11px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.toolbar .k{color:#7c8592}
.toolbar code{background:#222a34;padding:1px 6px;border-radius:3px;color:#b8c4d2}
.toolbar.nocalls{border-color:#5c4a26;background:#1d1a14}
.flag{background:#4a3a1c;color:#e6b96a;padding:2px 8px;border-radius:10px;font-size:10px}
.flag.ok{background:#1f3320;color:#87c98f}
.blk{border:1px solid #272e38;border-radius:5px;margin:6px 0;background:#12161d}
.blk>summary{cursor:pointer;padding:6px 10px;font-size:11px;color:#8d97a5;user-select:none;list-style:none;display:flex;gap:9px;align-items:center}
.blk>summary::-webkit-details-marker{display:none}
.blk>summary::before{content:"\\25B8";color:#5c6673;font-size:10px}
.blk[open]>summary::before{content:"\\25BE"}
.blk>summary .tag{font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.blk>summary .meta{color:#666f7c;font-weight:400;text-transform:none}
.blk .body{padding:0 10px 9px}
.blk.sys>summary .tag{color:#e0a33e}
.blk.usr>summary .tag{color:#6fa8e8}
.blk.asst>summary .tag{color:#5fc98c}
.blk.tres>summary .tag{color:#d78ad0}
.blk.toolcall>summary .tag{color:#e08a5f}
.blk.think>summary .tag{color:#8b93a1}
.blk.raw>summary .tag{color:#d9645c}
.blk.final>summary .tag{color:#7fd0d8}
pre.txt{margin:0;padding:9px 11px;background:#0c0f14;border:1px solid #232932;border-radius:4px;
  white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;max-height:430px;overflow:auto;
  font-size:12.5px;line-height:1.6;color:#c8d1dc}
.tcall{margin:6px 0}
.tname{color:#e08a5f;font-size:11px;margin-bottom:3px}
.empty{color:#5f6874;font-style:italic;font-size:12px;margin:4px 0}
.notice{background:#2a2016;border:1px solid #5a4526;color:#dbb379;padding:7px 11px;border-radius:5px;margin:7px 0;font-size:12px}
.vpanel{border:1px solid #2f3641;border-radius:6px;padding:11px;margin:8px 0;background:#141a22}
.vpanel.mismatch{border-color:#8a3d36;background:#1e1513}
.vpanel .banner{background:#40201c;border-left:3px solid #d9645c;color:#f0b3ac;padding:9px 12px;border-radius:4px;margin-bottom:10px;font-size:12.5px;line-height:1.6}
table.kv{border-collapse:collapse;width:100%;margin-bottom:8px}
table.kv th{text-align:left;color:#7d8794;font-weight:500;padding:3px 10px 3px 0;width:210px;vertical-align:top;font-size:11px}
table.kv td{padding:3px 0}
table.kv code{background:#222a34;padding:1px 6px;border-radius:3px;color:#dfe6ee;font-size:12px;
  word-break:break-word;display:inline-block;max-width:100%}
.ctrls{display:flex;gap:8px;margin:14px 0}
.ctrls button{background:#232a35;color:#c2cbd7;border:1px solid #333c49;border-radius:5px;padding:6px 13px;cursor:pointer;font:inherit;font-size:12px}
.ctrls button:hover{background:#2c3542;color:#fff}
footer{color:#5c6673;font-size:11px;padding:26px 0;text-align:center}
"""

JS = """
function setAll(open){document.querySelectorAll('details.blk').forEach(function(d){d.open=open;});}
document.addEventListener('DOMContentLoaded',function(){
  var e=document.getElementById('expand'),c=document.getElementById('collapse');
  if(e)e.addEventListener('click',function(){setAll(true);});
  if(c)c.addEventListener('click',function(){setAll(false);});
});
"""
