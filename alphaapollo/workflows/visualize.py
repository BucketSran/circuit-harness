#!/usr/bin/env python3
# ruff: noqa: E501 - embedded HTML/CSS stays readable and byte-stable as complete rules
"""Render AlphaApollo run directories into one self-contained HTML trajectory viewer.

The page shows the *complete* interaction for every trajectory: system prompt,
user prompt, the tools offered, tool_choice, assistant output, tool calls and
tool results, plus the scored outcome.

Usage:
    python -m alphaapollo.workflows.visualize RUN_DIR [RUN_DIR ...] -o OUT.html

A "run dir" is any directory containing cells of the shape
``<mode>/<problem>/sample-NNN/`` with ``canonical/trajectories.jsonl`` inside.
Missing or malformed files degrade to a visible notice instead of crashing.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

TASK_ID_RE = re.compile(
    r"^(?P<input>[^:]+):branch-(?P<branch>\d+):(?P<step>[^:]+):iteration-(?P<iter>\d+)$"
)


def read_json(path: Path) -> tuple[Any, str | None]:
    """Return (data, error). Never raises."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"missing file: {path.name}"
    except Exception as exc:  # noqa: BLE001 - a bad cell must not kill the report
        return None, f"{type(exc).__name__}: {exc}"


def read_jsonl(path: Path) -> tuple[list[Any], str | None]:
    """Return (records, error). Skips unparsable lines but reports the count."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], f"missing file: {path.name}"
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    out, bad = [], 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            bad += 1
    return out, (f"{bad} unparsable line(s) skipped" if bad else None)


def parse_task_id(task_id: str) -> tuple[str, int, int]:
    """-> (step, iteration, branch). Falls back gracefully on odd ids."""
    m = TASK_ID_RE.match(task_id or "")
    if not m:
        return (task_id or "step", 1, 0)
    return (m.group("step"), int(m.group("iter")), int(m.group("branch")))


def find_cells(run_dir: Path) -> list[Path]:
    """Every directory holding canonical/trajectories.jsonl, sorted by path."""
    if not run_dir.exists():
        return []
    seen = {p.parent.parent for p in run_dir.rglob("canonical/trajectories.jsonl")}
    # also surface cells that exist but have no trajectories yet (in-flight runs)
    for res in run_dir.rglob("result.json"):
        seen.add(res.parent)
    for art in run_dir.rglob("sample-*/artifacts"):
        seen.add(art.parent)
    return sorted(seen)


# --------------------------------------------------------------------------
# verifier verdict analysis  (the headline finding)
# --------------------------------------------------------------------------


def analyse_verifier_text(text: str) -> dict[str, Any]:
    """Try to parse a verifier payload the way the runtime would.

    Returns the claimed verdict (if the text *looks* like it says one) and the
    JSON error that caused it to be discarded. This is what makes the
    "valid-looking JSON, unparsable due to raw LaTeX backslashes" failure
    visible instead of silently becoming `inconclusive`.
    """
    info: dict[str, Any] = {"claimed_verdict": None, "parse_error": None, "parsed_ok": False}
    if not text:
        return info
    m = re.search(r"\{.*\}", text, re.S)
    candidate = m.group(0) if m else text
    try:
        data = json.loads(candidate)
        info["parsed_ok"] = True
        if isinstance(data, dict):
            info["claimed_verdict"] = data.get("verdict")
        return info
    except Exception as exc:  # noqa: BLE001
        info["parse_error"] = f"{type(exc).__name__}: {exc}"
    # The text did not parse. Recover what verdict it *claimed* via regex so the
    # discrepancy against the recorded verdict is explicit on the page.
    claim = re.search(r'"verdict"\s*:\s*"([^"]+)"', candidate)
    if claim:
        info["claimed_verdict"] = claim.group(1)
    # point at the offending escape when it is the classic LaTeX-backslash case
    bad = re.search(r'\\(?![\\/"bfnrtu])(.)', candidate)
    if bad:
        info["bad_escape"] = "\\" + bad.group(1)
    return info


# --------------------------------------------------------------------------
# html helpers
# --------------------------------------------------------------------------


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


# Special-token modes emit tool use inline as tags inside the assistant text,
# rather than as a structured `tool_calls` array. Both shapes must render.
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


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_tools(tools: Any, tool_choice: Any, tool_calls: Any) -> str:
    """Always render, even with zero calls — absence of tool use is a finding."""
    names = []
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function") if isinstance(t.get("function"), dict) else None
                names.append((fn or t).get("name") or "?")
    n_calls = len(tool_calls) if isinstance(tool_calls, list) else "unavailable"
    offered = ", ".join(names) if names else ("none" if isinstance(tools, list) else "unavailable")
    cls = "toolbar" + (" nocalls" if names and n_calls == 0 else "")
    note = ""
    if names and n_calls == 0:
        note = '<span class="flag">offered but never called</span>'
    return (
        f'<div class="{cls}">'
        f'<span class="k">tools offered:</span> <code>{esc(offered)}</code>'
        f'<span class="k">tool_choice:</span> <code>{esc(tool_choice if tool_choice is not None else "n/a")}</code>'
        f'<span class="k">calls:</span> <code>{n_calls}</code>{note}'
        f"</div>"
    )


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


def render_turn(turn: dict, prev_msg_count: int) -> tuple[str, int]:
    req = turn.get("generation_request") or {}
    resp = turn.get("generation_response") or {}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    idx = turn.get("index", 0)

    parts = [f'<div class="turn"><div class="turnhdr">turn {esc(idx)}</div>']
    parts.append(render_tools(req.get("tools"), req.get("tool_choice"), resp.get("tool_calls")))
    parts.append(render_messages(messages, prev_msg_count))

    text = content_to_text(resp.get("content"))
    reasoning = content_to_text(resp.get("reasoning_content"))
    finish = resp.get("finish_reason")
    if reasoning:
        parts.append(block("think", "reasoning", pre(reasoning), open_by_default=False))
    parts.append(
        block(
            "asst",
            "assistant",
            render_inline_text(text),
            meta=f"finish_reason={finish}" if finish else "",
        )
    )
    parts.append(render_tool_calls(resp.get("tool_calls")))
    parts.append("</div>")
    return "".join(parts), len(messages)


def render_verifier_panel(traj: dict, wf_output: dict | None, wf_step: dict | None) -> str:
    """Raw emitted JSON vs the verdict that was actually recorded."""
    raw = traj.get("final_text") or ""
    info = analyse_verifier_text(raw)
    recorded = (wf_output or {}).get("verdict")
    details = (wf_step or {}).get("details") if isinstance(wf_step, dict) else None
    recorded_err = None
    if isinstance(details, dict):
        recorded_err = details.get("parse_error")
    parse_error = recorded_err or info.get("parse_error")

    rows = [
        ("raw verdict claimed in text", info.get("claimed_verdict") or "(none found)"),
        ("verdict recorded by runtime", recorded if recorded is not None else "(none)"),
        ("raw JSON parses?", "yes" if info["parsed_ok"] else "NO"),
    ]
    if info.get("bad_escape"):
        rows.append(("first invalid escape", info["bad_escape"]))
    if parse_error:
        rows.append(("parse_error", parse_error))

    mismatch = (
        info.get("claimed_verdict")
        and recorded
        and str(info["claimed_verdict"]).lower() != str(recorded).lower()
    )
    cls = "vpanel mismatch" if mismatch else "vpanel"
    banner = ""
    if mismatch:
        banner = (
            '<div class="banner">Verdict discarded: the verifier emitted valid-looking JSON claiming '
            f"<b>{esc(info['claimed_verdict'])}</b>, but it failed to parse, so the runtime recorded "
            f"<b>{esc(recorded)}</b>. Unescaped LaTeX backslashes are invalid JSON escapes.</div>"
        )
    table = "".join(f"<tr><th>{esc(k)}</th><td><code>{esc(v)}</code></td></tr>" for k, v in rows)
    return (
        f'<div class="{cls}">{banner}<table class="kv">{table}</table>'
        + block("raw", "raw verifier output (verbatim)", pre(raw), open_by_default=bool(mismatch))
        + "</div>"
    )


def render_trajectory(traj: dict, wf_output: dict | None, wf_step: dict | None) -> str:
    task_id = traj.get("task_id", "")
    step, iteration, _branch = parse_task_id(task_id)
    turns = traj.get("turns") if isinstance(traj.get("turns"), list) else []

    head = (
        f'<div class="stephdr"><span class="pill s-{esc(step)}">{esc(step)}</span>'
        f'<span class="iter">iteration {esc(iteration)}</span>'
        f'<span class="tid">{esc(task_id)}</span>'
        f'<span class="tn">{len(turns)} turn(s)</span>'
        f'<span class="tn">termination: {esc(traj.get("termination_reason"))}</span></div>'
    )

    body = []
    if step == "verify":
        body.append(render_verifier_panel(traj, wf_output, wf_step))
    prev = 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        rendered, prev = render_turn(turn, prev)
        body.append(rendered)
    if not turns:
        body.append('<p class="empty">(no turns recorded)</p>')

    final_text = traj.get("final_text") or ""
    if final_text and step != "verify":
        body.append(
            block("final", "final_text for this step", pre(final_text), open_by_default=False)
        )

    return f'<section class="step">{head}<div class="stepbody">{"".join(body)}</div></section>'


def render_cell(cell: Path, root: Path) -> tuple[str, dict]:
    rel = cell.relative_to(root.parent) if root.parent in cell.parents else cell
    trajs, terr = read_jsonl(cell / "canonical" / "trajectories.jsonl")
    result, rerr = read_json(cell / "result.json")
    wf, werr = read_jsonl(cell / "canonical" / "workflow_results.jsonl")

    # index workflow step outputs by (step_id, iteration)
    step_out: dict[tuple[str, int], dict] = {}
    step_raw: dict[tuple[str, int], dict] = {}
    for rec in wf:
        for s in (rec.get("steps") or []) if isinstance(rec, dict) else []:
            if isinstance(s, dict):
                key = (s.get("step_id"), int(s.get("iteration") or 1))
                step_out[key] = s.get("output") or {}
                step_raw[key] = s

    parts = list(cell.parts)
    mode = parts[-3] if len(parts) >= 3 else "?"
    problem = parts[-2] if len(parts) >= 2 else "?"
    cid = f"{mode}--{problem}".replace("/", "-")

    result = result or {}
    ok = result.get("correct")
    status = "ok" if ok else ("bad" if ok is False else "unk")

    # summary header
    def kv(k, v):
        return f'<div class="cellkv"><span>{esc(k)}</span><b>{esc(v)}</b></div>'

    summary = "".join(
        [
            kv("final answer", result.get("final_answer", "—")),
            kv("gold answer", result.get("gold_answer", "—")),
            kv("correct", result.get("correct", "—")),
            kv("verdict", result.get("verdict", "—")),
            kv("rounds", result.get("rounds", "—")),
            kv("solver tool calls", result.get("solver_tool_calls", "—")),
            kv("prompt tokens", result.get("prompt_tokens", "—")),
            kv("completion tokens", result.get("completion_tokens", "—")),
            kv(
                "wall seconds",
                round(result["wall_seconds"], 1)
                if isinstance(result.get("wall_seconds"), (int, float))
                else "—",
            ),
            kv("termination", result.get("termination_reason", "—")),
        ]
    )

    notices = []
    for err in (terr, rerr, werr):
        if err:
            notices.append(f'<div class="notice">{esc(err)}</div>')
    if not trajs:
        notices.append(
            '<div class="notice">No trajectories on disk for this cell — the run may still be in flight.</div>'
        )

    # step-sequence strip (handles repeats: revise/verify can appear N times)
    seq = []
    for t in trajs:
        s, it, _ = parse_task_id(t.get("task_id", ""))
        seq.append(f'<span class="pill s-{esc(s)}">{esc(s)}<sub>{esc(it)}</sub></span>')
    seq_html = f'<div class="seq">{"&rarr;".join(seq)}</div>' if seq else ""

    steps_html = "".join(
        render_trajectory(
            t,
            step_out.get(parse_task_id(t.get("task_id", ""))[0:2]),
            step_raw.get(parse_task_id(t.get("task_id", ""))[0:2]),
        )
        for t in trajs
    )

    stats = {
        "mode": mode,
        "problem": problem,
        "cid": cid,
        "correct": ok,
        "solver_tool_calls": result.get("solver_tool_calls"),
        "verdict": result.get("verdict"),
        "rounds": result.get("rounds"),
        "n_steps": len(trajs),
    }

    html_out = (
        f'<article class="cell" id="{esc(cid)}" data-mode="{esc(mode)}">'
        f'<h2><span class="dot {status}"></span>{esc(problem)} '
        f'<span class="mode">{esc(mode)}</span></h2>'
        f'<div class="path">{esc(str(rel))}</div>'
        f"{''.join(notices)}"
        f'<div class="cellsum">{summary}</div>'
        f"{seq_html}"
        f"{steps_html}"
        f"</article>"
    )
    return html_out, stats


# --------------------------------------------------------------------------
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


def build_page(cells_html: list[str], stats: list[dict], sources: list[str]) -> str:
    toc = "".join(
        f'<a href="#{esc(s["cid"])}">{esc(s["problem"])} &middot; {esc(s["mode"])}</a>'
        for s in stats
    )
    n_zero = [s for s in stats if s.get("solver_tool_calls") == 0]
    n_incon = [s for s in stats if str(s.get("verdict")) == "inconclusive"]

    findings = f"""
<div class="findings">
<h3>Recorded outcomes</h3>
<ul>
<li>{len(n_zero)} of {len(stats)} runs record <code>solver_tool_calls = 0</code>.
Missing counts are not counted as zero.</li>
<li>{len(n_incon)} runs record <code>inconclusive</code>. This status alone does not
identify the cause; inspect the recorded verifier output and errors below.</li>
</ul>
</div>"""

    src = "".join(f"<li><code>{esc(s)}</code></li>" for s in sources)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaApollo &mdash; Complete Interaction Trajectories</title>
<style>{CSS}</style></head><body>
<header class="top">
<h1>AlphaApollo &mdash; Complete Interaction Trajectories</h1>
<div class="sub">system prompt &rarr; user prompt &rarr; tool call &rarr; tool response, for every step of every run &middot; generated {esc(ts)}</div>
</header>
<div class="wrap">
{findings}
<div class="ctrls"><button id="expand" type="button">Expand all</button><button id="collapse" type="button">Collapse all</button></div>
<nav class="toc">{toc}</nav>
{"".join(cells_html)}
<footer>Sources:<ul style="list-style:none;padding:0">{src}</ul>
Self-contained: no external CSS, JS, fonts or network requests.</footer>
</div>
<script>{JS}</script>
</body></html>"""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", type=Path, help="AlphaApollo run directories")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output HTML file")
    args = ap.parse_args(argv)

    cells_html: list[str] = []
    stats: list[dict] = []
    sources: list[str] = []

    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser()
        cells = find_cells(run_dir)
        sources.append(f"{run_dir} ({len(cells)} cell(s))")
        if not cells:
            print(f"warning: no cells found under {run_dir}", file=sys.stderr)
            continue
        for cell in cells:
            try:
                chtml, cstats = render_cell(cell, run_dir)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the page
                print(f"warning: failed to render {cell}: {exc}", file=sys.stderr)
                chtml = f'<article class="cell"><h2>{esc(cell.name)}</h2><div class="notice">render failed: {esc(str(exc))}</div></article>'
                cstats = {"mode": "?", "problem": cell.name, "cid": esc(cell.name)}
            cells_html.append(chtml)
            stats.append(cstats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(cells_html, stats, sources), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes, {len(stats)} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
