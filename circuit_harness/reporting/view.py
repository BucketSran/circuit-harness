"""Render Chips evidence with the shared trajectory viewer's escaped HTML blocks."""

from __future__ import annotations

import json

from circuit_harness.reporting.html import (
    CSS,
    JS,
    block,
    esc,
    pre,
    render_inline_text,
    render_messages,
    render_tool_calls,
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _display(value):
    if value is None:
        return "unavailable"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _timeline_table(report):
    rows = []
    for item in report["timeline"]:
        anchor = f"{item['kind']}-{item['id']}"
        values = [
            item["kind"],
            item["name"],
            item["elapsed_seconds"],
            item.get("server_action_seconds"),
            item.get("state") or item.get("stop_reason"),
        ]
        cells = "".join(f"<td>{esc(_display(value))}</td>" for value in values)
        rows.append(f'<tr>{cells}<td><a href="#{esc(anchor)}">details</a></td></tr>')
    return (
        '<h2>Timeline summary</h2><table class="episode-table"><thead><tr>'
        "<th>Kind</th><th>Action</th><th>Controller seconds</th><th>Server action seconds</th>"
        "<th>State / stop reason</th><th>Evidence</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><p>Intervals may overlap. Server action time includes "
        "its wrapper; missing values are not zero. The final evaluator is listed separately.</p>"
    )


def render_html(report: dict) -> str:
    parts = [
        "<h1>Chips Episode Report</h1>",
        "<p>Offline view of recorded evidence; not a new evaluation.</p>",
    ]
    for label, value in (
        ("Outcome", report["outcome"]),
        ("Saved experiment settings", report["experiment_settings"]),
        ("Metrics", report["metrics"]),
        ("Usage", report["usage"]),
        ("Evidence gaps", report["evidence_gaps"]),
        ("Per-request budget notices", report.get("budget_context", [])),
    ):
        parts.append(block("raw", label, pre(_json(value))))
    parts.append(_timeline_table(report))
    task = report["agent_input"]
    if task:
        parts.append(
            render_messages(
                [
                    {"role": "system", "content": task.get("system")},
                    {"role": "user", "content": task.get("prompt")},
                ],
                0,
            )
        )
        parts.append(
            block(
                "raw",
                "Harness input/tool snapshot (CLI may add context)",
                pre(_json(task)),
                open_by_default=False,
            )
        )
    parts.append(
        "<h2>Recorded agent events</h2><p>Event order is preserved. "
        "Only emitted text is visible; missing prompts, reasoning and timestamps "
        "are not inferred.</p>"
    )
    for index, event in enumerate(report["agent_events"]):
        body = render_inline_text(event.get("content") or "")
        if event.get("tool_call"):
            body += render_tool_calls([event["tool_call"]])
        for action in report["actions"]:
            if index + 1 in action["agent_event_indices"]:
                body += f'<a href="#tool-{esc(action["id"])}">Linked tool evidence</a>'
        parts.append(
            block(
                "asst",
                f"{index + 1}. {event.get('kind', 'unknown')}",
                body,
                meta=event.get("call_id") or "",
                open_by_default=False,
            )
        )
    if report.get("runtime_turns"):
        parts.append("<h2>Recorded Runtime turns</h2>")
        for turn in report["runtime_turns"]:
            parts.append(
                block(
                    "raw",
                    f"Runtime turn {turn.get('index')}",
                    pre(_json(turn)),
                    open_by_default=False,
                )
            )
    parts.append("<h2>Model and tool timeline</h2>")
    for row in report["timeline"]:
        parts.append(
            f'<section id="{esc(row["kind"])}-{esc(row["id"])}">'
            + block("tres", f"{row['name']} · {row['id']}", pre(_json(row)), open_by_default=False)
            + "</section>"
        )
    parts.append(
        block(
            "raw",
            "Candidate integrity and repair sequences",
            pre(
                _json(
                    {
                        "integrity": report["candidate_integrity"],
                        "repairs": report["repairs"],
                    }
                )
            ),
        )
    )
    for version in report["candidate_changes"]:
        parts.append(
            block(
                "raw",
                f"Candidate change · {version['action_id']}",
                pre(
                    version["diff"]
                    if version["diff"] is not None
                    else "Recorded restore; content unavailable for diff."
                ),
                meta=version["sha256"],
                open_by_default=False,
            )
        )
    parts.append(
        block(
            "raw",
            "Recorded final report",
            pre(
                _json(
                    {
                        "agent_report": report["recorded_report"],
                        "final_result": report["final_result"],
                        "archives": report["archives"],
                    }
                )
            ),
            open_by_default=False,
        )
    )
    parts.append(
        block("raw", "Source fingerprints", pre(_json(report["sources"])), open_by_default=False)
    )
    body = "\n".join(parts)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chips Episode Report · {esc(report["task_id"])}</title><style>{CSS}
.episode-table {{width:100%;border-collapse:collapse;font-size:12px;}}
.episode-table th,.episode-table td {{padding:8px;border-bottom:1px solid #303744;text-align:left;}}
a {{color:#8fb4e0;}} section:target {{outline:1px solid #8fb4e0;}}
</style></head>
<body><div class="wrap"><div class="ctrls"><button id="expand">Expand all</button>
<button id="collapse">Collapse all</button></div>{body}</div><script>{JS}</script></body></html>"""


def render_markdown(report: dict) -> str:
    lines = [
        "# Chips Episode Report",
        "",
        "Offline projection of saved evidence; not a new evaluation.",
        "",
    ]
    for label, key in (
        ("Outcome", "outcome"),
        ("Metrics", "metrics"),
        ("Usage", "usage"),
        ("Evidence gaps", "evidence_gaps"),
    ):
        lines.extend([f"## {label}", "", "```json", _json(report[key]), "```", ""])
    lines.extend(
        [
            "See report.html for inputs/events/results and timeline.csv for timings.",
            "Fingerprints: episode-report.json. Outputs contain private evidence.",
            "",
        ]
    )
    return "\n".join(lines)
