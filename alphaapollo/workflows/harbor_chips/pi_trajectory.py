"""Read native Pi v3 text sessions without executing commands or contacting providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harbor.models.trajectories import Trajectory
from jsonschema import Draft202012Validator


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in Pi session")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite value in Pi session")


def _text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(
        not isinstance(part, dict)
        or part.get("type") != "text"
        or not isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError("only text observations are supported")
    return "".join(part["text"] for part in content)


def _redact(value, literals):
    if isinstance(value, str):
        for literal in literals:
            value = value.replace(literal, "[REDACTED]")
    elif isinstance(value, list):
        value = [_redact(item, literals) for item in value]
    elif isinstance(value, dict):
        if any(literal in key for key in value for literal in literals):
            raise ValueError("redaction literal occurs in an object key; sanitize the source copy")
        value = {key: _redact(item, literals) for key, item in value.items()}
    return value


def read_pi_trajectory(path, *, agent_version, requested_thinking=None, redact=()):
    """Convert a linear Pi v3 session to ATIF, retaining native observation fidelity.

    System sections are joined with blank lines. This is an explicit reconstruction,
    not a claim that the provider received these exact bytes. Literal redaction is
    opt-in; output remains private and is never certified as safe to publish.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("session must be a regular file")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("session exceeds 32 MiB supported size")
    raw = path.read_bytes()
    entries = [
        (line, json.loads(value, object_pairs_hook=_unique, parse_constant=_nonfinite))
        for line, value in enumerate(raw.splitlines(), 1)
        if value.strip()
    ]
    records = [value for _, value in entries]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Pi records must be objects")
    if not records or records[0].get("type") != "session" or records[0].get("version") != 3:
        raise ValueError("expected a Pi v3 session")
    if not agent_version or any(not isinstance(x, str) or not x for x in redact):
        raise ValueError("agent version and redaction literals must be nonempty")
    steps, definitions, model, thinking = [], [], None, None
    pending, seen = {}, set()
    entry_ids, parent, tool_names = set(), None, set()
    for line, record in entries[1:]:
        entry_id = record.get("id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in entry_ids:
            raise ValueError("missing or duplicate Pi entry ID")
        if "parentId" not in record or record["parentId"] != parent:
            raise ValueError("branched or incomplete Pi history is unsupported")
        entry_ids.add(entry_id)
        parent = entry_id
        kind = record["type"]
        if kind == "model_change":
            if steps:
                raise ValueError("mid-session model changes are unsupported")
            model = record["modelId"]
            continue
        if kind == "thinking_level_change":
            if steps:
                raise ValueError("mid-session thinking changes are unsupported")
            thinking = record["thinkingLevel"]
            continue
        if kind != "message":
            raise ValueError(f"unsupported Pi record at line {line}")
        message = record["message"]
        role = message["role"]
        if message.get("toolsRemoved") or role != "system" and message.get("toolsAdded"):
            raise ValueError("mid-session tool definition changes are unsupported")
        content = message.get("content", [])
        if role == "toolResult":
            call_id = message["toolCallId"]
            if call_id not in pending or message["toolName"] != pending[call_id]:
                raise ValueError("unmatched tool result")
            del pending[call_id]
            details = message.get("details") or {}
            truncation = details.get("truncation") or {}
            # Never substitute fullOutputPath or the backend's full response.
            steps[-1].setdefault("observation", {"results": []})["results"].append(
                {
                    "source_call_id": call_id,
                    "content": _text(content),
                    "extra": {
                        "native_line": line,
                        "timestamp": record.get("timestamp"),
                        "is_error": message.get("isError"),
                        "truncation": {k: v for k, v in truncation.items() if k != "content"},
                    },
                }
            )
            continue
        if pending:
            raise ValueError("missing tool result before next message")
        step = {
            "step_id": len(steps) + 1,
            "timestamp": record.get("timestamp"),
            "source": "agent" if role == "assistant" else role,
            "message": "",
            "extra": {"native_line": line},
        }
        if role == "system":
            if steps:
                raise ValueError("only initial system context is supported")
            sections = message.get("sections") or {}
            if not isinstance(sections, dict) or any(
                not isinstance(v, str) for v in sections.values()
            ):
                raise ValueError("system sections must contain text")
            if sections and _text(content):
                raise ValueError("ambiguous Pi system content and sections")
            step["message"] = "\n\n".join(sections.values()) if sections else _text(content)
            step["extra"]["native_sections"] = sections
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in message.get("toolsAdded", [])
            ]
            for definition in definitions:
                function = definition["function"]
                if function["name"] in tool_names:
                    raise ValueError("duplicate tool definition")
                tool_names.add(function["name"])
                Draft202012Validator.check_schema(function["parameters"])
        elif role == "user":
            step["message"] = _text(content)
        elif role == "assistant":
            if message.get("model", model) != model:
                raise ValueError("assistant model differs from recorded model")
            if not isinstance(content, list):
                raise ValueError("expected Pi assistant content blocks")
            texts, thoughts, calls = [], [], []
            for block in content:
                if block["type"] == "text":
                    texts.append(block["text"])
                elif block["type"] == "thinking":
                    thoughts.append(block["thinking"])
                elif block["type"] == "toolCall":
                    if block["name"] not in tool_names:
                        raise ValueError("tool call lacks a native tool definition")
                    call_id = block["id"]
                    if call_id in seen:
                        raise ValueError("duplicate tool call ID")
                    seen.add(call_id)
                    pending[call_id] = block["name"]
                    calls.append(
                        {
                            "tool_call_id": call_id,
                            "function_name": block["name"],
                            "arguments": block["arguments"],
                        }
                    )
                else:
                    raise ValueError("unsupported assistant content block")
            step.update(
                message="".join(texts),
                model_name=message.get("model", model),
                reasoning_effort=thinking,
                reasoning_content="".join(thoughts) or None,
                tool_calls=calls or None,
            )
            step["extra"].update(
                stop_reason=message.get("stopReason"),
                provider=message.get("provider"),
                response_id=message.get("responseId"),
            )
            usage = message.get("usage")
            if usage:
                cached = usage.get("cacheRead", 0)
                cost = (usage.get("cost") or {}).get("total")
                step["metrics"] = {
                    "prompt_tokens": (
                        usage["input"] + cached + usage.get("cacheWrite", 0)
                        if "input" in usage
                        else None
                    ),
                    "completion_tokens": usage.get("output"),
                    "cached_tokens": cached,
                    "cost_usd": cost if cost and cost > 0 else None,
                    "extra": {"native_usage": usage},
                }
        else:
            raise ValueError("unsupported Pi message role")
        steps.append(step)
    if pending:
        raise ValueError("missing tool results at end of session")
    if not steps or steps[0]["source"] != "system" or not definitions or not model:
        raise ValueError("missing initial system, tool definitions or model")
    if len(steps) < 3 or steps[1]["source"] != "user":
        raise ValueError("missing initial task message before assistant work")
    if steps[-1]["source"] != "agent" or steps[-1]["extra"].get("stop_reason") != "stop":
        raise ValueError("session has no completed assistant response")
    result = {
        "schema_version": "ATIF-v1.8",
        "session_id": records[0]["id"],
        "agent": {
            "name": "pi",
            "version": agent_version,
            "model_name": model,
            "tool_definitions": definitions,
            "extra": {"requested_thinking": requested_thinking, "recorded_thinking": thinking},
        },
        "steps": steps,
        "extra": {
            "source_format": "pi-session-v3",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "context_fidelity": "native_reconstruction",
            "provider_requests": "unavailable",
            "token_ids": "unavailable",
            "sampling_logprobs": "unavailable",
            "redaction": {"literal_count": len(redact), "publication_safe": False},
        },
    }
    return Trajectory.model_validate(_redact(result, sorted(redact, key=len, reverse=True)))
