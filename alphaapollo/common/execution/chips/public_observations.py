"""Bounded views of public data; original execution evidence is never rewritten."""

from __future__ import annotations

import hashlib
import json
import math
import re

from .candidate_bundle import MAX_CANDIDATE_BYTES, regular_file

VIEW_VERSION = 1
INLINE_BYTES = 4096

READ_TOOLS = {
    "evas_observe": {
        "artifact_id": {"type": "string"},
        "signals": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
            "uniqueItems": True,
        },
        "start": {"type": "number"},
        "end": {"type": "number"},
        "max_points": {"type": "integer", "minimum": 2, "maximum": 128, "default": 64},
    },
    "evas_read_artifact": {
        "artifact_id": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 2048},
    },
}
READ_DESCRIPTIONS = {
    "evas_observe": (
        "Inspect saved EVAS signals over an inclusive time window; defaults to first 8 signals "
        "and at most 64 points. Uniform sampling can miss events/extrema: use a narrower window "
        "or full artifact for exact measurements. No simulation; consumes one action."
    ),
    "evas_read_artifact": (
        "Read a public JSON artifact in bounded character pages; use next_offset to continue. "
        "No simulation; consumes one action."
    ),
}


def encode(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def reference(artifact_id, content):
    data = content.encode()
    return {"id": artifact_id, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def manifest_view(manifest):
    content = encode(manifest)
    if "transient" not in manifest:
        summary = manifest
    else:
        transient = manifest["transient"]
        times = transient["output_times"]
        summary = {
            "sample_count": len(times),
            "time_range": [times[0], times[-1]] if times else None,
            "stop": transient["stop"],
            "max_step": transient["max_step"],
            "tolerances": manifest["tolerances"],
            "model_count": len(manifest["models"]),
            "instance_count": len(manifest["instances"]),
            "source_count": len(transient["sources"]),
        }
    return {
        "observation_view_version": VIEW_VERSION,
        "manifest": summary,
        "manifest_artifact": reference("manifest", content),
    }


def validate_read(tool, args):
    fields = READ_TOOLS[tool]
    if not isinstance(args, dict) or set(args) - set(fields) or "artifact_id" not in args:
        raise ValueError("invalid observation read arguments")
    if not isinstance(args["artifact_id"], str):
        raise ValueError("artifact_id must be a string")
    for key, value in args.items():
        if key == "artifact_id":
            continue
        rule = fields[key]
        if key == "signals":
            if (
                not isinstance(value, list)
                or not 1 <= len(value) <= 8
                or any(not isinstance(name, str) or not name for name in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError("select 1 to 8 unique signal names")
        elif key in {"start", "end"}:
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("time bounds must be finite numbers")
        elif type(value) is not int or not rule["minimum"] <= value <= rule.get(
            "maximum", 2**63 - 1
        ):
            raise ValueError("invalid observation read limit or offset")
    if "start" in args and "end" in args and args["start"] > args["end"]:
        raise ValueError("start must not exceed end")


def project_observations(action, result, identity, *, waveform):
    """Persist only allowlisted public data, separately from unmodified runner receipts."""
    projected = {**result, "observation_view_version": VIEW_VERSION}
    data = result.get("observations")
    if result.get("execution") != "ok" or data is None:
        return projected
    content = encode(data)
    path = action / "observation.json"
    path.write_bytes(content.encode())
    artifact = {
        **reference("observation:" + action.name, content),
        **identity,
        "format": "evas_waveform" if waveform else "json",
    }
    projected["observation_artifact"] = artifact
    if waveform:
        nodes, rows, times = data["nodes"], data["solutions"], data["transient"]["times"]
        summary = {
            "kind": "waveform_summary",
            "engine": data["engine"],
            "sample_count": len(times),
            "time_range": [times[0], times[-1]] if times else None,
            "node_count": len(nodes),
            "event_count": len(data["transient"]["events"]),
            "signals": [
                {
                    "name": node,
                    "min": min((row["voltages"][index] for row in rows), default=None),
                    "max": max((row["voltages"][index] for row in rows), default=None),
                }
                for index, node in enumerate(nodes[:32])
            ],
            "signals_truncated": len(nodes) > 32,
        }
        if len(encode(summary).encode()) > INLINE_BYTES:
            summary.pop("signals")
            summary.pop("engine")
            summary["signals_truncated"] = True
        projected["observations"] = summary
    elif len(content.encode()) > INLINE_BYTES:
        projected["observations"] = {"kind": "json_artifact", "inline_omitted": True}
    return projected


def _load_artifact(directory, config, artifact_id):
    if artifact_id == "manifest":
        content = encode(config["task"]["manifest"])
        return content, reference("manifest", content)
    match = re.fullmatch(r"observation:([A-Za-z0-9_-]{1,80})", artifact_id)
    if not match or "observations" not in config["task"]["feedback_fields"]:
        raise ValueError("unknown public artifact")
    prefix = "actions/" + match[1] + "/"
    response = json.loads(regular_file(directory, prefix + "response.json").read_text())
    result = response.get("result") if isinstance(response, dict) else None
    artifact = result.get("observation_artifact") if isinstance(result, dict) else None
    if (
        not isinstance(artifact, dict)
        or response.get("ok") is not True
        or result.get("execution") != "ok"
        or artifact.get("id") != artifact_id
        or artifact.get("format") not in ("evas_waveform", "json")
    ):
        raise ValueError("unavailable public artifact")
    path = regular_file(directory, prefix + "observation.json")
    if path.stat().st_size > MAX_CANDIDATE_BYTES:
        raise ValueError("oversized public artifact")
    content = path.read_bytes().decode()
    actual = reference(artifact_id, content)
    if any(artifact.get(key) != value for key, value in actual.items()):
        raise ValueError("public artifact integrity error")
    return content, artifact


def read_artifact(directory, config, args):
    content, artifact = _load_artifact(directory, config, args["artifact_id"])
    offset, limit = args.get("offset", 0), args.get("limit", 2048)
    if offset > len(content):
        raise ValueError("artifact offset exceeds content")
    end = min(len(content), offset + limit)
    return {
        "artifact": artifact,
        "offset": offset,
        "content": content[offset:end],
        "next_offset": end if end < len(content) else None,
        "total_characters": len(content),
        "observation_view_version": VIEW_VERSION,
    }


def observe(directory, config, args):
    content, artifact = _load_artifact(directory, config, args["artifact_id"])
    if artifact.get("format") != "evas_waveform":
        raise ValueError("artifact is not an EVAS waveform")
    data = json.loads(content)
    nodes, times = data["nodes"], data["transient"]["times"]
    signals = args.get("signals", nodes[:8])
    if any(name not in nodes for name in signals):
        raise ValueError("unknown signal")
    columns = [nodes.index(name) for name in signals]
    selected = [
        index
        for index, time in enumerate(times)
        if args.get("start", -math.inf) <= time <= args.get("end", math.inf)
    ]
    count, maximum = len(selected), args.get("max_points", 64)
    if count > maximum:
        selected = [selected[index * (count - 1) // (maximum - 1)] for index in range(maximum)]
    result = {
        "artifact": artifact,
        "signals": signals,
        "available_signal_count": len(nodes),
        "signals_omitted": len(nodes) - len(signals),
        "times": [times[index] for index in selected],
        "values": [
            [data["solutions"][index]["voltages"][column] for column in columns]
            for index in selected
        ],
        "matched_points": count,
        "sampled": count > maximum,
        "sampling": "uniform_index_endpoints" if count > maximum else "none",
        "observation_view_version": VIEW_VERSION,
    }
    if len(encode(result).encode()) > 24576:
        raise ValueError("waveform response too large; select fewer signals or points")
    return result
