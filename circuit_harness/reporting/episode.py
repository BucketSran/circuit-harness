"""Offline projection of saved Chips evidence, never an experiment runner.

Chips external-agent logs are not the canonical ``traj.jsonl`` event contract.
Keep their measurement rules here; reuse the shared viewer's rendering primitives
without inventing native Workflow turns or feeding foreign events to its reducer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

from circuit_harness.reporting.evidence import link_actions, read_final_evidence


class Evidence:
    """Read selected regular evidence files and fingerprint the exact bytes used."""

    def __init__(self):
        self.sources = {}
        self.gaps = []

    def read(self, path: Path, *, lines=False):
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"not a bounded regular evidence file: {path}")
        raw = path.read_bytes()
        self.sources[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        try:
            data = (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if lines
                else json.loads(raw)
            )
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"invalid JSON evidence: {path}") from exc
        records = data if lines else [data]
        if any(not isinstance(row, dict) for row in records):
            raise ValueError(f"expected JSON objects: {path}")
        return data

    def fingerprint(self, path: Path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"not a regular evidence file: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        value = {"sha256": digest, "bytes": path.stat().st_size}
        self.sources[str(path)] = value
        return value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _elapsed(start, end):
    return end - start if _number(start) and _number(end) and end >= start else None


def _output_budget(request, usage):
    output, reasoning = (usage or {}).get("output"), (usage or {}).get("reasoning")
    valid = type(output) is int and type(reasoning) is int and 0 <= reasoning <= output
    return {
        "configured_tokens": request.get("maxTokens"),
        "reported_reasoning_output_fraction": reasoning / output if valid and output > 0 else None,
        "reported_output_minus_reasoning_tokens": output - reasoning if valid else None,
        "scope": "Pi reported output detail; not thinking time or visible text length.",
    }


def _actions(evidence, agent):
    rows = []
    directory = agent / "tools"
    if not directory.is_dir():
        evidence.gaps.append("Tool action directory unavailable; count is not known to be zero.")
        return rows, None
    for path in sorted(directory.glob("*/request.json")):
        request = evidence.read(path)
        if (
            request.get("id") != path.parent.name
            or not isinstance(request.get("arguments"), dict)
            or not isinstance(request.get("tool"), str)
        ):
            raise ValueError(f"invalid tool action envelope: {path}")
        reply = evidence.read(path.with_name("response.json"))
        transport = evidence.read(path.with_name("transport.json"))
        timing = evidence.read(path.with_name("timing.json")) or {}
        result = (reply or {}).get("result", {})
        if not isinstance(result, dict):
            raise ValueError(f"invalid tool result: {path}")
        state = result.get("status", result.get("state"))
        failed = (
            (
                reply.get("ok") is False
                or state
                in {
                    "failed",
                    "error",
                    "timeout",
                    "invalid",
                    "simulation_error",
                    "infrastructure_error",
                    "cleanup_failed",
                }
            )
            if reply
            else None
        )
        start, end = timing.get("started_at"), timing.get("finished_at")
        rows.append(
            {
                "kind": "tool",
                "id": request["id"],
                "name": request.get("tool"),
                "started_at": start,
                "finished_at": end,
                "elapsed_seconds": _elapsed(start, end),
                "request": request,
                "response": reply,
                "transport": transport,
                "failed": failed,
                "state": state,
                "source": str(path),
            }
        )
        if reply is None or _elapsed(start, end) is None:
            evidence.gaps.append(f"Action {request['id']}: response or timing unavailable.")
    rows.sort(
        key=lambda row: (row["started_at"] if _number(row["started_at"]) else math.inf, row["id"])
    )
    return rows, len(rows)


def _model_requests(evidence, agent):
    path = agent / "model-budget.jsonl"
    records = evidence.read(path, lines=True)
    requests, stops, responses = [], [], []
    pending = None
    for index, event in enumerate(records or []):
        if event.get("event") == "request":
            pending = {
                "kind": "model",
                "id": event.get("call"),
                "name": "provider_request",
                "started_at": event.get("at"),
                "finished_at": None,
                "elapsed_seconds": None,
                "request": event,
                "stop_reason": None,
                "usage": None,
                "output_budget": _output_budget(event, None),
                "source": f"{path}:{index + 1}",
            }
            requests.append(pending)
        elif event.get("event") == "response":
            responses.append(event)
            if pending is None:
                evidence.gaps.append(f"Unmatched model response: {path}:{index + 1}")
                continue
            pending.update(
                finished_at=event.get("at"),
                usage=event.get("usage"),
                output_budget=_output_budget(pending["request"], event.get("usage")),
                stop_reason=event.get("stopReason"),
                elapsed_seconds=_elapsed(pending["started_at"], event.get("at")),
            )
            pending = None
        elif event.get("event") == "budget_stop":
            stops.append(event)
        elif event.get("event") != "budget_context":
            evidence.gaps.append(f"Unknown model-budget event: {path}:{index + 1}")
    fields = ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")
    totals = dict.fromkeys(fields)
    coverage = dict.fromkeys(fields, 0)
    for event in responses:
        usage = event.get("usage") or {}
        for key in fields:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key] = (totals[key] or 0) + value
                coverage[key] += 1
    if records is None:
        evidence.gaps.append(
            "Per-request model timing/usage unavailable; outcome usage is not a known total."
        )
    if any(row["elapsed_seconds"] is None for row in requests):
        evidence.gaps.append("Some model requests lack a matched response or valid timing.")
    return (
        requests,
        stops,
        {
            "totals": totals if responses else None,
            "coverage": coverage,
            "recorded_responses": len(responses),
            "scope": "sum_of_recorded_responses" if responses else "unavailable",
            "note": (
                "Missing fields are null; partial sums are lower bounds. "
                "Reasoning is an output detail, not added again. No billing amount is inferred."
            ),
        },
    )


def _native_requests(evidence, agent):
    """Read the native probe's HTTP boundary; its clock is relative, not epoch time."""
    requests, stops = [], []
    responses = 0
    for path in sorted((agent / "http").glob("*-request.json")):
        request = evidence.read(path)
        if request.get("state") == "rejected_before_send":
            stops.append(request)
            continue
        if request.get("state") != "dispatch_started":
            raise ValueError(f"unknown native request state: {path}")
        response = evidence.read(path.with_name(path.name.replace("-request", "-response")))
        if response and response.get("attempt") != request.get("attempt"):
            raise ValueError(f"native request/response identity mismatch: {path}")
        body = {}
        if response:
            responses += 1
            try:
                body = json.loads(response["body"])
            except (KeyError, TypeError, ValueError):
                evidence.gaps.append(f"Non-JSON HTTP response preserved: {path}")
            if not isinstance(body, dict):
                body = {}
        choices = body.get("choices") or [{}]
        requests.append(
            {
                "kind": "model",
                "id": request.get("attempt"),
                "name": "provider_request",
                "started_at": None,
                "finished_at": None,
                "elapsed_seconds": _elapsed(
                    request.get("elapsed_s"), (response or {}).get("elapsed_s")
                ),
                "request": request,
                "response": response,
                "stop_reason": choices[0].get("finish_reason"),
                "usage": body.get("usage"),
                "source": str(path),
            }
        )
    evidence.gaps.append(
        "Native HTTP offsets share a monotonic origin; no absolute clock anchor was recorded. "
        "Durations are available, but model/tool timeline interleaving is not inferred."
    )
    return (
        requests,
        stops,
        {
            "totals": None,
            "coverage": {},
            "recorded_responses": responses,
            "scope": "per_response_only",
            "note": (
                "Native provider usage is preserved on each request, "
                "without Pi token-field conversion."
            ),
        },
    )


def build_report(source: Path) -> dict:
    """Project one agent evidence directory or a unified experiment's agent child."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("source must be an existing evidence directory")
    agent = source / "agent" if (source / "agent").is_dir() else source
    evidence = Evidence()
    manifest = evidence.read(source / "experiment_manifest.json") or {}
    rows = evidence.read(source / "results.jsonl", lines=True) or []
    if len(rows) > 1:
        raise ValueError("select one episode; multiple result rows are not supported")
    row = rows[0] if rows else {}
    resolved = evidence.read(agent / "resolved_config.json") or {}
    operator = resolved.get("operator") or evidence.read(agent / "operator.json") or {}
    recorded_settings = {
        name: operator.get(name)
        for name in (
            "model",
            "provider",
            "transport",
            "policy_kind",
            "thinking",
            "reasoning_effort",
            "max_model_calls",
            "max_request_bytes",
            "max_output_tokens",
            "episode_timeout_s",
        )
    }
    settings = manifest.get("experiment_settings") or recorded_settings
    report = evidence.read(agent / "report.json") or {}
    outcomes = [
        (kind, evidence.read(agent / f"{kind}-outcome.json")) for kind in ("pi", "codex", "native")
    ]
    present = [(kind, outcome) for kind, outcome in outcomes if outcome is not None]
    if len(present) > 1:
        raise ValueError("ambiguous episode: multiple agent outcomes are present")
    kind, outcome = present[0] if present else (None, {})
    if not report and not outcome and not row and not (agent / "tools").is_dir():
        raise ValueError("no supported Chips episode evidence found")
    agent_input = evidence.read(agent / "agent-input.json")
    if agent_input is None:
        evidence.gaps.append(
            "Agent input snapshot unavailable; current prompts are not substituted."
        )
    evidence.gaps.append(
        "Tool intervals include overhead; SSH and solver time are not separately measured."
    )
    actions, count = _actions(evidence, agent)
    requests, stops, usage = (_native_requests if kind == "native" else _model_requests)(
        evidence, agent
    )
    captured = evidence.read(agent / "pi-wire-requests.jsonl", lines=True)
    if captured is not None:
        by_id = {request["id"]: request for request in requests}
        seen = set()
        for payload in captured:
            ident = payload.get("attempt")
            if ident in seen or ident not in by_id:
                raise ValueError("captured provider request has no unique budget record")
            seen.add(ident)
            by_id[ident]["provider_request"] = payload
        if len(seen) != len(requests):
            evidence.gaps.append(
                "Provider payload capture is partial; unmatched requests remain unavailable."
            )
    elif kind != "native":
        evidence.gaps.append(
            "Full provider requests/CLI context unavailable; current code is not substituted."
        )
    runtime = evidence.read(agent / "runtime-result.json") or {}
    if kind == "native" and not runtime:
        evidence.gaps.append(
            "Native Runtime result unavailable; HTTP evidence does not replace canonical turns."
        )
    usage["reported_outcome_usage"] = outcome.get("usage")
    changes, repairs, integrity = link_actions(actions, outcome.get("events", []), report)
    collection = evidence.read(agent / "collection.json")
    server_end = evidence.read(source / "session/episode-end.json")
    if collection is not None and server_end is not None and collection != server_end:
        raise ValueError("client/server episode collection differs")
    collection = collection or server_end or {}
    if collection.get("state") == "collected":
        if (
            collection.get("agent_submitted") is not False
            or collection.get("collection_source") != "episode_end"
        ):
            raise ValueError("invalid automatic collection receipt")
        digest = collection.get("candidate_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("collected candidate identity unavailable")
        integrity["collected"] = {"circuit.spi": digest}
    elif (
        collection.get("state") == "missing_candidate"
        and collection.get("candidate_sha256") is not None
    ):
        raise ValueError("missing candidate receipt contains a candidate")
    result, integrity, archives = read_final_evidence(
        evidence, source, agent, report, actions, integrity
    )
    success = (
        result.get("verdict") == "pass"
        if result.get("execution") == "ok" and result.get("verdict") in {"pass", "fail"}
        else None
    )
    score = result.get("score")
    analog_graded = (
        result.get("state") == "graded"
        and _number(score)
        and 0 <= score <= 1
        and result.get("container_exit") == 0
        and result.get("timed_out") is not True
    )
    if analog_graded:
        total, passed = result.get("tests_total"), result.get("tests_passed")
        if type(total) is int and type(passed) is int and total > 0 and 0 <= passed <= total:
            success = passed == total
    durations = [row["elapsed_seconds"] for row in actions if row["elapsed_seconds"] is not None]
    model_durations = [
        row["elapsed_seconds"] for row in requests if row["elapsed_seconds"] is not None
    ]
    simulations = [action for action in actions if action["name"].endswith("_simulate")]
    simulation_durations = [
        action["elapsed_seconds"] for action in simulations if action["elapsed_seconds"] is not None
    ]
    server_simulation_durations = [
        action["server_action_seconds"]
        for action in simulations
        if action.get("server_action_seconds") is not None
    ]
    status = row.get("status", report.get("state"))
    category = "success" if success else "graded" if analog_graded else status or "unclassified"
    if success and repairs:
        repaired = next(row for row in actions if row["id"] == repairs[-1]["successful_action"])
        if repaired["candidate"] == integrity["final"] == integrity["submitted"]:
            category = "success_after_repair"
    if success is False:
        category = "graded_failure"
    elif success is None and not analog_graded and status == "unsubmitted":
        limited = (
            bool(stops)
            or bool(requests and requests[-1]["stop_reason"] == "length")
            or (kind == "native" and outcome.get("termination_reason") == "max_turns")
        )
        category = "budget_exhausted_unsubmitted" if limited else "unsubmitted"
    agent_submitted = True if integrity["submitted"] else collection.get("agent_submitted")
    collection_source = "agent_submit" if agent_submitted else collection.get("collection_source")
    if integrity["submitted"] and collection.get("agent_submitted") is False:
        raise ValueError("collection receipt contradicts an acknowledged submission")
    harness_status = status or "unknown"
    if analog_graded or success is not None:
        harness_status = "finalized"
    elif result:
        harness_status = "final_evaluation_error"
    elif collection.get("state") == "missing_candidate":
        harness_status = "closed_without_candidate"
    elif collection.get("state") == "collected":
        harness_status = "awaiting_final"
    if integrity.get("collected") and analog_graded:
        category = (
            "collected_design_pass"
            if success is True
            else "collected_design_fail"
            if success is False
            else "collected_graded"
        )
    reason = (
        outcome.get("harness_termination_reason")
        or collection.get("termination_reason")
        or outcome.get("termination_reason")
    )
    acknowledged = integrity["submitted"] or integrity.get("collected")
    publicly_simulated = None
    if acknowledged and count is not None:
        publicly_simulated = any(
            action["name"].endswith("_simulate")
            and action["state"] in {"succeeded", "simulated"}
            and action["failed"] is False
            and action["candidate"] == acknowledged
            for action in actions
        )
        if not publicly_simulated and any(action["response"] is None for action in actions):
            publicly_simulated = None
    budget_events = evidence.read(agent / "model-budget.jsonl", lines=True) or []
    timeline = sorted(
        [*actions, *requests],
        key=lambda row: row["started_at"] if _number(row["started_at"]) else math.inf,
    )
    return {
        "schema_version": 1,
        "source": str(source),
        "agent_kind": kind,
        "task_id": result.get("task_id")
        or runtime.get("task_id")
        or report.get("task_id")
        or manifest.get("task_id")
        or row.get("task_id"),
        "experiment_settings": settings,
        "experiment_manifest": manifest,
        "recorded_result_row": row,
        "outcome": {
            "status": status,
            "category": category,
            "benchmark_success": None if analog_graded else success,
            "all_recorded_tests_passed": success if analog_graded else None,
            "verdict": result.get("verdict"),
            "score": score if analog_graded or success is not None else None,
            "authority": result.get("score_authority"),
            "success_criterion": "all_recorded_tests_passed"
            if analog_graded
            else "recorded_verdict",
            "termination_reason": reason,
            "raw_termination_reason": outcome.get("termination_reason"),
            "harness_status": harness_status,
            "agent_submitted": agent_submitted,
            "collection_source": collection_source,
            "final_candidate_publicly_simulated": publicly_simulated,
        },
        "metrics": {
            "tool_calls": count,
            "tool_elapsed_seconds": sum(durations) if durations else None,
            "timed_tool_calls": len(durations),
            "failed_tool_calls": sum(action["failed"] is True for action in actions)
            if count is not None
            else None,
            "unresolved_tool_calls": sum(action["response"] is None for action in actions)
            if count is not None
            else None,
            "model_requests": len(requests)
            if (agent / "model-budget.jsonl").is_file()
            or (kind == "native" and (agent / "http").is_dir())
            else None,
            "model_elapsed_seconds": sum(model_durations) if model_durations else None,
            "model_request_seconds_max": max(model_durations) if model_durations else None,
            "model_length_stops": sum(request["stop_reason"] == "length" for request in requests)
            if requests
            else None,
            "timed_model_requests": len(model_durations),
            "public_simulation_tool_calls": len(simulations) if count is not None else None,
            "public_simulation_elapsed_seconds": sum(simulation_durations)
            if simulation_durations
            else None,
            "public_simulation_server_seconds": sum(server_simulation_durations)
            if server_simulation_durations
            else None,
            "timed_server_simulations": len(server_simulation_durations),
            "recorded_wall_seconds": row.get("wall_elapsed_seconds"),
            "final_process_seconds": result.get("process", {}).get("elapsed_s"),
            "final_wall_seconds": result.get("wall_clock_s"),
            "timing_scope": (
                "Interval sums are not additive wall time when calls overlap. "
                "Model intervals include network, queuing and generation."
            ),
        },
        "usage": usage,
        "agent_input": agent_input,
        "agent_events": outcome.get("events", []),
        "runtime_turns": runtime.get("turns", []),
        "provider_metadata": outcome.get("provider_metadata"),
        "final_text": outcome.get("final_text"),
        "recorded_report": report,
        "candidate_changes": changes,
        "repairs": repairs,
        "candidate_integrity": integrity,
        "archives": archives,
        "final_result": result,
        "actions": actions,
        "model_requests": requests,
        "budget_stops": stops,
        "budget_context": [
            event for event in budget_events if event.get("event") == "budget_context"
        ],
        "collection_receipt": collection,
        "timeline": timeline,
        "sources": evidence.sources,
        "evidence_gaps": evidence.gaps,
    }


def write_report(report: dict, output: Path) -> None:
    """Write a fresh private derived report, never into the evidence tree."""
    from circuit_harness.reporting.view import render_html, render_markdown

    output = Path(output).absolute()
    source = Path(report["source"])
    if output.exists() or output.resolve().is_relative_to(source):
        raise ValueError("output must be a new directory outside the source evidence")
    content = {
        "episode-report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        "report.html": render_html(report),
        "report.md": render_markdown(report),
    }
    stream = io.StringIO()
    columns = ["kind", "id", "name", "started_at", "finished_at", "elapsed_seconds", "source"]
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["timeline"])
    content["timeline.csv"] = stream.getvalue()
    output.mkdir(parents=True, mode=0o700)
    for name, text in content.items():
        fd = os.open(output / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="saved Chips agent or experiment directory")
    parser.add_argument("--output", type=Path, required=True, help="new private report directory")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.source)
        write_report(report, args.output)
    except (ValueError, OSError) as exc:
        print(f"episode report: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output / 'report.html'} ({report['outcome']['category']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
