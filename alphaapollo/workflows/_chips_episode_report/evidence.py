"""Link saved Chips tool actions, candidate versions and independently sealed results."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import tarfile


def candidate_hashes(result):
    if isinstance(result.get("candidate"), dict):
        return {name: item["sha256"] for name, item in result["candidate"].items()}
    digest = result.get("candidate_sha256")
    return {"circuit.spi": digest} if digest else None


def link_actions(actions, events, recorded_report):
    """IDs join agent and tool logs; changed bytes plus ordered feedback evidence repairs."""
    by_id = {row["id"]: row for row in actions}
    for row in actions:
        row["agent_call_ids"] = []
        row["agent_event_indices"] = []
        row["candidate"] = candidate_hashes((row["response"] or {}).get("result", {}))
    for index, event in enumerate(events):
        if event.get("kind") != "tool_result":
            continue
        try:
            reply = json.loads(event.get("content") or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(reply, dict):
            continue
        row = by_id.get(reply.get("action_id"))
        if row is not None and row["response"] == reply:
            row["agent_event_indices"].append(index + 1)
            if event.get("call_id"):
                row["agent_call_ids"].append(event["call_id"])
    versions, contents, changes, repairs = {}, {}, [], []
    failed_simulation = None
    submitted = None
    for row in actions:
        args = row["request"]["arguments"]
        result = (row["response"] or {}).get("result", {})
        if (
            row["name"] in {"vabench_write", "analog_write", "analog_restore"}
            and (row["response"] or {}).get("ok") is True
        ):
            name = args.get("path", "circuit.spi")
            restoring = row["name"] == "analog_restore"
            reported = result.get("sha256", result.get("candidate_sha256"))
            if restoring and args.get("candidate_sha256") != reported:
                raise ValueError(f"restore differs from requested hash: {row['id']}")
            content = contents.get((name, reported)) if restoring else args.get("content")
            if not isinstance(content, str):
                if restoring:
                    changes.append(
                        {
                            "action_id": row["id"],
                            "path": name,
                            "sha256": reported,
                            "operation": "restore",
                            "previous_sha256": None,
                            "diff": None,
                            "baseline": "content_unavailable",
                        }
                    )
                    versions.pop(name, None)
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if reported is not None and reported != digest:
                raise ValueError(f"write content differs from reported hash: {row['id']}")
            previous = versions.get(name) if row["started_at"] is not None else None
            changes.append(
                {
                    "action_id": row["id"],
                    "operation": "restore" if restoring else "write",
                    "path": name,
                    "sha256": digest,
                    "previous_sha256": previous[0] if previous else None,
                    "diff": "".join(
                        difflib.unified_diff(
                            (previous[1] if previous else "").splitlines(keepends=True),
                            content.splitlines(keepends=True),
                            fromfile=f"previous/{name}",
                            tofile=f"{row['id']}/{name}",
                        )
                    ),
                    "baseline": "previous_observed_write" if previous else "first_observed_write",
                }
            )
            versions[name] = digest, content
            contents[name, digest] = content
        if row["name"] in {"vabench_simulate", "analog_simulate"}:
            if row["failed"] is True:
                failed_simulation = row
            elif row["state"] in {"succeeded", "simulated"} and row["failed"] is False:
                old = failed_simulation
                if (
                    old
                    and old["candidate"]
                    and row["candidate"]
                    and old["candidate"] != row["candidate"]
                    and old["finished_at"] is not None
                    and row["started_at"] is not None
                    and old["finished_at"] <= row["started_at"]
                ):
                    repairs.append(
                        {
                            "failed_action": old["id"],
                            "successful_action": row["id"],
                            "candidate_changed": True,
                            "evidence": "observed_sequence_not_causal_proof",
                        }
                    )
                failed_simulation = None
        if (
            row["name"] in {"vabench_submit", "analog_submit"}
            and row["state"] == "submitted"
            and (row["response"] or {}).get("ok") is True
        ):
            submitted = row["candidate"]
    final_candidate = candidate_hashes(recorded_report)
    if submitted and final_candidate and submitted != final_candidate:
        raise ValueError("reported final candidate differs from submission")
    integrity = {
        "status": "reported_match" if submitted and final_candidate else "unavailable",
        "submitted": submitted,
        "final": final_candidate,
        "scope": "Reported hashes only; archive bytes not verified.",
    }
    return changes, repairs, integrity


def read_final_evidence(evidence, source, agent, report, actions, integrity):
    """An extracted Analog episode binds frozen and scored bytes before showing a score."""
    result = report.get("result", {})
    transported = evidence.read(source / "final-result.json")
    final = evidence.read(source / "final/result.json")
    if transported is not None and final is not None:
        raise ValueError("ambiguous final evidence layouts")
    acknowledged = integrity["submitted"] or integrity.get("collected")
    if transported is not None:
        expected = {"circuit.spi": transported.get("frozen_candidate_sha256")}
        if not expected["circuit.spi"] or acknowledged != expected:
            raise ValueError("final result differs from acknowledged submission")
        integrity = {
            **integrity,
            "status": "reported_match",
            "submitted": integrity["submitted"],
            "final": expected,
            "scope": "Reported digest matched; remote bytes not read.",
        }
        evidence.gaps.append("Remote Analog archive is not downloaded by this offline report.")
        return transported, integrity, []
    if final is not None:
        frozen = evidence.read(source / "session/frozen.json") or {}
        expected = candidate_hashes(frozen)
        if not expected:
            raise ValueError("Analog final result has no frozen candidate identity")
        for name in ("session/frozen/circuit.spi", "final/inputs/circuit.spi"):
            digest = evidence.fingerprint(source / name)["sha256"]
            if digest != expected["circuit.spi"]:
                raise ValueError("final scorer candidate differs from frozen bytes")
        if acknowledged and acknowledged != expected:
            raise ValueError("frozen candidate differs from acknowledged submission")
        integrity = {
            **integrity,
            "status": "bytes_matched",
            "submitted": integrity["submitted"],
            "final": expected,
            "scope": "Frozen/input bytes matched; extracted final result is not an archive seal.",
        }
        evidence.gaps.append(
            "Analog extracted result is read as recorded; archive authenticity not certified."
        )
        result = final
    else:
        result, integrity, archives = _vabench_archives(evidence, agent, report, actions, integrity)
        return result, integrity, archives
    for row in actions:
        folder = source / "session/actions" / row["id"]
        if folder.is_dir():
            _server_action(
                row,
                evidence.read(folder / "request.json"),
                evidence.read(folder / "response.json"),
                evidence.read(folder / "measurement.json"),
            )
    return result, integrity, []


def _server_action(row, request, response, measurement):
    if request is not None and request != row["request"]:
        raise ValueError(f"client/server action request differs: {row['id']}")
    if response is not None and row["response"] is not None:
        client = {key: value for key, value in row["response"].items() if key != "action_id"}
        if client != response:
            raise ValueError(f"client/server action response differs: {row['id']}")
    value = (measurement or {}).get("elapsed_s")
    row["server_action_seconds"] = (
        value if (type(value) in (int, float) and math.isfinite(value) and value >= 0) else None
    )


def _find_archive(evidence, agent, expected_hash, public):
    if not expected_hash:
        return None
    root = agent / "archives"
    paths = (root / "episodes").glob("*/receipt.json") if public else root.glob("*/receipt.json")
    matches = []
    for path in paths:
        receipt = evidence.read(path)
        digest = receipt.get("sha256") if public else receipt.get("package", {}).get("sha256")
        if digest == expected_hash:
            matches.append((path.parent, receipt))
    if len(matches) > 1:
        raise ValueError("ambiguous archive identity")
    if not matches:
        evidence.gaps.append("Referenced public/final archive is unavailable locally.")
        return None
    record, receipt = matches[0]
    members = receipt.get("members", {})
    if len(members) > 5000 or sum(item["bytes"] for item in members.values()) > 256 * 1024 * 1024:
        raise ValueError("archive exceeds episode reporting limits")
    package = record / ("episode.tar.gz" if public else "job.tar.gz")
    if package.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("archive exceeds episode reporting limits")
    evidence.fingerprint(package)
    return record, receipt


def _vabench_archives(evidence, agent, report, actions, integrity):
    from alphaapollo.common.execution.chips.archive import verify_archive
    from alphaapollo.common.execution.chips.vabench_session import verify_episode_archive

    summaries = []
    result = report.get("result", {})
    public = _find_archive(evidence, agent, report.get("public_trace_sha256"), True)
    final = _find_archive(evidence, agent, report.get("final_archive_sha256"), False)
    expected = integrity["final"] or integrity["submitted"]
    for located, is_public in ((public, True), (final, False)):
        if located is None:
            continue
        record, _ = located
        receipt = verify_episode_archive(record) if is_public else verify_archive(record)
        hashes = candidate_hashes(receipt if is_public else receipt["request"]["identity"])
        if expected and expected != hashes:
            raise ValueError("archive candidate differs from submitted/final candidate")
        expected = hashes
        summaries.append(
            {
                "kind": "public" if is_public else "final",
                "path": str(record),
                "verified_members": len(receipt["members"]),
                "timings_s": receipt.get("timings_s"),
            }
        )
        if is_public:
            with tarfile.open(record / "episode.tar.gz", "r:gz") as archive:

                def read_member(name, members=receipt["members"]):
                    if name not in members:
                        return None
                    with archive.extractfile(name) as stream:
                        return json.load(stream)

                if candidate_hashes(read_member("frozen.json") or {}) != hashes:
                    raise ValueError("public archive freeze differs from receipt")
                for name, digest in (hashes or {}).items():
                    member = "candidate/" + name
                    if (
                        member not in receipt["members"]
                        or receipt["members"][member]["sha256"] != digest
                    ):
                        raise ValueError("public candidate bytes differ from freeze")
                for row in actions:
                    prefix = f"actions/{row['id']}/"
                    _server_action(
                        row,
                        read_member(prefix + "request.json"),
                        read_member(prefix + "response.json"),
                        read_member(prefix + "measurement.json"),
                    )
        else:
            recorded = receipt["completion"]["result"]
            if result and result != recorded:
                raise ValueError("final archive result differs from recorded report")
            result = recorded
    if public or final:
        integrity = {
            "status": "verified_archives"
            if public and final
            else "public_archive_verified"
            if public
            else "final_archive_verified",
            "submitted": integrity["submitted"],
            "final": expected,
            "scope": "Local archive/member hashes checked; this is not external certification.",
        }
    if not public or not final:
        evidence.gaps.append(
            "Both sealed public and final archives are needed for full VABench linkage."
        )
    return result, integrity, summaries
