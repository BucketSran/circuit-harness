"""Explicit offline ATIF selection for conversational SFT; no training or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from circuit_harness.data.io import commit_directory, file_sha256


class AtifDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["atif-sft-v1"] = "atif-sft-v1"
    parquet_encoding: Literal["json-columns"] = "json-columns"
    selection_policy: Literal["all_assistant_steps"] = "all_assistant_steps"
    reasoning: Literal["exclude", "include"]
    allow_reconstructed_context: bool
    sources: list[dict[str, Any]]
    splits: dict[str, int]
    files: dict[str, str]
    publication_safe: Literal[False] = False


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def checked_manifest(directory):
    """Validate dataset files before loading them; hashes are integrity, not signatures."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.is_symlink():
        raise ValueError("manifest must be a regular file")
    manifest = AtifDatasetManifest.model_validate_json(path.read_bytes())
    expected = {f"{split}.{suffix}" for split in manifest.splits for suffix in ("jsonl", "parquet")}
    if set(manifest.files) != expected or any(
        split not in {"train", "validation", "test"} for split in manifest.splits
    ):
        raise ValueError("unexpected dataset file inventory")
    for name, digest in manifest.files.items():
        file = directory / name
        if file.is_symlink() or not file.is_file() or file_sha256(file) != digest:
            raise ValueError("dataset integrity check failed")
    return manifest


def sft_record(trajectory, *, reasoning, allow_reconstructed_context=False):
    """Project text ATIF to ordinary messages/tools, keeping reward out of context."""
    if reasoning not in {"exclude", "include"}:
        raise ValueError("declare reasoning policy: exclude or include")
    if trajectory.continued_trajectory_ref or trajectory.subagent_trajectories:
        raise ValueError("continued/subagent trajectories require a separate selection policy")
    extra = trajectory.extra or {}
    if extra.get("context_fidelity") == "native_reconstruction" and not allow_reconstructed_context:
        raise ValueError("explicitly acknowledge reconstructed native context")
    circuit = extra.get("circuit", {})
    grade_status = circuit.get("grade_status")
    reward = circuit.get("reward")
    if grade_status == "verified":
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (float, int))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError("invalid independently verified reward")
    elif grade_status != "unscored" or reward is not None:
        raise ValueError("unverified trajectories must remain unscored with no reward")
    for key in ("task_id", "task_version", "candidate_sha256", "trial_id"):
        if not isinstance(circuit.get(key), str) or not circuit[key]:
            raise ValueError("missing circuit identity")
    definitions = trajectory.agent.tool_definitions
    if not definitions:
        raise ValueError("missing tool definitions")
    names = {tool["function"]["name"] for tool in definitions}
    messages, seen_calls = [], set()
    for step in trajectory.steps:
        if step.is_copied_context or step.llm_call_count == 0:
            raise ValueError("copied/deterministic context requires a separate loss policy")
        if not isinstance(step.message, str):
            raise ValueError("only text SFT is supported")
        text = step.message
        if reasoning == "include" and step.reasoning_content:
            text = "<think>\n" + step.reasoning_content + "\n</think>\n" + text
        message = {"role": "assistant" if step.source == "agent" else step.source, "content": text}
        calls = step.tool_calls or []
        ids = [call.tool_call_id for call in calls]
        if len(ids) != len(set(ids)) or seen_calls.intersection(ids):
            raise ValueError("duplicate tool call ID")
        seen_calls.update(ids)
        if any(call.function_name not in names for call in calls):
            raise ValueError("tool call lacks a definition")
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {"name": call.function_name, "arguments": call.arguments},
                }
                for call in calls
            ]
        messages.append(message)
        results = step.observation.results if step.observation else []
        if len(results) != len(ids) or {r.source_call_id for r in results} != set(ids):
            raise ValueError("tool calls and observations do not match")
        for result in results:
            if not isinstance(result.content, str) or result.subagent_trajectory_ref:
                raise ValueError("only direct text tool observations are supported")
            messages.append(
                {"role": "tool", "tool_call_id": result.source_call_id, "content": result.content}
            )
    if (
        not messages
        or messages[0]["role"] != "system"
        or len(messages) < 3
        or messages[1]["role"] != "user"
        or messages[-1]["role"] != "assistant"
        or messages[-1].get("tool_calls")
    ):
        raise ValueError("require initial system/user and completed final assistant message")
    return {
        "messages": messages,
        "tools": definitions,
        "extra_info": {
            **{
                key: circuit[key]
                for key in ("task_id", "task_version", "candidate_sha256", "trial_id")
            },
            "grade_status": grade_status,
            "reward": reward,
            "session_id": trajectory.session_id,
            "context_fidelity": extra.get("context_fidelity", "unspecified"),
            "native_sha256": extra.get("source_sha256"),
            "reasoning": reasoning,
        },
    }


def prepare_dataset(selection, output, *, reasoning, allow_reconstructed_context=False):
    """Export explicit task-group assignments, refusing cross-split leakage and overwrite.

    JSONL contains structured messages/tools. Parquet stores those same values in JSON
    columns so Arrow cannot union heterogeneous arguments/schemas and insert null keys.
    Use AtifSFTDataset to decode them without data loss.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from harbor.models.trajectories import Trajectory

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not selection:
        raise ValueError("selection must not be empty")
    rows, sources, groups, tasks, seen = {}, [], {}, {}, set()
    for entry in selection:
        if set(entry) != {"trajectory", "split", "task_group"}:
            raise ValueError("selection requires trajectory, split and task_group only")
        path, split, group = Path(entry["trajectory"]), entry["split"], entry["task_group"]
        if split not in {"train", "validation", "test"} or not isinstance(group, str) or not group:
            raise ValueError("declare a supported split and nonempty task_group")
        if path.is_symlink() or not path.is_file():
            raise ValueError("trajectory must be a regular file")
        if path.resolve().is_relative_to(output.resolve()):
            raise ValueError("source overlaps output")
        raw = path.read_bytes()
        value = Trajectory.model_validate_json(raw)
        row = sft_record(
            value, reasoning=reasoning, allow_reconstructed_context=allow_reconstructed_context
        )
        identity = row["extra_info"]
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen or identity["trial_id"] in seen or value.session_id in seen:
            raise ValueError("duplicate trajectory selection")
        seen.update((digest, identity["trial_id"], value.session_id))
        task = identity["task_id"]
        if task in tasks and tasks[task] != group:
            raise ValueError("the same task cannot use different task groups")
        tasks[task] = group
        if group in groups and groups[group] != split:
            raise ValueError("task group leaks across splits")
        groups[group] = split
        row["extra_info"].update(task_group=group, trajectory_sha256=digest)
        rows.setdefault(split, []).append(row)
        sources.append({**row["extra_info"], "split": split})
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".atif-sft-", dir=output.parent))
    try:
        files = {}
        for split, records in rows.items():
            jsonl = staged / f"{split}.jsonl"
            jsonl.write_text("\n".join(_json(row) for row in records) + "\n", encoding="utf-8")
            parquet = staged / f"{split}.parquet"
            encoded = [
                {key + "_json": _json(row[key]) for key in ("messages", "tools", "extra_info")}
                for row in records
            ]
            pq.write_table(pa.Table.from_pylist(encoded), parquet)
            for path in (jsonl, parquet):
                path.chmod(0o600)
                files[path.name] = file_sha256(path)
        manifest = AtifDatasetManifest(
            reasoning=reasoning,
            allow_reconstructed_context=allow_reconstructed_context,
            sources=sources,
            splits={split: len(records) for split, records in rows.items()},
            files=files,
        )
        manifest_path = staged / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o600)
        commit_directory(staged, output, replace=False)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, required=True, help="JSON array; paths relative to this file"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning", choices=("include", "exclude"), required=True)
    parser.add_argument("--allow-reconstructed-context", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if not isinstance(selection, list):
        raise ValueError("selection must be an array")
    for item in selection:
        item["trajectory"] = str(args.selection.parent / item["trajectory"])
    print(
        prepare_dataset(
            selection,
            args.output,
            reasoning=args.reasoning,
            allow_reconstructed_context=args.allow_reconstructed_context,
        )
    )


if __name__ == "__main__":
    main()
