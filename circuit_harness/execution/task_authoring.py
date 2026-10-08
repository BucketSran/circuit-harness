"""Operator confirmation of sourced task drafts, before simulation is allowed."""

import json
import math
import re
import time
from pathlib import Path, PurePosixPath

from .journal import digest, file_digest


def _validate_shape(draft: dict, requirements: dict) -> None:
    if (
        not isinstance(draft, dict)
        or set(draft) != {"schema_version", "task_id", "sources", "fields"}
        or type(draft["schema_version"]) is not int
        or draft["schema_version"] != 1
        or not isinstance(draft["task_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", draft["task_id"])
    ):
        raise ValueError("invalid task draft schema")
    if not isinstance(requirements, dict) or any(
        not isinstance(name, str)
        or not name
        or (unit is not None and (not isinstance(unit, str) or not unit))
        for name, unit in requirements.items()
    ):
        raise ValueError("requirements must map field names to canonical units or null")
    if not isinstance(draft["fields"], dict) or not 1 <= len(draft["fields"]) <= 128:
        raise ValueError("draft must have 1-128 fields")
    if not isinstance(draft["sources"], list) or not 1 <= len(draft["sources"]) <= 32:
        raise ValueError("draft must have 1-32 sources")
    try:
        if len(json.dumps(draft, allow_nan=False).encode()) > 1024 * 1024:
            raise ValueError("draft is too large")
    except (TypeError, OverflowError) as error:
        raise ValueError("draft must contain finite JSON data") from error
    for name, field in draft["fields"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(field, dict)
            or not {"value", "unit", "status", "evidence"} <= set(field)
            or set(field) - {"value", "unit", "status", "evidence", "note"}
            or field["status"] not in ("explicit", "inferred", "missing", "conflict")
            or not isinstance(field["evidence"], list)
            or len(field["evidence"]) > 32
        ):
            raise ValueError("invalid draft field")
        if field["unit"] is not None and (not isinstance(field["unit"], str) or not field["unit"]):
            raise ValueError("invalid field unit")
        if field["unit"] is not None and field["value"] is not None:
            if type(field["value"]) not in (int, float):
                raise ValueError("physical quantities must be numeric, not booleans or strings")
        if "note" in field and not isinstance(field["note"], str):
            raise ValueError("field note must be text")
        if field["status"] == "inferred" and not field.get("note", "").strip():
            raise ValueError("inferred values require a note for operator review")


def _verify_sources(sources: list, materials: Path) -> set[str]:
    root = Path(materials).resolve(strict=True)
    names = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError("invalid source record")
        name = source["path"]
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("invalid source path")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("source path must be a canonical relative path")
        if name in names:
            raise ValueError("duplicate source")
        names.add(name)
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("source path cannot contain a symlink")
        if (
            not path.is_file()
            or path.stat().st_size > 32 * 1024 * 1024
            or file_digest(path) != source["sha256"]
        ):
            raise ValueError(f"source missing, oversized or changed: {name}")
    return names


def inspect_draft(draft: dict, materials, requirements: dict) -> dict:
    """Inspect a draft against operator-owned required fields and units."""
    _validate_shape(draft, requirements)
    sources = _verify_sources(draft["sources"], materials)
    issues = [
        {"field": name, "reason": "missing"} for name in requirements if name not in draft["fields"]
    ]
    for name, field in draft["fields"].items():
        for evidence in field["evidence"]:
            if (
                not isinstance(evidence, dict)
                or set(evidence) != {"source", "page", "region"}
                or not isinstance(evidence["source"], str)
                or evidence["source"] not in sources
                or type(evidence["page"]) is not int
                or evidence["page"] < 1
                or not isinstance(evidence["region"], str)
                or not evidence["region"].strip()
            ):
                raise ValueError(f"invalid evidence for {name}")
        reasons = []
        if field["status"] in {"missing", "conflict"}:
            reasons.append(field["status"])
        elif field["value"] is None:
            reasons.append("missing")
        if name in requirements and field["unit"] != requirements[name]:
            reasons.append("unit_mismatch")
        if field["status"] == "explicit" and not field["evidence"]:
            reasons.append("missing_evidence")
        issues.extend({"field": name, "reason": reason} for reason in reasons)
    return {
        "status": "needs_clarification" if issues else "ready_for_confirmation",
        "issues": issues,
        "review_sha256": digest(
            json.dumps(
                {"draft": draft, "requirements": requirements},
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ),
    }


def confirm_draft(draft, materials, requirements, *, reviewer, kind):
    """Record an operator decision; this is not an authentication mechanism."""
    review = inspect_draft(draft, materials, requirements)
    if review["issues"]:
        raise ValueError("confirmation requires resolved fields and units")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("confirmation requires a named reviewer")
    if kind not in {"human", "scripted_confirmation"}:
        raise ValueError("invalid confirmation kind")
    return {
        "schema_version": 1,
        "review_sha256": review["review_sha256"],
        "reviewer": reviewer,
        "kind": kind,
        "confirmed_at": time.time(),
    }


def require_confirmation(draft, materials, requirements, receipt):
    """Return confirmed values or refuse to advance to simulation."""
    review = inspect_draft(draft, materials, requirements)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"schema_version", "review_sha256", "reviewer", "kind", "confirmed_at"}
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["kind"] not in ("human", "scripted_confirmation")
        or not isinstance(receipt["reviewer"], str)
        or not receipt["reviewer"].strip()
        or type(receipt["confirmed_at"]) not in (int, float)
        or not math.isfinite(receipt["confirmed_at"])
        or receipt["confirmed_at"] <= 0
        or receipt.get("review_sha256") != review["review_sha256"]
        or review["issues"]
    ):
        raise ValueError("missing or stale confirmation")
    return {name: field["value"] for name, field in draft["fields"].items()}
