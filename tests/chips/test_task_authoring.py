"""Constructed document fixtures; these tests do not certify expert review."""

import json
import stat

import pytest

from circuit_harness.execution.journal import file_digest
from circuit_harness.execution.task_authoring import (
    confirm_draft,
    inspect_draft,
    require_confirmation,
)


def draft_fixture(tmp_path):
    source = tmp_path / "spec.txt"
    source.write_text("Supply 1.8 V. Gain at a frequency to be confirmed.\n")
    return {
        "schema_version": 1,
        "task_id": "gain-d0",
        "sources": [{"path": source.name, "sha256": file_digest(source)}],
        "fields": {
            "supply": {
                "value": 1.8,
                "unit": "V",
                "status": "explicit",
                "evidence": [{"source": source.name, "page": 1, "region": "first sentence"}],
            }
        },
    }


def test_absent_required_condition_prevents_ready_state(tmp_path):
    draft = draft_fixture(tmp_path)
    review = inspect_draft(draft, tmp_path, {"supply": "V", "frequency": "Hz"})
    assert review["status"] == "needs_clarification"
    assert {"field": "frequency", "reason": "missing"} in review["issues"]


def test_source_changes_or_invented_locations_are_rejected(tmp_path):
    draft = draft_fixture(tmp_path)
    (tmp_path / "spec.txt").write_text("Supply 3.3 V.\n")
    with pytest.raises(ValueError, match="source.*changed"):
        inspect_draft(draft, tmp_path, {"supply": "V"})
    draft = draft_fixture(tmp_path)
    draft["fields"]["supply"]["evidence"][0]["source"] = "unprovided.png"
    with pytest.raises(ValueError, match="evidence"):
        inspect_draft(draft, tmp_path, {"supply": "V"})


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"status": "conflict"}, "conflict"),
        ({"value": None, "status": "missing"}, "missing"),
        ({"unit": None}, "unit_mismatch"),
        ({"unit": "mV"}, "unit_mismatch"),
        ({"evidence": []}, "missing_evidence"),
    ],
)
def test_unresolved_fields_are_not_silently_accepted(tmp_path, change, reason):
    draft = draft_fixture(tmp_path)
    draft["fields"]["supply"].update(change)
    review = inspect_draft(draft, tmp_path, {"supply": "V"})
    assert {"field": "supply", "reason": reason} in review["issues"]
    assert review["status"] == "needs_clarification"


def test_confirmation_is_required_and_binds_draft_and_operator_requirements(tmp_path):
    draft = draft_fixture(tmp_path)
    requirements = {"supply": "V"}
    with pytest.raises(ValueError, match="confirmation"):
        require_confirmation(draft, tmp_path, requirements, None)
    receipt = confirm_draft(
        draft, tmp_path, requirements, reviewer="fixture-author", kind="scripted_confirmation"
    )
    assert require_confirmation(draft, tmp_path, requirements, receipt) == {"supply": 1.8}
    draft["fields"]["supply"]["value"] = 3.3
    with pytest.raises(ValueError, match="confirmation"):
        require_confirmation(draft, tmp_path, requirements, receipt)
    draft["fields"]["supply"]["value"] = 1.8
    with pytest.raises(ValueError, match="confirmation"):
        require_confirmation(draft, tmp_path, {}, receipt)


@pytest.mark.parametrize(
    "case", ["traversal", "symlink", "duplicate", "status", "bool", "location"]
)
def test_untrusted_drafts_cannot_bypass_source_or_value_checks(tmp_path, case):
    draft = draft_fixture(tmp_path)
    if case == "traversal":
        draft["sources"][0]["path"] = "../" + tmp_path.name + "/spec.txt"
    elif case == "symlink":
        (tmp_path / "real.txt").write_bytes((tmp_path / "spec.txt").read_bytes())
        (tmp_path / "spec.txt").unlink()
        (tmp_path / "spec.txt").symlink_to(tmp_path / "real.txt")
    elif case == "duplicate":
        draft["sources"].append(dict(draft["sources"][0]))
    elif case == "status":
        draft["fields"]["supply"]["status"] = "approved_by_model"
    elif case == "bool":
        draft["fields"]["supply"]["value"] = True
    else:
        draft["fields"]["supply"]["evidence"][0]["page"] = False
    with pytest.raises(ValueError):
        inspect_draft(draft, tmp_path, {"supply": "V"})


def test_incomplete_receipt_and_unresolved_confirmation_are_rejected(tmp_path):
    draft = draft_fixture(tmp_path)
    review = inspect_draft(draft, tmp_path, {"supply": "V"})
    with pytest.raises(ValueError, match="confirmation"):
        require_confirmation(draft, tmp_path, {"supply": "V"}, review)
    draft["fields"]["supply"]["status"] = "conflict"
    with pytest.raises(ValueError, match="confirmation"):
        confirm_draft(draft, tmp_path, {"supply": "V"}, reviewer="author", kind="human")


def test_operator_cli_confirms_once_and_detects_later_source_change(tmp_path, capsys):
    from circuit_harness.task_authoring import main

    draft = draft_fixture(tmp_path)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft))
    rules = tmp_path / "requirements.json"
    rules.write_text(json.dumps({"supply": "V"}))
    args = ["--draft", str(path), "--materials", str(tmp_path), "--requirements", str(rules)]
    assert main(["inspect", *args]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready_for_confirmation"
    receipt = tmp_path / "confirmation.json"
    confirm = [
        "confirm",
        *args,
        "--reviewer",
        "fixture-author",
        "--kind",
        "scripted_confirmation",
        "--output",
        str(receipt),
    ]
    assert main(confirm) == 0
    capsys.readouterr()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert main(confirm) == 1  # An existing operator record is never overwritten.
    capsys.readouterr()
    assert main(["verify", *args, "--confirmation", str(receipt)]) == 0
    capsys.readouterr()
    (tmp_path / "spec.txt").write_text("New revision")
    assert main(["verify", *args, "--confirmation", str(receipt)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
