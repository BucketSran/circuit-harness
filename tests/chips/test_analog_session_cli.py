"""Session commands must survive packaging as the server-side offline bundle."""

import json
import subprocess
import sys

import pytest

from alphaapollo.common.execution.chips.bundle import build_cli
from alphaapollo.workflows import chips


@pytest.mark.parametrize("task_id", [None, "rlc-broadband-50-to-200-match"])
def test_session_cli_dispatches_actions(tmp_path, monkeypatch, capsys, task_id):
    seen = {}

    def fake_create(source, output, **options):
        seen.update(source=source, output=output, **options)
        return {"state": "ready"}

    def fake_action(directory, request):
        seen.update(directory=directory, request=request)
        return {"ok": True, "result": {"state": "submitted"}}

    monkeypatch.setattr(chips, "create_analog_session", fake_create, raising=False)
    monkeypatch.setattr(chips, "analog_session_action", fake_action, raising=False)
    assert (
        chips.main(
            [
                "analog-session",
                "--source-root",
                str(tmp_path / "source"),
                "--output",
                str(tmp_path / "session"),
                "--max-simulations",
                "2",
                *(["--task-id", task_id] if task_id else []),
            ]
        )
        == 0
    )
    assert seen["max_simulations"] == 2
    assert seen["task_id"] == (task_id or "rlc-rf-bandpass-100mhz")
    capsys.readouterr()
    request = {"id": "submit-1", "tool": "analog_submit", "arguments": {}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert (
        chips.main(
            ["analog-action", "--session", str(tmp_path / "session"), "--request", str(path)]
        )
        == 0
    )
    assert seen["request"] == request
    assert json.loads(capsys.readouterr().out)["result"]["state"] == "submitted"


def test_session_commands_exist_in_offline_bundle(tmp_path):
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    for command in (
        "analog-session",
        "analog-close",
        "analog-info",
        "analog-action",
        "analog-request",
        "analog-response",
        "analog-finalize",
        "analog-archive",
        "verify-analog-episode",
    ):
        completed = subprocess.run(
            [sys.executable, str(bundle), command, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_finalizer_cli_uses_operator_only_entry(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_finalize(session, output, **options):
        seen.update(session=session, output=output, **options)
        return {"state": "graded", "score": 0.0}

    monkeypatch.setattr(chips, "finalize_analog_session", fake_finalize, raising=False)
    assert (
        chips.main(
            [
                "analog-finalize",
                "--session",
                str(tmp_path / "session"),
                "--output",
                str(tmp_path / "score"),
            ]
        )
        == 0
    )
    assert seen["output"] == tmp_path / "score"
    assert json.loads(capsys.readouterr().out)["score"] == 0.0
