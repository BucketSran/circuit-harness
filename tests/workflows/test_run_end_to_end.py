from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from alphaapollo.data_preprocess import read_records
from alphaapollo.data_preprocess.prepare_custom_data import prepare as prepare_fixture_dataset
from alphaapollo.workflows.executor import WorkflowExecutor
from alphaapollo.workflows.main import main


def _scored_run(tmp_path: Path) -> tuple[Path, Path]:
    """Write a one-problem scored run whose fake backend answers correctly.

    Returns the config path and the output directory.
    """

    output = tmp_path / "run"
    dataset_root = tmp_path / "prepared"
    source = tmp_path / "fixture.jsonl"
    source.write_text(
        json.dumps({"problem_idx": 1, "problem": "What is 40 + 2?", "answer": 42}) + "\n",
        encoding="utf-8",
    )
    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=dataset_root,
        dataset_name="fixture_test",
        revision=None,
        splits=("test",),
        output_format="jsonl",
    )
    first_private = read_records(prepared.private_splits["test"])[0]
    config = {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "scored_run_e2e",
            "roles": [
                {
                    "id": "solver",
                    "target": "solver",
                    "system_prompt": "Solve the problem and give a boxed final answer.",
                }
            ],
            "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
            "entry_step": "solve",
        },
        "dataset": {
            "path": str(prepared.public_splits["test"]),
            "format": "jsonl",
            "input_key": "statement",
            "id_key": "task_uid",
        },
        "runtimes": {
            "solver": {
                "type": "alphaapollo",
                "options": {
                    "backend": {
                        "type": "fake",
                        "options": {"text": f"\\boxed{{{first_private['answer']}}}"},
                    },
                    "model": "offline",
                    "sampling": {"temperature": 0.0, "max_tokens": 32},
                    "max_turns": 1,
                },
            }
        },
        "verifiers": {},
        "environment": {"type": "text_only", "options": {}},
        "output": {"directory": str(output), "format": "jsonl"},
        "scoring": {
            "dataset_root": str(dataset_root),
            "source_id": "fixture_test",
            "version": "v1",
        },
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, output


def test_command_runs_workflow_and_writes_scored_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor_called = False
    original_run_batch = WorkflowExecutor.run_batch

    def recording_run_batch(self: WorkflowExecutor, inputs: object) -> object:
        nonlocal executor_called
        executor_called = True
        return original_run_batch(self, inputs)  # type: ignore[arg-type]

    monkeypatch.setattr(WorkflowExecutor, "run_batch", recording_run_batch)
    config_path, output = _scored_run(tmp_path)

    assert main(["--config", str(config_path)]) == 0
    assert executor_called is True
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    no_tool = report["conditions"]["no_tool"]
    assert no_tool["n_runs"] == 1
    assert no_tool["n_correct"] == 1
    assert no_tool["pass_at_1"] == 1.0
    assert (output / "no_tool" / "p0000" / "sample-000" / "result.json").is_file()
    assert (output / "no_tool" / "p0000" / "sample-000" / "traj.jsonl").is_file()
    assert (output / "report.md").is_file()


def test_command_exits_non_zero_when_every_cell_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that never ran must not exit 0 beside a report of 0.000.

    This is how an unreachable server or a model name the server 404s arrives:
    every cell raises, every record is a typed error, and the rates that follow
    measure nothing. The report is still written -- it holds the evidence.
    """

    def failing_run_batch(self: WorkflowExecutor, inputs: object) -> object:
        raise RuntimeError("404 The model 'served-model' does not exist")

    monkeypatch.setattr(WorkflowExecutor, "run_batch", failing_run_batch)
    config_path, output = _scored_run(tmp_path)

    assert main(["--config", str(config_path)]) == 1

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    no_tool = report["conditions"]["no_tool"]
    assert no_tool["n_runs"] == 1
    assert no_tool["n_error_runs"] == 1
    assert no_tool["pass_at_1"] == 0.0
    assert "NO RESULT" in (output / "report.md").read_text(encoding="utf-8")
    assert "measure nothing" in capsys.readouterr().err
    result = json.loads(
        (output / "no_tool" / "p0000" / "sample-000" / "result.json").read_text(encoding="utf-8")
    )
    assert "does not exist" in result["error"]
