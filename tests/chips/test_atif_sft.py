"""SFT conversion preserves native tool structures and split identity."""

import json

import pytest

pytest.importorskip("harbor")
pytest.importorskip("pyarrow")
from test_pi_trajectory import session

from alphaapollo.data_preprocess.prepare_atif import prepare_dataset
from alphaapollo.workflows.harbor_chips.pi_trajectory import read_pi_trajectory


def trajectory(tmp_path):
    path, _ = session(tmp_path)
    value = read_pi_trajectory(path, agent_version="fixture-v1")
    value.extra["circuit"] = {
        "task_id": "fixture-task",
        "task_version": "v1",
        "candidate_sha256": "c" * 64,
        "trial_id": "trial-fixture",
        "grade_status": "verified",
        "reward": 1,
    }
    target = tmp_path / "trajectory.json"
    target.write_text(value.model_dump_json())
    return target


def test_sft_keeps_tools_results_and_independent_reward_separate(tmp_path):
    path = trajectory(tmp_path)
    output = tmp_path / "dataset"
    prepare_dataset(
        [{"trajectory": str(path), "split": "train", "task_group": "fixture-family"}],
        output,
        reasoning="exclude",
        allow_reconstructed_context=True,
    )
    row = json.loads((output / "train.jsonl").read_text())
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["messages"][2]["tool_calls"][0]["function"]["arguments"] == {
        "command": "harness-public info"
    }
    assert row["messages"][3]["content"] == "tail only [truncated]"
    assert row["messages"][3]["tool_call_id"] == "c1"
    assert row["extra_info"]["reward"] == 1
    assert "reward" not in json.dumps(row["messages"])
    assert "Inspect the public feedback." not in json.dumps(row["messages"])
    import pyarrow.parquet as pq

    encoded = pq.read_table(output / "train.parquet").to_pylist()[0]
    assert json.loads(encoded["messages_json"]) == row["messages"]
    assert json.loads(encoded["tools_json"]) == row["tools"]


def test_reconstructed_context_requires_explicit_selection(tmp_path):
    path = trajectory(tmp_path)
    with pytest.raises(ValueError, match="acknowledge"):
        prepare_dataset(
            [{"trajectory": str(path), "split": "train", "task_group": "family"}],
            tmp_path / "out",
            reasoning="exclude",
        )


@pytest.mark.parametrize("grade_status,reward", [("unscored", None), ("verified", 0)])
def test_explicit_selection_preserves_unscored_and_zero_reward(tmp_path, grade_status, reward):
    path = trajectory(tmp_path)
    raw = json.loads(path.read_text())
    raw["extra"]["circuit"].update(grade_status=grade_status, reward=reward)
    path.write_text(json.dumps(raw))
    output = tmp_path / "dataset"
    prepare_dataset(
        [{"trajectory": str(path), "split": "train", "task_group": "family"}],
        output,
        reasoning="exclude",
        allow_reconstructed_context=True,
    )
    row = json.loads((output / "train.jsonl").read_text())
    assert row["extra_info"]["grade_status"] == grade_status
    assert row["extra_info"]["reward"] == reward
    assert "grade_status" not in json.dumps(row["messages"])


@pytest.mark.parametrize(
    "damage", ["duplicate", "split_leak", "task_alias", "unscored", "missing_result", "late_task"]
)
def test_invalid_training_selection_fails_before_publishing(tmp_path, damage):
    path = trajectory(tmp_path)
    raw = json.loads(path.read_text())
    entries = [{"trajectory": str(path), "split": "train", "task_group": "family"}]
    other = tmp_path / "other.json"
    if damage in {"split_leak", "task_alias"}:
        raw["session_id"] = "other-session"
        raw["extra"]["circuit"]["trial_id"] = "other-trial"
        entries.append(
            {
                "trajectory": str(other),
                "split": "validation",
                "task_group": "different" if damage == "task_alias" else "family",
            }
        )
    elif damage == "duplicate":
        entries.append(entries[0])
    elif damage == "unscored":
        raw["extra"]["circuit"]["grade_status"] = "unscored"
        entries[0]["trajectory"] = str(other)
    elif damage == "late_task":
        raw["steps"][1:3] = reversed(raw["steps"][1:3])
        for i, step in enumerate(raw["steps"], 1):
            step["step_id"] = i
        entries[0]["trajectory"] = str(other)
    else:
        raw["steps"][2].pop("observation")
        entries[0]["trajectory"] = str(other)
    other.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        prepare_dataset(
            entries, tmp_path / "out", reasoning="exclude", allow_reconstructed_context=True
        )
    assert not (tmp_path / "out").exists()
