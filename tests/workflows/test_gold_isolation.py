"""The gold answer must never reach the process that runs the model.

`ScoredTask` carries the gold. It is bound *after* execution is planned and is
only ever read by the scorer -- except in one place: `scoring.run_cell` builds a
`WorkflowInput` out of it when no public input was injected, through the task's
own `public_view` projection. These tests pin that seam from both ends -- what
`run_cell` is allowed to read, and what `public_view` is allowed to emit --
because nothing in the type system prevents a future field from carrying gold
across it.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from alphaapollo.workflows import data as data_module
from alphaapollo.workflows import scoring as scoring_module
from alphaapollo.workflows.config import ConfigError, parse_run_config
from alphaapollo.workflows.data import align_public_inputs_with_private_gold
from alphaapollo.workflows.records import ScoredTask, WorkflowInput

_GOLD = "31337"


def _prepared(
    tmp_path: Path,
    *,
    name: str = "isolation",
    columns: Mapping[str, object] | None = None,
) -> Path:
    """Build a two-row dataset whose gold is a string found nowhere else.

    ``columns`` are merged into every raw row. ``prepare_custom_data`` routes any column
    it does not map into the private ``extra`` blob, which is how a test can put
    something gold-adjacent on the private side of the boundary.
    """

    import json

    from alphaapollo.data_preprocess.prepare_custom_data import prepare as prepare_custom_data

    source = tmp_path / f"{name}.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "problem_idx": index,
                    "problem": f"Problem {index}?",
                    "answer": answer,
                    **dict(columns or {}),
                }
            )
            + "\n"
            for index, answer in ((1, 313), (2, 62))
        ),
        encoding="utf-8",
    )
    prepare_custom_data(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        metadata_keys=tuple(columns or {}),
        data_source=str(source),
        dataset_name=name,
        output_root=tmp_path / "prepared",
        revision=None,
        splits=("test",),
        output_format="jsonl",
    )
    return tmp_path / "prepared"


def _config(tmp_path: Path, root: Path, *, name: str = "isolation") -> object:
    public = root / name / "v1" / "public" / "test.jsonl"
    return parse_run_config(
        {
            "version": 1,
            "workflow": {
                "version": 1,
                "name": "isolation",
                "roles": [{"id": "solver", "target": "solver", "system_prompt": "Solve."}],
                "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
                "entry_step": "solve",
                "transitions": [],
            },
            "dataset": {
                "path": str(public),
                "format": "jsonl",
                "split": "test",
                "input_key": "statement",
                "id_key": "task_uid",
            },
            "runtimes": {
                "solver": {
                    "type": "alphaapollo",
                    "options": {
                        "backend": {"type": "fake", "options": {"text": "x"}},
                        "model": "offline",
                        "sampling": {"temperature": 0.0, "max_tokens": 8},
                        "max_turns": 1,
                    },
                }
            },
            "verifiers": {},
            "environment": {"type": "text_only", "options": {}},
            "output": {"directory": str(tmp_path / "out")},
            "scoring": {
                "dataset_root": str(root),
                "source_id": name,
                "version": "v1",
            },
        }
    )


def test_binding_keeps_public_text_and_drops_the_private_source(tmp_path: Path) -> None:
    """The rebuilt task must carry the public statement, not the private one.

    They are equal here by construction, but the rebuild is what guarantees the
    fallback in `run_cell` can only surface public text.
    """

    root = _prepared(tmp_path)
    config = _config(tmp_path, root)
    inputs = data_module.load_prepared_inputs(config)

    aligned = align_public_inputs_with_private_gold(config, inputs)

    assert len(aligned) == 2
    for task, item in aligned:
        assert task.problem == item.problem
        assert task.gold_answer
        # ``source_uri`` is the private row's provenance pointer; the scored task
        # no longer has a field for it, and nothing may smuggle it into metadata.
        assert not any("isolation.jsonl" in value for value in task.public_view().values()), (
            "private source_uri must not survive into the scored task's public view"
        )


def test_no_gold_reaches_a_workflow_input_built_from_a_scored_task() -> None:
    """`run_cell`'s fallback constructs the model's input from the gold-bearing task."""

    task = ScoredTask(id="p1", problem="Compute 1+1.", gold_answer=_GOLD, metadata={"year": "2026"})
    workflow_input = WorkflowInput(
        input_id="s0",
        problem=task.problem,
        metadata=task.public_view(),
    )

    serialized = repr(workflow_input)
    assert _GOLD not in serialized
    assert "gold" not in {key.lower() for key in workflow_input.metadata}


def test_the_public_projection_is_the_only_thing_the_fallback_crosses() -> None:
    """Guard the seam itself, not one instance of it.

    ``run_cell`` may read the gold-bearing task only through ``public_view``,
    and ``public_view`` may emit only the identifier plus the keys the binding
    allowlisted. Pinning both halves means a new field on ``ScoredTask`` cannot
    reach the model without failing here first, where the reason is written down.
    """

    source = inspect.getsource(scoring_module.run_cell)
    start = source.index("workflow_input = (")
    end = source.index(")", source.index("metadata=", start))
    fallback = source[start:end]

    read_attributes = {
        line.split("cell.problem.")[1].split(",")[0].split(")")[0].split("(")[0].strip()
        for line in fallback.splitlines()
        if "cell.problem." in line
    }
    assert read_attributes <= {"problem", "public_view"}, (
        f"run_cell reads {sorted(read_attributes)} from the gold-bearing ScoredTask; "
        "the public projection is ScoredTask.public_view, and picking fields off the "
        "task here moves the boundary rule away from the class that owns the gold"
    )

    task = ScoredTask(
        id="p1",
        problem="Compute 1+1.",
        gold_answer=_GOLD,
        metadata={"year": "2026"},
    )
    allowed = {"problem_id"} | set(data_module._PUBLIC_TASK_METADATA_KEYS)
    assert set(task.public_view()) <= allowed, (
        f"public_view emits {sorted(set(task.public_view()) - allowed)}, which reaches the "
        "model's prompt without having been allowlisted at the binding"
    )
    assert task.public_view() == {"problem_id": "p1", "year": "2026"}


def test_public_view_never_carries_the_gold_answer() -> None:
    """The one field that must never cross, checked by value and not by name."""

    task = ScoredTask(
        id="p1",
        problem="Compute 1+1.",
        gold_answer=_GOLD,
        metadata={"year": "2026"},
    )

    view = task.public_view()

    assert _GOLD not in view.values()
    assert _GOLD not in repr(view)
    assert "gold_answer" not in view
    assert not any("gold" in key.lower() or "answer" in key.lower() for key in view)


def test_a_dataset_without_a_year_yields_no_empty_metadata(tmp_path: Path) -> None:
    """A non-AIME row has no ``year``; the model must not be told ``year=""``.

    The dropped ``year`` field defaulted to the empty string and was handed to
    the model regardless. A mapping populated from the prepared row's ``extra``
    simply has no key when the dataset has no such column.
    """

    root = _prepared(tmp_path, name="noyear")
    config = _config(tmp_path, root, name="noyear")
    inputs = data_module.load_prepared_inputs(config)

    aligned = align_public_inputs_with_private_gold(config, inputs)

    assert aligned
    for task, _item in aligned:
        assert task.metadata == {}
        assert task.public_view() == {"problem_id": task.id}
        assert "" not in task.public_view().values()


def test_unmapped_private_columns_do_not_reach_the_public_view(tmp_path: Path) -> None:
    """``extra`` is a private catch-all, so only allowlisted keys may cross.

    ``prepare_custom_data`` routes explicitly selected metadata columns into ``extra``. A
    source row carrying a worked ``solution`` therefore parks the answer on the
    private side; copying ``extra`` wholesale would put it in the prompt.
    """

    root = _prepared(
        tmp_path,
        name="fixture2026",
        columns={"solution": f"The answer is {_GOLD}.", "notes": "internal", "year": "2026"},
    )
    config = _config(tmp_path, root, name="fixture2026")
    inputs = data_module.load_prepared_inputs(config)

    aligned = align_public_inputs_with_private_gold(config, inputs)

    assert aligned
    for task, _item in aligned:
        # The allowlisted key still crosses, unchanged, from explicit metadata.
        assert task.public_view() == {"problem_id": task.id, "year": "2026"}
        assert _GOLD not in repr(task.public_view())
        assert "solution" not in task.metadata
        assert "notes" not in task.metadata


def test_scoring_is_refused_without_a_prepared_private_projection(tmp_path: Path) -> None:
    root = _prepared(tmp_path)
    config = _config(tmp_path, root)
    inputs = data_module.load_prepared_inputs(config)
    without_scoring = replace(config, scoring=None)

    with pytest.raises(ConfigError, match="requires a private prepared dataset"):
        align_public_inputs_with_private_gold(without_scoring, inputs)
