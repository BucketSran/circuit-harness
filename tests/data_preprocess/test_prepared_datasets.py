from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from alphaapollo.data_preprocess import (
    MANIFEST_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
    ExistingPolicy,
    OutputFormat,
    canonical_json,
    default_root,
    io,
    open_dataset,
    raw_digest,
    read_manifest,
    read_prepared,
    read_records,
    task_uid,
)
from alphaapollo.data_preprocess.core import (
    BuildRequest,
    NormalizeContext,
    PreparedExample,
    prepare_dataset,
)
from alphaapollo.data_preprocess.prepare_custom_data import prepare as prepare_custom_data
from alphaapollo.data_preprocess.prepare_custom_data import prepare as prepare_fixture_dataset


def test_commit_directory_treats_backup_cleanup_as_post_commit(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    destination = tmp_path / "published"
    staged.mkdir()
    destination.mkdir()
    (staged / "value").write_text("new", encoding="utf-8")
    (destination / "value").write_text("old", encoding="utf-8")

    with patch("shutil.rmtree", side_effect=PermissionError("cleanup denied")):
        io.commit_directory(staged, destination, replace=True)

    assert (destination / "value").read_text(encoding="utf-8") == "new"
    assert not staged.exists()
    backups = tuple(tmp_path.glob(".published.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "value").read_text(encoding="utf-8") == "old"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


def _fixture_rows() -> list[dict]:
    return [
        {
            "problem_idx": 1,
            "problem": "What is 40 + 2?",
            "answer": 42,
            "year": 2026,
        },
        {
            "problem_idx": 2,
            "problem": "What is 6 * 7?",
            "answer": "042",
            "year": 2026,
        },
    ]


def _prepare_fixture_jsonl(tmp_path: Path, *, policy: ExistingPolicy = ExistingPolicy.REUSE):
    source = tmp_path / "fixture.jsonl"
    if not source.exists():
        _write_jsonl(source, _fixture_rows())
    return prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=tmp_path / "prepared",
        dataset_name="fixture_test",
        revision=None,
        splits=("test",),
        output_format=OutputFormat.JSONL,
        if_exists=policy,
    )


def test_default_root_honors_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHAAPOLLO_DATA_DIR", str(tmp_path))
    assert default_root() == tmp_path.resolve()


def test_identity_and_canonical_json_are_stable() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert task_uid("source", "1") == task_uid("source", "1")
    assert task_uid("source", "1") != task_uid("source", "2")


def test_fixture_prepare_separates_public_and_private_rows(tmp_path: Path) -> None:
    prepared = _prepare_fixture_jsonl(tmp_path)

    public = read_records(prepared.public_splits["test"])
    private = read_records(prepared.private_splits["test"])
    assert len(public) == len(private) == 2
    assert "answer" not in public[0]
    assert "raw_digest" not in public[0]
    assert public[0]["statement"] == "What is 40 + 2?"
    assert private[0]["answer"] == "42"
    assert private[1]["answer"] == "042"
    assert public[0]["task_uid"] == private[0]["task_uid"]


def test_read_prepared_joins_by_uid_and_validates_manifest(tmp_path: Path) -> None:
    prepared = _prepare_fixture_jsonl(tmp_path)
    examples = read_prepared("fixture_test", "v1", tmp_path / "prepared", split="test")

    assert [example.source_record_id for example in examples] == ["1", "2"]
    assert [example.answer for example in examples] == ["42", "042"]
    manifest = read_manifest(prepared.manifest_path)
    assert manifest.output_format is OutputFormat.JSONL
    assert manifest.splits["test"].rows == 2
    assert manifest.source_fingerprint


def test_reuse_skips_rebuild_when_source_is_unchanged(tmp_path: Path) -> None:
    first = _prepare_fixture_jsonl(tmp_path)
    first_manifest = first.manifest_path.read_bytes()

    second = _prepare_fixture_jsonl(tmp_path)

    assert second.build_id == first.build_id
    assert second.manifest_path.read_bytes() == first_manifest


def test_changed_local_source_requires_refresh_or_new_version(tmp_path: Path) -> None:
    first = _prepare_fixture_jsonl(tmp_path)
    source = tmp_path / "fixture.jsonl"
    changed = _fixture_rows()
    changed[0]["answer"] = 43
    _write_jsonl(source, changed)

    with pytest.raises(ValueError, match="different spec"):
        _prepare_fixture_jsonl(tmp_path)

    refreshed = _prepare_fixture_jsonl(tmp_path, policy=ExistingPolicy.REFRESH)
    assert refreshed.build_id != first.build_id
    assert read_prepared("fixture_test", "v1", tmp_path / "prepared")[0].answer == "43"


def test_error_policy_rejects_existing_dataset(tmp_path: Path) -> None:
    _prepare_fixture_jsonl(tmp_path)
    with pytest.raises(FileExistsError):
        _prepare_fixture_jsonl(tmp_path, policy=ExistingPolicy.ERROR)


def test_digest_tampering_is_detected(tmp_path: Path) -> None:
    prepared = _prepare_fixture_jsonl(tmp_path)
    prepared.private_splits["test"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        read_prepared("fixture_test", "v1", tmp_path / "prepared")


def test_json_output_is_supported(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "fixture.jsonl", _fixture_rows())
    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=tmp_path / "prepared",
        dataset_name="fixture_json",
        revision=None,
        splits=("test",),
        output_format="json",
    )
    assert prepared.public_splits["test"].suffix == ".json"
    assert len(read_records(prepared.public_splits["test"])) == 2


def test_parquet_is_the_default_output(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = _write_jsonl(tmp_path / "fixture.jsonl", _fixture_rows())
    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=tmp_path / "prepared",
        dataset_name="fixture_parquet",
        revision=None,
        splits=("test",),
    )
    assert prepared.public_splits["test"].suffix == ".parquet"
    assert len(read_records(prepared.public_splits["test"])) == 2


def test_custom_data_uses_explicit_column_mapping(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "custom.jsonl",
        [{"qid": "x", "question": "Q?", "gold": "A", "difficulty": "easy"}],
    )
    prepare_custom_data(
        data_source=str(source),
        dataset_name="custom_test",
        question_key="question",
        answer_key="gold",
        id_key="qid",
        metadata_keys=("difficulty",),
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )
    example = read_prepared("custom_test", "v1", tmp_path / "prepared")[0]
    assert example.statement == "Q?"
    assert example.answer == "A"
    assert example.extra == {"difficulty": "easy"}


def test_custom_metadata_cannot_include_answer_column(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "custom.jsonl", [{"q": "Q", "a": "A"}])
    with pytest.raises(ValueError, match="protected columns"):
        prepare_custom_data(
            data_source=str(source),
            dataset_name="custom_test",
            question_key="q",
            answer_key="a",
            metadata_keys=("a",),
            output_root=tmp_path,
            output_format="jsonl",
        )


@pytest.mark.parametrize("role", ["question_key", "id_key"])
def test_custom_answer_column_cannot_feed_a_public_column(tmp_path: Path, role: str) -> None:
    """`statement` and `source_record_id` are published, so neither may be the answer.

    Reusing the answer column for either one put the gold answer straight into the
    public row, which is the one thing this preparer exists to prevent.
    """

    source = _write_jsonl(tmp_path / "custom.jsonl", [{"q": "Q", "answer": "42"}])
    kwargs = {"question_key": "q", "id_key": None, role: "answer"}

    with pytest.raises(ValueError, match="published in the public row"):
        prepare_custom_data(
            data_source=str(source),
            dataset_name="custom_test",
            answer_key="answer",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
            **kwargs,
        )


def test_custom_multi_split_build_requires_an_id_key(tmp_path: Path) -> None:
    """Fallback ids are per-split row positions, so two splits always collide.

    Without the upfront check this failed deep in the build as an opaque
    ``duplicate task_uid`` naming a hash the caller has no way to trace.
    """

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_jsonl(source_dir / "train.jsonl", [{"q": "Q1", "a": "A1"}])
    _write_jsonl(source_dir / "test.jsonl", [{"q": "Q2", "a": "A2"}])

    with pytest.raises(ValueError, match="id_key is required"):
        prepare_custom_data(
            data_source=str(source_dir),
            dataset_name="custom_test",
            question_key="q",
            answer_key="a",
            splits=("train", "test"),
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )

    prepared = prepare_custom_data(
        data_source=str(source_dir),
        dataset_name="custom_test",
        question_key="q",
        answer_key="a",
        id_key="q",
        splits=("train", "test"),
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )
    assert set(prepared.public_splits) == {"train", "test"}


def test_reserved_private_fields_are_written_and_left_empty(tmp_path: Path) -> None:
    """`env_payload` and `reference_solution` are reserved, not forgotten.

    No shipped preparer fills them, but they are part of the persisted private
    row so the Workflow Environment channel and a future SFT path have a place
    to plug in. Removing them as dead fields would be a schema break.
    """

    prepared = _prepare_fixture_jsonl(tmp_path)

    private = read_records(prepared.private_splits["test"])
    assert private[0]["env_payload"] == "{}"
    assert private[0]["reference_solution"] == ""
    example = read_prepared("fixture_test", "v1", tmp_path / "prepared", split="test")[0]
    assert example.env_payload == {}
    assert example.reference_solution == ""


def test_raw_digest_is_recorded_but_never_verified(tmp_path: Path) -> None:
    """`raw_digest` is forensic evidence, not an enforced check — and it stays.

    It is the SHA-256 of the source row, written on every private row and read by
    nothing. That is deliberate: a prepared build keeps the digest but not the row
    it came from, so verifying it would mean re-fetching the source, which is just
    a rebuild. There is no cheap check to add, and adding one is not the fix.

    The field earns its place after a suspicion, not before one: if an upstream
    dataset silently edits rows while keeping the same revision — the one failure
    `source_revision` and `build_id` cannot see — you rebuild, diff this column,
    and the rows whose digest moved are named exactly. Deleting it as an unread
    field, the way `answer_space` was deleted, would remove that evidence.

    The contrast is the point, so this test pins both halves: corrupting a row's
    `raw_digest` (with the manifest kept consistent) reads back clean, while
    corrupting the split file itself is caught by the manifest's file digest.
    """

    prepared = _prepare_fixture_jsonl(tmp_path)
    root = tmp_path / "prepared"
    private_path = prepared.private_splits["test"]

    # It is a real digest of the real source row, not a placeholder.
    rows = read_records(private_path)
    assert [row["raw_digest"] for row in rows] == [raw_digest(raw) for raw in _fixture_rows()]

    # Corrupt one row digest, then re-stamp the manifest so the *file* digest
    # still matches. Nothing else looks at the row digest, so the read succeeds.
    rows[0]["raw_digest"] = "0" * 64
    _write_jsonl(private_path, rows)
    manifest = read_manifest(prepared.manifest_path)
    split = manifest.splits["test"]
    io.write_manifest(
        manifest.model_copy(
            update={
                "splits": {
                    "test": split.model_copy(
                        update={"private_sha256": io.file_sha256(private_path)}
                    )
                }
            }
        ),
        prepared.manifest_path,
    )

    examples = read_prepared("fixture_test", "v1", root, split="test")
    assert examples[0].raw_digest == "0" * 64
    assert examples[0].answer == "42"

    # The file digest, by contrast, is enforced on every open.
    private_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_prepared("fixture_test", "v1", root, split="test")


def test_a_hub_source_records_the_commit_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller should not have to paste a SHA to get a reproducible build.

    The commit is resolved at build time and written to the manifest, so a
    finished build still names exactly what it read.
    """

    import alphaapollo.data_preprocess.prepare_custom_data as module

    resolved = "0" * 40
    monkeypatch.setattr(module, "resolve_hub_revision", lambda _source: resolved)
    monkeypatch.setattr(
        io,
        "load_source",
        lambda data_source, *, revision, splits: {splits[0]: _fixture_rows()},
    )
    monkeypatch.setattr(io, "source_fingerprint", lambda *_args, **_kwargs: resolved)

    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source="owner/repository",
        revision=None,
        output_root=tmp_path,
        dataset_name="hub_test",
        splits=("test",),
        output_format="jsonl",
    )

    assert read_manifest(prepared.manifest_path).source_revision == resolved
    assert read_prepared("hub_test", "v1", tmp_path)[0].source_revision == resolved


def test_a_local_source_needs_no_revision(tmp_path: Path) -> None:
    prepared = _prepare_fixture_jsonl(tmp_path)

    assert read_manifest(prepared.manifest_path).source_revision is None


@pytest.mark.parametrize("preparer", ["fixture", "custom"])
def test_a_local_source_refuses_a_revision(tmp_path: Path, preparer: str) -> None:
    """A revision on a local build names a commit the build never read.

    It is not ignored: it reaches `manifest.source_revision`, every private row's
    `source_revision`, and `build_id`. Accepting it would let a build claim
    provenance for a commit whose bytes it may not contain.
    """

    source = _write_jsonl(tmp_path / "fixture.jsonl", _fixture_rows())
    kwargs: dict[str, object] = {
        "data_source": str(source),
        "output_root": tmp_path / "prepared",
        "revision": "0" * 40,
        "splits": ("test",),
        "output_format": OutputFormat.JSONL,
    }
    with pytest.raises(ValueError, match="a local file has no Hub commit"):
        if preparer == "fixture":
            prepare_fixture_dataset(
                question_key="problem",
                answer_key="answer",
                id_key="problem_idx",
                dataset_name="fixture_test",
                **kwargs,
            )  # type: ignore[arg-type]
        else:
            prepare_custom_data(  # type: ignore[arg-type]
                dataset_name="custom_test",
                question_key="problem",
                answer_key="answer",
                **kwargs,
            )

    assert not (tmp_path / "prepared").exists()


def test_an_existing_build_is_refused_before_the_source_is_read(tmp_path: Path) -> None:
    """`if_exists='error'` is answerable from the destination alone.

    Fingerprinting first would hash every split file for a build that was never
    allowed to be published.
    """

    _prepare_fixture_jsonl(tmp_path)
    calls: list[str] = []
    original = io.source_fingerprint

    def _counting(*args: object, **kwargs: object) -> str:
        calls.append("fingerprint")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    io.source_fingerprint = _counting  # type: ignore[assignment]
    try:
        with pytest.raises(FileExistsError):
            _prepare_fixture_jsonl(tmp_path, policy=ExistingPolicy.ERROR)
    finally:
        io.source_fingerprint = original  # type: ignore[assignment]

    assert calls == []


def test_a_malformed_prepared_row_names_the_record_it_came_from(tmp_path: Path) -> None:
    """A schema mismatch must be traceable to one row, not just to a field name."""

    prepared = _prepare_fixture_jsonl(tmp_path)
    private_path = prepared.private_splits["test"]
    rows = read_records(private_path)
    rows[1]["answer_space"] = "integer"
    _write_jsonl(private_path, rows)
    manifest = read_manifest(prepared.manifest_path)
    split = manifest.splits["test"]
    io.write_manifest(
        manifest.model_copy(
            update={
                "splits": {
                    "test": split.model_copy(
                        update={"private_sha256": io.file_sha256(private_path)}
                    )
                }
            }
        ),
        prepared.manifest_path,
    )

    with pytest.raises(ValueError) as failure:
        read_prepared("fixture_test", "v1", tmp_path / "prepared")

    message = str(failure.value)
    assert "row 1" in message
    assert "'test'" in message
    assert rows[1]["task_uid"] in message
    assert "answer_space" in message


def test_a_malformed_manifest_names_the_file(tmp_path: Path) -> None:
    prepared = _prepare_fixture_jsonl(tmp_path)
    prepared.manifest_path.write_text('{"dataset_name": ', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed prepared dataset manifest") as failure:
        read_prepared("fixture_test", "v1", tmp_path / "prepared")

    assert str(prepared.manifest_path) in str(failure.value)


def test_custom_data_also_requires_the_dataset_identity(tmp_path: Path) -> None:
    """`prepare_custom_data` must not default the name its sibling demands.

    The name is the output directory and the `source_id` hashed into every
    `task_uid`, so a default let two unrelated builds land on one prepared path
    and claim each other's identity.
    """

    source = _write_jsonl(tmp_path / "custom.jsonl", [{"q": "Q", "a": "A"}])

    with pytest.raises(TypeError, match="dataset_name"):
        prepare_custom_data(  # type: ignore[call-arg]
            data_source=str(source),
            question_key="q",
            answer_key="a",
            output_root=tmp_path / "prepared",
        )


def test_the_custom_data_cli_refuses_to_guess_the_dataset(tmp_path: Path) -> None:
    """The refusal lands in argument parsing, before any source is read."""

    from alphaapollo.data_preprocess.prepare_custom_data import main

    source = _write_jsonl(tmp_path / "custom.jsonl", [{"q": "Q", "a": "A"}])
    argv = [
        "--data-source",
        str(source),
        "--question-key",
        "q",
        "--answer-key",
        "a",
        "--output-root",
        str(tmp_path / "prepared"),
    ]

    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2
    assert not (tmp_path / "prepared").exists()

    assert main([*argv, "--dataset-name", "custom_test", "--format", "jsonl"]) == 0
    assert read_prepared("custom_test", "v1", tmp_path / "prepared")[0].answer == "A"


@pytest.mark.parametrize("output_format", ["parquet", "jsonl"])
def test_a_build_round_trips_through_open_dataset(tmp_path: Path, output_format: str) -> None:
    """prepare -> open_dataset -> read rows, in the shipped default and in JSONL."""

    if output_format == "parquet":
        pytest.importorskip("pyarrow")
    source = _write_jsonl(tmp_path / "fixture.jsonl", _fixture_rows())

    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=tmp_path / "prepared",
        dataset_name="fixture_round_trip",
        revision=None,
        splits=("test",),
        output_format=output_format,
    )

    reopened = open_dataset(prepared.root)
    assert reopened.build_id == prepared.build_id
    manifest = read_manifest(reopened.manifest_path)
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.row_schema_version == ROW_SCHEMA_VERSION
    assert len(read_records(reopened.public_splits["test"])) == 2

    examples = read_prepared("fixture_round_trip", "v1", tmp_path / "prepared", split="test")
    assert [example.answer for example in examples] == ["42", "042"]


def test_a_stale_row_schema_version_is_refused(tmp_path: Path) -> None:
    """A build's rows must match the code about to read them, or it is refused.

    `build_id` folds `ROW_SCHEMA_VERSION` in, but a reader holding a prepared
    directory cannot recompute the fingerprint without the source it no longer
    has. The recorded version is what makes the check answerable at open time,
    in the same place the file digests are verified.
    """

    prepared = _prepare_fixture_jsonl(tmp_path)
    manifest = read_manifest(prepared.manifest_path)
    stale = ROW_SCHEMA_VERSION - 1
    io.write_manifest(
        manifest.model_copy(update={"row_schema_version": stale}),
        prepared.manifest_path,
    )

    with pytest.raises(ValueError) as failure:
        read_prepared("fixture_test", "v1", tmp_path / "prepared")

    message = str(failure.value)
    assert f"row schema version {stale}" in message
    assert f"row schema version {ROW_SCHEMA_VERSION}" in message
    assert "rebuild" in message

    # The refusal never blocks the rebuild it asks for: `refresh` replaces the
    # stale directory without opening it first.
    refreshed = _prepare_fixture_jsonl(tmp_path, policy=ExistingPolicy.REFRESH)
    assert read_manifest(refreshed.manifest_path).row_schema_version == ROW_SCHEMA_VERSION


def test_a_manifest_predating_the_row_schema_version_is_refused(tmp_path: Path) -> None:
    """Every build prepared before the field existed must be rebuilt, deliberately.

    Such a manifest cannot state which row shape its files hold, so nothing can
    establish that they match the current code. Prepared builds are rebuilt from
    a recorded source in seconds and nothing outside this repository consumes
    one, so refusing by `schema_version` is preferred to guessing -- and it is a
    stated refusal, not a missing-key crash from the model.
    """

    prepared = _prepare_fixture_jsonl(tmp_path)
    legacy = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    legacy.pop("row_schema_version")
    legacy["schema_version"] = 1
    prepared.manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError) as failure:
        read_prepared("fixture_test", "v1", tmp_path / "prepared")

    message = str(failure.value)
    assert "schema_version 1" in message
    assert f"schema_version {MANIFEST_SCHEMA_VERSION}" in message
    assert str(prepared.manifest_path) in message


def test_reader_api_accepts_plain_string_paths(tmp_path: Path) -> None:
    """These are the package's public entry points; a str path is the obvious call."""

    prepared = _prepare_fixture_jsonl(tmp_path)

    assert read_records(str(prepared.public_splits["test"]))
    assert read_manifest(str(prepared.manifest_path)).build_id
    assert read_prepared("fixture_test", "v1", str(tmp_path / "prepared"))


def _generated_example(raw: dict, context: NormalizeContext) -> PreparedExample:
    record_id = str(raw["id"])
    return PreparedExample(
        task_uid=task_uid(context.source_id, record_id),
        source_id=context.source_id,
        source_record_id=record_id,
        source_order=context.source_order,
        split=context.split,
        statement=str(raw["statement"]),
        answer="42",
        grader_id="exact_match",
        source_uri="fixture://generated",
        source_revision="fixture-revision",
        dataset_license="unverified",
        raw_digest=raw_digest(raw),
        builder_version="1",
    )


def _prepare_generated(tmp_path: Path, *, data_source: str, build_options: dict | None = None):
    request = BuildRequest(
        dataset_name="generated",
        dataset_version="v1",
        data_source=data_source,
        revision=None,
        splits=("test",),
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
        if_exists=ExistingPolicy.REUSE,
    )
    return prepare_dataset(
        request,
        _generated_example,
        builder_name="fixture",
        builder_version="1",
        build_options=build_options,
        source_rows={"test": [{"id": "one", "statement": "What is 40 + 2?"}]},
    )


def test_generated_rows_refuse_a_data_source_the_build_never_read(tmp_path: Path) -> None:
    """`source_rows` skips both io helpers, so `data_source` must name the generator.

    Neither `io.source_fingerprint` nor `io.load_source` runs on this path. A
    file path would therefore be recorded in the manifest as the source of bytes
    nothing opened, and a bare Hub repo id would reach the manifest without the
    exact-revision guard both helpers enforce.
    """

    unread = _write_jsonl(tmp_path / "unread.jsonl", _fixture_rows())

    with pytest.raises(ValueError) as local_failure:
        _prepare_generated(tmp_path, data_source=str(unread))
    assert str(unread) in str(local_failure.value)
    assert "never read" in str(local_failure.value)

    with pytest.raises(ValueError) as hub_failure:
        _prepare_generated(tmp_path, data_source="an-org/a-dataset")
    assert "an-org/a-dataset" in str(hub_failure.value)
    assert "scheme-qualified pseudo-URI" in str(hub_failure.value)

    prepared = _prepare_generated(tmp_path, data_source="fixture://generated/one")
    assert read_manifest(prepared.manifest_path).data_source == "fixture://generated/one"
