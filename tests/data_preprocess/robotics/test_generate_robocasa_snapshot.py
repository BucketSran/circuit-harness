"""Generator contracts: output location, seed derivation, canonical rows.

The paths that reach the simulator are driven through a stub of the four
upstream names the generator imports (see ``_Runtime`` below), so the whole
module is covered without robocasa installed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from alphaapollo.data_preprocess.robotics import generate_robocasa_snapshot as gen
from alphaapollo.data_preprocess.robotics.prepare_robocasa import SNAPSHOT_PATH


def _committed_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_default_output_is_the_committed_snapshot_path() -> None:
    """One owner: a regeneration must overwrite the committed file, not a copy.

    The generator used to recompute the path one directory off, leaving a
    stageable stray file while the committed snapshot stayed untouched
    (review finding on #256). ``snapshots/`` also has to stay inside the
    robotics package, so the two modules that own the data reach it without
    climbing out of their own package.
    """

    assert gen.DEFAULT_OUTPUT == SNAPSHOT_PATH
    assert gen.DEFAULT_OUTPUT.name == "robocasa365_tasks.jsonl"
    assert gen.DEFAULT_OUTPUT.parent.name == "snapshots"
    assert gen.DEFAULT_OUTPUT.parent.parent == Path(gen.__file__).resolve().parent
    assert gen.DEFAULT_OUTPUT.is_file()


def test_task_seed_is_stable_and_task_sensitive() -> None:
    first = gen._task_seed(42, "AddIceCubes")
    assert first == gen._task_seed(42, "AddIceCubes")
    assert first != gen._task_seed(42, "AddMarshmallow")
    assert first != gen._task_seed(43, "AddIceCubes")
    assert 0 <= first < 2**31


def test_canonical_row_is_sorted_and_lossless_round_trip() -> None:
    row = {
        "id": "x-000",
        "task_name": "T",
        "instruction": "do it",
        "layout_id": 2,
        "style_id": 3,
        "scene_seed": 4,
        "horizon": 9,
    }
    rendered = gen._canonical_row(row)
    assert rendered == json.dumps(row, sort_keys=True, ensure_ascii=False)
    assert json.loads(rendered) == row


def test_known_unbuildable_covers_the_base_classes() -> None:
    assert gen.KNOWN_UNBUILDABLE == frozenset(
        {"ManipulateLowerDoor", "OpenDropDownDoor", "CloseDropDownDoor", "Kitchen"}
    )


def test_snapshot_horizons_are_task_specific() -> None:
    """The env constructor's horizon is a task-agnostic default; a constant
    horizon column means the official registry lookup was lost (#256).
    """

    assert len({row["horizon"] for row in _committed_rows()}) > 1


def test_meta_sidecar_describes_the_committed_manifest() -> None:
    meta = json.loads(SNAPSHOT_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    rows = _committed_rows()

    assert "derived" in meta["artifact"] and "not the official" in meta["artifact"]
    # The checkout the committed rows were built on, which is also what
    # UPSTREAM.md names and what the preparer's environment_version states.
    # A regeneration resolves these per run, so this pins the artifact, not
    # a generator constant.
    assert meta["upstream_pins"] == {"robocasa": "921c9a5", "robosuite": "5ce6643"}
    assert meta["protocol"]["generator_seed"] == gen.GENERATOR_SEED
    assert meta["row_count"] == len(rows)
    env_default = meta["horizon_env_default"]
    assert env_default and env_default == sorted(env_default)
    by_task = {row["task_name"]: row for row in rows}
    for task_name in env_default:
        # No official registry entry: the env constructor default applies.
        assert by_task[task_name]["horizon"] == 1000


def test_write_meta_records_the_run_it_was_given(tmp_path: Path) -> None:
    """The sidecar describes this run, not the module's defaults.

    It used to hardcode ``GENERATOR_SEED`` and a pair of pinned shas, so
    ``--seed 7`` produced rows the sidecar's own documented seed rule could
    not reproduce from the seed the sidecar recorded (#256 review).
    """

    meta_path = tmp_path / "snap.meta.json"

    gen._write_meta(
        meta_path,
        [{"task_name": "Beta"}, {"task_name": "Alpha"}],
        seed=7,
        upstream_pins={"robocasa": "1.0.1+abc1234", "robosuite": gen.UNRESOLVED_PIN},
        env_default_tasks=["Beta"],
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["row_count"] == 2
    assert meta["horizon_env_default"] == ["Beta"]
    assert meta["protocol"]["generator_seed"] == 7
    assert meta["upstream_pins"] == {
        "robocasa": "1.0.1+abc1234",
        "robosuite": gen.UNRESOLVED_PIN,
    }
    for key in ("artifact", "protocol", "upstream_pins", "procedure", "horizon_source"):
        assert meta[key]


# --- robocasa runtime stub -------------------------------------------------
#
# CONSTRUCTED, not captured: it reproduces the shape of the four upstream
# names the generator calls, not MuJoCo's physics. That is enough to drive
# every generator path -- the registry lookup, the two-phase build, the
# truncation refusal, --check, and main() -- on a machine with no simulator,
# and it is all any of them read. It does not cover whether a real scene
# builds or what a real ep_meta says; `--check` against the runtime is what
# covers that.

_OBJECTS = ("mug", "kettle", "bowl")


class _FakeEnv:
    """The subset of a robocasa env the generator touches."""

    def __init__(self, env_name: str, layout_id: int, style_id: int, seed: int) -> None:
        self.env_name = env_name
        self.layout_id = layout_id
        self.style_id = style_id
        self.seed = seed
        self.horizon = 500
        self.closed = False

    def reset(self) -> None:
        return None

    def get_ep_meta(self) -> dict[str, str]:
        if self.env_name == "SilentTask":
            return {"lang": "   "}
        return {"lang": f"move the {_OBJECTS[self.seed % len(_OBJECTS)]} for {self.env_name}"}

    def close(self) -> None:
        self.closed = True


class _Runtime:
    """Records every ``create_env`` call so the two phases stay observable."""

    def __init__(self, tasks: Sequence[str], *, failing: Sequence[str] = ()) -> None:
        self.tasks = {name: object() for name in tasks}
        self.failing = set(failing)
        self.calls: list[dict] = []
        self.envs: list[_FakeEnv] = []
        self.atomic = {"AlphaTask": {"horizon": 350}}
        self.composite = {"BetaTask": {"horizon": 900}}

    def create_env(
        self,
        *,
        env_name: str,
        seed: int,
        render_onscreen: bool,
        layout_and_style_ids=None,
    ) -> _FakeEnv:
        self.calls.append(
            {"env_name": env_name, "seed": seed, "layout_and_style_ids": layout_and_style_ids}
        )
        if env_name in self.failing:
            raise RuntimeError(f"{env_name} is not instantiable")
        if layout_and_style_ids is None:
            layout_id, style_id = seed % 9 + 1, seed % 11 + 1
        else:
            (layout_id, style_id) = layout_and_style_ids[0]
        env = _FakeEnv(env_name, layout_id, style_id, seed)
        self.envs.append(env)
        return env


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> _Runtime:
    stub = _Runtime(["AlphaTask", "BetaTask", "GammaTask", "Kitchen"], failing=["Kitchen"])
    package = ModuleType("robocasa")
    package.__path__ = []  # type: ignore[attr-defined]
    package.__version__ = "1.0.1"  # type: ignore[attr-defined]
    utils = ModuleType("robocasa.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    environments = ModuleType("robocasa.environments")
    environments.ALL_KITCHEN_ENVIRONMENTS = stub.tasks  # type: ignore[attr-defined]
    env_utils = ModuleType("robocasa.utils.env_utils")
    env_utils.create_env = stub.create_env  # type: ignore[attr-defined]
    dataset_registry = ModuleType("robocasa.utils.dataset_registry")
    dataset_registry.ATOMIC_TASK_DATASETS = stub.atomic  # type: ignore[attr-defined]
    dataset_registry.COMPOSITE_TASK_DATASETS = stub.composite  # type: ignore[attr-defined]
    for name, module in (
        ("robocasa", package),
        ("robocasa.utils", utils),
        ("robocasa.environments", environments),
        ("robocasa.utils.env_utils", env_utils),
        ("robocasa.utils.dataset_registry", dataset_registry),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return stub


def test_official_horizon_prefers_the_registry_over_the_env_default(runtime: _Runtime) -> None:
    assert gen._official_horizon("AlphaTask") == 350
    assert gen._official_horizon("BetaTask") == 900
    # No entry in either registry: the caller falls back to the env default.
    assert gen._official_horizon("GammaTask") is None


def test_replay_row_pins_the_pair_and_records_that_build(runtime: _Runtime) -> None:
    row = gen._replay_row("AlphaTask", 3, 7, 11)

    assert runtime.calls == [
        {"env_name": "AlphaTask", "seed": 11, "layout_and_style_ids": [(3, 7)]}
    ]
    assert row == {
        "id": "AlphaTask-000",
        "task_name": "AlphaTask",
        "layout_id": 3,
        "style_id": 7,
        "scene_seed": 11,
        "instruction": "move the bowl for AlphaTask",
        "horizon": 350,
    }
    assert all(env.closed for env in runtime.envs)


def test_replay_row_falls_back_to_the_env_horizon(runtime: _Runtime) -> None:
    assert gen._replay_row("GammaTask", 1, 1, 4)["horizon"] == 500


def test_replay_row_refuses_an_empty_instruction(runtime: _Runtime) -> None:
    runtime.tasks["SilentTask"] = object()
    with pytest.raises(ValueError, match="empty language instruction"):
        gen._replay_row("SilentTask", 1, 1, 1)
    assert all(env.closed for env in runtime.envs)


def test_build_row_samples_the_pair_then_replays_it(runtime: _Runtime) -> None:
    row = gen._build_row("AlphaTask", 20)

    # Phase one is unpinned; phase two replays the sampled pair verbatim.
    assert runtime.calls[0]["layout_and_style_ids"] is None
    assert runtime.calls[1]["layout_and_style_ids"] == [(20 % 9 + 1, 20 % 11 + 1)]
    assert (row["layout_id"], row["style_id"]) == (20 % 9 + 1, 20 % 11 + 1)


def test_generate_rows_covers_the_registry_and_skips_known_unbuildables(
    runtime: _Runtime,
) -> None:
    rows = gen.generate_rows(seed=7)

    assert [row["task_name"] for row in rows] == ["AlphaTask", "BetaTask", "GammaTask"]
    assert all(row["scene_seed"] == gen._task_seed(7, row["task_name"]) for row in rows)


def test_generate_rows_honors_the_limit(runtime: _Runtime) -> None:
    assert [row["task_name"] for row in gen.generate_rows(limit=2)] == ["AlphaTask", "BetaTask"]


def test_generate_rows_refuses_a_truncated_suite(runtime: _Runtime) -> None:
    """One unexpected failure must void the run, not shrink the suite."""

    runtime.failing.add("BetaTask")
    with pytest.raises(RuntimeError, match="refusing to write a truncated suite"):
        gen.generate_rows()


def test_check_accepts_rows_that_reconstruct(runtime: _Runtime, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.jsonl"
    rows = gen.generate_rows()
    snapshot.write_text("".join(f"{gen._canonical_row(row)}\n" for row in rows), encoding="utf-8")

    assert gen._check(snapshot) == 0


def test_check_refuses_an_edited_row(runtime: _Runtime, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.jsonl"
    rows = gen.generate_rows()
    rows[0]["instruction"] = "something the runtime never said"
    snapshot.write_text("".join(f"{gen._canonical_row(row)}\n" for row in rows), encoding="utf-8")

    assert gen._check(snapshot) == 1


def test_check_refuses_incomplete_task_coverage(runtime: _Runtime, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.jsonl"
    rows = gen.generate_rows()[:-1]
    snapshot.write_text("".join(f"{gen._canonical_row(row)}\n" for row in rows), encoding="utf-8")

    assert gen._check(snapshot) == 1


def test_check_refuses_an_empty_snapshot(runtime: _Runtime, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.jsonl"
    snapshot.write_text("\n", encoding="utf-8")

    assert gen._check(snapshot) == 1


def test_main_writes_the_snapshot_and_its_sidecar(runtime: _Runtime, tmp_path: Path) -> None:
    output = tmp_path / "out" / "snap.jsonl"

    assert gen.main(["--output", str(output)]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_name"] for row in rows] == ["AlphaTask", "BetaTask", "GammaTask"]
    meta = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["row_count"] == 3
    # GammaTask has no registry entry, so it keeps the env constructor default.
    assert meta["horizon_env_default"] == ["GammaTask"]
    assert meta["protocol"]["generator_seed"] == gen.GENERATOR_SEED
    # Resolved from what this run imported: the stub declares a version, and
    # robosuite is not installed here at all.
    assert meta["upstream_pins"] == {"robocasa": "1.0.1", "robosuite": gen.UNRESOLVED_PIN}


def test_main_records_the_seed_it_was_given(runtime: _Runtime, tmp_path: Path) -> None:
    """A non-default seed must reach the sidecar, or it documents another run."""

    output = tmp_path / "out" / "snap.jsonl"

    assert gen.main(["--seed", "7", "--output", str(output)]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["protocol"]["generator_seed"] == 7
    # The sidecar's own documented rule, applied to its own recorded seed,
    # reproduces the seeds the rows were built from.
    assert all(row["scene_seed"] == gen._task_seed(7, row["task_name"]) for row in rows)


def test_resolved_pin_reports_unresolved_rather_than_a_stale_guess() -> None:
    assert gen._resolved_pin("no_such_upstream_package") == gen.UNRESOLVED_PIN


@pytest.mark.parametrize(
    "argv", [["--limit", "2"], ["--seed", "7"], ["--seed", "7", "--limit", "1"]]
)
def test_main_refuses_to_write_the_committed_snapshot_from_a_partial_run(
    runtime: _Runtime, argv: list[str]
) -> None:
    """--output defaults to the committed suite, so a smoke run used to
    replace the vendored 370 rows in the working tree with no prompt (#256).
    """

    before = SNAPSHOT_PATH.read_bytes()
    sidecar = SNAPSHOT_PATH.with_suffix(".meta.json")
    sidecar_before = sidecar.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        gen.main(argv)

    assert excinfo.value.code == 2
    assert SNAPSHOT_PATH.read_bytes() == before
    assert sidecar.read_bytes() == sidecar_before
    # Refused before anything was built, not after an hour of scenes.
    assert runtime.calls == []


def test_main_accepts_a_partial_run_that_names_its_own_output(
    runtime: _Runtime, tmp_path: Path
) -> None:
    output = tmp_path / "smoke.jsonl"

    assert gen.main(["--limit", "1", "--seed", "7", "--output", str(output)]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_name"] for row in rows] == ["AlphaTask"]


def test_main_checks_instead_of_generating(runtime: _Runtime, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.jsonl"
    rows = gen.generate_rows()
    snapshot.write_text("".join(f"{gen._canonical_row(row)}\n" for row in rows), encoding="utf-8")
    before = len(runtime.calls)

    assert gen.main(["--check", "--snapshot", str(snapshot)]) == 0
    assert all(call["layout_and_style_ids"] is not None for call in runtime.calls[before:])
