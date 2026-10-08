from __future__ import annotations

from alphaapollo.common.execution import ArtifactStore
from alphaapollo.common.execution.tools.builtins.shell import OutputAccumulator


def test_accumulator_streams_to_disk_and_keeps_pi_tail(tmp_path) -> None:
    accumulator = OutputAccumulator(max_lines=2, max_bytes=1_000)

    assert accumulator.append(b"one\ntwo\n") == "one\ntwo\n"
    assert accumulator.append(b"three\n") == "three\n"
    snapshot = accumulator.finish()

    assert snapshot.content == "two\nthree"
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "lines"
    assert snapshot.truncation.total_lines == 3
    assert snapshot.capture_path.read_bytes() == b"one\ntwo\nthree\n"
    snapshot.capture_path.unlink()


def test_multimegabyte_capture_is_stored_without_the_old_one_megabyte_cap(tmp_path) -> None:
    payload = b"prefix\n" + (b"x" * (2 * 1024 * 1024)) + b"\nactual-tail\n"
    accumulator = OutputAccumulator()
    for offset in range(0, len(payload), 64 * 1024):
        accumulator.append(payload[offset : offset + 64 * 1024])
    snapshot = accumulator.finish()
    store = ArtifactStore(tmp_path / "artifacts")

    artifact = store.put_file(
        snapshot.capture_path,
        type_="bash_stdout",
        created_by="bash",
    )
    ref = store.ref(artifact)

    assert snapshot.truncation.total_bytes == len(payload)
    assert snapshot.truncation.truncated is True
    assert snapshot.content.endswith("actual-tail\n")
    assert store.get(ref) == payload
    snapshot.capture_path.unlink()
