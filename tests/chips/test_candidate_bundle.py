"""Frozen public deliverables preserve exact bytes, including incorrect syntax."""

import pytest

from alphaapollo.common.execution.chips.candidate_bundle import freeze_candidate, verify_candidate


def test_freeze_preserves_invalid_candidate_and_rejects_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "dut.va").write_bytes(b"not valid verilog\n")
    frozen = tmp_path / "frozen"
    receipt = freeze_candidate(
        source, frozen, ["dut.va"], task_id="synthetic", task_version="v1", reason="submit"
    )
    assert (frozen / "files/dut.va").read_bytes() == b"not valid verilog\n"
    assert verify_candidate(frozen) == receipt
    second = freeze_candidate(
        source,
        tmp_path / "second",
        ["dut.va"],
        task_id="synthetic",
        task_version="v1",
        reason="deadline",
    )
    assert second["candidate_sha256"] == receipt["candidate_sha256"]
    (frozen / "files/dut.va").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_candidate(frozen)


@pytest.mark.parametrize(
    "names", [["../dut.va"], ["dut.va", "dut.va"], ["dut.va", "dut.va/x"], ["/dut.va"]]
)
def test_unsafe_names_cannot_freeze(tmp_path, names):
    with pytest.raises(ValueError):
        freeze_candidate(
            tmp_path, tmp_path / "out", names, task_id="t", task_version="v", reason="submit"
        )


def test_symlink_and_missing_deliverable_cannot_freeze(tmp_path):
    (tmp_path / "dut.va").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        freeze_candidate(
            tmp_path, tmp_path / "out", ["dut.va"], task_id="t", task_version="v", reason="submit"
        )
