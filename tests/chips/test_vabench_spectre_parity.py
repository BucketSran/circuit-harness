"""Visible-waveform comparison is independent of the final EVAS score."""

import csv

import pytest

from circuit_harness.execution.vabench_spectre_parity import compare_visible_waveforms


def test_visible_psf_comparison_reports_event_and_quiet_error(tmp_path):
    psf = tmp_path / "tran.tran.tran"
    psf.write_text(
        "TRACE\n"
        '"data" "V"\n"clk" "V"\n"retimed_data" "V"\n"up" "V"\n"down" "V"\n'
        "VALUE\n"
        '"time" 0\n"data" 0\n"clk" 0\n"retimed_data" 0\n"up" 0\n"down" 0\n'
        '"time" 5e-10\n"data" 0.45\n"clk" 0\n"retimed_data" 0\n"up" 0.9\n"down" 0\n'
        '"time" 2e-9\n"data" 0.9\n"clk" 0\n"retimed_data" 0\n"up" 0.9\n"down" 0\n'
        "END\n"
    )
    csv_path = tmp_path / "tran.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "data", "clk", "retimed_data", "up", "down"))
        writer.writerows(
            ((0, 0, 0, 0, 0, 0), (5e-10, 0.45, 0, 0, 0.8, 0), (2e-9, 0.9, 0, 0, 0.9, 0))
        )
    report = compare_visible_waveforms(csv_path, psf)
    assert report["spectre_points"] == 3
    assert report["evas_points"] == 3
    assert report["signals"]["up"]["max_abs_error_v"] == pytest.approx(0.1)
    assert report["signals"]["up"]["quiet_max_abs_error_v"] == pytest.approx(0.0)
    assert report["signals"]["up"]["logic_mismatch_samples"] == 0


def test_visible_psf_rejects_wrong_trace_set(tmp_path):
    psf = tmp_path / "tran.tran.tran"
    psf.write_text('TRACE\n"private_score" "V"\nVALUE\nEND\n')
    csv_path = tmp_path / "tran.csv"
    csv_path.write_text("time,data,clk,retimed_data,up,down\n0,0,0,0,0,0\n1e-9,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="trace set"):
        compare_visible_waveforms(csv_path, psf)
