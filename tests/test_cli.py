"""Direct circuit operator commands."""

from circuit_harness.cli import main


def test_cli_report_escapes_logs_and_status(tmp_path):
    from circuit_harness.execution.journal import Journal

    Journal(tmp_path, call_id="test").emit("start", text="<script>alert(1)</script>")
    assert main(["report", str(tmp_path)]) == 0
    report = (tmp_path / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert main(["status", str(tmp_path)]) == 0
