"""Real local process output limits cover nested simulator artifacts."""

import sys
import threading
import time

from alphaapollo.common.execution.chips.journal import Journal
from alphaapollo.common.execution.chips.process import run_process


def test_nested_output_is_counted_and_process_group_is_cleaned(tmp_path):
    result = run_process(
        [
            sys.executable,
            "-c",
            "import pathlib,time; p=pathlib.Path('nested'); p.mkdir(); "
            "(p/'large').write_bytes(b'x'*65536); time.sleep(10)",
        ],
        directory=tmp_path,
        stage="probe",
        deadline=time.monotonic() + 2,
        cancel=threading.Event(),
        journal=Journal(tmp_path, call_id="nested"),
        max_output_bytes=32768,
    )
    assert result["execution"] == "output_limit"
    assert result["cleanup_confirmed"]
