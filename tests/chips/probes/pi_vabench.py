"""Real Pi and SSH/EVAS; scripted HTTP responses, not a real model."""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from alphaapollo.common.execution.chips.vabench_remote import RemoteVabench  # noqa: E402
from alphaapollo.workflows.chips_vabench_agent import run_pi  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--pi", type=Path, required=True)
parser.add_argument("--candidate", type=Path)
parser.add_argument("--artifact", default="dut.va")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--budget-only", action="store_true")
args = parser.parse_args()
if not args.budget_only and args.candidate is None:
    parser.error("--candidate is required for the repair fixture")
candidate = args.candidate.read_text() if args.candidate else ""
actions = [
    ("vabench_read", {"path": ""}),
    ("vabench_read", {"path": "task/instruction.md"}),
    ("vabench_write", {"path": args.artifact, "content": "invalid syntax\n"}),
    ("vabench_simulate", {}),
    ("vabench_write", {"path": args.artifact, "content": candidate}),
    ("vabench_simulate", {}),
    ("vabench_submit", {}),
]
requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        index = len(requests)
        requests.append(
            {
                "model": body.get("model"),
                "tools": [t["function"]["name"] for t in body.get("tools", [])],
                "messages": body["messages"],
            }
        )
        if index < len(actions):
            name, args = actions[index]
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"fixture-{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": "Submitted."}
            finish = "stop"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {
                "id": f"chat-{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"chat-{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
            },
        ]
        for chunk in chunks:
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
config = json.loads(args.config.read_text())
config.update(
    base_url=f"http://127.0.0.1:{server.server_port}/v1",
    model="fixture-model",
    pi_cli=str(args.pi.resolve()),
    episode_timeout_s=240,
    policy_kind="scripted_http_fixture",
    max_model_calls=1 if args.budget_only else 12,
)
os.environ["CHIPS_MODEL_KEY"] = "TEST_ONLY_NOT_A_REAL_KEY"
try:
    result = run_pi(config, args.output)
    print(
        json.dumps(
            {
                "termination_reason": result["termination_reason"],
                "events": len(result["events"]),
                "requests": len(requests),
            }
        )
    )
    (args.output / "provider-fixture.json").write_text(json.dumps(requests, indent=2))
    expected = 1 if args.budget_only else 8
    assert len(requests) == expected, {"requests": len(requests), "expected": expected}
    assert set(requests[0]["tools"]) == {
        "vabench_read",
        "vabench_write",
        "vabench_simulate",
        "vabench_submit",
    }
    if not args.budget_only:
        assert result["termination_reason"] == "final"
        replies = [
            json.loads(m["content"]) for m in requests[-1]["messages"] if m["role"] == "tool"
        ]
        assert all(r["ok"] for r in replies), replies
        assert replies[3]["result"]["status"] == "failed"
        assert replies[5]["result"]["status"] == "succeeded"
        assert replies[6]["result"]["status"] == "submitted"
        from alphaapollo.workflows.chips_vabench_agent import collect_result

        remote = RemoteVabench(config, args.output / "tools")
        remote.cli(
            "vabench-finalize",
            "--session",
            config["session"],
            "--root",
            config["job_root"],
            "--job-id",
            config["job_id"],
            "--archive-root",
            config["archive_root"],
            timeout=60,
        )
        assert collect_result(remote, config, args.output)
finally:
    server.shutdown()
