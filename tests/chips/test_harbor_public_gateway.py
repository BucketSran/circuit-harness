"""Real loopback transport and standalone CLI over synthetic public sessions."""

import asyncio
import json
import urllib.error
import urllib.request

from test_current_evas_session import make_session

from alphaapollo.workflows.harbor_chips.public_gateway import PublicSessionGateway


def request(gateway, path, body=None, *, token=None, method=None):
    headers = {"Authorization": "Bearer " + (gateway.token if token is None else token)}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(gateway.url + path, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(req, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        data = response.read()
        return response.status, json.loads(data) if data.startswith(b"{") else data.decode()


def test_authenticated_info_does_not_expose_host_paths(tmp_path):
    directory = make_session(tmp_path)

    async def scenario():
        gateway = await PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        ).start()
        try:
            status, info = await asyncio.to_thread(request, gateway, "/info")
            assert status == 200
            assert info["task_id"] == "synthetic"
            assert "public_workspace" not in info and "candidate_workspace" not in info
            assert str(directory) not in json.dumps(info)
            assert (await asyncio.to_thread(request, gateway, "/info", token="bad"))[0] == 401
            assert (await asyncio.to_thread(request, gateway, "/unknown"))[0] == 404
        finally:
            await gateway.close()

    asyncio.run(scenario())


def test_gateway_advertises_and_serves_bounded_waveform_reads(tmp_path, monkeypatch):
    from tests.chips.test_public_observations import waveform_session

    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])

    async def scenario():
        gateway = await PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        ).start()

        async def action(name, tool, **arguments):
            status, reply = await asyncio.to_thread(
                request,
                gateway,
                "/action",
                json.dumps(
                    {
                        "action_id": name,
                        "tool": tool,
                        "arguments": arguments,
                    }
                ).encode(),
            )
            assert status == 200 and reply["ok"], reply
            assert str(directory) not in json.dumps(reply)
            return reply["result"]

        try:
            _, info = await asyncio.to_thread(request, gateway, "/info")
            assert "evas_observe" in {row["function"]["name"] for row in info["tools"]}
            result = await action("sim", "evas_simulate")
            assert result["observations"]["sample_count"] == 1001
            window = await action(
                "window",
                "evas_observe",
                artifact_id=result["observation_artifact"]["id"],
                signals=["count"],
                start=0.499,
                end=0.5,
            )
            assert window["values"] == [[0], [1]]
            assert not window["sampled"]
        finally:
            await gateway.close()

    asyncio.run(scenario())


def test_actions_keep_ids_and_submit_hides_candidate_directory(tmp_path):
    directory = make_session(tmp_path)
    write = json.dumps(
        {
            "action_id": "write",
            "tool": "evas_write",
            "arguments": {"path": "dut.va", "content": "candidate"},
        }
    ).encode()

    async def scenario():
        gateway = await PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        ).start()
        try:
            status, reply = await asyncio.to_thread(request, gateway, "/action", write)
            assert status == 200 and reply["ok"]
            assert (await asyncio.to_thread(request, gateway, "/action", write))[1] == reply
            assert (await asyncio.to_thread(request, gateway, "/action", b"{}"))[0] == 400
            changed = write.replace(b"candidate", b"different")
            assert (await asyncio.to_thread(request, gateway, "/action", changed))[0] == 400
            submit = json.dumps(
                {"action_id": "submit", "tool": "evas_submit", "arguments": {}}
            ).encode()
            status, receipt = await asyncio.to_thread(request, gateway, "/action", submit)
            assert status == 200 and receipt["result"]["state"] == "submitted"
            assert "candidate_directory" not in receipt["result"]
            assert str(directory) not in json.dumps(receipt)
            assert (directory / "candidate/files/dut.va").read_text() == "candidate"
        finally:
            await gateway.close()

    asyncio.run(scenario())


def test_standalone_cli_uses_config_and_ignores_inherited_proxies(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    directory = make_session(tmp_path)
    client = (
        Path(__file__).resolve().parents[2] / "alphaapollo/workflows/harbor_chips/public_client.py"
    )

    async def scenario():
        gateway = await PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        ).start()
        config = tmp_path / "client.json"
        config.write_text(json.dumps({"url": gateway.url, "token": gateway.token, "timeout": 5}))
        env = {
            **os.environ,
            "HTTP_PROXY": "http://127.0.0.1:1",
            "http_proxy": "http://127.0.0.1:1",
            "NO_PROXY": "",
            "no_proxy": "",
        }

        async def cli(command, data=""):
            return await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-I", str(client), "--config", str(config), command],
                input=data,
                text=True,
                capture_output=True,
                env=env,
                timeout=10,
            )

        try:
            result = await cli("info")
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["task_id"] == "synthetic"
            payload = json.dumps(
                {
                    "action_id": "cli",
                    "tool": "evas_write",
                    "arguments": {"path": "dut.va", "content": "cli candidate"},
                }
            )
            assert json.loads((await cli("action", payload)).stdout)["ok"]
            assert (directory / "submission/dut.va").read_text() == "cli candidate"
            result = await cli("action", "invalid")
            assert result.returncode != 0 and gateway.token not in result.stderr
            config.write_text(json.dumps({"url": gateway.url, "token": "incorrect", "timeout": 5}))
            result = await cli("info")
            assert result.returncode != 0 and "incorrect" not in result.stderr
        finally:
            await gateway.close()

    asyncio.run(scenario())


async def raw_http(gateway, headers, body=b""):
    from urllib.parse import urlsplit

    parsed = urlsplit(gateway.url)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    try:
        writer.write(headers + body)
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=5)
        return int(raw.split(b" ", 2)[1]), raw
    finally:
        writer.close()
        await writer.wait_closed()


def test_request_size_timeout_and_auth_are_bounded(tmp_path):
    directory = make_session(tmp_path)

    async def scenario():
        gateway = await PublicSessionGateway(
            directory,
            bind_host="127.0.0.1",
            advertised_host="127.0.0.1",
            max_request_bytes=100,
            request_timeout_s=0.1,
        ).start()
        prefix = (
            "POST /action HTTP/1.1\r\nHost: localhost\r\n"
            f"Authorization: Bearer {gateway.token}\r\nConnection: close\r\n"
        ).encode()
        try:
            # Reject an oversized declaration without waiting for the body.
            status, raw = await raw_http(gateway, prefix + b"Content-Length: 101\r\n\r\n")
            assert status == 413
            status, _ = await raw_http(
                gateway,
                prefix + b"Transfer-Encoding: chunked\r\n\r\n",
                b"65\r\n" + b"x" * 101 + b"\r\n0\r\n\r\n",
            )
            assert status == 413
            status, _ = await raw_http(gateway, prefix + b"Content-Length: 1\r\n\r\n")
            assert status == 408
            status, raw = await raw_http(
                gateway,
                b"GET /info HTTP/1.1\r\nHost: localhost\r\n"
                b"Authorization: Bearer \xff\r\nConnection: close\r\n\r\n",
            )
            assert status == 401 and b"Traceback" not in raw
            assert not list((directory / "actions").iterdir())
        finally:
            await gateway.close()

    asyncio.run(scenario())


def test_disconnected_write_drains_before_freeze_and_shutdown_revokes_admissions(tmp_path):
    import fcntl
    from urllib.parse import urlsplit

    from alphaapollo.common.execution.chips.current_evas_session import close_session

    directory = make_session(tmp_path)

    async def scenario():
        gateway = await PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1", max_connections=1
        ).start()
        parsed = urlsplit(gateway.url)
        reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
        payload = json.dumps(
            {
                "action_id": "disconnected",
                "tool": "evas_write",
                "arguments": {"path": "dut.va", "content": "drained candidate"},
            }
        ).encode()
        lock = (directory / ".session.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        closing = None
        try:
            writer.write(
                (
                    "POST /action HTTP/1.1\r\nHost: localhost\r\n"
                    f"Authorization: Bearer {gateway.token}\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode()
                + payload
            )
            await writer.drain()
            async with asyncio.timeout(5):
                while (await asyncio.to_thread(request, gateway, "/info"))[0] != 503:
                    await asyncio.sleep(0.01)
            # The occupied admission slot cannot create a second action worker.
            assert (await asyncio.to_thread(request, gateway, "/action", payload))[0] == 503
            writer.close()
            await writer.wait_closed()
            closing = asyncio.create_task(gateway.close())
            await asyncio.sleep(0)
            assert not closing.done()
            # Cancellation must finish draining before escaping to the environment.
            closing.cancel()
            await asyncio.sleep(0.02)
            assert not closing.done()
            closing.cancel()
            await asyncio.sleep(0.02)
            assert not closing.done()
            try:
                status, _ = await asyncio.to_thread(request, gateway, "/action", payload)
                assert status == 503
            except urllib.error.URLError:
                pass  # The HTTP listener may already have stopped.
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            writer.close()
            if closing is not None:
                import pytest

                with pytest.raises(asyncio.CancelledError):
                    await closing
            await gateway.close()
        assert close_session(directory, "timeout")["state"] == "collected"
        assert (directory / "candidate/files/dut.va").read_text() == "drained candidate"
        assert json.loads((directory / "actions/disconnected/response.json").read_text())["ok"]

    asyncio.run(scenario())


def test_cli_does_not_retry_or_print_transport_error_details(tmp_path):
    import http.server
    import subprocess
    import sys
    import threading
    from pathlib import Path

    attempts = []
    token = "private-fixture-token"

    class BrokenGateway(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            attempts.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            # An invalid status line must not escape into CLI error output.
            self.wfile.write((token + "\r\n").encode())
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), BrokenGateway)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config = tmp_path / "client.json"
    config.write_text(
        json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "token": token, "timeout": 1})
    )
    client = (
        Path(__file__).resolve().parents[2] / "alphaapollo/workflows/harbor_chips/public_client.py"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(client), "--config", str(config), "action"],
            input='{"action_id":"uncertain","tool":"evas_submit","arguments":{}}',
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert result.returncode != 0 and not result.stdout
        assert token not in result.stderr and "Traceback" not in result.stderr
        assert attempts == ["/action"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_cancelled_start_cleans_listener_and_close_is_repeatable(tmp_path):
    import pytest

    directory = make_session(tmp_path)

    async def scenario():
        gateway = PublicSessionGateway(
            directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        )
        starting = asyncio.create_task(gateway.start())
        await asyncio.sleep(0)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        await gateway.close()
        with pytest.raises(urllib.error.URLError):
            await asyncio.to_thread(request, gateway, "/info")
        with pytest.raises(RuntimeError, match="cannot be started again"):
            await gateway.start()

    asyncio.run(scenario())
