"""Analog tools reuse the durable action transport with Analog commands."""

from alphaapollo.common.execution.chips.analog_remote import LocalAnalog, RemoteAnalog


def test_analog_transport_uses_analog_request_and_response(tmp_path, monkeypatch):
    config = {
        "host": "lab-host",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/episode",
    }
    for transport in (
        RemoteAnalog(config, tmp_path / "ssh"),
        LocalAnalog(config, tmp_path / "local"),
    ):
        calls = []

        def cli(*arguments, payload=None, calls=calls, **kwargs):
            calls.append((arguments, payload))
            return (
                {"state": "accepted"}
                if arguments[0] == "analog-request"
                else {"ok": True, "result": {"state": "submitted"}}
            )

        monkeypatch.setattr(transport, "cli", cli)
        reply = transport.call("analog_submit", {}, action_id="one")
        assert reply["ok"]
        assert [entry[0][0] for entry in calls] == ["analog-request", "analog-response"]
        assert calls[0][1] == {"id": "one", "tool": "analog_submit", "arguments": {}}
