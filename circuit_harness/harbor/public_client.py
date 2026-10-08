"""Standalone stdlib client uploaded beside /opt/harness/public.json.

Action stdin must include a stable action_id. Transport failures are uncertain;
this client never retries them or submits a replacement action automatically.
"""

import argparse
import http.client
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/opt/harness/public.json")
    parser.add_argument("command", choices=("info", "action"))
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        if not isinstance(config, dict):
            raise ValueError
        url, token = config["url"], config["token"]
        if not isinstance(url, str) or not isinstance(token, str):
            raise ValueError
        timeout = float(config["timeout"])
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not token
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError
        data = None
        if args.command == "action":
            data = sys.stdin.buffer.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024 or not isinstance(json.loads(data), dict):
                raise ValueError
        request = urllib.request.Request(
            url.rstrip("/") + "/" + args.command,
            data=data,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        # Agent-container proxy settings must never intercept the trial token.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError
        print(json.dumps(result))
        return 0
    except urllib.error.HTTPError as error:
        print(f"public gateway rejected request (HTTP {error.code})", file=sys.stderr)
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError):
        print(
            "public gateway unavailable; action outcome may be uncertain; no automatic retry",
            file=sys.stderr,
        )
    except (ValueError, TypeError, KeyError, RecursionError):
        print("invalid public client configuration, request or response", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
