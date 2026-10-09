"""Inspect task drafts and record explicit operator confirmation."""

import argparse
import json
import os
from pathlib import Path

from circuit_harness.execution.sessions.task_authoring import (
    confirm_draft,
    inspect_draft,
    require_confirmation,
)


def _read(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("input must be a regular JSON file of at most 1 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    """Operator CLI for task construction; not an Agent-visible approval tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "confirm", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--draft", type=Path, required=True)
        command.add_argument("--materials", type=Path, required=True)
        command.add_argument("--requirements", type=Path, required=True)
        if name == "confirm":
            command.add_argument("--reviewer", required=True)
            command.add_argument(
                "--kind", choices=("human", "scripted_confirmation"), required=True
            )
            command.add_argument("--output", type=Path, required=True)
        elif name == "verify":
            command.add_argument("--confirmation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        draft, requirements = _read(args.draft), _read(args.requirements)
        if args.command == "inspect":
            result = inspect_draft(draft, args.materials, requirements)
            code = int(result["status"] != "ready_for_confirmation")
        elif args.command == "confirm":
            result = confirm_draft(
                draft, args.materials, requirements, reviewer=args.reviewer, kind=args.kind
            )
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            code = 0
        else:
            values = require_confirmation(
                draft, args.materials, requirements, _read(args.confirmation)
            )
            result, code = {"status": "confirmed", "values": values}, 0
    except (ValueError, OSError) as error:
        result, code = {"status": "rejected", "reason": str(error)}, 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
