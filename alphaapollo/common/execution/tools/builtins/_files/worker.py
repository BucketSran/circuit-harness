# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Self-contained workspace worker injected into sandbox images."""

# Sandboxed worker source ----------------------------------------------------
# Keep this worker self-contained: the Podman baseline is python:3.11-slim and
# deliberately has no dependency on rg/fd/pathspec or a host bind mount.
_WORKSPACE_WORKER = r"""
import base64
import fnmatch
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT_LEXICAL = Path(WORKSPACE_ROOT)
ROOT = ROOT_LEXICAL.resolve(strict=True)
MAX_LINES = 2000
MAX_BYTES = 50 * 1024
# Reserve room for the actionable notice so the outer Bash/Podman accumulator
# does not tail-truncate a head-oriented workspace result.
CONTENT_MAX_LINES = MAX_LINES - 2
CONTENT_MAX_BYTES = MAX_BYTES - 512
GREP_MAX_LINE_LENGTH = 500
CONTROL_PREFIX = "__ALPHAAPOLLO_TOOL_ERROR__"
ARTIFACT_PREFIX = "__ALPHAAPOLLO_TOOL_ARTIFACT__"
ARTIFACT_PLACEHOLDER = "__ALPHAAPOLLO_TOOL_ARTIFACT_ID__"
CAPTURE_ARTIFACT = False


class ToolFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def fail(code, message):
    raise ToolFailure(code, message)


def emit_failure(error):
    payload = json.dumps(
        {"code": error.code, "message": error.message},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    print(CONTROL_PREFIX + payload, file=sys.stderr)


def lexical_candidate(raw, default=None):
    if raw is None:
        raw = default
    if not isinstance(raw, str) or not raw.strip():
        fail("invalid_path", "tool path must be a non-empty string")
    if "\x00" in raw:
        fail("invalid_path", "tool path must not contain NUL")
    pure = PurePosixPath(raw)
    if ".." in pure.parts:
        fail("path_forbidden", "tool path traversal is not allowed")
    candidate = Path(str(pure)) if pure.is_absolute() else ROOT_LEXICAL / str(pure)
    try:
        candidate.relative_to(ROOT_LEXICAL)
    except ValueError:
        fail("path_forbidden", "tool path must stay within /workspace")
    return candidate


def confined_path(raw, *, default=None, must_exist=False):
    candidate = lexical_candidate(raw, default)
    try:
        # Resolve existing symlink prefixes even when the final path is absent.
        # Checking existence first would misclassify ``escape/missing.txt`` as
        # merely absent when ``escape`` points outside /workspace.
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        fail("path_unavailable", "tool path could not be resolved")
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        fail("path_forbidden", "resolved tool path escapes /workspace")
    if must_exist and not resolved.exists():
        fail("path_not_found", f"path not found: {raw if raw is not None else default}")
    return resolved


def read_text(path):
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        fail("path_not_found", "file not found")
    except IsADirectoryError:
        fail("not_a_file", "path is a directory, not a file")
    except PermissionError:
        fail("permission_denied", "file is not readable")
    except OSError:
        fail("read_failed", "file could not be read")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        fail("not_text_file", "file is not valid UTF-8 text")


def truncate_head(text, *, max_lines=CONTENT_MAX_LINES, max_bytes=CONTENT_MAX_BYTES):
    raw_lines = text.split("\n")
    total_lines = len(raw_lines)
    total_bytes = len(text.encode("utf-8"))
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, False, None, total_lines, total_bytes, total_lines

    output = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(raw_lines[:max_lines]):
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output.append(line)
        output_bytes += line_bytes
    if len(output) >= max_lines and len(output) < total_lines:
        truncated_by = "lines"
    return "\n".join(output), True, truncated_by, total_lines, total_bytes, len(output)


def append_notice(content, notices):
    notices = [notice for notice in notices if notice]
    if not notices:
        return content
    separator = "\n\n" if content else ""
    return content + separator + "[" + ". ".join(notices) + "]"


def emit_result(full_content, output, notices, *, truncated):
    if truncated and CAPTURE_ARTIFACT:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="alphaapollo-tool-output-",
            suffix=".txt",
            dir="/tmp",
            delete=False,
        )
        try:
            handle.write(full_content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        metadata = json.dumps(
            {"path": handle.name},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        print(ARTIFACT_PREFIX + metadata, file=sys.stderr)
        notices.append(f"Full output artifact: {ARTIFACT_PLACEHOLDER}")
    print(append_notice(output, notices), end="")


def fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, content):
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fail("permission_denied", "parent directory is not writable")
    except OSError:
        fail("write_failed", "parent directory could not be created")

    # Re-resolve after mkdir so every now-existing component is checked.
    try:
        parent_resolved = parent.resolve(strict=True)
        parent_resolved.relative_to(ROOT)
    except (FileNotFoundError, ValueError, OSError):
        fail("path_forbidden", "resolved parent directory escapes /workspace")

    destination = path.resolve(strict=False)
    try:
        destination.relative_to(ROOT)
    except ValueError:
        fail("path_forbidden", "resolved destination escapes /workspace")
    if destination.exists() and destination.is_dir():
        fail("not_a_file", "destination is a directory")

    existing_mode = None
    try:
        existing_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        pass
    except OSError:
        fail("write_failed", "destination metadata could not be read")

    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".alphaapollo-tool-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        os.replace(temporary, destination)
        temporary = None
        fsync_directory(destination.parent)
    except PermissionError:
        fail("permission_denied", "destination is not writable")
    except OSError:
        fail("write_failed", "file could not be written atomically")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def positive_int(args, name, default):
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail("invalid_arguments", f"{name} must be a positive integer")
    return value


def non_negative_int(args, name, default):
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail("invalid_arguments", f"{name} must be a non-negative integer")
    return value


def tool_read(args):
    path = confined_path(args.get("path"), must_exist=True)
    if not path.is_file():
        fail("not_a_file", "path is not a regular file")
    text = read_text(path)
    lines = text.split("\n")
    offset = positive_int(args, "offset", 1)
    if offset > len(lines):
        fail("offset_out_of_range", f"offset {offset} is beyond end of file ({len(lines)} lines)")
    limit = args.get("limit")
    if limit is not None:
        limit = positive_int(args, "limit", 1)
    start = offset - 1
    selected = lines[start:] if limit is None else lines[start : start + limit]
    selected_text = "\n".join(selected)
    output, truncated, truncated_by, _, _, output_lines = truncate_head(selected_text)
    notices = []
    if truncated:
        if output_lines == 0:
            notices.append(f"Line {offset} exceeds the {MAX_BYTES // 1024}KB output limit")
        else:
            end = offset + output_lines - 1
            detail = f"Showing lines {offset}-{end} of {len(lines)}"
            if truncated_by == "bytes":
                detail += f" ({MAX_BYTES // 1024}KB limit)"
            notices.append(detail + f". Use offset={end + 1} to continue")
    elif start + len(selected) < len(lines):
        remaining = len(lines) - (start + len(selected))
        notices.append(
            f"{remaining} more lines in file. "
            f"Use offset={start + len(selected) + 1} to continue"
        )
    emit_result(selected_text, output, notices, truncated=truncated)


def tool_write(args):
    path = confined_path(args.get("path"), must_exist=False)
    content = args.get("content")
    if not isinstance(content, str):
        fail("invalid_arguments", "content must be a string")
    atomic_write(path, content)
    print(f"Successfully wrote {len(content.encode('utf-8'))} bytes to {args['path']}")


def tool_edit(args):
    path = confined_path(args.get("path"), must_exist=True)
    if not path.is_file():
        fail("not_a_file", "path is not a regular file")
    edits = args.get("edits")
    if not isinstance(edits, list) or not edits:
        fail("invalid_arguments", "edits must contain at least one replacement")
    raw = read_text(path)
    bom = "\ufeff" if raw.startswith("\ufeff") else ""
    content = raw[len(bom):]
    line_ending = "\r\n" if "\r\n" in content else "\n"
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    spans = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            fail("invalid_arguments", f"edits[{index}] must be an object")
        old = edit.get("oldText")
        new = edit.get("newText")
        if not isinstance(old, str) or not old:
            fail("invalid_arguments", f"edits[{index}].oldText must be a non-empty string")
        if not isinstance(new, str):
            fail("invalid_arguments", f"edits[{index}].newText must be a string")
        normalized_old = old.replace("\r\n", "\n").replace("\r", "\n")
        normalized_new = new.replace("\r\n", "\n").replace("\r", "\n")
        # ``str.count`` ignores overlapping occurrences (for example ``aa`` in
        # ``aaa``), but the edit contract requires one unique textual match.
        # Advance by one character so every possible start position is counted.
        match_starts = []
        cursor = 0
        while True:
            match_start = normalized_content.find(normalized_old, cursor)
            if match_start < 0:
                break
            match_starts.append(match_start)
            cursor = match_start + 1
        if not match_starts:
            fail("edit_no_match", f"edits[{index}].oldText did not match the file")
        if len(match_starts) != 1:
            fail(
                "edit_not_unique",
                f"edits[{index}].oldText matched {len(match_starts)} locations",
            )
        start = match_starts[0]
        spans.append((start, start + len(normalized_old), normalized_new, index))
    ordered = sorted(spans, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            fail("edit_overlap", f"edits[{previous[3]}] and edits[{current[3]}] overlap")
    updated = normalized_content
    for start, end, replacement, _ in reversed(ordered):
        updated = updated[:start] + replacement + updated[end:]
    if line_ending != "\n":
        updated = updated.replace("\n", line_ending)
    atomic_write(path, bom + updated)
    print(f"Successfully replaced {len(edits)} block(s) in {args['path']}.")


class IgnoreRule:
    def __init__(self, base, pattern, negated, directory_only):
        self.base = PurePosixPath(base)
        self.pattern = pattern
        self.negated = negated
        self.directory_only = directory_only

    def matches(self, relative, is_directory):
        relative = PurePosixPath(relative)
        try:
            sub = relative.relative_to(self.base)
        except ValueError:
            return False
        sub_text = sub.as_posix()
        if sub_text == ".":
            return False
        pattern = self.pattern
        anchored = pattern.startswith("/")
        if anchored:
            pattern = pattern[1:]
        candidates = [sub]
        if self.directory_only:
            candidates = [PurePosixPath(*sub.parts[:i]) for i in range(1, len(sub.parts) + 1)]
        for candidate in candidates:
            candidate_text = candidate.as_posix()
            candidate_is_dir = is_directory or candidate != sub
            if self.directory_only and not candidate_is_dir:
                continue
            if "/" not in pattern and not anchored:
                if any(fnmatch.fnmatchcase(part, pattern) for part in candidate.parts):
                    return True
            elif pattern.startswith("**/"):
                if PurePosixPath(candidate_text).match(pattern):
                    return True
                if PurePosixPath(candidate_text).match(pattern[3:]):
                    return True
            elif PurePosixPath("/" + candidate_text).match("/" + pattern):
                return True
        return False


def load_ignore_rules(directory, inherited):
    rules = list(inherited)
    ignore_file = directory / ".gitignore"
    if not ignore_file.is_file():
        return rules
    try:
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return rules
    try:
        base = directory.relative_to(ROOT).as_posix()
    except ValueError:
        return rules
    if base == ".":
        base = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        if not line:
            continue
        directory_only = line.endswith("/")
        line = line.rstrip("/")
        rules.append(IgnoreRule(base, line, negated, directory_only))
    return rules


def is_ignored(relative, is_directory, rules):
    ignored = False
    for rule in rules:
        if rule.matches(relative, is_directory):
            ignored = not rule.negated
    return ignored


def safe_resolved_entry(path):
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT)
        return resolved
    except (FileNotFoundError, ValueError, OSError):
        return None


def ancestor_ignore_rules(search_root):
    rules = []
    current = ROOT
    relative = search_root.relative_to(ROOT)
    for part in relative.parts:
        rules = load_ignore_rules(current, rules)
        current = current / part
    return rules


def walk_workspace(search_root):
    if search_root.is_file():
        yield search_root, search_root.name
        return
    rules_by_directory = {search_root: ancestor_ignore_rules(search_root)}
    for current_text, directories, files in os.walk(search_root, topdown=True, followlinks=False):
        current = Path(current_text)
        rules = load_ignore_rules(current, rules_by_directory.get(current, []))
        kept_directories = []
        for name in sorted(directories, key=lambda value: (value.casefold(), value)):
            candidate = current / name
            ignore_relative = candidate.relative_to(ROOT).as_posix()
            if name in {".git", "node_modules"}:
                continue
            if candidate.is_symlink() or safe_resolved_entry(candidate) is None:
                continue
            if is_ignored(ignore_relative, True, rules):
                continue
            rules_by_directory[candidate] = rules
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(files, key=lambda value: (value.casefold(), value)):
            candidate = current / name
            ignore_relative = candidate.relative_to(ROOT).as_posix()
            if is_ignored(ignore_relative, False, rules):
                continue
            resolved = safe_resolved_entry(candidate)
            if resolved is None or not resolved.is_file():
                continue
            relative = candidate.relative_to(search_root).as_posix()
            yield resolved, relative


def glob_matches(relative, pattern):
    path = PurePosixPath(relative)
    if path.match(pattern):
        return True
    if "/" not in pattern and fnmatch.fnmatchcase(path.name, pattern):
        return True
    if pattern.startswith("**/") and path.match(pattern[3:]):
        return True
    return False


def tool_find(args):
    search_root = confined_path(args.get("path"), default=".", must_exist=True)
    if not search_root.is_dir():
        fail("not_a_directory", "find path must be a directory")
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        fail("invalid_arguments", "pattern must be a non-empty string")
    limit = positive_int(args, "limit", 1000)
    results = []
    reached = False
    for _, relative in walk_workspace(search_root):
        if glob_matches(relative, pattern):
            if len(results) >= limit:
                reached = True
                break
            results.append(relative)
    if not results:
        print("No files found matching pattern")
        return
    raw = "\n".join(sorted(results, key=lambda value: (value.casefold(), value)))
    output, truncated, _, _, _, _ = truncate_head(raw, max_lines=10**9)
    notices = []
    if reached:
        notices.append(
            f"{limit} results limit reached. "
            f"Use limit={limit * 2} for more, or refine pattern"
        )
    if truncated:
        notices.append(f"{MAX_BYTES // 1024}KB limit reached")
    emit_result(raw, output, notices, truncated=truncated)


def truncate_grep_line(line):
    if len(line) <= GREP_MAX_LINE_LENGTH:
        return line, False
    return line[:GREP_MAX_LINE_LENGTH] + "... [truncated]", True


def tool_grep(args):
    search_root = confined_path(args.get("path"), default=".", must_exist=True)
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        fail("invalid_arguments", "pattern must be a non-empty string")
    literal = args.get("literal", False)
    ignore_case = args.get("ignoreCase", False)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        expression = re.compile(re.escape(pattern) if literal else pattern, flags)
    except re.error:
        fail("invalid_pattern", "pattern is not a valid regular expression")
    context = non_negative_int(args, "context", 0)
    limit = positive_int(args, "limit", 100)
    glob = args.get("glob")
    if glob is not None and (not isinstance(glob, str) or not glob):
        fail("invalid_arguments", "glob must be a non-empty string when provided")

    files = list(walk_workspace(search_root))
    matches = []
    reached = False
    lines_truncated = False
    for file_path, relative in files:
        if glob and not glob_matches(relative, glob):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        match_lines = [index for index, line in enumerate(lines) if expression.search(line)]
        for index in match_lines:
            if len(matches) >= limit:
                reached = True
                break
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            block = []
            for current in range(start, end):
                shown, was_truncated = truncate_grep_line(lines[current])
                lines_truncated = lines_truncated or was_truncated
                separator = ":" if current == index else "-"
                block.append(f"{relative}{separator}{current + 1}{separator} {shown}")
            matches.append("\n".join(block))
        if reached:
            break
    if not matches:
        print("No matches found")
        return
    raw = "\n--\n".join(matches) if context else "\n".join(matches)
    output, truncated, _, _, _, _ = truncate_head(raw, max_lines=10**9)
    notices = []
    if reached:
        notices.append(
            f"{limit} matches limit reached. "
            f"Use limit={limit * 2} for more, or refine pattern"
        )
    if truncated:
        notices.append(f"{MAX_BYTES // 1024}KB limit reached")
    if lines_truncated:
        notices.append(
            f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. "
            "Use read tool to see full lines"
        )
    emit_result(raw, output, notices, truncated=truncated)


def tool_ls(args):
    directory = confined_path(args.get("path"), default=".", must_exist=True)
    if not directory.is_dir():
        fail("not_a_directory", "ls path must be a directory")
    limit = positive_int(args, "limit", 500)
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        fail("permission_denied", "directory is not readable")
    except OSError:
        fail("read_failed", "directory could not be listed")
    formatted = []
    for entry in sorted(entries, key=lambda value: (value.name.casefold(), value.name)):
        resolved = safe_resolved_entry(entry)
        if resolved is None:
            # Do not expose entries whose target escapes the workspace.
            continue
        suffix = "/" if resolved.is_dir() else ""
        formatted.append(entry.name + suffix)
    reached = len(formatted) > limit
    selected = formatted[:limit]
    if not selected:
        print("(empty directory)")
        return
    raw = "\n".join(selected)
    output, truncated, _, _, _, _ = truncate_head(raw, max_lines=10**9)
    notices = []
    if reached:
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
    if truncated:
        notices.append(f"{MAX_BYTES // 1024}KB limit reached")
    emit_result(raw, output, notices, truncated=truncated)


TOOLS = {
    "read": tool_read,
    "write": tool_write,
    "edit": tool_edit,
    "grep": tool_grep,
    "find": tool_find,
    "ls": tool_ls,
}


def main():
    global CAPTURE_ARTIFACT
    payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    tool_id = payload.get("tool_id")
    arguments = payload.get("arguments")
    CAPTURE_ARTIFACT = payload.get("capture_artifact") is True
    if tool_id not in TOOLS or not isinstance(arguments, dict):
        fail("invalid_request", "workspace tool request is malformed")
    TOOLS[tool_id](arguments)


try:
    main()
except ToolFailure as error:
    emit_failure(error)
    raise SystemExit(2)
except Exception:
    emit_failure(ToolFailure("internal_error", "workspace tool failed unexpectedly"))
    raise SystemExit(2)
"""
