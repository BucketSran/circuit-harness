# External-agent protocol fixtures

These small streams exercise parser behavior. They are sanitized excerpts from
development CLI runs, not benchmark trajectories or training data:

| File | Source | Behavior preserved |
| --- | --- | --- |
| `pi_bash_stream.jsonl` | Pi 0.80.10, `pi -p --mode json` | Duplicate announcements of one tool call, tool output, final text and usage |
| `pi_empty_response.jsonl` | Pi 0.80.10, `pi -p --mode json` | Provider rejection with a zero process exit and empty assistant content |
| `codex_shell_stream.jsonl` | codex-cli 0.147.0, `codex exec - --json` | Startup warnings, one shell command and final text |
| `claude_code_bash_stream.jsonl` | Claude Code 2.1.231, `claude -p --output-format stream-json` | Tool invocation, result and session completion |

Session, request, tool and host identifiers use fixed fixture values. Timestamps
use a fixed epoch, paths use `/workspace` or configuration placeholders, and
reasoning text uses a generic fixture sentence. Provider signatures are redacted.
The arithmetic prompt/output and provider error classification are retained;
event order, tool-call associations and usage values remain available to parser
regressions. Use these excerpts for offline parser checks, not session lookup.

Earlier trimming removed unused Pi deltas, most Codex startup warnings and
machine-specific Claude initialization fields. Do not treat the excerpts as
complete native transcripts, timing measurements or proof of live CLI support.
New captures need the same content review before committing; a `.jsonl` extension
or placement under `tests/` does not make arbitrary run data publishable.
