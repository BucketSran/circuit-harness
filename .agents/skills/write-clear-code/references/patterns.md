# Harness structure references

Read only examples relevant to the changed boundary. Verify their current code before
reusing them; an example is not a required architecture.

## Execution and task policy

[Execution modules](../../../../alphaapollo/common/execution/chips/README.md) own simulator
processes, sessions, transport and evidence. Task-specific public feedback and grading remain
with the declared task. [Workflow modules](../../../../alphaapollo/workflows/README.md) compose
operations; do not make an executor select models or run the full experiment controller.

[Session transport](../../../../alphaapollo/common/execution/chips/session_transport.py)
provides stable action identity and same-action recovery. Preserve request matching,
unknown-state handling and result integrity when extending a backend.

## Public facade and implementation

The [external runtime entry](../../../../alphaapollo/reasoning/runtime/external_agent_runtime.py) and
[its support package](../../../../alphaapollo/reasoning/runtime/external) illustrate
a public entry with internal ownership. Before following the pattern, inspect import identity,
subclass/patch consumers and serialization. Do not add a package solely to reduce line counts.

## Resources and optional backends

[Podman execution](../../../../alphaapollo/common/execution/sandbox/_podman) is an existing
optional backend. Keep optional dependencies behind their declared integration boundary.
A runtime must either honor a granted capability or reject it before execution.

## Evidence and configuration

A recorded source/configuration digest is a claim about particular bytes, not proof of
numerical correctness. Check producer and consumer together when changing a record.
A migrated serialized format needs explicit compatibility handling; do not hide old data
behind a silently changed field meaning.
