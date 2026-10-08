# Harness structure references

Read the example relevant to the changed boundary and inspect its current code.

[Execution](../../../../circuit_harness/execution/README.md) owns simulator processes,
sessions, transport and evidence. [Harbor](../../../../circuit_harness/harbor/README.md)
owns composition with the external evaluation framework. Execution modules must not
select models or import the Harbor controller.

[Session transport](../../../../circuit_harness/execution/session_transport.py) preserves
stable action identity and same-action recovery. Unknown state is not permission to retry
as a new action. Preserve request matching and result integrity.

[Podman integration](../../../../circuit_harness/harbor/podman_environment.py) is an optional
backend. Keep imports within their declared integration boundary; honor each configured
capability or reject it before execution.

A source digest describes particular bytes, not numerical correctness. Check producer and
consumer together when changing records. [ATIF data](../../../../circuit_harness/data/README.md)
retains evidence and training-mask contracts without owning the user's trainer.
