# Domain documents

Read the [Circuit Harness glossary](../../GLOSSARY.md) for the platform's terminology.
Use the affected component's contract as the behavior authority:

- [Integration guide](../guides/integration.md): benchmark adapters, task execution paths and experiment setup.
- [Execution module guide](../../circuit_harness/execution/README.md): simulator,
  session, transport and evidence interfaces.
- [Development SOP](../chips/DEVELOPMENT_SOP.md): development, acceptance and delivery rules.
- [Experiment protocol](../chips/EXPERIMENT_PROTOCOL.md): declared conditions and evidence.
- [Test entry](../../tests/chips/README.md) and [validation record](../chips/VALIDATION.md):
  available checks and what actual evidence supports.
- [Repository scope](../chips/REPOSITORY_SCOPE.md): retained capabilities and publication boundaries.
- [Platform roadmap](../chips/ROADMAP.md) and [next work](../chips/NEXT_WORK.md): future outcomes and acceptance gaps.

EVAS semantics and algorithms belong to vaEVAS; task contents and checker rules belong to
its benchmark component. Read those contracts in the actual referenced checkout for
cross-repository work. Do not duplicate them here.

Use shared `domain-modeling` when a term changes or a design decision needs recording.
Update the glossary when a project term is agreed. Ordinary reversible workflow choices
belong in the existing SOP; create an ADR only for a durable tradeoff whose rationale would
otherwise be lost. Historical plans and file existence do not establish implemented support.
