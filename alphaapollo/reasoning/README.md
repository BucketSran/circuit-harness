# Reasoning

Two focused packages, plus the private record helpers they share:

- `runtime/`: the batch-first Generation-to-Environment loop and complete
  per-task trajectories;
- `verification/`: one shared batch interface for agent-backed and
  deterministic verification.

Workflow topology, prepared-input loading, persistence, and resource
composition live in `alphaapollo.workflows`. Reasoning does not import that
package and does not depend on Torch, Ray, TensorDict, DataProto, or trainer
state. There is no legacy compatibility import surface: use the singular
`reasoning.runtime` and `reasoning.verification` packages.

## Module map

```text
reasoning/
├── _immutable.py                   # recursive JSON freezing shared by every record
├── runtime/
│   ├── agent_runtime.py            # AgentTask/AgentTurn/AgentResult and the AgentRuntime ABC
│   ├── alphaapollo_agent_runtime.py# the batched model-to-Environment loop
│   ├── trajectory_slot.py          # per-task mutable state that loop drives
│   ├── external_agent_runtime.py   # third-party agents behind the same contract
│   └── external/                   # one module per agent vendor; see below
└── verification/
    ├── base.py                     # VerificationRequest/Result and the Verifier ABC
    ├── agent.py                    # agent-backed verification over an AgentRuntime
    ├── deterministic.py            # recomputation and the sole certification policy
    └── witness.py                  # durable, candidate-bound recomputation evidence
```

`runtime/external/` is the registry the composition root selects a vendor
through: `cli.py` owns spawning, timeouts, and the process group for all of
them; `agents/` holds one module per vendor supplying only its argument vector
and its stream parser; `bridge/` serves AlphaApollo's own tools to an agent
that would otherwise use its own, and only `bridge/mcp_server.py` needs the
optional `mcp` extra.

## Multimodal request history

Native runtimes may set
`runtimes.<name>.options.image_history_messages` to a positive integer. Before
each Generation request, AlphaApollo keeps image blocks only in the newest N
image-bearing messages; older messages retain their text and receive a neutral
omission placeholder. The complete trajectory remains unchanged. `null` keeps
the historical unbounded behavior.

The unit is deliberately a message, not an Environment turn: multiple image
blocks inside one message count once, while multiple image-bearing messages
returned by one continuation count separately. Chat-style `image_url` blocks,
Responses-style `input_image` blocks, and generic `image` blocks are recognized;
the replacement text block preserves the Chat or Responses content dialect.

## The invariant that matters

Passing is not certifying. `VerificationResult.certified` requires a `pass`
verdict, `trust_level >= 2`, `false_positive_risk == 0`, and a witness, and
`__post_init__` refuses any other combination. `DeterministicVerifier` grants
it only after re-reading the persisted witness through the checker that owns
the store, so evidence that no longer matches its content fails closed.
`AgentVerifier` never certifies. A certified answer that is nonetheless wrong
is a verifier false positive and is counted separately, because otherwise it
reads as ordinary model error.
