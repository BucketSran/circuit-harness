# AlphaApollo prompts

`alphaapollo.common.prompts` is the canonical source for model-facing prompts
shared by Learning, Reasoning, Workflow composition, and dataset export.

## Ownership

| Content | Canonical location | Reader |
| --- | --- | --- |
| Role system and user prompts | `resources/roles/` | Workflow and dataset export |
| Native tool schemas | `../execution/tools/base.py` | `ToolCatalog` and Workflow composition |
| Dynamic observations and legal actions | `../environment/` | Environment adapters |
| Provider message conversion | `../generation/` | Generation backends |
| Algorithm-only optimization prompts | owning package under `learning/` | owning algorithm |

Shared prompt resources are data and rendering contracts. Consumers still own
prompt selection, orchestration, termination, visibility, reward, and retry
policy. Runtime and Generation code transport resolved messages without adding
prompt policy.

## Reading prompts

Resource paths define their stable references: for example,
`resources/roles/model_default.yaml` is `roles.model_default`. IDs are not
repeated inside YAML files.

```python
from alphaapollo.common.prompts import resolve_prompt

prompt = resolve_prompt("roles.model_default")
initial = prompt.render_user(problem="What is AlphaApollo?")
```

`resolve_prompt()` caches immutable `PromptSpec` records, rejects unknown YAML
keys, and derives each template's required variables from its `{placeholders}`.
Rendering rejects missing and unexpected variables.

`prompt_ref` names a complete resource, not a default that later configuration
may partially replace. It owns both `system` and `user_template`. A Workflow
step cannot override the referenced user template. During dataset export,
`task.public.prompt` (when present) is passed as the value of `{problem}` rather
than replacing the resource template; `{statement}` remains the raw public
statement.

An empty `system` means no system message. `roles.model_default` therefore sends
only its rendered user message. This makes no promise that a provider or model
will add a semantic default system instruction.

Only resources packaged below `alphaapollo.common.prompts.resources` can be
resolved by `prompt_ref`. Project-defined prompt text remains supported through
the inline fields offered by each consumer; arbitrary filesystem paths and
plugin search paths are intentionally outside the current resolver contract.

## Versions and digests

`version` is an author-maintained revision number, not a historical version
selector. Any change to `system`, `user_template`, `continuation_template`, or
`visual_template` that can change rendered text—including whitespace—must
increment it. Comments and formatting outside scalar values do not. Historical
reproduction uses the Git commit containing the resource together with the
recorded `prompt_digest`; the resolver always loads the resource in the current
installation.

`prompt_digest` is a digest of the rendered first-turn system/user text only. It
does not cover tool schemas, continuation or visual prompts, provider chat
templates, model identity, or sampling parameters, and must not be treated as a
complete Generation request fingerprint.

## Consumer variables

Prompt syntax is validated centrally, while each consumer owns the variables it
can supply. Built-in resources and consumers have these contracts:

| Resource or consumer | Template | Available variables |
| --- | --- | --- |
| Workflow `roles.*` | `user_template` | `branch_index`, `candidate`, `feedback`, `input_id`, `iteration`, `metadata`, `model`, `previous_output`, `problem`, `request_id`, `role`, `tools` |
| Dataset export with `prompt_ref` | `user_template` | `problem`, `statement` |

A referenced resource is rejected if its placeholders exceed its consumer's
available variables. Contract tests render every variant of every packaged
resource with the variables supplied by its corresponding consumer.

## Tool schemas

Tool schemas are execution contracts, not duplicated prompt text. Grant tools
by ID and resolve their schemas from the canonical catalog:

```python
from alphaapollo.common.execution.tools import ToolCatalog

schema = ToolCatalog().get("bash").to_openai_tool()
```

Workflow composition reads role tool IDs, resolves them through `ToolCatalog`,
and passes the schemas as `GenerationRequest.tools`. Prompt resources never
copy tool names, parameters, or descriptions.

## Composition

1. A Workflow or Learning configuration selects a `prompt_ref`.
2. `resolve_prompt()` validates and loads the packaged resource.
3. The consumer renders task or environment variables.
4. Workflow resolves granted tools separately through `ToolCatalog`.
5. Runtime receives resolved messages and tool schemas without rewriting them.
6. Dataset exports record the rendered prompt digest and, when used, the prompt
   reference and version.

Inline Workflow prompts remain supported for compatibility. A role must use
either `prompt_ref` or inline `system_prompt`/`input_template`; combining them,
including a step-level input override, is rejected instead of relying on
implicit precedence or string merging.
