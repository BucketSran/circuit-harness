# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble a Common ``GenerationResponse`` into verl-compatible rollout tensors.

This is the learning-side bridge that proves the point of the whole generation
seam: a neutral, framework-free :class:`~alphaapollo.common.generation.GenerationResponse`
carries exactly what verl needs to build the same padded ``DataProto`` its own
trainer would.

The padding scheme here is a byte-for-byte port of verl's
``AgentLoopWorker._agent_loop_postprocess`` (``verl/experimental/agent_loop/
agent_loop.py``), so a DataProto built from our neutral types is identical to a
native verl rollout for the same prompt and sampled tokens:

  * prompt ids: left-padded to ``prompt_length``
  * response ids: right-padded to ``response_length``
  * ``response_mask`` = (1-for-LLM / 0-for-tool mask, right-padded) * response
    attention mask, so padding is always 0
  * ``input_ids`` / ``attention_mask`` = prompt-block concatenated with
    response-block
  * ``position_ids`` = ``clip(cumsum(attention_mask) - 1, min=0)`` — verl's
    ``compute_position_id_with_mask`` (``verl/utils/model.py``); note trailing
    padding keeps the max position (this matches the *code*, not the idealized
    doc-comment which shows zeros)
  * ``rollout_log_probs``: response logprobs right-padded with 0.0
  * ``rm_scores``: emitted only when an explicit reward is supplied, with the
    reward at the last real response position

The core (:func:`build_rollout_row`) works on plain Python lists so the index
math is unit-testable with no torch. :func:`stack_to_dataproto` is the thin edge
that turns rows into batched tensors and a verl ``DataProto``; it imports torch
and verl lazily and only there. Multi-turn masking (tool tokens -> 0) is the
reasoning layer's job: it fills ``response_mask`` before this runs. Here a
single-turn response is all-ones by construction.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alphaapollo.common.generation import GenerationResponse


def _require_int_tokens(values: Sequence[Any], *, name: str) -> None:
    """Reject non-integer / boolean token ids before any float conversion.

    ``build_rollout_row`` casts token ids to float for a uniform row dtype and the
    stacker later restores int64 tensors; a fractional or boolean id would be
    silently truncated there, so it must be refused at the boundary instead. Any
    integral type is accepted (Python ``int`` and NumPy integers alike, since
    Learning feeds rows out of NumPy/torch); ``bool`` is rejected as a token id.
    """
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise RolloutTensorError(f"{name} must contain only integer token ids")


def _require_finite(value: Any, *, name: str) -> None:
    # Accept any real number (Python or NumPy float/int); reject bool, non-numeric,
    # and NaN/Inf. math.isfinite converts NumPy scalars to float cleanly.
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise RolloutTensorError(f"{name} must be a finite number")


# The verl DataProto field names this builder may emit, in a stable order.
REQUIRED_ROLLOUT_TENSOR_FIELDS = (
    "prompts",
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
)
OPTIONAL_ROLLOUT_TENSOR_FIELDS = ("rm_scores",)
ROLLOUT_TENSOR_FIELDS = REQUIRED_ROLLOUT_TENSOR_FIELDS + OPTIONAL_ROLLOUT_TENSOR_FIELDS


class RolloutTensorError(ValueError):
    """Raised when a rollout row cannot be assembled into verl tensors."""


def build_rollout_row(
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    response_mask: Sequence[int],
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
    response_logprobs: Sequence[float] | None = None,
    reward_score: float | None = None,
) -> dict[str, list[float]]:
    """Pad one trajectory into verl's rollout row, as plain lists.

    ``response_mask`` is the 1-for-LLM / 0-for-tool mask over ``response_ids``
    (both the same length). For a single-turn response it is all ones; the
    reasoning layer supplies the tool zeros for multi-turn. Returns a dict keyed
    by :data:`ROLLOUT_TENSOR_FIELDS`, every value length ``prompt_length +
    response_length`` for ``input_ids`` / ``attention_mask`` / ``position_ids``
    and ``response_length`` for the response-side fields.
    """
    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    response_mask = list(response_mask)
    # Validate values before any float cast: the stacker restores int64 tensors, so
    # a fractional token id, a non-binary mask, or a NaN/Inf logprob would silently
    # corrupt the row rather than raise.
    _require_int_tokens(prompt_ids, name="prompt_ids")
    _require_int_tokens(response_ids, name="response_ids")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, numbers.Integral):
        raise RolloutTensorError("pad_token_id must be an integer token id")
    for mask_value in response_mask:
        if isinstance(mask_value, bool) or mask_value not in (0, 1):
            raise RolloutTensorError("response_mask must contain only 0 or 1")
    if response_logprobs is not None:
        for logprob in response_logprobs:
            _require_finite(logprob, name="response_logprobs")
    if reward_score is not None:
        _require_finite(reward_score, name="reward_score")
        if len(response_ids) == 0:
            # verl places the reward at the last real response token; with none there
            # is nowhere to place it, so fail loudly instead of dropping it silently.
            raise RolloutTensorError("cannot place reward_score on an empty response")
    if len(response_ids) != len(response_mask):
        raise RolloutTensorError(
            f"response_ids ({len(response_ids)}) and response_mask ({len(response_mask)}) "
            "must be the same length"
        )
    if len(prompt_ids) > prompt_length:
        raise RolloutTensorError(
            f"prompt of {len(prompt_ids)} tokens exceeds prompt_length {prompt_length}"
        )
    if len(response_ids) > response_length:
        raise RolloutTensorError(
            f"response of {len(response_ids)} tokens exceeds response_length {response_length}"
        )
    if response_logprobs is not None and len(response_logprobs) != len(response_ids):
        raise RolloutTensorError("response_logprobs must align with response_ids")

    real_prompt = len(prompt_ids)
    real_resp = len(response_ids)
    left_pad = prompt_length - real_prompt
    right_pad = response_length - real_resp

    # prompt: LEFT pad (verl: [0,0,0,0,1,2,3,4]); response: RIGHT pad.
    prompt_block = [pad_token_id] * left_pad + prompt_ids
    prompt_attn = [0] * left_pad + [1] * real_prompt
    response_block = response_ids + [pad_token_id] * right_pad
    response_attn = [1] * real_resp + [0] * right_pad

    # response_mask = (mask right-padded with 0) * response attention mask.
    mask_padded = response_mask + [0] * right_pad
    padded_response_mask = [m * a for m, a in zip(mask_padded, response_attn, strict=True)]

    input_ids = prompt_block + response_block
    attention_mask = prompt_attn + response_attn

    # position_ids = clip(cumsum(attention_mask) - 1, min=0). Trailing padding
    # keeps the running max, matching verl's compute_position_id_with_mask.
    position_ids: list[int] = []
    running = 0
    for flag in attention_mask:
        running += flag
        position_ids.append(running - 1 if running > 0 else 0)

    row: dict[str, list[float]] = {
        "prompts": [float(x) for x in prompt_block],
        "responses": [float(x) for x in response_block],
        "response_mask": [float(x) for x in padded_response_mask],
        "input_ids": [float(x) for x in input_ids],
        "attention_mask": [float(x) for x in attention_mask],
        "position_ids": [float(x) for x in position_ids],
    }

    if response_logprobs is None and any(m == 1 for m in padded_response_mask):
        # Zeros at loss positions would masquerade as sampling-time logprobs.
        # Absent logprobs are only honest where nothing is trained (mask all 0);
        # a trainable span must carry its real per-token logprobs.
        raise RolloutTensorError(
            "response_logprobs is required when response_mask marks trainable (mask=1) "
            "positions; refusing to emit fabricated zero logprobs at loss positions"
        )
    if response_logprobs is not None:
        row["rollout_log_probs"] = [float(x) for x in response_logprobs] + [0.0] * right_pad
    else:
        row["rollout_log_probs"] = [0.0] * response_length

    if reward_score is not None:
        rm_scores = [0.0] * response_length
        if real_resp > 0:
            # verl places the trajectory reward at the last real response token.
            rm_scores[real_resp - 1] = float(reward_score)
        row["rm_scores"] = rm_scores

    return row


def rollout_response_row(
    response: GenerationResponse,
    *,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
    reward_score: float | None = None,
) -> dict[str, list[float]]:
    """Build one rollout row from a single-turn ``common.generation.GenerationResponse``.

    This is the connecting seam: a token-native Generation response carries exactly
    what verl's DataProto needs. A single generation is one turn with an all-ones
    ``response_mask`` (no tool tokens); multi-turn trajectories go through
    :func:`assemble_trajectory_row`.
    """
    if not response.is_trainable:
        raise RolloutTensorError(
            "GenerationResponse is not trainable (no token-native projection); use a "
            "token-native backend (verl LLM server, or OpenAI + return_token_ids)"
        )
    assert response.prompt_token_ids is not None  # guaranteed by is_trainable
    assert response.response_token_ids is not None
    assert response.response_logprobs is not None
    return build_rollout_row(
        prompt_ids=response.prompt_token_ids,
        response_ids=response.response_token_ids,
        response_mask=[1] * len(response.response_token_ids),
        prompt_length=prompt_length,
        response_length=response_length,
        pad_token_id=pad_token_id,
        response_logprobs=response.response_logprobs,
        reward_score=reward_score,
    )


@dataclass(frozen=True)
class TurnSegment:
    """One contiguous span of a multi-turn trajectory's response side.

    ``trainable`` is True for model-generated tokens (mask 1) and False for
    inserted tool/observation tokens (mask 0). A non-trainable span carries no
    policy logprobs; they are filled with 0.0 and excluded from loss by the mask,
    matching verl's ``response_mask``.
    """

    token_ids: Sequence[int]
    trainable: bool
    logprobs: Sequence[float] | None = None

    def __post_init__(self) -> None:
        # trainable drives the response mask (1 vs 0); a truthy non-bool such as the
        # string "false" would flip a tool span into a trained one, so require a bool.
        if not isinstance(self.trainable, bool):
            raise RolloutTensorError("TurnSegment.trainable must be a bool")
        # A tool/observation span carries no policy logprobs; supplying them is a
        # misclassified-span bug we reject rather than silently discard.
        if not self.trainable and self.logprobs is not None:
            raise RolloutTensorError("a non-trainable TurnSegment must not carry logprobs")


def assemble_trajectory_row(
    prompt_ids: Sequence[int],
    segments: Sequence[TurnSegment],
    *,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
    reward_score: float | None = None,
) -> dict[str, list[float]]:
    """Concatenate multi-turn segments into one verl-shaped rollout row.

    Reference for what the reasoning AgentRuntime does across turns: assistant
    spans (mask 1) and tool spans (mask 0) are concatenated in order into one
    ``response_ids`` / ``response_mask`` / logprob sequence, then padded exactly
    like a single generation. This is what makes a multi-turn Trajectory's
    DataProto match verl's tool-agent-loop output.
    """
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    for segment in segments:
        ids = list(segment.token_ids)
        response_ids.extend(ids)
        response_mask.extend([1 if segment.trainable else 0] * len(ids))
        if segment.trainable:
            if segment.logprobs is None or len(segment.logprobs) != len(ids):
                raise RolloutTensorError("a trainable segment must carry aligned logprobs")
            # Validate before the float() cast: otherwise a bool or a numeric string
            # would be silently laundered into a valid logprob on the multi-turn path,
            # bypassing build_rollout_row's finiteness/type check downstream.
            for logprob in segment.logprobs:
                _require_finite(logprob, name="segment logprobs")
            response_logprobs.extend(float(x) for x in segment.logprobs)
        else:
            # tool/observation tokens have no policy logprob; mask 0 drops them.
            response_logprobs.extend([0.0] * len(ids))
    return build_rollout_row(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        pad_token_id=pad_token_id,
        response_logprobs=response_logprobs,
        reward_score=reward_score,
    )


def stack_to_dataproto(
    rows: Sequence[dict[str, list[float]]],
    *,
    provenances: Sequence[Any] | None = None,
    non_tensors: dict[str, Any] | None = None,
    meta_info: dict[str, Any] | None = None,
    dataproto_class: type[Any] | None = None,
) -> Any:
    """Stack rollout rows into a batched verl ``DataProto`` (lazy torch + verl).

    The int fields become ``int64`` tensors and the float fields ``float32``,
    matching a native verl rollout's dtypes. ``dataproto_class`` can be injected
    for tests; by default verl's ``DataProto`` is imported lazily so importing
    this module never drags in verl.

    ``provenances`` (one per row) is preserved as a batch-aligned ``provenance``
    non-tensor so Learning can verify the rollout came from the expected policy
    weights / tokenizer / chat template before an update. This adapter only
    *retains* provenance; it does not enforce policy match -- that gate lives in
    the Learning training loop, not in this backend-neutral conversion.
    """
    if not rows:
        raise RolloutTensorError("cannot stack an empty batch of rollout rows")

    # Pure-list validation first (no torch): a batch must not mix presence of an
    # optional field across rows, or a missing value would become a fabricated zero.
    fields_to_stack = list(REQUIRED_ROLLOUT_TENSOR_FIELDS)
    for field_name in OPTIONAL_ROLLOUT_TENSOR_FIELDS:
        presence = [field_name in row for row in rows]
        if any(presence) and not all(presence):
            raise RolloutTensorError(
                f"optional field {field_name!r} must be present in every row or none; "
                "refusing to turn a missing value into a fabricated zero"
            )
        if all(presence):
            fields_to_stack.append(field_name)

    import torch  # lazy: torch belongs to the learning runtime, not the import path

    int_fields = {
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
    }

    tensors: dict[str, Any] = {}
    for field_name in fields_to_stack:
        column = [row[field_name] for row in rows]
        if field_name in int_fields:
            tensors[field_name] = torch.tensor(column, dtype=torch.int64)
        else:
            tensors[field_name] = torch.tensor(column, dtype=torch.float32)

    if dataproto_class is None:
        try:
            from verl.protocol import DataProto
        except ImportError as exc:  # pragma: no cover - server runtime concern
            raise ImportError(
                "verl is required to build a DataProto; install the learning runtime "
                "or inject dataproto_class"
            ) from exc
        dataproto_class = DataProto

    combined_non_tensors = dict(non_tensors or {})
    if provenances is not None:
        if len(provenances) != len(rows):
            raise RolloutTensorError(
                f"provenances ({len(provenances)}) must align with rows ({len(rows)})"
            )
        if "provenance" in combined_non_tensors:
            raise RolloutTensorError("non_tensors already carries a 'provenance' key")
        import numpy as np  # lazy: numpy rides with the learning runtime, not the import path

        combined_non_tensors["provenance"] = np.array(
            [_provenance_to_dict(item) for item in provenances], dtype=object
        )

    return dataproto_class.from_dict(
        tensors=tensors,
        non_tensors=combined_non_tensors,
        meta_info=meta_info or {},
    )


def _provenance_to_dict(value: Any) -> dict[str, Any]:
    """Normalize a Provenance (or mapping) into a plain dict for the non-tensor."""

    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise RolloutTensorError("each provenance must be a Provenance dataclass or a mapping")


__all__ = [
    "OPTIONAL_ROLLOUT_TENSOR_FIELDS",
    "REQUIRED_ROLLOUT_TENSOR_FIELDS",
    "ROLLOUT_TENSOR_FIELDS",
    "RolloutTensorError",
    "TurnSegment",
    "assemble_trajectory_row",
    "build_rollout_row",
    "rollout_response_row",
    "stack_to_dataproto",
]
