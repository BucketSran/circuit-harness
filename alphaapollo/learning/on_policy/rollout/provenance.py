# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Learning-owned rollout policy provenance without trainer dependencies."""

from __future__ import annotations

import json
from typing import Any

from alphaapollo.common.generation import fingerprint


def tokenizer_fingerprints(tokenizer: Any, tokenizer_id: str) -> tuple[str, str]:
    """Fingerprint the tokenizer revision and exact chat template used for rollout."""

    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    revision = str(
        getattr(tokenizer, "_commit_hash", "")
        or init_kwargs.get("_commit_hash", "")
        or init_kwargs.get("revision", "")
    )
    added_tokens = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    tokenizer_identity = json.dumps(
        {
            "id": tokenizer_id,
            "revision": revision,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "added_tokens": sorted(
                (str(token), int(index)) for token, index in added_tokens.items()
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        get_chat_template = getattr(tokenizer, "get_chat_template", None)
        chat_template = get_chat_template() if callable(get_chat_template) else None
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise ValueError("training tokenizer must expose the chat template used for rollout")

    return fingerprint(tokenizer_identity), fingerprint(chat_template)


def synchronize_rollout_weights(
    *,
    checkpoint_manager: Any,
    generation_backend: Any,
    global_step: int,
    previous_sync_count: int,
) -> int:
    """Synchronize replicas, publish their new identity, and return its counter."""

    checkpoint_manager.update_weights(global_step)
    sync_count = previous_sync_count + 1
    generation_backend.update_weights_version(f"step-{global_step}-sync-{sync_count}")
    return sync_count


__all__ = ["synchronize_rollout_weights", "tokenizer_fingerprints"]
