"""Fingerprint tokenizer semantics without retaining a training runtime."""

import hashlib
import json
from typing import Any

def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash tokenization semantics while ignoring where the tokenizer was loaded from."""

    payload = {
        "chat_template": str(getattr(tokenizer, "chat_template", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        payload["backend_tokenizer"] = backend_tokenizer.to_str()
    elif callable(getattr(tokenizer, "get_vocab", None)):
        payload["vocab"] = sorted(
            (str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()
        )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
