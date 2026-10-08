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

"""Single-generation generation contract and pluggable backends (Issue #185)."""

from alphaapollo.common.generation.backends.openai_compatible import (
    OpenAICompatibleGenerationBackend,
)
from alphaapollo.common.generation.backends.verl_llm_server import (
    VerlLLMServerGenerationBackend,
)
from alphaapollo.common.generation.base import (
    BATCH_EXECUTION_CLIENT_CONCURRENT,
    BATCH_EXECUTION_SEQUENTIAL,
    BATCH_EXECUTION_SERVER_BATCHED,
    POLICY_SOURCE_LOCAL_ENGINE,
    POLICY_SOURCE_REMOTE_ENGINE,
    BackendCapabilities,
    CapabilityError,
    GenerationBackend,
    GenerationError,
    GenerationRequest,
    GenerationResponse,
    Provenance,
    SamplingOptions,
    ToolCall,
    fingerprint,
    reject_standard_key_overrides,
    require_trainable,
    verify_batch_identity,
    with_backend_metadata,
)

__all__ = [
    "BATCH_EXECUTION_CLIENT_CONCURRENT",
    "BATCH_EXECUTION_SEQUENTIAL",
    "BATCH_EXECUTION_SERVER_BATCHED",
    "POLICY_SOURCE_LOCAL_ENGINE",
    "POLICY_SOURCE_REMOTE_ENGINE",
    "BackendCapabilities",
    "CapabilityError",
    "OpenAICompatibleGenerationBackend",
    "Provenance",
    "GenerationBackend",
    "GenerationError",
    "GenerationRequest",
    "GenerationResponse",
    "SamplingOptions",
    "ToolCall",
    "VerlLLMServerGenerationBackend",
    "fingerprint",
    "reject_standard_key_overrides",
    "require_trainable",
    "verify_batch_identity",
    "with_backend_metadata",
]
