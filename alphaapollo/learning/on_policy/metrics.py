# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""AlphaApollo trainer episode metrics (stateless, absent-safe).

Emits episode/* metrics from DataProto non_tensor_batch fields.
Keys are only emitted when their source field is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from verl import DataProto


def compute_episode_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute episode-level training metrics from a DataProto batch.

    Absent-safe: returns only keys present in non_tensor_batch.
    Single-turn proof (no traj_uid/tool_callings/success_rate) yields empty dict cleanly.

    Key-name compatibility: the live multi-turn rollout writes ``tool_callings`` and
    ``success_rate`` into non_tensor_batch (see rollout_loop.py / env_manager.py), while
    earlier synthetic fixtures used ``tool_call_count`` / ``success``. Both spellings are
    accepted so this helper works on real training batches and on the legacy unit fixtures.

    Args:
        batch: DataProto whose non_tensor_batch may contain any subset of
            {traj_uid, tool_callings|tool_call_count, success_rate|success}.
            Missing fields are silently skipped — no KeyError or synthetic defaults.

    Returns:
        dict with a subset of:
            episode/num_trajectories: int — distinct traj_uid values.
            episode/tool_call_count: float — mean tool-call count per trajectory.
            episode/success_rate: float — mean success rate (cast to float).
    """
    metrics: dict[str, Any] = {}
    ntb = batch.non_tensor_batch
    if "traj_uid" in ntb:
        metrics["episode/num_trajectories"] = len(set(ntb["traj_uid"].tolist()))
    # tool calls: live rollout writes "tool_callings"; accept legacy "tool_call_count".
    for tc_key in ("tool_callings", "tool_call_count"):
        if tc_key in ntb:
            metrics["episode/tool_call_count"] = float(
                np.mean(np.asarray(ntb[tc_key], dtype=float))
            )
            break
    # success: live rollout writes "success_rate" (a batch-mean win rate broadcast to
    # every row); accept legacy boolean "success".
    for sc_key in ("success_rate", "success"):
        if sc_key in ntb:
            metrics["episode/success_rate"] = float(np.mean(np.asarray(ntb[sc_key], dtype=float)))
            break
    return metrics
