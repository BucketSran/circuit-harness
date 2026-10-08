# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The environment-graded grader id.

Robotics rows (and any environment-backed task) are not scored by comparing
strings: the execution backend's success predicate decides the outcome at
rollout time. Rows from such preparers carry
``grader_id="environment_success"`` so the id resolves everywhere a
prepared dataset is read, while ``offline_scoreable = False`` tells the
offline scorer's configuration-time gate
(``registry.require_offline_grader``) to refuse the dataset before a
matrix runs.

``grade`` itself must not raise. Raising put the refusal inside the
per-cell ``except Exception`` in ``workflows/scoring.py``, so an
unroutable dataset ran the whole matrix and failed in every cell instead
of being refused at configuration time. A grader with no verdict to give
returns ``None`` — the documented "could not be scored at all" state —
rather than inventing a third kind of outcome.
"""

from __future__ import annotations

from alphaapollo.common.grader.base import Verdict

__all__ = ["EnvironmentSuccessGrader"]


class EnvironmentSuccessGrader:
    """Resolves the id; carries no offline verdict."""

    grader_id = "environment_success"
    # Read by ``registry.require_offline_grader``: the marker every
    # configuration-time offline-scoring gate checks so it refuses this
    # dataset before the run rather than once per cell.
    offline_scoreable = False

    def extract(self, text: str | None) -> str | None:
        # There is no answer text to extract: the environment is the oracle.
        return None

    def grade(self, candidate: str | None, gold: str | None) -> Verdict:
        # Unscoreable, not wrong: the verdict lives in the backend's success
        # predicate, and nothing here can stand in for it.
        return None
