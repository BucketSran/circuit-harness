"""Frozen candidate grading under Harbor's independently bounded verifier phase."""

import asyncio
import json
import math
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from .config import FinalEvaluationConfig, validate_final_separation


class FrozenCandidateVerifier(BaseVerifier):
    def __init__(
        self,
        *args,
        config_path=None,
        task_bindings=None,
        task_bindings_sha256=None,
        task_binding_receipts=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.final_config = None
        if config_path is not None and task_bindings is not None:
            raise ValueError("scalar config_path and task_bindings are mutually exclusive")
        if task_bindings is not None:
            from .task_bindings import resolve_task_binding, write_binding_receipt

            task_dir = self.task.paths.task_dir.resolve()
            if task_bindings_sha256 is not None and str(task_dir) not in (
                task_binding_receipts or {}
            ):
                raise ValueError("compiled task binding pins are incomplete")
            selection = resolve_task_binding(
                task_bindings,
                task_dir,
                public_roots=[self.trial_paths.trial_dir.parent],
                expected_manifest_sha256=task_bindings_sha256,
                expected_receipt=(task_binding_receipts or {}).get(str(task_dir)),
            )
            config_path = selection.final_config_path
            self.task_binding = selection.receipt
            write_binding_receipt(selection, self.trial_paths.trial_dir)
        if config_path is not None:
            path = Path(config_path).resolve()
            if path.is_relative_to(self.task.paths.task_dir.resolve()) or path.is_relative_to(
                self.trial_paths.trial_dir.resolve()
            ):
                raise ValueError(
                    "verifier config must be outside task and trial export directories"
                )
            self.final_config = FinalEvaluationConfig.model_validate_json(path.read_text())

    async def evaluate(self, environment, candidate):
        from circuit_harness.execution.benchmark_spectre import package_identity

        config = self.final_config
        if config is None:
            config = environment.settings.final_evaluation()
        validate_final_separation(environment.settings, config.task_package, config.remote)
        package = package_identity(config.task_package, purpose="final")
        if any(
            package["manifest"][key] != environment.settings.task[key]
            for key in ("task_id", "task_version")
        ):
            raise ValueError("public and final task declarations differ")
        if config.backend == "benchmark_opensource":
            from .local_evaluation import evaluate_replay

            result = await evaluate_replay(candidate, config, self.trial_paths.verifier_dir)
            return {"state": "completed", **result}
        from circuit_harness.execution.benchmark_remote import RemoteBenchmarkSpectre

        transport = RemoteBenchmarkSpectre(
            config.remote, self.trial_paths.verifier_dir / "transport"
        )
        job_id = f"harbor-{environment.context_id}"
        try:
            state = await asyncio.to_thread(
                transport.submit,
                Path(candidate["candidate_directory"]),
                config.task_package,
                job_id,
            )
            while True:
                if state.get("state") == "finished":
                    # Completion is published before the archive. Keep querying
                    # this job under the verifier deadline until transfer is safe.
                    archive = state.get("archive")
                    archive_state = archive.get("state") if isinstance(archive, dict) else None
                    if archive_state == "verified":
                        break
                    if archive_state not in ("pending", "running"):
                        raise RuntimeError("Final job archive is unavailable or invalid")
                elif state.get("state") not in ("running", "unknown", "queued"):
                    raise RuntimeError("Final job has an invalid execution state")
                await asyncio.sleep(0.25)
                state = await asyncio.to_thread(transport.query, job_id)
            result = await asyncio.to_thread(transport.retrieve, job_id)
            return {"state": "completed", **result}
        finally:
            # Interrupt this client's SSH wait only; preserve the durable server job.
            transport.interrupt_wait()

    async def verify(self):
        candidate = self.environment.frozen
        if candidate is None or not all(
            candidate.get(key) for key in ("candidate_directory", "candidate_sha256")
        ):
            raise RuntimeError("Verifier requires an already frozen candidate")
        evaluation = self.evaluate(self.environment, candidate)
        if self.final_config is None and hasattr(self.environment.settings, "final_timeout_s"):
            # Preserve legacy native configuration. New profiles use Harbor's
            # verifier phase deadline without a duplicate timeout setting.
            report = await asyncio.wait_for(
                evaluation, timeout=self.environment.settings.final_timeout_s
            )
        else:
            report = await evaluation
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        score = report.get("score")
        if (
            report.get("state") != "completed"
            or report.get("execution") != "ok"
            or report.get("verdict") not in ("pass", "fail")
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise RuntimeError("Final evaluation did not produce a valid task score")
        if report.get("candidate_sha256") != candidate["candidate_sha256"]:
            raise RuntimeError("Final evaluation candidate identity mismatch")
        task = self.environment.settings.task
        if (
            any(report.get(key) != task[key] for key in ("task_id", "task_version"))
            or report.get("purpose") != "final"
        ):
            raise RuntimeError("Final evaluation task identity or purpose mismatch")
        return VerifierResult(rewards={"reward": score})
