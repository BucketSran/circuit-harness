"""Real native Harbor Docker transfers and grading of synthetic file contracts."""

import asyncio
import json
import os
from uuid import uuid4

import pytest

pytest.importorskip("harbor")

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode
from harbor.models.trial.paths import TrialPaths
from test_artifact_delivery import verifier_fixture

from circuit_harness.harbor.artifact_delivery import ArtifactDelivery


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_ARTIFACT_IMAGE"), reason="requires an explicitly installed image"
)
@pytest.mark.parametrize("layout", ["single", "multi", "directory", "wrong-helper"])
def test_native_docker_delivery_and_separate_original_verifier(tmp_path, layout):
    image = os.environ["CHIPS_TEST_ARTIFACT_IMAGE"]
    assert image.startswith("sha256:"), "use an existing immutable image ID; tests never pull"
    starter = tmp_path / "starter"
    (starter / "testbench").mkdir(parents=True)
    (starter / "circuit.va").write_text("STARTER")
    (starter / "testbench/public.py").write_text("TRUSTED PUBLIC")
    expected = {"circuit.va": "CORRECT", "testbench/public.py": "TRUSTED PUBLIC"}
    artifacts = ["/app/circuit.va"]
    fixed = []
    commands = ["printf CORRECT > /app/circuit.va", "chmod 755 /app/circuit.va"]
    if layout != "single":
        (starter / "modules").mkdir()
        (starter / "modules/helper.va").write_text("OLD HELPER")
        commands.append("printf 'NEW HELPER' > /app/modules/helper.va")
        expected["modules/helper.va"] = "NEW HELPER"
        artifacts.append("/app/modules" if layout == "directory" else "/app/modules/helper.va")
        if layout == "wrong-helper":
            commands.append("printf 'WRONG HELPER' > /app/modules/helper.va")
    if layout == "directory":
        (starter / "modules/old.va").write_text("OLD")
        (starter / "modules/fixed.inc").write_text("TRUSTED FIXED")
        (starter / "modules/fixed.sh").write_text("#!/bin/sh\nprintf TRUSTED\n")
        (starter / "modules/fixed.sh").chmod(0o755)
        fixed = ["/app/modules/fixed.inc", "/app/modules/fixed.sh"]
        commands.extend(
            [
                "mv /app/modules/helper.va /app/modules/renamed.va",
                "rm /app/modules/old.va",
                "mkdir /app/modules/empty",
                "printf NEW > /app/modules/new.va",
                "printf TAMPERED > /app/modules/fixed.inc",
                "printf TAMPERED > /app/modules/fixed.sh",
                "chmod 644 /app/modules/fixed.sh",
            ]
        )
        del expected["modules/helper.va"]
        expected.update(
            {
                "modules/renamed.va": "NEW HELPER",
                "modules/new.va": "NEW",
                "modules/fixed.inc": "TRUSTED FIXED",
                "modules/fixed.sh": "#!/bin/sh\nprintf TRUSTED\n",
            }
        )
    commands.extend(
        [
            "printf TAMPERED > /app/testbench/public.py",
            "printf SCRATCH > /app/scratch.log",
        ]
    )
    definition = tmp_path / "definition"
    definition.mkdir()
    (definition / "Dockerfile").write_text("FROM fixture\n")
    token = "delivery-" + uuid4().hex[:10]
    paths = {role: TrialPaths(tmp_path / role) for role in ("agent", "verifier")}
    paths["verifier"].verifier_dir.mkdir(parents=True)
    config = EnvironmentConfig(docker_image=image, network_mode=NetworkMode.NO_NETWORK)
    environments = {}
    for role in paths:
        environments[role] = DockerEnvironment(
            environment_dir=definition,
            environment_name="delivery-fixture",
            session_id=f"{token}__{role}",
            trial_paths=paths[role],
            task_env_config=config,
            mounts=[]
            if role == "agent"
            else [
                {
                    "type": "bind",
                    "source": str(paths[role].verifier_dir),
                    "target": "/logs/verifier",
                }
            ],
        )

    async def run():
        source, target = environments["agent"], environments["verifier"]
        try:
            await source.start(force_build=False)
            await target.start(force_build=False)
            await source.upload_dir(starter, "/app")
            await target.upload_dir(starter, "/app")
            for command in commands:
                result = await source.exec(command)
                assert result.return_code == 0, result
            hidden = await source.exec("test ! -e /tests && test ! -e /solution")
            assert hidden.return_code == 0
            verifier = verifier_fixture(tmp_path, target, expected, paths=paths["verifier"])
            delivery = ArtifactDelivery(
                starter,
                artifacts,
                task_id=f"synthetic-{layout}",
                task_version="v1",
                fixed_paths=fixed,
                verifier_task=verifier.task,
            )
            frozen = await delivery.collect(source, tmp_path / "frozen")
            result = await delivery.grade(verifier, frozen)
            assert result.rewards == {"reward": 0 if layout == "wrong-helper" else 1}
            receipt = json.loads(
                (paths["verifier"].verifier_dir / "delivery-result.json").read_text()
            )
            assert receipt["candidate_sha256"] == delivery.verify(frozen)["candidate_sha256"]
            hidden = await source.exec("test ! -e /tests && test ! -e /logs/verifier/reward.json")
            assert hidden.return_code == 0
            if layout == "directory":
                probe = await target.exec(
                    "test -d /app/modules/empty && test ! -e /app/modules/old.va"
                )
                assert probe.return_code == 0
                probe = await target.exec("/app/modules/fixed.sh")
                assert probe.return_code == 0 and probe.stdout == "TRUSTED"
        finally:
            for environment in reversed(list(environments.values())):
                await environment.stop(delete=True)

    asyncio.run(asyncio.wait_for(run(), timeout=180))
