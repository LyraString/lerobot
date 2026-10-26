"""Validate the paired deployment templates without opening hardware or a model."""

from pathlib import Path

import draccus

from lerobot.remote_inference.configs import ServerConfig
from lerobot.robots.dual_ur5_cb3.dual_ur5_cb3 import STATE_NAMES
from lerobot.scripts.lerobot_rollout import RolloutConfig


def test_paired_ur5_templates_match_names_rates_and_camera_schema():
    examples = Path(__file__).resolve().parents[2] / "examples" / "ur5_real"
    server = draccus.parse(ServerConfig, config_path=examples / "remote_fastwam_server.yaml", args=[])
    client = draccus.parse(RolloutConfig, config_path=examples / "remote_rollout.yaml", args=[])

    assert client.inference.type == "remote"
    assert client.policy is None
    assert not client.robot.enable_motion
    assert not client.robot.enable_gripper
    assert not client.return_to_initial_position
    assert client.robot.type == server.robot_type == "dual_ur5_cb3"
    assert client.inference.deployment == server.deployment
    assert client.inference.semantics == server.semantics
    assert client.fps == server.execution.action_fps == 20
    assert server.action_feature.names == STATE_NAMES
    state = next(feature for feature in server.features if feature.name == "observation.state")
    assert state.names == STATE_NAMES
    for name, camera in client.robot.cameras.items():
        feature = next(feature for feature in server.features if feature.name == f"observation.images.{name}")
        assert feature.kind == "rgb"
        assert feature.dtype == "uint8"
        assert feature.shape == (camera.height, camera.width, 3)
