import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.dual_ur5_cb3 import DualUR5CB3, DualUR5CB3Config, dual_ur5_cb3 as module
from lerobot.robots.dual_ur5_cb3.dual_ur5_cb3 import STATE_NAMES
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.feature_utils import hw_to_dataset_features


class FakeGripper:
    def __init__(self):
        self.position = 51
        self.active = True
        self.commands = []
        self.disconnected = False

    def connect(self, *args):
        self.connection = args

    def disconnect(self):
        self.disconnected = True

    def get_open_position(self):
        return 0

    def get_closed_position(self):
        return 255

    def get_current_position(self):
        return self.position

    def is_active(self):
        return self.active

    def move(self, position, speed, force):
        self.commands.append((position, speed, force))
        return True, position


class FakeController:
    def __init__(self, robot_ip, **kwargs):
        self.robot_ip = robot_ip
        self.kwargs = kwargs
        self.joints = np.arange(6, dtype=float) / 10 + (0 if robot_ip.endswith("50") else 1)
        self.rtde_r = MagicMock()
        self.rtde_r.getActualQ.side_effect = lambda: self.joints.copy()
        self.rtde_r.isConnected.return_value = True
        self.rtde_r.getSafetyMode.return_value = 1
        self.rtde_r.getRobotMode.return_value = 7
        self.rtde_c = MagicMock()
        self.gripper = FakeGripper()
        self.commands = []
        self.closed = False
        self.on_servo = None

    def get_current_joint_positions(self):
        return self.joints.copy()

    def servo_joints(self, command):
        self.commands.append(command.copy())
        if self.on_servo is not None:
            self.on_servo()

    def close(self):
        self.closed = True


class FakeCamera:
    def __init__(self):
        self.is_connected = False
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.error = None

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def read_latest(self, max_age_ms):
        self.max_age_ms = max_age_ms
        if self.error is not None:
            raise self.error
        return self.image


@pytest.fixture
def driver(monkeypatch):
    controllers = {}

    def make_controller(robot_ip, **kwargs):
        controller = FakeController(robot_ip, **kwargs)
        controllers[robot_ip] = controller
        return controller

    fake = SimpleNamespace(
        URController=MagicMock(side_effect=make_controller),
        rtde_receive=SimpleNamespace(
            RTDEReceiveInterface=MagicMock(side_effect=lambda ip: make_controller(ip).rtde_r)
        ),
        RobotiqGripper=FakeGripper,
        controllers=controllers,
    )
    monkeypatch.setattr(module, "_load_driver", lambda root: fake)
    monkeypatch.setattr(
        module, "make_cameras_from_configs", lambda cameras: {name: FakeCamera() for name in cameras}
    )
    threads = MagicMock()
    threads.return_value.is_alive.return_value = False
    monkeypatch.setattr(module, "Thread", threads)
    return fake


@pytest.fixture
def robot(driver, tmp_path):
    instance = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path))
    yield instance
    instance.disconnect()


def action_for(robot, delta=0):
    observation = robot.get_observation()
    action = {name: observation[name] for name in STATE_NAMES}
    for name in STATE_NAMES[:12]:
        action[name] += delta
    return action


def arm_robot(driver, tmp_path, **kwargs):
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, enable_motion=True, **kwargs))
    robot.connect()
    return robot


def test_factory_and_feature_order(robot):
    made = make_robot_from_config(robot.config)
    assert isinstance(made, DualUR5CB3)
    assert list(robot.action_features) == list(STATE_NAMES)
    features = hw_to_dataset_features(robot.observation_features, "observation")
    assert features["observation.state"]["shape"] == (14,)
    assert features["observation.state"]["names"] == list(STATE_NAMES)


def test_official_remote_hold_retains_applied_targets_without_camera_reads(driver, tmp_path, monkeypatch):
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    robot = arm_robot(driver, tmp_path)
    try:
        wrapper = ThreadSafeRobot(robot)
        wrapper.configure_position_hold()
        measured = wrapper.get_observation()
        monkeypatch.setattr(robot, "get_observation", lambda: pytest.fail("hold must not read cameras"))
        wrapper.hold()
        np.testing.assert_allclose(robot._target, [measured[name] for name in STATE_NAMES])
        target = {name: measured[name] + (0.1 if name in STATE_NAMES[:12] else 0) for name in STATE_NAMES}
        applied = wrapper.send_action(target)
        previous_refresh = robot._last_action_time
        wrapper.hold()
        np.testing.assert_allclose(robot._target, [applied[name] for name in STATE_NAMES])
        assert robot._last_action_time >= previous_refresh
        assert not wrapper.hardware_failure
        assert all(not controller.gripper.commands for controller in robot.controllers.values())
    finally:
        robot.disconnect()


def test_read_only_connect_and_action_do_not_create_motion_interface(robot, driver):
    robot.connect()
    driver.URController.assert_not_called()
    observation = robot.get_observation()
    np.testing.assert_allclose([observation[name] for name in STATE_NAMES[:6]], np.arange(6) / 10)
    np.testing.assert_allclose([observation[name] for name in STATE_NAMES[6:12]], np.arange(6) / 10 + 1)
    assert observation["left_gripper.pos"] == pytest.approx(0.2)
    assert observation["right_gripper.pos"] == pytest.approx(0.2)
    sent = robot.send_action(action_for(robot, delta=0.01))
    assert sent == {name: observation[name] for name in STATE_NAMES}
    assert not robot._threads
    assert not robot._ready.is_set()
    assert all(not controller.gripper.commands for controller in robot.controllers.values())
    assert all(camera.max_age_ms == 200 for camera in robot.cameras.values())
    robot.disconnect()
    assert not robot.is_connected
    for controller in driver.controllers.values():
        controller.rtde_r.disconnect.assert_called_once()


@pytest.mark.parametrize(
    "updates",
    [
        {"servo_time": 0},
        {"max_joint_velocity": -1},
        {"max_relative_target": float("nan")},
        {"action_timeout_s": 0.001},
        {"joint_limit": float("inf")},
        {"lookahead_time": 0.01},
        {"servo_gain": 99},
        {"gripper_force": 256},
        {"camera_max_age_ms": 0},
        {"right_robot_ip": "192.168.1.50"},
        {"enable_motion": True, "dry_run_report_only": True},
    ],
)
def test_invalid_config(updates):
    with pytest.raises(ValueError):
        DualUR5CB3Config(**updates)


def test_connection_guards(robot):
    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    robot.connect()
    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()


@pytest.mark.parametrize("invalid", ["missing", "extra", "nan", "jump", "joint_limit"])
def test_invalid_action_latches_fault(robot, invalid):
    robot.connect()
    action = action_for(robot)
    if invalid == "missing":
        del action[STATE_NAMES[0]]
    elif invalid == "extra":
        action["unexpected.pos"] = 0
    elif invalid == "nan":
        action[STATE_NAMES[0]] = float("nan")
    elif invalid == "joint_limit":
        action[STATE_NAMES[0]] = 2 * math.pi + 0.01
    else:
        action[STATE_NAMES[0]] += robot.config.max_relative_target + 0.01
    with pytest.raises(ValueError):
        robot.send_action(action)
    assert robot._stop.is_set()
    with pytest.raises(RuntimeError, match="servo stopped"):
        robot.get_observation()


@pytest.mark.parametrize("enable_motion", [False, True])
@pytest.mark.parametrize(
    "joint_index, limit_name, direction",
    [
        (0, "joint_limit", 1),
        (11, "joint_limit", -1),
        (5, "max_relative_target", 1),
        (6, "max_relative_target", -1),
    ],
)
def test_joint_limit_error_reports_rejected_values(
    driver, tmp_path, enable_motion, joint_index, limit_name, direction
):
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, enable_motion=enable_motion))
    robot.connect()
    try:
        action = action_for(robot)
        joint_name = STATE_NAMES[joint_index]
        measured = action[joint_name]
        if limit_name == "joint_limit":
            action[joint_name] = direction * (robot.config.joint_limit + 0.01)
        else:
            action[joint_name] += direction * (robot.config.max_relative_target + 0.01)
        with pytest.raises(ValueError, match=f"Joint target exceeds {limit_name}") as raised:
            robot.send_action(action)
        message = str(raised.value)
        assert joint_name in message
        assert f"({getattr(robot.config, limit_name):.6f} rad)" in message
        assert f"target={action[joint_name]:.6f} rad" in message
        assert f"measured={measured:.6f} rad" in message
        assert f"delta={action[joint_name] - measured:.6f} rad" in message
        assert f"enable_motion={enable_motion}" in message
        assert "previous_target=unavailable" in message
        assert "last_sent_command=unavailable" in message
        assert "rejected action was not sent" in message
        assert robot._target is None
        assert not robot._ready.is_set()
        assert robot._stop.is_set()
        assert all(not controller.commands for controller in driver.controllers.values())
        assert all(not controller.gripper.commands for controller in driver.controllers.values())
    finally:
        robot.disconnect()


def test_report_only_dry_run_records_relative_violation_without_motion(driver, tmp_path, caplog):
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, dry_run_report_only=True))
    robot.connect()
    try:
        observation = robot.get_observation()
        action = action_for(robot)
        driver.controllers[robot.config.left_robot_ip].joints[5] = 3.005212
        action[STATE_NAMES[5]] = 3.315335
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=module.__name__):
            sent = robot.send_action(action)
            robot.send_action(action)
        assert sent[STATE_NAMES[5]] == pytest.approx(3.005212)
        assert sent[STATE_NAMES[0]] == observation[STATE_NAMES[0]]
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Dry-run action #1" in message
        assert "left_joint_6.pos" in message
        assert "delta=0.310123 rad" in message
        assert "not approved for motion" in message
        assert any("measured:" in record.getMessage() for record in caplog.records)
        assert not robot._stop.is_set()
        assert robot._target is None
        assert robot._last_action_time == 0
        assert not robot._ready.is_set()
        assert not robot._threads
        robot.get_observation()
        driver.URController.assert_not_called()
        assert all(not controller.commands for controller in driver.controllers.values())
        assert all(not controller.gripper.commands for controller in robot.controllers.values())
    finally:
        robot.disconnect()


@pytest.mark.parametrize("invalid", ["missing", "extra", "nan", "joint_limit", "state"])
def test_report_only_dry_run_still_rejects_invalid_data(robot, driver, invalid):
    robot.config.dry_run_report_only = True
    robot.connect()
    action = action_for(robot)
    if invalid == "missing":
        del action[STATE_NAMES[0]]
    elif invalid == "extra":
        action["unexpected.pos"] = 0
    elif invalid == "nan":
        action[STATE_NAMES[0]] = float("nan")
    elif invalid == "joint_limit":
        action[STATE_NAMES[0]] = 2 * math.pi + 0.01
    else:
        driver.controllers[robot.config.left_robot_ip].joints[0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        robot.send_action(action)
    assert robot._stop.is_set()
    assert robot._target is None
    assert not robot._ready.is_set()


def test_motion_checks_remain_strict_if_report_only_flag_is_changed(driver, tmp_path):
    robot = arm_robot(driver, tmp_path)
    try:
        assert not robot.config.clip_relative_target
        robot.config.dry_run_report_only = True
        with pytest.raises(ValueError, match="max_relative_target"):
            robot.send_action(action_for(robot, delta=0.31))
        assert robot._stop.is_set()
        assert robot._target is None
        assert not robot._ready.is_set()
        assert all(not controller.commands for controller in driver.controllers.values())
    finally:
        robot.disconnect()


@pytest.mark.parametrize("side,joint_index", [("left", 4), ("right", 10)])
@pytest.mark.parametrize("direction", [-1, 1])
def test_motion_clips_relative_target_with_official_helper(
    driver, monkeypatch, tmp_path, side, joint_index, direction
):
    safe_goals = MagicMock(wraps=module.ensure_safe_goal_position)
    monkeypatch.setattr(module, "ensure_safe_goal_position", safe_goals)
    robot = arm_robot(driver, tmp_path, clip_relative_target=True)
    try:
        action = action_for(robot)
        joint_name = STATE_NAMES[joint_index]
        measured = action[joint_name]
        action[joint_name] += direction * 0.7
        sent = robot.send_action(action)

        safe_goals.assert_called_once()
        assert safe_goals.call_args.args[0][joint_name] == (action[joint_name], measured)
        assert safe_goals.call_args.args[1] == 0.3
        assert sent[joint_name] == pytest.approx(measured + direction * 0.3)
        assert robot._target[joint_index] == pytest.approx(sent[joint_name])
        assert not robot._stop.is_set()
        assert robot._worker_error is None
        assert all(not controller.commands for controller in robot.controllers.values())

        controller = robot.controllers[side]
        controller.on_servo = robot._stop.set
        robot._run_servo(side)
        assert controller.commands[0][joint_index % 6] == pytest.approx(
            measured + direction * robot.config.max_joint_velocity * robot.config.servo_time
        )
        robot._stop.clear()
        action[joint_name] += direction * 0.1
        sent = robot.send_action(action)
        assert sent[joint_name] == pytest.approx(measured + direction * 0.3)
        assert safe_goals.call_count == 2
        assert robot._target_action_index == 2
        assert not robot._stop.is_set()
    finally:
        robot.disconnect()


@pytest.mark.parametrize("invalid", ["missing", "extra", "nan", "joint_limit", "state"])
def test_clipping_does_not_bypass_invalid_action_or_state(driver, tmp_path, invalid):
    robot = arm_robot(driver, tmp_path, clip_relative_target=True)
    try:
        action = action_for(robot)
        if invalid == "missing":
            del action[STATE_NAMES[0]]
        elif invalid == "extra":
            action["unexpected.pos"] = 0
        elif invalid == "nan":
            action[STATE_NAMES[0]] = float("nan")
        elif invalid == "joint_limit":
            action[STATE_NAMES[0]] = robot.config.joint_limit + 0.01
        else:
            robot.controllers["left"].joints[0] = float("nan")
        with pytest.raises((ValueError, RuntimeError)):
            robot.send_action(action)
        assert robot._stop.is_set()
        assert robot._target is None
        assert not robot._ready.is_set()
        assert all(not controller.commands for controller in robot.controllers.values())
    finally:
        robot.disconnect()


@pytest.mark.parametrize("direction", [-1, 1])
def test_clipped_target_cannot_exceed_absolute_joint_limit(driver, tmp_path, direction):
    robot = arm_robot(driver, tmp_path, clip_relative_target=True)
    try:
        robot.controllers["left"].joints[4] = direction * (robot.config.joint_limit + 0.4)
        action = action_for(robot)
        action[STATE_NAMES[4]] = 0
        with pytest.raises(ValueError, match="Clipped joint target exceeds joint_limit"):
            robot.send_action(action)
        assert robot._stop.is_set()
        assert robot._target is None
        assert not robot._ready.is_set()
        assert all(not controller.commands for controller in robot.controllers.values())
    finally:
        robot.disconnect()


@pytest.mark.parametrize("report_only", [False, True])
def test_clipping_setting_does_not_change_read_only_checks(driver, monkeypatch, tmp_path, report_only):
    safe_goals = MagicMock(wraps=module.ensure_safe_goal_position)
    monkeypatch.setattr(module, "ensure_safe_goal_position", safe_goals)
    robot = DualUR5CB3(
        DualUR5CB3Config(calibration_dir=tmp_path, clip_relative_target=True, dry_run_report_only=report_only)
    )
    robot.connect()
    try:
        action = action_for(robot, delta=0.31)
        if report_only:
            sent = robot.send_action(action)
            assert sent[STATE_NAMES[0]] == pytest.approx(0)
            assert not robot._stop.is_set()
        else:
            with pytest.raises(ValueError, match="max_relative_target"):
                robot.send_action(action)
            assert robot._stop.is_set()
        safe_goals.assert_not_called()
        assert robot._target is None
        assert not robot._ready.is_set()
        driver.URController.assert_not_called()
    finally:
        robot.disconnect()


def test_camera_failure_latches_fault(robot):
    robot.connect()
    robot.cameras["exterior"].error = TimeoutError("stale frame")
    with pytest.raises(TimeoutError, match="stale frame"):
        robot.get_observation()
    assert robot._stop.is_set()


def test_bad_camera_shape_is_rejected(robot):
    robot.connect()
    robot.cameras["left_wrist"].image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="invalid image"):
        robot.get_observation()


def test_active_connect_starts_no_servo_until_first_action(driver, tmp_path):
    robot = arm_robot(driver, tmp_path)
    try:
        assert not robot._ready.is_set()
        assert all(not controller.commands for controller in robot.controllers.values())
        assert all(not controller.gripper.commands for controller in robot.controllers.values())
        for controller in robot.controllers.values():
            controller.rtde_c.moveJ.assert_not_called()
            assert controller.kwargs["servo_time"] == 0.017
    finally:
        robot.disconnect()


def test_inactive_gripper_does_not_activate_it(driver, tmp_path):
    original = driver.URController.side_effect

    def inactive_controller(robot_ip, **kwargs):
        controller = original(robot_ip, **kwargs)
        controller.gripper.active = False
        return controller

    driver.URController.side_effect = inactive_controller
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, enable_motion=True, enable_gripper=True))
    with pytest.raises(RuntimeError, match="inactive"):
        robot.connect()
    assert not robot.is_connected
    assert driver.controllers["192.168.1.50"].closed
    assert not driver.controllers["192.168.1.50"].gripper.commands


def test_real_servo_threads_share_watchdog_and_stop(driver, monkeypatch, tmp_path):
    from threading import Thread

    monkeypatch.setattr(module, "Thread", Thread)
    robot = arm_robot(driver, tmp_path, action_timeout_s=0.08)
    try:
        assert all(not controller.commands for controller in robot.controllers.values())
        robot.send_action(action_for(robot))
        assert robot._stop.wait(timeout=2)
        for thread in robot._threads:
            thread.join(timeout=1)
            assert not thread.is_alive()
        assert isinstance(robot._worker_error, TimeoutError)
        for controller in robot.controllers.values():
            assert controller.commands
            controller.rtde_c.servoStop.assert_called_once()
    finally:
        robot.disconnect()


@pytest.mark.parametrize("max_joint_velocity", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_servo_velocity_limit_and_disabled_gripper(
    driver, tmp_path, max_joint_velocity, clip_relative_target
):
    robot = arm_robot(
        driver, tmp_path, max_joint_velocity=max_joint_velocity, clip_relative_target=clip_relative_target
    )
    try:
        actual = robot.controllers["left"].joints.copy()
        robot.send_action(action_for(robot, delta=0.1))
        controller = robot.controllers["left"]
        controller.on_servo = robot._stop.set
        robot._run_servo("left")
        np.testing.assert_allclose(
            controller.commands[0] - actual,
            robot.config.max_joint_velocity * robot.config.servo_time,
        )
        assert not controller.gripper.commands
        controller.rtde_c.servoStop.assert_called_once()
    finally:
        robot.disconnect()


def test_diagnostic_trace_preserves_rejected_target_and_servo_command(driver, tmp_path):
    path = tmp_path / "diagnostics/rollout.jsonl"
    robot = arm_robot(driver, tmp_path, diagnostics_path=path)
    action = action_for(robot, delta=0.1)
    actual = robot.controllers["left"].joints.copy()
    try:
        robot.send_action(action)
        controller = robot.controllers["left"]
        controller.on_servo = robot._stop.set
        robot._run_servo("left")
        robot._stop.clear()
        next_action = action.copy()
        next_action[STATE_NAMES[0]] = float(actual[0] + 0.31)
        with pytest.raises(ValueError, match="max_relative_target"):
            robot.send_action(next_action)
        assert robot._stop.is_set()
        assert robot._target_action_index == 1
        assert len(controller.commands) == 1
    finally:
        robot.disconnect()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["metadata", "accepted", "rejected", "disconnect"]
    assert rows[0]["max_joint_velocity_rad_s"] == 0.6
    assert rows[0]["max_relative_target_rad"] == 0.3
    assert rows[1]["action_index"] == 1
    assert rows[1]["previous_accepted_target"] is None
    rejected = rows[2]
    assert rejected["action_index"] == 2
    assert rejected["target"][0] == pytest.approx(actual[0] + 0.31)
    assert rejected["measured"][0] == pytest.approx(actual[0])
    assert rejected["previous_accepted_target"] == [action[name] for name in STATE_NAMES]
    assert "max_relative_target" in rejected["error"]
    servo = rejected["servo"]["left"]
    assert servo["action_index"] == 1
    np.testing.assert_allclose(servo["command"], actual + 0.6 * 0.017)
    np.testing.assert_allclose(servo["target"], actual + 0.1)
    assert servo["period_s"] is None
    assert servo["servo_call_s"] >= 0
    assert robot._diagnostics_file is None


def test_reported_left_joint_5_target_is_clipped_and_preserved_in_trace(driver, tmp_path):
    path = tmp_path / "clipped.jsonl"
    robot = arm_robot(driver, tmp_path, clip_relative_target=True, diagnostics_path=path)
    try:
        controller = robot.controllers["left"]
        controller.joints[4] = -0.783614460621969
        action = action_for(robot)
        action[STATE_NAMES[4]] = -0.4437938630580902
        sent = robot.send_action(action)
        assert sent[STATE_NAMES[4]] == pytest.approx(-0.483614460621969)
        assert not robot._stop.is_set()
        assert robot._worker_error is None
        controller.on_servo = robot._stop.set
        robot._run_servo("left")
        assert controller.commands[0][4] == pytest.approx(controller.joints[4] + 0.6 * 0.017)
    finally:
        robot.disconnect()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["metadata", "accepted", "disconnect"]
    assert rows[0]["clip_relative_target"]
    accepted = rows[1]
    assert accepted["clipped"]
    assert accepted["requested_target"] == [action[name] for name in STATE_NAMES]
    assert accepted["target"] == [sent[name] for name in STATE_NAMES]
    assert accepted["target"][4] == pytest.approx(-0.483614460621969)
    assert accepted["error"] is None
    assert np.max(np.abs(np.array(accepted["target"][:12]) - accepted["measured"][:12])) <= 0.3 + 1e-12


@pytest.mark.parametrize("side,joint_index", [("left", 4), ("right", 10)])
def test_joint_error_reports_last_sent_command_without_trace_file(driver, tmp_path, side, joint_index):
    robot = arm_robot(driver, tmp_path)
    try:
        assert robot._diagnostics_file is None
        action = action_for(robot, delta=0.1)
        joint_name = STATE_NAMES[joint_index]
        controller = robot.controllers[side]
        measured = controller.joints[joint_index % 6]
        robot.send_action(action)
        controller.on_servo = robot._stop.set
        robot._run_servo(side)
        robot._stop.clear()
        servo = robot._servo_status[side]
        servo["completed_at_s"] -= 0.5
        command = controller.commands[0][joint_index % 6]
        next_action = action.copy()
        next_action[joint_name] = float(measured + 0.31)

        with pytest.raises(ValueError, match="max_relative_target") as raised:
            robot.send_action(next_action)

        message = str(raised.value)
        assert f"previous_target={action[joint_name]:.6f} rad" in message
        assert "target_step_delta=0.210000 rad" in message
        assert f"last_sent_command={command:.6f} rad" in message
        assert f"target_minus_last_sent={next_action[joint_name] - command:.6f} rad" in message
        assert f"last_sent_minus_measured={command - measured:.6f} rad" in message
        assert "last_command_action_index=1" in message
        age_match = re.search(r"last_command_age_ms=(\d+\.\d+)", message)
        assert age_match is not None
        assert float(age_match.group(1)) >= 500
        assert "action_index=2; max_joint_velocity=0.600000 rad/s" in message
        assert robot._target_action_index == 1
        assert robot._stop.is_set()
        assert len(controller.commands) == 1
        assert robot._diagnostics_file is None
    finally:
        robot.disconnect()


def test_existing_diagnostic_trace_is_not_overwritten(driver, tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("existing trace\n")
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, diagnostics_path=path))

    with pytest.raises(FileExistsError):
        robot.connect()

    assert path.read_text() == "existing trace\n"
    driver.URController.assert_not_called()
    driver.rtde_receive.RTDEReceiveInterface.assert_not_called()
    assert not any(camera.is_connected for camera in robot.cameras.values())


def test_servo_diagnostics_measure_period_and_keep_previous_accepted_action(driver, tmp_path):
    path = tmp_path / "servo.jsonl"
    robot = arm_robot(driver, tmp_path, diagnostics_path=path)
    try:
        robot.send_action(action_for(robot, delta=0.1))
        controller = robot.controllers["left"]

        def stop_after_two_commands():
            if len(controller.commands) >= 2:
                robot._stop.set()

        controller.on_servo = stop_after_two_commands
        robot._run_servo("left")
        status = robot._servo_status["left"]
        assert status["period_s"] > 0
        assert status["completed_at_s"] >= status["started_at_s"]
        assert status["action_index"] == 1
        np.testing.assert_allclose(status["command"], controller.commands[-1])
    finally:
        robot.disconnect()


def test_dry_run_diagnostics_never_create_motion_interface(driver, tmp_path):
    path = tmp_path / "dry_run.jsonl"
    robot = DualUR5CB3(
        DualUR5CB3Config(calibration_dir=tmp_path, diagnostics_path=path, dry_run_report_only=True)
    )
    robot.connect()
    try:
        robot.send_action(action_for(robot, delta=0.31))
        assert not robot._stop.is_set()
        assert robot._target is None
    finally:
        robot.disconnect()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["event"] == "dry_run"
    assert "max_relative_target" in rows[1]["error"]
    assert rows[1]["servo"] == {}
    driver.URController.assert_not_called()


@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_diagnostic_write_failure_stops_before_accepting_action(driver, tmp_path, clip_relative_target):
    robot = arm_robot(driver, tmp_path, clip_relative_target=clip_relative_target)
    writer = MagicMock()
    writer.write.side_effect = OSError("disk failure")
    robot._diagnostics_file = writer
    robot._diagnostics_started = time.monotonic()
    try:
        with pytest.raises(OSError, match="disk failure"):
            robot.send_action(action_for(robot, delta=0.34 if clip_relative_target else 0))
        assert robot._stop.is_set()
        assert robot._target is None
        assert not robot._ready.is_set()
        assert all(not controller.commands for controller in robot.controllers.values())
    finally:
        robot.disconnect()
    writer.close.assert_called_once()


def test_nonfinite_rejected_action_is_recorded_as_error_without_nan(driver, tmp_path):
    path = tmp_path / "invalid.jsonl"
    robot = arm_robot(driver, tmp_path, diagnostics_path=path)
    try:
        action = action_for(robot)
        action[STATE_NAMES[0]] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            robot.send_action(action)
    finally:
        robot.disconnect()

    rejected = json.loads(path.read_text().splitlines()[1])
    assert rejected["event"] == "rejected"
    assert rejected["target"] is None
    assert "finite" in rejected["error"]
    assert "NaN" not in path.read_text()


@pytest.mark.parametrize("side,expected", [("left", 255), ("right", 0)])
@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_gripper_mapping_and_clipping(driver, tmp_path, side, expected, clip_relative_target):
    robot = arm_robot(driver, tmp_path, enable_gripper=True, clip_relative_target=clip_relative_target)
    try:
        action = action_for(robot)
        action["left_gripper.pos"] = 2
        action["right_gripper.pos"] = -1
        robot.send_action(action)
        controller = robot.controllers[side]
        controller.on_servo = lambda: setattr(controller, "on_servo", robot._stop.set)
        robot._run_servo(side)
        assert controller.gripper.commands == [(expected, 255, 128)]
        assert robot._target[13] == 0
    finally:
        robot.disconnect()


@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_action_watchdog_stops_without_sending_stale_target(driver, tmp_path, clip_relative_target):
    robot = arm_robot(driver, tmp_path, clip_relative_target=clip_relative_target)
    try:
        robot.send_action(action_for(robot))
        robot._last_action_time = time.monotonic() - 2
        robot._run_servo("right")
        assert isinstance(robot._worker_error, TimeoutError)
        assert robot._stop.is_set()
        assert not robot.controllers["right"].commands
        with pytest.raises(RuntimeError, match="servo stopped"):
            robot.send_action(action_for(robot))
    finally:
        robot.disconnect()


@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_tracking_error_stops_servo(driver, tmp_path, clip_relative_target):
    robot = arm_robot(driver, tmp_path, clip_relative_target=clip_relative_target)
    try:
        robot.send_action(action_for(robot))
        controller = robot.controllers["left"]

        def lose_tracking():
            controller.joints += 1

        controller.on_servo = lose_tracking
        robot._run_servo("left")
        assert robot._stop.is_set()
        assert isinstance(robot._worker_error, RuntimeError)
        controller.rtde_c.servoStop.assert_called_once()
    finally:
        robot.disconnect()


@pytest.mark.parametrize("clip_relative_target", [False, True])
def test_single_arm_safety_fault_stops_both(driver, tmp_path, clip_relative_target):
    robot = arm_robot(driver, tmp_path, clip_relative_target=clip_relative_target)
    try:
        robot.send_action(action_for(robot))
        robot.controllers["right"].rtde_r.getSafetyMode.return_value = 3
        robot._run_servo("right")
        assert robot._stop.is_set()
        assert not robot.controllers["right"].commands
    finally:
        robot.disconnect()


def test_reconnect_clears_fault(robot):
    robot.connect()
    robot._fail(RuntimeError("old fault"))
    robot.disconnect()
    robot.connect()
    assert robot._worker_error is None
    assert not robot._stop.is_set()
    assert not robot._ready.is_set()


def test_partial_connect_failure_releases_existing_devices(driver, monkeypatch, tmp_path):
    def connect_first_only(robot_ip, **kwargs):
        if robot_ip.endswith("60"):
            raise ConnectionError("right unavailable")
        controller = FakeController(robot_ip, **kwargs)
        driver.controllers[robot_ip] = controller
        return controller

    driver.URController.side_effect = connect_first_only
    robot = DualUR5CB3(DualUR5CB3Config(calibration_dir=tmp_path, enable_motion=True))
    with pytest.raises(ConnectionError, match="right unavailable"):
        robot.connect()
    assert driver.controllers["192.168.1.50"].closed
    assert not robot.is_connected
    assert all(not camera.is_connected for camera in robot.cameras.values())


def test_source_loader_missing_and_wrong_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "require_package", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="collection driver not found"):
        module._load_driver(tmp_path)
    expected = tmp_path / "xrobotoolkit_teleop/hardware/interface/universal_robots.py"
    expected.parent.mkdir(parents=True)
    expected.touch()
    fake_module = ModuleType("driver")
    fake_module.__file__ = str(tmp_path / "wrong_driver.py")
    monkeypatch.setattr(module, "import_module", lambda name: fake_module)
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="different collection driver"):
        module._load_driver(tmp_path)
    assert sys.path == before


def test_example_yaml_parses_without_connecting_hardware(monkeypatch, tmp_path):
    import draccus

    from lerobot.configs import parser
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.rollout.configs import RolloutConfig

    checkpoint = tmp_path / "checkpoint"
    SmolVLAConfig(device="cpu").save_pretrained(checkpoint)
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", f"--policy.path={checkpoint}"])
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    example = Path("examples/ur5_real/rollout.yaml")
    cleaned = parser.extract_path_fields_from_config(str(example), ["policy"])
    config = draccus.parse(RolloutConfig, config_path=cleaned, args=[])
    assert config.robot.type == "dual_ur5_cb3"
    assert not config.robot.enable_motion
    assert not config.robot.enable_gripper
    assert config.robot.max_joint_velocity == 1.0
    assert config.robot.max_relative_target == 0.5
    assert config.robot.clip_relative_target
    assert config.robot.max_tracking_error == 0.3
    assert config.robot.servo_time == 0.017
    assert config.policy.n_action_steps == 50
    assert config.inference.type == "rtc"
    assert config.inference.rtc.enabled
    assert config.inference.rtc.mode == "guided"
    assert config.inference.rtc.execution_horizon == 30
    assert config.inference.queue_threshold == 30
    assert config.interactive
    assert not config.return_to_initial_position
    assert config.fps == 20


@pytest.mark.parametrize("inference_delay", [12, 13, 18])
def test_example_rtc_retains_guidance_after_measured_inference_delay(inference_delay):
    import yaml

    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    config = yaml.safe_load(Path("examples/ur5_real/rollout.yaml").read_text())
    inference = config["inference"]
    rtc = RTCConfig(**inference["rtc"])
    available_steps = inference["queue_threshold"]
    weights = RTCProcessor(rtc).get_prefix_weights(
        inference_delay, min(rtc.execution_horizon, available_steps), 50
    )

    assert inference_delay < available_steps
    assert (weights[inference_delay : inference_delay + 5] > 0).all()
    assert config["robot"]["max_relative_target"] == 0.5
    assert config["robot"]["clip_relative_target"]
    assert config["robot"]["max_joint_velocity"] == 1.0


@pytest.mark.parametrize(
    "side,measured,requested,expected",
    [
        ("left", -0.783614461, -0.443793863, -0.443793863),
        ("right", -0.914020, -1.214902, -1.214902),
        ("left", -1.232441, -0.919662, -0.919662),
        ("left", -1.232441, -0.6, -0.732441),
    ],
)
def test_relaxed_example_preserves_normal_targets_and_limits_large_offsets(
    driver, tmp_path, monkeypatch, side, measured, requested, expected
):
    import yaml

    example = yaml.safe_load(Path("examples/ur5_real/rollout.yaml").read_text())["robot"]
    limits = {
        name: example[name]
        for name in (
            "servo_time",
            "max_joint_velocity",
            "max_relative_target",
            "clip_relative_target",
            "max_tracking_error",
        )
    }
    robot = arm_robot(driver, tmp_path, **limits)
    safe_goals = MagicMock(wraps=module.ensure_safe_goal_position)
    monkeypatch.setattr(module, "ensure_safe_goal_position", safe_goals)
    try:
        controller = robot.controllers[side]
        controller.joints[4] = measured
        joint_name = f"{side}_joint_5.pos"
        action = action_for(robot)
        action[joint_name] = requested
        sent = robot.send_action(action)
        assert sent[joint_name] == pytest.approx(expected)
        if requested == expected:
            safe_goals.assert_not_called()
        else:
            safe_goals.assert_called_once()
            assert safe_goals.call_args.args[1] == 0.5
        controller.on_servo = robot._stop.set
        robot._run_servo(side)
        assert controller.commands[0][4] == pytest.approx(
            measured + math.copysign(0.017, expected - measured)
        )
        assert robot._worker_error is None
        assert robot.config.max_tracking_error == 0.3
    finally:
        robot.disconnect()


def test_official_sync_engine_preserves_14d_state_and_action_order(robot):
    import torch

    from lerobot.inference.sync import SyncInferenceEngine
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor import make_default_policy_processor_steps, make_policy_processor_pipelines
    from lerobot.utils.feature_utils import build_dataset_frame, dataset_to_policy_features

    features = {
        **hw_to_dataset_features(robot.observation_features, "observation"),
        **hw_to_dataset_features(robot.action_features, "action"),
    }
    policy_features = dataset_to_policy_features(features)
    config = SmolVLAConfig(
        device="cpu",
        input_features={name: feature for name, feature in policy_features.items() if name != "action"},
        output_features={"action": policy_features["action"]},
    )
    means = torch.arange(14, dtype=torch.float32) / 10
    steps = make_default_policy_processor_steps(
        config,
        {
            "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
            "action": {"mean": means, "std": torch.ones(14)},
        },
    )
    preprocessor, postprocessor = make_policy_processor_pipelines(
        input_steps=[steps.normalize], output_steps=[steps.unnormalize]
    )

    class FakePolicy:
        def __init__(self):
            self.config = config

        def select_action(self, batch):
            assert batch["observation.state"].shape == (1, 14)
            for name in ("exterior", "left_wrist", "right_wrist"):
                assert batch[f"observation.images.{name}"].shape == (1, 3, 480, 640)
            return torch.zeros(1, 14)

    engine = SyncInferenceEngine(
        FakePolicy(), preprocessor, postprocessor, features, list(STATE_NAMES), "test task", "cpu", robot.name
    )
    robot.connect()
    frame = build_dataset_frame(features, robot.get_observation(), "observation")
    predicted = engine.get_action(frame)
    torch.testing.assert_close(predicted, means)
