import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.ur5_real import home


@pytest.fixture
def initial_state():
    return np.concatenate((np.linspace(-1, 1, 12), [3 / 255, 3 / 255]))


@pytest.fixture
def dataset(monkeypatch, initial_state):
    instance = SimpleNamespace(
        features={"observation.state": {"names": [name.removesuffix(".pos") for name in home.STATE_NAMES]}},
        meta=SimpleNamespace(episodes=[{"episode_index": 4, "dataset_from_index": 100}]),
        get_raw_item=MagicMock(
            return_value={
                "episode_index": 4,
                "frame_index": 0,
                "observation.state": initial_state,
                "action": np.full(14, 99),
            }
        ),
    )
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(home, "LeRobotDataset", factory)
    return instance, factory


def test_target_is_first_measured_state_without_loading_videos(dataset, initial_state):
    instance, factory = dataset
    state = home.load_initial_state(Path("demo"), "local/demo", 4)

    np.testing.assert_array_equal(state, initial_state)
    factory.assert_called_once_with("local/demo", root=Path("demo"), download_videos=False)
    instance.get_raw_item.assert_called_once_with(100)


@pytest.mark.parametrize("invalid", ["names", "episode", "frame", "shape", "nan", "joint", "gripper"])
def test_invalid_target_rejected(dataset, invalid):
    instance, _ = dataset
    frame = instance.get_raw_item.return_value
    if invalid == "names":
        instance.features["observation.state"]["names"].reverse()
    elif invalid == "episode":
        frame["episode_index"] = 0
    elif invalid == "frame":
        frame["frame_index"] = 1
    elif invalid == "shape":
        frame["observation.state"] = np.zeros(12)
    elif invalid == "nan":
        frame["observation.state"][0] = np.nan
    elif invalid == "joint":
        frame["observation.state"][0] = 2 * np.pi + 0.01
    else:
        frame["observation.state"][12] = 1.01
    with pytest.raises(ValueError):
        home.load_initial_state(Path("demo"), "local/demo", 4)


def test_dataset_preview_never_connects(monkeypatch, initial_state):
    monkeypatch.setattr(sys, "argv", ["home.py", "--pose-source=dataset"])
    monkeypatch.setattr(home, "load_initial_state", lambda *args: initial_state)
    load_driver = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(home, "_load_driver", load_driver)
    monkeypatch.setattr(home, "execute_home", execute)

    home.main()

    load_driver.assert_called_once()
    execute.assert_not_called()


def test_noninteractive_execution_rejected_before_loading_driver(monkeypatch, initial_state):
    monkeypatch.setattr(sys, "argv", ["home.py", "--execute"])
    monkeypatch.setattr(home, "load_initial_state", lambda *args: initial_state)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    load_driver = MagicMock()
    monkeypatch.setattr(home, "_load_driver", load_driver)

    with pytest.raises(SystemExit):
        home.main()

    load_driver.assert_not_called()


def test_collection_joints_use_existing_constants_without_rounding():
    driver = SimpleNamespace(
        LEFT_INITIAL_JOINT_DEG=np.array([39.79, -65.51, -106.48, -126.25, -20.00, 171.02]),
        RIGHT_INITIAL_JOINT_DEG=np.array([-28.806, -86.372, 103.030, -172.883, -19.005, 17.054]),
    )

    joints = home.load_collection_joints(driver)

    np.testing.assert_array_equal(
        joints, np.deg2rad(np.concatenate((driver.LEFT_INITIAL_JOINT_DEG, driver.RIGHT_INITIAL_JOINT_DEG)))
    )


@pytest.mark.parametrize("angles", [[0] * 5, [0] * 5 + [np.nan], [0] * 5 + [361]])
def test_invalid_collection_home_angles_rejected(angles):
    driver = SimpleNamespace(LEFT_INITIAL_JOINT_DEG=angles, RIGHT_INITIAL_JOINT_DEG=angles)

    with pytest.raises(ValueError, match="six finite"):
        home.load_collection_joints(driver)


def test_default_preview_uses_collection_constants_without_hardware(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["home.py"])
    driver = SimpleNamespace(LEFT_INITIAL_JOINT_DEG=np.arange(6), RIGHT_INITIAL_JOINT_DEG=-np.arange(6))
    monkeypatch.setattr(home, "_load_driver", lambda *args: driver)
    dataset_loader = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(home, "load_initial_state", dataset_loader)
    monkeypatch.setattr(home, "execute_home", execute)

    home.main()

    dataset_loader.assert_not_called()
    execute.assert_not_called()


def test_gripper_restoration_requires_recorded_targets(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["home.py", "--execute", "--restore-grippers"])
    load_driver = MagicMock()
    monkeypatch.setattr(home, "_load_driver", load_driver)

    with pytest.raises(SystemExit):
        home.main()

    load_driver.assert_not_called()


@pytest.fixture
def hardware(monkeypatch):
    events, readers = [], {}

    def make_reader(driver, robot_ip):
        reader = MagicMock()
        reader.joints = np.zeros(6)
        reader.get_current_joint_positions.side_effect = lambda: reader.joints.copy()
        reader.rtde_r.isConnected.return_value = True
        reader.rtde_r.getSafetyMode.return_value = 1
        reader.rtde_r.getRobotMode.return_value = 7
        reader.rtde_r.getActualQd.return_value = np.zeros(6)
        reader.gripper.is_active.return_value = True
        readers[robot_ip] = reader
        return reader

    def make_motion(robot_ip, initial_joint_positions):
        reader = readers[robot_ip]
        controller = MagicMock()
        controller.rtde_r = reader.rtde_r
        controller.get_current_joint_positions = reader.get_current_joint_positions

        def return_home():
            events.append(("home", robot_ip))
            reader.joints = initial_joint_positions.copy()

        controller.return_home.side_effect = return_home
        controller.close.side_effect = lambda: events.append(("close", robot_ip))
        return controller

    driver = SimpleNamespace(
        HOME_VELOCITY=0.2, HOME_ACCELERATION=0.3, URController=MagicMock(side_effect=make_motion)
    )
    monkeypatch.setattr(home, "_ReadOnlyController", make_reader)
    monkeypatch.setattr(home, "confirm", lambda *args: None)
    return driver, readers, events


@pytest.mark.parametrize("first_arm", ["left", "right"])
@pytest.mark.parametrize("pose_source", ["collection", "dataset"])
def test_home_is_sequential_and_uses_exact_joints(hardware, initial_state, first_arm, pose_source):
    driver, readers, events = hardware
    robot_ips = {"left": "left_ip", "right": "right_ip"}

    state = initial_state[:12] if pose_source == "collection" else initial_state
    home.execute_home(state, driver, robot_ips, first_arm, grippers=False)

    order = [robot_ips[first_arm], robot_ips["right" if first_arm == "left" else "left"]]
    assert events == [(operation, robot_ip) for robot_ip in order for operation in ("home", "close")]
    for side, offset in (("left", 0), ("right", 6)):
        reader = readers[robot_ips[side]]
        np.testing.assert_array_equal(reader.joints, initial_state[offset : offset + 6])
        reader.close.assert_called_once()
        reader.gripper.move.assert_not_called()
        reader.gripper.activate.assert_not_called()


def test_cancel_before_first_motion_closes_readers(hardware, initial_state, monkeypatch):
    driver, readers, events = hardware
    monkeypatch.setattr(home, "confirm", MagicMock(side_effect=RuntimeError("Cancelled")))

    with pytest.raises(RuntimeError, match="Cancelled"):
        home.execute_home(initial_state, driver, {"left": "left_ip", "right": "right_ip"}, "left", False)

    driver.URController.assert_not_called()
    assert not events
    for reader in readers.values():
        reader.close.assert_called_once()


def test_failed_first_home_prevents_second_arm_motion(hardware, initial_state):
    driver, readers, events = hardware
    original_factory = driver.URController.side_effect

    def failed_motion(**kwargs):
        controller = original_factory(**kwargs)
        controller.return_home.side_effect = RuntimeError("Failed moveJ")
        return controller

    driver.URController.side_effect = failed_motion
    with pytest.raises(RuntimeError, match="Failed moveJ"):
        home.execute_home(initial_state, driver, {"left": "left_ip", "right": "right_ip"}, "left", False)

    assert driver.URController.call_count == 1
    assert events == [("close", "left_ip")]
    for reader in readers.values():
        reader.close.assert_called_once()


def test_missed_target_prevents_second_arm_motion(hardware, initial_state):
    driver, _, events = hardware
    original_factory = driver.URController.side_effect

    def missed_target(**kwargs):
        controller = original_factory(**kwargs)
        controller.return_home.side_effect = lambda: None
        return controller

    driver.URController.side_effect = missed_target
    with pytest.raises(RuntimeError, match="did not reach"):
        home.execute_home(initial_state, driver, {"left": "left_ip", "right": "right_ip"}, "left", False)

    assert driver.URController.call_count == 1
    assert events == [("close", "left_ip")]


def test_inactive_gripper_prevents_homing_when_restoration_requested(hardware, initial_state, monkeypatch):
    driver, readers, events = hardware
    original_reader = home._ReadOnlyController

    def inactive_reader(*args):
        reader = original_reader(*args)
        reader.gripper.is_active.return_value = False
        return reader

    monkeypatch.setattr(home, "_ReadOnlyController", inactive_reader)
    with pytest.raises(RuntimeError, match="inactive"):
        home.execute_home(initial_state, driver, {"left": "left_ip", "right": "right_ip"}, "left", True)

    driver.URController.assert_not_called()
    assert not events
    for reader in readers.values():
        reader.close.assert_called_once()


def test_motion_during_confirmation_cancels_before_control_connection(hardware, initial_state, monkeypatch):
    driver, readers, events = hardware

    def changed_pose(*args):
        readers["right_ip"].joints[0] += 0.1

    monkeypatch.setattr(home, "confirm", changed_pose)
    with pytest.raises(RuntimeError, match="during confirmation"):
        home.execute_home(initial_state, driver, {"left": "left_ip", "right": "right_ip"}, "left", False)

    driver.URController.assert_not_called()
    assert not events


@pytest.mark.parametrize("fault", ["moving", "safety", "mode", "nan"])
def test_bad_telemetry_prevents_motion(fault):
    controller = MagicMock()
    controller.rtde_r.isConnected.return_value = True
    controller.rtde_r.getSafetyMode.return_value = 1
    controller.rtde_r.getRobotMode.return_value = 7
    controller.rtde_r.getActualQd.return_value = np.zeros(6)
    controller.get_current_joint_positions.return_value = np.zeros(6)
    if fault == "moving":
        controller.rtde_r.getActualQd.return_value = np.full(6, 0.02)
    elif fault == "safety":
        controller.rtde_r.getSafetyMode.return_value = 3
    elif fault == "mode":
        controller.rtde_r.getRobotMode.return_value = 5
    else:
        controller.get_current_joint_positions.return_value = np.full(6, np.nan)

    with pytest.raises(RuntimeError):
        home.check_stationary("left", controller)


def test_restore_gripper_requires_activation_before_command(monkeypatch):
    controller = MagicMock()
    controller.gripper.is_active.return_value = False
    confirmation = MagicMock()
    monkeypatch.setattr(home, "confirm", confirmation)

    with pytest.raises(RuntimeError, match="inactive"):
        home.restore_gripper("left", controller, 0.01)

    confirmation.assert_not_called()
    controller.gripper.move.assert_not_called()
    controller.gripper.activate.assert_not_called()


def test_restore_gripper_uses_recorded_fraction_without_calibration(monkeypatch):
    controller = MagicMock()
    gripper = controller.gripper
    gripper.is_active.return_value = True
    gripper.get_open_position.return_value = 0
    gripper.get_closed_position.return_value = 255
    gripper.move.return_value = (True, 3)
    gripper.get_current_position.return_value = 3
    confirmation = MagicMock()
    monkeypatch.setattr(home, "confirm", confirmation)

    home.restore_gripper("right", controller, 3 / 255)

    confirmation.assert_called_once()
    gripper.move.assert_called_once_with(3, 64, 32)
    gripper.activate.assert_not_called()
