"""Preview or explicitly restore the dual-UR5 home pose from the collection driver."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.robots.dual_ur5_cb3.dual_ur5_cb3 import STATE_NAMES, _load_driver, _ReadOnlyController

logger = logging.getLogger(__name__)


def load_collection_joints(driver) -> np.ndarray:
    angles = np.asarray([driver.LEFT_INITIAL_JOINT_DEG, driver.RIGHT_INITIAL_JOINT_DEG], dtype=np.float64)
    if angles.shape != (2, 6) or not np.isfinite(angles).all() or np.any(np.abs(angles) > 360):
        raise ValueError("Collection home pose must contain two sets of six finite joint angles in degrees.")
    return np.deg2rad(angles.reshape(12))


def load_initial_state(root: Path, repo_id: str, episode_index: int) -> np.ndarray:
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    names = dataset.features["observation.state"]["names"]
    if names is None or [name.removesuffix(".pos") for name in names] != [
        name.removesuffix(".pos") for name in STATE_NAMES
    ]:
        raise ValueError("Dataset state names/order do not match the dual-UR5 convention.")
    episode = next((item for item in dataset.meta.episodes if item["episode_index"] == episode_index), None)
    if episode is None:
        raise ValueError(f"Episode {episode_index} does not exist.")
    frame = dataset.get_raw_item(int(episode["dataset_from_index"]))
    if int(frame["episode_index"]) != episode_index or int(frame["frame_index"]) != 0:
        raise ValueError("Episode metadata does not point to its first frame.")
    state = np.asarray(frame["observation.state"], dtype=np.float64)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Initial state must contain 14 finite values.")
    if np.any(np.abs(state[:12]) > 2 * np.pi) or np.any((state[12:] < 0) | (state[12:] > 1)):
        raise ValueError("Initial state exceeds the expected joint/gripper ranges.")
    return state


def check_stationary(side: str, controller) -> np.ndarray:
    receiver = controller.rtde_r
    if not receiver.isConnected() or receiver.getSafetyMode() not in (1, 2) or receiver.getRobotMode() != 7:
        raise RuntimeError(f"{side} UR5 must be running and in normal/reduced safety mode.")
    joints = np.asarray(controller.get_current_joint_positions(), dtype=np.float64)
    speeds = np.asarray(receiver.getActualQd(), dtype=np.float64)
    if (
        joints.shape != (6,)
        or speeds.shape != (6,)
        or not np.isfinite(joints).all()
        or not np.isfinite(speeds).all()
    ):
        raise RuntimeError(f"{side} UR5 returned invalid joint telemetry.")
    if np.any(np.abs(speeds) > 0.01):
        raise RuntimeError(f"{side} UR5 is moving; stop all other controllers before homing.")
    return joints


def confirm(command: str, message: str) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Execution requires an interactive terminal for confirmation.")
    if input(f"{message}\nType {command} to proceed: ").strip() != command:
        raise RuntimeError("Cancelled; no further movement will be commanded.")


def close_motion_controller(controller) -> None:
    try:
        controller.close()
    finally:
        for interface in (controller.rtde_r, controller.rtde_c):
            try:
                interface.disconnect()
            except Exception:
                logger.exception("Could not disconnect a homing RTDE interface")


def restore_gripper(side: str, controller, fraction: float) -> None:
    gripper = controller.gripper
    if not gripper.is_active():
        raise RuntimeError(f"{side} gripper is inactive; this tool never activates/calibrates it.")
    opened, closed = gripper.get_open_position(), gripper.get_closed_position()
    if closed <= opened:
        raise RuntimeError(f"{side} gripper has invalid endpoints.")
    position = round(opened + fraction * (closed - opened))
    confirm(
        f"GRIPPER {side.upper()}", f"Check {side} gripper is empty and its opening/closing path is clear."
    )
    accepted, _ = gripper.move(position, 64, 32)
    if not accepted:
        raise RuntimeError(f"{side} gripper rejected its target.")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if abs(gripper.get_current_position() - position) <= 3:
            print(f"{side} gripper restored to the recorded starting position.")
            return
        time.sleep(0.05)
    raise TimeoutError(f"{side} gripper did not reach its starting position; inspect it manually.")


def execute_home(state: np.ndarray, driver, robot_ips: dict[str, str], first_arm: str, grippers: bool):
    readers = {}
    try:
        for side, robot_ip in robot_ips.items():
            readers[side] = _ReadOnlyController(driver, robot_ip)
            joints = check_stationary(side, readers[side])
            offset = 0 if side == "left" else 6
            print(f"{side} current degrees: {np.rad2deg(joints).round(5).tolist()}")
            print(
                f"{side} target minus current degrees: {np.rad2deg(state[offset : offset + 6] - joints).round(5).tolist()}"
            )
            if grippers and not readers[side].gripper.is_active():
                raise RuntimeError(f"{side} gripper is inactive; activate it separately only when safe.")
        order = (first_arm, "right" if first_arm == "left" else "left")
        for side in order:
            before = {name: check_stationary(name, reader) for name, reader in readers.items()}
            confirm(
                f"MOVE {side.upper()}",
                f"{side} arm will moveJ to the recorded pose at {driver.HOME_VELOCITY} rad/s, "
                f"{driver.HOME_ACCELERATION} rad/s^2. Verify its entire path, including the other arm. "
                "No collision checking is performed. Keep the pendant stop accessible.",
            )
            for name, reader in readers.items():
                if np.max(np.abs(check_stationary(name, reader) - before[name])) > 0.005:
                    raise RuntimeError(f"{name} arm moved during confirmation; rerun and inspect the path.")
            offset = 0 if side == "left" else 6
            target = state[offset : offset + 6].copy()
            controller = driver.URController(robot_ip=robot_ips[side], initial_joint_positions=target)
            try:
                check_stationary(side, controller)
                for name, reader in readers.items():
                    if np.max(np.abs(check_stationary(name, reader) - before[name])) > 0.005:
                        raise RuntimeError(f"{name} arm moved while connecting; homing cancelled.")
                controller.return_home()
                actual = check_stationary(side, controller)
                error = float(np.max(np.abs(actual - target)))
                if error > 0.005:
                    raise RuntimeError(f"{side} homing did not reach its target: max error {error:.6f} rad.")
                print(f"{side} arm restored; max measured joint error {error:.6f} rad.")
            finally:
                close_motion_controller(controller)
        if grippers:
            for side in order:
                restore_gripper(side, readers[side], float(state[12 if side == "left" else 13]))
        print("Homing finished. Reset the task objects before starting rollout.")
    finally:
        for reader in readers.values():
            try:
                reader.close()
            except Exception:
                logger.exception("Could not close a homing read-only connection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-source", choices=("collection", "dataset"), default="collection")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/carto317/XRoboToolkit-Teleop-Sample-Python/datasets/dual_ur5_demo_coloured"),
    )
    parser.add_argument("--repo-id", default="local/dual-ur5-demo-coloured")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--teleop-root", type=Path, default=Path("/home/carto317/XRoboToolkit-Teleop-Sample-Python")
    )
    parser.add_argument("--left-robot-ip", default="192.168.1.50")
    parser.add_argument("--right-robot-ip", default="192.168.1.60")
    parser.add_argument("--first-arm", choices=("left", "right"), default="left")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--restore-grippers", action="store_true")
    args = parser.parse_args()
    if not args.left_robot_ip or not args.right_robot_ip or args.left_robot_ip == args.right_robot_ip:
        parser.error("Specify distinct, nonempty robot IP addresses.")
    if args.restore_grippers and not args.execute:
        parser.error("--restore-grippers requires --execute")
    if args.restore_grippers and args.pose_source != "dataset":
        parser.error(
            "Collection home angles contain no gripper targets; --restore-grippers needs --pose-source=dataset."
        )
    if args.execute and not sys.stdin.isatty():
        parser.error("Execution requires an interactive terminal; no hardware connections were created.")
    driver = _load_driver(args.teleop_root)
    if args.pose_source == "collection":
        state = load_collection_joints(driver)
        print("Target: LEFT_INITIAL_JOINT_DEG / RIGHT_INITIAL_JOINT_DEG from the collection driver.")
    else:
        state = load_initial_state(args.dataset_root, args.repo_id, args.episode)
        print(f"Target: episode {args.episode}, first measured observation.state (not predicted action).")
    print(f"left joint radians: {state[:6].tolist()}")
    print(f"right joint radians: {state[6:12].tolist()}")
    print(f"left joint degrees: {np.rad2deg(state[:6]).round(5).tolist()}")
    print(f"right joint degrees: {np.rad2deg(state[6:12]).round(5).tolist()}")
    if state.size == 14:
        print(f"gripper closure fractions: {state[12:].tolist()}")
    if not args.execute:
        print("Preview only: no robot, gripper or camera connection was created. Add --execute to move.")
        return
    print("Stop rollout/VR/other RTDE controllers first. Ctrl+C is not a hardware emergency stop.")
    execute_home(
        state,
        driver,
        {"left": args.left_robot_ip, "right": args.right_robot_ip},
        args.first_arm,
        args.restore_grippers,
    )


if __name__ == "__main__":
    main()
