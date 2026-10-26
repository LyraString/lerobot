from __future__ import annotations

import json
import logging
import sys
import time
from importlib import import_module
from pathlib import Path
from threading import Event, Lock, Thread
from types import ModuleType
from typing import Any, TextIO

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from .config_dual_ur5_cb3 import DualUR5CB3Config

logger = logging.getLogger(__name__)

STATE_NAMES = (
    *(f"left_joint_{index}.pos" for index in range(1, 7)),
    *(f"right_joint_{index}.pos" for index in range(1, 7)),
    "left_gripper.pos",
    "right_gripper.pos",
)


def _load_driver(root: Path) -> ModuleType:
    require_package("ur-rtde", extra="ur5", import_name="rtde_control")
    root = root.resolve()
    expected = root / "xrobotoolkit_teleop/hardware/interface/universal_robots.py"
    if not expected.is_file():
        raise FileNotFoundError(f"CB3 collection driver not found: {expected}. Set --robot.teleop_root.")
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        module = import_module("xrobotoolkit_teleop.hardware.interface.universal_robots")
    finally:
        if inserted:
            sys.path.remove(str(root))
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"A different collection driver is already imported: {module.__file__}")
    return module


class _ReadOnlyController:
    def __init__(self, driver: ModuleType, robot_ip: str) -> None:
        self.rtde_r = driver.rtde_receive.RTDEReceiveInterface(robot_ip)
        self.gripper = driver.RobotiqGripper()
        try:
            self.gripper.connect(robot_ip, 63352)
        except Exception:
            try:
                self.gripper.disconnect()
            finally:
                self.rtde_r.disconnect()
            raise

    def get_current_joint_positions(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)

    def close(self) -> None:
        try:
            self.gripper.disconnect()
        finally:
            self.rtde_r.disconnect()


class DualUR5CB3(Robot):
    """Bridge the workstation's existing CB3/Robotiq driver to LeRobot rollout."""

    config_class = DualUR5CB3Config
    name = "dual_ur5_cb3"
    supports_position_hold = True

    def __init__(self, config: DualUR5CB3Config) -> None:
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.controllers: dict[str, Any] = {}
        self._is_connected = False
        self._lock = Lock()
        self._stop = Event()
        self._ready = Event()
        self._threads: list[Thread] = []
        self._target: np.ndarray | None = None
        self._last_action_time = 0.0
        self._last_dry_run_log = 0.0
        self._last_dry_run_warning = 0.0
        self._dry_run_action_count = 0
        self._worker_error: Exception | None = None
        self._diagnostics_file: TextIO | None = None
        self._diagnostics_started = 0.0
        self._action_count = 0
        self._target_action_index = 0
        self._servo_status: dict[str, dict[str, Any]] = {}

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **dict.fromkeys(STATE_NAMES, float),
            **{name: (config.height, config.width, 3) for name, config in self.config.cameras.items()},
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(STATE_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Joint radians and the collection driver's gripper endpoints need no VR calibration."""

    def configure(self) -> None:
        """Do not activate grippers, set TCP/payload, or move to an initial pose."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        driver = _load_driver(self.config.teleop_root)
        self._stop.clear()
        self._ready.clear()
        self._worker_error = None
        self._target = None
        self._threads = []
        self._last_dry_run_log = 0.0
        self._last_dry_run_warning = 0.0
        self._dry_run_action_count = 0
        self._action_count = 0
        self._target_action_index = 0
        self._servo_status = {}
        try:
            self._open_diagnostics()
            for camera in self.cameras.values():
                camera.connect()
            for side, robot_ip in (
                ("left", self.config.left_robot_ip),
                ("right", self.config.right_robot_ip),
            ):
                if self.config.enable_motion:
                    controller = driver.URController(
                        robot_ip=robot_ip,
                        initial_joint_positions=np.zeros(6),
                        servo_time=self.config.servo_time,
                        lookahead_time=self.config.lookahead_time,
                        servo_gain=self.config.servo_gain,
                        gripper_speed=self.config.gripper_speed,
                        gripper_force=self.config.gripper_force,
                    )
                else:
                    controller = _ReadOnlyController(driver, robot_ip)
                self.controllers[side] = controller
                joints = self._read_joints(side)
                if self.config.enable_motion:
                    controller.initial_joint_positions = joints.copy()
                    self._check_safety(side)
                    if self.config.enable_gripper and not controller.gripper.is_active():
                        raise RuntimeError(
                            f"{side} Robotiq is inactive; activate it manually before rollout."
                        )
            self._is_connected = True
            self._read_state()
            if self.config.enable_motion:
                for side in self.controllers:
                    thread = Thread(target=self._run_servo, args=(side,), name=f"ur5-{side}", daemon=True)
                    self._threads.append(thread)
                    thread.start()
            logger.warning(
                "Dual UR5 CB3 connected: motion=%s, gripper_commands=%s, relative_target_mode=%s. "
                "No homing or activation performed.",
                self.config.enable_motion,
                self.config.enable_motion and self.config.enable_gripper,
                "clip" if self.config.enable_motion and self.config.clip_relative_target else "reject",
            )
        except Exception:
            self.disconnect()
            raise

    def _open_diagnostics(self) -> None:
        path = self.config.diagnostics_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._diagnostics_started = time.monotonic()
        self._diagnostics_file = path.open("x", encoding="utf-8", buffering=1)
        metadata = {
            "event": "metadata",
            "timestamp": time.time(),
            "state_names": STATE_NAMES,
            "enable_motion": self.config.enable_motion,
            "enable_gripper": self.config.enable_gripper,
            "servo_time_s": self.config.servo_time,
            "max_joint_velocity_rad_s": self.config.max_joint_velocity,
            "max_relative_target_rad": self.config.max_relative_target,
            "clip_relative_target": self.config.clip_relative_target,
            "max_tracking_error_rad": self.config.max_tracking_error,
        }
        self._diagnostics_file.write(json.dumps(metadata, allow_nan=False) + "\n")
        logger.info("UR5 diagnostic trace: %s", path.resolve())

    def _record_diagnostics(
        self,
        status: str,
        target: np.ndarray | None,
        state: np.ndarray | None,
        error: str | None = None,
        *,
        requested_target: np.ndarray | None = None,
    ) -> None:
        if self._diagnostics_file is None:
            return
        with self._lock:
            servo = dict(self._servo_status)
            previous_target = None if self._target is None else self._target.tolist()
        record = {
            "event": status,
            "elapsed_s": time.monotonic() - self._diagnostics_started,
            "action_index": self._action_count,
            "target": target.tolist() if target is not None and np.isfinite(target).all() else None,
            "requested_target": (
                requested_target.tolist()
                if requested_target is not None and np.isfinite(requested_target).all()
                else None
            ),
            "clipped": requested_target is not None,
            "measured": state.tolist() if state is not None and np.isfinite(state).all() else None,
            "previous_accepted_target": previous_target,
            "servo": servo,
            "error": error,
        }
        self._diagnostics_file.write(json.dumps(record, allow_nan=False) + "\n")

    def _read_joints(self, side: str) -> np.ndarray:
        joints = np.asarray(self.controllers[side].get_current_joint_positions(), dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError(f"{side} UR5 returned invalid joint positions: {joints}")
        return joints

    def _check_safety(self, side: str) -> None:
        receiver = self.controllers[side].rtde_r
        if (
            not receiver.isConnected()
            or receiver.getSafetyMode() not in (1, 2)
            or receiver.getRobotMode() != 7
        ):
            raise RuntimeError(
                f"{side} UR5 is disconnected, not running, or not in normal/reduced safety mode."
            )

    def _read_state(self) -> np.ndarray:
        state = np.concatenate([self._read_joints("left"), self._read_joints("right"), np.zeros(2)])
        for index, side in enumerate(("left", "right"), start=12):
            gripper = self.controllers[side].gripper
            opened, closed = gripper.get_open_position(), gripper.get_closed_position()
            if closed <= opened:
                raise RuntimeError(f"{side} Robotiq has invalid open/closed endpoints.")
            fraction = (gripper.get_current_position() - opened) / (closed - opened)
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise RuntimeError(f"{side} Robotiq returned an invalid position: {fraction}")
            state[index] = fraction
        return state

    def _check_worker(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError(
                "CB3 servo stopped; disconnect and inspect the fault before restarting."
            ) from error

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        try:
            self._check_worker()
            state = self._read_state()
            observation = dict(zip(STATE_NAMES, state.tolist(), strict=True))
            for name, camera in self.cameras.items():
                image = camera.read_latest(max_age_ms=self.config.camera_max_age_ms)
                if image.shape != self.observation_features[name] or image.dtype != np.uint8:
                    raise RuntimeError(
                        f"{name} camera returned an invalid image: {image.shape}, {image.dtype}"
                    )
                observation[name] = image
            return observation
        except Exception as error:
            self._fail(error)
            raise

    def _joint_target_error_details(self, index: int, target: np.ndarray, state: np.ndarray) -> str:
        side = "left" if index < 6 else "right"
        with self._lock:
            previous_target = None if self._target is None else float(self._target[index])
            servo = self._servo_status.get(side)
        details = [
            f"{STATE_NAMES[index]}: target={target[index]:.6f} rad",
            f"measured={state[index]:.6f} rad",
            f"delta={target[index] - state[index]:.6f} rad",
        ]
        if previous_target is None:
            details.append("previous_target=unavailable")
        else:
            details.extend(
                (
                    f"previous_target={previous_target:.6f} rad",
                    f"target_step_delta={target[index] - previous_target:.6f} rad",
                )
            )
        if servo is None:
            details.append("last_sent_command=unavailable")
        else:
            command = servo["command"][index % 6]
            age_ms = (time.monotonic() - self._diagnostics_started - servo["completed_at_s"]) * 1000
            details.extend(
                (
                    f"last_sent_command={command:.6f} rad",
                    f"target_minus_last_sent={target[index] - command:.6f} rad",
                    f"last_sent_minus_measured={command - state[index]:.6f} rad",
                    f"last_command_action_index={servo['action_index']}",
                    f"last_command_age_ms={age_ms:.3f}",
                )
            )
        return ", ".join(details)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        target = state = requested_target = None
        self._action_count += 1
        try:
            self._check_worker()
            if set(action) != set(STATE_NAMES):
                raise ValueError(
                    f"Expected exactly these 14 action keys, in radians/closure fraction: {STATE_NAMES}"
                )
            target = np.asarray([action[name] for name in STATE_NAMES], dtype=np.float64)
            if target.shape != (14,) or not np.isfinite(target).all():
                raise ValueError("Action must contain 14 finite scalar values.")
            state = self._read_state()
            if not self.config.enable_motion:
                self._dry_run_action_count += 1
            relative_violation = None
            for limit_name, values, limit in (
                ("joint_limit", np.abs(target[:12]), self.config.joint_limit),
                (
                    "max_relative_target relative to measured joints",
                    np.abs(target[:12] - state[:12]),
                    self.config.max_relative_target,
                ),
            ):
                violations = np.flatnonzero(values > limit)
                if violations.size:
                    details = "; ".join(
                        self._joint_target_error_details(index, target, state) for index in violations
                    )
                    message = (
                        f"Joint target exceeds {limit_name} ({limit:.6f} rad). {details}. "
                        f"enable_motion={self.config.enable_motion}; action_index={self._action_count}; "
                        f"max_joint_velocity={self.config.max_joint_velocity:.6f} rad/s; "
                        "rejected action was not sent."
                    )
                    if (
                        limit_name == "max_relative_target relative to measured joints"
                        and self.config.enable_motion
                        and self.config.clip_relative_target
                    ):
                        requested_target = target.copy()
                        goal_present_pos = {
                            name: (float(target[index]), float(state[index]))
                            for index, name in enumerate(STATE_NAMES[:12])
                        }
                        safe_goals = ensure_safe_goal_position(
                            goal_present_pos, float(self.config.max_relative_target)
                        )
                        target[:12] = [safe_goals[name] for name in STATE_NAMES[:12]]
                        if np.any(np.abs(target[:12]) > self.config.joint_limit):
                            raise ValueError(
                                "Clipped joint target exceeds joint_limit; rejected action was not sent."
                            )
                    elif (
                        limit_name == "max_relative_target relative to measured joints"
                        and self.config.dry_run_report_only
                        and not self.config.enable_motion
                    ):
                        relative_violation = message
                    else:
                        if not self.config.enable_motion:
                            message += f" Dry-run action #{self._dry_run_action_count}."
                        raise ValueError(message)
            target[12:] = np.clip(target[12:], 0, 1)
            if self.config.enable_motion:
                for side in self.controllers:
                    self._check_safety(side)
                self._record_diagnostics("accepted", target, state, requested_target=requested_target)
                with self._lock:
                    self._target = target.copy()
                    self._last_action_time = time.monotonic()
                    self._target_action_index = self._action_count
                self._ready.set()
                sent = target.copy()
                if not self.config.enable_gripper:
                    sent[12:] = state[12:]
            else:
                self._record_diagnostics("dry_run", target, state, error=relative_violation)
                now = time.monotonic()
                if relative_violation is not None and now - self._last_dry_run_warning >= 1.0:
                    logger.warning(
                        "Dry-run action #%d: %s Diagnostic only; not approved for motion.",
                        self._dry_run_action_count,
                        relative_violation,
                    )
                    self._last_dry_run_warning = now
                if now - self._last_dry_run_log >= 1.0:
                    logger.info(
                        "Dry-run action #%d target (not sent): %s; measured: %s; max_joint_delta=%.6f rad",
                        self._dry_run_action_count,
                        target.tolist(),
                        state.tolist(),
                        np.max(np.abs(target[:12] - state[:12])),
                    )
                    self._last_dry_run_log = now
                sent = state
            return dict(zip(STATE_NAMES, sent.tolist(), strict=True))
        except Exception as error:
            self._fail(error)
            try:
                self._record_diagnostics(
                    "rejected", target, state, error=str(error), requested_target=requested_target
                )
            except Exception:
                logger.exception("Could not record rejected UR5 action")
            raise

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._worker_error is None:
                self._worker_error = error
        self._stop.set()

    def _run_servo(self, side: str) -> None:
        controller = self.controllers[side]
        offset = 0 if side == "left" else 6
        command = None
        last_gripper = None
        servo_started = False
        previous_started = None
        try:
            while not self._stop.is_set():
                if not self._ready.wait(timeout=self.config.servo_time):
                    continue
                if self._stop.is_set():
                    break
                started = time.monotonic()
                with self._lock:
                    if self._target is None:
                        raise RuntimeError("No joint target is available.")
                    target = self._target.copy()
                    last_action_time = self._last_action_time
                    action_index = self._target_action_index
                if started - last_action_time > self.config.action_timeout_s:
                    raise TimeoutError("No fresh policy action; stopping both UR5 arms.")
                self._check_safety(side)
                actual = self._read_joints(side)
                if command is None:
                    command = actual.copy()
                if np.any(np.abs(command - actual) > self.config.max_tracking_error):
                    raise RuntimeError(f"{side} UR5 tracking error exceeds max_tracking_error.")
                max_step = self.config.max_joint_velocity * self.config.servo_time
                command += np.clip(target[offset : offset + 6] - command, -max_step, max_step)
                servo_started = True
                servo_call_started = time.monotonic()
                controller.servo_joints(command)
                completed = time.monotonic()
                status = {
                    "action_index": action_index,
                    "started_at_s": started - self._diagnostics_started,
                    "completed_at_s": completed - self._diagnostics_started,
                    "period_s": None if previous_started is None else started - previous_started,
                    "servo_call_s": completed - servo_call_started,
                    "target": target[offset : offset + 6].tolist(),
                    "command": command.tolist(),
                    "measured_before_send": actual.tolist(),
                }
                with self._lock:
                    self._servo_status[side] = status
                previous_started = started
                if self.config.enable_gripper and not self._stop.is_set():
                    gripper = controller.gripper
                    opened, closed = gripper.get_open_position(), gripper.get_closed_position()
                    position = round(opened + target[12 + offset // 6] * (closed - opened))
                    if position != last_gripper:
                        success, _ = gripper.move(
                            position, self.config.gripper_speed, self.config.gripper_force
                        )
                        if not success:
                            raise RuntimeError(f"{side} Robotiq rejected its position command.")
                        last_gripper = position
                self._stop.wait(max(0, self.config.servo_time - (time.monotonic() - started)))
        except Exception as error:
            logger.exception("%s CB3 servo failed", side)
            self._fail(error)
        finally:
            if servo_started:
                try:
                    controller.rtde_c.servoStop()
                except Exception:
                    logger.exception("Could not stop %s CB3 servo", side)

    def disconnect(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.error(
                    "Servo thread %s did not stop; verify both robots at the teach pendants.", thread.name
                )
        self._threads = []
        for side, controller in self.controllers.items():
            try:
                controller.close()
            except Exception:
                logger.exception("Could not close %s CB3 controller", side)
            if self.config.enable_motion:
                for interface in (controller.rtde_r, controller.rtde_c):
                    try:
                        interface.disconnect()
                    except Exception:
                        logger.exception("Could not disconnect %s RTDE interface", side)
        self.controllers.clear()
        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception:
                    logger.exception("Could not disconnect camera")
        self._is_connected = False
        if self._diagnostics_file is not None:
            try:
                self._record_diagnostics(
                    "disconnect",
                    self._target,
                    None,
                    error=None if self._worker_error is None else str(self._worker_error),
                )
            except Exception:
                logger.exception("Could not finalize UR5 diagnostic trace")
            finally:
                self._diagnostics_file.close()
                self._diagnostics_file = None
