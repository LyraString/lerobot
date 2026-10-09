import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


def default_cameras() -> dict[str, CameraConfig]:
    return {
        name: RealSenseCameraConfig(serial_number_or_name=serial, fps=30, width=640, height=480)
        for name, serial in (
            ("exterior", "937622072836"),
            ("left_wrist", "941322071481"),
            ("right_wrist", "045322072148"),
        )
    }


@RobotConfig.register_subclass("dual_ur5_cb3")
@dataclass
class DualUR5CB3Config(RobotConfig):
    teleop_root: Path = Path("/home/carto317/XRoboToolkit-Teleop-Sample-Python")
    left_robot_ip: str = "192.168.1.50"
    right_robot_ip: str = "192.168.1.60"
    enable_motion: bool = False
    enable_gripper: bool = False
    dry_run_report_only: bool = False
    diagnostics_path: Path | None = None
    servo_time: float = 0.017
    lookahead_time: float = 0.1
    servo_gain: float = 300.0
    joint_limit: float = 2 * math.pi
    max_joint_velocity: float = 0.6
    max_relative_target: float = 0.3
    clip_relative_target: bool = False
    max_tracking_error: float = 0.3
    action_timeout_s: float = 1.0
    gripper_speed: int = 255
    gripper_force: int = 128
    camera_max_age_ms: int = 200
    cameras: dict[str, CameraConfig] = field(default_factory=default_cameras)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.teleop_root = Path(self.teleop_root).expanduser()
        if self.diagnostics_path is not None:
            self.diagnostics_path = Path(self.diagnostics_path).expanduser()
        if self.dry_run_report_only and self.enable_motion:
            raise ValueError("dry_run_report_only requires enable_motion=false.")
        if not self.left_robot_ip or not self.right_robot_ip or self.left_robot_ip == self.right_robot_ip:
            raise ValueError("Specify distinct, nonempty left and right UR5 IP addresses.")
        for name in (
            "servo_time",
            "joint_limit",
            "max_joint_velocity",
            "max_relative_target",
            "max_tracking_error",
            "action_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.03 <= self.lookahead_time <= 0.2 or not 100 <= self.servo_gain <= 2000:
            raise ValueError("Use lookahead_time in [0.03, 0.2] and servo_gain in [100, 2000].")
        if self.action_timeout_s <= self.servo_time:
            raise ValueError("action_timeout_s must exceed servo_time.")
        if self.camera_max_age_ms <= 0:
            raise ValueError("camera_max_age_ms must be positive.")
        if not 0 <= self.gripper_speed <= 255 or not 0 <= self.gripper_force <= 255:
            raise ValueError("Robotiq speed and force must be in [0, 255].")
