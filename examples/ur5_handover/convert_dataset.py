#!/usr/bin/env python

"""Convert canonical UR5 handover demonstrations to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets import LeRobotDataset


CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
EXPECTED_SCHEMA = "robotwin_canonical_teleop"
EXPECTED_ACTION_SEMANTICS = "actual_relative_delta_pose_euler_xyz"
EXPECTED_CONTROL_MODE = "bimanual_pd_ee_delta_pose"
DEFAULT_HELDOUT_EPISODES = (6, 33, 49, 55, 57, 66, 76, 78)
STATE_AXES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (*[f"joint_{index}" for index in range(6)], "gripper")
)
ACTION_AXES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
)


@dataclass(frozen=True)
class EpisodePaths:
    source_index: int
    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "episode.json"

    @property
    def hdf5(self) -> Path:
        return self.root / "data.hdf5"

    def video(self, camera_name: str) -> Path:
        return self.root / f"{camera_name}.mp4"


class VideoReader:
    def __init__(self, path: Path, *, expected_frames: int, expected_fps: int):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if frame_count != expected_frames:
            raise ValueError(f"{path}: expected {expected_frames} frames, found {frame_count}")
        if not np.isclose(fps, expected_fps, atol=1.0e-3):
            raise ValueError(f"{path}: expected {expected_fps} FPS, found {fps}")

    def read_rgb(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Video ended early: {self.path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def assert_exhausted_after_final_observation(self) -> None:
        ok, _ = self.capture.read()
        if not ok:
            raise RuntimeError(f"Missing final observation frame: {self.path}")
        ok, _ = self.capture.read()
        if ok:
            raise ValueError(f"Video contains extra frames: {self.path}")

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=None,
        help="Optional correction dataset. All successful episodes are appended as dataset_source=1.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5-handover")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("train", "heldout", "all"),
        default="train",
        help=(
            "Select source episodes before applying --num-episodes. "
            "Defaults to train to avoid stats leakage."
        ),
    )
    parser.add_argument(
        "--heldout-episodes",
        default=",".join(str(index) for index in DEFAULT_HELDOUT_EPISODES),
        help="Comma-separated source episode indices recorded in conversion_manifest.json.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    return parser.parse_args()


def parse_episode_index(path: Path) -> int:
    try:
        return int(path.name.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"Invalid episode directory name: {path.name}") from exc


def discover_episodes(raw_root: Path) -> list[EpisodePaths]:
    episode_dirs = sorted(path for path in (raw_root / "episodes").glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {raw_root / 'episodes'}")
    return [EpisodePaths(parse_episode_index(path), path) for path in episode_dirs]


def select_episodes(
    episodes: list[EpisodePaths],
    *,
    split: str,
    heldout: set[int],
    num_episodes: int | None,
) -> list[EpisodePaths]:
    if split == "train":
        selected = [episode for episode in episodes if episode.source_index not in heldout]
    elif split == "heldout":
        selected = [episode for episode in episodes if episode.source_index in heldout]
    else:
        selected = episodes
    if num_episodes is not None:
        if num_episodes <= 0:
            raise ValueError("--num-episodes must be positive")
        selected = selected[:num_episodes]
    if not selected:
        raise ValueError(f"No episodes selected for split={split!r}")
    return selected


def load_episode_metadata(episode: EpisodePaths) -> dict:
    metadata = json.loads(episode.metadata.read_text(encoding="utf-8"))
    expected = {
        "schema_name": EXPECTED_SCHEMA,
        "control_freq_hz": 20,
        "control_mode": EXPECTED_CONTROL_MODE,
        "translation_unit": "meter",
        "rotation_unit": "radian",
        "action_semantics": EXPECTED_ACTION_SEMANTICS,
        "translation_frame": "per_arm_root_aligned",
        "gripper_semantics": "absolute_1_open_0_closed",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{episode.root.name}: incompatible metadata: {mismatches}")
    if not metadata.get("success", False):
        raise ValueError(f"{episode.root.name}: only successful demonstrations are accepted")
    return metadata


def validate_arrays(episode: EpisodePaths, states: np.ndarray, actions: np.ndarray) -> None:
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"{episode.root.name}: expected state shape [T+1, 14], got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"{episode.root.name}: expected action shape [T, 14], got {actions.shape}")
    if states.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f"{episode.root.name}: expected observations=actions+1, "
            f"got {states.shape[0]} and {actions.shape[0]}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{episode.root.name}: state or action contains NaN/Inf")

    for offset in (0, 7):
        translation_norm = np.linalg.norm(actions[:, offset : offset + 3], axis=1)
        rotation_norm = np.linalg.norm(actions[:, offset + 3 : offset + 6], axis=1)
        if np.any(translation_norm > 0.015 + 1.0e-6):
            raise ValueError(f"{episode.root.name}: translation delta exceeds 0.015 m")
        if np.any(rotation_norm > 0.10 + 1.0e-6):
            raise ValueError(f"{episode.root.name}: rotation delta exceeds 0.10 rad")


def dataset_features(image_shape: tuple[int, int, int]) -> dict:
    return {
        **{
            f"observation.images.{camera_name}": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            }
            for camera_name in CAMERA_NAMES
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": list(STATE_AXES)},
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": list(ACTION_AXES)},
        },
        "source_episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "source_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "dataset_source": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }


def inspect_image_shape(episode: EpisodePaths) -> tuple[int, int, int]:
    shapes = set()
    for camera_name in CAMERA_NAMES:
        capture = cv2.VideoCapture(str(episode.video(camera_name)))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {episode.video(camera_name)}")
            shapes.add(
                (
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    3,
                )
            )
        finally:
            capture.release()
    if len(shapes) != 1:
        raise ValueError(f"Camera resolutions differ in {episode.root.name}: {sorted(shapes)}")
    return shapes.pop()


def convert_episode(
    dataset: LeRobotDataset, episode: EpisodePaths, *, dataset_source: int
) -> tuple[int, str]:
    metadata = load_episode_metadata(episode)
    with h5py.File(episode.hdf5, "r") as h5_file:
        states = np.asarray(h5_file["observations/robot/agent_qpos"], dtype=np.float32)
        actions = np.asarray(h5_file["actions/control/delta_pose_actual"], dtype=np.float32)
    validate_arrays(episode, states, actions)

    expected_observations = actions.shape[0] + 1
    with ExitStack() as stack:
        readers = {
            camera_name: stack.enter_context(
                VideoReader(
                    episode.video(camera_name),
                    expected_frames=expected_observations,
                    expected_fps=int(metadata["control_freq_hz"]),
                )
            )
            for camera_name in CAMERA_NAMES
        }
        for frame_index, action in enumerate(actions):
            frame = {
                "task": metadata["task_instruction"],
                "observation.state": states[frame_index],
                "action": action,
                "source_episode_index": np.asarray([episode.source_index], dtype=np.int64),
                "source_frame_index": np.asarray([frame_index], dtype=np.int64),
                "dataset_source": np.asarray([dataset_source], dtype=np.int64),
            }
            frame.update(
                {
                    f"observation.images.{camera_name}": reader.read_rgb()
                    for camera_name, reader in readers.items()
                }
            )
            dataset.add_frame(frame)

        for reader in readers.values():
            reader.assert_exhausted_after_final_observation()

    dataset.save_episode()
    return int(actions.shape[0]), str(metadata["task_instruction"])


def parse_heldout_episodes(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_root = args.raw_root.expanduser().resolve()
    correction_root = args.correction_root.expanduser().resolve() if args.correction_root else None
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    heldout = parse_heldout_episodes(args.heldout_episodes)
    primary_episodes = select_episodes(
        discover_episodes(raw_root),
        split=args.split,
        heldout=heldout,
        num_episodes=args.num_episodes,
    )
    episode_groups = [(0, "original", raw_root, primary_episodes)]
    if correction_root is not None:
        correction_episodes = select_episodes(
            discover_episodes(correction_root), split="all", heldout=set(), num_episodes=None
        )
        episode_groups.append((1, "correction", correction_root, correction_episodes))

    image_shape = inspect_image_shape(primary_episodes[0])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=20,
        robot_type="bimanual_ur5",
        features=dataset_features(image_shape),
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )

    converted = []
    try:
        total_episodes = sum(len(group[3]) for group in episode_groups)
        for dataset_source, source_name, source_root, episodes in episode_groups:
            for episode in episodes:
                output_episode_index = len(converted)
                logging.info(
                    "Converting %s episode %d (%d/%d)",
                    source_name,
                    episode.source_index,
                    output_episode_index + 1,
                    total_episodes,
                )
                frame_count, task = convert_episode(
                    dataset, episode, dataset_source=dataset_source
                )
                converted.append(
                    {
                        "output_episode_index": output_episode_index,
                        "dataset_source": dataset_source,
                        "source_name": source_name,
                        "source_root": str(source_root),
                        "source_episode_index": episode.source_index,
                        "frames": frame_count,
                        "task": task,
                    }
                )
    finally:
        dataset.finalize()

    selected_source_indices = {
        item["source_episode_index"] for item in converted if item["dataset_source"] == 0
    }
    manifest = {
        "source_root": str(raw_root),
        "correction_root": str(correction_root) if correction_root is not None else None,
        "dataset_sources": {"0": "original", "1": "correction"} if correction_root else {"0": "original"},
        "repo_id": args.repo_id,
        "split": args.split,
        "fps": 20,
        "state_semantics": "left/right arm joint positions followed by absolute gripper state",
        "action_semantics": EXPECTED_ACTION_SEMANTICS,
        "action_order": list(ACTION_AXES),
        "observation_action_alignment": (
            "observation[t] -> action[t]; final observation is not a training frame"
        ),
        "train_source_episodes": sorted(selected_source_indices - heldout),
        "heldout_source_episodes": sorted(selected_source_indices & heldout),
        "episodes": converted,
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info(
        "Converted %d episodes and %d frames to %s",
        len(converted),
        sum(item["frames"] for item in converted),
        output_root,
    )


if __name__ == "__main__":
    main()
