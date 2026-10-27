"""Compare SmolVLA chunks with demonstration actions without connecting to any hardware."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors


def summarize_errors(predictions, targets, states, valid):
    errors = np.abs(predictions - targets)
    hold_errors = np.abs(states[:, None, :] - targets)
    rows = []
    horizons = [step for step in (0, 1, 5, 10, 20, 30, 40, 49) if step < predictions.shape[1]]
    for step in horizons:
        selected = valid[:, step]
        if not selected.any():
            continue
        joints = errors[selected, step, :12]
        rows.append(
            {
                "step": step,
                "samples": int(selected.sum()),
                "joint_mae_rad": float(joints.mean()),
                "joint_max_rad": float(joints.max()),
                "sample_max_joint_error_p90_rad": float(np.quantile(joints.max(axis=1), 0.9)),
                "hold_joint_mae_rad": float(hold_errors[selected, step, :12].mean()),
                "gripper_mae_fraction": errors[selected, step, 12:].mean(axis=0).tolist(),
            }
        )
    return {
        "joint_mae_rad": float(errors[valid, :12].mean()),
        "hold_joint_mae_rad": float(hold_errors[valid, :12].mean()),
        "per_joint_mae_rad": errors[valid, :12].mean(axis=0).tolist(),
        "gripper_mae_fraction": errors[valid, 12:].mean(axis=0).tolist(),
        "horizons": rows,
    }


def save_views(sample, keys, path):
    sheet = Image.new("RGB", (320 * len(keys), 265), "white")
    drawing = ImageDraw.Draw(sheet)
    for column, key in enumerate(keys):
        pixels = sample[key]
        if pixels.ndim == 4:
            pixels = pixels[-1]
        pixels = pixels.permute(1, 2, 0).numpy()
        picture = Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8))
        picture.thumbnail((320, 240))
        sheet.paste(picture, (column * 320, 25))
        drawing.text((column * 320 + 5, 5), key.removeprefix("observation.images."), fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=Path("policy/SmolVLA"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/carto317/XRoboToolkit-Teleop-Sample-Python/datasets/dual_ur5_demo_coloured"),
    )
    parser.add_argument("--repo-id", default="local/dual-ur5-demo-coloured")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples_per_episode < 1:
        parser.error("--samples-per-episode must be positive")

    config = PreTrainedConfig.from_pretrained(args.policy_path)
    if config.type != "smolvla" or config.action_feature.shape != (14,):
        parser.error("This diagnostic expects a 14-dimensional dual-UR5 SmolVLA policy")
    config.device = args.device
    config.rtc_config = None
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend=args.video_backend,
    )
    requested_episodes = set(args.episodes) if args.episodes is not None else None
    episodes = [
        episode
        for episode in metadata.episodes
        if requested_episodes is None or episode["episode_index"] in requested_episodes
    ]
    if not episodes or (
        requested_episodes is not None
        and requested_episodes != {episode["episode_index"] for episode in episodes}
    ):
        parser.error("Requested episodes are missing from the dataset")

    output_dir = args.output_dir or Path("outputs/ur5_offline_eval") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output_dir.resolve()}", flush=True)
    print("Loading checkpoint; no robot or camera connection is created.", flush=True)
    policy = get_policy_class(config.type).from_pretrained(args.policy_path, config=config)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    predictions, targets, states, masks, sample_details = [], [], [], [], []
    total_samples = len(episodes) * args.samples_per_episode
    for episode in episodes:
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        for fraction in np.arange(args.samples_per_episode) / args.samples_per_episode:
            index = start + min(int((end - start) * fraction), end - start - 1)
            sample = dataset[index]
            observation = {key: sample[key] for key in config.input_features}
            observation["task"] = sample["task"]
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            torch.manual_seed(args.seed + index)
            batch = preprocessor(observation)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            with torch.no_grad():
                predicted = postprocessor(policy.predict_action_chunk(batch)).squeeze(0).cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            target = sample["action"].numpy()
            state = sample["observation.state"].numpy().reshape(-1, 14)[-1]
            valid = ~sample["action_is_pad"].numpy()
            if predicted.shape != target.shape or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid prediction at frame {index}: {predicted.shape}")
            predictions.append(predicted)
            targets.append(target)
            states.append(state)
            masks.append(valid)
            sample_details.append(
                {
                    "episode": int(episode["episode_index"]),
                    "frame": index - start,
                    "index": index,
                    "fraction": float(fraction),
                    "latency_s": elapsed,
                    "first_step_joint_mae_rad": float(np.abs(predicted[0, :12] - target[0, :12]).mean()),
                }
            )
            if episode == episodes[0]:
                save_views(sample, list(config.image_features), output_dir / f"views_{index:06d}.jpg")
            count = len(predictions)
            if count == 1 or count % 10 == 0 or count == total_samples:
                print(f"Evaluated {count}/{total_samples}; last chunk {elapsed:.3f}s", flush=True)

    predictions, targets, states, valid = map(np.stack, (predictions, targets, states, masks))
    report = {
        "policy_path": str(args.policy_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "seed": args.seed,
        "episodes": len(episodes),
        "samples": len(predictions),
        "fps": metadata.fps,
        "rtc_enabled": False,
        "notes": "Demonstration-frame prediction accuracy, not closed-loop success or hardware validation.",
        "metrics": summarize_errors(predictions, targets, states, valid),
        "by_phase": {
            str(fraction): summarize_errors(
                predictions[selected], targets[selected], states[selected], valid[selected]
            )
            for fraction in sorted({sample["fraction"] for sample in sample_details})
            if (selected := np.array([sample["fraction"] == fraction for sample in sample_details])).any()
        },
        "sample_details": sample_details,
    }
    np.savez_compressed(
        output_dir / "chunks.npz", predictions=predictions, targets=targets, states=states, valid=valid
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"Report: {(output_dir / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
