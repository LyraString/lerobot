"""Offline tensor smoke check using cached LIBERO data and SmolVLA weights.

This adapts feature metadata and normalization to LIBERO in memory. It does
NOT fine-tune the checkpoint or validate the meaning/quality of its actions.
No simulator or robot is connected. Do not deploy these outputs as controls.

Run from the repository root with HF_HUB_OFFLINE=1 and uv run --no-sync.
"""

import argparse

import torch
from huggingface_hub import snapshot_download

from lerobot.configs import FeatureType
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.feature_utils import dataset_to_policy_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-id", default="lerobot/libero")
    parser.add_argument("--dataset-revision", default="v3.0")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a CPU-only check.")
    torch.manual_seed(args.seed)

    # Resolve existing snapshots explicitly: the dataset constructor otherwise
    # queries remote version refs even when its files are already cached.
    model_root = snapshot_download(
        args.model_id, revision=args.model_revision, local_files_only=True
    )
    dataset_root = snapshot_download(
        args.dataset_id,
        repo_type="dataset",
        revision=args.dataset_revision,
        cache_dir=HF_LEROBOT_HUB_CACHE,
        local_files_only=True,
    )
    print("Model snapshot:", model_root, flush=True)
    print("Dataset snapshot:", dataset_root, flush=True)
    dataset = LeRobotDataset(args.dataset_id, root=dataset_root, episodes=[args.episode])
    frame = dict(dataset[0])
    reference_action = frame.pop("action")  # Never feed the demonstration action to inference.

    config = PreTrainedConfig.from_pretrained(model_root)
    print("Original checkpoint state/action shapes:",
          config.input_features["observation.state"].shape, config.action_feature.shape)
    features = dataset_to_policy_features(dataset.features)
    config.input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    config.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    config.device = args.device
    if config.input_features["observation.state"].shape[0] > config.max_state_dim:
        raise ValueError("Dataset state exceeds model padding capacity.")
    if config.action_feature.shape[0] > config.max_action_dim:
        raise ValueError("Dataset action exceeds model padding capacity.")

    # SmolVLA projects padded state/action vectors. Metadata can be adapted
    # without changing the learned tensors, but this does not teach LIBERO.
    policy = SmolVLAPolicy.from_pretrained(model_root, config=config, strict=True).eval()
    preprocess, postprocess = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    batch = preprocess(frame)
    print("Task:", frame["task"])
    print("Processed tensors:", {k: tuple(v.shape) for k, v in batch.items()
                                 if isinstance(v, torch.Tensor)})
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite model input: {key}")

    policy.reset()
    with torch.inference_mode():
        normalized_action = policy.select_action(batch)
        action = postprocess(normalized_action)
    expected_shape = (1, *config.action_feature.shape)
    if tuple(action.shape) != expected_shape:
        raise ValueError(f"Expected {expected_shape}, got {tuple(action.shape)}")
    if not torch.isfinite(normalized_action).all() or not torch.isfinite(action).all():
        raise ValueError("Model returned NaN/Inf.")
    print("Device:", next(policy.parameters()).device)
    print("Action shape:", tuple(action.shape))
    print("Normalized prediction:", normalized_action.detach().cpu().tolist())
    print("Postprocessed prediction:", action.detach().cpu().tolist())
    print("Dataset action (context only; not a quality metric):", reference_action.tolist())
    print("PASS: finite action tensor produced. This is NOT a LIBERO task evaluation.")


if __name__ == "__main__":
    main()
