#!/usr/bin/env bash
# Copyright 2026 LeRobot contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Hybrid scheme from OpenPI #842: Transformer LoRA + full SigLIP/action projections.
# The example config starts from lerobot/pi05_base, the general pretrained checkpoint.
# HF_TOKEN and WANDB_API_KEY may be exported in a subshell for temporary accounts.
# Optional: PI05_LIBERO_ROOT, PI05_LOG_PATH, CUDA_VISIBLE_DEVICES, and CLI overrides.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config_path="examples/training/pi05_base_libero_hybrid_lora.json"

libero_root="$(uv run --no-sync python - "$config_path" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

config = json.loads(Path(sys.argv[1]).read_text())
dataset_config = config["dataset"]
download_kwargs = {
    "repo_id": dataset_config["repo_id"],
    "repo_type": "dataset",
    "revision": dataset_config.get("revision"),
    "cache_dir": HF_LEROBOT_HUB_CACHE,
}
root = os.environ.get("PI05_LIBERO_ROOT")
if root is None:
    try:
        root = snapshot_download(**download_kwargs, local_files_only=True)
    except LocalEntryNotFoundError:
        root = None
    if root is None or not (Path(root) / "meta/info.json").is_file():
        root = snapshot_download(
            **download_kwargs,
            allow_patterns=["meta/**", "data/**", "videos/**"],
        )

stats = json.loads((Path(root) / "meta/stats.json").read_text())
for feature in ("action", "observation.state"):
    for quantile in ("q01", "q99"):
        if quantile not in stats.get(feature, {}):
            raise ValueError(f"{feature} needs {quantile} for QUANTILES normalization: {root}")
print(Path(root).resolve())
PY
)"

log_path="${PI05_LOG_PATH:-outputs/train/pi05_base_libero_hybrid_lora_r16_30k_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$log_path")"
printf 'Config: %s\nDataset: %s\nConsole log: %s\n' "$config_path" "$libero_root" "$log_path"

uv run --no-sync lerobot-train \
  --config_path="$config_path" \
  --dataset.root="$libero_root" \
  "$@" \
  2>&1 | tee "$log_path"
