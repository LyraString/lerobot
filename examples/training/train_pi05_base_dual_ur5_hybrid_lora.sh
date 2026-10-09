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

# Train the general PI05 base on local dual-UR5 joint-position demonstrations.
# Optional: PI05_UR5_DATASET_ROOT, PI05_LOG_PATH, CUDA_VISIBLE_DEVICES, and CLI overrides.
# HF_TOKEN and WANDB_API_KEY may be exported in a subshell for temporary accounts.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config_path="examples/training/pi05_base_dual_ur5_hybrid_lora.json"
train_args=(--config_path="$config_path")
if [[ -n "${PI05_UR5_DATASET_ROOT:-}" ]]; then
  train_args+=(--dataset.root="$PI05_UR5_DATASET_ROOT")
fi

log_path="${PI05_LOG_PATH:-outputs/train/pi05_base_dual_ur5_hybrid_lora_r16_10k_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$log_path")"
printf 'Config: %s\nConsole log: %s\n' "$config_path" "$log_path"

uv run --no-sync lerobot-train "${train_args[@]}" "$@" 2>&1 | tee "$log_path"
