#!/usr/bin/env bash
# UR5 真机 rollout + saliency 冒烟（tasks 6.2）
# 与 rollout.yaml 的差异：打开 display（saliency rerun 流的前提）、启用 saliency。
# 产出：outputs/saliency/<时间戳>/ 下每相机一个热力图叠加 mp4（含注意力预算条），
# rerun 中 saliency/<camera> 实体与 saliency/budget/* 曲线。
set -euo pipefail

METHOD="${1:-attention}"   # attention | grad_x_input

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml \
  --robot.enable_motion=true \
  --robot.enable_gripper=true \
  --robot.diagnostics_path="outputs/ur5_handoff_$(date +%Y%m%d_%H%M%S).jsonl" \
  --fps=20 --interpolation_multiplier=1 \
  --interactive=true --duration=0 \
  --display_data=true \
  --saliency.enabled=true \
  --saliency.method="${METHOD}"
