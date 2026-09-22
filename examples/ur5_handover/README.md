# UR5 handover dataset conversion

This converter ports the canonical `HandoverBlockUR5-v2` demonstrations to a
local LeRobot dataset. It preserves the three RGB cameras, 14-dimensional
bimanual proprioceptive state, 14-dimensional actual-relative delta-pose action,
20 Hz control rate, and task instruction.

Each training row uses `observation[t] -> action[t]`. The final observation in
each source episode is validated but is not written as a training row.

## Smoke test

Use a new output path because the converter refuses to overwrite data:

```bash
uv run --no-sync python examples/ur5_handover/convert_dataset.py \
  --raw-root=/home/idiot/zhj/zhj_maniskill/data/HandoverBlockUR5-v2/pico_handover_clean_v2 \
  --output-root=/tmp/ur5_handover_lerobot_smoke \
  --repo-id=local/ur5-handover-smoke \
  --num-episodes=1
```

## Full conversion

```bash
uv run --no-sync python examples/ur5_handover/convert_dataset.py \
  --raw-root=/home/idiot/zhj/zhj_maniskill/data/HandoverBlockUR5-v2/pico_handover_clean_v2 \
  --output-root=/home/idiot/zhj/ler/datasets/ur5_handover_clean_v2 \
  --repo-id=local/ur5-handover-clean-v2
```

The default `--split=train` excludes source episodes
`6,33,49,55,57,66,76,78`, so normalization statistics are computed only from
the 72 training demonstrations. Convert the eight held-out demonstrations to a
different output path when offline validation needs them:

```bash
uv run --no-sync python examples/ur5_handover/convert_dataset.py \
  --raw-root=/home/idiot/zhj/zhj_maniskill/data/HandoverBlockUR5-v2/pico_handover_clean_v2 \
  --output-root=/home/idiot/zhj/ler/datasets/ur5_handover_clean_v2_heldout \
  --repo-id=local/ur5-handover-clean-v2-heldout \
  --split=heldout
```

Each generated `conversion_manifest.json` records source-to-output episode
mapping and split membership. Conversion is local and does not push the dataset
to Hugging Face Hub.

## SmolVLA evaluation

The policy server runs in the LeRobot environment while the evaluator runs in
the ManiSkill environment. The evaluator restores the eight held-out recorded
initial states, predicts 50-step chunks, executes 10 actions before replanning,
and converts each actual-relative delta immediately before that control step.

Start the 50k checkpoint server in the LeRobot checkout:

```bash
cd /home/idiot/zhj/ler/lerobot
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
uv run --no-sync python examples/ur5_handover/policy_server.py \
  --checkpoint=outputs/train/ur5_handover_smolvla_full_50k/checkpoints/050000/pretrained_model \
  --device=cuda
```

Run its held-out evaluation from a second terminal:

```bash
cd /home/idiot/zhj/zhj_maniskill
CUDA_VISIBLE_DEVICES=2 \
/home/idiot/anaconda3/envs/zhjrl/bin/python script/handover/eval_smolvla.py \
  --policy-label=smolvla-050000 \
  --output=tmp/handover/smolvla_050000/results.json
```

Stop the first server after the evaluation, then repeat with the 100k checkpoint:

```bash
cd /home/idiot/zhj/ler/lerobot
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
uv run --no-sync python examples/ur5_handover/policy_server.py \
  --checkpoint=outputs/train/ur5_handover_smolvla_full_50k/checkpoints/100000/pretrained_model \
  --device=cuda
```

```bash
cd /home/idiot/zhj/zhj_maniskill
CUDA_VISIBLE_DEVICES=2 \
/home/idiot/anaconda3/envs/zhjrl/bin/python script/handover/eval_smolvla.py \
  --policy-label=smolvla-100000 \
  --output=tmp/handover/smolvla_100000/results.json
```

Both runs use held-out episodes `6,33,49,55,57,66,76,78`, policy seed `999`,
an execution horizon of 10, and at most 1000 control steps. Results and videos
are written incrementally, so rerunning the same command resumes missing
episodes. Compare the `aggregate` objects in the two result JSON files.

## VR correction collection

Start the selected policy server as above. In a second terminal, run the
correction collector from the ManiSkill checkout:

```bash
cd /home/idiot/zhj/zhj_maniskill
CUDA_VISIBLE_DEVICES=2 \
/home/idiot/anaconda3/envs/zhjrl/bin/python \
  script/handover/collect_smolvla_corrections.py \
  smolvla_corrections_round01 \
  --policy-label=smolvla-050000 \
  --seed=2000
```

Each reset waits for `B` before starting the policy. Press `A` to pause and
prepare a takeover, align both controller triggers with the current gripper
commands, then press `B` to begin VR control and correction recording. Grip and
Trigger keep the standard VR clutch and gripper behavior. Successful corrections
are saved automatically. Autonomous policy successes are counted but do not
create dataset episodes. Press `X` to discard and retry the current seed, or `Y`
to discard and continue with the next seed.

Corrections are stored under
`data/HandoverBlockUR5-v2/smolvla_corrections_round01`. Only the human-controlled
suffix is recorded; policy actions before takeover are excluded from training
data. This raw correction dataset is intentionally kept separate until the data
aggregation phase.

## DAgger-style fine-tuning

Convert the 72 original training demonstrations and all successful round-01
corrections into one dataset. The conversion stores `dataset_source=0` for the
original data and `dataset_source=1` for corrections:

```bash
cd /home/idiot/zhj/ler/lerobot
uv run --no-sync python examples/ur5_handover/convert_dataset.py \
  --raw-root=/home/idiot/zhj/zhj_maniskill/data/HandoverBlockUR5-v2/pico_handover_clean_v2 \
  --correction-root=/home/idiot/zhj/zhj_maniskill/data/HandoverBlockUR5-v2/smolvla_corrections_round01 \
  --output-root=/home/idiot/zhj/ler/datasets/ur5_handover_dagger_round01 \
  --repo-id=local/ur5-handover-dagger-round01 \
  --split=train
```

Start a new full-model fine-tuning run from the 50k policy checkpoint. This is
not a resume: optimizer and scheduler state start fresh. `steps=20000` counts
micro-batches; with gradient accumulation 2 this produces 10,000 optimizer
updates and an effective batch size of 24. The Accelerate-wrapped scheduler
advances only on optimizer updates, so its 500-step warmup and 10,000-step decay
span the whole run. Each sampler epoch contains exactly 75% original frames and
25% correction frames. The pretrained checkpoint's normalization statistics
remain authoritative.

```bash
cd /home/idiot/zhj/ler/lerobot

(
  read -rsp "请输入你的 W&B API Key: " WANDB_API_KEY
  echo
  [[ -n "$WANDB_API_KEY" ]] || exit 1
  export WANDB_API_KEY

  CUDA_VISIBLE_DEVICES=1 \
  uv run --no-sync lerobot-train \
    --policy.path=outputs/train/ur5_handover_smolvla_full_50k/checkpoints/050000/pretrained_model \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.optimizer_lr=1e-5 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=10000 \
    --dataset.repo_id=local/ur5-handover-dagger-round01 \
    --dataset.root=/home/idiot/zhj/ler/datasets/ur5_handover_dagger_round01 \
    --dataset.sampling_group_key=dataset_source \
    --dataset.sampling_group_weights='[0.75,0.25]' \
    --preserve_pretrained_processor_stats=true \
    --policy.device=cuda \
    --accelerator.mixed_precision=bf16 \
    --batch_size=12 \
    --accelerator.gradient_accumulation.steps=2 \
    --steps=20000 \
    --save_freq=4000 \
    --output_dir=outputs/train/ur5_handover_smolvla_dagger_r01_full_10k \
    --job_name=ur5_handover_smolvla_dagger_r01_full_10k \
    --wandb.enable=true \
    --wandb.mode=online \
    --wandb.entity=lyrastring-zhejiang-university \
    --wandb.project=smolvla-ur5-handover \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=false
)
```
