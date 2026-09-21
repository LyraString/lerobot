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
