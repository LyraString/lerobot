## Why

UR5 分支需要在真机 rollout 时具备两类可解释性/消融能力：查看策略动作的显著性来源（saliency map），以及在运行中受控地移除指定相机视角（view ablation）。这两个能力已在 `feat/rollout-attention-saliency` 分支实现，但该分支基于较旧的 main，需要把这两项特性移植到基于较新 main（含 BYOS 策略机制）的 UR5 分支上。

## What Changes

- 移植 **rollout-saliency**：新增 `rollout/saliency/` 包，支持两种 method——`grad_x_input`（像素敏感度图）与 `attention`（Patch 注意力图）；含注意力预算条、叠加热力图 mp4 录制、rerun 实时流、以及解释「下一入队动作 a_d」的精确回放与噪声注入机制。仅移植 **SmolVLA** explainer。
- 移植 **rollout-view-ablation**：新增 `rollout/view_mask/` 包，支持 CLI 指定初始屏蔽视角、运行中单键锁存切换、黑图替换策略输入、revision 化的原子 Mask Transition（APPLYING/ACTIVE/FAILED）、数据集逐帧 mask 字段与事件 JSONL、rerun MASKED 标注。
- 移植支撑机制：`policies/rtc/` 的噪声注入与 provenance（merge 元数据、mask revision），`policies/smolvla/` 的注意力 capture tap，`policies/common/flow_matching.py` 的 `preserve_graph`，`utils/rerun_visualization.py` 的 saliency 实体。
- 合并策略：`RolloutStrategy.setup()` 采用 BYOS 的默认实现，在其中集中挂载 `_attach_view_mask(ctx)`；各内置策略的 `setup()` 改为 `super().setup(ctx)` 后只保留自身额外工作。**BREAKING**（对内部策略实现者）：`_attach_view_mask` 非幂等，各策略**禁止**再显式调用它。
- 排除项：pi0/pi05 explainer、dm05 插件、openpi 远程 client、pi05 SO-101 训练配方、ee6d、脚本套件、`openspec/` 与 `CONTEXT.md` 过程文档、分支版 `pyproject.toml`。

## Capabilities

### New Capabilities

- `rollout-saliency`：rollout 显著性可视化——两种 method 的语义与产物、叠加热力图视频、注意力预算条、rerun 实时流、精确回放与噪声注入约束。
- `rollout-view-ablation`：rollout 视角消融——视角选择与黑图语义、动态切换与状态机、记录与数据集字段、显示标注、跨策略一致性与启动校验。

### Modified Capabilities

（无）

## Impact

- **代码**：新增 `src/lerobot/rollout/saliency/`、`src/lerobot/rollout/view_mask/`；修改 `rollout/{configs,context,controller,interactive,inference/*,strategies/*}.py`、`policies/rtc/{__init__,action_queue,modeling_rtc}.py`、`policies/common/flow_matching.py`、`policies/smolvla/{modeling_smolvla,smolvlm_with_expert}.py`、`utils/{rerun_visualization,keyboard_input,action_interpolator}.py`、`processor/converters.py`、`scripts/lerobot_rollout.py`。
- **测试**：移植 `tests/test_saliency.py`（裁剪至 SmolVLA 用例）、`tests/rollout/test_view_mask*.py`、`tests/utils/test_rerun_visualization.py`。
- **依赖**：不新增第三方依赖（opencv/rerun 已在）。
- **数据集 schema**：启用 view ablation 时新增逐帧 `view_mask`/`view_mask_revision` 字段与 `meta/view_mask_events.jsonl`；旧 schema 数据集不允许续写。
