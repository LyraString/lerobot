# Tasks: 移植 rollout-saliency 与 rollout-view-ablation 到 UR5

## 1. 准备

- [x] 1.1 commit 或 stash 当前工作区改动（`pyproject.toml`、`src/lerobot/configs/parser.py`、`scripts/lerobot_rollout.py`、测试、`uv.lock` 等），确保切分支不丢改动
- [x] 1.2 从 UR5 新建并切换到 `feat/ur5-saliency` 分支
- [x] 1.3 记录基线：确认 `git merge-base UR5 317lab/feat/rollout-attention-saliency` 与两分支相对 base 的改动文件清单，作为后续核对的依据

## 2. 新增包（纯拷贝）

- [x] 2.1 拷贝 `src/lerobot/rollout/saliency/` 全部文件，删除 `pi.py` 与 `pi_attention.py`
- [x] 2.2 修改 `saliency/__init__.py`：移除 `pi`/`pi_attention` 的 import 与 `__all__` 导出，保留 `SmolVLAExplainer`/`SmolVLAAttentionExplainer`/`Explainer`/`registry`
- [x] 2.3 拷贝 `src/lerobot/rollout/view_mask/` 全部文件（11 个）
- [x] 2.4 校验：`uv run python -c "import lerobot.rollout.saliency, lerobot.rollout.view_mask"` 无 import 错误

## 3. 取分支版本（UR5 未改过的文件）

- [x] 3.1 取分支版本覆盖模型侧：`policies/common/flow_matching.py`、`policies/smolvla/{modeling_smolvla,smolvlm_with_expert}.py`
- [x] 3.2 取分支版本覆盖 RTC：`policies/rtc/{__init__,action_queue,modeling_rtc}.py`
- [x] 3.3 取分支版本覆盖 rollout 引擎：`rollout/inference/{base,factory,rtc,sync}.py`、`rollout/controller.py`、`rollout/interactive.py`
- [x] 3.4 取分支版本覆盖策略基类与三个策略：`rollout/strategies/{base,episodic,highlight,sentry}.py`
- [x] 3.5 取分支版本覆盖 utils：`utils/{rerun_visualization,keyboard_input,action_interpolator}.py`
- [x] 3.6 逐文件 `git diff` 复核，确认未意外带入 pi/dm05/openpi/训练配方改动

## 4. 三方合并（两侧都改的文件）

- [x] 4.1 `rollout/configs.py`：保留 UR5 策略配置重构，加入 `SaliencyConfig` + `ViewMaskConfig` 字段与 `validate_saliency`/`validate_view_mask` 调用（**禁止**采用分支版 `pyproject.toml`）
- [x] 4.2 `rollout/context.py`：保留 UR5 改动，加入 saliency manager 与 view_mask runtime 装配
- [x] 4.3 `rollout/strategies/core.py`：保留 BYOS 默认 `setup()`，在其中 `_init_engine` 后加入 `_attach_view_mask`；保留分支的 view_mask 系列方法（`_attach_view_mask`/`_view_mask_tick`/`_view_mask_frame_fields`/`_log_view_mask` 等）
- [x] 4.4 `rollout/strategies/dagger.py`：`setup()` 改为 `super().setup(ctx)`，合并 UR5 的 num_episodes/streaming 提示与分支的 view_mask 逻辑，**禁止**二次调用 `_attach_view_mask`
- [x] 4.5 `rollout/strategies/{base,sentry,highlight,episodic}.py`：`setup()` 改为 `super().setup(ctx)`（移除各自的 `_init_engine` 显式调用与 branch 的 `_attach_view_mask` 调用），保留各自额外初始化
- [x] 4.6 `scripts/lerobot_rollout.py`：合并 saliency 启动接线（`set_view_mask`）与退出 `saliency.stop()`，保留 UR5 改动
- [x] 4.7 `utils/constants.py` 与 `processor/converters.py`：两侧新增常量/透传 keep-both（`RUNTIME_CONTEXT` 与 `LANGUAGE_*`/`MESSAGES_RENDERED`）
- [x] 4.8 `.gitignore`：合并两侧新增忽略项
- [x] 4.9 `uv.lock`：用 `uv lock` 重新生成，**禁止**手工合并 94 段冲突
- [x] 4.10 `rollout/strategies/factory.py` 保留 UR5 版本，**禁止**用分支版本覆盖

## 5. 测试移植

- [x] 5.1 移植 `tests/test_saliency.py`，裁剪掉 pi0/pi05 用例，仅保留 SmolVLA
- [x] 5.2 移植 `tests/rollout/test_view_mask.py` 与 `tests/rollout/test_view_mask_integration.py`
- [x] 5.3 移植 `tests/utils/test_rerun_visualization.py` 的 saliency 相关用例
- [x] 5.4 为「默认 `setup()` 集中接线 View Mask」补一个断言：策略经 `super().setup(ctx)` 后 view ablation 生效且 flush hook 只注册一次
- [x] 5.5 为「未启用 view_mask 时零成本」补断言：禁用时 `_attach_view_mask` 不改变引擎/策略状态

## 6. 验证与收尾

- [x] 6.1 跑移植后的 saliency/view_mask 测试与 `tests/test_rollout.py`、`tests/test_interactive_rollout.py`，全部通过
- [ ] 6.2 【需真机 + 联网模型】冒烟：SmolVLA UR5 rollout `--saliency.enabled=true --saliency.method=attention --inference.type=rtc --display_mode=rerun --display_data=true`，确认热力图视频与预算条产出
- [ ] 6.3 【需真机】冒烟：`--view_mask.enabled=true` 下用 F1/F2/F3 切换视角，确认黑图替换、revision 递增与事件 JSONL
- [x] 6.4 回归：`--saliency.enabled=false` 与 `--view_mask.enabled=false` 时，rollout 行为与移植前一致
- [x] 6.5 `pre-commit run --all-files` 通过（ruff/typos 等）
