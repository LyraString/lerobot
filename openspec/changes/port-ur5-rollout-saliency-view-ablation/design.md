## Context

`feat/rollout-attention-saliency` 分支（基于旧 main `fbb811fc`）实现了 rollout saliency 与 view ablation，并捆带了 view_mask、dm05 插件、openpi client、pi05 训练配方、ee6d 等多项特性。UR5 分支基于较新 main（含 `3f2c29ef feat(rollout): bring your own strategy` BYOS），且工作区已有未提交改动。

两分支尖端差异达 219 文件 / +21463 / −6592，不可整体 merge。共同祖先与各侧改动（相对 base 的文件数：分支 149、UR5 84）表明大部分 saliency 依赖文件在 UR5 与 base 完全一致，可直接取分支版本；仅 14 个文件两侧都改，其中 9 个真实冲突（`uv.lock` 94 段，其余 8 文件各 1–2 段）。

关键约束：`git merge-tree` 两方向冲突结构相同，故选择方向不影响冲突量，只影响合并后的基线——必须选择「分支 → UR5」以保留较新 main 与 BYOS。

## Goals / Non-Goals

**Goals:**

- 把 rollout saliency（仅 SmolVLA explainer）与 view ablation 移植到 UR5 基线。
- 保留 UR5 的 BYOS 默认 `setup()` 语义，并让 view ablation 通过该单点对所有策略默认生效。
- 与上游分支保持可对齐（未来可再 pull 这两项特性）。
- 不引入第三方依赖、不破坏 UR5 现有工作。

**Non-Goals:**

- 不移植 pi0/pi05 explainer、dm05 插件、openpi client、pi05 SO-101 训练配方、ee6d、脚本套件。
- 不移植 `openspec/` 与 `CONTEXT.md` 过程文档、分支版 `pyproject.toml`。
- 不修改 saliency/view_mask 的既有语义（本变更是移植，不是重设计）。

## Decisions

### 决策 1：采用白名单精挑移植，禁止整体 merge 分支

分支上无关特性占比高，整体 merge 会连带 dm05/openpi/训练配方等，且分支版 `pyproject.toml` 含 `openpi-client` 本地路径依赖会破坏 UR5 依赖。**必须**逐个特性挑选。

替代方案：整分支 merge（否决——带入无关特性与坏依赖）；rebase（否决——两分支分叉过深且方向错）。

### 决策 2：文件分三类处理

- **纯新增**（`rollout/saliency/`、`rollout/view_mask/`）：**必须**整体拷贝，`saliency/__init__.py` 禁止导出 pi 相关符号。
- **UR5 未改过（与 base 一致）**：**必须**直接取分支版本，零冲突。
- **两侧都改**：**必须**手工三方合并，逐个确认 keep-both 语义。

### 决策 3：`setup()` 采用 BYOS 默认实现并集中接线 View Mask

合并后的 `RolloutStrategy.setup()`：

```python
def setup(self, ctx):            # 具体方法，保留 BYOS 语义
    self._init_engine(ctx)
    self._attach_view_mask(ctx)  # --view_mask.enabled=false 时 no-op
```

各内置策略（base/sentry/highlight/dagger/episodic）**必须**改为 `super().setup(ctx)` 后仅保留自身额外初始化。

理由：分支原实现因 `setup()` 是 `@abc.abstractmethod` 而被迫在每个策略手加 `_attach_view_mask`；BYOS 把 `setup()` 变具体后，单点接线可让内置与第三方策略默认一致（契合 openspec「跨策略一致」目标）。**禁止**策略在 `super().setup(ctx)` 之后再显式调 `_attach_view_mask`（非幂等，会重复 `add_flush_hook`）。

替代方案：保留逐策略显式调用（否决——与默认实现冗余，且第三方策略静默失去 view ablation）。

### 决策 4：RTC 注入与 provenance 机制整体取分支版本

saliency 的噪声注入 + 回放快照（`saliency_sink`、`_build_saliency_snapshot`、`accepts_noise_injection`）、view_mask 的 revision 判定，均位于 `rollout/inference/rtc.py`；`policies/rtc/{action_queue,modeling_rtc,__init__}.py` 提供 `ActionQueueMergeResult` 与 `preserve_graph`。这些文件 UR5 未改过，**必须**整体取分支版本，**禁止**手工重写，以免破坏 `preserve_graph` 的有限差分语义。

### 决策 5：`strategies/factory.py` 保留 UR5 版

分支未改该文件；UR5 的 BYOS 回退（`make_device_from_device_class`）是新增能力，**必须**保留，不参与合并。

### 决策 6：`constants.py` / `converters.py` 采用相邻新增的 keep-both

两侧都在 `QUERY_TEXT` 后新增不同常量（分支 `RUNTIME_CONTEXT`，UR5 `LANGUAGE_*`/`MESSAGES_RENDERED`），**必须**同时保留；`converters.py` 的 `RUNTIME_CONTEXT` 透传与 UR5 改动同理共存。

### 决策 7：先切专用分支，禁止直接改 UR5

**必须**新建 `feat/ur5-saliency`（工作区已有未提交改动，需先 commit 或 stash）再动手；`UR5` 分支在移植验证通过前保持不动，便于整体回滚。

### 决策 8：测试裁剪到 SmolVLA

`tests/test_saliency.py` 含 pi 用例，**必须**裁剪为仅 SmolVLA 用例；view_mask 与 rerun 测试整体移植。**禁止**为通过测试而放宽 spec 声明的容差。

## Risks / Trade-offs

- **`rollout/inference/rtc.py` 移植语义漂移** → 取分支版本后用 saliency 测试与「回放—前向一致性（1e-5）」验收；不做任何手工改写。
- **`_attach_view_mask` 非幂等的重复注册** → 单点接线 + spec 明令禁止策略二次调用；code review 检查 `super().setup(ctx)` 后无显式调用。
- **`uv.lock` 冲突（94 段）** → **必须**用 `uv lock` 重新生成而非手工合并；分支版 `pyproject.toml` 禁止采用。
- **带上 view_mask 的连带范围** → 已确认用户接受；禁用时惰性，不影响现有 rollout。
- **数据集 schema 变更影响续写** → 启用 view ablation 时旧 schema 数据集必须 fail-fast；spec 已覆盖。
- **工作区脏改动在切分支时丢失** → 切分支前必须先 commit/stash。

## Migration Plan

1. commit/stash 现有工作区改动；从 UR5 新建 `feat/ur5-saliency`。
2. 拷贝 `saliency/`（去 pi）、`view_mask/` 两个新包。
3. 取分支版本覆盖「UR5 未改过」文件。
4. 三方合并 9 个冲突文件（`configs.py`、`context.py`、`strategies/{core,dagger}.py`、`scripts/lerobot_rollout.py`、`constants.py`、`converters.py`、`.gitignore`、`uv.lock`），`uv.lock` 用 `uv lock` 重生成。
5. 移植测试并裁剪。
6. 验证：smolvla UR5 rollout `--saliency.enabled=true --saliency.method=attention --inference.type=rtc --display_mode=rerun --display_data=true`；以及 view ablation 热键切换。
7. 回滚策略：整个移植在独立分支，未合并前 `git branch -D feat/ur5-saliency` 即可丢弃。

## Open Questions

- 移植后是否需要在 `docs/source/inference.mdx` 补充 saliency/view ablation 用户文档？（默认：先不补，功能验证后再定）
