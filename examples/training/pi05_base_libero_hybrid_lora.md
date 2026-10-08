# PI05 base 在 LIBERO 上的混合 LoRA 配置对照

本配置使用通用预训练权重 `lerobot/pi05_base`。它在 LeRobot 中匹配
OpenPI issue #842 的可训练参数范围和主要超参数，并非作者实验的完整复现。
相同的可训练参数数量不代表输入处理、训练目标和数值实现相同。

## 使用方式

从仓库根目录运行，使用当前环境的 `HF_TOKEN` 和 `WANDB_API_KEY`：

```bash
CUDA_VISIBLE_DEVICES=1 \
bash examples/training/train_pi05_base_libero_hybrid_lora.sh
```

启动脚本调用标准 `lerobot-train`，不自行实现训练循环。GPU 由调用方选择，
数据集版本及训练参数读取 JSON 配置。可通过 `PI05_LIBERO_ROOT` 指定已有的
本地数据集，通过 `PI05_LOG_PATH` 指定日志路径，也可追加标准训练 CLI 参数。

仅需标准训练入口时，可以直接使用配置文件：

```bash
uv run --no-sync lerobot-train \
  --config_path=examples/training/pi05_base_libero_hybrid_lora.json
```

对照代码版本：`sen-code-lost/openpi` 的
`813c9b19f8a5058f71de8b462ae69d08bf60d8e1`。

- [实验报告](https://github.com/Physical-Intelligence/openpi/issues/842)
- [公开训练配置](https://github.com/sen-code-lost/openpi/blob/813c9b19f8a5058f71de8b462ae69d08bf60d8e1/src/openpi/training/config.py#L769)
- [公开评测脚本](https://github.com/sen-code-lost/openpi/blob/813c9b19f8a5058f71de8b462ae69d08bf60d8e1/examples/libero/main.py#L29)

## 已匹配的设置

| 设置 | 当前 LeRobot 配置 | 作者公开配置或脚本 |
| --- | --- | --- |
| 训练预测长度 | `chunk_size=10` | `action_horizon=10` |
| 执行长度 | `n_action_steps=5` | 评测脚本默认 `replan_steps=5` |
| Transformer LoRA | VLM 和 expert 的 q/k/v/o、gate/up/down | 同左 |
| LoRA rank / alpha | 16 / 16；dropout=0；rsLoRA 关闭 | 同左 |
| 全量训练模块 | SigLIP vision tower、action in/out、time MLP in/out | 同左 |
| 可训练参数 | 441,150,224 | 441,150,224 |
| 单卡 batch / 累积 | 16 / 1 | batch 16；每个 batch 更新一次 |
| 更新次数 | 30,000 | 30,000 |
| 学习率 | peak 1e-4、warmup 1,000、decay 30,000、end 1e-5 | 同左 |
| AdamW | betas=(0.9, 0.95)、eps=1e-8、clip=1、weight_decay=1e-10 | 同左 |
| 训练 seed | 42 | `TrainConfig` 默认 42 |
| EMA | 关闭 | 关闭 |
| 归一化模式 | STATE/ACTION 使用 QUANTILES | 分位数归一化 |
| LIBERO 动作 | 使用原始 delta 动作，不再转相对动作 | `extra_delta_transform=False` |

作者没有公开完整运行命令，因此这些是公开代码中的设置或默认值，不能确认
它们在报告 96.95% 的那次实验中均未被覆盖。尤其是初始 PyTorch 权重目录只有
占位符，无法确认作者使用了通用 base 还是 LIBERO 微调权重。

## 仍有差异或不能确认的设置

| 项目 | 当前 LeRobot 实现 | 作者公开代码 |
| --- | --- | --- |
| State 与 prompt | 保留原生 PI05 的离散 state prompt | `discrete_state_input=False`，tokenizer 不加入 state；PI05 的 expert 也不使用 PI0 的连续 state projection |
| 训练图像增强 | 当前配置关闭 dataset image transforms；PI05 只做 resize 和缩放 | forward 的预处理启用裁剪、旋转、亮度、对比度和饱和度增强 |
| Loss 的动作维度 | 仅对 LIBERO 的实际 7 维动作求 loss | 对 padding 后的 32 维 flow-matching loss 求平均 |
| LoRA 初始化 | PEFT 默认 A 为 Kaiming uniform、B 为 0 | 自定义 A 为 normal(std=0.01)、B 为 0 |
| 精度与实现 | PEFT、Accelerate bf16 autocast；本地 vision tower 原始权重保留 FP32 | 原生 LoRA；主要权重 BF16，仅指定 vision embedding 等权重保留 FP32 |
| 数据与统计来源 | `lerobot/libero` 的指定 revision 和已有 q01/q99 | `physical-intelligence/libero` 与单独计算的 OpenPI norm stats；具体统计文件未公布 |
| 保存频率 / workers | 每 10k 步保存；8 workers | 默认每 1k 步保存；2 workers；影响存储和吞吐 |
| 正式评测 | 需使用本次训练保存的 processor，明确设置执行步数 | 公开脚本默认每任务 50 次、seed=7；报告实际评测命令未公布 |

训练预测长度与执行长度独立。预测 10 步、执行前 5 步后重新观察，是普通的
滚动重规划。本配置没有启用 RTC；RTC 还涉及异步推理、推理延迟和动作片段衔接，
并不要求预测长度必须是 50。

`n_action_steps` 不改变训练动作序列或训练 loss，所以从 10 改为 5 不需要重训。
`chunk_size` 则决定训练目标长度，改为 50 会成为另一组训练实验。

## 已完成的验证及限制

2026-10-06 的短跑从 `lerobot/pi05_base` 成功加载全部权重，用 batch 16 完成
3 次优化器更新，显存峰值约 20.2 GiB。保存的 adapter 包含全部 441,150,224
个可训练参数，检查确认 VLM LoRA、expert LoRA、完整视觉权重和动作投影均更新。

短跑时使用旧配置的 `n_action_steps=10`、`weight_decay=0.01`、`seed=1000`。
后续已将这三项修正为公开代码的 5、1e-10、42，并通过配置解析和 scheduler /
optimizer preset 检查。当前配置于 2026-10-07 完成单卡 30,000 次优化器更新，
最终 loss 为 0.116，训练日志中的显存峰值约 20.2 GiB。

随后使用 30k checkpoint 及其保存的 processor 完成四个 LIBERO suite 的评测：
每个任务 50 次、共 2,000 次，成功 1,881 次，总成功率为 94.05%。

| Suite | 成功次数 / 评测次数 | 成功率 |
| --- | --- | --- |
| LIBERO-Spatial | 476 / 500 | 95.2% |
| LIBERO-Object | 490 / 500 | 98.0% |
| LIBERO-Goal | 477 / 500 | 95.4% |
| LIBERO-10 | 438 / 500 | 87.6% |

本地结果文件为 `outputs/eval/pi05_base_libero_hybrid_lora_r16_30k/eval_info.json`。
这些结果验证了当前 LeRobot 配置的训练和评测流程；上述实现差异仍然存在，
不能据此认定完整复现了 issue 作者的实验。
