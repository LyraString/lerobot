# PI05 base 双 UR5 四卡全量微调

使用 `lerobot/pi05_base`，训练视觉编码器、语言主干、动作专家和动作/时间投影，
关闭 PEFT 和原生 LoRA。数据保持三路相机、14 维绝对动作及数据集分位数归一化。

## 环境与启动

在仓库根目录安装依赖：

```bash
uv pip install -e ".[pi,training]"
```

在仓库根目录激活虚拟环境后启动：

```bash
source .venv/bin/activate
bash examples/training/train_pi05_base_dual_ur5_full_finetune.sh
```

脚本采用官方多卡写法，由 `torchrun` 启动四个 `lerobot-train` 进程，
通过 `--config_path` 读取 JSON 中的训练参数、输出目录和任务名。
启动前将脚本中的 `CUDA_VISIBLE_DEVICES=0,1,2,3` 改为选定的四张空闲 GPU。
新建训练任务时，在 JSON 中设置新的 `output_dir`，已有训练使用下方的恢复命令。

W&B 默认开启，项目为 `pi05-dual-ur5`；训练前在当前虚拟环境执行 `wandb login`。
每 10 步记录训练指标，关闭模型 artifact 上传，本地 checkpoint 照常保存。

## 配置与步数

| 设置 | 默认值 |
| --- | --- |
| 并行方式 | FSDP2，`parallelism.dp_shard=4` |
| 精度 | FP32 主参数，BF16 混合精度计算 |
| 每卡 batch / 梯度累计 | 32 / 1，全局有效 batch 128 |
| 训练预算 | 448 步及优化器更新，约 2 遍训练集 |
| 学习率 | peak 2.5e-5，warmup 15，decay 448，end 2.5e-6 |
| 验证 / 保存 / 日志 | 每 224 / 224 / 10 步 |
| 数据划分 | 41 条训练演示（28,585 帧），最后 5 条验证演示（3,236 帧） |
| 动作调度 | 预测 50 步，执行前 25 步 |

`policy.dtype=float32` 保留基座权重精度；`accelerator.mixed_precision=bf16`
让 FSDP2 在前后向计算时转换参数精度。开启 `policy.gradient_checkpointing` 降低激活显存。

`steps`、scheduler 和保存/验证间隔均按微批次计数。改变 batch 或梯度累计后，
需要重新计算训练步数和这些间隔，才能符合预期的样本预算与训练遍数。
当前每步处理 `32 × 4 = 128` 个采样帧，每遍约需 `ceil(28585 / 128) = 224` 步。

pi0.5 联合注意力直接调用投影层，因此默认按 MLP、视觉编码层和 Embedding 分片，
保留原模型的参数名称及计算路径，避免使用不会进入的 decoder 层 forward 作为分片边界。

## 保存与恢复

每份 checkpoint 包含完整的 `pretrained_model/model.safetensors`、处理器和训练配置，
以及 `training_state/` 中的分片优化器状态。恢复训练时保留整个 checkpoint，
使用四卡启动并传入保存的配置，将 `<run>` 替换为训练输出目录名：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  "$(which lerobot-train)" \
  --config_path="outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true
```

部署时使用 `pretrained_model/`；它是完整模型权重，不依赖 LoRA adapter。
保存分片优化器前，各卡同步并释放未使用的 CUDA 缓存，为 NCCL 额外通信资源留出显存。

## 已完成的验证

- 两卡缩小模型的 FP32/BF16 梯度对比通过，覆盖真实 pi0.5 前向、梯度检查点和梯度累计。
- 预训练权重加载与原有原生 LoRA 相关的 11 项回归检查通过。
- 完整 `pi05_base` 和真实三路 UR5 数据完成四卡 batch32、BF16 的 448 步训练，
  第 224、448 步的模型、优化器分片、调度和随机数状态均已保存。
- 原 112 步保存失败场景在释放 CUDA 空闲缓存后验证通过；保存时 0 号卡占用
  从约 39.05 GiB 降至 13.94 GiB，未再出现 NCCL 显存不足。

断点恢复及实机推理尚未验证。训练产物与日志保存在本地 `outputs/`，不纳入版本控制。
