# PI05 base 在双 UR5 数据上的原生 LoRA 微调

本配置继续使用通用预训练权重 `lerobot/pi05_base` 和已有 UR5 数据，
在标准 `lerobot-train` 中启用原生 LoRA，不依赖 PEFT 包装器。
参考 [OpenPI issue #842](https://github.com/Physical-Intelligence/openpi/issues/842)
及作者分支的 `813c9b19f8a5058f71de8b462ae69d08bf60d8e1` 实现。

## 配置入口与训练命令

编辑同目录的 `pi05_base_dual_ur5_native_lora.json`，或通过标准 CLI 覆盖：

```bash
cd /home/idiot/zhj/ler/lerobot

(
  run_tag=$(date +%Y%m%d_%H%M%S)
  run_name="pi05_base_dual_ur5_native_lora_r128_b8_acc2_${run_tag}"

  # HF_TOKEN and WANDB_API_KEY are inherited from this subshell's environment.
  set -o pipefail
  mkdir -p outputs/train
  CUDA_VISIBLE_DEVICES=1 uv run --no-sync lerobot-train \
    --config_path=examples/training/pi05_base_dual_ur5_native_lora.json \
    --output_dir="outputs/train/${run_name}" \
    --job_name="${run_name}" \
    2>&1 | tee "outputs/train/${run_name}.log"
)
```

| 设置                   | 当前值                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| LoRA rank / alpha      | `policy.native_lora.rank=128` / `alpha=128`                                                |
| LoRA 初始化            | A 为 normal(std=0.01)，B 为 0；dropout=0，缩放 alpha/rank                                  |
| 精度                   | `dtype=bfloat16`，`use_openpi_precision=true`，BF16 autocast                               |
| 训练范围               | VLM / expert 的 q/k/v/o、gate/up/down 使用 LoRA；视觉编码器及 action/time 投影直接全量训练 |
| Batch / 梯度累计       | 8 / 2；单卡有效 batch 16                                                                   |
| 总训练预算             | 20,000 个微批次，10,000 次参数更新，约 160,000 帧                                          |
| 学习率                 | peak 1e-4，warmup 666，decay 20,000，end 1e-5                                              |
| 保存 / 验证 / 日志间隔 | 4,000 / 2,000 / 200 个微批次                                                               |
| 动作长度               | 预测 50 步，默认执行前 25 步                                                               |
| 数据 / 验证划分        | 原始双 UR5 数据；41 条训练演示、5 条验证演示                                               |
| 归一化                 | STATE/ACTION 使用本数据集 QUANTILES，VISUAL 使用 IDENTITY                                  |

`peft` 必须为 `null`，`policy.use_peft` 保持 `false`。
原生 LoRA 与 PEFT 同时启用会报错。调整 rank 时同时调整 alpha，可保持 alpha/rank 不变。

## 与此前 PEFT 实验的区别

原生 LoRA 直接持有原线性层的 weight/bias，不复制基座权重；适配器沿用原层精度，
不额外将 BF16 中间张量转成 FP32。全量训练模块在原模型中直接解冻，
不创建 PEFT `modules_to_save` 的冻结副本。

`use_openpi_precision=true` 时，BF16 模型的视觉编码器和多模态投影使用 BF16；
vision patch/position embedding 和指定的 Transformer norm 保留 FP32。
SigLIP 的 embedding 输出进入编码层前转换到编码层精度，参考作者的实现。
action/time 投影仍为 FP32。该设置会改变数值精度，不能视为与旧 FP32 训练完全相同。

默认 `native_lora=null`、`use_openpi_precision=false`，保留此前训练和 PEFT checkpoint 的行为。
新方案仍使用 LeRobot 的 state prompt、实际动作维度 loss 和原有数据处理；
这里只参考作者的 LoRA 与精度实现，不宣称完整复现其 LIBERO 实验或 16 GB 显存报告。
原 issue 报告的是 rank 16、batch 16；本配置是 rank 128、batch 8、三路 UR5 图像和 50 步动作。

## 保存、恢复与工作站加载

每次保存生成完整 `model.safetensors`，包含基座、LoRA 及全量训练的模块，
并保存标准 config、train config、processor 和 tokenizer。文件比旧 adapter checkpoint 大。
复制整个 `pretrained_model/` 后，模型权重不再依赖单独下载 `lerobot/pi05_base`。
本配置的模型文件约 8.34 GiB，优化器状态约 2.32 GiB；每份训练 checkpoint
约 10.7 GiB。按当前保存间隔保留五份，需预留约 53 GiB，另加日志和运行余量。

工作站需要包含本次原生 LoRA 支持的 LeRobot 代码，才能识别新配置和适配器层。
通过标准 `--policy.path=.../pretrained_model` 加载，处理器仍使用训练保存的 UR5 统计量。
本配置保持原始 14 维绝对动作、相机排列、数据单位和 50 / 25 调度设置。

`--resume=true --config_path=.../pretrained_model/train_config.json` 继续使用标准训练恢复流程。
需要保留同一 checkpoint 下的 `training_state/`；仅做 rollout 不需要该目录。
原生 LoRA 和 PEFT 的 checkpoint 不直接互换，不覆盖旧实验。

原生 checkpoint 缺失适配器或读取失败会报错，不返回随机初始化模型。
首次从没有 LoRA 的 base 加载时，才允许初始化新的 A/B 适配器。

## 本机验证

2026-10-12 在单张 RTX 4090 上，使用真实 UR5 训练数据完成 rank 128、batch 8、
梯度累计 2 的 6 个微批次（3 次参数更新），训练 PyTorch 分配峰值为 15.36 GiB。
loss 和梯度有限，并完成 16 个验证帧的离线 loss 检查及标准 checkpoint 保存。
这不是完整 10k 次更新训练，也不代表实机成功率或长期运行的最终显存峰值。

随后从完整 checkpoint 重新加载全部权重，用真实 UR5 样本预测 `(1, 50, 14)`
动作，并通过保存的 processor 反归一化；输出均有限。检查确认 LoRA B 非零、
视觉投影权重更新、抽查的冻结 VLM 投影与原始 base 一致。
这次单样本推理的 PyTorch 分配峰值约为 8.57 GiB。

验证日志和结果保存在
`outputs/train/pi05_native_lora_validation_r128_b8_acc2_20261012_185553.log`
和同名目录的 `validation_info.json`。短验证的大 checkpoint 文件在完成重载检查后清理。
对应原生 LoRA、历史观测、RTC 及 PEFT revision 的单元检查为 41 passed；
额外的视觉训练集成检查中有 3 个自带跳过标记，原测试注明会产生 OOM。
