# PI05 base 在实机双 UR5 数据上的混合 LoRA 微调

从通用预训练模型 `lerobot/pi05_base` 开始，沿用 LIBERO 实验的混合 LoRA 和优化器配置：
VLM 和 action expert 的 attention / MLP 使用 rank 16 的 LoRA，视觉编码器和
action / time 投影全量训练。这不是从 LIBERO 微调 checkpoint 继续训练。
为增加实机推理的时间余量，预测 / 默认执行长度改为 50 / 25。

SmolVLA 实机训练用于核对数据特征与标准训练流程；不继承它的全参数微调方式、
MEAN_STD 归一化或学习率配置。训练仍调用标准 `lerobot-train`，
不需要另写 UR5 数据转换、训练循环或连接实机控制程序。

## 数据与动作定义

当前本地数据集为 `/home/idiot/zhj/ler/datasets/dual_ur5_demo_coloured`，
LeRobot v3 格式，共 46 个 episode、31,821 帧，采样频率 20 Hz。
任务文本为 `Pick up the object and place it on the scales`。

| 特征                | 定义                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `observation.state` | 14 维：左臂 6 关节、右臂 6 关节、左夹爪、右夹爪                                                  |
| `action`            | 同样的 14 维顺序；按采集数据中的绝对关节位置目标训练                                             |
| 三路 RGB            | `observation.images.exterior`、`observation.images.left_wrist`、`observation.images.right_wrist` |
| 相机分辨率          | 数据为 640×480，PI05 内部保留宽高比并 pad / resize 到 224×224                                    |

关节数值与弧度位置一致，夹爪数据位于 `[0, 1]`。这是从元数据与数值检查得到的判断，
不是对实机采集程序单位、夹爪开合方向和硬件控制接口的验证。
本配置沿用已有 SmolVLA 数据中的动作目标，不将它转换为 LIBERO 的末端增量动作。

配置保持 `use_relative_actions=false`，不额外做 `action - state`，也不改变数据。
STATE 和 ACTION 使用该数据集已有的 q01/q99 做 QUANTILES 归一化，
`preserve_pretrained_processor_stats=false`；不沿用 LIBERO 或通用 base 的统计量。
三个真实相机保持原键名与排列，不添加空相机，不开启历史观测记忆。

## 首轮训练

从仓库根目录启动，GPU 由调用方选择；凭据从当前进程环境读取：

```bash
CUDA_VISIBLE_DEVICES=1 \
bash examples/training/train_pi05_base_dual_ur5_hybrid_lora.sh
```

也可以直接调用标准训练入口：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync lerobot-train \
  --config_path=examples/training/pi05_base_dual_ur5_hybrid_lora.json
```

`PI05_UR5_DATASET_ROOT` 可以覆盖本地数据路径，`PI05_LOG_PATH` 可以指定控制台日志。
启动脚本也接受标准训练 CLI 覆盖，例如 `--wandb.enable=false`。
机器路径、GPU、账号与实验参数均在配置、环境变量或 CLI 中设置。

混合 LoRA、精度和 AdamW 参数与 `pi05_base_libero_hybrid_lora.json` 保持一致。
训练预算按 UR5 数据量缩短为 10k 步，warmup 从 LIBERO 的 1,000 步按比例缩短
为 333 步，decay 改为 10k；预测 / 默认执行长度从 10 / 5 调整为 50 / 25：

| 设置                       | 当前 UR5 配置                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------- |
| 预训练权重                 | `lerobot/pi05_base`                                                                |
| Transformer LoRA           | VLM 和 expert 的 q/k/v/o、gate/up/down；rank / alpha = 16 / 16                     |
| 全参数训练模块             | `vision_tower`、`action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out` |
| 精度与显存优化             | BF16 权重、BF16 autocast、gradient checkpointing                                   |
| 单卡 batch / 梯度累计      | 16 / 1                                                                             |
| 训练更新 / 保存间隔        | 10,000 / 2,000                                                                     |
| 离线验证                   | `eval_split=0.1`；每 1,000 步计算全部验证帧的 loss                                 |
| 学习率                     | peak 1e-4、warmup 333、decay 10,000、end 1e-5                                      |
| AdamW                      | betas=(0.9, 0.95)、eps=1e-8、clip=1、weight_decay=1e-10                            |
| 预测 / 默认执行长度        | 50 / 25；LIBERO 配置为 10 / 5                                                      |
| 归一化                     | STATE/ACTION 使用 QUANTILES，VISUAL 使用 IDENTITY                                  |
| Seed / EMA / torch.compile | 42 / 关闭 / 关闭                                                                   |

数据相关的适配为：本地 UR5 数据路径、14 维状态与动作、三路真实相机，
`empty_cameras=0`，以及 UR5 自己的归一化统计量和任务文本。W&B 项目与输出目录
单独命名。沿用已有 SmolVLA 实机实验的 episode 验证划分，`eval_split=0.1`：
本数据集只有一个任务，标准加载器将最后 5 条完整轨迹（episode 41–45）留作验证，
前 41 条（episode 0–40）用于训练，分别为 3,236 和 28,585 帧。
这是按完整 episode 划分，不把同一条轨迹的相邻帧分到训练和验证两边。

`eval_steps=1000`，`max_eval_samples=0` 表示每次使用全部验证帧，在终端和 W&B
记录验证 loss。验证使用 `eval()` 和 `no_grad()`，不会更新模型参数；完成后继续训练。
归一化仍使用原数据集 `meta/stats.json` 的全量统计，和已有 SmolVLA 训练流程一致；
没有重新按训练子集计算统计量。Flow-matching loss 中的噪声与时间仍随机采样，
观察多次验证的趋势，不把单次波动直接当成过拟合。

本轮 10,000 次更新共采样约 160,000 帧，约为训练子集的 5.60 遍；
每 2,000 步保存，可对比 2k、4k、6k、8k、10k checkpoint。验证每 1k 步进行，
共 10 次；每次保存 checkpoint 时也会计算验证 loss。作为对照，LIBERO 的
273,465 帧数据在 batch 16 下训练 30k 步约为 1.76 遍。按帧数对齐采样遍数不保证
新任务训练充分，本轮用验证趋势判断拟合情况。训练和验证 loss 均不代表实机成功率。
三路真实图像的计算量可能与 LIBERO 中的两路真实图像
加一路空图像不同，相同 batch 不保证相同峰值显存。

本配置预测 50 个动作，默认执行前 25 个再重新观察：在 20 Hz 下分别对应
2.5 秒的预测序列和 1.25 秒的动作执行段。该执行长度是策略 `select_action`
接口的默认行为，工作站的 chunk 调度由其控制程序决定。模型推理和网络延迟会影响实际时间。
执行段延长会降低重规划频率，也会延后对新观测和动作误差的响应。预测长度增加则会
增加动作专家的计算量和训练显存；此前 10 步预测的推理耗时不能直接视为 50 步的耗时。
这是同步滚动重规划的配置，不是 RTC 训练；若以后使用 RTC，需要根据实测推理延迟
另行配置控制程序，不能假定 `n_action_steps` 就是 RTC 的执行调度。

若降低 batch 并提高累积，注意本仓库的 `steps` 和 scheduler 按微批次计数。
例如 batch 8、累积 2 要达到本轮同样的 160k 样本、10k 优化器更新，应将
`steps` / decay 改为 20k、warmup 改为 666、save_freq 改为 4k、eval_steps 改为 2k，
并选择新的输出目录。

本轮使用新的 `_10k` 输出目录和日志名称，从通用 base 重新开始训练，`resume=false`。
先前由用户停止的 `_30k` 运行目录与日志保留，不复用其输出路径。

## 传输训练结果

训练完成后使用：

```text
outputs/train/pi05_base_dual_ur5_hybrid_lora_r16_10k/checkpoints/010000/pretrained_model/
```

传输整个 `pretrained_model/` 目录，包括 `config.json`、`adapter_config.json`、
`adapter_model.safetensors`、pre/post processor 的配置和统计量，以及 tokenizer。
这是 PEFT adapter，加载时还需要可用的 `lerobot/pi05_base` 权重，不能只复制
adapter 权重就假定可以完全离线启动。

本例只负责本机训练，不依赖另一台机器上的 UR5 采集或控制脚本。
训练完成后由用户把 checkpoint 传到工作站，并在现有实机控制程序中加载。
输入需保持同样的状态和相机定义，使用本次保存的 processor 得到 14 维绝对动作目标。
训练配置不会连接机器人，也不改变工作站上的控制程序。

## 本机验证记录

2026-10-09 完成以下 CPU 检查，未启动新的 10k 训练或加载策略权重：

- 标准 `TrainPipelineConfig.from_pretrained` 解析与 `validate` 通过；优化器和
  scheduler preset 与上表一致。与 LIBERO JSON 对照，policy 的显式设置仅
  `empty_cameras`、`chunk_size`、`n_action_steps` 与 warmup / decay 步数不同，
  PEFT、精度和 AdamW 超参数一致。
- 标准数据加载器划分为 41 个训练 episode / 28,585 帧和 5 个验证 episode /
  3,236 帧，episode 集合不重叠；训练和验证 batch 的 state 均为
  `(2, 14)`、动作目标为 `(2, 50, 14)`，三路 RGB 均成功解码。
- 加载 `lerobot/pi05_base` 保存的 processor，并覆盖为 UR5 特征与 q01/q99；
  task/state tokenization 输出 `(2, 200)`，归一化结果有限，动作反归一化还原检查通过。
- JSON、`bash -n` 检查通过；通过隔离的 `uvx` 环境运行本次文件的大小、
  冲突、末尾换行、空白、拼写与 Markdown 格式检查，均通过。
- 完整 pre-commit 未完成：初始化 gitleaks 环境时下载的 Go 压缩包不完整，
  解压报 `EOFError`；gitleaks 检查未运行。

此前 30k 配置曾开始训练，之后由用户停止。新的 10k 配置尚未进行 GPU 训练，
其完整训练和验证结果待实际运行确认；50 / 25 部署推理延迟与实机成功率也尚未测量。
