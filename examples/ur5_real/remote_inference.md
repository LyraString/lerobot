# 双臂 UR5：官方远程 FastWAM 推理

本分支移植官方 `lerobot-rollout --inference.type=remote` 和
`lerobot-policy-server`，通信、会话管理和动作队列使用官方 Zenoh 实现。
工作站采集观测并执行动作；服务器加载 checkpoint 及其 processor，返回反归一化动作块。
工作站只需要能 SSH 到服务器，服务器不需要反向 SSH 到工作站。

## 当前两台机器的启动命令

工作站：`/home/carto317/zhj/lerobot`。服务器：`zhj@10.13.21.170`，
仓库是 `/data0/zhj/lerobot`。两端使用 `feat/official-remote-inference`。
服务器 YAML 已填写 `fastwam_base_dual_ur5_full_finetune_text_cache_bs4`
的 `003574/pretrained_model`；已核对数据集 20 Hz、三路相机和左臂/右臂/夹爪的 14 维顺序。

终端一登录服务器，选择当时空闲的 GPU，然后启动模型服务：

```bash
ssh zhj@10.13.21.170
cd /data0/zhj/lerobot
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync lerobot-policy-server \
  --config_path=examples/ur5_real/remote_fastwam_server.yaml
```

等日志出现 `Policy server ready`。终端二在工作站保持 SSH 隧道：

```bash
ssh -N \
  -L 127.0.0.1:17447:127.0.0.1:7447 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  zhj@10.13.21.170
```

终端三在工作站运行只读验证：

```bash
cd /home/carto317/zhj/lerobot
uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/remote_rollout.yaml \
  --robot.diagnostics_path="outputs/ur5_remote_$(date +%Y%m%d_%H%M%S).jsonl"
```

出现交互提示后输入 `/start`。默认采集并请求推理 5 秒，不下发运动或夹爪命令。
真实观测、动作顺序及耗时检查通过后，再启动运动：

```bash
uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/remote_rollout.yaml \
  --robot.enable_motion=true --robot.enable_gripper=true \
  --interactive=true --duration=0 \
  --robot.diagnostics_path="outputs/ur5_remote_$(date +%Y%m%d_%H%M%S).jsonl"
```

输入 `/start` 开始，`/stop` 结束当前会话。此模型的文本缓存仅包含
`Pick up the object and place it on the scales`；服务端 `execution.warmup_task`
和客户端 `task` 均使用这个任务。改成其他文字前需要相应的文本嵌入。

2026-10-26 的真实权重/模拟观测验证已通过服务端预热与 SSH 隧道。
两次 JPEG 请求往返为 1.19 秒和 0.97 秒，一次 raw 请求为 1.44 秒；
均返回有限的 `[10, 14]` 动作并通过客户端校验。这里使用三路随机图像和零状态，
不涉及机器人，也不验证任务成功率。单次策略计算约 0.64–0.75 秒。
当前 10 步只覆盖 0.5 秒，短于这次测得的推理往返时间，连续执行会等待新动作；
20 Hz 是动作播放频率，不是模型推理频率。
先完成只读观测验证，再根据实测耗时决定执行片段、合并方式和模型加速设置；
仅加大 `n_action_steps` 不能据此保证连续执行，仍须检查动作年龄和新旧块的衔接。

服务器安装通信依赖时保留了原有 Torch 等训练依赖。切换分支前的训练草稿完整保存在
`refs/codex/fastwam-training-wip-20261026`，包括当时未跟踪的训练配置和文本缓存生成脚本。
将来需要恢复训练现场时，先停止推理服务，确认当前工作区干净，再运行：

```bash
cd /data0/zhj/lerobot
git switch feat/fastwam-libero
git stash apply refs/codex/fastwam-training-wip-20261026
```

## 移植范围

- 官方 `ae2b57f12223c8c5184fcd814eb63cd9ce7de203`：PR #4937，包含推理模块迁移、
  远程入口、测试和旧 `async_inference` 的移除。它的早期版本 #4836 曾被回退，
  不要再移植那个提交及其回退提交。
- 官方 `61373ee1`：PR #4965，更新远程部署文档。
- 解决当前分支与上游的冲突；补入上游 relative-action processor 的必要依赖。
  不引入当前分支没有的 LaWAM/DM05 等无关策略。
- 按本次选择使用官方控制循环，移除旧 rollout 的显著性/视角屏蔽扩展及其测试。
  UR5 驱动、单位约定、伺服保护和诊断记录继续作为硬件适配层使用。
- 增加 FastWAM 的动作块声明：预测长度来自 `action_horizon`，执行片段来自
  `n_action_steps`；未来视频帧索引仅用于训练，在线只传当前观测。仅声明 `chunk` 模式。
- FastWAM 保留模型内的图像缩放，校验相机名称和状态维度；不改变网络协议或模型计算。
- UR5 声明支持保持位置，供官方客户端在等待动作时刷新最近下发的目标。
- 从服务器训练代码移入已保存的文本缓存 processor 及配置兼容项；补入上游
  Adam/AdamW 的 `fused` 字段，保留 checkpoint 的 `optimizer_fused` 设置。
  服务端支持显式 `execution.warmup_task`，让预热使用 checkpoint 已缓存的真实任务。
- 缓存嵌入存在时，不再从同一批次的任务元数据生成另一份 prompt，避免真实核心
  拒绝同时提供 prompt 和 context；显式传入二者仍由模型做互斥校验。

上游后续的外部 VLM、G1 SONIC 和全仓类型检查改动不属于这个双臂 FastWAM 移植。
两台机器应使用本分支同一个经过验证的提交，不要混用旧 gRPC 服务和新 Zenoh 客户端。

## 校外服务器

在服务器上的本分支仓库根目录安装依赖：

```bash
uv sync --locked --extra remote --extra fastwam --extra dataset
```

检查 `pretrained_model` 包含模型配置、权重、保存的 pre/postprocessor 和归一化统计；
冻结的 Wan VAE、文本编码器和 tokenizer 所需缓存也必须可用。
不要用 LIBERO 的动作维度、夹爪翻转设置或统计量替代实机训练配置。

`remote_fastwam_server.yaml` 已与 gpu57 的 003574 checkpoint 配置和数据集元数据核对；
切换其他模型时重新核对训练时的相机名称、状态/动作顺序、
`action_dim`、`proprio_dim`、`action_horizon`、`n_action_steps` 和采样频率。
顺序为左臂六关节、右臂六关节、左夹爪、右夹爪；关节用弧度，夹爪 0=开、1=闭。
FastWAM 在模型内把每张原始 RGB 图缩放到 checkpoint 配置的每相机尺寸后拼接。

启动时覆盖服务器本地模型路径：

```bash
uv run --no-sync lerobot-policy-server \
  --config_path=examples/ur5_real/remote_fastwam_server.yaml \
  --model.repo_or_path=/实际模型目录/pretrained_model
```

已有训练环境可以只补装通信依赖和刷新 editable 安装，避免重新同步整套 GPU 依赖：

```bash
uv pip install --python .venv/bin/python --no-deps -e . \
  'eclipse-zenoh==1.10.1' 'msgpack==1.2.3'
```

离线环境仅在所需组件缓存齐全时增加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
先等服务器完成模型加载、processor 校验和预热，出现 `Policy server ready`，再启动客户端。
服务仅绑定服务器的 `127.0.0.1:7447`。

## 工作站

在工作站的本分支仓库根目录安装客户端及设备依赖：

```bash
uv sync --locked --extra remote --extra core_scripts --extra ur5
```

终端一建立隧道，并保持运行；替换 SSH 用户名、地址及端口：

```bash
ssh -N \
  -L 127.0.0.1:17447:127.0.0.1:7447 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  用户名@服务器地址
```

SSH 非默认端口时增加 `-p 端口号`。SSH 服务需允许 TCP 端口转发。

核对工作站配置里的 UR5 IP、相机序列号和 `teleop_root`。
终端二先运行默认只读配置：

```bash
uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/remote_rollout.yaml \
  --robot.diagnostics_path="outputs/ur5_remote_$(date +%Y%m%d_%H%M%S).jsonl"
```

就绪后输入 `/start`，默认验证 5 秒；`/stop` 结束会话。
`enable_motion=false`、`enable_gripper=false`，仍会连接硬件、读相机并校验模型动作。
客户端没有 `policy.path`：旧 `rollout.yaml` 含本地模型与 RTC 配置，不能直接混用。
`deployment`、`semantics` 和 20 Hz 执行频率必须与服务端一致。

## 运行验收

先确认真实 checkpoint 能通过服务端预热，再用现场只读观测检查状态/动作单位、
顺序、相机和反归一化。记录上传、推理和下载的总耗时及波动。
`refill_seconds=0.75`、JPEG quality 90、最大观测年龄 2 秒只是起始配置；
需根据训练设置、图像质量和实际网络测量调整，不能据此认定满足 20 Hz 闭环。

当前执行余量应覆盖一次请求总耗时及其波动。缩短执行片段会减少延迟余量；
动作队列空、过期数据、失联及模式切换按官方等待/失效逻辑处理。
保持位置会刷新最近应用的关节与夹爪目标，它不表示物理立即停止。
UR5 原有动作超时与故障锁存不做放宽；远程保持位置也不代替控制器侧停止机制。
现场验证通过后才使用已有运动/夹爪开关；`/reset` 仍会尝试回到程序启动位置。

## 软件验证范围

官方 inference/remote/rollout 测试覆盖独立进程、Zenoh、本地动作队列、失联、超时与旧动作拒绝。
新增 FastWAM 测试保存并重新加载 checkpoint 和真实 processor，比较 `select_action`
与远程 runner 的动作，并实际通过 Zenoh 验证 raw/JPEG 两种传输。
这些测试用小型替代网络避免加载数十 GB 权重，不能视为真实 FastWAM 权重、校外链路或实机验收。
新增 UR5 测试以模拟驱动验证官方保持位置逻辑；不会连接机器人。

本次离线回归：605 passed、2 skipped；跳过的是需要额外 `zenohd` 二进制的
router/mTLS 测试。直接 TCP/Zenoh、raw/JPEG、独立进程和 FastWAM 处理管线测试均已执行。
`uv lock --check`、相关 Ruff 检查及格式检查通过；两个官方入口的 `--help` 可正常运行。

真实 checkpoint 配置和完整 processor 文件另行复制到本地做 CPU 验证：
14 维状态、`[1, 128, 4096]` 文本嵌入和 14 维反归一化动作均可加载并产生有限值。
文本缓存兼容和任务预热补丁的回归为 316 passed、2 skipped，覆盖
`tests/inference`、`tests/remote_inference`、FastWAM 策略和优化器测试。
真实预热发现的缓存条件冲突修复后，FastWAM 相关 15 项测试再次通过；
gpu57 的 003574 权重完成两次预热，并通过上述 JPEG/raw 隧道验证。
测试服务及临时 SSH 隧道在验证后关闭，使用时按前面的三个终端命令重新启动。

参考：[官方远程推理文档](https://huggingface.co/docs/lerobot/main/en/remote_inference)。
