# 双 UR5 CB3 实机 Rollout

复用本仓库的 `lerobot-rollout`，无需自建策略服务器、通信协议或 VR/IK 推理流程。

跨机器运行 FastWAM 使用官方远程入口，见 [远程推理说明](remote_inference.md)。
该入口的工作站和服务器配置分别是 `remote_rollout.yaml` 与 `remote_fastwam_server.yaml`；
下文 `rollout.yaml` 仍用于本机 SmolVLA 推理。
UR5/Robotiq 部分直接导入采集仓库现有的
`xrobotoolkit_teleop/hardware/interface/universal_robots.py`，不复制驱动、不安装
UR5e 社区插件，也不需要把整个 XRoboToolkit 项目及其依赖装入 LeRobot 环境。
相机和策略执行使用 LeRobot 自带的 RealSense、processor 和 rollout 实现。

默认使用现成的 RTC 后端：控制线程按 20 Hz 消费动作队列、发布最新观测，
`RTCInference` 后台线程同时生成下一个 50 步动作块。新块用旧块的重叠前缀进行
`guided` 衔接，并按推理延迟跳过已经过去的时间步，不等待旧块完全执行完。
该模式使用现有 SmolVLA 权重，不要求重新训练，也不需要独立策略服务器。
配置没有 `n_action_steps` 截断；RTC 通过 `predict_action_chunk()` 生成完整动作块。
相关实现见 `src/lerobot/rollout/inference/rtc.py` 和 `docs/source/rtc.mdx`。

## 安装和检查

在 LeRobot 仓库根目录运行：

```bash
cd /home/carto317/zhj/lerobot
uv sync --locked --extra core_scripts --extra smolvla --extra ur5
uv run --no-sync lerobot-rollout --help
```

`ur5` extra 使用与采集环境一致的 `ur-rtde==1.6.3`。
采集仓库必须仍位于 `robot.teleop_root` 指定的位置。
模型加载还需要 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的缓存；如果本机没有缓存，
联网时先运行：

```bash
uv run --no-sync hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

## 观测和动作约定

适配层严格保持原录制器的排列顺序：

| 下标 | 状态 | 反归一化后的动作 |
| --- | --- | --- |
| `0:6` | 左臂实际关节角，弧度 | 左臂绝对关节目标，弧度 |
| `6:12` | 右臂实际关节角，弧度 | 右臂绝对关节目标，弧度 |
| `12` | 左夹爪开合比例 | 左夹爪目标比例 |
| `13` | 右夹爪开合比例 | 右夹爪目标比例 |

夹爪沿用驱动的开合端点，`0` 为打开、`1` 为关闭。
LeRobot 硬件字段带 `.pos` 后缀，但拼接后的状态和动作仍然是同样的 14 维数值。
三路图像使用 RGB、640×480，名称为 `exterior`、`left_wrist`、`right_wrist`。
配置中的 IP、相机序列号与任务文本来自采集项目，启动前必须现场核验。

## 默认：真实观测和推理，不下发运动

先停止 VR 采集程序，关闭 RealSense Viewer，避免多个进程争用 RTDE 控制或相机。
把两臂和任务物体准备到训练演示的起始场景。

```bash
uv run --no-sync lerobot-rollout --config_path=examples/ur5_real/rollout.yaml
```

默认 `robot.enable_motion=false`：仅使用 RTDE 接收接口和 Robotiq 的查询命令，
不会创建 RTDE 运动控制接口、激活/校准夹爪、下发关节目标或自动回位。
配置启用交互式运行：程序就绪后，在终端输入 `/start` 开始 5 秒推理；
输入 `/stop` 或按 `Ctrl+C` 结束会话。`/start` 之前仍会初始化相机并读取硬件。

`send_action()` 在此模式只校验预测，不发送；`display_data` 显示的已发送动作是实测状态，
不是预测轨迹。交互模式会隐藏 INFO 日志；需要检查预测目标时，可以保持 motion=false，
使用下面的非交互式 5 秒 dry-run，终端约每秒打印一次预测目标（不会发送）：

```bash
uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml --interactive=false
```

dry-run 并不等于已经验证了碰撞、任务成功率或真实执行时序。

需要观察多次异步补队列时，保持只读并延长测试时间：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml \
  --robot.enable_motion=false --robot.enable_gripper=false \
  --duration=20 --interactive=false
```

日志应出现 `Creating inference engine: rtc` 和 `RTC inference thread started`。
检查结尾的 `Cadence summary` 和任何 `RTC inference error`；首次动作块生成前允许
短暂出现 `ticks with no action to send`，但稳定运行后持续断粮需要排查。
主循环接近 20 Hz 并不表示模型每秒完成 20 次推理，也不代表已经验证运动安全。

### dry-run 报关节限位错误

只读模式仍会校验预测动作，超限会结束推理；这不是已经向机器人发送了超限目标。
`Joint target exceeds joint_limit` 表示预测的绝对关节角超出配置的 `±6.283185 rad`；
`Joint target exceeds max_relative_target` 表示预测角度与当前实测角度的差超过配置阈值；
本示例为 `0.5 rad`，适配器类默认仍为 `0.3 rad`。
异常会列出关节名称、`target`、`measured`、`delta`，所有角度单位均为弧度。
只读日志还记录动作序号和实测状态，用于区分首条预测就超限与之后逐渐偏离。
保留完整报错，核验训练演示起始场景、关节角数值表示、动作维度顺序和反归一化，
不要通过放宽阈值、自动裁剪或对角度取模来绕过检查。配置的 `joint_limit` 只是适配层
检查，不替代示教器的安全限制。排查期间保持两个 `enable_*` 开关为 `false`。

只读时机器人不会跟随动作，实际观测保持不动，而 RTC 仍生成和消费未来目标。
后续目标因此可能与静止实测角度相差超过 `max_relative_target`；这不是已经发生的跟踪误差，
也不能仅凭这一点认定模型预测正确或 RTC 有问题。静止观测下长时间推理不是真实闭环。

如需继续测量 RTC 队列与节拍，可显式启用仅用于只读的诊断选项：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml \
  --robot.enable_motion=false --robot.enable_gripper=false \
  --robot.dry_run_report_only=true --duration=20 --interactive=false
```

`dry_run_report_only` 默认关闭，仅把相对偏差超限记录为限频告警，继续测节拍；
不发送或裁剪关节目标。绝对限位、动作/状态格式、非有限数值和相机检查仍会终止运行。
这个选项与 `enable_motion=true` 同时使用会在配置解析时直接报错，不能用于实机跳过保护。
只读模式不使用 `clip_relative_target`，因此仍可检查原始预测是否超过相对偏差。
诊断运行正常结束不表示动作通过安全检查，也不应作为直接开启运动的依据。

## 回到示教起始姿态

Rollout 的“initial position”指程序启动时读取的姿态，不是训练演示的起点；
不要在失败姿态启动 rollout 后用 `/reset` 期待回到示教姿态。
采集仓库的 `teleop_dual_ur5e_hardware.py --reset` 则使用驱动中的硬编码角度，
并调用夹爪激活/校准，不能当成完全无副作用的关节回位。

默认直接复用采集仓库 `universal_robots.py` 中的 `LEFT_INITIAL_JOINT_DEG` 与
`RIGHT_INITIAL_JOINT_DEG`，不需要读取数据集，不复制角度常量。在 LeRobot 仓库中先预览：

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python examples/ur5_real/home.py
```

预览只导入驱动并显示角度，不连接机器人、夹爪或相机。
如果需要特定演示首帧的实测姿态，而不是采集程序的配置起点，可以显式加
`--pose-source=dataset --episode=<编号>`；该模式默认读取 `dual_ur5_demo_coloured`。
其他数据集可用 `--dataset-root` 和 `--repo-id` 指定，不读取模型动作或平均姿态。

先退出 rollout/VR/其他 RTDE 控制程序，确认双臂已启用且静止，移除夹持物体，
在现场核验整个回位路径及双臂/桌面间隙，保持示教器停车按钮可立即使用，再执行：

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python examples/ur5_real/home.py --execute
```

程序读取当前姿态并显示偏差，随后分别要求输入 `MOVE LEFT`、`MOVE RIGHT`。
每臂复用采集驱动的 `return_home()`，按现有 `0.2 rad/s`、`0.3 rad/s²` 参数执行
同步 `moveJ`，先完成一臂、核对关节误差，再让另一臂运动；可用 `--first-arm=right`
调整顺序。配置起点直接由原驱动的度数转换为弧度，数据集起点则保持原始弧度值；
显示时的舍入不影响下发目标。
不使用 VR、IK、策略或相机，也不放宽 rollout 的保护阈值。

默认只恢复双臂，不发送夹爪运动、激活或校准。采集驱动的角度常量不包含夹爪状态。
两侧夹爪已激活且空载时，可以显式使用 `--pose-source=dataset --restore-grippers`
恢复演示第一帧的关节姿态与开合状态；每个夹爪还要单独输入 `GRIPPER LEFT`
或 `GRIPPER RIGHT` 确认。夹爪未激活时直接拒绝执行，不会自动激活它。

`moveJ` 只指定关节终点，不进行避障；顺序回位也不保证不会碰撞。
当前姿态路径不安全时，应先用示教器/自由驱动移到可安全回位的位置，不能盲目执行。
`Ctrl+C` 和 Python 清理不是实机急停保证。回位后重新摆放物体与秤，再启动 rollout。

## 人工监督下启用关节运动

完成现场风险评估，核验双臂/桌面/工具间隙、示教器安全限制、负载/TCP 和急停措施，
确认没有其他控制进程，再显式启用：

```bash
uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml \
  --robot.enable_motion=true
```

此时会建立原驱动的 RTDE 控制连接，但伺服线程在收到第一条策略动作前不发送目标。
适配层不调用 `reset()`、`return_home()`、`moveJ()` 或夹爪 `activate()`。
`/start` 才开始策略动作；默认仍不发送夹爪运动。需要完整任务执行时，先人工确认
夹爪已激活并可以正常使用，再加入 `--robot.enable_gripper=true`。
任何夹爪初始化本身都可能产生动作，不应在夹持物体时未经核验自动执行。

## 时序和软件保护

- 策略动作按训练数据的 20 Hz 调度，50 步对应 2.5 秒的预测时域。
- `queue_threshold=30` 表示剩余不超过 30 步时后台开始推理；在 20 Hz 下约有
  1.5 秒的队列余量。`execution_horizon=30` 是重叠前缀引导范围，不是只执行 30 步。
  该范围必须大于推理期间经过的步数，换块后才仍有用于衔接的引导步。例如本机推理
  约需 12–18 步，设成 10 会使新块中所有受引导的步都被延迟补偿跳过，执行部分
  失去衔接约束。新块就绪后会替换未来动作，因此每块不一定全部执行。
  30 步配置仍需验证稳态不断粮和块间衔接；它不是硬约束，也不保证动作安全。
- 两侧伺服线程复用原驱动的 `servo_joints()`，保持采集时的 17 ms 参数；
  模型推理不直接阻塞这两个线程。它们是普通 Python 线程，不是硬实时控制器。
- 每个伺服周期将关节目标变化限制为 `max_joint_velocity * servo_time`。
  本示例将 `max_joint_velocity` 从 `0.6` 放宽到 `1.0 rad/s`，
  对应每 17 ms 最多变化 `0.017 rad`；目标相对实测位置的窗口从 `0.3` 放宽到
  `0.5 rad`，已发送指令的实际跟踪误差保护仍为 `0.3 rad`。
  适配器类默认仍为 `0.6 rad/s` 和 `0.3 rad`，只使用本示例配置时启用放宽值。
  最近一次 10 Hz × 2 实机日志中，50 ms 目标变化率最大约 `0.953 rad/s`；
  放宽值用于减少正常目标的限幅，不代表更快运动已通过现场安全验证。
  这不是完整示教轨迹的速度上界，快速段仍可能产生积压。
  这是目标变化限速，不是实际关节速度/加速度或碰撞安全保证。
- 关节目标必须有限、在 `±joint_limit` 内。本示例显式设置 `clip_relative_target=true`：
  运动模式复用 LeRobot 的 `ensure_safe_goal_position()`，将超偏差目标裁剪到实测角度
  的 `±max_relative_target` 范围，告警并继续，不因原始策略目标超偏差直接退出。
  裁剪后的参考仍经过伺服线程限速；实际跟踪超过 `max_tracking_error` 时仍停止两臂。
  不启用此配置则保持原来的相对目标超限停机；适配器类默认仍为关闭。
  这改变了原始策略目标越界时的处理方式，不消除目标积压，也不是轨迹或碰撞安全保证。
- 后台推理期间，控制线程继续消费旧块并更新关节目标；底层伺服线程跟踪最新目标。
  若动作队列断粮，底层维持最近目标，超过 `action_timeout_s` 没有新动作则两臂停止。
  默认 1 秒是可配置的软件超时，不是安全认证参数；必须测量推理延迟并验证停车行为，
  不要为绕过超时而盲目增大它。20 Hz 动作调度不是 20 Hz 重新推理，也不保证硬实时。
- 相机读到超过 `camera_max_age_ms` 的旧帧、状态/动作错误、单臂/夹爪故障，都会终止执行。
  故障会锁存，检查现场后退出并重新连接；不会自动恢复保护停止。
- `return_to_initial_position=false` 禁止 LeRobot 退出时自动回程；不要在首轮验证中改为 true。
- 交互式 `/stop` 结束会话并断开硬件。时长结束后停止发送动作，运动线程
  会在超时后停车并锁存错误，此时需要退出并重新启动，不能直接开始下一段。
- 不要把 `/reset` 当作暂停或急停：官方交互命令会显式尝试回到启动位置，
  不受 `return_to_initial_position=false` 这个退出选项约束。本示例只使用 `/start` 和 `/stop`。

这些软件检查不是双臂碰撞检测，也不能处理进程被强制杀死、操作系统挂起或所有网络故障。
现场必须验证机器人控制器侧停止行为，不能只依赖 Python 的清理逻辑。
初次验证从很短的 `duration` 和人工监督开始，未通过验证不得无人值守运行。

## 实机跟踪诊断

交接时触发 `max_relative_target`，不能只凭最后一帧认定模型突跳或机器人跟踪滞后。
在现场确认可进行短程测试、双臂/夹爪及任务物体已复位后，保留原保护阈值，开启
可选 JSONL 记录（默认关闭）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync lerobot-rollout \
  --config_path=examples/ur5_real/rollout.yaml \
  --robot.enable_motion=true --robot.enable_gripper=true \
  --robot.diagnostics_path="outputs/ur5_handoff_$(date +%Y%m%d_%H%M%S).jsonl" \
  --interactive=true --duration=0
```

就绪后输入 `/start`，测试结束输入 `/stop`。接近交接时出现危险或异常动作，使用
现场停车措施，不必等待软件保护；不要在出现危险或异常动作后通过放宽
`max_relative_target` 或提高限速来继续运行。
若再次触发保护，程序停止并退出，文件保留被拒绝的目标和故障原因。
相对目标裁剪只告警并继续；绝对关节目标超限、实际跟踪故障、UR 状态异常和超时
仍会停止。持续裁剪表示策略目标与执行进度不匹配，不应把没有退出当成任务成功。

即使没有启用 JSONL，关节目标超限的报错也包含上一条通过检查的 `previous_target`、
`target_step_delta`、最近成功下发的 `last_sent_command`、目标与该指令的差
`target_minus_last_sent`、该指令与当前实测的差 `last_sent_minus_measured`，以及
该指令的动作序号与距当前的时间 `last_command_age_ms`。尚未下发过指令时显示
`unavailable`，不会以实测位置代替发送记录。该快照是异步采样，不代表完整执行历史。
插值启用时，`target` 与 `previous_target` 是插值后的目标，不一定是模型原始时间步。

日志包含通过检查、可能已裁剪的参考 `target`、实测 `measured`、上一条通过检查的目标
`previous_accepted_target`，以及各臂最近成功调用 `servo_joints()` 时的 `command`、
目标、实测状态、策略动作序号、伺服周期 `period_s` 与驱动调用时长 `servo_call_s`。
`accepted` 只表示通过目标检查，不代表机器人实际到达；伺服快照是异步采样，
`completed_at_s` 可用于识别它是否已经过时。发生关节相对目标裁剪时，`clipped=true`，
`requested_target` 保留裁剪前的请求；未发生裁剪则该字段为 `null`。首行记录当次的
开关、速度和保护参数，包括 `clip_relative_target`。
文件使用独占创建，不覆盖已有日志；磁盘写入失败会停止接受动作。记录发生在控制
线程而非伺服线程，但仍有额外开销，需结合当次节拍报告评估。

## 策略离线诊断

实机动作不理想时，先检查策略在示教画面上的预测，不必连接机器人或相机：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python \
  examples/ur5_real/eval_policy_offline.py
```

默认读取 `./policy/SmolVLA` 与采集仓库的 `dual_ur5_demo_coloured`，使用 checkpoint 保存的
processor、官方 `LeRobotDataset` 和 `predict_action_chunk()`，关闭 RTC 前缀引导以测量
策略自身的预测基线。每条 episode 均匀取五个阶段，每个样本预测完整 50 步；
示教动作不会送入策略输入，统计排除超出 episode 末尾的补齐帧。

报告保存在 `outputs/ur5_offline_eval/<时间>/`：`report.json` 包含不同预测时间步的
关节 MAE/最大误差、夹爪比例误差和“保持当前姿态”的比较基线；`chunks.npz` 保存
预测/示教动作与观测状态；`views_*.jpg` 保存第一条 episode 的三路相机画面。
仅测试少量样本可加 `--episodes 0 --samples-per-episode=2`；其他路径可用
`--policy-path`、`--dataset-root`、`--repo-id` 指定。

离线 MAE 不是任务成功率，也不验证 RTC 衔接、实机跟踪或碰撞安全；
平均误差较小仍可能存在较大的单关节误差。不要据此放宽保护或直接启用运动。
先核验三路相机角色/视角、物体与秤的位置、两臂及夹爪起始状态是否与示教一致。
`enable_gripper=false` 不执行抓取动作，夹爪状态仍作为真实观测输入策略；
不能把这种测试当成完整抓放任务的成功率测试。关节限速是否导致滞后，应通过
短程诊断记录中的策略目标与实测姿态比较，不能仅凭 20 Hz 节拍推断。

## 假设备测试

假设备测试不导入真实驱动，不连接机器人或相机：

```bash
uv run --extra test pytest tests/robots/test_dual_ur5_cb3.py -q
```

当前适配器在 `lerobot-rollout` 入口注册；本示例不改造 VR 采集脚本，也不替换已有数据。
