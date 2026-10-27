## Purpose

为 UR5 分支的 `lerobot-rollout` 提供注意力/像素敏感度可视化（saliency）能力：支持 `grad_x_input`（像素级敏感度图）与 `attention`（SmolVLA cross-attention 8×8 patch 图与注意力预算条）两种方法，通过快照精确回放产出热力图叠加视频与 rerun 曲线，且任何失败都不得打断策略控制回路。本变更仅移植 SmolVLA 解释器。

## ADDED Requirements

### Requirement: 方法选择与校验

`--saliency.method` SHALL 接受 `grad_x_input` 与 `attention`，默认 MUST 为 `grad_x_input`。非法值 MUST 在配置校验期 fail-fast。两种方法 MUST 支持本地 SmolVLA，沿用「仅 RTC 推理」「display 开启要求 `--display_mode=rerun` 且 `--display_data=true`」的约束。目标动作语义 MUST 为「下一入队动作 a_d」，不承诺与原固定 a₀ 结果逐位一致。

#### Scenario: 默认方法不变
- **WHEN** 只启用 saliency 未指定 method
- **THEN** 解析为 `grad_x_input`，并解释下一入队动作

#### Scenario: 非法方法值被拒绝
- **WHEN** method 为 `foo`
- **THEN** 启动前报错并列出合法值

#### Scenario: attention 在非 RTC 推理下被拒绝
- **WHEN** `method=attention` 与非 RTC 推理同时启用
- **THEN** 启动前报错并说明噪声注入约束

#### Scenario: SmolVLA 两种方法可用
- **WHEN** 本地 SmolVLA 在合法 RTC 与显示配置下选择任一方法
- **THEN** 加载对应解释器，输出每个有效相机的地图

### Requirement: 像素敏感度图语义（grad_x_input）

grad_x_input SHALL 使用与控制模型相同的预处理 batch、noise、生成时 RTC 参数重放；目标 MUST 为队列实际保留的首个策略空间动作 a_d，score 为其实际动作维度的平方 L2，地图为 `(grad * img).abs().sum(channel)`。MUST 保留动作值加权语义，禁止使用后处理电机命令替代目标。MUST 验证包括 guided leftover 在内的完整输入梯度，不得以数值重放一致替代梯度验证。

#### Scenario: 动作值加权语义保持
- **WHEN** 指令幅度接近零
- **THEN** 敏感度相应变小，不因合法零值告警或产生 NaN

#### Scenario: 非零目标索引
- **WHEN** 队列保留的首个动作在原块中索引为 4
- **THEN** score 取 `actions[0,4]`，而非 `actions[0,0]`

#### Scenario: guided RTC 梯度完整性
- **WHEN** 微型模型以非空 leftover 做重放及有限差分比较
- **THEN** 输入梯度在声明的数值容差内一致，截断梯度不得作为完整敏感度验收通过

### Requirement: 注意力截取与隔离

SmolVLA MUST 保留实例级 eager capture，默认 None 且无额外张量复制，softmax 后 cast 前的 float32 probs 仅由解释器副本消费；只捕获 cross 事件。禁止全局注册或给控制模型安装捕获器；停止时 MUST 卸载。每次捕获 MUST 立即选择目标 query、平均 heads 并以 float32 累加，禁止长期保留全矩阵。目标 query MUST 为 d。层和 step 身份 MUST 来自真实模型与采样位置。

#### Scenario: 控制模型不捕获
- **WHEN** 控制模型未安装捕获器
- **THEN** 不触发捕获或解释器张量复制，仅保留原有 None 判断

#### Scenario: 只消费 cross 事件
- **WHEN** SmolVLA 使用 16 层、10 steps、每两层一次 self-attention，选择器默认全选
- **THEN** 消费 80 次 cross 事件，忽略 prefill/self 事件，并每次只消费 query d

#### Scenario: 停止后捕获器卸载
- **WHEN** saliency 会话停止
- **THEN** 解释器副本上的 capture 被卸载，控制模型全程未安装捕获器

### Requirement: Patch 注意力图语义

attention SHALL 对所选真实层 L、去噪步骤 S 的笛卡尔积，先平均各事件 heads，再等权平均事件；默认使用全部合法层与 steps。地图 MUST 保留原始 attention 份额，禁止在图像列内重归一化。相机列与网格 MUST 按实际 prefix 顺序、视觉配置及 token 数校验，含缺失相机占位；SmolVLA 当前为每相机 64 token / 8×8。运行时布局不匹配 MUST 报错并沿用 saliency 故障隔离；letterbox 无有效几何时 MUST 启动前报错。SmolVLA MUST 使用 `resize_imgs_with_padding` 的宽高约定。

#### Scenario: 列归属正确
- **WHEN** 假注意力仅给第二相机全部 patch 列质量
- **THEN** 仅第二相机原始地图非零，网格符合 SmolVLA 的 8×8

#### Scenario: 原始份额保留绝对量
- **WHEN** 全部相机质量合计 0.2
- **THEN** 原始相机地图之和为 0.2，不重新缩放为 1；显示颜色另由 norm_scope 决定

#### Scenario: 布局不匹配时 fail-fast
- **WHEN** key 长度不符合实际相机、文本、状态和动作布局
- **THEN** 明确报错并走旁路降级，不猜测切片也不影响控制

#### Scenario: 单组聚合
- **WHEN** 选择三个合法层和两个 steps
- **THEN** 对六个组合等权平均，每相机输出一张图；预算使用同样六个组合

### Requirement: 注意力预算条

attention SHALL 使用与地图相同的目标 query 与聚合范围，计算各相机及非视觉分类的原始质量。SmolVLA 分类 MUST 为 camera / language / state。预算 MUST 以 float32 归约；float32 路径的测试总和与 1 的舍入误差容差 MUST 为 1e-4，低精度路径以受控舍入测试确定固定阈值，不得自适应放宽以掩盖分类遗漏。MUST NOT 增加默认逐帧总和超差禁用门禁；禁止丢弃分类后重归一化或时间平滑。MUST 将带百分比的条烧录入各相机视频，并将每个实际分类记录到默认可见的 Rerun `saliency/budget/<类别或相机>` 曲线。

#### Scenario: 五份总和为 1
- **WHEN** 三相机 SmolVLA 完成预算计算
- **THEN** 三相机与 language / state 五份合计在 1e-4 内为 1

#### Scenario: 视频自包含
- **WHEN** 播放任一相机 saliency mp4
- **THEN** 顶部可读该策略全部预算分类和百分比，无需 Rerun

### Requirement: Patch 图渲染

Patch 图 SHALL 最近邻块状放大叠加，禁止双线性/高斯模糊，patch 内热力分量分段常数，边界 MUST 有网格。letterbox MUST 按 SmolVLA 真实 resize/padding 几何映射，禁止假设居中或统一 padding；仅渲染 patch 与内容区交集，禁止对内容区重采样 token 网格。黑边 token 不渲染但质量保留于预算。色标 MUST 沿用 run/chunk。画面 MUST 标明实际层/step 选择、mean、动作索引与观测关联信息。

#### Scenario: 块状而非平滑
- **WHEN** 8×8 地图叠加到相机帧
- **THEN** patch 内分段常数，边界清晰且无插值渐变

#### Scenario: letterbox 忠实映射
- **WHEN** SmolVLA 将 1280×720 图像缩至 512×512、顶部补 224 像素
- **THEN** 720p 显示内容的水平网格位于 y=80/240/400/560，底部不多出条带

### Requirement: 精确回放、降级与防护

两方法 MUST 共享输入快照、噪声和生成条件，在测试/离线验收中与控制目标动作进行重放一致性验证；默认运行 MUST NOT 要求逐帧携带目标动作或执行一致性门禁。attention 外层 MUST 使用 no_grad，允许 RTC 原有内部必要梯度上下文。测试 MUST 声明 dtype、设备、compiled/eager 与 atol/rtol；同副本同设备 float32 capture 基准使用绝对误差 1e-5，不得直接套用到低精度/跨设备路径。capture/计算单次失败 MUST 丢帧告警，连续两次 MUST 禁用旁路，禁止打断控制。非有限地图 MUST 丢帧；仅预算非有限 MUST warn-once 并丢弃该帧预算/曲线，保留有效地图。

#### Scenario: 回放与直接前向一致
- **WHEN** 测试中同一副本同设备以相同输入噪声和参数直接前向及重放
- **THEN** float32 capture 结果差异小于 1e-5；该测试不引入运行时逐帧门禁

#### Scenario: capture 失败不中断 rollout
- **WHEN** 捕获器连续两次抛出异常
- **THEN** 两帧丢弃并禁用 saliency，控制继续

#### Scenario: 非有限预算不丢地图
- **WHEN** language / state 等预算非有限而地图有限
- **THEN** 只丢预算条与曲线，保留地图并 warn-once

### Requirement: 下一入队动作与两阶段快照

正常 Policy 模式 SHALL 在推理前复制不可变输入快照，在 chunk 被 RTC 接受且保留非空动作后才提交。MUST 由队列实际裁切结果提供 `first_queued_action_index=d`，与生成前 `inference_delay` 分离；两方法均解释 a_d。快照 MUST 携带 chunk 身份、观测时间、epoch 与遮罩版本。禁止把「入队」等同「最终实际执行」。被拒绝、全部裁去或失效的快照 MUST 正常丢弃，不记连续失败；禁止钳制非法 d 到末动作。

#### Scenario: 估计与实际延迟不同
- **WHEN** 生成条件 delay=2 而实际队列裁去 4 项
- **THEN** 重放仍使用 delay=2，目标索引为 4

#### Scenario: 无有效动作
- **WHEN** chunk 被 trained RTC 拒绝或全部动作被裁去
- **THEN** 不提交 saliency 快照

#### Scenario: reset 后失效
- **WHEN** 快照接受后、发布前 reset epoch 或 View Mask revision 变化
- **THEN** 旧快照不发布，也不增加错误计数

### Requirement: 单组 attention 选择器

系统 SHALL 提供 `attention_layers`、`attention_steps`；None 表示全选，使用真实零基层号和从首次去噪开始的 step 序号。空列表、重复、负数、越界及 SmolVLA 非 cross 层 MUST 启动前拒绝并列合法范围；非 attention 方法配置非默认选择器 MUST 拒绝。MUST 校验每个选中层/step 捕获完整，禁止用未知重复/遗漏事件数改变平均权重。首版 MUST 仅输出单组结果。

#### Scenario: 默认与显式全选一致
- **WHEN** 对同一快照分别省略选择器及列出全部合法索引
- **THEN** 原始地图与预算在数值容差内一致

#### Scenario: 非法层号
- **WHEN** SmolVLA 指定只产生 self-attention 的层
- **THEN** 启动报错并列出可捕获的真实层号

### Requirement: 视角消融下的 saliency 必须复现实际 masked batch

启用 View Ablation 时，saliency snapshot SHALL 在 View Mask processor 之后捕获，并 MUST 携带生成该 batch 的 `view_mask_revision` 与 masked canonical views。explainer MUST 回放策略实际收到的黑图 batch；被 mask 视角的 gradient/attention 结果即使非零也必须保留，禁止人为清零、隐藏或解释为 logical missing-view attention mask。

#### Scenario: 黑图视角产生非零敏感度
- **WHEN** masked 视角的实际黑图输入产生非零 gradient × input 或 attention 质量
- **THEN** saliency map 保留模型真实值，overlay 标记该视角为 MASKED，控制回路与原始相机显示不受影响

#### Scenario: 回放 batch 与执行 batch 一致
- **WHEN** revision N 的 policy action 生成 saliency snapshot
- **THEN** snapshot 中的图像 tensor、mask 集合、revision、noise 与 chunk 参数均对应 revision N 实际执行 batch

### Requirement: saliency 必须丢弃过期 View Mask revision

Mask Transition 后，saliency manager/recorder MUST 在回放和发布前检查 snapshot revision；任何不等于最新可执行 revision 的 snapshot MUST 被丢弃。APPLYING 与 FAILED 期间禁止为未执行动作发布新 saliency。该丢弃属于正常 provenance 过滤，禁止计入 saliency 连续失败降级计数，也禁止影响 policy 控制。

#### Scenario: recorder queue 中的旧 snapshot 被丢弃
- **WHEN** revision N snapshot 已排入异步 recorder，但回放前 View Mask 已切换为 revision N+1
- **THEN** revision N snapshot 不生成 overlay、视频帧或 Rerun saliency 数据，且不增加错误计数

#### Scenario: APPLYING 不展示旧动作解释
- **WHEN** Mask Application State 为 APPLYING 且系统正保持旧 revision 的最后动作
- **THEN** saliency UI 不把旧 snapshot 显示为新 revision 的解释，并明确显示 APPLYING 状态

### Requirement: 训练与默认推理保持原有语义

实现 MUST 保持公共 Policy 推理签名、no_grad 边界、训练 forward/loss、参数可训练性与 state_dict 格式不变。MUST NOT 为解释移除共享 RTC 的 detach 或改变其默认 autograd 设置。完整输入梯度所需调整 MUST 限于解释器专用路径，并通过前向等价与有限差分两项验收；不能满足时 MUST 不将该路径标为支持，不静默回退截断梯度。解释器 MUST 使用独立配置和模型实例，不临时改写控制模型共享配置。队列元数据暴露 MUST 保持既有裁切、锁及接受决策；saliency 关闭时 MUST 不增加副本加载、噪声采样、快照和捕获。

#### Scenario: 共享模型代码训练回归
- **WHEN** 对比任何被修改的共享模型函数与改动前基线，并固定 batch、noise、time 和随机状态
- **THEN** 训练 loss、应训练参数梯度和冻结状态保持一致，state_dict 键不变

#### Scenario: 默认推理回归
- **WHEN** saliency 关闭且输入与噪声固定
- **THEN** 默认推理输出与队列裁切行为保持；不加载副本，也不额外采样、捕获、复制快照

#### Scenario: 副本隔离
- **WHEN** 初始化、运行及关闭解释器
- **THEN** 控制模型的配置、参数可训练性、梯度上下文及捕获器保持原状

### Requirement: 聚合扩展边界

捕获 MUST 提供真实 layer/step 身份与目标 query 的 head 均值，由独立的单组选择/累加逻辑消费；地图和预算 MUST 共享聚合结果。首版 MUST NOT 实现多组输出或保存全部 attention 矩阵。后续多组对比 SHALL 能复用同一事件流，不要求每组重新运行模型。

#### Scenario: 单组模块边界
- **WHEN** 用合成事件测试单组选择与累加
- **THEN** 无需加载模型即可验证筛选和平均；模型捕获逻辑不依赖视频输出或组名
