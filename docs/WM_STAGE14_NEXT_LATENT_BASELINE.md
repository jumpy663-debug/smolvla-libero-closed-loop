# WM 阶段 4：动作条件 Next-Latent Baseline

本阶段首次训练 World Model，但只回答一个受控问题：

> 在预测下一帧冻结视觉 latent 时，提供正确动作是否比不提供动作更好？

为了避免把模型容量、初始化或采样差异误认为动作贡献，action-conditioned 与 no-action 使用完全
相同的架构、初始权重、训练 batch 顺序和优化器设置。Test split 仍然没有参与。

## 数据协议

| 划分 | Episode | 帧 | 一步 transition | 用途 |
| --- | ---: | ---: | ---: | --- |
| Train | 346 | 42,611 | 42,265 | 更新模型参数与计算归一化 |
| Validation | 43 | 5,211 | 5,168 | 选择 checkpoint 和最终评测 |
| Test | 43 | 未提取 | 0 | 完全保留 |

每个 transition 使用：

- 当前双相机视觉 token：`[32, 384]`；
- 当前机器人状态：8 维；
- 当前动作：7 维；
- 目标：下一帧双相机视觉 token。

所有 latent、状态与动作只使用 train 帧计算 mean/std。Episode 最后一帧没有 next-latent，不生成
transition，因此不会跨 episode 边界。

## 模型与严格对照

模型是一个 566,046 参数的 token-wise residual MLP：

1. 将每个 384 维 token 投影到 192 维；
2. 对当前 32 个 token 求全局均值，并与状态、动作拼接成 context；
3. 加入 32 个可学习位置 embedding；
4. 经过 2 个 residual MLP block；
5. 预测 normalized latent delta，再与当前 latent 相加。

输出层权重和 bias 均初始化为 0，因此 step 0 的模型逐 bit 等价于 persistence：

```text
predicted_latent(t+1) = latent(t)
```

两个训练变体：

- `action_conditioned`：输入标准化后的真实动作；
- `no_action`：相同 7 维接口恒为 normalized zero。

两者共享：

- 初始化 SHA-256：
  `ce5211a7325246bafb66875dd54420a8e992e99862fb46b2219835377f0ebbab`
- batch schedule SHA-256：
  `c41ac282bee7d002046695aea1f257a3d715602831942cc0f3b8dc85e0922ce9`
- seed 1400、batch size 128、学习率 `1e-3`、weight decay `1e-4`；
- 5,000 step、BF16 autocast 训练、FP32 validation；
- 每 250 step 完整评测，以最低 validation normalized MSE 选择 checkpoint。

两个变体的最优 checkpoint 都出现在 step 4,500，随后 validation 没有继续改善。

## 一步预测结果

| 模型 | Validation raw MSE | 相对 persistence 改善 | Token cosine |
| --- | ---: | ---: | ---: |
| Persistence | 0.219152 | — | — |
| No-action | 0.172660 | 21.215% | 0.973576 |
| Action-conditioned | **0.171457** | **21.764%** | **0.973746** |

严格配对下，action-conditioned 相对 no-action 的 raw MSE 改善为 **0.697%**。在以
persistence raw MSE 定义的 validation 高动态 top 25% transition 上，改善为 **0.530%**。

按 LIBERO Spatial 任务分别计算，action-conditioned 在 **10/10** 个任务上 raw MSE 都低于
no-action，但幅度均较小。

## 动作反事实

对同一个 action-conditioned checkpoint 进行三种输入评测：

| 动作输入 | Validation raw MSE | 相对正确动作退化 |
| --- | ---: | ---: |
| 正确动作 | **0.171457** | — |
| Normalized zero | 0.174886 | 1.961% |
| 全局随机打乱动作 | 0.177620 | 3.470% |

打乱使用固定 seed 1417，在全部 5,168 个 validation transition 上做全局随机排列，而不是只把
动作移动到相邻帧。正确动作优于 no-action、zero 和 shuffled，说明模型确实利用了动作信息，
而不仅是从当前视觉与状态预测平均运动。

## Episode-cluster bootstrap

由于 0.697% 是小幅增益，不能只看全局平均值。本阶段进一步以 episode 为独立 cluster，对
43 个 validation episode 进行 10,000 次有放回重采样：

| 指标 | 结果 |
| --- | ---: |
| 动作模型更好的 episode | 36/43 |
| 动作模型更差的 episode | 7/43 |
| 平局 | 0 |
| Float64 聚合改善 | 0.696% |
| 95% cluster-bootstrap CI | **[0.484%, 0.916%]** |
| Bootstrap 正增益概率 | 100% |

Bootstrap seed 为 1423，指标在每次重采样中仍按帧数和 token 数加权。该区间支持“在这组
validation episode 上存在稳定但较小的一步动作贡献”，不支持夸大为显著的长期控制提升。

## 本机资源

RTX 4060 Laptop GPU（8 GB）上的本地测量：

| 训练变体 | 时间 | 峰值 PyTorch GPU allocation |
| --- | ---: | ---: |
| Action-conditioned | 50.08 秒 | 241,809,408 字节 |
| No-action | 50.32 秒 | 241,809,408 字节 |

每个最优 checkpoint 约 2.16 MiB，加载后完整 validation 指标与保存前完全一致。时间仅为本机
测量，不是跨机器 benchmark。

## 复现

训练两个严格配对模型：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/wm/stage14_next_latent_baseline.py
```

运行 episode-cluster bootstrap：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/wm/stage14_paired_bootstrap.py
```

公开结果位于
[`results/wm/stage14_next_latent_baseline`](../results/wm/stage14_next_latent_baseline)：

- `report.json`：训练配置、一步指标、分任务结果和动作反事实；
- `training_curve.csv`：两个模型的配对训练/验证曲线；
- `normalization.json`：只由 train 帧计算的归一化参数；
- `paired_bootstrap.json`：episode-cluster bootstrap 方法、哈希和置信区间；
- `performance.json`：本地训练时间与显存。

两个 checkpoint 位于 `outputs/wm/stage14_next_latent_baseline/`，受 `.gitignore` 排除。

## 结论边界与下一阶段

- 结果证明的是冻结 latent 的**一步**预测利用了动作，不是像素重建质量或闭环任务成功率。
- 0.697% 仍是小幅改善；强视觉连续性让 no-action/persistence 本身已经很强。
- Validation 上的 cluster-bootstrap 区间不能替代新的 test split 或不同随机种子复现。
- 当前模型只在专家轨迹上训练，尚未接触策略失败后的离分布状态。
- 尚未测试误差是否会在 5、10、25 步递归 rollout 中累积。

动作信号通过了预设的三项判据，因此下一阶段可以进入 multi-step latent rollout：冻结当前
checkpoint，对比正确动作、zero、shuffled 和 no-action 的 1/5/10/25 步误差增长曲线，再决定
是否值得把 WM 接入策略评估或失败预警。
