# WM 阶段 5：多步 Latent Rollout 与误差累积

本阶段属于原定 World Model 路线中的多步 rollout 验证。Stage 14 已确认一步预测存在稳定但较小的
动作信号，因此本阶段冻结两个最优 checkpoint，进一步回答：

> 动作条件优势能否在递归预测中保持？模型误差会以多快的速度累积？

本阶段不更新任何模型参数，不使用 test，也没有接入 LIBERO 闭环控制。

## 评测协议

为了保证 H=1/5/10/25 比较的是同一批样本，只使用能够完整 rollout 25 步的 validation 起点：

| 项目 | 数量 |
| --- | ---: |
| Validation episode | 43 |
| 公共 rollout 起点 | 4,136 |
| Horizon | 1、5、10、25 |
| Test episode 使用数 | 0 |

视觉 latent 采用递归预测：第 `k` 步模型输入的 latent 是第 `k-1` 步的预测，而不是 ground truth。
但 Stage 14 没有训练 state dynamics，因此每一步仍输入对应时刻的 ground-truth
proprioceptive state。本文将其明确称为 **oracle-state latent rollout**，不能描述为完全自主的
世界模型模拟。

评测六种模式：

- `conditioned_correct`：动作模型 + 正确动作序列；
- `conditioned_zero`：动作模型 + normalized zero；
- `conditioned_shuffled`：动作模型 + seed 1517 全局打乱动作；
- `no_action`：no-action checkpoint；
- `persistence`：始终复制 rollout 初始 latent；
- `conditioned_teacher_forced`：每步从 ground-truth 当前 latent 做一步预测，用于分离局部误差与
  递归累积。

## 多步 Raw MSE

所有 horizon 使用相同 4,136 个起点：

| 模式 | H=1 | H=5 | H=10 | H=25 |
| --- | ---: | ---: | ---: | ---: |
| Action + correct | **0.175869** | **0.381760** | **0.582892** | **0.919794** |
| No-action | 0.176878 | 0.394114 | 0.610887 | 0.976970 |
| Action + zero | 0.179462 | 0.421223 | 0.675742 | 1.063338 |
| Action + shuffled | 0.182159 | 0.427979 | 0.676495 | 1.042190 |
| Persistence | 0.226615 | 0.589283 | 0.955799 | 1.498744 |
| Teacher-forced correct | 0.175869 | 0.176977 | 0.176619 | 0.173698 |

动作模型相对 no-action 的优势随 horizon 增大：

| Horizon | 相对 no-action 改善 | 相对 zero 改善 | 相对 shuffled 改善 |
| ---: | ---: | ---: | ---: |
| 1 | 0.570% | 2.002% | 3.453% |
| 5 | 3.135% | 9.369% | 10.799% |
| 10 | 4.583% | 13.740% | 13.836% |
| 25 | 5.852% | 13.499% | 11.744% |

这说明正确动作在较长 rollout 中提供了越来越有用的条件信息。与此同时，所有递归模型的绝对
误差都快速增长，不能只报告相对优势。

## 递归误差累积

Action + correct 相对 teacher-forced 一步预测的 raw MSE 比值：

| Horizon | Recursive / teacher-forced |
| ---: | ---: |
| 1 | 1.00× |
| 5 | 2.16× |
| 10 | 3.30× |
| 25 | 5.30× |

Teacher-forced error 在四个 horizon 上保持约 0.174—0.177，而递归误差到 H25 增至 0.920。
因此当前瓶颈已经从“一步模型是否读取动作”转变为“递归分布漂移和误差累积”。

## Episode-cluster bootstrap

每个 horizon 对 43 个 validation episode 进行 10,000 次 cluster bootstrap：

| Horizon | 动作模型胜出 episode | 95% CI | 正增益概率 |
| ---: | ---: | ---: | ---: |
| 1 | 36/43 | [0.334%, 0.821%] | 100% |
| 5 | 42/43 | [2.576%, 3.717%] | 100% |
| 10 | 42/43 | [3.765%, 5.414%] | 100% |
| 25 | 38/43 | [4.233%, 7.496%] | 100% |

四个区间均为正，支持总体动作优势不是少数长 episode 单独造成。

## 分任务 H25

H25 上 action-conditioned 相对 no-action：

| Task | 改善 |
| ---: | ---: |
| 0 | +1.154% |
| 1 | **-4.031%** |
| 2 | +7.206% |
| 3 | +7.428% |
| 4 | +2.735% |
| 5 | +4.824% |
| 6 | +11.414% |
| 7 | +7.414% |
| 8 | +10.577% |
| 9 | +5.909% |

结果是 9/10 个任务方向为正。Task 1 是明确反例，说明总体多步增益不能被表述为每个任务都稳定
改善，也提示单一全局 dynamics 模型可能在特定场景中发生动作条件误用。

## 本机资源

在 RTX 4060 Laptop GPU（8 GB）、batch size 256 上：

- 完整六模式、四 horizon 评测约 12.69 秒；
- 峰值 PyTorch GPU allocation 为 190,401,024 字节，约 181.6 MiB；
- 没有训练或保存新 checkpoint。

## 复现

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/wm/stage15_multistep_rollout.py
```

公开结果位于
[`results/wm/stage15_multistep_rollout`](../results/wm/stage15_multistep_rollout)：

- `report.json`：协议、聚合指标、配对差异和边界；
- `rollout_metrics.csv`：六种模式在四个 horizon 的完整指标；
- `task_h25.csv`：Task 0—9 的 H25 结果；
- `paired_bootstrap.json`：四个 horizon 的 episode-cluster bootstrap；
- `performance.json`：本地评测时间和显存。

## 结论边界与下一阶段

- 当前使用 oracle ground-truth state，不是视觉和状态都递归预测的完整 WM。
- 评测发生在专家 validation 轨迹上，不包含策略失败后的离分布状态。
- Latent MSE 与 cosine 不是像素质量、机器人成功率或安全性的校准代理。
- 虽然相对 no-action 的优势增长，但 H25 绝对误差仍是 teacher-forced 的 5.30 倍。
- Task 1 的 H25 结果为负，不能宣称动作条件对所有任务都有效。

下一步不应立即声称 WM 能改善 SmolVLA。更合理的是先构造 rollout-error / disagreement 分数，
检查它是否能区分已有闭环成功与失败；由于旧 rollout 没有同步双相机与状态，这一步需要按
Stage 11 的回采索引重新采集一小批带完整观测的闭环轨迹。
