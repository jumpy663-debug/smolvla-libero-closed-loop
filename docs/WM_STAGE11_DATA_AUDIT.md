# WM 阶段 0—1：基线冻结、数据审计与确定性划分

本阶段只建立 World Model（WM）扩展的数据基础，不提取视觉特征，也不训练模型。原有
SmolVLA × LIBERO 闭环结果冻结为 `v0.1.0`，后续开发在
`feature/latent-world-model` 分支进行。

## 阶段结论

- `lerobot/libero` 固定到 revision
  `2e98211ba27db9322efbaa5ed108b34bb03ca163`。
- LIBERO Spatial 共审计 10 个任务、432 个专家 episode、52,970 帧。
- 按任务分别进行 episode 级确定性 80/10/10 划分，最终得到
  346/43/43 个训练/验证/测试 episode，不存在 episode 交叉。
- 现有 78 个闭环 rollout 均有动作、奖励和渲染视频，但没有同步保存双相机策略观测与
  proprioception，因此不进入 WM-v1 训练集。
- 阶段 2 可以直接从专家 demonstrations 开始表征提取；闭环 rollout 的重新采集推迟到
  表征方案确定之后，避免先生成一批格式不合适的数据。

## 专家数据审计

| 划分 | Episode 数 | 帧数 | H=10 窗口 | H=25 窗口 | H=50 窗口 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 346 | 42,611 | 39,151 | 33,961 | 25,311 |
| 验证 | 43 | 5,211 | 4,781 | 4,136 | 3,061 |
| 测试 | 43 | 5,148 | 4,718 | 4,073 | 2,998 |
| 合计 | 432 | 52,970 | 48,650 | 42,170 | 31,370 |

拆分使用 `seed=1000`，对每个任务内的 episode ID 计算 SHA-256 排名，再按
80/10/10 分配。这样既保持每个任务都出现在三个集合中，也不依赖不同 NumPy 版本的随机数
实现。测试集在 WM 架构和超参数冻结前不参与选择。

LIBERO benchmark task ID 与 LeRobot dataset task index 的映射为：

| Benchmark task | Dataset task | Episode 数 | 帧数 | 训练/验证/测试 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 34 | 45 | 4,487 | 37/4/4 |
| 1 | 37 | 45 | 5,940 | 37/4/4 |
| 2 | 38 | 46 | 5,343 | 36/5/5 |
| 3 | 35 | 43 | 4,287 | 35/4/4 |
| 4 | 31 | 42 | 6,257 | 34/4/4 |
| 5 | 32 | 39 | 4,472 | 31/4/4 |
| 6 | 30 | 46 | 5,775 | 36/5/5 |
| 7 | 33 | 35 | 4,747 | 27/4/4 |
| 8 | 36 | 47 | 5,570 | 37/5/5 |
| 9 | 39 | 44 | 6,092 | 36/4/4 |

### 数据分片兼容性发现

本地固定 revision 中共有 377 个 Parquet 分片和 74 个视频文件，分别占 20,264,464 字节和
1,915,134,134 字节。逐分片读取后确认其覆盖全部 1,693 个 episode 和 273,465 帧。

但 `meta/episodes` 中的 `data/file_index` 只覆盖 0—68；与真实分片内容比对时，
1,690/1,693 个 episode 的索引不一致。因此 manifest 没有直接信任该字段，而是扫描每个
Parquet 分片中的 `episode_index` 建立真实映射。帧数、首尾 `frame_index`、单 episode
任务一致性和文件存在性均已校验。

## 现有闭环 rollout 审计

| 策略与执行窗口 | Episode 数 | 成功数 | WM-v1 可用 |
| --- | ---: | ---: | --- |
| 预训练，H=50 | 30 | 18 | 否 |
| 预训练，H=25 | 12 | 6 | 否 |
| 预训练，H=10 | 12 | 7 | 否 |
| 动作专家微调，H=10 | 12 | 6 | 否 |
| LoRA rank-16，H=10 | 12 | 7 | 否 |

这些产物中的 NPZ 只包含 `actions`、`rewards`、`successes`、`dones` 和
`done_indices`；MP4 是环境渲染画面，不等价于带时间同步的双相机策略输入。将其直接与动作
拼接训练会造成模态和时序含义不一致，所以本阶段只保留它们的路径、哈希与确定性回采索引。

## 阶段 2 的资源预算

若先采用双相机、每相机 16 个 token、384 维、FP16 的冻结视觉特征，每帧约需 24,576 字节：

- 全部 Spatial demonstrations：约 1.21 GiB；
- Task 4/5/7/8 困难子集：约 493 MiB。

这只计算视觉 token，不含状态、动作、索引和存储格式开销，但说明 8 GB 显卡和本地磁盘足以
先完成小规模表征 pilot，不需要在当前阶段租卡。

## 复现

从仓库根目录运行：

```bash
uv run --no-sync python scripts/wm/stage11_build_dataset_manifest.py \
  --rollout-root /path/to/outputs/reproduction/smolvla_libero \
  --split-seed 1000
```

脚本默认拒绝覆盖内容不同的已有结果。只有在明确重新生成 Stage 11 产物时才使用
`--overwrite`。

公开结果位于 [`results/wm/stage11_data_audit`](../results/wm/stage11_data_audit)，包括：

- `expert_episodes.csv`：专家 episode、任务、真实数据分片、双相机视频区间和有效窗口数；
- `rollout_episodes.csv`：既有 rollout 的配置、成功标签、产物哈希和模态可用性；
- `splits.json`：确定性划分和 seed；
- `report.json`：完整审计摘要、库存哈希、空间预算与阶段 2 readiness。

## 结论边界

- Expert metadata 没有暴露 LIBERO simulator initial-state ID，因此无法证明 demonstrations 与
  闭环 benchmark 初始状态不存在语义重合；当前保证的是严格 episode 级不泄漏。
- 本阶段没有任何 WM 训练结果，不把数据审计描述为模型性能提升。
- 现有 rollout 仍可用于只看渲染画面与动作的探索实验，但这类实验必须与 WM-v1 的双相机、
  proprioception 设定分开报告。
