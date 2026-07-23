# WM 阶段 3：全量连续特征缓存

本阶段把 Stage 12 的冻结 DINOv2 双相机表征从 80 个稀疏样本扩展到完整
train/validation demonstrations，并建立可中断恢复、逐 episode 审计的特征缓存。仍未训练
World Model，test split 继续完全保留。

## 最终数据规模

| 划分 | Episode | 连续帧 | 双相机图像 | 缓存字节 |
| --- | ---: | ---: | ---: | ---: |
| Train | 346 | 42,611 | 85,222 | 1,051,044,243 |
| Validation | 43 | 5,211 | 10,422 | 128,535,683 |
| 合计 | 389 | 47,822 | 95,644 | 1,179,579,926 |

缓存约 1.10 GiB，共 389 个 Safetensors shard，每个 episode 对应一个文件。Stage 11 中的
43 个 test episode 没有被解码或缓存，`test_split_cached_episodes=0`。

## 分片格式

每个 episode shard 包含：

| Tensor | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `visual_tokens` | `[L, 2, 16, 384]` | FP16 | 两路相机的冻结 DINOv2 空间 token |
| `observation_state` | `[L, 8]` | FP32 | 机器人状态 |
| `action` | `[L, 7]` | FP32 | 专家动作 |
| `dataset_index` | `[L]` | Int64 | 原始 LeRobot 全局行索引 |
| `frame_index` | `[L]` | Int64 | Episode 内连续帧索引 |
| `timestamp` | `[L]` | FP32 | Episode 内时间戳 |
| `is_terminal` | `[L]` | Bool | Episode 最后一帧标记 |

Shard metadata 记录 dataset/model revision、split、task、episode、长度、语言指令和双相机解码
像素 SHA-256。这样后续构造 next-latent 样本时不会跨 episode 边界，也能追溯每个 latent 的
原始观测与动作。

## 断点恢复

写入流程采用：

1. 解码一个完整 episode 的两路视频；
2. 按固定 batch size 编码；
3. 先写入临时 Safetensors；
4. 原子替换为正式 shard；
5. 立即重载并验证 keys、shape、dtype、finite、连续帧和 terminal mask；
6. 更新 `progress.json`。

若进程中断，重新运行时会逐个验证已有 shard，只对缺失项重新解码和编码。全量提取完成后的
第二次完整复跑得到：

```text
new=0, resumed=389
```

389 个 shard 全部通过验证并跳过，resume 复跑总耗时约 9.12 秒。

## 数值一致性处理

初次尝试期间 Windows GPU 驱动发生 device-lost。重启后对同一 episode 比较 batch 64 与
batch 32 的输出，虽然差异很小，但并非逐 bit 一致：

- 不同 FP16 元素：14,418；
- 最大绝对差：0.0078125；
- 平均绝对差：约 0.00000218。

DINOv2 不含依赖 batch 统计量的 BatchNorm，但不同 batch shape 仍可能选择不同 CUDA kernel，
产生舍入差异。因此没有混用两组缓存，而是保留原分片备份后，以 batch 32 统一重建全部
389 个 episode。最终确定性重算也使用 batch 32。

## 验证结果

脚本固定 `recheck_seed=1300`，重新解码并编码：

- Train episode 1445；
- Validation episode 1516。

两者的解码像素 SHA-256 和全部缓存 tensor 均逐 bit 一致。

之后又运行独立审计，逐 shard 完成：

- 389/389 文件存在、文件大小和 SHA-256 匹配；
- 47,822/47,822 帧的 state、action、timestamp、dataset index 与原始 Parquet 逐元素一致；
- 全部视觉 token、状态和动作都是有限值；
- 每个 episode 的 `frame_index` 从 0 连续递增；
- 每个 shard 恰有一个 terminal，且位于最后一帧；
- Train/validation task 0—9 全覆盖；
- 43 个 test episode 与缓存 episode 集合完全不相交。

特征分布摘要：

| 指标 | 数值 |
| --- | ---: |
| Token 均值 | 0.064766 |
| Token 标准差 | 1.806306 |
| 跨帧平均 channel 标准差 | 0.430078 |
| 双相机 embedding cosine 均值 | 0.476468 |
| 双相机 cosine 范围 | 0.199293—0.794309 |

这些统计用于检查数值异常和表征塌缩，不代表任务性能。

## 本机性能

统一缓存使用 RTX 4060 Laptop GPU（8 GB）、batch size 32：

| 指标 | 本地测量 |
| --- | ---: |
| 完整提取总时间 | 703.70 秒 |
| 视频解码时间 | 105.00 秒 |
| 冻结编码时间 | 587.32 秒 |
| PyTorch 峰值 GPU allocation | 306,543,104 字节（约 292.3 MiB） |
| 完整 resume 验证 | 9.12 秒 |

时间只是本机单次测量，不作为跨机器 benchmark。结果进一步说明当前显卡足够完成 WM 数据准备，
无需租用云 GPU。

## 复现

从仓库根目录运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/wm/stage13_full_feature_cache.py \
  --batch-size 32
```

实际特征保存在：

```text
outputs/wm/stage13_full_feature_cache/shards/
```

`outputs/` 受 `.gitignore` 排除。公开审计产物位于
[`results/wm/stage13_full_feature_cache`](../results/wm/stage13_full_feature_cache)：

- `shards.csv`：389 个 shard 的相对路径、任务、长度、字节数、特征哈希与像素哈希；
- `report.json`：数据规模、tensor schema、库存哈希、特征统计和确定性重算结果；
- `performance.json`：首次完整提取和 resume 复跑的本地性能记录。

## 结论边界与下一阶段

- 当前缓存只包含冻结通用视觉表征，没有证明其保留所有精细接触与夹爪状态信息。
- 全量缓存仍是专家 demonstration 分布，不包含策略失败后的离分布状态。
- Test split 目前没有特征缓存；在模型结构和超参数冻结前不应提取或使用。
- 本阶段没有 next-latent loss、rollout prediction 或闭环提升结果。

下一阶段可以训练最小 action-conditioned next-latent baseline：只用 train shard 更新参数，只用
validation shard 选择 checkpoint，并先回答“动作条件是否优于无动作预测”这个最基本的问题。
