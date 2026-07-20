# SmolVLA LIBERO：阶段 8 本地训练链路检查

## 结论

RTX 4060 本地训练链路通过了全部预设检查：

- Task 5 数据筛选与多模态 batch 解码
- 单 batch forward、backward、梯度裁剪与 optimizer update
- 完整策略和 optimizer checkpoint 保存/重载
- 使用固定 8 个训练样本的 200-step tiny-overfit

该阶段证明训练基础设施和本地硬件满足后续实验需求，但不代表闭环 LIBERO 成功率已经提升。

## 数据审计

- 数据集：`lerobot/libero`，本地 snapshot `2e98211ba27db9322efbaa5ed108b34bb03ca163`
- 完整数据：1,693 个 episode、273,465 帧、40 个任务
- 仿真 Task 5 对应数据集 task index 32：
  `pick up the black bowl on the ramekin and place it on the plate`
- Task 5 子集：39 条 demonstration、4,472 帧
- 观测：两个 256×256 RGB 相机和 8 维状态
- 目标：10 Hz 下 50×7 的动作块
- Tiny training probe：从 8 条不同 demonstration 各取一个中点动作块
- 只读 probe：从另外 2 条 demonstration 各取一个中点动作块

全部数据已经存在于本地，无需额外下载。

## 显存友好的训练配置

| 设置 | 数值 |
| --- | --- |
| 总参数 | 450,046,176 |
| 可训练参数 | 99,880,992（22.19%） |
| 动作专家可训练参数 | 98,245,840 |
| 投影层可训练参数 | 1,635,152 |
| VLM 与视觉编码器 | 冻结 |
| Batch size | 1 |
| 精度 | BF16 autocast |
| 学习率 | 1e-5 |
| 训练步数 | 200 |
| CUDA 峰值已分配显存 | 1.80 GB |
| CUDA 峰值保留显存 | 2.03 GB |

8 GB RTX 4060 Laptop GPU 足以运行该动作专家训练配置；验证流水线和开展小规模实验无需租用 GPU。

## 单 batch 与 checkpoint 检查

- 第一步 loss：0.42149
- 裁剪前梯度范数：8.6873
- `model.state_proj.weight` 最大更新量：1.001×10⁻⁵
- 完整策略 checkpoint：906.7 MB
- Optimizer state：412.7 MB，共 155 个参数状态条目
- 重载时间：5.24 秒
- 重载前后可训练参数 SHA-256 完全一致

Checkpoint 包含策略、配置、preprocessor、postprocessor、optimizer state 和训练步数。

## Tiny-overfit 结果

| 指标 | 初始 | 最终 | 变化 |
| --- | ---: | ---: | ---: |
| 8 个训练 probe 的固定平均 loss | 0.05835 | 0.04946 | **−15.24%** |
| 2 个留出 probe 的固定平均 loss | 0.03507 | 0.03560 | +1.50% |
| 前/后 20 步随机训练 loss 均值 | 0.11940 | 0.07425 | −37.81% |

- 最小随机训练 loss：0.00849
- 最终随机训练 loss：0.01798
- 训练循环 wall time：41.84 秒
- 平均 optimizer step：0.160 秒

预设门槛要求 8 个固定训练 probe 的平均 loss 至少下降 10%，实际下降 15.24%，因此通过。两个留出
样本只作为 sanity check，数量远不足以衡量泛化能力。

## 保留的学习率失败实验

首次 100-step 尝试直接使用 checkpoint 的 1e-4 峰值学习率，却没有复现原训练的 1000-step warmup。
单个固定 probe 从 0.02951 上升到 0.24708，因此该运行被标记为
`completed_without_overfit_gate` 并保留，没有隐藏。

修正实验使用 1e-5，接近原线性 warmup 在约第 100 步达到的量级，同时把脆弱的单 probe 判据替换
为全部 8 个固定训练 probe 的平均值。

## 解释

- 本地 forward/backward、optimizer 和 checkpoint 生命周期全部正常。
- 动作专家训练可以轻松放入本地 GPU 显存。
- 即使是小规模适配，学习率 warmup 也很重要。
- Tiny-overfit checkpoint 不是闭环候选模型：它只使用 8 个孤立动作块，200-step 权重没有被冒充为
  正式训练策略。
- 下一步应使用完整 demonstration、episode-level train/validation 划分、warmup scheduler、周期验证
  和 H=10 配对闭环评测进行正式短微调。

## 复现命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_CACHE=outputs/stage8_training_smoke/hf_datasets_cache \
uv run --no-sync python scripts/stage8_training_smoke.py \
  --output-dir outputs/stage8_training_smoke_lr1e5 \
  --steps 200 \
  --learning-rate 1e-5
```
