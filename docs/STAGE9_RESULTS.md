# SmolVLA LIBERO：阶段 9 动作专家微调结果

## 结论

Task 5 动作专家正式微调和配对闭环评测均成功完成。训练过程稳定，但微调策略没有改善目标任务，
并让一个原本成功的任务回合退化。

这是模型层面的负结果，不是训练或评测基础设施失败。

## 训练配置

- 初始化：固定版本的 `lerobot/smolvla_libero` checkpoint
- 目标：LIBERO Spatial Task 5，将 ramekin 上的黑色碗放到盘子中
- 数据：Task 5 全部 39 条 demonstration，按 episode 确定性划分
- 训练集：31 个 episode、3,563 帧
- 验证集：8 个不重叠 episode、909 帧
- 可训练模块：动作专家与状态/动作投影层
- 冻结模块：VLM 与视觉编码器
- 可训练参数：99,880,992 / 450,046,176（22.19%）
- Batch size：4
- 训练步数：1,000
- 调度：前 100 步线性 warmup 至 1e-5，随后 cosine decay 至 1e-6
- 精度：BF16 autocast

最初的诊断实验尝试 warmup 到 1e-4，验证 loss 在第 200 步由 0.06634 上升到 0.28306，因此在约
第 300 步停止且没有保存 checkpoint。正式稳定实验使用阶段 8 已验证的 1e-5 峰值学习率。

## 离线指标

| 指标 | 初始/前期 | 最终 | 相对变化 |
| --- | ---: | ---: | ---: |
| 固定验证 loss | 0.06634 | 0.06480 | −2.32% |
| 随机训练 loss 均值 | 前 100 步：0.07722 | 后 100 步：0.06305 | −18.35% |

- 最小随机训练 loss：0.01204
- 1,000 个 loss 和梯度范数全部为有限值
- 训练时间：324.39 秒（5.4 分钟）
- 平均 optimizer step：0.274 秒
- CUDA 峰值已分配/保留显存：2.195/2.248 GB
- 最终策略和 optimizer checkpoint 精确重载，可训练参数哈希完全一致

最终验证 loss 是所有已记录 checkpoint 边界中的最佳值，但相对预训练初始化的改善幅度很小。

## H=10 严格配对闭环评测

微调策略与预训练策略使用完全相同的任务、初始状态、seed、分辨率、pre/postprocessing 和 H=10
执行窗口。

| Task | 预训练 H=10 | 动作专家微调 H=10 | 变化 |
| ---: | ---: | ---: | ---: |
| 5（训练目标） | 0/3 | 0/3 | 0 |
| 4 | 2/3 | 2/3 | 0 |
| 7 | 3/3 | 2/3 | −1 |
| 8 | 2/3 | 2/3 | 0 |
| **总计** | **7/12（58.3%）** | **6/12（50.0%）** | **−1** |

配对状态转移：

- 失败 → 成功：0
- 成功 → 失败：1
- 成功 → 成功：6
- 失败 → 失败：5

Task 7、初始状态 2 从预训练策略 131 步成功，退化为运行到 280 步上限仍失败。视频显示微调策略
能够到达目标区域，但没有完成稳定放置。Task 5 仍为 0/3，全部运行到时间上限。

## 解释

- 更低的离线 flow-matching loss 不能作为闭环控制改善的充分证据。
- 目标 checkpoint 已经使用同一公开 LIBERO 数据集训练，额外的单任务 behavior cloning 主要是在
  已有分布上继续拟合，而不是补充缺失行为。
- 更新完整动作专家会改变相近的碗放置任务；Task 7 的 success-to-failure 是负迁移的直接证据。
- 结果说明必须使用配对仿真评测，并为 replay/多任务正则化提供了动机，而不能仅凭离线 loss 选择
  checkpoint。
- LoRA 对照仍然有意义，因为它可以检验较低容量的 adapter 是否更好地保留 Task 7；但减少可训练
  参数本身不应被预期为自动解决 Task 5。

## 产物审计

- 12/12 个评测视频均可在 360×360 下正常解码，SHA-256 与报告一致
- 4/4 个轨迹归档包含有限动作和预期数组
- 共解码 2,361 个评测视频帧
- 闭环评测时间：733.67 秒（12.2 分钟）

## 复现命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/stage9_task5_finetune.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
uv run --no-sync python scripts/stage5_small_benchmark.py \
  --checkpoint-dir outputs/stage9_task5_finetune/checkpoint_final/pretrained_model \
  --output-dir outputs/stage9_task5_finetune_eval \
  --task-ids 5 4 7 8 \
  --episodes-per-task 3 \
  --n-action-steps 10 \
  --fine-tune-report outputs/stage9_task5_finetune/report.json \
  --comparison-report outputs/stage7_action_horizon_ablation/report.json
```
