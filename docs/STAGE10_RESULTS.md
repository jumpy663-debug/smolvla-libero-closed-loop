# SmolVLA LIBERO：阶段 10 LoRA 严格对照结果

## 结论

在与阶段 9 动作专家微调完全一致的控制条件下，完成了 rank-16 LoRA 训练和闭环评测。LoRA 的
闭环成功数为 7/12，与预训练策略相同，并保留全部 7 条配对成功轨迹；动作专家微调则因 Task 7
丢失一条成功而降至 6/12。

LoRA 没有解决 Task 5：目标任务仍为 0/3。因此本实验支持的是“更低适配成本和更少遗忘”，不能
表述为“目标任务能力得到提升”。

## 严格控制条件

LoRA 与动作专家微调使用相同的：

- 固定版本 `lerobot/smolvla_libero` 初始化和本地 SmolVLM backbone
- 39 条 Task 5 demonstration，以及确定性的 31/8 episode 划分
- 3,563 个训练帧和 909 个不重叠验证帧
- seed 1000、batch size 4、1,000 个 optimizer step、BF16 autocast
- 前 100 步 warmup 至 1e-5，随后 cosine decay 至 1e-6
- 8 个固定验证 probe，包括完全相同的噪声和 flow time
- H=10 下相同的仿真任务、初始状态、seed 和预处理

Stage 10 脚本会在训练前读取 Stage 9 报告并核对这些控制变量，任何一项不同都会拒绝运行。

## LoRA 配置

- 方法：LoRA，rank 16，alpha 16，dropout 0
- 动作专家 q/v adapter：645,120 个参数
- 状态/动作/时间投影 adapter：97,536 个参数
- 非预期可训练参数：0
- 可训练参数总计：742,656 / 450,788,832（0.165%）
- Adapter 产物：3.00 MB
- Optimizer 产物：6.00 MB

视觉编码器、VLM、基础动作专家和基础投影层全部保持冻结。

## 训练效率对比

| 指标 | 动作专家微调 | LoRA | LoRA 变化 |
| --- | ---: | ---: | ---: |
| 可训练参数 | 99,880,992 | 742,656 | 少 134.5×（−99.26%） |
| 可训练参数占比 | 22.19% | 0.165% | −22.03 个百分点 |
| 固定验证 loss | 0.066339→0.064797 | 0.066339→0.064686 | −2.32% 对比 −2.49% |
| 最佳固定验证 loss | 第 1000 步 0.064797 | 第 500 步 0.064298 | — |
| 训练时间 | 324.39 秒 | 279.24 秒 | −13.92% |
| CUDA 峰值已分配显存 | 2.195 GB | 1.737 GB | −20.85% |
| 模型/adapter 产物 | 906.71 MB | 3.00 MB | 小 302.0× |
| Optimizer 产物 | 412.75 MB | 6.00 MB | 小 68.7× |

全部 1,000 个 LoRA loss 和梯度范数均为有限值。Adapter 与 optimizer 成功重载；重载前后可训练
参数 SHA-256 和 8 个固定验证 loss 完全一致。

## H=10 严格配对闭环评测

| Task | 预训练 | 动作专家微调 | LoRA |
| ---: | ---: | ---: | ---: |
| 5（训练目标） | 0/3 | 0/3 | 0/3 |
| 4 | 2/3 | 2/3 | 2/3 |
| 7 | 3/3 | 2/3 | 3/3 |
| 8 | 2/3 | 2/3 | 2/3 |
| **总计** | **7/12（58.3%）** | **6/12（50.0%）** | **7/12（58.3%）** |

LoRA 相对预训练策略的配对状态转移：

- 失败 → 成功：0
- 成功 → 失败：0
- 成功 → 成功：7
- 失败 → 失败：5

LoRA 的回合步数分别为：Task 5 `[280, 280, 280]`、Task 4 `[126, 280, 131]`、Task 7
`[114, 115, 126]`、Task 8 `[91, 95, 280]`。7 条成功与预训练策略是完全相同的配对初始状态；
少量步数差异没有改变成功分类。

## Adapter 评测路径

最终闭环直接使用未合并的 PEFT adapter 覆盖固定 Stage 2 checkpoint。一次诊断性 BF16 merge 让
某个固定 probe loss 最多变化 0.00856，而直接重载 adapter 的 loss 误差为 0。这与较小低秩更新在
BF16 执行前折叠进基础权重时发生数值改变的现象一致，因此非等价的 merged path 被拒绝，没有用于
正式 benchmark。

## 解释

- 两种方法都让离线 imitation loss 改善约 2.5%，但都没有改变 Task 5 的 0/3。离线 loss 仍然
  不能代理稀疏闭环成功率。
- LoRA 保留了预训练成功集合，而动作专家微调出现一条回归。在本次固定配对实验中，限制适配容量
  减少了 catastrophic interference。
- LoRA 用少 134.5× 的可训练参数和 3 MB adapter 达到更好的工程结果，但不能据此声称它产生了
  目标任务能力增益。
- Task 5 未改变说明下一步应优先改变训练信号或数据，而不是盲目增大 adapter：例如任务平衡 replay、
  失败状态数据或闭环训练目标。

## 产物审计

- 12/12 个评测视频 SHA-256 与记录一致，可在 360×360 下正常解码
- 4/4 个轨迹归档哈希一致，动作和 reward 均为有限值
- 共解码 2,210 个评测视频帧
- 闭环评测时间：662.40 秒（11.0 分钟）
- CUDA 评测峰值已分配显存：1.049 GB

## 复现命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/stage10_task5_lora.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
uv run --no-sync python scripts/stage5_small_benchmark.py \
  --checkpoint-dir outputs/stage2_checkpoint/checkpoint \
  --adapter-dir outputs/stage10_task5_lora/checkpoint_final/adapter \
  --output-dir outputs/stage10_task5_lora_eval \
  --task-ids 5 4 7 8 \
  --episodes-per-task 3 \
  --n-action-steps 10 \
  --fine-tune-report outputs/stage10_task5_lora/report.json \
  --comparison-report outputs/stage7_action_horizon_ablation/report.json
```
