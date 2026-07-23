# WM 阶段 2：冻结视觉表征 Pilot

本阶段验证 World Model 的输入表征链路，不训练预测模型。目标是确认 LIBERO 双相机视频能够与
Parquet 中的机器人状态、动作严格对齐，并在本地 RTX 4060 Laptop GPU 上稳定生成可缓存的
视觉 token。

## 表征方案

冻结编码器采用：

- 模型：`facebook/dinov2-small`
- revision：`ed25f3a31f01632728cabb09d1542f84ab7b0056`
- 参数量：22,056,576
- 可训练参数：0
- 输入：两路 `256×256` RGB 图像
- 预处理：短边缩放至 256，再中心裁剪至 `224×224`
- 原始 patch：`16×16` 个、每个 384 维
- 池化：每个不重叠 `4×4` patch block 做均值，得到每相机 `4×4=16` 个 token
- 最终布局：`[sample, camera, spatial_token, channel]`

之所以先使用 DINOv2-S/14，是因为它体积小、输入接口稳定，并保留空间 token，而不是只留下单个
全局向量。这使后续 WM 可以先验证“动作条件下的潜变量动力学”，而不必同时训练视觉编码器。
这不是在本阶段宣称 DINOv2 一定是最优机器人表征。

## Pilot 数据

Stage 11 的固定 manifest 和划分保持不变，使用 `selection_seed=1200`：

| 项目 | 数量 |
| --- | ---: |
| 覆盖 split | train、validation |
| 每个 split 覆盖任务 | Task 0—9 |
| 每任务选取 episode | 1 |
| Episode 总数 | 20 |
| 每 episode 均匀取帧 | 4 |
| 对齐时刻数 | 80 |
| 解码图像数 | 160 |

每个时刻保存：

- 双相机视觉 token：`[2, 16, 384]`，FP16；
- `observation.state`：8 维，FP32；
- `action`：7 维，FP32；
- split、task、episode、frame 索引；
- 视频相对路径、精确查询时间戳和解码像素 SHA-256。

视觉缓存整体形状为 `[80, 2, 16, 384]`，与状态 `[80, 8]`、动作 `[80, 7]` 一一对齐。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| Train/validation Task 0—9 覆盖 | 通过 |
| 双相机 AV1 视频解码 | 通过 |
| 状态与动作有限值 | 通过 |
| 视觉 token 有限值 | 通过 |
| 同一 batch 重复冻结前向 | 逐 bit 一致 |
| Safetensors 保存后重载 | 逐 bit 一致 |
| 完整流程第二次复跑 | 通过，未产生非一致输出 |
| 独立样本均值向量 | 80/80 唯一 |

特征分布摘要：

| 指标 | 数值 |
| --- | ---: |
| 全部 token 均值 | 0.065703 |
| 全部 token 标准差 | 1.804675 |
| 跨样本平均 channel 标准差 | 0.431335 |
| 双相机均值 embedding cosine 平均值 | 0.487019 |
| 双相机 cosine 范围 | 0.236574—0.724938 |

这些结果只能支持“特征没有数值异常或完全塌缩”。Cosine 差异也符合两路相机视角不同的事实，
不能直接解释为任务信息质量。

## 本机资源

本次在 NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）上以 batch size 16 运行：

- 冻结编码器峰值 PyTorch GPU allocation：202,955,264 字节，约 193.6 MiB；
- 两次复跑中，80 个时刻的视频解码约 2.99—3.08 秒；
- 两次复跑中，160 张图像的编码约 0.62—0.82 秒；
- Pilot Safetensors 缓存 1,974,480 字节，约 1.88 MiB。

时间是单次本地测量，不作为跨机器 benchmark。显存数据是 PyTorch allocator 的峰值，不包括
驱动和 CUDA context 的全部占用。该结果说明当前 8 GB 显卡足以进入全量冻结特征提取阶段，
目前没有租卡必要。

## 复现

先保证 Stage 11 manifest 与固定 DINOv2 snapshot 已存在，然后从仓库根目录运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/wm/stage12_representation_pilot.py \
  --batch-size 16
```

公开结果位于
[`results/wm/stage12_representation_pilot`](../results/wm/stage12_representation_pilot)：

- `samples.csv`：80 个样本的可复现选择、时间戳和像素哈希；
- `report.json`：编码器 revision、模型文件哈希、张量形状、特征统计和缓存哈希。

约 1.9 MiB 的 `pilot_features.safetensors` 位于 `outputs/`，受 `.gitignore` 排除。它可以由脚本
从公开 manifest 重新生成，因此不提交二进制特征。

## 结论边界与下一阶段

- Pilot 只对每个 episode 均匀抽取 4 帧，尚未验证连续时间窗口的数据吞吐。
- 当前只验证表征，不包含 latent dynamics、next-latent prediction loss 或闭环策略干预。
- DINOv2 来自通用视觉预训练，不一定编码夹爪接触、精细位姿等机器人控制关键信息。
- 测试 split 完全未参与本阶段，继续保留到 WM 设计和超参数冻结之后。

下一阶段可以按同一格式提取连续的 train/validation 全量特征，并加入可恢复的分片缓存与
resume 校验；完成后才开始最小 latent dynamics baseline。
