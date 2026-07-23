# WM 阶段 6：完整观测闭环回采

Stage 15 已确认动作条件 latent dynamics 在多步预测中保持总体优势，但既有 78 个闭环
rollout 只保存动作、奖励和渲染视频，缺少与策略执行同步的双相机观测及 proprioception，无法
用于判断 WM 误差是否与策略失败相关。

本阶段按 Stage 11 保留的确定性索引重新执行一小批 SmolVLA 闭环轨迹，补齐后续失败信号分析
所需的数据模态。本阶段只回采和审计数据，不计算 WM failure score，也不声称 WM 可以预测或
改善闭环成功率。

## 固定协议

| 项目 | 设置 |
| --- | --- |
| 环境 | LIBERO Spatial |
| 策略 | 固定 revision 的预训练 SmolVLA |
| 动作执行窗口 | H=10 |
| 任务 | Task 4、5、7、8 |
| 初始状态 | 每任务 0、1、2 |
| Seed | 1000、1001、1002 |
| 分辨率 | 360×360 |
| 回合数 | 12 |
| 专家 test episode 使用数 | 0 |

这 12 个条件与 Stage 7 的 H=10 困难任务子集严格配对。旧结果包含 7 个成功和 5 个失败，
因此在不额外搜索 seed 的情况下同时覆盖两类闭环结果。

## 保存模态与时序

每个 episode 独立保存为本地压缩 NPZ：

- `camera1`：agentview，`uint8 RGB`；
- `camera2`：eye-in-hand，`uint8 RGB`；
- `eef_pos`、`eef_quat`、`eef_mat`；
- `gripper_qpos`、`gripper_qvel`；
- `joint_pos`、`joint_vel`；
- 实际经过 postprocessor 并传入 LIBERO 的 7 维 action；
- reward、success、累计 done 和 `transition_valid`。

若回合有 `T` 个控制步，则每路观测和状态保存 `T+1` 帧，action/label 保存 `T` 帧。正常
transition 的含义为：

```text
(observation[t], state[t], action[t]) -> observation[t + 1]
```

### 成功终止的特殊处理

当前 `LiberoEnv.step()` 在任务成功后会立即 reset，并把下一个初始状态作为 step 返回值。
因此成功回合的最后一张图并不是真实 terminal observation，最后一条 transition 跨越了自动
reset。本阶段保留该观测用于审计，但将对应 `transition_valid[-1]` 标为 `False`，后续 WM
评分必须排除它。

失败回合由 280 步时限结束，最后一个 next-observation 仍属于原回合，因此全部 transition
有效。

## 回采结果

| Task | 成功/回合 | 控制步 | 有效 transition |
| ---: | ---: | ---: | ---: |
| 4 | 2/3 | 536 | 534 |
| 5 | 0/3 | 840 | 840 |
| 7 | 3/3 | 362 | 359 |
| 8 | 2/3 | 468 | 466 |
| **合计** | **7/12** | **2,206** | **2,199** |

12 个回合的 success 和 steps 均与 Stage 7 完全一致，step delta 全部为 0。这比只核对最终成功
标签更严格地验证了自定义采集循环与原官方闭环执行路径等价。

## 数据完整性审计

- 12/12 episode 同时包含两路 360×360 `uint8` 图像、完整 robot state、动作和标签；
- 12/12 满足 `observation_count = steps + 1`；
- 7 个成功回合各排除一条自动 reset transition，共得到 2,199 条有效 transition；
- 每个 NPZ 都在写入后重新打开，验证 shape、dtype、有限值和时序长度；
- 同时记录文件 SHA-256 和与 NPZ 容器无关的数组内容 SHA-256；
- 独立复算再次通过全部 12 个文件哈希和 12 个内容哈希；
- 公开 CSV/report 不含本机绝对路径，原始图像归档由 `.gitignore` 排除。

原始压缩归档共 593,029,103 字节，约 565.6 MiB。它们保存在本地
`outputs/wm/stage16_closed_loop_recollection`，不会推送至 GitHub。

## 本机资源

在 RTX 4060 Laptop GPU（8 GB）上：

- Task 4 三回合试采约 218.6 秒；
- 正式调用复用 Task 4，并完成其余九回合及公开结果汇总约 552.2 秒；
- 峰值 PyTorch GPU allocation 为 1,045,809,152 字节，约 997.4 MiB；
- 全程使用 EGL，无需真机或云端显卡。

## 复现

从本地 LeRobot 仓库根目录执行，并将脚本和输出路径替换为当前项目位置：

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python \
  /path/to/smolvla-libero-closed-loop/scripts/wm/stage16_collect_closed_loop.py \
  --task-ids 4 5 7 8 \
  --output-dir /path/to/smolvla-libero-closed-loop/outputs/wm/stage16_closed_loop_recollection \
  --public-results-dir /path/to/smolvla-libero-closed-loop/results/wm/stage16_closed_loop_recollection
```

脚本支持 task 级 resume：只有 task report 和三个 episode shard 同时存在且全部哈希、内容和
schema 复验通过时才跳过；部分输出会被拒绝，而不会静默覆盖。

公开结果位于
[`results/wm/stage16_closed_loop_recollection`](../results/wm/stage16_closed_loop_recollection)：

- `episodes.csv`：12 个回合的标签、步数、模态可用性和双重哈希；
- `report.json`：固定协议、总体统计、时序边界和 Stage 17 readiness。

## 结论边界与下一阶段

- 当前只有 12 个 episode、4 个 task，适合探索，不足以训练复杂 failure classifier。
- 同一批数据将用于提出和初步检查 score，不能冒充独立测试集。
- 成功回合缺少最后一动作后的真实 terminal observation。
- 本阶段没有计算 latent error，也没有证明其与成功率相关。

下一阶段将冻结 Stage 14 dynamics 和 DINOv2 encoder，对 2,199 条有效闭环 transition 计算
一步误差、短 horizon rollout error 和 action counterfactual disagreement。应先报告逐 episode
时间曲线和成功/失败的无监督效应量；若没有稳定分离，不继续包装成 failure detector。
