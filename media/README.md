# 展示媒体与来源说明

`task7_paired_comparison.mp4` 展示 Task 7、初始状态 2、H=10 下的严格配对结果：

| 画面 | 原始视频 | 结果 |
| --- | --- | --- |
| 预训练 | `stage7_action_horizon_ablation/videos/horizon_10/task_07/episode_02.mp4` | 成功，131 步 |
| 动作专家微调 | `stage9_task5_finetune_eval/videos/task_07/episode_02.mp4` | 失败，280 步 |
| LoRA | `stage10_task5_lora_eval/videos/task_07/episode_02.mp4` | 成功，126 步 |

原始视频以 80 FPS 保存。展示版本将其放慢到 20 FPS 并水平拼接；两个成功回合结束后停留在最后
一帧，直至动作专家微调回合运行到 280 步上限。视频没有混用不同任务或不同初始状态的画面。

文件校验值：

```text
6778f184da8a5e0f0444e141f62cd2489378962e46c4a864fa6b1c3d3b877eed  task7_paired_comparison.mp4
367ccbf40d60a42afcde557f191ea662ae4af2b764a1bfdce7891a8cbf2e34f9  task7_paired_comparison.jpg
```

`results_overview.svg` 根据 [`../results/results_summary.json`](../results/results_summary.json)
中经过审计的实验数据绘制；
`results_overview.png` 是适合幻灯片和网申附件使用的 1200×680 位图版本。

```text
a397e772dd94bf0331c54eac29568c6b51fd58297c200174f21a145d42069b6c  results_overview.svg
283b1a3a104a0fb9f60e70df83713144eb8e7599ecbca748f37b1eb8d90dae13  results_overview.png
```

`wm_stage17_failure_signal.svg` 根据
[`../results/wm/stage17_failure_signal`](../results/wm/stage17_failure_signal)
中的 episode 级分数与统计量绘制，展示 pooled H10 error 的初步排序信号以及
task-centered 后接近随机的限制。

```text
108f350819234d93434ba2e3d3823455c44561c0f206456f735f87f9c2267319  wm_stage17_failure_signal.svg
```

`wm_stage18_initial_state_scout.svg` 根据
[`../results/wm/stage18_initial_state_scout`](../results/wm/stage18_initial_state_scout)
中的 48 条 score-blind 标签筛选结果绘制，展示逐任务成功/失败分布、预注册平衡门槛和阻塞
cell。

```text
941aeed8439973c991fede5a88dbb3633acc84fd9811cc187f13a6eebe1c1cf7  wm_stage18_initial_state_scout.svg
```

`wm_stage19_balanced_cohort.svg` 根据
[`../results/wm/stage19_balanced_cohort`](../results/wm/stage19_balanced_cohort)
中的 Task 8 固定块扩展与三任务门槛审计结果绘制，展示为什么本阶段没有生成不满足预注册要求的
calibration/test 队列。

```text
ff437eb6a42745c79f43879db95465424906ae0807afed864e7e96049af8a412  wm_stage19_balanced_cohort.svg
```
