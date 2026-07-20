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
