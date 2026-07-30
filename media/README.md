# 展示媒体与来源说明

## 统计扩展

`statistical_results_overview.svg/png` 由
[`../results/statistical_evaluation/statistics.json`](../results/statistical_evaluation/statistics.json)
生成，展示 100-state 基线、40-state Action Horizon/模型适配配对、114 个失败分类和 20-state
随机性审计。图中“缩短 H”只指 H=25/H=10 相对 H=50；图中同时明确 H=10 与 H=25 不显著。

```text
9aed2cab1a3a5e6a9d8d032bff844b295437f2d04856d9c6ee35012a66aaf9f8  statistical_results_overview.svg
801a156c22ac3f3f287be7af05ff89efa2861f1b5bc01f9d6b522ed8f8f191d9  statistical_results_overview.png
```

`statistical_horizon_task8_init9.mp4` 展示 Task 8、init 9、环境 seed 1009、策略 seed 1009 的严格
配对：H=50/H=25 均在 280 步失败，H=10 在 89 步成功。三个源视频分别为：

| H | 源 episode | 源视频 SHA-256 |
| ---: | --- | --- |
| 50 | `formal-pretrained-h50-task08-init09-env1009-policy1009` | `241f38e844b6533dad0601fe801bd4cc8a8761d421e7631ff961c2f39e87fc5e` |
| 25 | `formal-pretrained-h25-task08-init09-env1009-policy1009` | `b1e24fdac2d945a685b53f5438587268ec22389121e9dfaf937021366060d72a` |
| 10 | `formal-pretrained-h10-task08-init09-env1009-policy1009` | `df3f3eea4e9851b6099cbbed0014a502a8807ea0268c904554baf04115a85756` |

公开视频以 20 FPS 同屏播放；短回合在末帧停留，不混用状态或 seed。

```text
eddc398b12f28c727e1984fb4d6642f3e248dd348e3566bed2527c57ce15363a  statistical_horizon_task8_init9.mp4
44a3a6c2a79cc507484e3c1848dd7bdc70b8bd63247d86e1012740e6291891c7  statistical_horizon_task8_init9.jpg
```

`statistical_failure_examples.jpg` 从经过视频哈希验证的 12 帧 contact sheet 中各选择一条 H=50
失败：抓取失败、抓取后掉落、放置未稳定和放置位置错误。

```text
e96896bddd2e63274364d4041bc7e00c39c2306fc593d4556389f9f1f83c9d23  statistical_failure_examples.jpg
```

所有源路径、任务、状态、seed、配置、源哈希和公开产物哈希以
[`../results/statistical_evaluation/media_manifest.json`](../results/statistical_evaluation/media_manifest.json)
为准。公开产物可由
[`../scripts/stage32_build_statistical_public_artifacts.py`](../scripts/stage32_build_statistical_public_artifacts.py)
重建。

## 早期配对媒体

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
