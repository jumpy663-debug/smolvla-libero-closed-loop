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

`wm_stage20_paired_interventions.svg` 根据
[`../results/wm/stage20_paired_interventions`](../results/wm/stage20_paired_interventions)
中的 fresh-reset 筛选、六组配对步数与故障有效性结果绘制。它展示 nominal 与 persistent
motion-dropout 的严格同状态对照，不包含 WM 评分结果。

```text
16c2e97ed39b371ccd9f124e14af37d678b2b23b5116b87c5499ee9d66148f7b  wm_stage20_paired_interventions.svg
```

`wm_stage21_paired_wm_scoring.svg` 根据
[`../results/wm/stage21_paired_wm_scoring`](../results/wm/stage21_paired_wm_scoring)
中的六组 pair-level 差分与精确检验结果绘制。左图展示预注册 commanded-error 主指标的方向
不一致与 decision rule 失败；右图展示仅作为诊断的 commanded-minus-executed gap，并明确标注
Holm 校正后的统计边界。

```text
e477967f5b3eec28628321546158e617e6ee4c4bfa96cc8858edc9956db0193f  wm_stage21_paired_wm_scoring.svg
```

`wm_stage22_confirmatory_interventions.svg` 根据
[`../results/wm/stage22_confirmatory_interventions`](../results/wm/stage22_confirmatory_interventions)
中的六组三条件轨迹绘制，展示两种温和动作干预保留的 EEF/像素运动、episode 步数增长，以及
18/18 回合成功的独立 score-blind 确认集。

```text
1b3442a53e4eba346d7471342d6f279e9c4a3173cf920614614c198371e74d7d  wm_stage22_confirmatory_interventions.svg
```

`wm_stage23_confirmatory_wm_scoring.svg` 根据
[`../results/wm/stage23_confirmatory_wm_scoring`](../results/wm/stage23_confirmatory_wm_scoring)
中的六个 pair 级确认分数绘制。左图展示预注册主检验的 5/6 同向结果；右图区分 0.5× 衰减
和 3-step 延迟的支持性结果，并保留效应 shrinkage 与不可直接外推为 shield 的边界。

```text
786b12032ec746de1a7b9d46516c922c46421d98b3a3202477d1c0cdeabb9e30  wm_stage23_confirmatory_wm_scoring.svg
```

`wm_stage24_online_sidecar.svg` 根据
[`../results/wm/stage24_online_sidecar`](../results/wm/stage24_online_sidecar)
中的 1,080 窗口因果回放和真实在线配对结果绘制。两段 MP4 分别展示 normal 与 0.5× 动作衰减
的 90 步 SmolVLA/LIBERO 监控前缀，左上角叠加 observation index 和最近一次 H10 gap；WM
输出没有参与动作生成。

```text
64615ea2d98383f0b4bc2d9411cdf7c71e6f432f6ca3f8872fd43fac1cf3d7ae  wm_stage24_online_sidecar.svg
1740ac09e7dadfbbfc9a6a5a28b8d36a98b9ac6d1b945d76bf17f6735bd9f0bf  wm_stage24_nominal_monitor.mp4
6aa57778fafaec16871b3582ba358c967bc9d4d27efdbb670c11c553cc9c5bbd  wm_stage24_motion_attenuation_0p5_monitor.mp4
```

`wm_stage25_detector_calibration.svg` 根据
[`../results/wm/stage25_detector_calibration`](../results/wm/stage25_detector_calibration)
中的 calibration-only threshold 与 6 个留出 test pair 绘制。它同时展示 WM detector 的正常
误报、两类故障检出，以及直接 action mismatch 基线为何阻止本阶段直接外推为 shield。

```text
25901dbb237c687526ecf80391c0e4bf64a5ea251027354e6f462ec7cafddcc6  wm_stage25_detector_calibration.svg
```

`wm_stage26_hidden_dynamics_cohort.svg` 根据
[`../results/wm/stage26_hidden_dynamics_cohort`](../results/wm/stage26_hidden_dynamics_cohort)
中的 score-blind pilot 门控和 6 对正式 discovery 诊断绘制。它展示 0.5× arm actuator
gain 通过、10× joint damping 被拒绝，以及通过条件在 action 接口逐位相同的前提下造成的
EEF/视觉分叉；图中不包含任何 WM 分数。

```text
64a18b9e10c79609af393b8475f446b09c0c94bf0e127a4e1e17c683f39afc3a  wm_stage26_hidden_dynamics_cohort.svg
```
