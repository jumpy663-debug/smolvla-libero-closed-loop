# World Model 公开结果

该目录只保存体积小、可审计且不含本机绝对路径的结果，不保存原始数据集、checkpoint、视觉
特征缓存或批量视频。

- [`stage11_data_audit`](stage11_data_audit)：LIBERO Spatial 专家数据 manifest、确定性
  train/validation/test 划分、既有闭环 rollout 模态审计和 SHA-256 摘要。
- [`stage12_representation_pilot`](stage12_representation_pilot)：冻结 DINOv2 双相机表征的
  80 样本 pilot、逐样本像素哈希、特征统计和缓存哈希。
- [`stage13_full_feature_cache`](stage13_full_feature_cache)：389 个 train/validation
  episode 的分片 manifest、库存哈希、全量特征统计和 resume 性能。
- [`stage14_next_latent_baseline`](stage14_next_latent_baseline)：严格配对的
  action-conditioned/no-action 一步预测、动作反事实、训练曲线和 episode-cluster bootstrap。
- [`stage15_multistep_rollout`](stage15_multistep_rollout)：公共 validation 窗口上的
  1/5/10/25 步递归 rollout、teacher-forced 对照、分任务结果和 bootstrap。
- [`stage16_closed_loop_recollection`](stage16_closed_loop_recollection)：12 个完整观测闭环
  回合的标签、步数、模态可用性、文件哈希和容器无关内容哈希。

实验设置、限制和字段解释见：

- [`docs/WM_STAGE11_DATA_AUDIT.md`](../../docs/WM_STAGE11_DATA_AUDIT.md)
- [`docs/WM_STAGE12_REPRESENTATION_PILOT.md`](../../docs/WM_STAGE12_REPRESENTATION_PILOT.md)
- [`docs/WM_STAGE13_FULL_FEATURE_CACHE.md`](../../docs/WM_STAGE13_FULL_FEATURE_CACHE.md)
- [`docs/WM_STAGE14_NEXT_LATENT_BASELINE.md`](../../docs/WM_STAGE14_NEXT_LATENT_BASELINE.md)
- [`docs/WM_STAGE15_MULTISTEP_ROLLOUT.md`](../../docs/WM_STAGE15_MULTISTEP_ROLLOUT.md)
- [`docs/WM_STAGE16_CLOSED_LOOP_RECOLLECTION.md`](../../docs/WM_STAGE16_CLOSED_LOOP_RECOLLECTION.md)
