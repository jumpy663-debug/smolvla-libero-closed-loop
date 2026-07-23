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

实验设置、限制和字段解释见：

- [`docs/WM_STAGE11_DATA_AUDIT.md`](../../docs/WM_STAGE11_DATA_AUDIT.md)
- [`docs/WM_STAGE12_REPRESENTATION_PILOT.md`](../../docs/WM_STAGE12_REPRESENTATION_PILOT.md)
- [`docs/WM_STAGE13_FULL_FEATURE_CACHE.md`](../../docs/WM_STAGE13_FULL_FEATURE_CACHE.md)
- [`docs/WM_STAGE14_NEXT_LATENT_BASELINE.md`](../../docs/WM_STAGE14_NEXT_LATENT_BASELINE.md)
