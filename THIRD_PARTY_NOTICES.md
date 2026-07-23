# 第三方项目与模型

本仓库提供独立的实验脚本和本地实验结果，不重新分发模型权重、数据集或 LIBERO assets。

- [LeRobot](https://github.com/huggingface/lerobot)：实验代码依赖并固定到提交
  `b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf`；上游采用 Apache-2.0 许可证。
- [SmolVLA LIBERO checkpoint](https://huggingface.co/lerobot/smolvla_libero)：运行时从
  Hugging Face Hub 下载，实验固定 revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`。
- [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)：
  SmolVLA 使用的视觉语言主干，实验固定 revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`。
- [DINOv2-Small](https://huggingface.co/facebook/dinov2-small)：WM 表征 pilot 使用的冻结视觉
  编码器，实验固定 revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`；模型权重不在
  本仓库重新分发。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 与
  [LeRobot LIBERO 数据集](https://huggingface.co/datasets/lerobot/libero)：分别提供仿真任务和
  demonstrations；数据实验固定 revision
  `2e98211ba27db9322efbaa5ed108b34bb03ca163`。

使用上述第三方资源时，应同时遵守各自仓库或模型卡中的许可证和使用条款。
