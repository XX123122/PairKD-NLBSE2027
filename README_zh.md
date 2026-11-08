# PairKD 投稿仓库本地候选

本目录只包含原 PKP2 工具赛所需的源码、四页具名论文、推理 notebook、结果汇总和复现说明。原提交权重与阈值保持不变，不包含最近四轮实验权重。

入口是 [英文 README](README.md)。源码可在这个目录初始化为新 Git 仓库；不要上传整个实验工作区。模型 ZIP 约 202 MB，应放在独立 release 或模型托管处，不能直接当普通 Git 文件提交。目录外 `PairKD_RELEASE_ASSETS_20261107` 存放原包与原配 notebook。

当前仓库为私人仓库：https://github.com/XX123122/PairKD-NLBSE2027。只有获授权账号能访问，不是公开发行或正式提交。模型以私人仓库 Release 附件提供。待解决：真实仓库/模型 URL、许可最终选定及第三方分发范围、缺失的原训练资产/完整源链、组织者入口的实际兼容性。详细现状见 [RELEASE_STATUS](docs/RELEASE_STATUS.md)。整理和合成测试不会调用真实模型；11-05 的 50 次前向预算已用完，本次新增前向为零。
