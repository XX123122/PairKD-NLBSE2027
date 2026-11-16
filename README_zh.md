# PairKD 工具赛源码

源码与文档：https://github.com/XX123122/PairKD-NLBSE2027

模型及原教师/划分附件：https://github.com/XX123122/PairKD-NLBSE2027/releases/tag/v0.2.0

请从 [英文 README](README.md) 开始。正式模型为原 PKP2-seed13 epoch4；实验编辑权重不参与提交。代码采用 MIT，保留上游来源声明。请使用独立 Release 模型附件，不将模型 ZIP 推进普通 Git 历史。

新增了按官方 train 和固定划分生成训练输入、生成带哈希记录的新 seed13 头、缓存教师 TRAIN logits、绑定学生训练 job 的复现路径。原教师权重及划分随附件提供，原始数据从官方获取。此路径不保证历史 checkpoint 逐位一致；此前原教师的上游训练链不在该重跑路径内。本轮没有新训练或前向。

完整新环境重训和组织者接口尚未实际复测，不将它们写成已通过。详见 [复现范围](docs/REPRODUCIBILITY.md) 和 [发布状态](docs/RELEASE_STATUS.md)。仓库发布不等于比赛表单已提交。
