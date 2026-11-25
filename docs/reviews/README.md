# 成员交付记录

由实际负责人提交 `weibo_cov_audit.md`、`misbot_audit.md`、`experiments.md`。

每份包含：实际执行者与日期、数据来源/版本/哈希、命令与环境、输出路径、核查结果、可靠结论、未完成事项、个人贡献与 AI 使用。只提交可公开的聚合记录。

## Xiaofeng LI

- [x] 核对官方 MisBot 活跃账号文件、镜像、标签与脱敏状态。两份文件逐字节一致。[来源审计](misbot_audit.md)
- [x] 用官方文件复跑并复算三种基线，记录账号切分和文本重叠。[基线指标表](../../results/misbot_benchmark_summary.csv)
- [x] 冻结本地模型和验证阈值，验证话题内用户等权评分接口。[模型交接元数据](../../results/misbot_style_model.json)
- [ ] 生成官方 Weibo-COV 的帖子风格分数与话题内用户等权结果。当前工作区等待规范 CSV 和来源清单。[数据接口](../../data/README.md#weibo)
