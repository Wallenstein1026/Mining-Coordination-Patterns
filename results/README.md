# 当前结果与待完成实验

`misbot_benchmark.json` 和 `misbot_metrics.csv` 记录用 **MisBot 官方活跃账号文件**复跑的辅助基线。该文件与此前使用的镜像逐字节一致。`misbot_auc_intervals.csv` 保留此前镜像运行的区间估计。这些指标评估 MisBot 源域账号标签。[输入来源和哈希](../data/misbot_manifest.json)

`misbot_benchmark_summary.csv` 记录复跑的测试指标、样本数与输入哈希。模型交接元数据在 `misbot_style_model.json`，来源、文本重叠检查和复算命令见 [MisBot 审计](../docs/reviews/misbot_audit.md)。

当前包没有官方 Weibo-COV 实验结果，也没有已退出主实验的独立微博样本结果。正式重跑后只提交聚合数据审计、指标、消融、方法比较与验证记录。账号划分、预测和模型均留在本地。
