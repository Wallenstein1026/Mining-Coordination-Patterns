# 已交付结果

- `weibo_data_audit.json`、`weibo_root_coverage.json`：官方月表固定子集及根帖覆盖审计。
- `misbot_benchmark_summary.csv`、`misbot_benchmark.json`：官方文件核验后的三种账号基线；官方与早期镜像字节相同。
- `misbot_weibo_style_receipt.json`：规范微博上的辅助风格分数交接元数据，不包含逐帖数据。
- `experiments/`：Chang 提交的主实验和补充比较；主预测 Windows 结果保持原样，不被本次复算静默覆盖。
- `final_review/`：最终独立审查、macOS 对照、PDF 规格与哈希。

聚合结果不代替原始数据重跑；时间戳、候选覆盖、迁移/CIB 真值及跨平台差异见 [最终审查](../docs/FINAL_REVIEW.md)。原始内容、账号级划分/预测、模型和本机私有包不发布。
