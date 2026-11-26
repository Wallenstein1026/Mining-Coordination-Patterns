# 官方 Weibo-COV 数据接口

`weibo_cov.py`：固定选择、流式读取、SQLite 去重、规范表及来源清单签发。

`verify_weibo_handoff.py`：通过实际读取入口校验规范表；加 `--check-raw` 时重哈希已取得月表。

`audit_weibo_root_coverage.py`：在原始月表中扩大根帖端点查找范围。存在性匹配不代表逐跳传播树。

输入、固定配置与聚合审计详见 `data/README.md`、`configs/weibo_subset.json`、`results/weibo_data_audit.json`。导入器使用 POSIX `fcntl`，原始数据导入建议 Linux/macOS 或 WSL；规范表分析可在已验证依赖的其他环境运行。
