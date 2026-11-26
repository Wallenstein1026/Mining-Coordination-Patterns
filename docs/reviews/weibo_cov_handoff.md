# Chang XU 数据交接单 · Weibo-COV 2.0

**可开始验收。** 2020-09-01 至 2020-11-30，254,624 帖，199,851 个用户，91 天，规范表 150.2 MB。 从三份官方月表经过预先固定的 ID 哈希抽样得到。数据版本以完整 SHA-256 为准。

```text
99aa5cfb20806185327f19c787a0fe0378b747a8fba43892ca4a387f6a9ee4ea
```

## 接收数据

研究服务器项目目录为 `/data4/datamining`。Chang 做聚类与预测，Xiaofeng 给同一批微博计算风格分数，两人均需访问规范表。Jiaxin 整合海报与报告时，聚合审计、指标表和图即可。原始月表留给 Qixiang 复核和未来扩容，逐行追溯或独立重建时再安排访问。

在已有授权的环境读取，或请 Qixiang/管理员把私有数据交接包交到接收者自己的受控工作目录。目前原始数据目录和规范表目录仅运行账号可读。使用各人自己的授权账号，接收者在自己的代码工作目录写入分析结果。

- 代码由团队 Git 分支 `data/weibo-cov` 或公开交付包提供。
- 私有数据包位于服务器 `.local/handoff/weibo_2020-09_11_private.tar.gz`。内含 `data/processed/weibo.csv` 与 `data/weibo_manifest.json`。如果使用独立工作副本，在仓库根目录解包到这两个相对位置。Git 中的 `results/weibo_data_audit.json` 随代码提供。
- 下载端三份 CSV 共 4,567,113,805 字节，只有使用 `--check-raw` 或完整重建时需要；普通验收和后续实验读规范表即可。

## Chang 验收

在仓库根目录运行以下命令（Python 3.12，已安装仓库依赖）。脚本先校验规范表哈希，再通过 `run_analysis.load_weibo()` 读取并核对行数、用户、日期、缺失及根帖匹配。该函数会在接收者的 `results/` 中写入聚合读取摘要。

```bash
.venv/bin/python src/adapters/verify_weibo_handoff.py
```

需要从三张原始月表完整重建时，依次执行以下命令。

```bash
.venv/bin/python src/adapters/weibo_cov.py
.venv/bin/python src/adapters/verify_weibo_handoff.py --check-raw
.venv/bin/python src/adapters/audit_weibo_root_coverage.py
```

**本次发送方复核结果。** 254,624 帖，199,851 个用户，91 天，规范表 150.2 MB。 实际范围 2020-09-01 00:00:00.000000 至 2020-11-30 23:59:00.000000。带根帖引用的帖子共 209,524 条；在规范子集内能匹配 604 条（0.29%），把查找范围扩大到三个月的原始表后能匹配 27,494 条（13.12%）。

**接收回执（由 Chang 本人填写）。** 已于［日期］在［自己的环境／代码提交］通过校验，规范表 SHA-256 为［输出］，有效帖［数量］，用户［数量］，覆盖［起止］，原始月表校验［已执行／本环境未运行］。

通过后按逐日覆盖冻结前向切分，核对每段正负例、候选覆盖和次日标签的边界。Xiaofeng 已完成同一版本规范表的帖子风格分数和话题内用户等权汇总。Jiaxin 可以先用本次审计描述研究样本，预测增益与 CIB 结论须等待对应实验。[风格分数交接](misbot_weibo_handoff.md)

## 后续扩展

保留当前配置和产物。新建独立的数据版本和输出目录后，可先对三个月保留全部帖子（哈希模数设为 1、余数 0），或把日期扩大到 13 张官方月表、维持 1/32。全时段全量需要再同时设为全保留。按官方文件清单，13 张月表合计 39.83 GB，还缺 10 张约 35.26 GB；需为原始表、SQLite 去重索引、规范表和实验中间结果预留空间。现有描述性分析会一次读入规范表并构建稠密事务矩阵，全量分析应先压测并改为分块/稀疏处理。更换数据版本后要重报覆盖率、重做切分与评估。

[正式审计](weibo_cov_audit.md) · [聚合结果](../../results/weibo_data_audit.json) · [完整数据发布说明](https://github.com/nghuyong/weibo-cov#weibo-cov-20)
