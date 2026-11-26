# AIAA 3111 · Group 28

**Mining Coordination Patterns in Emerging Social Media Events: CIB-Informed Hot-Topic Clustering and Pattern Mining**

Jiaxin DONG · Qixiang HUA · Xiaofeng LI · Chang XU

[Poster.pdf](Poster.pdf) · [Report.pdf](Report.pdf) · [海报预览](Poster_preview.png) · [最终审查](docs/FINAL_REVIEW.md) · [贡献说明](docs/WORK_DONE.md)

官方 Weibo-COV、MisBot 与三位成员的交付已整合，并于 2026-11-26 独立复核。报告为 **9 页正文 + 1 页披露/参考文献**；海报为 **80 × 100 cm 竖版**，包含三组核心矢量图，明确展示 CIB 与 Human-Manipulation Index (HMI)。

## 主要结论与边界

- 官方 Weibo-COV 2.0，2020 年 9–11 月：从 8,149,927 帖固定选取约 1/32，保留 254,624 帖、199,851 用户、91 天；1,422 个合格话题。
- 七组融合聚类 silhouette 0.1564。HMI 和 MisBot 风格信号帮助描述分组，但不构成 CIB 真值，也不一致改善紧凑性。
- 两种项集算法在 0.3% 支持度下均得 54 项集；发现可重复的六小时时间模式。分组差异是回顾性关联。
- 正式时间切分中，完整 RF 的 AP 为 0.2369，活跃度基线为 0.1476。每日审核五个话题时，主运行命中 84/175（精确率 48.0%）；独立 macOS 运行命中 83/175（47.4%），AP 0.2365。
- **候选覆盖仅 30.7%，主运行全样本召回仅 3.38%。** 单独加入协调/HMI 特征块的 AP 增益只有 0.0013（四个滚动分量加 HMI），主要增益来自时间模式。不能据此声称预测所有热点或识别真实操纵。

原 Windows 结果作为成员交付的主结果保留。相同输入/配置的 macOS 复算存在少量 RF 数值差异，已列入 [对照表](results/final_review/forecast_replication_comparison.csv)。只有五个测试周块，首次测试前的外部冻结时间戳未核实，补充实验复用了测试期；论文和海报均披露这些限制。

## 环境

Python 3.12，依赖按 `requirements.txt` 固定。从仓库根目录运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。分析入口可读取已规范化数据；原始月表导入器使用 POSIX `fcntl` 文件锁，建议在 Linux/macOS 或 WSL 导入。PDF 几何/文本检查使用 Poppler：macOS `brew install poppler`，Ubuntu `sudo apt-get install poppler-utils`。

## 不需要原始数据的操作

```bash
python -m unittest discover -s tests -v
python src/check_repository.py
python src/build_deliverables.py
python src/validate_deliverables.py
```

单元测试中的官方 README 示例测试需要额外设置 `WEIBO_PUBLISHER_EXAMPLE`，缺少时会明确跳过；最终本地审查已准备哈希锁定示例，10 项全通过。PDF 构建只读取公开汇总结果，不代表重新执行了数据实验。最终输出在 `output/pdf/`，同时生成同内容的根目录提交文件。

## 数据与重新执行实验

原始记录、用户标识、账号级预测、模型和私有评分包不入库。官方数据入口、提取码、固定抽样和文件哈希见 [数据接口](data/README.md)、[数据来源](docs/DATA_STATUS.md) 与 [正式审计](docs/reviews/weibo_cov_audit.md)。官方原始文件需从作者发布入口下载；百度网盘大文件可能要求客户端。公开仓库不提供任何账号登录凭据。

1. 将官方 2020-09、10、11 月 CSV 放入 `data/raw/original_weibo_cov/`，使用固定的 `configs/weibo_subset.json`：

```bash
python src/adapters/weibo_cov.py
python src/adapters/verify_weibo_handoff.py --check-raw
```

也可在组内受控环境接收规范表和 manifest，放到 `data/processed/weibo.csv`、`data/weibo_manifest.json` 后运行不带 `--check-raw` 的验收。公开 [release manifest](data/weibo_release_manifest.json) 仅含相对路径和哈希，可用于核对版本；不能代替真实数据。

2. 将官方 MisBot 活跃账号文件按 [manifest](data/misbot_manifest.json) 放置并检查：

```bash
python src/run_analysis.py --stage misbot
python src/validate_misbot.py
```

3. 使用成员交付的**正式独立入口**复跑，先描述性、验证期，再按已保存配置报告测试期：

```bash
python src/run_experiments.py describe
python src/run_experiments.py forecast
python src/run_experiments.py forecast --held-out
python src/mining_extensions.py
python src/forecast_extensions.py
```

结果写入 Git 忽略的 `.local/experiments/standalone/`。主预测独立构造过去可见候选、滚动 HMI 和训练期模式/聚类；不读取全期话题筛选、全期 HMI 或未来训练的 MisBot 模型。`predict_hotspots.py` 和 `validate_forecast.py` 现在只是这套实现的兼容入口，默认验证期。

可选回顾性辅助信号实验：`python src/run_experiments.py describe --with-misbot`。已有组内私有评分包时，运行 `python src/misbot_handoff_experiment.py /path/to/PRIVATE_PACKAGE.tar.gz`，独立核对全部帖子和话题汇总后复跑消融。只能加载可信本地生成的 pickle。

## 团队贡献与最终提交

| 成员 | 已完成工作 |
|---|---|
| Jiaxin DONG | 基本框架、小公开数据集先行测试、统筹检查整合、最终 poster/report |
| Qixiang HUA | 官方微博文件、固定子集、流式导入、质量与根帖覆盖审计 |
| Xiaofeng LI | 官方 MisBot 核验、账号基线、帖子/话题辅助风格分数交接 |
| Chang XU | 正式聚类/HMI/模式/预测实验、消融、敏感性与图表交付 |

详见 [贡献与 AI 披露](docs/WORK_DONE.md)、[交付验收](docs/TEAM_TASKS.md) 及报告第 6 节。代码、执行、审查、图文均有 Codex 辅助；没有虚构未记录的独立人工签收。成员须按课程要求理解、核验并改写后提交 Canvas。

海报截止：2026-11-27 23:59，北京时间。现场展示：2026-12-04 15:00–16:20。最终 `Group_28_Project.zip` 包含报告、代码、README、汇总证据和数据下载说明；海报另行提交。
