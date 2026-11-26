> 2026-11-26 最终整合注：本文记录成员提交的 Windows 主结果；独立 macOS 复算支持主要结论，但 RF 存在少量差异，详见 [最终审查](../FINAL_REVIEW.md)。不把以下数值称为跨平台逐位完全一致。

# Chang XU 实验验收

2026-11-25（北京时间）｜分支 `experiments/mining-and-forecast`。原定实验均有结果，正式预测配置、聚合表、复算记录和三张核心矢量图已整理为仓库交付材料。 [分工](../TEAM_TASKS.md#chang-xu回答挖出的东西有没有用)

数据为官方 Weibo-COV 2.0 的 2020 年 9–11 月固定子集，共 254,624 帖、199,851 位用户。Codex 在接收环境核对了规范表的 SHA-256（文件校验指纹）、91 天的逐日计数、作者数及根帖匹配数。原始月表校验由发送方记录。 [本地接收回执](../../results/experiments/weibo_receipt.json) · [数据交接](weibo_cov_handoff.md)

## 逐项核对

| 分工项 | 状态及实际结论 | 对应图表 | 还缺什么 |
| --- | --- | --- | --- |
| 1. 审核入口，定任务、阈值、指标、对照和时间切分 | **已登记并复跑。** 预测次日热点以至少 4 帖、3 位作者、相对前 7 天的发帖份额至少翻倍为门槛。训练截至 10-09，验证 10-11 至 10-24，测试 10-26 至 11-29，均为 2020 年预测发起日。独立入口读取仓库配置。 | [预测配置](../../configs/forecast.json) · [验证期记录](../../results/experiments/forecast_development_summary.json) · [正式测试记录](../../results/experiments/forecast_summary.json) | 保存的验证期与正式测试输出采用同一口径。配置是在本次交付补录，首次测试前的外部时间戳待证。 |
| 2a. 文本、行为、融合各跑 K 均值和 Ward，比较轮廓系数及条件稳定性 | **完成。** 同一融合坐标下，七组融合 K 均值的轮廓系数最高（0.1564，分离仍有限）。固定七组重抽样，文本 K 均值的调整 Rand 指数（ARI，分组一致程度）均值最高（0.8864）。抽样后重做特征，融合方案的平均 ARI 为 0.751，最低为 0.548。 | [图 03](../../figures/experiments/03_method_comparison.pdf) · [图 15](../../figures/experiments/15_descriptive_comparisons.pdf) · [图 16](../../figures/experiments/16_mining_extensions.pdf) · [聚类对照表](../../results/experiments/clustering_grid.csv) | 分工要求的对照齐全。标题盲评为单次人工智能小样本初筛。 |
| 2b. 分别加入 HMI 和用户等权风格分数，再联合加入 | **完成。** HMI 是四项发帖行为的相对综合分。单独加入 HMI 或用户等权风格分数时，轮廓系数从 0.1564 分别降到 0.1232 和 0.1413。联合权重 0.25 时，轮廓系数为 0.1540，抽样平均 ARI 从 0.7609 升到 0.8271。交接包 254,624 条逐帖分数全部对齐，1,422 个话题的汇总分数复算通过。 | [图 12](../../figures/experiments/12_cib_informed_profiles.pdf) · [图 20](../../figures/experiments/20_misbot_style_diagnostics.pdf) · [图 19](../../figures/experiments/19_misbot_style_ablation.pdf) · [七组消融表](../../results/experiments/cib_informed_ablation.csv) | 组员已核对官方输入与镜像逐字节一致。本地分数包与已发布回执一致。微博缺少人工真值，分数用于描述文本风格。 |
| 2c. 对照 Apriori 与 FP-growth，按 HMI 和增长解释时段模式 | **完成。** 两算法在三档支持度下得到同样的标签组合。最低支持度 0.3% 时均得 54 组，Apriori 重跑中位耗时 0.162 秒，FP-growth 为 0.617 秒。高 HMI 话题更常出现白天集中模式（39.47% 对 20.09%）。同一话题在热点日的午后至夜间集中模式占比高 28.08 个百分点（每日至少 4 帖）。 | [图 13](../../figures/experiments/13_hmi_context_patterns.pdf) · 图 15、16 · [计时表](../../results/experiments/itemset_repeated_runtime.csv) · [同话题模式对照表](../../results/experiments/matched_topic_motifs.csv) | 分工要求的对照齐全，耗时和时段差异的结论限于本机或本样本。 |
| 3. 日末预测次日热点，比较近期量、当前增长、仅活跃度、加 HMI、再加模式或聚类 | **完成。** 两个排序规则和三档输入各配逻辑回归、随机森林，共八组。时段模式和聚类中心只从训练期确定，主对照模型按验证期选择。补充拆分显示，单加时段模式比例的平均精确率（AP，衡量排序效果）为 0.2352，完整随机森林为 0.2369。 | [图 14](../../figures/experiments/14_nextday_prediction.pdf) · [图 17](../../figures/experiments/17_forecast_extensions.pdf) · [八组正式结果](../../results/experiments/forecast_metrics.csv) · [输入拆分表](../../results/experiments/feature_ablation.csv) | 补充消融复用了原测试期。若采用新特征或候选规则作为最终方案，需用新留出期验证。 |
| 4. 报告 AP、每日前五名精确率和召回率、测试正例、候选覆盖及周分块区间 | **完成。** 35 天的候选清单含 763 个热点，样本共有 2,485 个热点，覆盖率 **30.70%**。完整随机森林 AP **0.2369**，前五名精确率 **48.00%**（84/175），候选内召回率 **11.01%**（84/763），全样本召回率 **3.38%**（84/2,485）。相较仅活跃度，AP 增量 **0.0893**，五周均为正，95% 周分块区间 **[0.0604, 0.1204]**。单加 HMI 的 AP 增量仅 0.0013，部分周为负。 | 图 14、17 · [指标与区间](../../results/experiments/forecast_summary.json) · [逐周增益表](../../results/experiments/forecast_weekly_ap_deltas.csv) | 所需指标齐全。区间只有五个周块，证据范围为该样本、时段及候选规则。 |
| 5. 分开全期 HMI 与历史窗口 HMI，检查 MisBot 的时间可用性 | **完成。** 描述性聚类用全期 HMI，历史预测用预测日末可见的近 28 天特征。截断验证期中点后的数据，既有候选、预测输入及训练期选出的模式保持一致。MisBot 分数仅用于回顾性聚类。 | 图 19 · [预测复算记录](../../results/experiments/forecast_test_validation.json) · [MisBot 交接复算](../../results/experiments/misbot_handoff_validation.json) | 将来若让 MisBot 进入历史预测，须先证明当日能取得模型及分数。 |
| 6. 审核传播字段，高低 HMI 各检查至少三个正常案例 | **完成。** `origin_weibo` 是根帖引用，209,524 条非空引用中仅 604 条在本子集匹配（0.29%）。图 06 只表示标签共现。另按规模和活跃天数匹配三对高低 HMI 话题，六例均能对应公开事件，高组的文本重复率均较高。集中报道或共用措辞是与数据相符的解释。 | [图 06](../../figures/experiments/06_hashtag_network.pdf) · [图 11](../../figures/experiments/11_hotspot_growth.pdf) · [图 18](../../figures/experiments/18_matched_cases.pdf) · [案例汇总表](../../results/experiments/matched_case_aggregates.csv) · [事件出处](../../EXPERIENCE.md#103-配对案例与根帖覆盖) · [根帖覆盖](../../results/experiments/observed_root_references.json) | 六例的事件背景已核对。样本只支持根帖覆盖审计和案例解释，逐跳传播需要父帖关系。 |
| 7. 提交报告、聚合表、复算记录、约三张核心矢量图 | **交付材料齐备。** 描述性检查 22 项、回顾性解释检查 35 项、验证期预测检查 29 项、正式预测检查 38 项、MisBot 交接检查 20 项。已有隔离目录复跑和指标比对。 | [描述性复算](../../results/experiments/descriptive_validation.json) · [回顾性复算](../../results/experiments/retrospective_validation.json) · [复跑记录](../../results/experiments/release_verification.json) · 下列三张图 | 组员可凭其授权输入复跑，独立签收回执待组员填写。 |

七组文字与行为聚类的预测消融另见图 23。验证期中，完整模型比仅活跃度的 AP 高 0.0192，两者每日前五名的总命中数均为 21。 [图 23](../../figures/experiments/23_forecast_cluster7_ablation.pdf) · [结果表](../../results/experiments/forecast_cluster7_validation_metrics.csv)

## 三张核心矢量图

- **图 15**　固定七组时，文本 K 均值的抽样分组最稳定。 [PDF](../../figures/experiments/15_descriptive_comparisons.pdf)
- **图 19**　联合加入低权重 HMI 和用户等权风格分数后，抽样稳定性均值升高，分离度接近原融合基线。 [PDF](../../figures/experiments/19_misbot_style_ablation.pdf)
- **图 14**　这五周里，完整随机森林比仅活跃度随机森林多命中 40 个热点，AP 从 0.1476 升到 0.2369。 [PDF](../../figures/experiments/14_nextday_prediction.pdf)

## 复跑与贡献

使用 Python 3.12、仓库固定依赖及中央处理器（CPU）。按手册先运行 `describe`，再运行 `forecast` 和 `forecast --held-out`，随后运行独立的 MisBot 及两套补充实验入口。仓库中的聚合表与核心图可直接查看。复跑产生的逐帖或逐话题明细写入 Git 忽略目录 `.local/experiments/standalone/`。 [环境依赖](../../requirements.txt) · [复跑命令](../../EXPERIENCE.md#9-重跑命令与产物位置) · [补充实验命令](../../EXPERIENCE.md#105-补充实验复跑)

组员从作者公布的入口取得 MisBot 活跃账号文件，与早期镜像逐字节比对一致。实验进一步核验了私有交接包、发布回执和微博子集的对应关系。 [来源审计](misbot_audit.md) · [已发布回执](../../results/misbot_weibo_style_receipt.json) · [交接复算](../../results/experiments/misbot_handoff_validation.json)

Chang XU 负责上述挖掘、预测和证据交付。Codex 按其要求协助编写独立入口、执行复算和整理报告。微博规范表由 Qixiang HUA 交接，MisBot 风格分数由 Xiaofeng LI 交接。组员复跑回执请记录复跑人、日期、代码提交、规范表校验指纹，以及正式预测的正例数、前五名命中数和 AP 增量。
