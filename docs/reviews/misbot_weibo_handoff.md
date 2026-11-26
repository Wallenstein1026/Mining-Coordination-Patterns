# Xiaofeng LI · Weibo-COV 风格分数交接

2026-11-25 已在正式 Weibo-COV 2.0 研究子集上评分。输入覆盖 2020-09-01 至 2020-11-30，共 254,624 帖、199,851 名用户。模型用已核验的官方 MisBot 活跃账号文件训练。Xiaofeng 提供官方 MisBot 文件与 Weibo-COV 交接包，Codex 协助验收、实现评分入口、复算并打包。[Weibo-COV 数据审计](weibo_cov_audit.md) · [MisBot 审计](misbot_audit.md)

## 接收和使用

私有包 `misbot_weibo_2020-09_11_570a297f_private.tar.gz` 含两张表和 `receipt.json`。两张表在组内受控环境使用。包的 SHA-256 为 `e21596656f776b134eadde3f059a27937900c3c9c37ded52d822491ee60d8b04`。逐表哈希和输入版本见[聚合回执](../../results/misbot_weibo_style_receipt.json)。

| 文件 | 行数 | 字段和接合方式 |
|---|---:|---|
| `post_style_scores.csv` | 254,624 | `mid`、`bot_style`。以字符串读取 `mid`，与同版本规范表一对一接合。 |
| `topic_style_scores.csv` | 1,422 | `topic`、`posts`、`users`、`bot_style`、`user_balanced_style`。以 `topic` 与描述性话题表接合，核对帖数和用户数。 |

话题须至少有 30 帖、5 名用户，广义采集标签沿用主分析排除规则。话题表的 `bot_style` 是帖子等权均值。`user_balanced_style` 先计算每名用户在该话题内的帖子均值，再把这些用户均值等权平均。帖子可以属于多个话题。

Chang 可用 `user_balanced_style` 做描述性消融，并与 `bot_style` 对照高活跃用户对聚合分数的影响。分数是从 MisBot 迁移的连续文本风格信号。源域账号阈值 0.37 仅用于复现基线。COVID 目标域标签、校准和历史预测时点的模型可用性仍需独立证据。

## 复跑与验收

在已具备受核验的官方 MisBot 和 Weibo-COV 文件的仓库根目录运行以下命令。已有可信本地冻结模型时可直接验证并评分。

```bash
python src/adapters/verify_weibo_handoff.py
python src/run_analysis.py --stage misbot
python src/validate_misbot.py
python src/score_weibo_style.py
```

评分入口检查发布版规范表和本地模型哈希，并把两张 CSV 和 `receipt.json` 写入 `.local/misbot_weibo_style/` 下的结果目录。本次核对了所有帖子 ID、两个 CSV 的哈希和压缩包内文件，抽样重算了 1,280 条帖子的模型分数，并独立复算了全部 1,422 个话题的帖数、用户数和两种聚合。组员可使用随包回执核对文件后接手消融分析。
