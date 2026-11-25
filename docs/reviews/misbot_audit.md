# MisBot 来源审计与基线复核

工作包负责人为 Xiaofeng LI，分支为 `models/misbot`。2026-11-24 由 Codex 审计镜像、修改代码并复跑基线。2026-11-25 由 Xiaofeng 下载官方文件，Codex 完成两份文件的比对和官方输入的复跑。[小组分工](../TEAM_TASKS.md)

## 数据版本和标签

- 上游作者把 `User_Instances/train_data_sampled.jsonl` 定义为活跃的人工标注账号子集。论文 §2.4 报告三名标注者按多数票给出账号标签，活跃子集含 18,132 个 bot 和 30,404 个 genuine 账号。实验使用 `1=bot`、`0=genuine`。作者另行发布了弱监督的 `inference_labels.json` 和帖子真假样本，二者均在这个训练分支的输入范围之外。[官方仓库](https://github.com/whr000001/MisBot) · [论文 §2.4](https://aclanthology.org/2025.emnlp-main.1604/) · [官方字段示例](https://github.com/whr000001/MisBot/blob/main/load_example.py)
- 11 月 25 日取得作者公开 Drive 中的原文件。官方文件与镜像均为 68,846,699 字节，`shasum -a 256` 均得到 `01d8170eb494367694c3403bb33b702eca19c35469c56df529c3e025778b7e16`，`cmp -s` 逐字节比对通过。本次复跑使用官方下载文件，manifest 标记为 `official`。[官方 Drive 文件](https://drive.google.com/file/d/1XK4X9hp3NZWmOyxIPe_cq00sLEQqH6sk/view) · [镜像目录](https://github.com/VladPlusIn/Research_Code/tree/main/MisBot/MisBot/User_Instances) · [输入 manifest](../../data/misbot_manifest.json)
- 文件含 48,536 个账号。两个类别的计数及三维数值、二十维类别字段与上游资料相符。每行有 5 条非空文本，共 242,680 条。
- 作者在 2025-08-25 公告发布了遮蔽可识别信息的版本。镜像中 48,536 个 `uid` 都是带字母前缀的编码，原文仍含 `@`（56,853 条）和 URL 形式字符串（5,195 条）。这只说明已有字段的外观，无法据此证明完全匿名。基线使用帖子风格特征和账号统计字段，描述字段仅在结构审计中计数。账号表、帖子、预测和模型文件保留本地。[作者的发布记录](https://github.com/whr000001/MisBot#updating)

## 账号切分和指标

按账号标签分层，固定 seed 28，切分 60/20/20。三个模型只在训练集拟合。验证集以 0.10 至 0.90、步长 0.01 的网格最大化账号平衡准确率，同分取最低阈值。三个模型在本次运行都选到 0.37。Text-style LR 先以帖子特征学习账号标签，再对一个账号的各帖子分数取均值用于阈值选择和评估。官方文件复跑得到的三组测试指标与镜像运行一致。

| 集合 | 账号 | 正例账号 | 帖子 |
|---|---:|---:|---:|
| 训练 | 29,121 | 10,879 | 145,605 |
| 验证 | 9,707 | 3,626 | 48,535 |
| 测试 | 9,708 | 3,627 | 48,540 |

| 模型 | 测试 ROC AUC | AP | F1 | 平衡准确率 | 阈值 |
|---|---:|---:|---:|---:|---:|
| Text-style LR | 0.6893 | 0.5657 | 0.5574 | 0.6328 | 0.37 |
| Metadata LR | 0.6102 | 0.4624 | 0.5222 | 0.5879 | 0.37 |
| Metadata RF | 0.6211 | 0.4900 | 0.5155 | 0.5859 | 0.37 |

三组交叉均无重叠 `uid`。原文按字符串逐字比较。规范化比较先移除 URL、部分 `@` 提及和空白、统一为小写，并排除空文本。帖子计数包括同一字符串的多次出现。

| 集合对 | 共享原文种数 | 原文命中帖子（前集 / 后集） | 共享规范化文本种数 | 规范化命中帖子（前集 / 后集） |
|---|---:|---:|---:|---:|
| 训练 / 验证 | 818 | 5,795 / 2,560 | 977 | 6,716 / 2,997 |
| 训练 / 测试 | 870 | 5,901 / 2,549 | 1,044 | 6,900 / 3,004 |
| 验证 / 测试 | 511 | 2,111 / 2,061 | 614 | 2,492 / 2,420 |

测试帖中剔除与训练集重合的规范化文本及规范化为空的帖子后，还剩 45,408 条，帖子级 AUC 从 0.6388 变为 0.6364。这个敏感性检查保留原训练模型，评估单位为继承账号标签的帖子。账号级指标仍对应完整的测试账号集合。账号独立与文本独立是两项不同的检查。

## 复跑和交接

在仓库根目录使用 Python 3.12 和 `requirements.txt` 指定的版本。复跑使用 Python 3.12.9、NumPy 1.26.4、pandas 2.2.2 和 scikit-learn 1.5.1。

```bash
python src/run_analysis.py --stage misbot
python src/validate_misbot.py
python -m unittest discover -s tests
```

运行后可核对[测试指标表](../../results/misbot_benchmark_summary.csv)、[样本及重叠审计](../../results/misbot_benchmark.json)和[模型交接元数据](../../results/misbot_style_model.json)。验证入口从本地预测逐项复算验证阈值和测试指标，并核对源文件与本地模型哈希。模型 SHA-256 对应本次本地序列化文件，跨环境重训时应重查分数与新文件哈希。

`results/text_style_model.pkl` 由可信本地运行生成。已用该模型在官方 Weibo-COV 的 254,624 帖上计算 `bot_style`，并独立复算 1,422 个话题的 `user_balanced_style`。评分入口、字段和已核验的输入版本见[风格分数交接](misbot_weibo_handoff.md)。

COVID 域的时间、话题、用户构成及转发写法可能与 MisBot 活跃账号子集不同，且目标域缺少账号及 CIB 真值。风格分数用于描述性消融，0.37 仅供源域账号基线复现。目标域准确率和 CIB 概率需要相应真值与校准。历史预测沿用其独立的过去可见特征。
