# Chang XU 接收说明 · 官方微博研究子集

**准备日 2026-11-25，北京时间。** 数据负责人 Qixiang HUA，实验接收人 Chang XU。运行目录为服务器 `/data4/datamining`，工作分支为 `data/weibo-cov`。准备时已部署读取和审计工具，三个正式月表均待接收。正式状态请查 `results/weibo_data_audit.json`。其中 `awaiting_monthly_csv` 表示等待，`monthly_import_verified` 表示导入与行数检查已完成，独立验收仍请执行第 3 步。

## 1. Qixiang 接收三个文件

在 [作者的 Weibo-COV 2.0 分享](https://pan.baidu.com/s/1mxU5RbnGBNRvR4Ci-9d0Hg?pwd=jffm)（提取码 `jffm`）获取以下文件，保持原始文件名和字节内容。存放目录为 `/data4/datamining/data/raw/original_weibo_cov/`。仅使用可访问该语料的授权工作环境。

| 月表 | 目录记录字节数 |
|---|---:|
| `2020-09.csv` | 2,040,438,498 |
| `2020-10.csv` | 1,276,271,699 |
| `2020-11.csv` | 1,250,403,608 |
| 合计 | 4,567,113,805 |

只需月度帖子表。完整研究窗口为 2020-09-01 到 2020-11-30；固定按帖 ID 哈希取约 1/32，所有天使用同一规则。用客户端把 CSV 下到个人电脑时，可用文件校验和核对服务器收到的同一份文件。

## 2. Qixiang 运行导入和审计

在服务器仓库根目录执行。第一条命令随时可用。只到一两个月时，第二条先输出已到月份的聚合 JSON，可将实际覆盖和每日帖数先发 Chang。等文件齐全后，运行第三条。导入器使用 Python 标准库，支持带引号和换行的 UTF-8 CSV，磁盘 SQLite 负责全局去重，导入器不依赖 MisBot。

```bash
cd /data4/datamining
.venv/bin/python src/adapters/weibo_cov.py --inventory
.venv/bin/python src/adapters/weibo_cov.py --audit-available
.venv/bin/python src/adapters/weibo_cov.py
```

成功后，读取以下三项。

| 文件 | 用途 |
|---|---|
| `data/processed/weibo.csv` | 本地规范表；主键 `mid`。下游必需列 `mid,user_id,created_at,content`，另有根帖、赞转评及其缺失标记 |
| `data/weibo_manifest.json` | 文件来源、版本、固定规则、原始与规范文件的真实字节摘要。与规范表配套放置 |
| `results/weibo_data_audit.json` | 可公开聚合结果，含进出行数、日期覆盖、逐日数量、互动缺失与根帖匹配分母；指标定义见[审计说明](weibo_cov_audit.md) |

## 3. Chang 独立验收

在自己的授权环境，用相同代码和固定配置重建，或使用同一受控目录中的规范表、manifest 和聚合审计。服务器已有 `.venv`；若在其他机器，自行用 Python 3.12 安装仓库 `requirements.txt`。在仓库根目录执行：

```bash
.venv/bin/python src/adapters/verify_weibo_handoff.py
```

有原始月表时，增加 `--check-raw`，对三个输入逐个重算 SHA-256。验收程序调用项目实际使用的 `run_analysis.load_weibo()`，重算有效行、唯一用户、日期边界、逐日数量、字段缺失和根帖匹配，核对规范表哈希。正常退出后回传以下一行，方括号按实际输出填写：

> Chang 已于［时间］在［环境／代码版本］复核。规范表 SHA-256 为［哈希］，有效帖［行数］，独立用户［数量］，实际覆盖［起止时间］，原始月表校验［通过／本环境未执行］。

拿到逐日覆盖后再填写 `configs/forecast.json`，保留次日目标的切分隔离日，检查各段正负例及候选覆盖。全期统计仅供描述，预测特征限当时可见信息。全样本根帖匹配率与抽样子集匹配率应区分。文案或海报使用的具体数值须来自本次导入。若 18:00 仍缺月表，直接向 Jiaxin 同步文件名、实际已到数据及下次更新时间。

## 可发群的当前状态

> 我已准备好 9–11 月官方微博导入器、固定抽样规则和质量审计。代码已经按作者示例做过接口验证。当前完整月表待到位；第一份到位后我先发逐日覆盖，三份到齐后发最终哈希、有效行数、日期范围和根帖匹配率。Chang 请用验收命令独立读取并回传摘要，再冻结预测切分。

原文、用户标识和逐帖表留在本地受控目录。代码、配置、公开聚合审计和接收说明可以随团队 PR 提交。个人贡献按实际操作填写，含 AI 使用。
