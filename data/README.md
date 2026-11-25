# 本地数据接口

原始文件放 `data/raw/original_weibo_cov/`、`data/raw/misbot/`，规范化研究子集放 `data/processed/weibo.csv`；这些目录被 Git 忽略。不要上传社交媒体原文、用户标识或模型。

## Weibo

规范 CSV 必需列：`mid,user_id,created_at,content`；`created_at` 为北京时间、无时区字符串。官方 `_id` 映射到 `mid`，保留 `origin_weibo`、互动计数与缺失标记等额外列。根帖引用不默认是逐跳父节点。

导入器生成 `data/weibo_manifest.json`，含数据集、版本、来源网址、时区、固定抽样配置、配置文件哈希、原始文件名/字节数/哈希、规范表相对路径及 SHA-256。运行验收时，规范表和这份清单须来自同一版本。公开可查的聚合审计在 [weibo_data_audit.json](../results/weibo_data_audit.json)。

从三张原始月表重建请运行 `python src/adapters/weibo_cov.py`。接收现成规范表的成员可以直接运行 `python src/adapters/verify_weibo_handoff.py`。[数据交接说明](../docs/reviews/weibo_cov_handoff.md)

## MisBot

现有基线要求每行一个账号 JSON，含 `uid,label,numerical,categorical,tweet`。官方结构不同应显式适配，不能悄悄换字段或标签。

`data/misbot_manifest.json` 须包含 `file`（本地相对路径）、`sha256`、`source_url`、`version`、`release_kind`（`official` 或 `mirror`）。下载镜像不能代替官方核验。来源说明见 [数据状态](../docs/DATA_STATUS.md)。

当前 manifest 使用 `data/raw/misbot/official_train_data_sampled.jsonl`。可将作者公开的 `User_Instances/train_data_sampled.jsonl` 下载到此路径，运行入口会检查文件哈希。[官方文件](https://drive.google.com/file/d/1XK4X9hp3NZWmOyxIPe_cq00sLEQqH6sk/view)

来源清单/manifest 不包含 cookie、下载签名、个人路径或账号级内容；原始与规范表均留在本地。
