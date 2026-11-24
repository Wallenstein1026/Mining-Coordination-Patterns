# 本地数据接口

原始文件放 `data/raw/original_weibo_cov/`、`data/raw/misbot/`，规范化研究子集放 `data/processed/weibo.csv`；这些目录被 Git 忽略。不要上传社交媒体原文、用户标识或模型。

## Weibo

规范 CSV 必需列：`mid,user_id,created_at,content`；`created_at` 为北京时间、无时区字符串。官方 `_id` 映射到 `mid`，保留 `origin_weibo`、互动计数与缺失标记等额外列。根帖引用不默认是逐跳父节点。

`data/weibo_manifest.json` 是实际取得文件后的来源记录，当前不预造。至少包含：

```json
{
  "dataset": "nghuyong/Weibo-COV",
  "version": "2.0",
  "source_url": "https://github.com/nghuyong/weibo-cov",
  "normalized_file": "data/processed/weibo.csv",
  "normalized_sha256": "填写实际规范CSV的64位SHA-256",
  "timezone": "Asia/Shanghai",
  "selection": "填写预先固定的日期范围与选择规则",
  "raw_files": [{"filename": "实际文件名", "bytes": 0, "sha256": "填写实际值"}]
}
```

源文件可有多个，实际大小、计数、清洗规则和关系覆盖在 `docs/reviews/weibo_cov_audit.md` 与聚合审计中记录。不要直接复制占位内容作为已经审计的证明。

## MisBot

现有基线要求每行一个账号 JSON，含 `uid,label,numerical,categorical,tweet`。官方结构不同应显式适配，不能悄悄换字段或标签。

`data/misbot_manifest.json` 须包含 `file`（本地相对路径）、`sha256`、`source_url`、`version`、`release_kind`（`official` 或 `mirror`）。下载镜像不能代替官方核验。来源说明见 [数据状态](../docs/DATA_STATUS.md)。

来源清单/manifest 不包含 cookie、下载签名、个人路径或账号级内容；原始与规范表均留在本地。
