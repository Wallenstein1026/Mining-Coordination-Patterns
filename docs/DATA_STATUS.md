# 数据状态与获取记录

更新于 2026-11-26。

| 项目 | 目前可核实状态 |
|---|---|
| 先前微博样本 | 曾取得并测试 48,617 条四字段数据（已退出主实验），来源记录为 bys1417/weibo-cov；确实执行过实验，按组内决定退出最终主实验 |
| 官方 nghuyong Weibo-COV 2.0 | 官方 Weibo-COV 2.0 的 2020 年 9–11 月三份月表已导入。原始 8,149,927 条，按预先固定的帖子 ID 哈希桶选取约 1/32，规范表保留 254,624 帖、199,851 个用户，覆盖 91/91 天。 |
| MisBot 活跃用户 | 已取得官方文件并确认与镜像逐字节一致，已用官方文件复跑账号独立基线 |
| 官方数据上的挖掘/预测结论 | Chang 已完成聚类、模式与前向预测，最终整合已收到真实规范表并独立复跑，详见 [最终审查](FINAL_REVIEW.md)。 |

## 官方 Weibo-COV

官方仓库：<https://github.com/nghuyong/weibo-cov>

- 1.0：<https://pan.baidu.com/s/1SwbkEnuXrUFmRj1lx_AQlg?pwd=r8gn>
- 2.0：<https://pan.baidu.com/s/1mxU5RbnGBNRvR4Ci-9d0Hg?pwd=jffm>，提取码 `jffm`。
- 2.0 目录有 2019-12 至 2020-12 的月度 CSV，以及 `user.csv`。公开文件大小清单见 [official_access.json](../data/official_access.json)。清单不是语料内容，不代表下载完成。
- 官方 README 声明关键词过滤微博覆盖 2019-12-01 至 2020-12-30。使用哪批文件、日期、计数与哈希须以实际下载审计为准。
- `origin_weibo` 可能提供根帖引用，不能未经审计就当完整逐跳传播树。

1.0 分享目录也已核对，提供的 `weibo-cov.csv.tar.gz` 约 9.38 GB、`all-geo.csv.tar.gz` 约 5.09 GB，同样不是可从网页取得的小文件。

## MisBot

官方：<https://github.com/whr000001/MisBot>

官方文本入口：<https://drive.google.com/drive/folders/1IfXTZjYHkgoolY1s8f20EdbG95L-1qb2?usp=sharing>

当前已取得的镜像来源：<https://github.com/VladPlusIn/Research_Code/tree/main/MisBot/MisBot/User_Instances>

官方文件和镜像均为 68,846,699 字节，SHA-256 均为 `01d8170eb494367694c3403bb33b702eca19c35469c56df529c3e025778b7e16`。逐字节比对通过。基线评估的是 MisBot 活跃账号标签。[MisBot 审计](reviews/misbot_audit.md) · [官方文件](https://drive.google.com/file/d/1XK4X9hp3NZWmOyxIPe_cq00sLEQqH6sk/view)

## 获取历史与当前版本

2026-11-24 通过公开提取码打开分享，成功列出官方文件并单独选中 2019-12.csv。正常下载按钮最终提示：“下载内容超50mb，请使用百度网盘客户端保存”。该目录最小 CSV 为 60,624,172 bytes，所有文件都超过网页提示阈值。未保存 CSV，不能据此运行官方实验。

2026-11-25，Qixiang 提供三份已下载月表。服务器文件与下载端 SHA-256 逐份一致，抽样与质量复核已完成。[正式数据审计](reviews/weibo_cov_audit.md) · [Chang 交接说明](reviews/weibo_cov_handoff.md)

## 获取限制

只记录实际尝试及结果；登录、验证码或客户端要求出现时按正常流程处理，不把 HTML 网页当 CSV。仅将不含 cookie、签名、个人账号的数据来源记录入库。最终评估不依赖预制验证集，而应按任务构造时间切分。
