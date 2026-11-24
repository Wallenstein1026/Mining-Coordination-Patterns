# 数据状态与获取记录

截至 2026-11-24。数据文件存在、来源确认、实验运行和结果有效是不同的状态。

| 项目 | 目前可核实状态 |
|---|---|
| 先前微博样本 | 本地已有 48,617 条四字段数据，来源记录为 bys1417/weibo-cov；确实执行过实验，按组内决定退出最终主实验 |
| 官方 nghuyong Weibo-COV 2.0 | 已用公开提取码成功打开官方网盘目录，读取月度文件清单；已尝试正常下载最小的 2019-12.csv（60,624,172 bytes），页面要求 >50 MB 使用百度网盘客户端；未取得正文 |
| MisBot 活跃用户镜像 | 已按哈希核验并复跑账号独立基线，已核对上游标签定义；官方文件的完整哈希一致性仍待确认 |
| 官方数据上的挖掘/预测结论 | 需要实际 CSV、来源/选择审计、切分配置与正式重跑；当前不把先前样本的数字移作官方结果 |

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

镜像文件 SHA-256 为 `01d8170eb494367694c3403bb33b702eca19c35469c56df529c3e025778b7e16`。上游同名文件大小与镜像一致，已下载的文件前缀也一致。使用镜像得到的基线评估的是 MisBot 活跃账号标签，其指标适用范围见 [MisBot 审计](reviews/misbot_audit.md)。

## 实际阻塞与下一步

2026-11-24 通过公开提取码打开分享，成功列出官方文件并单独选中 2019-12.csv。正常下载按钮最终提示：“下载内容超50mb，请使用百度网盘客户端保存”。该目录最小 CSV 为 60,624,172 bytes，所有文件都超过网页提示阈值。未保存 CSV，不能据此运行官方实验。

需要通过百度网盘客户端取得文件，并存入本地 `data/raw/original_weibo_cov/`；数据本身无需上传 GitHub。优先按事先固定的连续月份取得研究窗口，审计覆盖后再冻结时间切分。官方目录清单、已成功的数据访问与失败的文件下载分开记录，不能互相替代。

## 获取限制

只记录实际尝试及结果；登录、验证码或客户端要求出现时按正常流程处理，不把 HTML 网页当 CSV。仅将不含 cookie、签名、个人账号的数据来源记录入库。最终评估不依赖预制验证集，而应按任务构造时间切分。
