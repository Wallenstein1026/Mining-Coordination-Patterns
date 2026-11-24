# AIAA 3111 · Group 28

**Mining Coordination Patterns in Emerging Social Media Events: CIB-Informed Hot-Topic Clustering and Pattern Mining**

Jiaxin DONG · Qixiang HUA · Xiaofeng LI · Chang XU

本仓库包含当前方法代码、数据接口、验证方案和团队任务。已成功访问官方 Weibo-COV 目录；正常下载提示大于 50 MB 必须使用百度网盘客户端，CSV 尚未取得，尚未生成官方语料结果。原始社交媒体数据不上传 GitHub。

[已完成工作](docs/WORK_DONE.md) · [四人分工与交付](docs/TEAM_TASKS.md) · [方法与验证](docs/METHODS.md) · [数据来源和当前状态](docs/DATA_STATUS.md) · [课程要求对照](docs/REQUIREMENTS_ALIGNMENT.md)

## 主线

1. 按文本、时间和参与特征发现并聚类升温话题。
2. 计算 Human-Manipulation Index (HMI)，结合 MisBot 的辅助风格信号解释协调现象。
3. 使用 Apriori / FP-growth 与时间 motif 挖掘潜在重复模式；有可信关系字段时再分析可观察传播结构。
4. 按时间划分训练/验证/测试，检验 HMI / 模式能否比近期发帖量等基线提高次日热点预测能力。

CIB（Coordinated Inauthentic Behavior）涉及协调和身份/来源伪装；HMI 或机器人分数都不能单独证明 CIB。当前不宣称预测有效或识别了真实操纵账号。

## 当前完成程度

数据清洗、辅助分类、聚类、HMI、项集/时间模式和前向预测的代码原型已编写。之前确实在已下载的四字段微博样本和 MisBot 镜像上执行过实验；这些工作不能写成“没有做实验”，也不能换名当作官方语料结果。该样本按组内决定不进入最终主实验，本包不含它的结果或历史海报。

已保留 [MisBot 镜像辅助基线](results/README.md) 和 [ROC 图](figures/misbot_mirror_roc.pdf)，其官方来源核验仍待完成。

本次仓库检查通过，只代表代码/文件结构可交付；官方语料重跑、预测验证和最终 PDF 仍需要完成。

## 环境与运行

Python 3.12，从仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/check_repository.py
```

Windows 使用 `.venv\Scripts\activate`。数据接口详见 [data/README.md](data/README.md)。填入真实官方文件及本地 manifest 后，可先运行描述性流程：

```bash
python src/run_analysis.py
python src/supplement_analysis.py
python src/proposal_analysis.py
python src/validate_results.py
python src/validate_proposal_results.py
```

只复核 MisBot：`python src/run_analysis.py --stage misbot`。该入口要求实际文件与 `data/misbot_manifest.json`，不会自动把未经核验的镜像叫作官方数据。

前向验证需先冻结 `configs/forecast.json` 的日期，再运行：

```bash
python src/predict_hotspots.py
python src/validate_forecast.py
```

日期尚未固定时入口明确报错。描述性流程使用全期话题筛选和 HMI，其结果只能解释回顾性模式；预测入口独立构造过去可见的候选/特征，不读取全期聚类或未来训练的 MisBot 分数。

## GitHub 上传

当前目录可直接上传至新仓库，也可在此运行以下命令并替换地址：

```bash
git init -b main
git add .
git commit -m "Prepare Group 28 methods and team work packages"
git remote add origin https://github.com/YOUR_USERNAME/aiaa3111-group28.git
git push -u origin main
```

未创建远程仓库、发布 Issue 或推送代码。三个成员任务模板已放入 `.github/ISSUE_TEMPLATE/`；建 Issue 后按实际 GitHub 账号 assign。

## 提交时间与披露

Poster.pdf：80 × 100 cm 竖版，11 月 27 日 23:59 前由一人提交 Canvas；现场展示 12 月 4 日 15:00–16:20，均为北京时间。最终海报和报告由审核后的新结果生成，不在此包中放置过期稿。

现有代码、探索实验、图表及文字由 Codex 协助。成员应亲自检查、运行、理解和修订负责部分，按实际行为记录贡献；计划分工不等于已完成贡献。详见 [WORK_DONE.md](docs/WORK_DONE.md) 和 [RIGHTS.md](RIGHTS.md)。
