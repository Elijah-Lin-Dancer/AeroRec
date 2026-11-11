# AeroRec · 出行智能推荐中台

> 🌐 [English](README.md) | 简体中文

**一套端到端的推荐系统实现——四路召回、DeepFM 排序、五步混排、服务化 API 与双语面板——上面挂着一条研究主线：让曝光偏差变得「可测量」，而不是靠猜。**

[![CI](https://github.com/Elijah-Lin-Dancer/AeroRec/actions/workflows/repro.yml/badge.svg)](https://github.com/Elijah-Lin-Dancer/AeroRec/actions/workflows/repro.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-Streamlit%20Cloud-ff4b4b)](https://aerorec-tourism.streamlit.app)
[![Homepage](https://img.shields.io/badge/homepage-GitHub%20Pages-2563eb)](https://elijah-lin-dancer.github.io/AeroRec/)
[![Preprint](https://img.shields.io/badge/preprint-10.20944%2Fpreprints202610.2582.v2-c7215f)](https://doi.org/10.20944/preprints202610.2582.v2)
[![Zenodo](https://img.shields.io/badge/zenodo-10.5281%2Fzenodo.24003810-blue)](https://doi.org/10.5281/zenodo.24003810)

AeroRec 在一份**可复现的合成数据**上复现工业推荐链路：四路召回 → 加权融合 → DeepFM / LR 多目标排序 → 五步混排 → FastAPI 服务 + 中英双语七页面板。数据生成器是解析式的，因此潜在 CTR 与记录策略的 propensity 都有**闭式真值**——偏差可以精确测量而非估计，本页每个结论背后都有一个数字。

**[在线 Demo](https://aerorec-tourism.streamlit.app)** · **[项目主页](https://elijah-lin-dancer.github.io/AeroRec/)** · **[论文预印本](https://doi.org/10.20944/preprints202610.2582.v2)** · **[技术报告](docs/bias_report.md)**

> **由来**　这个项目起于一段实习。实习期间参与一家航空公司的出行产品项目，我第一次近距离看到推荐系统在真实业务里的约束；实习结束后，我把当时没来得及深究的问题带了回来——曝光偏差到底有多大，什么时候才真的值得去纠。这个仓库就是拿测量而不是直觉去回答它们。

---

## 🎯 为什么看这个项目

1. **跑通的是整条链路，不是一个模型。** 热门 / ItemCF / Two-Tower / 规则四路召回 → 位置分加权融合 → DeepFM 出 CTR、LR 出 CVR → 归一化 / 加权 / 分层 / 打散 / 去重截断 → REST API 与七页面板。`151` 项测试，`ruff` + `mypy` 全绿。
2. **偏差对着已知真值来量。** 有闭式真值，排序损伤就藏不住：随曝光偏斜加剧，Spearman **+0.8665 → −0.5054**，榜单已经反转，而「平均 CTR 看着正常」。去偏存在可测边界 **ESS/N ≈ 0.358**，且这个量不需要真值就能算。
3. **数字逐字节可复现。** 7 组偏差实验在**仅 numpy** 下 **19 秒**跑完；产物在 CI 中重新生成并逐字节比对（`git diff --exit-code`）；论文里 **207** 个数字全部对产物审计，**0 差异**。
4. **能塞进 690 MB 内存。** 训练与推理解耦：离线用 torch，在线用纯 numpy 前向（最大偏差 **1.19e-07**），**服务期不加载 torch**——这一条由 CI 断言。在线：P50 **0.71 ms** / P99 **0.93 ms**，**1379 QPS**，**129 MB** 常驻内存。

## 📸 界面预览

<p align="center">
  <img src="docs/assets/overview_en.png" width="78%" alt="项目总览页">
</p>
<p align="center">
  <img src="docs/assets/demo_en.png" width="49%" alt="实时推荐">
  <img src="docs/assets/bias_en.png" width="49%" alt="偏差诊断">
  <img src="docs/assets/model_en.png" width="49%" alt="模型与验证">
  <img src="docs/assets/bias_zh.png" width="49%" alt="偏差诊断中文界面">
</p>

> 应用默认打开**项目总览**页——一屏讲清「做了什么、四个关键数字、怎么引用」。其余依次是实时推荐、偏差扫描与模型页；最后一张是同一个偏差页的中文界面。全站**中英文一键切换**（侧边栏顶部，默认英文）。

## 🧭 系统架构

```mermaid
flowchart TB
    GEN["合成数据生成器<br/>10万用户 × 10万物品 × 500万曝光<br/>含隐式非线性交叉"]
    REC["召回层<br/>热门 / ItemCF / Two-Tower / 规则"]
    FUS["四路召回融合"]
    RANK["排序层<br/>DeepFM CTR + LR CVR 多目标打分"]
    MIX["5步混排<br/>归一化 / 加权 / 分层 / 打散 / 去重"]
    DEG["降级兜底<br/>异常时切历史平均CTR"]
    SVC["服务层<br/>FastAPI / Streamlit 面板"]
    OUT["Top-K 推荐 可解释"]

    GEN --> REC
    REC --> FUS
    FUS --> RANK
    RANK --> MIX
    MIX --> OUT
    DEG -. 服务异常 .-> RANK
    SVC -- 请求 --> REC
    MIX -- 响应 --> SVC
```

训练与推理解耦：离线用 torch 训练并导出 numpy 权重，在线只跑纯 numpy 前向——**服务期不加载 torch**。

## ✨ 功能总览

- [x] 多场景推荐：航线 / 酒店 / 商城 / 营销 / 直播 五大类目混排
- [x] 热门兜底召回 + 物品协同过滤（ItemCF）召回基线
- [x] Two-Tower 向量召回（PyTorch 双塔 + 负采样）
- [x] 规则召回：里程兑换 / 里程票零库存 / 常驻地热门航线 / 多段行程优先返程
- [x] 四路召回融合（位置分加权 + 降级兜底）
- [x] 轻量特征库：召回 / 排序 / 冷启动共用特征（dense + sparse）
- [x] CTR 动态排序：归一化 → CTR 加权 → 分层排序 → 同品类打散 → 分页去重
- [x] DeepFM 排序模型（PyTorch，FM + DNN）
- [x] 降级兜底：CTR 服务超时 → 历史平均 CTR 兜底
- [x] 离线评估：LR vs DeepFM 的 AUC 对比 + 推荐列表真实 CTR 对比
- [x] 服务化 API（FastAPI：`/health` `/recommend` `/explain` `/ab`）
- [x] Streamlit **七页、中英双语可切换**面板（总览 / 演示 / 偏差诊断 / 排序流程 / 真实 CTR / A/B / 模型）
- [x] 在线推理基准：micro 档冷启动 init ≈ 0.08 s、单线程 P50 ≈ 0.7 ms / P99 ≈ 1 ms、峰值 RSS ≈ 129 MB
- [x] 模拟 A/B（多目标 CTR + CVR，离线反事实重放）
- [x] **真实 CTR 口径评估 + Oracle 上界**（不依赖任何平滑估计，结论可引用）
- [x] **偏差传播诊断 + 去偏验证**（传播矩阵 / IPS·SNIPS·DR / 有效边界 `ESS/N ≈ 0.358`）
- [x] **真实数据迁移性**：Criteo 100 万行 AUC **0.7733**、UCI Adult **0.9183**
- [x] 三档数据规模：`full` / `lite` / `micro`，同一套代码、只差规模
- [x] 在线 Demo：GitHub Pages 项目主页 + Streamlit Community Cloud

## ⚡ 快速开始

```bash
# 30 秒看效果 —— 只需 numpy + streamlit，不装 torch，约 0.3 GB
pip install -r requirements.txt
streamlit run app.py

# 完整复现 —— 追加 torch / scikit-learn / pandas
pip install -r requirements_full.txt
python -m src.eval.evaluate              # LR vs DeepFM 的 AUC + 真实 CTR 口径 + oracle 上界
python scripts/run_bias_experiments.py   # 7 组偏差实验 -> data/bias/*.json（约 19 秒，仅 numpy）
python -m pytest -q                      # 151 项测试
```

| 命令 | 作用 | 依赖 |
|---|---|---|
| `python -m src.pipeline_demo` | 造数 → 四路召回 → 融合 → 出推荐 | 完整依赖，约 500 万行曝光 |
| `python -m src.eval.evaluate` | AUC 对比 + 真实 CTR 排序 + oracle 上界 | 完整依赖 |
| `python -m src.build_online_artifacts micro` | 离线训练并把 numpy 权重导出到 `artifacts/micro/` | torch |
| `uvicorn src.serve.app:app --port 8000` | REST 服务：`/health` `/recommend` `/explain` `/ab` | 完整依赖 |
| `python scripts/run_bias_experiments.py` | 传播 / 注入 / 边界 / 扫描 / 用户侧等实验 | 仅 numpy |
| `python scripts/build_bias_report.py` | 生成自包含 HTML 报告（内联 SVG、零 JS） | 仅 numpy |
| `python scripts/bench_inference.py --mode micro` | 延迟 / QPS / 内存基准 → `data/bench.json` | 仅 numpy |
| `python scripts/audit_paper_numbers.py --strict` | 复核论文全部 207 个数字 | 仅 numpy |
| `python scripts/make_paper_figures.py` | 由产物重画 6 张论文配图 | matplotlib |
| `python scripts/validate_real_data_deepfm.py --dataset adult` | 真实数据迁移性检查（亦可 `--dataset criteo`） | 完整依赖 |

**三档数据规模**，通过 `AEROREC_MODE=micro|lite|full` 切换（默认 `micro`）：

| 档位 | 规模 | 峰值内存 | 用途 |
|---|---|---|---|
| `full` | 10 万用户 × 10 万物品 × 500 万曝光 | 约 1.5 GB | 论文级、可引用结果 |
| `lite` | 2 万 × 2 万 × 100 万 | 约 0.93 GB | 2 GB 实例，可在线训练 |
| `micro` | 1 万 × 1 万 × 50 万 | **约 0.30 GB** | 免费实例（690 MB 下限），离线训练 + numpy 推理 |

三档共用同一套算法代码，只改数据规模与特征存储实现。

## 📊 结果一览

**离线 AUC（合成数据，6 万测试样本）**

| 模型 | AUC |
|---|---|
| Logistic Regression | 0.7421 |
| **DeepFM** | **0.8019**（+6.0 pp） |

**真实 CTR 口径（不依赖任何平滑估计；100 用户 × Top-10）**

| 方法 | 平均真实 CTR | 相对热门 | 占 Oracle 上界 |
|---|---|---|---|
| 随机（下界参照） | 0.1652 | −66.4% | 23.1% |
| 热门（工业常用基线） | 0.4922 | — | 68.8% |
| **DeepFM + 多目标混排** | **0.6323** | **+28.5%** | **88.4%** |
| Oracle 上界（候选池最优） | 0.7153 | +45.3% | 100% |

以上数字全部由「快速开始」的两条命令直接产出（`python -m src.eval.evaluate` 与 `python scripts/run_bias_experiments.py`）。

<details>
<summary>研究主线结果（偏差传播 / 有效边界 / 用户侧）</summary>

- 偏差的首要危害是**排序反转**：曝光偏斜加剧时 Spearman **+0.8665 → −0.5054**（τ = 3，p < 0.001）——平均 CTR 看着正常，榜单已经变了。
- 去偏有效边界 **ESS/N ≈ 0.358**：低于它，方差代价超过纠偏收益；记录数扩 33 倍、种子 30 → 250 后边界只移动 0.005。
- 用户侧自选择须按 user–item 粒度建模：naive 表观 CTR 虚增 **2.5 倍**（0.2811 → 0.7025），按错误粒度去偏留残差 +0.222。
- 完整表格与推导见 [docs/tech-report.md](docs/tech-report.md) 与[自包含可视化报告](data/bias/report.html)。

</details>

<details>
<summary>Oracle 上界怎么构造 / 为什么不用平滑口径 / 模拟 A/B 的边界</summary>

- **Oracle 构造**：池 = 热门候选 ∪ 规则候选 ∪ 3000 个随机物品，在池内按该用户的**真实 CTR** 降序取 Top-K。池必须包含其他方法的结局，否则不构成真正的上界（也不能用全局平均真实 CTR 排序代替——那会退化成热门榜变体）。
- **为什么主结论不用平滑口径**：热门基线按贝叶斯平滑 CTR（α = 5）排序，而日志是随机曝光——平滑值被严重高估（完整档 top200 平滑 0.6231 vs 真实 0.2528；平滑榜 top10 与真实 top10 仅重合 2/10），会高估增益。因此同时给出不依赖平滑的口径，并把平滑口径的失真作为偏差危害的一个实测样本。
- **模拟 A/B 只作流程对照**：离线反事实重放，不含网络效应与反馈循环，仅验证统计流程（两比例 z 检验）实现正确。

</details>

## 🗂️ 目录结构

<details>
<summary>展开目录树</summary>

```text
src/
  config.py / config_lite.py / config_micro.py   三档配置（仅规模不同）
  config_scale.py            通过 AEROREC_MODE 解析档位
  pipeline_demo.py           造数 -> 召回 -> 融合 -> 推荐
  build_online_artifacts.py  离线训练 + numpy 权重导出
  data/                      合成数据生成器（pandas / 纯 numpy）、Criteo、Adult
  feature/                   特征存储（pandas）+ numpy 存储（lite / micro）
  recall/                    热门 / ItemCF 基线、two_tower、规则、融合
  rank/                      deepfm、trainer、numpy_deepfm、numpy_lr、mixer、degrade
  eval/                      metrics、evaluate（AUC + 真实 CTR + oracle）、ab
  serve/                     engine（含 lite / micro）、FastAPI 应用、i18n 文案表
  bias/                      policy、injectors、metrics、propagation、estimators、
                             boundary（ESS 预警）、report（自包含 HTML）
app.py                       Streamlit 演示（七页、双语、默认 micro）
scripts/                     偏差实验、报告、基准、论文配图、数字审计、真实数据验证
tests/                       151 项测试，覆盖 data / rank / api / bias / coat / serve
docs/                        bias_report.md、tech-report.md、paper/、index.html（主页）
data/bias/                   7 份实验产物 + report.html
data/real_validation/        Criteo（100 万 / 20 万）与 Adult 验证结果
artifacts/micro/             预训练 numpy 权重（约 29 KB）
requirements.txt             仅部署所需：numpy + streamlit（Streamlit Cloud 读这份）
requirements_full.txt        torch / sklearn / pandas / fastapi / pytest / matplotlib
```

</details>

## 🧱 技术栈

| 层 | 实现 |
|---|---|
| 数据 | 带解析真值的合成生成器（`full` 用 pandas，`lite` / `micro` 用纯 numpy）；Criteo 与 UCI Adult 加载器用于迁移性检查 |
| 召回 | 热门、ItemCF、带负采样的 PyTorch Two-Tower、出行业务规则、加权融合 |
| 排序 | DeepFM（torch）出 CTR，逻辑回归出 CVR；导出为 numpy 供在线推理 |
| 服务 | FastAPI、Streamlit、纯 numpy 前向（运行期不依赖 torch / sklearn / pandas） |
| 研究 | IPS / SNIPS / DR 估计量、bootstrap 置信区间（B = 10000）、ESS 边界、闭式 propensity |
| 质量 | pytest（151 项）、ruff、mypy、GitHub Actions（产物逐字节零漂移门禁 + 论文数字审计） |

## ❓ 常见问题

<details>
<summary><strong>为什么要用合成数据，而不是公开数据集？</strong></summary>

本项目有两条结论——oracle 上界与无偏真实 CTR 口径——要求已知每个 *(用户, 物品)* 的潜在 CTR。真实日志只给点击（0/1），永远给不出 CTR 本身，所以这两件事在真实数据上原则上做不到。此外公开的出行数据集并不存在（涉及隐私与商业保密），而固定种子让整条链路可以一条命令复现。代价是明说的：合成数据覆盖不了真实噪声、长尾与概念漂移，因此这里的指标衡量的是**受控条件下的方法有效性**，不是线上可迁移性。

</details>

<details>
<summary><strong>在线路径为什么不用 torch？</strong></summary>

Streamlit 免费实例有 **690 MB 内存下限**。实测框架基线：`torch + sklearn + numpy + streamlit` = **673 MB**，只剩 16.7 MB 给数据，必然 OOM；`numpy + streamlit` = **73 MB**。于是训练留在离线，权重导出为 `.npz`（约 29 KB），在线只重跑 numpy 前向。逐样本校验最大偏差 **1.19e-07**，CI 断言 `torch_loaded = false`。

</details>

<details>
<summary><strong>ESS/N ≈ 0.358 是普适阈值吗？</strong></summary>

不是。它是在本项目的合成规模与先验下、用闭式 propensity 测出来的。真实系统得先估计 propensity，这会引入本实验不包含的误差——实际边界会**更严**。能迁移的是方法（信任何去偏估计之前先看 ESS/N），不是这个数字。

</details>

<details>
<summary><strong>混排的基础分权重为什么默认 0？</strong></summary>

因为候选池已经来自热门与规则召回，混排阶段再注入热门等于把偏差重新引回来。真实 CTR 口径实测：`w_base = 0.4` → +6.3%，`0.2` → +17.2%，**`0.0` → +20.1%**。效应的正负取决于注入了什么：与现有信号独立的信息有益，重复的热门信息有害。

</details>

<details>
<summary><strong>怎么复现本页的数字？</strong></summary>

```bash
pip install -r requirements_full.txt
python -m src.eval.evaluate              # AUC 0.8019、真实 CTR 0.6323、oracle 0.7153
python scripts/run_bias_experiments.py   # ESS/N 边界 0.358、Spearman -0.5054
python scripts/audit_paper_numbers.py --strict   # 207 项检查，0 差异
```

偏差这条线只需要 numpy，约 19 秒跑完。AUC 的可复现依赖一个细节：DeepFM 的权重初始化必须排在 `torch.manual_seed(SEED)` **之后**，否则 AUC 会跨环境漂移约 0.004。

</details>

<details>
<summary><strong>预印本 v1 到 v2 改了什么？</strong></summary>

数值只有一处：小规模去偏边界原报 `ESS/N = 0.3576`，该值无法由已发布代码复现；同配置重跑已提交的流水线得到 `0.3625`（正文与图注各出现一次）。另补了一句规模说明与作者贡献声明。结论——边界对流水线规模稳健——不变。v1 保留独立 DOI 继续在线，是该值的原始记录，对应仓库状态打了 `preprint-v1` 标签。

</details>

<details>
<summary><strong>Demo 需要 GPU 或付费实例吗？</strong></summary>

不需要。免费档就是设计目标：带浏览器驱动实测峰值约 297 MB，冷启动约 1 秒，请求时不训练。本地推理基准为 P50 0.71 ms / P99 0.93 ms、129 MB 常驻内存；免费实例会慢一些，但同一量级。

</details>

## 📏 诚实边界

主动列出，避免被误读：

- 数据为**本项目合成**（受控实验），只用于演示工程链路与测量方法；不涉任何真实业务数据，不与任何真实企业系统关联。
- 生成公式注入隐式非线性交叉项，让深度模型有东西可学；模型只拿到原始特征、交叉需自行学习——这是明确的受控实验设置，不是对真实流量的断言。
- Two-Tower / DeepFM 等均为已有方法的标准实现，本项目**不提出新算法**。
- 实习经历仅作为动机来源（业务规则与流程启发），实习中未涉及深度排序模型的训练。

<details>
<summary><strong>已知局限（8 条）</strong></summary>

1. **合成指标向真实出行流量的外推仍未证明**：迁移性已在 Criteo（AUC 0.7733）与 UCI Adult（0.9183，同数据 LR 基线 0.9113）上验证，但两者都不是出行域流量。
2. **Criteo 上深度交叉无可见优势**：DeepFM 0.7733 vs LR 0.7716（差 0.0017）；20 万行 3 种子 0.7586 ± 0.0007。只声称「同一实现能在真实 CTR 数据上学到强信号（≫ 0.5）」，不声称深度模型带来增量——Criteo 以高基数稀疏特征为主，线性模型本就接近强基线，与文献观察一致。
3. **A/B 是离线反事实重放**，不含网络效应、用户反馈循环与长短期收益差异。
4. **出行业务规则的收益未经真实数据检验**（里程兑换、多段行程优先等规则在合成数据上自洽）。
5. **训练预算必须按「梯度更新次数」设定**：固定 epoch 会让小档位静默欠拟合（1.2 万样本 × 2 epoch ≈ 12 次更新时 AUC 仅 0.5109）；正确做法是用 `plan_epochs(target_updates=400)` 按目标更新次数反推 epoch。
6. **本项目的贡献是「工程链路复现 + 评估口径的诚实性」，不是算法创新。**
7. **偏差实验的 propensity 闭式已知，边界数值不可迁移**：真实系统必须估计 propensity，会引入本实验没有的误差，实际边界会更严苛；可迁移的是方法（用 ESS/N 作预警），不是 0.358 这个数。
8. **实验规模有限、未覆盖用户侧偏好偏差**：传播矩阵 500 物品 × 5000 请求 × 5 坑位（30 种子），边界实验 2 万记录（250 种子）；仅建模曝光/位置偏差，未涉及同一用户内跨物品的选择偏差。

</details>

## 📝 更新日志

| 版本 | 日期 | 要点 |
|---|---|---|
| `v1.1.0` | 2026-11-06 | 首个 GitHub Release；归档至 Zenodo（概念 DOI + 版本 DOI）；已发布标签由仓库规则集锁定为不可变 |
| `v1.0.0` | 2026-11-05 | 论文数字全量审计与溯源文档；预印本 v1 线（2026-10-27 发布） |
| `v0.2.0` | 2026-10-15 | 7 组偏差实验的锚点——论文各表均可由该标签复现 |

含逐周里程碑的完整历史见 [CHANGELOG.md](CHANGELOG.md)。

## 📄 研究成果

<details>
<summary>预印本、DOI 与引用格式</summary>

> **Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study** — Yulin Zhao，*Preprints* 2026, 202610.2582（v2 发布于 2026-11-06；v1 发布于 2026-10-27）。14 页、6 图 8 表，CC BY 4.0。

- 预印本 v2：[10.20944/preprints202610.2582.v2](https://doi.org/10.20944/preprints202610.2582.v2) · [稿件页](https://www.preprints.org/manuscript/202610.2582/v2) · [PDF](docs/paper/AeroRec-preprints.org-v2.pdf)
- 预印本 v1：[10.20944/preprints202610.2582.v1](https://doi.org/10.20944/preprints202610.2582.v1) · [PDF](docs/paper/AeroRec-preprints.org-v1.pdf)
- 软件归档：Zenodo [概念 DOI](https://doi.org/10.5281/zenodo.24003810)（始终指向最新）· [版本 DOI](https://doi.org/10.5281/zenodo.24003811)（`v1.1.0`，`preprint-v2` 的逐字节快照）
- 论文各表各图到产物的逐条溯源：[docs/paper/PROVENANCE.md](docs/paper/PROVENANCE.md)

```bibtex
@article{zhao2026aerorec,
  title   = {Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study},
  author  = {Zhao, Yulin},
  journal = {Preprints},
  year    = {2026},
  number  = {202610.2582},
  doi     = {10.20944/preprints202610.2582.v2}
}

@software{zhao2026aerorec_zenodo,
  title   = {AeroRec: an oracle-based diagnosis platform for bias propagation and debiasing boundaries in recommender systems},
  author  = {Zhao, Yulin},
  year    = {2026},
  version = {1.1.0},
  doi     = {10.5281/zenodo.24003811},
  license = {MIT}
}
```

</details>

## 数据与许可

| 数据 | 是否入库 | 条款 |
|---|---|---|
| 合成数据（`data/processed*`、`data/raw*`） | 否——一条命令重新生成 | 本仓库生成，随仓库以 MIT 发布 |
| UCI Adult（`data/adult/`） | 否——按需加载 | 仅限研究用途；引用 Kohavi & Becker (1996) |
| Criteo 100 万样本（`data/criteo/`） | 否——需自行获取 | CC-BY-NC-SA 4.0（非商业） |
| 验证结果（`data/real_validation/*.json`） | 是 | 仅聚合指标，MIT |

真实原始数据一律不入库，只提交验证结果 JSON 作为证据。

## 许可

MIT — 见 [LICENSE](LICENSE)。
