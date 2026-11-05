# Changelog

本项目所有显著变更记录于此。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号在首次正式发布前以里程碑（W1–W11 = 第 1–11 周）+ 日期标识。

## [Unreleased]

### Added
- **论文数字溯源审计**：新增 `docs/paper/PROVENANCE.md`（8 张表 / 6 张图 → 产物文件的
  逐项对应 + 冻结文件清单）与 `scripts/audit_paper_numbers.py`（逐单元格比对，支持
  `--strict` 接 CI）。另用内嵌位图比对确认仓库 6 张图与已发布 PDF 的 Figure 2–6 一一对应
  （MAE ≈ 0.2）。首次全量核对 207 项中有 1 项已知差异（§4.3 的 `0.3576`），该差异已在下条
  修复，现为 **207 项全部一致、0 豁免**。
- **软件归档元数据**：新增 `CITATION.cff`（GitHub 侧栏 "Cite this repository" 可直接
  产出 BibTeX，并由 `preferred-citation` 指向预印本）与 `.zenodo.json`（Zenodo 归档页
  标题/作者/许可/关键词，并通过 `related_identifiers` 与预印本 DOI 双向绑定）。
  代码与结果文件与 `preprint-v1` 标签逐字节一致，本次仅新增元数据。
- **论文预印本正式发布**：Preprints.org V1（posted 2026-10-27），
  DOI [10.20944/preprints202610.2582.v1](https://doi.org/10.20944/preprints202610.2582.v1)；
  发布版 PDF 归档于 `docs/paper/AeroRec-preprints.org-v1.pdf`（平台排版版，
  14 页，CC BY 4.0），README（中/英）新增 DOI 徽章、发布展示区、
  引用格式与 BibTeX（`zhao2026aerorec`）。
- **Streamlit 新增「项目总览」页（Overview），并设为默认落地页**（全站第 7 页）：
  一屏回答「这个项目做了什么 / 四个关键数字（+28.5% 真实 CTR、88.4% Oracle 上界、
  ESS/N 0.358、Criteo AUC 0.7733）/ 研究主线三条结论 / 为什么值得在意 / 如何引用」，
  含 BibTeX 与诚实边界声明。动机：多数访客既不读论文、也不翻仓库源码。
- **偏差诊断页扩为两个标签**：「曝光扫描（τ）」保留原交互；新增
  「去偏有效边界（ESS/N）」标签，把 `data/bias/boundary.json` 的 6 档边界直接画成
  表格 + 曲线（含容差三态判定与 fail-safe 稳健性说明），页面不手打任何边界值。
- **模型与验证页新增两节**：第二个真实数据集 UCI Adult（AUC 0.9183 / LR 0.9113 /
  48,842 行）与「复现性是被强制的」区块（133 测试、7 JSON 逐字节门禁、
  零重依赖断言、内存纪律断言、显式播种）。
- **侧边栏与页脚新增预印本入口**（DOI · 出版社页面 · 发布版 PDF），页头正下方加
  「已发表预印本」条 —— 首屏即见，不再只藏在 README 里。
- **论文 6 图入库** `docs/paper/figures/`，配套 `scripts/make_paper_figures.py`
  （直接读 `data/bias/*.json`，固定种子，可逐字节重生成；dpi 300–340 满足
  ChinaXiv 有效 dpi ≥300）与 `docs/paper/README.md`（材料清单 + 图↔产物同源说明）。
- **Pages 主页补两项扩展实验**：用户侧偏差的 user-item 粒度结论（β 扫描表）与
  propensity 估计误差的稳健性（Δ=−0.0022，p=0.87，零漏报），并嵌入论文图 2/图 3。

### Changed
- **仓库主 README 语言切换为英文**：英文版升为主 `README.md`，
  中文版移至 `README.zh.md`；两个版本的语言切换链接、
  Documentation 表与引用点同步更新，英文主 README 新增
  徽章行（含 DOI）与 "Published Preprint" 展示区。
- **英文主 README 由摘要版补齐为完整版**（154 → 622 行）：新增截图、背景、21 项功能
  清单、快速开始全命令、数据与复现（为什么合成数据 / 数据许可表 / 复现性 CI /
  三档规模 / 训练推理解耦长文 / 推理基准）、方法与四组离线结果、目录结构、
  诚实边界与已知局限，与中文版信息对等；中文版保留更详尽的工程论证。
- **Pages 主页与文档同步到最新状态**：导航由「六页」改「七页」（含项目总览）、
  测试数 107 → 133、Adult 对照列补 LR 基线、页脚加 DOI。
- `docs/tech-report.md`、`docs/bias_report.md`、`docs/research_highlight.md`
  均在开头标注已发表预印本与 DOI；`bias_report.md` 的产物清单由 3 个更正为 7 个。
- 演示截图全部重截（`docs/assets/`，2720×3000 与旧图同规格，隐藏 Streamlit 工具栏），
  并新增 `overview_en.png`；两版 README 的截图区改为「总览页 1 张大图 + 4 张 2×2」。

### Fixed
- **论文 §4.3 与 Fig. 3 图注的跨规模边界值（`0.3576` → `0.3625`）**：预印本 v1 报告
  小规模配置（60 物品 × 600 请求 × 30 种子）下边界为 `0.3576`，该值来自早期实现的一次
  运行、当时未落盘，仓库内无产物支撑，当前代码按同配置复算为 `0.3625`。v2 更正稿已把
  正文与图注改为 `0.3625`，并补上量化依据——两个配置落在同一网格档（π_ratio = 40），
  差 0.005，而相邻网格点的 ESS/N 间距为 0.29，故"对规模稳健"的结论成立。
  `audit_paper_numbers.py` 的 `EXPECTED_GAPS` 随之清空。
- **审计脚本会给自己发豁免**：原逻辑"复算值与论文相差 < 0.01 即自动登记为已知差异"，
  等于让不合格项永久通过。现改为严格比对 + `2×10⁻⁴` 浮点容差，超差即判不一致。
- **交互偏差页 τ=0 处的 L1 伪相关**：τ=0 时 L1 估计 `tmean·C·π` 是常数向量，
  `_rank` 的稳定排序把「全等值」排成物品下标，与真值求秩相关得出纯伪相关
  （60 物品、单种子下约 +0.176，图上表现为一条陡降红线）。现对常数估计把
  spearman 约定为 0（数学上常数序列不携带排序信息）。**只改交互展示**，
  不动 `src/bias/propagation.py` 的统计口径，故 `data/bias/*.json` 与论文数字
  逐字节不变（落库 500 物品/30 种子版该值为 −0.0197，本就近零）。
- **边界表判定与增益量级不符**：原按 gain 正负二分会把 gain=−0.0001 报成
  「去偏有害」、与 ESS/N=0.905 自相矛盾；改为带 `tolerance_used`（0.001）的
  三态判定（有益 / 无显著差异 / 有害），边界档单独标注。
- **中文 README 两处指向不存在章节的引用**（「见下『真实数据验证』一节」、
  「见『真实数据验证』一节」），改指实际存在的「数据许可」小节与结果 JSON。
- **中文 README 边界表漏 `[0.002, 2.0]` 一档**（JSON 有 6 行、文档只列 5 行）。
- **英文 README 的 `[README.md](README.md)` 自链**（License 节）改为指向本文档的
  Data licensing 小节。
- **CHANGELOG 图数描述** 4 图 → 6 图。

## [0.2.0] - 2026-10-15

论文预印本版本锚点：**本 tag 包含论文全部 7 个偏差实验的可复现代码**
（v0.1.0 仅含 propagation / injection / boundary 三实验，论文中
Table 5–8 对应的 4 个实验在其后加入）。

### Added
- 扩展实验（改进计划 #4/#5/#6）：`src/bias/pi_estimation.py`、
  `src/bias/userbias.py` → `data/bias/propensity_estimation.json`、
  `data/bias/userbias.json`；
- 历史补测落盘：`estimand_check.json`、`boundary_sweep.json`；
- 论文预印本 PDF（`docs/paper/AeroRec-preprint.pdf`，**6 图 8 表**：另有框架图与
  用户侧偏差图）及生成图 `docs/paper/figures/`（数据直读产物 JSON，可逐字节重生成）。

### Changed
- 零漂移门禁扩至 7 个产物 JSON；
- `docs/tech-report.md`、`docs/research_highlight.md` 代码锚点同步 v0.2.0。

## [0.1.0] - 2026-10-15

首个版本锚点（tag `v0.1.0`）。自里程碑 W11 以来的主要变更：

### Added
- **偏差研究扩展实验**（改进计划 #4/#5/#6）：propensity 估计误差
  （`data/bias/propensity_estimation.json`——日志估计 π̂ 替代闭式真 π，
  本规模下 SNIPS 误差变化不显著、ESS 预警 fail-safe）与用户侧选择偏差
  （`data/bias/userbias.json`——user-item 粒度自选模型，量化「粒度决定
  去偏成败」：物品边际去偏残余 +0.06~+0.22，正确粒度 0.004–0.037）；
- **bootstrap 显著性检验**：`src/bias/significance.py`（numpy-only，B=10000
  固定种子），多种子指标全量落盘 `*_std` 与 `significance` 字段；
- **历史补测落盘**：`estimand_check.json`（§4 历史口径）与
  `boundary_sweep.json`（τ×N 网格），[待补测] 清单清零；
- **扩规模新基线**：items 500 / requests 5000（方案 A 重跑），文档全量对齐；
- **文档**：`docs/tech-report.md`（面向论文读者的技术文档）、
  `docs/bias_report.md` §8 扩展实验章节。

### Changed
- 传播/注入 RNG 统一为 CRN + 单次模拟解析扫描（消除伪顶点/伪回撤伪影）；
- 零漂移门禁扩至 7 个产物 JSON（repro.yml）；
- README 移除工程台账（Roadmap）聚焦展示，研究主线入口补
  `tech-report.md`。

### Fixed
- `validate_real_data*.py` 显式限制 torch 线程数（受限 CPU 配额下防超订）；
- 文档数字与产物全面对齐（历史多处文档↔产物偏离已修正）。

### 新增
- **测试纳入 CI**：`repro.yml` 新增 `tests` job——安装完整依赖（numpy 锁定 2.3.5）跑
  `pytest` 全量测试 + `pytest-cov` 覆盖率报告；真实数据（Criteo / Adult）在文件缺失时自动 skip。
- **静态检查纳入 CI**：新增 `lint` job 跑 `ruff`（E/F/I/UP/RUF100 规则集）与 `mypy`（46 个源文件 0 问题）；
  规则集聚焦「能抓真问题、可全量通过」，配置见 `pyproject.toml`（新增）。
- **在线引擎测试**：`tests/test_mixer.py`（5 步混排：去重/打散/加权）与
  `tests/test_online_engine.py`（micro 引擎召回组件 + 加载已入库工件的端到端 `recommend` / `benchmark_rankers`）。
- **API 层测试**：`tests/test_api.py`——用桩引擎验证参数校验（400）、引擎冷启动失败（503）、
  内部错误不泄露栈（500）。
- **FastAPI 异常处理**（`src/serve/app.py`）：此前四个端点**零异常处理**，越界 `user_id` 会直接 500
  并泄露内部栈。现约定：400 = 参数不合法（user_id 越界 / top_k 非法）、503 = 引擎未就绪、
  500 = 内部错误（服务端记完整栈，客户端只拿通用信息）。
- **真实数据多种子误差棒**（`scripts/validate_real_data_deepfm.py`）：新增 `--seeds N`，
  >1 时重复训练并报告 DeepFM 验证 AUC 的 mean±std 与逐种子值；
  种子序列为 `SEED..SEED+N-1`，故 `--seeds 1` 与历史单种子结果逐字节可复现。
  实测（Criteo 200k，seeds 42/43/44）：AUC **0.7586 ± 0.0007**（逐种子 0.7583 / 0.7593 / 0.7581，
  见 `data/real_validation/result_criteo_200k_3seeds.json`）——种子 42 与既有单种子结果 0.7583 逐位一致。
  相应地 `DeepFMRanker` 新增可选 `seed` 参数（默认项目基准种子，行为不变）。
- **数据许可标注**：README「数据与复现」新增「数据许可」小节（合成数据 MIT / UCI Adult 引用要求 /
  Criteo CC-BY-NC-SA 4.0 / 验证 JSON 仅含聚合指标）。
- **CHANGELOG**（本文件）。

### 修复
- **`recommend(top_k=...)` 参数被忽略**（`engine.py` / `engine_lite.py` / `engine_micro.py` 三处同源）：
  `top_k` 被计算后从未传给 `Mixer`（其用构造时的 `self.top_k` 截断），导致 API 的 `?top_k=` 形同虚设。
  现按 `top_k` 截断最终列表；默认 `top_k=self.top_k` 时行为与之前逐字节一致。
- **测试过严导致的确定性红灯**：`test_mixing_injection_monotone_in_w_base` 断言 `w_base` 逐点严格单调，
  但 `mixing_second_injection` 每个 `w_base` 用独立 RNG 重抽曝光，相邻点 L2 噪声不同，
  局部小幅回撤属抽样噪声而非机制失效（实测 `[0.0032, 0.2606, 0.2097, 0.947, 1.0]`）。
  放宽为趋势断言：端点确定性（`w=1 ⇒ spearman=1.0`）+ 整体上行（`sp[-1] − sp[0] > 0.5`）。
  该测试此前长期潜伏为红灯——因为 CI 此前不跑 pytest，只有零漂移门禁。

### 移除
- `requirements_streamlit.txt`：与根目录 `requirements.txt`（部署专用锁定版）内容重叠，零引用，删除。

---

## [W11] 2026-10 — 真实数据迁移性验证（路径 B）

- Criteo Display Advertising Challenge（13 稠密 + 26 稀疏 → p(click)）100 万行：DeepFM 验证 AUC **0.7733**；
  20 万行档 **0.7583**；UCI Adult **0.9183**。同一 `DeepFMRanker` 实现，模型零改动。
- 如实记录：Criteo 上深度交叉项未显现明显优势，以高基数稀疏特征为主，LR 基线已接近——**不声称**深度交叉在真实数据带来增量。
- 结果 JSON 入库（`data/real_validation/`），原始数据不入库。

## [W9–W10] 2026-10 — 研究主线：偏差传播诊断 + 去偏验证

- M1–M7：曝光策略与闭式 propensity（`policy.py`）→ 偏差度量（`metrics.py`）→
  链路传播矩阵（`propagation.py`：L0 真值 / L1 召回 / L2 排序 / L3 混排）→
  IPS / SNIPS / DR 去偏（`estimators.py`）→ 有效边界扫描（`boundary.py`：**ESS/N 是无需真值的去偏预警指标**）→
  HTML 报告（`report.py`）+ `docs/bias_report.md`。
- 关键认识（均实测修正）：① 偏差的大头不是整体偏高而是**排序被判反**（spearman +0.83 → −0.47）；
  ② IPS 纠正的是**估计目标**（T_pop vs T_exp），不是"坏估计"；③ π 的两种约定（包含概率 vs 分布概率）不可混用；
  ④ 去偏亏本的真正驱动是 **π 动态范围（ESS）**，不是偏差幅度。

## [W8] 2026-10 — 可复现性修复 + 三档部署形态

- 修复指标不可复现的根因（权重初始化未在播种之后），同步全部文档数字。
- 训练预算改为**锚定梯度更新次数**，修复 lite 档欠训（AUC 0.51 → 0.80）。
- 三档形态：full（10 万×10 万×500 万，离线）/ lite（2 万×2 万×100 万）/ micro（在线，加载 numpy 权重）。
- micro 档离线导出：DeepFM/LR 权重 → `artifacts/micro/*.npz`（29KB），在线纯 numpy 前向，
  PyTorch vs numpy 最大偏差 1.19e-07。

## [W1–W7] 2026-09 — 链路主体与 Demo

- 数据 / 特征层 → 召回（热门 / ItemCF / Two-Tower / 规则 / 融合）→ 排序（DeepFM + 多目标 CTR/CVR）→
  5 步混排 → 降级兜底 → 离线评估（AUC / Recall@k）→ 模拟 A/B + **真实 CTR 口径对照（含 Oracle 上界）** →
  `RecommendationEngine` + FastAPI + Streamlit 交互面板（中英双语、偏差诊断页、推理基准）。
- 部署决策：最终采用 Streamlit Community Cloud（HuggingFace Spaces 经实测未采用，原因见 README「部署决策」）。

## [CI] 2026-10 — 复现性流水线

- `repro.yml`：① numpy-only 重跑偏差实验 + `git diff --exit-code` **零漂移硬门禁**；
  ② 在线引擎基准 + 断言 `torch_loaded = false`；③（本次新增）pytest 全量 + 覆盖率；④（本次新增）ruff + mypy。
- 依赖锁定：`requirements.txt`（在线，精确版本）与 `requirements_full.txt`（开发/复现，主版本上限）分离。
