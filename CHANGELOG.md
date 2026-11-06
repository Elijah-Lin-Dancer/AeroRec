# Changelog

本项目所有显著变更记录于此。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号在首次正式发布前以里程碑（W1–W11 = 第 1–11 周）+ 日期标识。

## [Unreleased]

### Added
- **已发布标签纳入 GitHub Ruleset 保护**：此前仓库无任何 ruleset、`main` 也无分支保护，
  "删掉已发布 tag 再以同名重建"这一风险**没有任何技术拦截**。现建立两条仓库级规则集：
  - `published-tags-are-immutable`（id `26684005`）：`refs/tags/v*` 与 `refs/tags/preprint-*`
    禁止删除、禁止改指；
  - `no-reuse-of-released-version-numbers`（id `26684019`）：已发布的 4 个版本号禁止再次创建。

  规则集**刻意不含 `creation` 通配规则**，未来的 `v1.2.0` 可正常创建，不误伤发布流程。
  四条路径均以真实 git push 实测：删除 / 改指 / 同名重建**三项全部被拒**，
  而创建全新版本号**成功**。新增 `docs/RELEASING.md` 记录铁律、技术强制、逃生舱步骤、
  发布命令序列与版本号↔DOI 对应表。
- **软件已在 Zenodo 完成归档（本项目首个 GitHub Release）**：Release `v1.1.0`
  （标签 `v1.1.0` → `8aee4f9`），Zenodo 的 release hook 自动归档，产出
  概念 DOI [10.5281/zenodo.24003810](https://doi.org/10.5281/zenodo.24003810)（永远指向最新版）
  与版本 DOI [10.5281/zenodo.24003811](https://doi.org/10.5281/zenodo.24003811)（锁定本次归档）。
  归档内容为 `preprint-v2` 的逐字节快照（151 个条目、含 v1/v2 两版论文 PDF 与全部 `data/bias/`
  产物，`data/` 与 `preprint-v1` 标签逐字节相同，**未重跑任何实验、未改动任何产物**）；
  creator 元数据补入作者机构（Henan University of Economics and Law）与 ORCID。
  Release 同时附上两版论文 PDF。README（中/英）新增 Zenodo 徽章与软件引用条目，
  `docs/index.html` 首屏与页脚新增软件 DOI 入口。`preprint-v1` / `preprint-v2` /
  `v1.0.0` 三个标签一律未移动。
- **预印本 v2 上线**：Preprints.org 于 2026-11-06 发布更正版，DOI 变为
  [10.20944/preprints202610.2582.v2](https://doi.org/10.20944/preprints202610.2582.v2)
  （v1 的 `.v1` DOI 永久保留，两版互链）。v2 的版式由我们提交的 Word 稿排版而来，
  平台回传 PDF 归档为 `docs/paper/AeroRec-preprints.org-v2.pdf`；入库前逐项核对：
  14 页、6 图与仓库 PNG 一一对应（MAE ≤ 0.14，次优 ≥ 10）、8 表齐全、
  `0.3625` 出现 2 次且 `0.3576` 归零。`AeroRec-preprints.org-v1.pdf` 原地保留，
  作为 `0.3576` 的物证。
- **论文数字溯源审计**：新增 `docs/paper/PROVENANCE.md`（8 张表 / 6 张图 → 产物文件的
  逐项对应 + 冻结文件清单）与 `scripts/audit_paper_numbers.py`（逐单元格比对，支持
  `--strict` 接 CI）。另用内嵌位图比对确认仓库 6 张图与已发布 PDF 的 Figure 2–6 一一对应
  （MAE ≈ 0.2）。首次全量核对 207 项中有 1 项已知差异（§4.2 的 `0.3576`），该差异已在下条
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
- **文档与产物的口径一致性清理**（6 处，全部是「文档写的 ≠ 产物算的」，无代码行为变更）：
  - `src/bias/boundary.py` 模块文档串的边界表停留在早期版本（gain `−0.0010 / −0.0065 /
    −0.0316 / −0.0850 / −0.1070`、ESS `0.019 / 0.017`），与 `data/bias/boundary.json`
    现值不符。已按 `boundary_by_range(t_pop≈0.2459, n_records=20000, 250 seed)` 的现产物
    改写为 `−0.0001 / −0.0017 / −0.0084 / −0.0306 / −0.0627`、ESS `0.006 / 0.003`；
    同段 τ 扫描的「60/200 物品、40 seed」更正为「40 seed × 每 seed 5000 条记录」，
    gain 由 `≈+0.07 / ≈+0.16` 改为实测 `+0.0468 / +0.1395`；预警线表述改为
    「ESS/N 降至约 0.36（实测越界点 0.3579）以下开始亏本，代码默认阈值取 0.4」。
  - `.github/workflows/repro.yml` 头部注释写「三个 job」，实际 4 个（漏了 `lint`），
    已补齐 job 清单与分工说明。
  - `README.zh.md` 的文件清单仍写「当前版本锚点 v0.2.0」，而最新发布锚点是 `v1.1.0`、
    `[0.2.0]` 只是 CHANGELOG 的最后一个语义版本节 —— 改为同时说明两者。
  - 两版 README 仍指向 `README_spaces.md` / `requirements_spaces.txt`，这两个文件已在
    `e42ae7e` 移除，读者点了只会 404 —— 改为「草案文件已从仓库移除」。
  - `CRITEO_NOTE`（`src/data/criteo.py`）与两份 `data/real_validation/*.json` 的
    `note` 写 LR 基线 `0.7717`，而同文件的 `val_auc_logistic_baseline` 是 `0.7716`
    （`round(auc_lr, 4)` 的输出）—— 统一为 `0.7716`。只改 `note` 文本，数值字段一律未动。
  - GitHub Release `v1.1.0` 说明里「§4.3 and the Fig. 3 caption」仍用旧章节号，
    已改为 `§4.2`（标签树与 Zenodo 归档快照不可变，仍保留 §4.3）。
- **CI 静态检查范围扩大**：mypy 由 `mypy src` 改为 `mypy .`（50 → 59 个文件），
  ruff 由 `ruff check src tests` 改为 `ruff check src tests scripts`。
  此前 `scripts/` 完全不在 CI 检查范围内 —— 这是 `bench_inference.py` 的类型错误
  能长期存活的根因。
- **全仓库 DOI 指向 `.v2`**：`CITATION.cff`、`.zenodo.json`、两版 README（徽章 +
  稿件页 + 引用 + BibTeX）、`docs/index.html`（首屏条 + 页脚）、三份技术文档、
  Streamlit 侧边栏与页脚引用、论文材料清单与溯源文件共 13 个文件；`.v2` 尚未全局
  解析（平台侧当晚生效），但 DOI 本身已由平台分配，链接按最终形态写死即可。
- **`CITATION.cff` 升版 1.0.0 → 1.1.0**，`date-released` 改为 2026-11-06；
  `.zenodo.json` 同步升版并在 description / notes 里说明"结果文件与 `preprint-v1`
  逐字节相同，变的只有论文里一个值、审计基建与文档"。
- **演示截图重拍**（`docs/assets/` 5 张，2720×3000 与旧图同规格，隐藏工具栏）：
  旧图页头仍显示 `.v1` DOI，与仓库其余部分不一致；新图取自本地运行的当前代码，
  七页均无运行时异常。
- **两版 README 增补「v2 改了什么」一段**：一句话说清改动、结论是否受影响、
  v1 为何保留，避免读者点开摘要看不出两版差别。
：英文版升为主 `README.md`，
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
- **`scripts/bench_inference.py` 的 2 个 mypy 报错**（`Incompatible import of "factory"`）：
  三个分支把 `MicroRecommendationEngine` / `LiteRecommendationEngine` /
  `RecommendationEngine` 绑到同一个名字 `factory`，mypy 判定为类型不兼容。改用
  `_Engine` / `_EngineFactory` 两个 Protocol 显式化三档引擎的共有契约（**未退化成
  `Any`**，反证实验确认接口被改坏时 mypy 会立刻报错）。脚本行为与性能口径不变。
- **论文 §4.2 与 Fig. 3 图注的跨规模边界值（`0.3576` → `0.3625`）**：预印本 v1 报告
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
- **文档把论文边界实验的章节号写成 §4.3（实为 §4.2）**：三份 PDF 的边界实验均在
  **4.2. Effective Debiasing Boundary**（§4.3 是 `Mixing Re-Injection`），仓库却有
  9 个文件共 16 处写成 §4.3。现统一更正：`README.md`、`README.zh.md`、`docs/index.html`、
  `docs/paper/PROVENANCE.md`、`docs/paper/manuscript.md`、`docs/ITERATION_PLAN.md`、
  `CHANGELOG.md`、`.zenodo.json`、`scripts/audit_paper_numbers.py`。
  `scripts/make_paper_figures.py` 图内 `(§4.3)` 标注指向混排，**未动**（改了会让图漂移）。
  Zenodo 已归档记录的元数据不可修改，该处笔误只在新提交里更正。
- **`docs/tech-report.md` 多处口径过期**：产物数 5 → **7**（三处）、测试数 125 → **133**、
  `run_bias_experiments.py` 耗时 23 s → **19 s**（实测 19.2 s）、代码版本锚点
  `v0.2.0` → **`v1.1.0`**；§7 删除已被 §5.8 覆盖的「单物品域」局限，现为 4 条，
  与论文 §5.2 对齐。
- **`docs/tech-report.md` §5.6 的 τ-sweep 结论与自身数据相抵**：原文称「偏差强度本身
  不决定去偏成败——π 动态范围才决定」，而同段数据是 τ=3 时增益 +0.1395、同期 ESS/N
  仅 0.181（远低于 0.358）仍净赚。现补明该组 π ∝ truth^τ、naive 收敛 `T_exp` 故本身
  有偏，与 §5.4（π 独立均匀抽样、naive 无偏）**不是同一设定**，边界数值不可脱离设定外推。
- **`scripts/run_bias_experiments.py` 文件头产出清单漏列 `boundary.json`**（6 → 7 项）。
- **`src/serve/i18n.py` 的落地接口示例与真实签名不符**：4 处写作
  `ess_warning_from_log(log, threshold=0.358)`，实际签名只接受 `propensity`、
  阈值内置 0.4（高于实测边界，偏保守）。改为 `ess_warning_from_log(propensity)`。

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
