# Changelog

本项目所有显著变更记录于此。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号在首次正式发布前以里程碑（W1–W11 = 第 1–11 周）+ 日期标识。

## [Unreleased]

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
