# AeroRec Technical Report

> 面向论文读者的技术文档。与 `README.md`（面向访客的展示页）并列；实验数字的
> 唯一来源是 `data/bias/*.json` 与 `docs/bias_report.md`，本文档与其逐项一致（H-6）。
> 复现命令见第 8 节；代码版本锚点为 tag `v0.1.0`。

## 1. Overview

AeroRec 是一个出行域推荐系统的完整实现，附带一条**可测量的曝光偏差研究线**。
系统的 ground truth 由解析函数生成（oracle 设定）：真实 CTR 已知、曝光策略的
propensity 闭式可算，因此「偏差 = 有偏观测 − 已知真值」可以被**精确测量**而非
估计——这是本研究线的核心方法学前提。

两条产出线：

1. **系统工程**：四路召回 → DeepFM 排序 → 5 步混排 → 降级兜底的完整链路，
   FastAPI 服务化 + Streamlit 交互面板 + 在线推理基准。
2. **偏差研究**：曝光偏差在「召回 → 排序 → 混排」链路上的传播测量、混排基础分的
   二次注入效应、IPS/SNIPS 去偏估计量的有效边界标定，以及全链路的统计显著性检验。

## 2. System architecture

| 层 | 组件 | 说明 |
|---|---|---|
| 召回 | Two-Tower 向量召回 + 规则召回 `RuleRecall` | 四路召回融合（向量 / 规则 / 热门 / 近期） |
| 排序 | DeepFM | 与真实数据验证共用同一实现（见 §3） |
| 混排 | 5 步 Mixer | 基础分权重 `w_base` 默认取 0.0（依据见 §5.3） |
| 兜底 | Degrade | 降级策略，保证链路可用性 |
| 服务 | FastAPI + Streamlit | 在线侧仅依赖 numpy 推理路径，不加载 torch |

配置分四档（`src/config*.py`）：lite / micro / scale 与默认档，分别用于
快速验证、在线演示、扩规模与本地开发。

## 3. Data and ground truth

| 数据 | 真实 | 用途 | 关键结果 |
|---|---|---|---|
| 合成出行域流量（`compute_true_ctr`） | 否（oracle） | 偏差研究全链路 | 见 §5 |
| Criteo Display Advertising Challenge（100 万行） | 是 | 模型迁移性验证 | DeepFM 验证 AUC **0.7733** |
| UCI Adult / Census Income | 是 | 迁移性验证（非 CTR 域） | AUC **0.9183** |

合成指标的诚实口径：离线评估 AUC **0.8019**、相对热门基线 **+28.5%**——
两份真实数据证明模型实现能在真实数据上学到信号，但它们都不是出行域流量，
合成指标不因真实数据验证而升级为「真实性能」。

## 4. Offline evaluation

离线评估采用真实 CTR 口径 + Oracle 上界双口径（`src/eval/`），
训练显式播种：同一份代码在固定种子下可复现；进程全局 RNG 的残余状态会带来
约 0.004 的 AUC 漂移，因此跨环境比较必须固定种子（详见 `docs/bias_report.md` §1）。

## 5. Bias study

### 5.1 两种偏差机制

- **机制 A（水平 / 稀疏收缩）**：平滑 CTR `CTR_hat_i = (click_i + α)/(impr_i + α)`
  对曝光少的物品把估计拽向先验 `α/(0+α) = 1.0`。即使曝光完全均匀（τ=0）也存在。
- **机制 B（排序 / 曝光偏斜）**：曝光策略 π ∝ truth^τ（τ 为偏斜强度）下，
  冷门物品曝光趋零、被拽向高估，而其真实 CTR 低 → **排序被反转**。

度量同时报告两类：逐物品相对误差 `rel_err`（水平）与 Spearman 秩相关（排序）。
只看均值会因重标定抵消而得出「偏差可控」的错误结论。

### 5.2 偏差传播矩阵（`propagation.json`）

配置：items=500，requests=5000，slots=5，α=5.0，w_base=0，taus={0, 0.5, 1, 2, 3}；
**30 种子均值±std**，每 (seed, τ) 以 `default_rng(seed+s)` 重置（共同随机数，
τ 间差异只来自曝光策略）。

| τ | L2 rel_err | L2 spearman | 曝光 Gini | KL | impr_min |
|---|---|---|---|---|---|
| 0.0 | 0.4112±0.0125 | **+0.8665**±0.0104 | 0.0794 | 0.0100 | 30.1 |
| 0.5 | 0.5049±0.0144 | +0.7999±0.0166 | 0.1608 | 0.0410 | 15.3 |
| 1.0 | 0.6706±0.0215 | +0.6316±0.0235 | 0.2850 | 0.1315 | 5.4 |
| 2.0 | 1.2457±0.0219 | +0.0345±0.0306 | 0.5156 | 0.4710 | 0.1 |
| 3.0 | 2.0133±0.0286 | **−0.5054**±0.0188 | 0.6921 | 0.9716 | 0.0 |

排序相关性在 τ≈2–3 之间穿越零点：偏差的主要危害是**把排序搞反**，而非整体偏高。

### 5.3 二次注入（`injection.json`）

固定 τ=2，扫描混排基础分权重 w_base∈{0, 0.2, 0.4, 0.8, 1.0}。曝光与点击只依赖
τ，与 w_base 无关，故只模拟一次曝光、逐 w_base 解析重算
`L3 = (1−w)·L2 + w·truth`——曲线不含抽样噪声，w_base 间逐点严格可比。
`injection_spearman`（相对 L2 的增量）：0 → 0.1929 → 0.4287 → 0.9147 → **0.9655**，
严格单调；w=1.0 时 L3 恰等于真值（rel_err=0、spearman=1.0，实现自检）。
在强偏差场景下 `w_base` 改善混排级排序；注入**重复**的热度信息有害
（项目默认取 0.0），注入独立信息有益——注入的符号由「信息是否已在候选集中」判别。

### 5.4 去偏有效边界（`boundary.json`）

核心实验：扫 π 的动态范围（6 档，pi_ratio 1 → 2×10⁶；n_records=20000，
250 种子平均），标定 IPS/SNIPS 由赚转亏的 ESS/N 阈值。

- `cross_ess_ratio = 0.3579`（扩规模前 0.3576，对规模稳健）：ESS/N 低于约 **0.358**
  时去偏收益转负且超出噪声水平（容差 0.0010）。
- `naive_err` 各档恒为 0.0024；高偏场景 `snips_err` 升至 0.0651——**SNIPS 在高偏下
  反而劣于 naive**。
- 判据提取用「亏损超出噪声水平（3 × loss floor）」而非「gain 首次变负」
  （后者会在 ESS/N=1 处误触发）。
- 落地接口：`ess_warning_from_log(propensity)`——ESS/N 无需 ground-truth 即可计算，
  可作为真实系统的去偏预警指标。

### 5.5 估计目标对照（`estimand_check.json`）

配置：60 物品、400 请求 × 5 坑位（报告 §4 历史口径），30 种子。验证
**naive 收敛到 `T_exp = Σ truth·π`、SNIPS 收敛到 `T_pop = mean(truth)`**：

| τ | T_exp | naive | SNIPS | naive→T_exp 误差 | SNIPS→T_pop 误差 | ESS (N=2000) |
|---|---|---|---|---|---|---|
| 0.0 | 0.2525 | 0.2510 | 0.2510 | 0.0078 | 0.0078 | 2000 |
| 1.0 | 0.3045 | 0.3072 | 0.2547 | 0.0079 | 0.0079 | 1634 |
| 3.0 | 0.4028 | **0.4025** | 0.2477 | 0.0100 | 0.0169 | 400 |

naive 对 `T_exp` 的误差始终在 ≈1 个抽样标准误的噪声量级——即使 `T_exp` 漂到 0.40。
之前当作「偏差」的 0.2525 → 0.4028 漂移不是估计量出错，而是**估计目标变了**；
IPS/SNIPS 做的是在两个估计目标之间转换，而不是修复坏估计。SNIPS 贴 `T_pop`
的代价是 ESS 由 2000 降到 400、τ=3 时 std 放大到 ±0.021。

### 5.6 去偏收益的 τ-sweep（`boundary_sweep.json`）

60 物品、40 种子，N∈{1000, 2000, 5000, 10000}。N=5000 时去偏收益 gain
（naive_err − snips_err）随 τ 单调上升：τ=1 → **+0.0468±0.0062**，τ=2 → +0.0976±0.0083，
τ=3 → **+0.1395±0.0094**。τ 只拉大 `T_exp` 与 `T_pop` 的差距（estimand 之差），
而 propensity 闭式已知，因此偏差强度本身不决定去偏成败——π 动态范围才决定。

### 5.7 Propensity estimation error (improvement-plan #4)

Config: 60 items, 400 requests × 5 slots, 30 seeds; π̂ estimated from the log
only (frequency λ=0 / Laplace λ=1), never touching ground truth. At this scale
(2,000 records) replacing the closed-form π with π̂ changes SNIPS error
insignificantly (delta ≈ 0, p = 0.87) — the oracle conclusions are robust to
propensity estimation error. The ESS/N warning is conservative under π̂
(false-alarm rate 0.13 at τ=2, miss rate 0 — fail-safe direction). Caveat:
estimation error shrinks as √N; raw estimates carry 1/π̂ explosion risk at
smaller N or stronger skew, mitigated by λ>0 smoothing.

### 5.8 User-side selection bias (improvement-plan #5/#6)

Structure: 100 users × 60 items, `truth_ui = base_i × affinity_ui` (oracle);
users self-select via `p_ui ∝ softmax(β·log-affinity)` (β = self-selection
strength, analogous to τ), 2,000 sessions × 5 views, 30 seeds.

| β | T_pop | T_exp_user | naive | SNIPS(true p) err | SNIPS(item-marginal) err | residual |
|---|---|---|---|---|---|---|
| 0.0 | 0.2811 | 0.2811 | 0.2814 | 0.0039 | 0.0039 | +0.0000 |
| 1.0 | 0.2811 | 0.4066 | 0.4072 | 0.0040 | 0.0668 | +0.0628 |
| 2.0 | 0.2811 | 0.5317 | 0.5314 | 0.0071 | 0.1342 | +0.1272 |
| 4.0 | 0.2811 | 0.7021 | 0.7025 | 0.0373 | 0.2594 | +0.2220 |

Findings (all p < 0.001): (1) self-selection drifts the naive estimand from
0.281 to 0.702 — pooled CTR appears to grow 2.5× while truth is unchanged;
(2) debiasing at item-marginal granularity (ignoring the user dimension)
leaves a significant residual (+0.063 → +0.222), while correct user-item-level
propensity debiasing keeps error at 0.004–0.037. **User-side bias must be
modeled at the user-item granularity.**

## 6. Statistical treatment

- **多种子**：传播/注入 30 种子、边界 250 种子、τ-sweep 40 种子；所有跨种子指标
  以均值±std 落盘（`*_std` 字段）。
- **显著性**：bootstrap 百分位置信区间 + 重抽样 p 值（B=10000，固定 bootstrap
  种子），纯 numpy 实现（`src/bias/significance.py`），不引入 scipy——研究路径
  零重依赖。关键检验：τ=3 排序反转 p<0.001；τ=0→3 配对下降 p<0.001；
  注入相邻档增量全部 p<0.001；boundary 边界档亏损 p<0.001 而无损档
  （gain≡0）p=1.0 正确不显著。bootstrap p 的最小可达值为 1/(B+1)=0.0001，
  应读作「p < 0.001」。
- **零漂移门禁**：CI 在干净环境重跑全部实验，`git diff --exit-code` 逐字节校验
  5 个产物 JSON；任何「改代码没更新产物」的提交直接变红。

## 7. Limitations

- **合成环境**：ground truth 与 propensity 均为解析已知（oracle 设定）。
  真实系统必须估计 propensity，会引入本实验不存在的误差，实际有效边界会更严苛。
- **边界数值不可迁移**：`ESS/N ≈ 0.358` 与容差 0.001 在本项目合成数据的规模与
  先验下测出。可迁移的是方法（ESS/N 预警），不是数值。
- **单物品域**：未涉及用户侧偏好偏差（同一用户内跨物品的选择偏差）。
- **真实数据非出行域**：Criteo / Adult 只验证模型实现的迁移性，不改变合成
  指标的口径。
- **本项目不提出新算法**：所有估计量（IPS/SNIPS/DR）与度量均为文献标准方法，
  实现参考 Steck 2011、Schnabel et al. 2016、Joachims et al.、Gilotte et al. 2018。

## 8. Reproducibility

```bash
pip install numpy==2.3.5          # 研究路径唯一依赖
python scripts/run_bias_experiments.py    # 生成 5 个产物 JSON
python scripts/build_bias_report.py       # 自包含 HTML 报告
python -m pytest                          # 全量测试（125 项）
```

- 实验耗时约 23 s（单机、单进程）；随机种子、bootstrap 种子、RNG 策略
  （`rng_policy` 字段）均写入产物。
- 零漂移门禁：`.github/workflows/repro.yml`，对 5 个产物 JSON 逐字节校验。
- 在线侧内存纪律：服务进程不加载 torch（峰值 RSS ≈ 129 MB，
  单线程 P50 0.7 ms / P95 0.85 ms / P99 0.95 ms，QPS ≈ 1378，micro 档实测）。
- 代码版本锚点：tag **`v0.1.0`**。
- 更完整的机理叙事与被修正的错误见 `docs/bias_report.md`；研究贡献一页纸摘要见
  `docs/research_highlight.md`。
