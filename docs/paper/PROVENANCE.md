# 论文数字溯源（Provenance）

预印本已发布：*Bias propagation and effective boundaries of debiasing in recommender systems:
An oracle-based diagnosis study*, Preprints **2026**, 202610.2582 ·
DOI [10.20944/preprints202610.2582.v1](https://doi.org/10.20944/preprints202610.2582.v1)

论文一旦发布就不能再改。所以"可复现"这件事只能由仓库承担：**论文里的每一个数字，都必须能在本仓库的某个产物文件里找到出处，并且能由某个脚本重新算出来。**

本文档就是这张对应表。改动仓库前请先读一遍——尤其是"不允许改动的文件"一节。

---

## 1. 一句话规则

> 论文里出现的任何数值，其唯一合法来源是 `data/` 下的 JSON；JSON 的唯一合法来源是 `scripts/` 下的脚本。

不允许"手工修正一个看起来不对的数字"。如果产物与论文不一致，要么是脚本有 bug，要么是论文写错了——两个都得留下记录，不能悄悄对齐。

---

## 2. 八张表的对应关系

| 论文中的表 | 产物文件 | 关键字段 |
|---|---|---|
| Table 1 实验设置 | 各实验 JSON 的顶层元数据 | `n_requests` / `n_records` / `n_seeds` / `n_users` / `n_sessions` |
| Table 2 偏差传播矩阵 | `data/bias/propagation.json` | `rows[].L1_rel_err` / `L2_rel_err` / `L2_spearman` / `L2_rank_mae` / `impr_min` / `impr_max` / `exposure_kl` / `exposure_gini` |
| Table 3 去偏边界 | `data/bias/boundary.json` | `rows[].pi_ratio` / `ess_ratio` / `naive_err` / `snips_err` / `gain`；`cross_ess_ratio` |
| Table 4 二次注入 | `data/bias/injection.json` | `rows[].w_base` / `injection_spearman` |
| Table 5 估计目标对照 | `data/bias/estimand_check.json` | `rows[].T_exp` / `naive` / `snips` / `naive_err_vs_texp` / `snips_err_vs_tpop` / `ess` |
| Table 6 用户侧偏差 | `data/bias/userbias.json` | `rows[].beta` / `T_pop` / `T_exp_user` / `naive` / `snips_true_err` / `snips_agg_err` / `agg_residual` / `exposure_gini` |
| Table 7 propensity 估计误差 | `data/bias/propensity_estimation.json` | `rows[].snips_true_err` / `snips_hat_err` / `delta_err` / `ess_ratio_true` / `warning_hat`；`significance.tau*_raw` |
| Table 8 真实数据 | `data/real_validation/result.json`（Criteo 1M）、`result_adult.json`（UCI Adult） | `val_auc_deepfm` / `val_auc_logistic_baseline` |

### 正文里的散落数字

| 断言 | 出处 |
|---|---|
| Spearman 从 +0.87 反转到 −0.51 | `propagation.json` → `rows[τ=0].L2_spearman` = +0.8665，`rows[τ=3].L2_spearman` = −0.5054 |
| τ = 3 时 bootstrap p < 0.001 | `propagation.json` → `significance.tau3.L2_spearman_vs_zero.p` = 0.0001 |
| \|Δ\| ≤ 0.0023 | `propensity_estimation.json` → `max(abs(rows[].delta_err))` |
| τ = 3 raw 的 p = 0.87 | `propensity_estimation.json` → `significance.tau3_raw.delta_err_gt_zero.p` = 0.8684 |
| τ = 2 时 13% 误报、零漏报 | `propensity_estimation.json` → `rows[τ=2, raw].warning_hat` = 0.1333，`warning_true` = 0.0 |
| 平滑后 0.286 vs 真值 0.200 | `propensity_estimation.json` → `rows[τ=3, laplace1].ess_ratio_hat` = 0.2864 |
| τ-sweep 增益 +0.0468 / +0.0976 / +0.1395 | `boundary_sweep.json` → `grid[τ=1/2/3, n_records=5000].gain`，`gain_std` = 0.0062 / 0.0083 / 0.0094 |
| 用户侧 0.281 → 0.702 | `userbias.json` → `rows[].naive` 首尾 |

## 3. 六张图的对应关系

| 图 | 文件 | 数据来源 |
|---|---|---|
| Fig. 1 框架图 | `fig1_framework.png` | 手绘示意图，无数据依赖 |
| Fig. 2 传播曲线 | `fig2_propagation.png` | `propagation.json` |
| Fig. 3 边界曲线 | `fig3_boundary.png` | `boundary.json` |
| Fig. 4 二次注入 | `fig4_injection.png` | `injection.json` |
| Fig. 5 估计目标 | `fig5_estimand.png` | `estimand_check.json` |
| Fig. 6 用户侧 | `fig6_userbias.png` | `userbias.json` |

重生成：`python scripts/make_paper_figures.py`。输出与仓库内文件**逐字节一致**（已核对 sha256）。

---

## 4. 怎么验证

```bash
python scripts/audit_paper_numbers.py           # 逐单元格比对 8 张表 + 关键断言
python scripts/audit_paper_numbers.py --strict  # 有不一致则退出码 1（可接 CI）
```

当前结果：**核对 207 项，206 一致，1 项已知差异，0 不一致**。

图与 PDF 的一致性用另一条路验证：从 `AeroRec-preprints.org-v1.pdf` 抽出内嵌位图，与
`docs/paper/figures/*.png` 做灰度缩略图比对，6 张全部一一对应（MAE ≈ 0.2，次优匹配 ≥ 7）。

---

## 5. 已知差异（唯一一处）

**论文 §4.3 与 Fig. 3 图注中的 `0.3576`。**

- 论文原文：边界在小规模配置（60 物品 × 600 请求 × 30 种子）下为 0.3576，扩规模到 500 物品 × 5000 请求 × 250 种子后为 0.3579，因此"对规模稳健"。
- 当前实现：按同配置调用 `boundary_by_range(t_pop=0.25, n_records=600, n_seeds=30)` 得 **0.3625**；扩规模配置得 0.3579（与论文一致）。
- 差异 0.0049（约 1.4%）。该数字来自早期实现的某次运行，仓库内没有任何产物文件支撑，也无法在当前代码下逐位复现。
- **结论不受影响**：0.3625 与 0.3579 的相对差仍是 1.3%，"边界对规模稳健"成立。
- 处理方式：论文不改（已发布）；仓库保留原报告值并在 `scripts/audit_paper_numbers.py` 中登记为已知差异；本文件记录复算值。

如果将来要发预印本 v2，这一处应当在正文里换成 0.3625，或明确写成"0.358–0.363 区间"。

---

## 6. 不允许改动的文件（冻结）

以下 11 个文件是论文的数值底座，**只能新增，不能覆盖**：

```
data/bias/propagation.json            data/real_validation/result.json
data/bias/injection.json              data/real_validation/result_criteo_200k.json
data/bias/boundary.json               data/real_validation/result_criteo_200k_3seeds.json
data/bias/estimand_check.json         data/real_validation/result_adult.json
data/bias/boundary_sweep.json
data/bias/propensity_estimation.json
data/bias/userbias.json
```

同理，`docs/paper/figures/fig2–fig6` 就是已发布 PDF 里 Figure 2–6 的原图，重生成后必须与
PDF 保持一致；若因新实验需要改图，另存到 `docs/thesis/figures/`，不要覆盖。

## 7. 标签

| 标签 | 指向 | 含义 |
|---|---|---|
| `preprint-v1` | `c41997b` | 论文全部数字的产出状态 |
| `v1.0.0` | `1fc5e0f` | Zenodo 归档版本（仅新增元数据，代码与产物同 `preprint-v1`） |

核对历史状态：`git diff preprint-v1 -- data/`（应为空）。
