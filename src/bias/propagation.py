"""链路偏差传播：量化「召回 → 排序 → 混排」各级把估计扭曲了多少。

研究问题
--------
热门基线的稀疏性偏差与混排 `w_base` 的热度二次注入，是两个独立的偏差现象；
本模块把它们统一成一个问题：

    偏差在链路上是**累积**的，而且**可以在下游被二次注入**。

核心机制（先讲清楚，否则度量必错）
----------------------------------
在有偏曝光下，「物品 i 的曝光次数」服从多项分布，其期望 ∝ π_i。
而排序级唯一的 CTR 估计是**平滑 CTR**：

    CTR_hat_i = (click_i + α) / (impr_i + α)

这个估计量对**曝光次数少**的物品会把结果拽向先验 α/(0+α) = 1.0。
于是产生**两种性质完全不同**的偏差：

  机制 A（稀疏收缩，水平偏差 / level）
      α 相对曝光量不可忽略时，估计被整体拽向 1。
      **即使曝光完全均匀（τ=0）也照样存在**——它是"样本量不够"的偏差，
      与曝光策略无关。表现：冷门物品的平滑 CTR 被系统性高估
      （实测见 `data/bias/propagation.json` 的 `L2_signed` 列）。

  机制 B（曝光偏斜，排序偏差 / ordering）
      τ 越大，impr_i 的分布越悬殊。曝光少的物品被拽向 1.0，
      而它们的真实 CTR 往往很低 → **最冷门的反而被估成最好**，排序被反转。
      表现：与真值的秩相关随 τ 下降，强偏差下跌破 0（实测见
      `data/bias/propagation.json` 的 `L2_spearman` 列）。

**关键认识**：偏差的大头不是"整体偏高"，而是**排序被判反**。
因此度量必须同时报告「水平」与「排序」两类，只看均值会得出错误结论
（本项目实际踩过：只比均值时 L1_bias 在 τ=0..3 上恒为 0，现象被对齐抹掉）。

三级链路模型
------------
    L0 真值    : 各物品真实 CTR（ground-truth，链路不可见）
    L1 召回级  : 只依据曝光次数、不做点击平滑的估计（策略直接后果）
    L2 排序级  : 有偏日志上的平滑 CTR（**链路真实可见的唯一信息**）
    L3 混排级  : L2 再叠一次热度基础分 w_base（二次注入）

为了让各级**同尺度可比**，L1 采用与真值同为 CTR 量纲、且随曝光单调的估计：
直接用 `曝光份额 π` 的尺度对齐版本，而非把 π 当成 CTR 本身。
"""
from __future__ import annotations

import numpy as np

from src.bias.metrics import exposure_gini, exposure_kl, smoothed_ctr
from src.bias.policy import popularity_propensity, sample_exposure
from src.bias.significance import bootstrap_mean_test, paired_diff_test


# ----------------------------------------------------------------------
# 逐物品误差度量（偏差传播的正确口径）
# ----------------------------------------------------------------------
def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


def _rank(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def _spearman(a, b) -> float:
    return _pearson(_rank(a), _rank(b))


def stage_error(est, truth) -> dict:
    """逐物品衡量某一级估计的偏差——**偏差传播的正确度量**。

    为什么不能只看均值
    ------------------
    曝光偏差不改变"整体平均 CTR"，它改变的是**哪些物品被高估/低估**。
    只比较向量均值时，重标定（rescaling）会让逐物品偏差在求均值时抵消，
    于是每一级的水平偏差都恒为 0（实测 L1_bias 在 τ=0..3 上恒为 0.0000）。
    因此必须用**逐物品**、对整体标定不敏感的量。

    返回四项，分属两种机制：
      level（水平/机制 A）
        - `rel_err` : mean(|est−truth| / truth)，整体量纲误差
        - `signed`  : mean(est−truth)，有符号（>0 = 系统性高估）
      ordering（排序/机制 B）
        - `spearman`: 与真值的秩相关（1=完全同序，<0=排序被反转）
        - `rank_mae`: 平均秩错位 / N
    """
    est = np.asarray(est, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n = len(truth)
    if n == 0:
        return {"signed": 0.0, "rel_err": 0.0, "rank_mae": 0.0,
                "spearman": 0.0, "pearson": 0.0}
    return {
        "signed": float(np.mean(est - truth)),
        "rel_err": float(np.mean(np.abs(est - truth) / (np.abs(truth) + 1e-6))),
        "rank_mae": float(np.mean(np.abs(_rank(est) - _rank(truth))) / max(n, 1)),
        "spearman": _spearman(est, truth),
        "pearson": _pearson(est, truth),
    }


# ----------------------------------------------------------------------
# 链路模拟
# ----------------------------------------------------------------------
def run_stage_estimates(true_ctr, n_items: int, tau: float, n_requests: int,
                        slots: int, alpha: float = 5.0, w_base: float = 0.0,
                        rng: np.random.Generator | None = None) -> dict:
    """按曝光策略 τ 生成有偏曝光，逐级给出「估计 CTR」，与真值同尺度可比。

    与真值同尺度的构造（关键）
    -------------------------
    - L0 真值 : truth_i
    - L1 召回 : 只用**曝光份额**推断热度，再映射到 CTR 量纲：
                est_L1_i = truth_mean · N · π_i
                这样它随策略单调、与真值同为 CTR 量纲，且**不做均值对齐**
                （对齐会把 τ 的影响抹掉）。
    - L2 排序 : 平滑 CTR，链路真实可见的唯一信息。
    - L3 混排 : (1−w)·L2 + w·truth（基础分用 popularity 代理，本项目里就是历史 CTR）。
    """
    rng = rng or np.random.default_rng(0)
    T = np.asarray(true_ctr, dtype=np.float64)
    if T.ndim == 1:
        T = T[None, :]
    R, C = T.shape
    truth = T.mean(axis=0)
    tmean = float(truth.mean())

    # ---- 曝光策略：π ∝ truth^τ（用真实 CTR 作热度代理，工业上即历史 CTR）----
    if tau == 0.0:
        pi = np.full(C, 1.0 / C)
    else:
        pi = popularity_propensity(truth, tau=tau)
    pi = pi / max(pi.sum(), 1e-12)

    # ---- 逐请求抽曝光，累计曝光/点击 ----
    K = int(min(slots, C))
    imps = np.zeros(C, dtype=np.float64)
    clicks = np.zeros(C, dtype=np.float64)
    for _ in range(int(n_requests)):
        idx = sample_exposure(pi, K, rng, replace=True)
        p = T[0, idx] if T.shape[0] == 1 else T[_ % R, idx]
        clicks[idx] += (rng.random(len(idx)) < p).astype(np.float64)
        imps[idx] += 1.0

    # ---- 各级估计（全部 CTR 量纲）----
    # L1：只用曝光份额推断的相对热度（不做均值对齐）
    l1_est = tmean * C * pi
    # L2：平滑 CTR（唯一的链路可见信息）
    l2_est = smoothed_ctr(clicks, imps, alpha=alpha)
    # L3：混排叠加热度基础分（凸组合，保持 CTR 量纲）
    l3_est = (1.0 - w_base) * l2_est + w_base * truth

    return {
        "true_ctr": truth,
        "L0_truth": truth,
        "L1_recall": l1_est,
        "L2_rank": l2_est,
        "L3_mix": l3_est,
        "impressions": imps,
        "clicks": clicks,
        "exposure_freq": imps / max(imps.sum(), 1e-12),
        "pi": pi,
        "alpha": float(alpha),
    }


def _aggregate_rows(per_seed_rows: list[list[dict]], key_field: str) -> list[dict]:
    """跨种子聚合逐行指标：数值字段取均值，n_seeds>1 时附 `*_std`。

    口径与 `boundary.py` 的多种子聚合一致：`np.mean` / `np.std`（ddof=0）、
    round 4。种子序列为 `default_rng(seed + s)`（s = 0..n_seeds-1）。
    n_seeds=1 时不添加 std 字段，行结构与单种子完全一致（向后兼容）。
    """
    n = len(per_seed_rows)
    first = per_seed_rows[0]
    out: list[dict] = []
    for i in range(len(first)):
        row: dict = {key_field: first[i][key_field]}
        for f in first[i]:
            if f == key_field:
                continue
            vals = [pr[i][f] for pr in per_seed_rows]
            row[f] = round(float(np.mean(vals)), 4)
            if n > 1:
                row[f + "_std"] = round(float(np.std(vals)), 4)
        out.append(row)
    return out


# ----------------------------------------------------------------------
# 传播矩阵：核心产出
# ----------------------------------------------------------------------
def propagation_matrix(true_ctr, n_items: int, taus, n_requests=500, slots=5,
                       alpha=5.0, w_base=0.0, seed=42, n_seeds: int = 1) -> dict:
    """在多个偏差强度 τ 上算「链路各级偏差传播」矩阵。

    主指标为**逐物品相对误差 rel_err**（水平，机制 A）与
    **秩相关 spearman**（排序，机制 B）。两者都会随 τ 变化——
    这正是"偏差在链路上累积"的定量证据。

    n_items 参数保留以兼容调用方语义（全局物品数 = 候选数）。

    多种子（n_seeds > 1）：种子 s ∈ [0, n_seeds)，每个种子内部仍按 CRN
    （每 τ 以 default_rng(seed+s) 重置）跑完整条 τ 扫描；指标跨种子取
    均值并附 `*_std` 误差棒（口径同 boundary.py：np.mean/np.std, round 4）。
    n_seeds=1 时输出与单种子完全一致（不添加 std 字段，向后兼容）。
    """

    def _rows_for_seed(s: int) -> list[dict]:
        out: list[dict] = []
        for tau in taus:
            # CRN（共同随机数）：每个 τ 都从同一初始状态重置 RNG——随机流相同、
            # 仅曝光策略 π 不同，τ 间差异只来自参数本身而非不同的抽样序列；
            # 且与扫描顺序无关、新增 τ 不影响既有 τ 的结果（demo 与报告逐点一致）。
            rng = np.random.default_rng(seed + s)
            r = run_stage_estimates(true_ctr, n_items, tau=tau, n_requests=n_requests,
                                    slots=slots, alpha=alpha, w_base=w_base, rng=rng)
            t = r["true_ctr"]
            e1 = stage_error(r["L1_recall"], t)
            e2 = stage_error(r["L2_rank"], t)
            e3 = stage_error(r["L3_mix"], t)
            out.append({
                "tau": float(tau),
                # 水平（机制 A）
                "L1_rel_err": round(e1["rel_err"], 4),
                "L2_rel_err": round(e2["rel_err"], 4),
                "L3_rel_err": round(e3["rel_err"], 4),
                "L2_incremental_rel": round(e2["rel_err"] - e1["rel_err"], 4),
                "L3_incremental_rel": round(e3["rel_err"] - e2["rel_err"], 4),
                "L2_signed": round(e2["signed"], 4),
                # 排序（机制 B）
                "L1_spearman": round(e1["spearman"], 4),
                "L2_spearman": round(e2["spearman"], 4),
                "L3_spearman": round(e3["spearman"], 4),
                "L2_rank_mae": round(e2["rank_mae"], 4),
                # 曝光自省
                "impr_min": int(r["impressions"].min()),
                "impr_max": int(r["impressions"].max()),
                "exposure_kl": round(exposure_kl(r["exposure_freq"]), 4),
                "exposure_gini": round(exposure_gini(r["exposure_freq"]), 4),
            })
        return out

    n = max(1, int(n_seeds))
    per_seed = [_rows_for_seed(s) for s in range(n)]
    rows = _aggregate_rows(per_seed, "tau")
    rng_policy = "common_random_numbers: 每个 tau 均以 default_rng(seed) 重置，" \
                 "tau 间差异只来自曝光策略而非抽样序列"
    if n > 1:
        rng_policy += (f"；跨 {n} 个种子（default_rng(seed+s), s=0..{n - 1}）"
                       "取均值±std（口径同 boundary.py）")
    # 显著性（n_seeds>1 才有 per-seed 样本）：逐 τ 的机制 A/B 指标 vs 0，
    # 以及首末 τ 的排序质量同种子配对差（H1: 下降）。
    sig: dict = {}
    if n > 1:
        for i, tau in enumerate(taus):
            sp = [r[i]["L2_spearman"] for r in per_seed]
            rel = [r[i]["L2_rel_err"] for r in per_seed]
            sig[f"tau{tau:g}"] = {
                "L2_spearman_vs_zero": bootstrap_mean_test(sp),
                "L2_rel_err_vs_zero": bootstrap_mean_test(rel, alternative="greater"),
            }
        sig["tau_first_vs_last_spearman_paired"] = paired_diff_test(
            [r[-1]["L2_spearman"] for r in per_seed],
            [r[0]["L2_spearman"] for r in per_seed],
            alternative="less")
    out = {
        "taus": [float(x) for x in taus],
        "rows": rows,
        "stage_order": ["L0_truth", "L1_recall", "L2_rank", "L3_mix"],
        "metric": "rel_err（水平/机制A）+ spearman（排序/机制B）",
        "w_base": float(w_base),
        "alpha": float(alpha),
        "n_requests": int(n_requests),
        "slots": int(slots),
        "n_seeds": n,
        "seed_base": int(seed),
        "rng_policy": rng_policy,
    }
    if sig:
        out["significance"] = sig
    return out


def mixing_second_injection(true_ctr, n_items: int, tau: float, w_bases,
                            n_requests=500, slots=5, alpha=5.0, seed=42,
                            n_seeds: int = 1) -> dict:
    """固定曝光偏差 τ，扫描混排基础分 w_base —— 量化**二次注入**。

    实现（与早期版本的关键差别）
    --------------------------
    曝光与点击只依赖 τ，与 w_base 无关；而 L3 = (1−w)·L2 + w·truth 是解析式。
    因此曝光/点击**只模拟一次**（rng = default_rng(seed)），再对每个 w_base
    解析重算 L3——注入曲线不含抽样噪声，w_base 间严格可比。
    早期版本每个 w_base 独立重抽曝光（不同随机流），把采样噪声混进了
    注入增量，曾产生 w=0.4 处的伪回撤；该缺陷已由本实现根除。

    预期：w_base 越大，L3 越靠近真值，spearman 随之上升、rel_err 随之下降
    （具体数值见 `data/bias/injection.json`，文档不得另行手写）。

    多种子（n_seeds > 1）：每个种子独立「单次模拟 + 解析扫描」
    （default_rng(seed+s)），指标跨种子取均值并附 `*_std`；w=1.0 端点
    跨种子确定性（L3≡truth），均值仍严格为 1.0、std=0。
    n_seeds=1 时输出与单种子完全一致（向后兼容）。
    """
    n = max(1, int(n_seeds))

    def _rows_for_seed(s: int) -> list[dict]:
        # 曝光与 L2 对整场扫描只算一次；w_base 仅进入 L3 的解析凸组合
        rng = np.random.default_rng(seed + s)
        r = run_stage_estimates(true_ctr, n_items, tau=tau, n_requests=n_requests,
                                slots=slots, alpha=alpha, w_base=0.0, rng=rng)
        t = r["true_ctr"]
        l2 = np.asarray(r["L2_rank"], dtype=np.float64)
        e2 = stage_error(l2, t)

        out: list[dict] = []
        for wb in w_bases:
            l3 = (1.0 - float(wb)) * l2 + float(wb) * t
            e3 = stage_error(l3, t)
            out.append({
                "w_base": float(wb),
                "L2_rel_err": round(e2["rel_err"], 4),
                "L3_rel_err": round(e3["rel_err"], 4),
                "injection_rel": round(e3["rel_err"] - e2["rel_err"], 4),
                "L2_spearman": round(e2["spearman"], 4),
                "L3_spearman": round(e3["spearman"], 4),
                "injection_spearman": round(e3["spearman"] - e2["spearman"], 4),
            })
        return out

    per_seed = [_rows_for_seed(s) for s in range(n)]
    rows = _aggregate_rows(per_seed, "w_base")
    rng_policy = ("single_simulation_analytic_sweep: 曝光只模拟一次"
                  "（default_rng(seed)），L3=(1-w)*L2+w*truth 逐 w_base 解析重算")
    if n > 1:
        rng_policy += f"；跨 {n} 个种子（default_rng(seed+s)）取均值±std"
    # 显著性（n_seeds>1 才有 per-seed 样本）：各档注入增益 vs 0，
    # 相邻 w_base 档的同种子配对增量（H1: >0，注入单调改善）。
    sig: dict = {}
    if n > 1:
        for i, wb in enumerate(w_bases):
            sig[f"w{wb:g}_spearman_vs_zero"] = bootstrap_mean_test(
                [r[i]["injection_spearman"] for r in per_seed])
        for i in range(len(w_bases) - 1):
            sig[f"w{w_bases[i]:g}_to_{w_bases[i + 1]:g}_paired"] = paired_diff_test(
                [r[i + 1]["injection_spearman"] for r in per_seed],
                [r[i]["injection_spearman"] for r in per_seed],
                alternative="greater")
    out = {
        "tau": float(tau),
        "rows": rows,
        "metric": "rel_err + spearman",
        "n_requests": int(n_requests),
        "slots": int(slots),
        "n_seeds": n,
        "seed_base": int(seed),
        "rng_policy": rng_policy,
    }
    if sig:
        out["significance"] = sig
    return out
