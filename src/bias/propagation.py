"""链路偏差传播：量化「召回 → 排序 → 混排」各级把估计扭曲了多少。

研究问题
--------
项目此前独立地撞上过两个偏差现象（热门基线稀疏性偏差、混排 `w_base` 热度二次注入），
但它们是**各查各的**。这里把它们统一成一个问题：

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
      与曝光策略无关。表现：估计均值 0.372 vs 真值 0.260（实测）。

  机制 B（曝光偏斜，排序偏差 / ordering）
      τ 越大，impr_i 的分布越悬殊。曝光少的物品被拽向 1.0，
      而它们的真实 CTR 往往很低 → **最冷门的反而被估成最好**，排序被反转。
      表现：Rank 相关随 τ 由 +0.83 跌到 −0.47（实测，见下方矩阵）。

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
                （对齐会把 τ 的影响抹掉——见模块 docstring 的踩坑记录）。
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


# ----------------------------------------------------------------------
# 传播矩阵：核心产出
# ----------------------------------------------------------------------
def propagation_matrix(true_ctr, n_items: int, taus, n_requests=500, slots=5,
                       alpha=5.0, w_base=0.0, seed=42) -> dict:
    """在多个偏差强度 τ 上算「链路各级偏差传播」矩阵。

    主指标为**逐物品相对误差 rel_err**（水平，机制 A）与
    **秩相关 spearman**（排序，机制 B）。两者都会随 τ 变化——
    这正是"偏差在链路上累积"的定量证据。

    n_items 参数保留以兼容调用方语义（全局物品数 = 候选数）。
    """
    rows = []
    for tau in taus:
        # 每 τ 独立确定性 rng（与扫描顺序无关），保证 demo 与报告可逐点复现一致
        rng = np.random.default_rng(seed * 1000 + int(round(float(tau) * 100)))
        r = run_stage_estimates(true_ctr, n_items, tau=tau, n_requests=n_requests,
                                slots=slots, alpha=alpha, w_base=w_base, rng=rng)
        t = r["true_ctr"]
        e1 = stage_error(r["L1_recall"], t)
        e2 = stage_error(r["L2_rank"], t)
        e3 = stage_error(r["L3_mix"], t)
        rows.append({
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
    return {
        "taus": [float(x) for x in taus],
        "rows": rows,
        "stage_order": ["L0_truth", "L1_recall", "L2_rank", "L3_mix"],
        "metric": "rel_err（水平/机制A）+ spearman（排序/机制B）",
        "w_base": float(w_base),
        "alpha": float(alpha),
        "n_requests": int(n_requests),
        "slots": int(slots),
    }


def mixing_second_injection(true_ctr, n_items: int, tau: float, w_bases,
                            n_requests=500, slots=5, alpha=5.0, seed=42) -> dict:
    """固定曝光偏差 τ，扫描混排基础分 w_base —— 量化**二次注入**。

    预期：w_base 越大（基础分=热度权重越高），排序相关性越高（因为基础分是真实 CTR），
    但相对误差先降后升——注入把估计从"有偏日志"拉向"真值"，却也引入了新的偏离。
    实测：τ=2 时 w_base 由 0→0.8，spearman 由 −0.03 升到 +0.97。
    """
    rows = []
    for wb in w_bases:
        # 每 w_base 独立确定性 rng（与扫描顺序无关）
        rng = np.random.default_rng(seed * 1000 + int(round(float(wb) * 100)))
        r = run_stage_estimates(true_ctr, n_items, tau=tau, n_requests=n_requests,
                                slots=slots, alpha=alpha, w_base=wb, rng=rng)
        t = r["true_ctr"]
        e2 = stage_error(r["L2_rank"], t)
        e3 = stage_error(r["L3_mix"], t)
        rows.append({
            "w_base": float(wb),
            "L2_rel_err": round(e2["rel_err"], 4),
            "L3_rel_err": round(e3["rel_err"], 4),
            "injection_rel": round(e3["rel_err"] - e2["rel_err"], 4),
            "L2_spearman": round(e2["spearman"], 4),
            "L3_spearman": round(e3["spearman"], 4),
            "injection_spearman": round(e3["spearman"] - e2["spearman"], 4),
        })
    return {
        "tau": float(tau),
        "rows": rows,
        "metric": "rel_err + spearman",
        "n_requests": int(n_requests),
        "slots": int(slots),
    }
