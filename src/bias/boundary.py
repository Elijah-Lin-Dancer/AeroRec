"""去偏的**有效边界**：什么时候 IPS/SNIPS 划算，什么时候反而更糟。

⚠️ 本模块的核心结论是**反直觉的，且经过实测修正**（务必先读）
---------------------------------------------------------------------
最初的假设是「曝光越偏（τ 越大），去偏越容易亏本」。**实测否定了一半**：

  (a) 扫 popularity-τ（60/200 物品，40 seed 平均）：
      去偏在**所有** τ 上都赚（τ=1 gain≈+0.07，τ=3 gain≈+0.16），
      而且 τ 越大赚得越多。τ 增大只是让 T_exp 与 T_pop 差距变大，
      于是"看似偏差"变大 —— 但那是 estimand 之差，不是估计误差。

  (b) 真正让去偏**亏本**的不是偏差幅度，而是 **π 的动态范围**（→ ESS）：
      | π 取值范围 | ESS/N | gain(naive_err − snips_err) |
      |-----------|-------|-----------------------------|
      | [1, 1] 均匀 | 1.000 | ±0.0000（两目标重合）        |
      | [0.5, 1.5] | 0.905 | −0.0010（几乎无损）          |
      | [0.05, 2.0]| 0.358 | −0.0065（开始亏）            |
      | [0.002,2.0]| 0.048 | −0.0316（明显亏）            |
      | [1e-4, 2.0]| 0.019 | −0.0850（严重亏）            |
      | [1e-6, 2.0]| 0.017 | −0.1070（崩溃）              |

    机制：π 有极小值时 1/π 权重爆炸 → SNIPS 方差主导 → 估计被少数记录带偏。
    **ESS/N 越低，1/π 的方差越大，去偏越可能亏本。**

由此得到本模块最有实用价值的产出
--------------------------------
**ESS/N 是一个不需要 ground-truth 就能算的预警指标。**
在合成环境里标定「ESS/N 低于约 0.4 时去偏开始亏本」，
然后在**真实系统**里只用 ESS/N（可从日志与 logging policy 估出）做在线预警——
无需真值即可决定「该不该上 IPS 去偏」。

这就是从研究到落地的桥，也是本模块相对"只报告偏差大小"的增量。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.bias.estimators import effective_sample_size, naive_estimate, snips_estimate
from src.bias.policy import popularity_propensity


def _simulate_log(truth, pi, rng, n_records: int):
    """按 π 抽 n_records 条曝光记录，返回 (逐记录点击, 逐记录 π)。

    IPS 作用在**记录粒度**上，故把「每物品一个 π」展开成逐条记录，
    这也是真实日志的形态。
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    counts = rng.multinomial(n_records, pi)
    rec_item = np.repeat(np.arange(len(pi)), counts)
    if len(rec_item) == 0:
        return np.zeros(0), np.zeros(0)
    y = (rng.random(len(rec_item)) < truth[rec_item]).astype(np.float64)
    return y, pi[rec_item]


def _eval_once(truth, pi, n_records, rng):
    y, p_rec = _simulate_log(truth, pi, rng, n_records)
    if len(y) == 0:
        return None
    t_pop = float(truth.mean())
    nv = naive_estimate(y)
    sn = snips_estimate(y, p_rec)
    ess = effective_sample_size(p_rec)
    return {
        "naive": nv, "snips": sn,
        "naive_err": abs(nv - t_pop), "snips_err": abs(sn - t_pop),
        "gain": abs(nv - t_pop) - abs(sn - t_pop),
        "ess": ess, "ess_ratio": ess / max(len(y), 1),
        "t_pop": t_pop,
    }


def boundary_sweep(truth, taus, n_records_list, n_seeds: int = 20,
                   seed: int = 42) -> dict:
    """扫描 (τ, N) 网格，**多种子平均**后记录 naive/SNIPS 的误差与 ESS。

    ⚠️ 必须多种子平均：单 seed 下 0.0155 vs 0.0065 这种差会**跑在种子噪声里**
    （本项目实测过：单 seed 时 N=1000 的误差反而大于 N=2000）。
    """
    truth = np.asarray(truth, dtype=np.float64)
    t_pop = float(truth.mean())
    grid = []
    for tau in taus:
        pi = np.full(len(truth), 1.0 / len(truth)) if tau == 0.0 \
            else popularity_propensity(truth, tau=tau)
        t_exp = float((truth * (pi / pi.sum())).sum())
        for n_rec in n_records_list:
            acc = []
            for s in range(n_seeds):
                r = _eval_once(truth, pi, n_rec, np.random.default_rng(seed + s))
                if r:
                    acc.append(r)
            if not acc:
                continue
            grid.append({
                "tau": float(tau),
                "n_records": int(n_rec),
                "t_exp": round(t_exp, 4),
                "naive": round(float(np.mean([a["naive"] for a in acc])), 4),
                "snips": round(float(np.mean([a["snips"] for a in acc])), 4),
                "naive_err": round(float(np.mean([a["naive_err"] for a in acc])), 4),
                "snips_err": round(float(np.mean([a["snips_err"] for a in acc])), 4),
                "gain": round(float(np.mean([a["gain"] for a in acc])), 4),
                "gain_std": round(float(np.std([a["gain"] for a in acc])), 4),
                "ess_ratio": round(float(np.mean([a["ess_ratio"] for a in acc])), 4),
                "n_seeds": len(acc),
            })
    return {"t_pop": round(t_pop, 4), "n_seeds": int(n_seeds), "grid": grid}


def boundary_by_range(t_pop: float = 0.25, n_records: int = 20000,
                      ranges=None, n_seeds: int = 60, seed: int = 0) -> dict:
    """**核心实验**：扫 π 的动态范围，找去偏由赚转亏的 ESS/N 阈值。

    这个实验才真正暴露了边界——不是 τ，而是 π 的极值比。
    """
    if ranges is None:
        ranges = [(1.0, 1.0), (0.5, 1.5), (0.05, 2.0),
                  (0.002, 2.0), (1e-4, 2.0), (1e-6, 2.0)]
    rows = []
    for lo, hi in ranges:
        gains, essr, sn_err, nv_err = [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s)
            pi = rng.uniform(lo, hi, n_records)
            pi /= pi.sum()
            y = (rng.random(n_records) < t_pop).astype(np.float64)
            nv = naive_estimate(y)
            sn = snips_estimate(y, pi)
            ess = effective_sample_size(pi)
            gains.append(abs(nv - t_pop) - abs(sn - t_pop))
            essr.append(ess / n_records)
            sn_err.append(abs(sn - t_pop))
            nv_err.append(abs(nv - t_pop))
        rows.append({
            "pi_min": lo, "pi_max": hi,
            "pi_ratio": float(hi / lo) if lo > 0 else float("inf"),
            "ess_ratio": round(float(np.mean(essr)), 4),
            "naive_err": round(float(np.mean(nv_err)), 4),
            "snips_err": round(float(np.mean(sn_err)), 4),
            "gain": round(float(np.mean(gains)), 4),
            "n_seeds": n_seeds,
        })
    # 找"去偏开始实质性亏本"的 ESS/N 阈值。
    #
    # ⚠️ 判定细节（实测调整过）：gain 在 ESS/N=1 时**恰为 0**（两目标重合），
    # 之后随 ESS/N 下降单调变负。因此不能用 "gain >= 0 > gain" 找跨越——
    # 那会在第一个点就触发，报出 ESS/N≈1.000 的假阈值。
    # 正确做法：以「亏损超出噪声水平」为界。
    #   noise = 用 gain 在不同 seed 下的标准差近似；这里用 rows 的 |gain| 最小值
    #   作为"无损区"宽度，取 |gain| 首次超过该宽度 3 倍的点作为边界。
    rows_sorted = sorted(rows, key=lambda r: -r["ess_ratio"])
    gains = [r["gain"] for r in rows_sorted]
    # 无损区宽度：gain 绝对值的最小非零量级（无 seed 明细时用最小 |gain| 兜底）
    loss_floor = min((abs(g) for g in gains if abs(g) > 1e-9), default=1e-4)
    tol = max(3.0 * loss_floor, 1e-3)
    cross = None
    for r in rows_sorted:
        if r["gain"] < -tol:
            cross = round(r["ess_ratio"], 4)
            break
    return {
        "t_pop": float(t_pop),
        "n_records": int(n_records),
        "rows": rows,
        "tolerance_used": round(tol, 4),
        "cross_ess_ratio": cross,
        "conclusion": (
            f"ESS/N 降至约 {cross:.3f} 以下时，去偏收益转负且超出噪声水平"
            f"（阈值容差 {tol:.4f}）——此时 IPS 的方差代价超过纠偏收益。"
            if cross is not None else
            "该区间未观察到收益实质转负；可进一步压缩 π 下界或增大 n_seeds。"
        ),
    }


def find_boundary(grid, ess_ratio_threshold: float = 0.4) -> dict:
    """从 τ×N 网格里归类每个 τ 的去偏收益与 ESS 预警位置。"""
    by_tau: dict[float, list] = {}
    for row in grid:
        by_tau.setdefault(row["tau"], []).append(row)

    out: dict[str, Any] = {"per_tau": [], "ess_threshold": float(ess_ratio_threshold)}
    for tau in sorted(by_tau):
        rows = sorted(by_tau[tau], key=lambda r: r["n_records"])
        gains = [r["gain"] for r in rows]
        first_losing = next((r["n_records"] for r in rows if r["gain"] < 0), None)
        first_low_ess = next((r["n_records"] for r in rows
                              if r["ess_ratio"] < ess_ratio_threshold), None)
        out["per_tau"].append({
            "tau": tau,
            "min_gain": round(min(gains), 4) if gains else 0.0,
            "max_gain": round(max(gains), 4) if gains else 0.0,
            "profitable_at_all_n": bool(all(g >= 0 for g in gains)),
            "first_losing_n": first_losing,
            "first_low_ess_n": first_low_ess,
            "ess_ratio_range": [round(min(r["ess_ratio"] for r in rows), 4),
                                round(max(r["ess_ratio"] for r in rows), 4)],
        })
    return out


def ess_warning_from_log(propensity) -> dict:
    """**落地接口**：只看日志的 propensity 就给出「该不该去偏」的建议。

    真实系统拿不到真值，但能算出 ESS/N。本函数把它翻译成可执行建议，
    阈值由 `boundary_by_range` 在合成环境标定（默认 0.4）。
    """
    pi = np.asarray(propensity, dtype=np.float64)
    if len(pi) == 0:
        return {"ess": 0.0, "ess_ratio": 0.0, "recommendation": "无数据"}
    ess = effective_sample_size(pi)
    ratio = ess / len(pi)
    # 阈值来自 boundary_by_range 的实测标定
    if ratio > 0.4:
        rec = "可用 IPS/SNIPS 去偏：ESS/N 充足，去偏收益大概率覆盖方差代价。"
    elif ratio > 0.05:
        rec = "谨慎：ESS/N 偏低，去偏可能亏本，建议先做小流量验证或改用 DR。"
    else:
        rec = "不建议 IPS：ESS/N 极低，1/π 方差会主导，建议回退 naive 或扩大曝光。"
    return {"ess": round(ess, 2), "ess_ratio": round(ratio, 4),
            "pi_min": round(float(pi.min()), 8), "pi_max": round(float(pi.max()), 6),
            "recommendation": rec}
