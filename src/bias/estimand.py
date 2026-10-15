"""估计目标对照实验：naive 收敛到 `T_exp`、SNIPS 收敛到 `T_pop` 的实测验证。

背景（报告 §4 的历史口径补测）
------------------------------
M4 阶段的一个重要认识：naive 池化 CTR 对 `T_exp = Σ truth·π` 几乎无偏——
之前当作「偏差」的漂移不是估计量出错，而是**估计目标变了**。该结论当时的
数据（60 物品、400 请求）未落盘，本模块以固定种子补测并落盘产物，
由零漂移门禁逐字节复现。

设计
----
对每个 τ：π = popularity_propensity(truth, τ)（τ=0 均匀），模拟
`n_requests × slots` 条曝光记录（记录粒度，与真实日志同构）：

  - `naive`        = 记录点击均值            → 应贴 `T_exp`
  - `snips`        = Σ(y/π)/Σ(1/π)           → 应贴 `T_pop`
  - `T_exp`        = Σ truth·π（闭式，无需模拟）
  - `T_pop`        = mean(truth)
  - `ess` / `ess_ratio`：记录级 1/π 的有效样本量（ESS/N 即去偏预警指标）

多种子口径与 `propagation.py` 一致（`default_rng(seed+s)`，跨种子均值±std、
bootstrap 显著性；n_seeds=1 不写 std/显著性，向后兼容）。
"""
from __future__ import annotations

import numpy as np

from src.bias.estimators import effective_sample_size, naive_estimate, snips_estimate
from src.bias.policy import popularity_propensity, sample_exposure
from src.bias.significance import bootstrap_mean_test

_R4 = 4  # round 4


def _rows_for_seed(true_ctr, taus, n_requests: int, slots: int, seed: int, s: int) -> list[dict]:
    rng = np.random.default_rng(seed + s)
    T = np.asarray(true_ctr, dtype=np.float64)
    truth = T.mean(axis=0) if T.ndim == 2 else T
    K = int(min(slots, len(truth)))
    out: list[dict] = []
    for tau in taus:
        pi = np.full(len(truth), 1.0 / len(truth)) if tau == 0.0 \
            else popularity_propensity(truth, tau=tau)
        pi = pi / max(pi.sum(), 1e-12)
        # 记录粒度模拟：每请求抽 slots 个物品
        n_rec = int(n_requests) * K
        rec_item = np.empty(n_rec, dtype=np.int64)
        rec_click = np.empty(n_rec, dtype=np.float64)
        pos = 0
        for _ in range(int(n_requests)):
            idx = sample_exposure(pi, K, rng, replace=True)
            m = len(idx)
            rec_item[pos:pos + m] = idx
            rec_click[pos:pos + m] = (rng.random(m) < truth[idx]).astype(np.float64)
            pos += m
        p_rec = pi[rec_item]
        t_pop = float(truth.mean())
        t_exp = float((truth * pi).sum())
        nv = naive_estimate(rec_click)
        sn = snips_estimate(rec_click, p_rec)
        ess = effective_sample_size(p_rec)
        out.append({
            "tau": float(tau),
            "T_pop": round(t_pop, _R4),
            "T_exp": round(t_exp, _R4),
            "naive": round(nv, _R4),
            "snips": round(sn, _R4),
            "naive_err_vs_texp": round(abs(nv - t_exp), _R4),
            "snips_err_vs_tpop": round(abs(sn - t_pop), _R4),
            "ess": round(ess, _R4),
            "ess_ratio": round(ess / max(n_rec, 1), _R4),
            "n_records": n_rec,
        })
    return out


def _aggregate(per_seed_rows: list[list[dict]], key_field: str) -> list[dict]:
    n = len(per_seed_rows)
    first = per_seed_rows[0]
    out: list[dict] = []
    for i in range(len(first)):
        row: dict = {key_field: first[i][key_field]}
        for f in first[i]:
            if f == key_field:
                continue
            if f == "n_records":  # 配置量，非随机量：原样保留
                row[f] = first[i][f]
                continue
            vals = [pr[i][f] for pr in per_seed_rows]
            row[f] = round(float(np.mean(vals)), _R4)
            if n > 1:
                row[f + "_std"] = round(float(np.std(vals)), _R4)
        out.append(row)
    return out


def estimand_check(true_ctr, taus, n_requests: int = 400, slots: int = 5,
                   seed: int = 42, n_seeds: int = 30) -> dict:
    """逐 τ 验证「naive→T_exp、SNIPS→T_pop」的估计目标分离（多种子均值±std）。"""
    n = max(1, int(n_seeds))
    per_seed = [_rows_for_seed(true_ctr, taus, n_requests, slots, seed, s) for s in range(n)]
    rows = _aggregate(per_seed, "tau")

    sig: dict = {}
    if n > 1:
        for i, tau in enumerate(taus):
            sig[f"tau{tau:g}"] = {
                # H0: naive 对 T_exp 无偏（two-sided，预期不显著）
                "naive_err_vs_texp": bootstrap_mean_test(
                    [r[i]["naive_err_vs_texp"] for r in per_seed]),
                # H0: SNIPS 对 T_pop 无偏（two-sided）
                "snips_err_vs_tpop": bootstrap_mean_test(
                    [r[i]["snips_err_vs_tpop"] for r in per_seed]),
            }

    out: dict = {
        "taus": [float(x) for x in taus],
        "rows": rows,
        "n_requests": int(n_requests),
        "slots": int(slots),
        "n_seeds": n,
        "seed_base": int(seed),
        "claim": "naive 池化 CTR 收敛到 T_exp=Σtruth·π（无偏于曝光加权总体）；"
                 "SNIPS 收敛到 T_pop=mean(truth)。两者之差是 estimand 之差，非估计量误差。",
    }
    if n > 1:
        out["rng_policy"] = (f"common_random_numbers: 每 (seed, tau) 以 default_rng(seed+s) 重置，"
                             f"跨 {n} 种子取均值±std")
        out["significance"] = sig
    return out
