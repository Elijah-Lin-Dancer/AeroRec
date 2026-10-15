"""propensity 估计误差实验：用**估计的** π̂ 替代闭式真 π，量化代价。

背景（改进计划 #4）
------------------
主实验（`boundary.py` / `estimand_check.py`）全部使用**闭式已知**的 propensity
——这是 oracle 设定的最大优势，也是最大不真实之处：真实系统拿不到真 π，
只能从日志**估计**。本实验回答两个问题：

  Q1 用 π̂ 去偏，比用真 π 去偏多付多少误差？（SNIPS 误差增量 delta）
  Q2 用 π̂ 算的 ESS/N 预警会误判吗？（阈值 0.358 的误报/漏报率）

估计器（真实系统可得，不用任何真值信息）
----------------------------------------
π̂_i = (impr_i + λ) / (Σ impr + λ·C)，λ 为 Laplace 平滑强度。
  - λ=0（raw）：频率估计；τ 大时冷门物品 impr≈0 → π̂≈0，1/π̂ 权重爆炸；
  - λ=1（laplace）：+1 平滑，把没见过的物品拉向均匀先验。

模拟日志只含曝光与点击（与真实日志同构），估计只用日志。
口径与主实验一致：多种子（`default_rng(seed+s)`）、均值±std、bootstrap 显著性。
"""
from __future__ import annotations

import numpy as np

from src.bias.estimators import effective_sample_size, naive_estimate, snips_estimate
from src.bias.policy import popularity_propensity, sample_exposure
from src.bias.significance import bootstrap_mean_test, paired_diff_test

_R4 = 4

_LAMBDA_KEY = {0.0: "raw", 1.0: "laplace1"}


def _rows_for_seed(true_ctr, taus, n_requests: int, slots: int, lambdas,
                   ess_threshold: float, seed: int, s: int) -> list[dict]:
    rng = np.random.default_rng(seed + s)
    T = np.asarray(true_ctr, dtype=np.float64)
    truth = T.mean(axis=0) if T.ndim == 2 else T
    C = len(truth)
    K = int(min(slots, C))
    t_pop = float(truth.mean())
    out: list[dict] = []
    for tau in taus:
        pi = np.full(C, 1.0 / C) if tau == 0.0 \
            else popularity_propensity(truth, tau=tau)
        pi = pi / max(pi.sum(), 1e-12)
        # 记录粒度模拟日志（估计器看不到 π）
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
        p_true = pi[rec_item]
        nv = naive_estimate(rec_click)
        sn_true = snips_estimate(rec_click, p_true)
        ess_true = effective_sample_size(p_true) / n_rec
        imps = np.bincount(rec_item, minlength=C)
        for lam in lambdas:
            pi_hat = (imps + lam) / (n_rec + lam * C)
            sn_hat = snips_estimate(rec_click, pi_hat[rec_item])
            ess_hat = effective_sample_size(pi_hat[rec_item]) / n_rec
            out.append({
                "tau": float(tau),
                "estimator": _LAMBDA_KEY.get(float(lam), f"laplace{lam:g}"),
                "T_pop": round(t_pop, _R4),
                "naive": round(nv, _R4),
                "snips_true_err": round(abs(sn_true - t_pop), _R4),
                "snips_hat_err": round(abs(sn_hat - t_pop), _R4),
                "delta_err": round(abs(sn_hat - t_pop) - abs(sn_true - t_pop), _R4),
                "ess_ratio_true": round(ess_true, _R4),
                "ess_ratio_hat": round(ess_hat, _R4),
                "ess_drop": round(ess_hat - ess_true, _R4),
                # 0/1 值：跨种子均值即误判概率
                "warning_true": int(ess_true < ess_threshold),
                "warning_hat": int(ess_hat < ess_threshold),
                "n_records": n_rec,
            })
    return out


def _aggregate(per_seed_rows: list[list[dict]], keys: tuple[str, str]) -> list[dict]:
    n = len(per_seed_rows)
    first = per_seed_rows[0]
    out: list[dict] = []
    for i in range(len(first)):
        row: dict = {k: first[i][k] for k in keys}
        for f in first[i]:
            if f in keys or f == "n_records":
                if f == "n_records":
                    row[f] = first[i][f]
                continue
            vals = [pr[i][f] for pr in per_seed_rows]
            row[f] = round(float(np.mean(vals)), _R4)
            if n > 1:
                row[f + "_std"] = round(float(np.std(vals)), _R4)
        out.append(row)
    return out


def propensity_estimation_experiment(true_ctr, taus, n_requests: int = 5000,
                                     slots: int = 5, lambdas=(0.0, 1.0),
                                     ess_threshold: float = 0.358,
                                     seed: int = 42, n_seeds: int = 30) -> dict:
    """扫 (τ, λ)：用日志估计 π̂ 去偏/预警 vs 闭式真 π 的代价（多种子均值±std）。"""
    n = max(1, int(n_seeds))
    per_seed = [_rows_for_seed(true_ctr, taus, n_requests, slots, lambdas,
                               ess_threshold, seed, s) for s in range(n)]
    rows = _aggregate(per_seed, ("tau", "estimator"))

    sig: dict = {}
    if n > 1:
        for i, tau in enumerate(taus):
            for lam in lambdas:
                key = f"tau{tau:g}_{_LAMBDA_KEY.get(float(lam), f'laplace{lam:g}')}"
                idx = [j for j, r in enumerate(per_seed[0])
                       if r["tau"] == float(tau)
                       and r["estimator"] == _LAMBDA_KEY.get(float(lam), f"laplace{lam:g}")]
                j = idx[0]
                sig[key] = {
                    # H1: 用 π̂ 去偏比真 π 更差（delta_err > 0）
                    "delta_err_gt_zero": bootstrap_mean_test(
                        [r[j]["delta_err"] for r in per_seed], alternative="greater"),
                    # H1: ESS 预警被低估（ess_drop < 0）
                    "ess_drop_lt_zero": bootstrap_mean_test(
                        [r[j]["ess_drop"] for r in per_seed], alternative="less"),
                }
        # 配对（同种子同 τ）：SNIPS(π̂) − SNIPS(π*) 误差配对差（λ=raw，τ 最大处最严酷）
        sig["paired_snips_err_delta_max_tau"] = paired_diff_test(
            [r[-len(lambdas)]["snips_hat_err"] for r in per_seed],
            [r[-len(lambdas)]["snips_true_err"] for r in per_seed],
            alternative="greater")

    out: dict = {
        "taus": [float(x) for x in taus],
        "estimators": [_LAMBDA_KEY.get(float(x), f"laplace{x:g}") for x in lambdas],
        "rows": rows,
        "n_requests": int(n_requests),
        "slots": int(slots),
        "ess_threshold": float(ess_threshold),
        "n_seeds": n,
        "seed_base": int(seed),
        "claim": "真实系统只能从日志估计 propensity。本实验量化「用 π̂ 替代闭式真 π」"
                 "对 SNIPS 去偏误差与 ESS/N 预警的影响；warning_* 跨种子均值即误判概率。",
    }
    if n > 1:
        out["rng_policy"] = (f"common_random_numbers: 每 (seed, tau) 以 default_rng(seed+s) 重置，"
                             f"跨 {n} 种子取均值±std")
        out["significance"] = sig
    return out
