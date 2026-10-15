"""用户侧选择偏差实验：user-item 粒度的偏差生成、测量与去偏（改进计划 #5/#6）。

背景
----
主实验链路是**物品粒度**的：曝光偏差只由物品热度（π ∝ truth^τ）驱动，没有用户。
真实系统的选择偏差大量来自**用户自选**——用户按自身亲和浏览，导致
「哪些 (user, item) 对被观测」本身有偏。本实验把维度补上：

  - 结构（#5）：U 个用户 × C 个物品，真实 CTR 为矩阵 truth_ui（oracle 已知）；
  - 偏差（#6）：用户自选模型——用户 u 以 softmax(β·a_ui) 的概率浏览物品 i，
    β 为**自选强度**（类比物品粒度的 τ）：β=0 无偏差，β 越大越只看亲和高的对。

估计目标（两类，分离 estimand）
------------------------------
  - `T_pop`     = mean(truth_ui)：全体 (u,i) 对的平均真实 CTR（均匀浏览的理想值）
  - `T_exp_user`= mean_u Σ_i p_ui·truth_ui：自选分布下的平均（naive 的收敛目标）

三个估计器（核心对比，#6 的论点所在）
------------------------------------
  1. `naive`           ：日志点击均值 → 收敛到 T_exp_user（自选加权总体，高估）
  2. `snips_true`      ：记录级 propensity 用**真** p_ui → 收敛到 T_pop（正确去偏）
  3. `snips_agg`       ：记录级 propensity 用**物品边际** mean_u p_ui ——
     模拟「忽视用户维度、只在物品粒度建模曝光」的常见错误做法 → 有残余偏差。

预期：β=0 时 p_ui 均匀，三者目标重合；β↑ → naive 高估、snips_agg 残余偏差
增大、snips_true 不动——「用户侧偏差必须在 user-item 粒度建模」。

口径与主实验一致：多种子、均值±std、bootstrap 显著性、numpy-only。
"""
from __future__ import annotations

import numpy as np

from src.bias.estimators import naive_estimate, snips_estimate
from src.bias.metrics import exposure_gini
from src.bias.significance import bootstrap_mean_test

_R4 = 4


def _user_truth(base_ctr: np.ndarray, n_users: int, affinity_sigma: float,
                rng: np.random.Generator) -> np.ndarray:
    """构造 user×item 真实 CTR 矩阵（oracle）：物品基础 CTR × 用户亲和乘子。"""
    base = np.asarray(base_ctr, dtype=np.float64).ravel()
    aff = np.exp(rng.normal(0.0, affinity_sigma, size=(n_users, len(base))))
    return np.clip(base[None, :] * aff, 0.01, 0.9)


def _rows_for_seed(base_ctr, betas, n_users: int, n_sessions: int, slots: int,
                   affinity_sigma: float, seed: int, s: int) -> list[dict]:
    rng = np.random.default_rng(seed + s)
    truth = _user_truth(base_ctr, n_users, affinity_sigma, rng)  # (U, C)
    U, C = truth.shape
    K = int(min(slots, C))
    t_pop = float(truth.mean())
    out: list[dict] = []
    for beta in betas:
        # 每用户的选择分布：softmax(β·affinity)（闭式，oracle propensity）
        logits = beta * np.log(np.clip(truth / truth.mean(axis=1, keepdims=True), 1e-6, None))
        # 亲和即 truth/base；用 log-affinity 作得分，β 控制集中度
        p_u = np.exp(logits - logits.max(axis=1, keepdims=True))
        p_u /= p_u.sum(axis=1, keepdims=True)  # (U, C)
        n_rec = int(n_sessions) * K
        rec_u = np.empty(n_rec, dtype=np.int64)
        rec_item = np.empty(n_rec, dtype=np.int64)
        rec_click = np.empty(n_rec, dtype=np.float64)
        pos = 0
        for _ in range(int(n_sessions)):
            u = int(rng.integers(U))
            idx = rng.choice(C, size=K, replace=True, p=p_u[u])
            m = len(idx)
            rec_u[pos:pos + m] = u
            rec_item[pos:pos + m] = idx
            rec_click[pos:pos + m] = (rng.random(m) < truth[u, idx]).astype(np.float64)
            pos += m
        p_true_rec = p_u[rec_u, rec_item]
        # 物品边际 propensity（忽视用户维度的做法）：π_agg_i = mean_u p_ui
        p_agg_rec = p_u.mean(axis=0)[rec_item]
        t_exp_user = float(np.mean((p_u * truth).sum(axis=1)))
        nv = naive_estimate(rec_click)
        sn_true = snips_estimate(rec_click, p_true_rec)
        sn_agg = snips_estimate(rec_click, p_agg_rec)
        # (u,i) 对的曝光集中度
        pair = rec_u * C + rec_item
        freq = np.bincount(pair, minlength=U * C)
        out.append({
            "beta": float(beta),
            "T_pop": round(t_pop, _R4),
            "T_exp_user": round(t_exp_user, _R4),
            "naive": round(nv, _R4),
            "snips_true_err": round(abs(sn_true - t_pop), _R4),
            "snips_agg_err": round(abs(sn_agg - t_pop), _R4),
            "agg_residual": round(abs(sn_agg - t_pop) - abs(sn_true - t_pop), _R4),
            "exposure_gini": round(exposure_gini(freq / max(freq.sum(), 1e-12)), _R4),
            "n_records": n_rec,
        })
    return out


def _aggregate(per_seed_rows: list[list[dict]], key: str) -> list[dict]:
    n = len(per_seed_rows)
    first = per_seed_rows[0]
    out: list[dict] = []
    for i in range(len(first)):
        row: dict = {key: first[i][key]}
        for f in first[i]:
            if f == key or f == "n_records":
                if f == "n_records":
                    row[f] = first[i][f]
                continue
            vals = [pr[i][f] for pr in per_seed_rows]
            row[f] = round(float(np.mean(vals)), _R4)
            if n > 1:
                row[f + "_std"] = round(float(np.std(vals)), _R4)
        out.append(row)
    return out


def user_bias_experiment(base_ctr, betas=(0.0, 1.0, 2.0, 4.0),
                         n_users: int = 100, n_sessions: int = 2000,
                         slots: int = 5, affinity_sigma: float = 0.5,
                         seed: int = 42, n_seeds: int = 30) -> dict:
    """扫自选强度 β：用户侧偏差的生成、估计目标分离与粒度去偏对比。"""
    n = max(1, int(n_seeds))
    per_seed = [_rows_for_seed(base_ctr, betas, n_users, n_sessions, slots,
                               affinity_sigma, seed, s) for s in range(n)]
    rows = _aggregate(per_seed, "beta")

    sig: dict = {}
    if n > 1:
        for i, beta in enumerate(betas):
            j = i  # 行序与 betas 一致
            sig[f"beta{beta:g}"] = {
                # H1: naive 高估自选加权总体（naive − T_exp_user > 0 → 偏差为正）
                "naive_minus_texp": bootstrap_mean_test(
                    [r[j]["naive"] - r[j]["T_exp_user"] for r in per_seed]),
                # H1: 忽视用户维度的去偏有残余偏差（agg_residual > 0）
                "agg_residual_gt_zero": bootstrap_mean_test(
                    [r[j]["agg_residual"] for r in per_seed], alternative="greater"),
            }

    out: dict = {
        "betas": [float(x) for x in betas],
        "rows": rows,
        "n_users": int(n_users),
        "n_sessions": int(n_sessions),
        "slots": int(slots),
        "affinity_sigma": float(affinity_sigma),
        "n_seeds": n,
        "seed_base": int(seed),
        "claim": "用户自选（softmax 亲和，β 为强度）产生 user-item 粒度的选择偏差："
                 "naive 收敛到自选加权总体（高估），物品边际 propensity 去偏有残余偏差，"
                 "user-item 粒度的正确 propensity 去偏才能恢复 T_pop。",
    }
    if n > 1:
        out["rng_policy"] = (f"common_random_numbers: 每 (seed, beta) 以 default_rng(seed+s) 重置，"
                             f"跨 {n} 种子取均值±std")
        out["significance"] = sig
    return out
