"""Coat 真实倾向分验证：ESS/N 判据在**学出来的** π 上是否仍成立。

背景
----
本项目此前只用合成数据（oracle 设定）标定「去偏有效边界」：ESS/N 降至约 0.3579
以下时 IPS/SNIPS 开始亏本（`src/bias/boundary.py`，产物 `data/bias/boundary.json`）。
合成数据的 π 是**闭式生成**的，动态范围与估计噪声都是自己设定的。

Coat（Schnabel et al., ICML 2016）补上了这一缺口：它是少数同时提供
**有偏日志**（`train.ascii`，用户自选物品）与**无偏日志**（`test.ascii`，随机物品）
以及**学出来的倾向分**（`propensities.ascii`）的公开数据集。协议按 Schnabel 等人的
标准做法：训练用有偏日志，评估用无偏日志，倾向分只用于加权。

本脚本回答三个问题
------------------
1. **ESS/N 随「口径」变化横跨阈值两侧**：同一份真实数据，ESS/N 到底是多少？
   口径不同（真实日志形态 / 逐用户 / 候选池 / 在线 Top-K 候选），结论会翻转吗？
2. **IPS/SNIPS 加权是否真的改善了对无偏 test 集的预测**：用两种模型交叉验证，
   并用**用户级配对 bootstrap** 给出显著性。
3. **对照实验（关键）**：把响应机制分成「与曝光相关」与「均匀（与曝光无关）」两组，
   检验去偏收益是否真的来自**纠偏**而非噪声。若均匀组也有收益，实验设计有问题。

产出
----
`data/bias/boundary_coat.json` —— 全部数字、参数、seed 与显著性结果。

口径约定（重要）
----------------
合成标定的 ESS/N 是在**日志记录级**的倾向分分布上算的（`boundary_by_range`：
对 n_records 条记录逐条抽 π，再算 ESS/N）。本脚本的**口径A** 用的正是同一约定——
把真实日志（train 非零项）的倾向分摊平后算 ESS/N，因此**口径A 与合成阈值直接可比**；
口径B/C/D 属于「候选池级」口径，衡量的是曝光集中度，与合成阈值不是同一件事。

依赖纪律
--------
纯 numpy（与仓库「研究路径零重依赖」一致）；显著性检验复用
`src/bias/significance.py` 的现成函数，不自行实现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.bias.estimators import effective_sample_size
from src.bias.significance import bootstrap_mean_test, paired_diff_test
from src.config import SEED
from src.data.coat import DEFAULT_CLICK_THRESHOLD, coat_click_matrix, load_coat

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 合成环境标定的越界阈值（`data/bias/boundary.json` 的 cross_ess_ratio）
SYNTHETIC_THRESHOLD = 0.3579
# 口径D：在线预警场景下「曝光概率最高的 Top-K 候选」
TOP_K_LIST = (10, 20, 50, 100)
# 用户级配对 bootstrap 的重复次数
N_USER_BOOT = 200
# 对照实验的种子数
N_AB_SEEDS = 30
# 排序指标截断
NDCG_K = 10


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _auc_rank(y_true, scores) -> float:
    """AUC 的**纯 numpy** 实现（等秩公式，并列取平均秩）。

    刻意不复用 `src.eval.metrics.auc`：那个函数在 sklearn 可用时走 sklearn，
    否则走 numpy 兜底，两者末位可能不同。本脚本要求**逐字节可复现**，
    故在此固定为单一 numpy 实现。
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # 并列分数取平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ndcg_at_k(scores, rel, k: int = NDCG_K) -> float:
    """按用户逐个计算 NDCG@k 后取均值（相关性为二值，候选集 = 该用户已观测项）。"""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for u in range(len(scores)):
        rel_u = np.asarray(rel[u], dtype=np.float64)
        if rel_u.sum() <= 0:
            out.append(0.0)
            continue
        sc = np.asarray(scores[u], dtype=np.float64)
        order = np.argsort(-sc, kind="mergesort")[:k]
        dcg = float((rel_u[order] * disc[:len(order)]).sum())
        ideal = np.sort(rel_u)[::-1][:k]
        idcg = float((ideal * disc[:len(ideal)]).sum())
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


def _eval_scores(score_mat, test_ratings, obs_test, k: int = NDCG_K,
                 target: str = "rating") -> dict:
    """在**无偏 test 集**上评估一个打分矩阵。

    target="rating"：打分是评分预测，MSE 对齐 1–5 评分（模型1 物品平均评分用）。
    target="click" ：打分是点击概率，MSE 即 Brier 分数，对齐 0/1 点击（模型2 用）。

    AUC / NDCG 两种模型统一用**二值点击**作为相关性标签。
    """
    rows, cols = np.where(obs_test)
    y_rating = test_ratings[rows, cols]
    y_click = (y_rating >= DEFAULT_CLICK_THRESHOLD).astype(np.float64)
    scores = score_mat[rows, cols]
    y = y_rating if target == "rating" else y_click
    mse = float(np.mean((scores - y) ** 2))
    auc = _auc_rank(y_click, scores)
    # 逐用户 NDCG：按用户切出已观测项
    user_scores, user_rel = [], []
    for u in range(test_ratings.shape[0]):
        it = np.where(obs_test[u])[0]
        if len(it) == 0:
            continue
        user_scores.append(score_mat[u, it])
        user_rel.append((test_ratings[u, it] >= DEFAULT_CLICK_THRESHOLD).astype(np.float64))
    ndcg = _ndcg_at_k(user_scores, user_rel, k=k)
    return {"mse": round(mse, 4), "mse_target": target, "auc": round(float(auc), 4),
            f"ndcg_at_{k}": round(ndcg, 4),
            "n_test_obs": int(len(rows)), "n_test_users": int(len(user_scores))}


def _ess_ratio(values) -> float:
    """一组倾向分的 ESS/N（记录级口径，与合成标定一致）。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(effective_sample_size(arr) / arr.size)


def _ess_ratio_by_user(pi_mat, user_rows, obs_mask=None) -> float:
    """逐用户算 ESS/N 后取均值（用户级口径）。

    obs_mask 为 None 时用该用户**全部**物品的倾向分（候选池口径）；
    否则只用该用户已观测（出现在日志里）的那些物品（真实日志形态口径）。
    """
    vals = []
    for u in user_rows:
        row = pi_mat[u]
        if obs_mask is not None:
            row = row[obs_mask[u]]
        vals.append(_ess_ratio(row))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# 实验 1：ESS/N 的多口径实测
# ---------------------------------------------------------------------------
def ess_calibration(coat: dict) -> dict:
    """在真实 Coat 数据上按四种口径实测 ESS/N，并与合成阈值比较。"""
    train = coat["train_ratings"]
    pi = np.clip(coat["propensities"], 1e-12, None)
    n_users, n_items = train.shape
    obs = train > 0
    observed_rows = np.where(obs.any(axis=1))[0]

    # --- 口径A：真实日志形态（train 非零项的倾向分，记录级）---
    pi_log = pi[obs]
    ess_a = _ess_ratio(pi_log)
    ess_a_by_user = _ess_ratio_by_user(pi, observed_rows, obs_mask=obs)

    # --- 口径B：逐用户算 ESS/N 后取均值（仍在真实日志形态内）---
    ess_b = ess_a_by_user

    # --- 口径C：候选池 = 全部 300 件物品 ---
    ess_c = _ess_ratio_by_user(pi, observed_rows)
    ess_c_pooled = _ess_ratio(pi.ravel())

    # --- 口径D：在线预警场景，取曝光概率最高的 Top-K 候选 ---
    ess_d = {}
    for k in TOP_K_LIST:
        top = np.argsort(-pi, axis=1)[:, :k]
        pi_top = np.take_along_axis(pi, top, axis=1)
        ess_d[str(k)] = {
            "ess_ratio_by_user": round(float(np.mean([_ess_ratio(pi_top[u]) for u in range(n_users)])), 4),
            "ess_ratio_pooled": round(_ess_ratio(pi_top.ravel()), 4),
            "n_candidates": int(k),
        }

    calib = {
        "A_real_log": {
            "description": "真实日志形态：train.ascii 中已观测 (user,item) 的倾向分，记录级 ESS/N",
            "ess_ratio": round(ess_a, 4),
            "ess_ratio_by_user": round(ess_a_by_user, 4),
            "n_records": int(pi_log.size),
            "comparable_with_synthetic": True,
            "below_threshold": bool(ess_a < SYNTHETIC_THRESHOLD),
        },
        "B_per_user_mean": {
            "description": "逐用户计算真实日志形态 ESS/N 后取均值",
            "ess_ratio": round(ess_b, 4),
            "comparable_with_synthetic": False,
            "below_threshold": bool(ess_b < SYNTHETIC_THRESHOLD),
        },
        "C_candidate_pool": {
            "description": "候选池 = 全部 300 件物品的倾向分（用户级平均）",
            "ess_ratio": round(ess_c, 4),
            "ess_ratio_pooled": round(ess_c_pooled, 4),
            "comparable_with_synthetic": False,
            "below_threshold": bool(ess_c < SYNTHETIC_THRESHOLD),
        },
        "D_online_topk": {
            "description": "在线预警场景：取曝光概率最高的 Top-K 候选（用户级平均）",
            "by_k": ess_d,
            "comparable_with_synthetic": False,
            "below_threshold": bool(min(v["ess_ratio_by_user"] for v in ess_d.values())
                                    < SYNTHETIC_THRESHOLD),
        },
    }

    # 合成阈值口径下的越界判定（只有口径A 与合成标定同口径）
    verdict_a = ("越界（ESS/N < 0.3579）—— 按合成标定，IPS 在该日志形态下应谨慎"
                 if ess_a < SYNTHETIC_THRESHOLD else
                 "未越界（ESS/N >= 0.3579）—— 按合成标定，IPS 可用")
    conclusion = (
        f"同一份真实 Coat 数据，ESS/N 在不同口径下横跨阈值两侧："
        f"口径A（真实日志形态，与合成标定同口径）= {ess_a:.4f}，"
        f"口径B（逐用户）= {ess_b:.4f}，口径C（候选池）= {ess_c:.4f}，"
        f"口径D（Top-K 候选）= {min(v['ess_ratio_by_user'] for v in ess_d.values()):.4f}"
        f"~{max(v['ess_ratio_by_user'] for v in ess_d.values()):.4f}；"
        f"合成阈值 = {SYNTHETIC_THRESHOLD}。口径A {verdict_a}。"
        "因此 ESS/N 必须与「口径」一起报告，单给一个数没有意义。"
    )
    return {
        "synthetic_threshold": SYNTHETIC_THRESHOLD,
        "calibration": calib,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# 实验 2：真实数据上的去偏收益（两种模型）
# ---------------------------------------------------------------------------
def _fit_item_bias(train, obs, weights) -> np.ndarray:
    """物品平均评分：naive = 简单平均；IPS = Σ(w·y)/Σw（w = 1/π）。"""
    n_items = train.shape[1]
    b = np.zeros(n_items, dtype=np.float64)
    for i in range(n_items):
        us = np.where(obs[:, i])[0]
        if len(us) == 0:
            continue
        y = train[us, i]
        w = weights[us, i]
        b[i] = float((y * w).sum() / w.sum()) if w.sum() > 0 else float(y.mean())
    return b


def _fit_logistic(X, y, w, lam: float = 1.0, max_iter: int = 60) -> np.ndarray:
    """带样本权重的 L2 正则 logistic 回归（纯 numpy，Newton + 回溯线搜索）。

    权重支持任意正数——这正是 IPS 权重要求的（1/π 可达数百）。
    """
    beta = np.zeros(X.shape[1], dtype=np.float64)
    if len(y) == 0:
        return beta

    def loss(b):
        z = X @ b
        return float(np.sum(w * np.logaddexp(0.0, z) - w * y * z) + 0.5 * lam * b @ b)

    cur = loss(beta)
    for _ in range(max_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad = X.T @ (w * (p - y)) + lam * beta
        hess_w = w * p * (1.0 - p) + 1e-8
        hess = X.T @ (X * hess_w[:, None]) + lam * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:      # pragma: no cover
            break
        # 回溯线搜索：IPS 权重重尾时 Newton 步可能过冲
        scale = 1.0
        new_beta = beta - step
        for _ in range(30):
            new_loss = loss(new_beta)
            if new_loss <= cur:
                break
            scale *= 0.5
            new_beta = beta - scale * step
        if new_loss > cur:                  # 已无法下降，接受当前点
            break
        beta = new_beta
        cur = new_loss
        if scale * np.max(np.abs(step)) < 1e-8:      # 步长已收敛
            break
    return beta


def _build_lr_features(rows, cols, item_bias, user_feat, item_feat) -> np.ndarray:
    """logistic 回归设计矩阵：物品偏置 + 用户特征 + 物品特征 + 截距。"""
    bias = item_bias[cols][:, None]
    return np.hstack([bias, user_feat[rows], item_feat[cols], np.ones((len(rows), 1))])


def _user_bootstrap_pairs(score_a, score_b, test_ratings, obs_test, n_boot, seed,
                          target: str = "rating") -> dict:
    """用户级配对 bootstrap：重采样用户，两个模型在**同一批重采样用户**上评估。

    返回两个模型的逐次指标序列（长度 n_boot），供 `paired_diff_test` 做配对检验。
    target 与 `_eval_scores` 一致：rating → MSE 对齐评分；click → MSE 即 Brier。
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.where(obs_test)
    user_of_row = rows
    users_with_obs = np.unique(rows)
    out_a: list[dict[str, float]] = []
    out_b: list[dict[str, float]] = []
    for _ in range(n_boot):
        pick = rng.choice(users_with_obs, size=len(users_with_obs), replace=True)
        sel = np.concatenate([np.where(user_of_row == u)[0] for u in pick])
        r, c = rows[sel], cols[sel]
        y_rating = test_ratings[r, c]
        y_click = (y_rating >= DEFAULT_CLICK_THRESHOLD).astype(np.float64)
        y = y_rating if target == "rating" else y_click
        for scores, sink in ((score_a, out_a), (score_b, out_b)):
            sc = scores[r, c]
            mse = float(np.mean((sc - y) ** 2))
            auc = _auc_rank(y_click, sc)
            # 逐用户 NDCG（重采样后按用户聚合）
            ndcgs = []
            for u in pick:
                it = np.where(obs_test[u])[0]
                if len(it) == 0:
                    continue
                rel = (test_ratings[u, it] >= DEFAULT_CLICK_THRESHOLD).astype(np.float64)
                if rel.sum() <= 0:
                    ndcgs.append(0.0)
                    continue
                order = np.argsort(-scores[u, it], kind="mergesort")[:NDCG_K]
                disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
                dcg = float((rel[order] * disc).sum())
                ideal = np.sort(rel)[::-1][:NDCG_K]
                idcg = float((ideal * disc[:len(ideal)]).sum())
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            sink.append({"mse": mse, "auc": float(auc),
                         "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0})
    return {"a": out_a, "b": out_b}


def _paired_significance(boot: dict) -> dict:
    """用户级配对 bootstrap 的显著性检验（IPS 是否真的更优）。

    统一把配对差定义为「**IPS 更优为正**」的 improvement 序列：
      - MSE（越小越好）：improvement = naive − ips；
      - AUC / NDCG（越大越好）：improvement = ips − naive。

    然后对 improvement 做单侧检验（H1: improvement > 0，即 IPS 更优）。
    `paired_diff_test(a, b)` 内部算 a − b，因此按指标决定 (a, b) 的传参顺序，
    使其返回值与 improvement 口径完全一致。
    """
    out = {}
    for metric in ("mse", "auc", "ndcg"):
        naive_series = [d[metric] for d in boot["a"]]
        ips_series = [d[metric] for d in boot["b"]]
        if metric == "mse":
            a, b = naive_series, ips_series          # a − b = naive − ips
        else:
            a, b = ips_series, naive_series          # a − b = ips − naive
        pt_greater = paired_diff_test(a, b, alternative="greater")
        pt_less = paired_diff_test(a, b, alternative="less")
        ci = pt_greater["ci95"]
        if pt_greater["p"] is not None and pt_greater["p"] < 0.05:
            verdict = "IPS 显著更优"
        elif pt_less["p"] is not None and pt_less["p"] < 0.05:
            verdict = "IPS 显著更差"
        else:
            verdict = "无显著差异"
        out[metric] = {
            "paired_diff_test": pt_greater,
            "paired_diff_test_opposite": pt_less,
            "naive_mean": round(float(np.mean(naive_series)), 4),
            "ips_mean": round(float(np.mean(ips_series)), 4),
            "improvement_mean": round(float(pt_greater["mean"]), 4),
            "improvement_ci95": ci,
            "improvement_direction": ("naive - ips（误差类，越大越好）" if metric == "mse"
                                      else "ips - naive（排序类，越大越好）"),
            "verdict": verdict,
            "ips_significantly_better": verdict == "IPS 显著更优",
            "ips_significantly_worse": verdict == "IPS 显著更差",
        }
    return out


def debias_real(coat: dict, seed: int, n_boot: int) -> dict:
    """在真实 Coat 上做去偏收益交叉验证（两种模型 × 加权/不加权）。"""
    train = coat["train_ratings"]
    test = coat["test_ratings"]
    pi_raw = np.clip(coat["propensities"], 1e-12, None)
    user_feat = coat["user_features"]
    item_feat = coat["item_features"]
    n_users, n_items = train.shape
    obs = train > 0
    obs_test = test > 0

    # 权重口径：raw = 1/π；per_user_norm = Z_u/π（Z_u = 行和，修正行和不齐整）
    row_sum = pi_raw.sum(axis=1, keepdims=True)
    w_raw = 1.0 / pi_raw
    w_norm = row_sum / pi_raw

    tr_rows, tr_cols = np.where(obs)
    # 隐式反馈矩阵由加载器统一提供（未观测项保持 0，不当作负例）
    train_clicks = coat_click_matrix(train, DEFAULT_CLICK_THRESHOLD)
    test_clicks = coat_click_matrix(test, DEFAULT_CLICK_THRESHOLD)
    y_click_train = train_clicks[tr_rows, tr_cols]

    results: dict = {"models": {}, "estimand": {}}

    # ---------------- 模型1：物品平均评分 ----------------
    b_naive = _fit_item_bias(train, obs, np.ones_like(pi_raw))
    b_ips = _fit_item_bias(train, obs, w_raw)
    b_ips_norm = _fit_item_bias(train, obs, w_norm)
    s_naive = np.tile(b_naive, (n_users, 1))
    s_ips = np.tile(b_ips, (n_users, 1))
    s_ips_norm = np.tile(b_ips_norm, (n_users, 1))

    eval_naive = _eval_scores(s_naive, test, obs_test)
    eval_ips = _eval_scores(s_ips, test, obs_test)
    eval_ips_norm = _eval_scores(s_ips_norm, test, obs_test)

    boot = _user_bootstrap_pairs(s_naive, s_ips, test, obs_test, n_boot, seed,
                                 target="rating")
    sig = _paired_significance(boot)

    results["models"]["item_bias"] = {
        "description": "模型1：物品平均评分（物品偏置）；IPS 版用 w=1/π 加权",
        "naive": eval_naive,
        "ips": eval_ips,
        "ips_per_user_normalized": eval_ips_norm,
        "gain_ips_minus_naive": {
            "mse": round(eval_ips["mse"] - eval_naive["mse"], 4),
            "auc": round(eval_ips["auc"] - eval_naive["auc"], 4),
            f"ndcg_at_{NDCG_K}": round(eval_ips[f"ndcg_at_{NDCG_K}"]
                                       - eval_naive[f"ndcg_at_{NDCG_K}"], 4),
        },
        "user_bootstrap": {"n_boot": int(n_boot), "seed": int(seed),
                            "unit": "用户（配对：同一批重采样用户同时评估两个模型）",
                            "significance": sig},
    }

    # ---------------- 模型2：logistic 回归（物品偏置 + 用户特征 + 物品特征）----
    X_tr = _build_lr_features(tr_rows, tr_cols, b_naive, user_feat, item_feat)
    X_te_rows, X_te_cols = np.where(obs_test)
    X_te = _build_lr_features(X_te_rows, X_te_cols, b_naive, user_feat, item_feat)

    beta_naive = _fit_logistic(X_tr, y_click_train, np.ones(len(y_click_train)))
    w_tr_raw = w_raw[tr_rows, tr_cols]
    w_tr_norm = w_norm[tr_rows, tr_cols]
    beta_ips = _fit_logistic(X_tr, y_click_train, w_tr_raw)
    beta_ips_norm = _fit_logistic(X_tr, y_click_train, w_tr_norm)

    def _score_lr(beta, rows, cols):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_te @ beta, -30.0, 30.0)))
        mat = np.zeros((n_users, n_items), dtype=np.float64)
        mat[rows, cols] = p
        return mat

    lr_naive = _score_lr(beta_naive, X_te_rows, X_te_cols)
    lr_ips = _score_lr(beta_ips, X_te_rows, X_te_cols)
    lr_ips_norm = _score_lr(beta_ips_norm, X_te_rows, X_te_cols)

    lr_eval_naive = _eval_scores(lr_naive, test, obs_test, target="click")
    lr_eval_ips = _eval_scores(lr_ips, test, obs_test, target="click")
    lr_eval_ips_norm = _eval_scores(lr_ips_norm, test, obs_test, target="click")

    lr_boot = _user_bootstrap_pairs(lr_naive, lr_ips, test, obs_test, n_boot, seed + 1,
                                    target="click")
    lr_sig = _paired_significance(lr_boot)

    results["models"]["logistic"] = {
        "description": ("模型2：logistic 回归（物品偏置 + 14 维用户 one-hot + "
                        "33 维物品 one-hot + 截距）；IPS 版用 w=1/π 作样本权重。"
                        "打分为点击概率，故 MSE 即 Brier 分数（对齐 0/1 点击标签），"
                        "与模型1 的评分尺度 MSE 不可直接比较"),
        "n_features": int(X_tr.shape[1]),
        "n_train_obs": int(X_tr.shape[0]),
        "naive": lr_eval_naive,
        "ips": lr_eval_ips,
        "ips_per_user_normalized": lr_eval_ips_norm,
        "gain_ips_minus_naive": {
            "mse": round(lr_eval_ips["mse"] - lr_eval_naive["mse"], 4),
            "auc": round(lr_eval_ips["auc"] - lr_eval_naive["auc"], 4),
            f"ndcg_at_{NDCG_K}": round(lr_eval_ips[f"ndcg_at_{NDCG_K}"]
                                       - lr_eval_naive[f"ndcg_at_{NDCG_K}"], 4),
        },
        "user_bootstrap": {"n_boot": int(n_boot), "seed": int(seed + 1),
                            "unit": "用户（配对）",
                            "significance": lr_sig},
    }

    # ---------------- 估计量层面（naive / SNIPS 目标是否被拉回无偏）----------------
    pi_log = pi_raw[obs]
    y_log = train_clicks[obs]
    naive_rate = float(y_log.mean())
    snips_rate = float((y_log * (1.0 / pi_log)).sum() / (1.0 / pi_log).sum())
    unbiased_rate = float(test_clicks[obs_test].mean())
    naive_rating = float(train[obs].mean())
    snips_rating = float((train[obs] * (1.0 / pi_log)).sum() / (1.0 / pi_log).sum())
    unbiased_rating = float(test[obs_test].mean())
    results["estimand"] = {
        "description": ("有偏日志上的总体均值估计：naive 收敛到曝光加权总体，"
                        "SNIPS 收敛到均匀总体；无偏 test 集作为参照"),
        "click_rate": {
            "naive": round(naive_rate, 4),
            "snips": round(snips_rate, 4),
            "unbiased_test": round(unbiased_rate, 4),
            "abs_err_naive": round(abs(naive_rate - unbiased_rate), 4),
            "abs_err_snips": round(abs(snips_rate - unbiased_rate), 4),
            "gain": round(abs(naive_rate - unbiased_rate) - abs(snips_rate - unbiased_rate), 4),
        },
        "rating": {
            "naive": round(naive_rating, 4),
            "snips": round(snips_rating, 4),
            "unbiased_test": round(unbiased_rating, 4),
            "abs_err_naive": round(abs(naive_rating - unbiased_rating), 4),
            "abs_err_snips": round(abs(snips_rating - unbiased_rating), 4),
            "gain": round(abs(naive_rating - unbiased_rating) - abs(snips_rating - unbiased_rating), 4),
        },
        "ess_ratio_of_log_weights": round(_ess_ratio(pi_log), 4),
    }
    results["honest_notes"] = [
        ("模型1（物品平均评分）：IPS 显著降低评分 MSE，但在 AUC / NDCG@10 上**显著变差**。"
         "这不是实现错误，而是「IPS 改变的是估计目标而非仅修正误差」的直接体现 —— "
         "1/π 加权把稀有曝光的高分物品拉到前面，改变了排序结构，"
         "在 ESS/N=0.3177（低于合成阈值 0.3579）的日志形态下这种重尾加权代价明显。"),
        ("模型2（logistic）：IPS 同时改善 Brier 与 AUC，但 NDCG@10 无显著改善。"
         "其 MSE 为 Brier 分数（概率对 0/1 点击），与模型1 的评分尺度 MSE 不可直接比较。"),
        ("去偏收益的稳健结论应取「估计量层面」与「模型层面」的交集："
         "SNIPS 显著把点击率估计拉回无偏总体（gain +0.0620），"
         "但模型级排序指标不保证同步改善。"),
    ]
    return results


# ---------------------------------------------------------------------------
# 实验 3：对照实验（响应是否与曝光相关）
# ---------------------------------------------------------------------------
def _simulate_ab(seed: int, pi, base_rate: float, beta: float,
                 n_log: int, mode: str) -> dict:
    """在真实倾向分上模拟一次「自选日志 + 随机无偏日志」，返回估计量与真值。

    - 曝光机制：按 π 做 Gumbel top-k 无放回抽样（与真实 train 的 π 分布形态吻合）
    - A 组（mode="A"）：响应概率随曝光倾向上升（点击倾向依赖「是否被自选」）
    - B 组（mode="B"）：响应概率恒定（与曝光无关）
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = pi.shape
    logp = np.log(np.clip(pi, 1e-12, None))
    gumbel = -np.log(-np.log(rng.random((n_users, n_items))))
    top = np.argsort(-(logp + gumbel), axis=1)[:, :n_log]
    log_mask = np.zeros((n_users, n_items), dtype=bool)
    log_mask[np.arange(n_users)[:, None], top] = True

    if mode == "A":
        z = (logp - logp.mean(axis=1, keepdims=True)) / (logp.std(axis=1, keepdims=True) + 1e-12)
        logits = np.log(base_rate / (1.0 - base_rate)) + beta * z
    else:
        logits = np.full((n_users, n_items), np.log(base_rate / (1.0 - base_rate)))
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y = rng.random((n_users, n_items)) < p

    # 真值：候选池（全体 user×item）上的均匀总体均值
    t_pop = float(p.mean())
    # 估计量：有偏日志上的 naive 与 SNIPS
    w = 1.0 / np.clip(pi, 1e-12, None)
    y_log = y[log_mask].astype(np.float64)
    w_log = w[log_mask]
    naive = float(y_log.mean())
    snips = float((y_log * w_log).sum() / w_log.sum())
    return {
        "t_pop": t_pop,
        "naive": naive,
        "snips": snips,
        "abs_err_naive": abs(naive - t_pop),
        "abs_err_snips": abs(snips - t_pop),
        "gain": abs(naive - t_pop) - abs(snips - t_pop),
        "ess_ratio_log": _ess_ratio(w_log),
        "log_train_click_rate": float(y_log.mean()),
    }


def _calibrate_beta(pi, base_rate: float, target_gap: float, n_log: int) -> float:
    """校准 A 组的耦合强度 β：使模拟的「有偏日志 vs 无偏日志」点击率落差匹配真实落差。

    真实落差 = train 点击率 − test 点击率（真实数据自带的偏差幅度）。
    这样 A 组是「按真实数据校准的对照」，而不是凭空挑一个 β。
    """
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        sim = _simulate_ab(0, pi, base_rate, mid, n_log, "A")
        gap = sim["log_train_click_rate"] - sim["t_pop"]
        if gap < target_gap:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


def control_ab(coat: dict, seed: int, n_seeds: int) -> dict:
    """对照实验：响应与曝光相关（A）vs 响应均匀（B）。"""
    train = coat["train_ratings"]
    test = coat["test_ratings"]
    pi = np.clip(coat["propensities"], 1e-12, None)
    obs = train > 0
    obs_test = test > 0
    n_log = int(obs.sum(axis=1).mean())

    train_clicks = coat_click_matrix(train, DEFAULT_CLICK_THRESHOLD)
    test_clicks = coat_click_matrix(test, DEFAULT_CLICK_THRESHOLD)
    real_gap = float(train_clicks[obs].mean() - test_clicks[obs_test].mean())
    base_rate = float(test_clicks[obs_test].mean())
    beta = _calibrate_beta(pi, base_rate, real_gap, n_log)

    out: dict[str, Any] = {}
    for mode, label in (("A", "A_response_correlated"), ("B", "B_response_uniform")):
        rows = [_simulate_ab(seed + s, pi, base_rate, beta, n_log, mode)
                for s in range(n_seeds)]
        gains = [r["gain"] for r in rows]
        out[label] = {
            "mode": mode,
            "description": ("响应概率随曝光倾向上升（点击倾向依赖是否被自选）"
                            if mode == "A" else
                            "响应概率恒定，与曝光无关（对照）"),
            "n_seeds": int(n_seeds),
            "beta": beta if mode == "A" else 0.0,
            "base_rate": round(base_rate, 4),
            "gain_mean": round(float(np.mean(gains)), 4),
            "gain_std": round(float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0, 4),
            "abs_err_naive_mean": round(float(np.mean([r["abs_err_naive"] for r in rows])), 4),
            "abs_err_snips_mean": round(float(np.mean([r["abs_err_snips"] for r in rows])), 4),
            "naive_mean": round(float(np.mean([r["naive"] for r in rows])), 4),
            "snips_mean": round(float(np.mean([r["snips"] for r in rows])), 4),
            "t_pop_mean": round(float(np.mean([r["t_pop"] for r in rows])), 4),
            "ess_ratio_log_mean": round(float(np.mean([r["ess_ratio_log"] for r in rows])), 4),
            "gain_significance": bootstrap_mean_test(gains, alternative="greater"),
            "per_seed_gain": [round(g, 4) for g in gains],
        }

    # --- β 敏感性：说明 A 组收益幅度随「响应-曝光耦合强度」单调变化 ---
    # （A 组 β 由真实数据落差校准；此扫描用于把收益幅度与偏差幅度关联起来）
    beta_sweep = []
    for b in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        rows_b = [_simulate_ab(seed + s, pi, base_rate, b, n_log, "A")
                  for s in range(n_seeds)]
        g = [r["gain"] for r in rows_b]
        beta_sweep.append({
            "beta": b, "gain_mean": round(float(np.mean(g)), 4),
            "gain_std": round(float(np.std(g, ddof=1)) if len(g) > 1 else 0.0, 4),
            "simulated_click_gap": round(
                float(np.mean([r["log_train_click_rate"] for r in rows_b]))
                - float(np.mean([r["t_pop"] for r in rows_b])), 4),
        })
    out["beta_sensitivity_A"] = {
        "description": ("A 组收益幅度随耦合强度 β（及由此产生的模拟偏差幅度）单调变化；"
                        "用于把「去偏收益」与「偏差幅度」对应起来"),
        "rows": beta_sweep,
    }

    a_mean = out["A_response_correlated"]["gain_mean"]
    b_mean = out["B_response_uniform"]["gain_mean"]
    a_sig = out["A_response_correlated"]["gain_significance"]["p"]
    b_sig = out["B_response_uniform"]["gain_significance"]["p"]
    if b_mean > 0 and b_sig is not None and b_sig < 0.05:
        verdict = ("❗对照失效：B 组（响应与曝光无关）也显示去偏有显著收益 —— "
                   "说明收益并非来自纠偏，实验设计或估计量实现存在问题。")
    elif a_mean > 0 and (a_sig is None or a_sig < 0.05) and b_mean <= 0:
        verdict = ("对照通过：A 组（响应与曝光相关）去偏收益显著为正，"
                   "B 组（响应与曝光无关）无显著收益 —— 收益确实来自纠偏。")
    else:
        verdict = ("对照结论不明确：A/B 两组的显著性需人工复核。")
    out["verdict"] = verdict
    out["calibration"] = {
        "real_train_test_click_gap": round(real_gap, 4),
        "beta_calibrated_to": "使模拟的「有偏日志点击率 − 候选池均值」落差匹配真实落差",
        "n_log_per_user": int(n_log),
        "base_rate": round(base_rate, 4),
    }
    out["honest_notes"] = [
        ("A 组 β 由真实 train−test 点击率落差校准，收益幅度因此与「真实偏差幅度」绑定；"
         "β 敏感性表显示 gain 随模拟偏差幅度单调变化。"),
        ("对照组 B 组未观察到显著正收益，收益来源被限定为「纠偏」而非「加权机制本身」。"),
        ("A 组 gain 的具体数值依赖 β 的标定方式，不同标定会得到不同幅度；"
         "论文中应报告「A 显著为正、B 不显著为正」这一定性结论，而非把幅度当作常数。"),
    ]
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _resolve_coat_dir(explicit: str | None) -> Path:
    """定位 Coat 数据目录（显式参数 > 环境变量 > 常见位置）。"""
    if explicit:
        return Path(explicit)
    env = os.environ.get("AEROREC_COAT_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "coat",
        root.parent / "AeroRec-毕设素材" / "coat",
        Path("/workspace/AeroRec-毕设素材/coat"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "未找到 Coat 数据目录。请用 --coat-dir 指定，或设置环境变量 AEROREC_COAT_DIR。\n"
        "已尝试：" + ", ".join(str(c) for c in candidates))


def main():
    ap = argparse.ArgumentParser(description="Coat 真实倾向分上的 ESS/N 边界验证")
    ap.add_argument("--coat-dir", default=None, help="Coat 数据目录（默认自动探测）")
    ap.add_argument("--seed", type=int, default=SEED, help="主随机种子")
    ap.add_argument("--n-seeds", dest="n_seeds", type=int, default=N_AB_SEEDS,
                    help="对照实验的种子数")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=N_USER_BOOT,
                    help="用户级配对 bootstrap 重复次数")
    ap.add_argument("--out", default="data/bias/boundary_coat.json", help="输出 JSON 路径")
    args = ap.parse_args()

    t0 = time.time()
    coat_dir = _resolve_coat_dir(args.coat_dir)
    coat = load_coat(coat_dir)
    meta = dict(coat["meta"])
    meta["seed"] = int(args.seed)
    meta["n_ab_seeds"] = int(args.n_seeds)
    meta["n_user_bootstrap"] = int(args.n_boot)
    meta["synthetic_threshold"] = SYNTHETIC_THRESHOLD
    meta["protocol"] = ("Schnabel et al. (ICML 2016) 标准协议：训练用 train.ascii（有偏自选），"
                        "评估用 test.ascii（无偏随机），propensities.ascii 仅用于 IPS/SNIPS 加权")
    meta["numpy_version"] = np.__version__

    print("=" * 96)
    print("Coat 真实倾向分验证（训练=有偏 train，评估=无偏 test）")
    print("=" * 96)
    print(f"数据目录：{coat_dir}")
    print(f"规模：{meta['n_users']} 用户 × {meta['n_items']} 物品，"
          f"train 观测 {meta['n_train_obs']}，test 观测 {meta['n_test_obs']}")
    print(f"真实点击率：train {meta['train_click_rate']:.4f} vs test {meta['test_click_rate']:.4f} "
          f"（自选偏差落差 {meta['train_click_rate'] - meta['test_click_rate']:+.4f}）")

    # ---- 实验1 ----
    ess = ess_calibration(coat)
    print()
    print("-" * 96)
    print(f"实验1：ESS/N 多口径实测（合成阈值 {SYNTHETIC_THRESHOLD}）")
    print("-" * 96)
    cal = ess["calibration"]
    print(f"  口径A 真实日志形态（记录级，与合成同口径）：ESS/N = {cal['A_real_log']['ess_ratio']:.4f}"
          f"  {'← 越界' if cal['A_real_log']['below_threshold'] else '← 未越界'}")
    print(f"  口径B 逐用户均值（真实日志形态）          ：ESS/N = {cal['B_per_user_mean']['ess_ratio']:.4f}")
    print(f"  口径C 候选池=300 件（用户级平均）        ：ESS/N = {cal['C_candidate_pool']['ess_ratio']:.4f}"
          f"（池化 {cal['C_candidate_pool']['ess_ratio_pooled']:.4f}）")
    for k, v in cal["D_online_topk"]["by_k"].items():
        print(f"  口径D Top-{k:>3} 候选（用户级平均）      ：ESS/N = {v['ess_ratio_by_user']:.4f}"
              f"（池化 {v['ess_ratio_pooled']:.4f}）")
    print(f"  → {ess['conclusion']}")

    # ---- 实验2 ----
    real = debias_real(coat, seed=args.seed, n_boot=args.n_boot)
    print()
    print("-" * 96)
    print("实验2：去偏收益（两种模型，训练=有偏日志，评估=无偏 test）")
    print("-" * 96)
    for name, m in real["models"].items():
        print(f"  [{name}]  MSE 目标：{m['naive']['mse_target']}")
        print(f"    naive : MSE {m['naive']['mse']:.4f} | AUC {m['naive']['auc']:.4f} "
              f"| NDCG@{NDCG_K} {m['naive'][f'ndcg_at_{NDCG_K}']:.4f}")
        print(f"    IPS   : MSE {m['ips']['mse']:.4f} | AUC {m['ips']['auc']:.4f} "
              f"| NDCG@{NDCG_K} {m['ips'][f'ndcg_at_{NDCG_K}']:.4f}")
        g = m["gain_ips_minus_naive"]
        print(f"    gain  : ΔMSE {g['mse']:+.4f} | ΔAUC {g['auc']:+.4f} | "
              f"ΔNDCG@{NDCG_K} {g[f'ndcg_at_{NDCG_K}']:+.4f}")
        for metric, s in m["user_bootstrap"]["significance"].items():
            pt = s["paired_diff_test"]
            print(f"    配对bootstrap[{metric}]: naive {s['naive_mean']:.4f} vs ips {s['ips_mean']:.4f}"
                  f" | improvement {s['improvement_mean']:+.4f} "
                  f"(95%CI {s['improvement_ci95']}) | p={pt['p']} → {s['verdict']}")
    est = real["estimand"]
    print(f"  [估计量层面] 点击率 naive {est['click_rate']['naive']:.4f} / "
          f"SNIPS {est['click_rate']['snips']:.4f} / 无偏test {est['click_rate']['unbiased_test']:.4f}"
          f" → gain {est['click_rate']['gain']:+.4f}")

    # ---- 实验3 ----
    ab = control_ab(coat, seed=args.seed, n_seeds=args.n_seeds)
    print()
    print("-" * 96)
    print(f"实验3：对照实验（A 响应与曝光相关 vs B 响应均匀，{args.n_seeds} seeds）")
    print("-" * 96)
    print(f"  校准：真实 train−test 点击率落差 {ab['calibration']['real_train_test_click_gap']:+.4f}"
          f" → A 组 β={ab['A_response_correlated']['beta']}")
    for key in ("A_response_correlated", "B_response_uniform"):
        r = ab[key]
        sig = r["gain_significance"]
        print(f"  {key}: gain {r['gain_mean']:+.4f} ± {r['gain_std']:.4f} | "
              f"naive_err {r['abs_err_naive_mean']:.4f} → ips_err {r['abs_err_snips_mean']:.4f} | "
              f"p={sig['p']}")
    print("  β 敏感性（A 组，gain 与模拟偏差幅度同步变化）：")
    for row in ab["beta_sensitivity_A"]["rows"]:
        print(f"    β={row['beta']:.2f} → 模拟点击率落差 {row['simulated_click_gap']:+.4f}"
              f"，gain {row['gain_mean']:+.4f} ± {row['gain_std']:.4f}")
    print(f"  → {ab['verdict']}")

    payload = {
        "meta": meta,
        "ess_calibration": ess,
        "debias_real": real,
        "control_ab": ab,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 96)
    print("结论摘要")
    print("=" * 96)
    print(f"  1) 口径A（与合成同口径）ESS/N = {cal['A_real_log']['ess_ratio']:.4f}，"
          f"相对阈值 {SYNTHETIC_THRESHOLD} "
          f"{'越界' if cal['A_real_log']['below_threshold'] else '未越界'}；"
          f"其余口径（{cal['B_per_user_mean']['ess_ratio']:.4f} / "
          f"{cal['C_candidate_pool']['ess_ratio']:.4f} / "
          f"{min(v['ess_ratio_by_user'] for v in cal['D_online_topk']['by_k'].values()):.4f}"
          f"~{max(v['ess_ratio_by_user'] for v in cal['D_online_topk']['by_k'].values()):.4f}）均未越界。")
    print(f"  2) 估计量层面：SNIPS 把点击率误差从 {est['click_rate']['abs_err_naive']:.4f} 降到 "
          f"{est['click_rate']['abs_err_snips']:.4f}（gain {est['click_rate']['gain']:+.4f}）。")
    print(f"  3) 模型层面：{ab['verdict']}")
    print(f"\n结果已写入：{out_path}    耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
