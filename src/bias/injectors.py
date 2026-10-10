"""偏差注入器：把「曝光偏差」施加到一份诚实日志上，得到「有偏日志」。

设计原则
--------
偏差注入**只改「谁被曝光」，不改「点击行为」**。
这保证了一个干净的对照：

    真值（ground-truth）  = compute_true_ctr(用户, 物品)     —— 不随策略变化
    有偏观测               = 在策略曝光日志上统计到的点击率     —— 随策略变化

两者的差就是**偏差本身**。真实日志里这个差是混在一起的（你既不知道真值，
也不知道曝光策略），这里能把它**单独拆出来**。

与既有代码的关系
----------------
项目里已有一个实例：`Mixer(w_base>0)` 在混排阶段二次注入热度偏差
（实测 w_base=0.4 时增益被拉低，见 README）。本模块把这类现象**一般化**为
可复用的注入算子，而不是散落在各处的特例。
"""
from __future__ import annotations

import numpy as np

from src.bias.policy import ExposurePolicy, sample_exposure


def inject_exposure_popularity(item_pool, popularity, n_slots: int,
                               tau: float, rng: np.random.Generator,
                               replace: bool = False):
    """按热度策略 h^τ 抽曝光位，返回 (曝光物品下标, 闭式 π)。

    参数
    ----
    item_pool   : 候选物品下标（决定「在谁里面抽」）。
    popularity  : 与 item_pool 等长的热度向量（或全库热度，配合 pool 取）。
    n_slots     : 每次请求曝光几个坑位。
    tau         : 偏差强度；**τ=0 → 均匀曝光，即无偏**。

    返回
    ----
    (exposed_idx, pi)
        exposed_idx : 被曝光的候选在 item_pool 中的**下标**。
        pi          : 与 item_pool 等长的闭式 propensity。
    """
    pool = np.asarray(item_pool, dtype=np.int64)
    h = np.asarray(popularity, dtype=np.float64)
    if h.shape[0] != pool.shape[0]:
        h = h[pool]                          # 传的是全库热度 → 按池取
    pol = ExposurePolicy("popularity", strength=tau)
    pi = pol.propensity(len(pool), popularity=h)
    idx = sample_exposure(pi, n_slots, rng, replace=replace)
    return idx, pi


def inject_exposure_position(scores, n_slots: int, eta: float,
                             rng: np.random.Generator,
                             replace: bool = False):
    """按位次权重 1/rank^η 抽曝光位，模拟**位置偏差**。

    scores 越大越靠前；返回 (曝光下标, 闭式 π)。
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    pol = ExposurePolicy("position", strength=eta)
    pi = pol.propensity(n, scores=s)
    idx = sample_exposure(pi, n_slots, rng, replace=replace)
    return idx, pi


def inject_uniform(n_slots: int, rng: np.random.Generator,
                   replace: bool = False):
    """均匀曝光（无偏对照组）。返回 (曝光下标, 闭式 π)。"""
    pol = ExposurePolicy("uniform")
    pi = pol.propensity(n_slots)             # 注意：均匀策略下 π 只看候选数
    idx = sample_exposure(pi, n_slots, rng, replace=replace)
    return idx, pi


def build_biased_log(true_ctr, propensity, n_requests: int, slots: int,
                     rng: np.random.Generator, replace: bool = True):
    """从「真实 CTR + 曝光策略」生成一份有偏日志，并保留每条记录的 propensity。

    这是去偏实验的**唯一数据入口**：产出的日志同时携带
    `clicked`（有偏观测）与 `propensity`（用于 IPS 加权），
    以及 `true_ctr`（用于算真值，只在评估时用，估计器不许碰）。

    参数
    ----
    true_ctr    : (n_requests, n_candidates)，每个请求下每个候选的真实点击概率。
                  行内为候选集，调用方需保证同一列在不同请求间是**同一个物品**
                  （否则 IPS 的权重没有跨请求可比性）。
    propensity  : (n_requests, n_candidates) 或 (n_candidates,)，闭式曝光概率。
    n_requests  : 请求数 R。
    slots       : 每请求曝光坑位数 K。
    replace     : 曝光是否有放回（默认 True，与工业 CTR 日志更接近，且 π 口径干净）。

    返回
    ----
    dict：
      request_idx (R*K,)  请求号
      candidate_idx (R*K,) 候选列号（= 物品号）
      clicked (R*K,)      是否点击（伯努利 true_ctr 抽样）
      propensity (R*K,)   该条记录的曝光概率（**估计器唯一可用**）
      true_ctr (R*K,)     真实点击率（**仅评估用**）
    """
    T = np.asarray(true_ctr, dtype=np.float64)
    if T.ndim == 1:
        T = T[None, :]
    R, C = T.shape
    R = int(n_requests) if n_requests else R
    if R != T.shape[0]:
        # 允许调用方只给一行、按需重复（同分布）
        T = np.repeat(T[:1], R, axis=0)

    P = np.asarray(propensity, dtype=np.float64)
    if P.ndim == 1:
        P = np.repeat(P[None, :], R, axis=0)

    K = int(min(slots, C) if not replace else slots)
    req = np.empty(R * K, dtype=np.int64)
    cand = np.empty(R * K, dtype=np.int64)
    prop = np.empty(R * K, dtype=np.float64)

    for r in range(R):
        idx = sample_exposure(P[r], K, rng, replace=replace)
        req[r * K:(r + 1) * K] = r
        cand[r * K:(r + 1) * K] = idx
        prop[r * K:(r + 1) * K] = P[r][idx]

    tc = T[req, cand]
    clicked = (rng.random(len(tc)) < tc).astype(np.int64)

    return {
        "request_idx": req,
        "candidate_idx": cand,
        "clicked": clicked,
        "propensity": prop,
        "true_ctr": tc,
        "n_requests": R,
        "slots": K,
    }
