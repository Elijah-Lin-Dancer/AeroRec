"""偏差度量：把「某个估计量相对 ground-truth 的偏离」量化清楚。

三种口径，对应三种不同的偏差来源
--------------------------------
1. **曝光偏差（exposure bias）**
   有偏策略下，各物品被曝光的频率 π̂ 偏离理想均匀曝光。度量：
   - `exposure_kl`  : KL(π̂ ‖ uniform)，即曝光分布偏离均匀的程度（nats）。
   - `exposure_gini` : 曝光集中度（0=完全均匀，→1=极度集中）。
   这是**链路最上游**的偏差，也是后面所有偏差的源头。

2. **统计偏差（statistical bias）**
   用有偏日志估计「某物品/某方法的真实 CTR」，估计值与真值之差。度量：
   - `ctr_bias`      : 平滑 CTR − 真实 CTR（**这正是 README 里已记录的现象**）。
   - `ctr_mae`       : 绝对偏差的均值。
   这一类就是「热门基线稀疏性偏差」。

3. **排序偏差（ranking bias）**
   偏差是否改变了「谁排前面」这一结论。度量：
   - `topk_overlap`  : 有偏榜 Top-K 与真实榜 Top-K 的重合比例。
   - `ndcg_at_k`     : 以真实 CTR 为相关性标签的 NDCG@K。
   这是**对业务最有杀伤力**的一类——分数错了不要紧，排错了才致命。

为什么全部相对 ground-truth
---------------------------
真实场景下这些量都无法计算（没有真值）。本模块存在的意义正是：
把「偏差」从「真实系统里只能定性感知」变成「合成环境里可以定量测量」。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ----------------------------------------------------------------------
# ① 曝光偏差
# ----------------------------------------------------------------------
def exposure_distribution(candidate_idx, n_candidates: int) -> np.ndarray:
    """把曝光记录聚合成曝光频率分布（和为 1）。"""
    idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(idx) == 0 or n_candidates <= 0:
        return np.zeros(max(n_candidates, 0))
    counts = np.bincount(idx, minlength=n_candidates)[:n_candidates]
    return counts / max(counts.sum(), 1)


def exposure_kl(exposure_freq, uniform: bool = True) -> float:
    """KL(P ‖ U)：曝光分布相对均匀分布的偏离，单位 nats。

    uniform=False 时以经验均匀（1/N）为准，语义相同；保留参数是为了显式化口径。
    """
    p = np.asarray(exposure_freq, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    q = np.full(n, 1.0 / n)
    p = p / max(p.sum(), _EPS)
    # 只在 p>0 处求和（0·log0 = 0）
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + _EPS) / (q[mask] + _EPS))))


def exposure_gini(exposure_freq) -> float:
    """曝光集中度的 Gini 系数：0=完全均匀，→1=极度集中。"""
    x = np.sort(np.asarray(exposure_freq, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * x.sum()) - (n + 1.0) / n)


# ----------------------------------------------------------------------
# ② 统计偏差：估计 CTR vs 真实 CTR
# ----------------------------------------------------------------------
def ctr_bias(est_ctr, true_ctr) -> float:
    """平均有符号偏差：估计值 − 真值。正值=系统性高估。"""
    e = np.asarray(est_ctr, dtype=np.float64)
    t = np.asarray(true_ctr, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    return float(np.mean(e - t))


def ctr_mae(est_ctr, true_ctr) -> float:
    """平均绝对偏差，衡量偏离幅度（不区分高估/低估）。"""
    e = np.asarray(est_ctr, dtype=np.float64)
    t = np.asarray(true_ctr, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    return float(np.mean(np.abs(e - t)))


def smoothed_ctr(clicks, impressions, alpha: float = 5.0) -> np.ndarray:
    """贝叶斯平滑 CTR：(click + α) / (impr + α)。

    这正是 `PopularRecall` 用的口径。α 越小越容易被「偶然点击一次」带偏——
    在有偏曝光日志上这一点会被放大（曝光本身就偏向热门）。
    """
    c = np.asarray(clicks, dtype=np.float64)
    i = np.asarray(impressions, dtype=np.float64)
    return (c + alpha) / (i + alpha)


# ----------------------------------------------------------------------
# ③ 排序偏差
# ----------------------------------------------------------------------
def topk_overlap(rank_a, rank_b, k: int = 10) -> float:
    """两个榜单 Top-K 的重合比例（Jaccard 式，按集合交/并）。"""
    a = list(rank_a)[:k]
    b = list(rank_b)[:k]
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 0.0


def ndcg_at_k(ranked_items, relevance: dict, k: int = 10) -> float:
    """以 relevance（这里是**真实 CTR**）为标签的 NDCG@K。

    把真实 CTR 当作相关性分数，衡量「排序把高真实价值的物品排到前面了吗」。
    """
    items = list(ranked_items)[:k]
    if not items:
        return 0.0
    gains = np.array([relevance.get(it, 0.0) for it in items], dtype=np.float64)
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))

    # 理想排序：在**同一候选集**内按 relevance 降序（严格上界）
    ideal = np.sort(np.array(list(relevance.values()), dtype=np.float64))[::-1][:k]
    if len(ideal) < len(gains):
        ideal = np.concatenate([ideal, np.zeros(len(gains) - len(ideal))])
    idcg = float(np.sum(ideal * disc))
    dcg = float(np.sum(gains * disc))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# 汇总：一次算全
# ----------------------------------------------------------------------
def bias_summary(exposure_freq=None, est_ctr=None, true_ctr=None) -> dict:
    """把可算的偏差指标打包，便于实验脚本统一记录。"""
    out: dict = {}
    if exposure_freq is not None:
        out["exposure_kl_nats"] = round(exposure_kl(exposure_freq), 4)
        out["exposure_gini"] = round(exposure_gini(exposure_freq), 4)
    if est_ctr is not None and true_ctr is not None:
        out["ctr_bias"] = round(ctr_bias(est_ctr, true_ctr), 4)
        out["ctr_mae"] = round(ctr_mae(est_ctr, true_ctr), 4)
    return out


def reproduce_known_phenomenon(pop_recall, users, items, top_n: int = 200,
                               sample_user: int = 0) -> dict:
    """复现「热门基线平滑 CTR 偏差」这一已知现象。

    复用 `src.eval.ab.baseline_bias_table`，把既有现象纳入偏差框架——
    这是「新旧两条线接上」的证据：偏差诊断不是新开一摊，而是把已发现的问题形式化。

    返回：smooth_ctr / true_ctr / bias / top_n，或在不可用时带 error 字段。
    """
    from src.eval.ab import baseline_bias_table
    return baseline_bias_table(pop_recall, users, items, top_n=top_n,
                               sample_user=sample_user)
