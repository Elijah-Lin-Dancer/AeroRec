"""离线评估指标：AUC、Recall@k、HitRate@k、NDCG@k。"""
import numpy as np


def auc(y_true, scores) -> float:
    """ROC-AUC（离线评估用）。

    ⚠️ 此处**惰性导入 sklearn**：本模块经 `src/eval/__init__.py` 被在线低内存路径
    间接导入（`engine_micro` → `src.eval.ab`），而在线路径不调用本函数。
    若在模块顶层 `import sklearn`，会让在线侧白白付出约 150MB 导入开销。
    sklearn 不可用时退化为**纯 numpy 的等秩公式**实现（数值等价）：
        AUC = (Σ rank_pos - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except ImportError:
        return _auc_numpy(y_true, scores)


def _auc_numpy(y_true, scores) -> float:
    """纯 numpy 的 AUC（Mann-Whitney U 等秩公式），与 sklearn 数值一致。"""
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # 处理并列分数：取平均秩
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_k(ranked, true_items, k: int) -> float:
    return len(set(ranked[:k]) & set(true_items)) / max(1, len(true_items))


def hit_rate_at_k(ranked, true_items, k: int) -> float:
    return float(len(set(ranked[:k]) & set(true_items)) > 0)


def ndcg_at_k(ranked, true_items, k: int) -> float:
    true = set(true_items)
    dcg = 0.0
    for i, it in enumerate(ranked[:k]):
        if it in true:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true))))
    return dcg / idcg if idcg > 0 else 0.0
