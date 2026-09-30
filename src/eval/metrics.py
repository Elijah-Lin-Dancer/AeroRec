"""离线评估指标：AUC、Recall@k、HitRate@k、NDCG@k。"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, scores) -> float:
    return roc_auc_score(y_true, scores)


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
