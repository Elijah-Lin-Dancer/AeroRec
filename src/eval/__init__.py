"""评估层：离线指标 + 模型对比。"""
from src.eval.metrics import auc, recall_at_k, hit_rate_at_k, ndcg_at_k

__all__ = ["auc", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
