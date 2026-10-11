"""评估层：离线指标 + 模型对比。"""
from src.eval.metrics import auc, hit_rate_at_k, ndcg_at_k, recall_at_k

__all__ = ["auc", "hit_rate_at_k", "ndcg_at_k", "recall_at_k"]
