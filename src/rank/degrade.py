"""降级兜底：CTR 预估失败 / 超时时，退化为历史平均 CTR。

对应实习中的降级策略：CTR 接口超时 → 用近 7 天历史平均 CTR 兜底，保证核心链路不因
算法服务故障而中断。合成数据无时间戳，故"近 7 天"用全量曝光日志的平均 CTR 近似，
工程语义（缓存一份聚合 CTR 做兜底）保持一致。
"""
import numpy as np
import pandas as pd


def compute_recent_ctr(logs: pd.DataFrame, alpha: float = 5.0) -> dict:
    """计算物品级历史平均 CTR（贝叶斯平滑），返回 {item_id: ctr}。"""
    g = logs.groupby("item_id")["clicked"].agg(["sum", "count"])
    ctr = (g["sum"] + alpha) / (g["count"] + alpha)
    return ctr.to_dict()


class DegradeRanker:
    """包装 CTR 预估器，失败时降级到历史平均 CTR。"""

    def __init__(self, predictor, fallback_ctr_map: dict, fallback_default: float = 0.05):
        self.predictor = predictor            # callable(dense, sparse) -> ctr array
        self.fallback_ctr_map = fallback_ctr_map
        self.fallback_default = fallback_default

    def predict_ctr(self, dense, sparse, item_ids) -> np.ndarray:
        try:
            return np.asarray(self.predictor(dense, sparse), dtype=float)
        except Exception as e:
            print(f"[degrade] CTR 预估失败：{e}，降级为历史平均 CTR")
            return np.array([self.fallback_ctr_map.get(i, self.fallback_default)
                             for i in item_ids], dtype=float)
