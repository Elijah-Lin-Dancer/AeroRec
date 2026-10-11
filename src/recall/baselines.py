"""召回基线：热门兜底 + ItemCF（物品协同过滤）。

召回层负责从全量物品中产出"候选集"，交给下游排序层精排。
这里实现两类可解释、易复现的召回，作为中台基线与兜底来源。
"""
from collections import defaultdict

import numpy as np
import pandas as pd


class PopularRecall:
    """全局热门兜底：按历史点击率（含曝光平滑）取 Top 物品。"""

    def __init__(self, top_k: int = 200, alpha: float = 5.0):
        self.top_k = top_k
        self.alpha = alpha  # 贝叶斯平滑先验强度

    def fit(self, logs: pd.DataFrame):
        g = logs.groupby("item_id")["clicked"].agg(["sum", "count"])
        # 贝叶斯平滑后的 CTR
        g["ctr"] = (g["sum"] + self.alpha) / (g["count"] + self.alpha)
        self.top_items_ = g.sort_values("ctr", ascending=False).head(self.top_k)
        return self

    def recall(self, k: int = 10) -> list:
        return self.top_items_.head(k).index.tolist()


class ItemCFRecall:
    """物品协同过滤：基于共点击相似度，给用户推荐与其历史点击物品相似者。"""

    def __init__(self, top_k: int = 200, sim_threshold: float = 0.0):
        self.top_k = top_k
        self.sim_threshold = sim_threshold

    def fit(self, logs: pd.DataFrame):
        # 仅用"点击"行为构建物品共现
        clicked = logs[logs["clicked"] == 1]
        cooc: dict[tuple[int, int], int] = defaultdict(int)
        user_items = defaultdict(set)
        for uid, iid in zip(clicked["user_id"], clicked["item_id"]):
            user_items[uid].add(iid)
        for item_set in user_items.values():
            item_list = list(item_set)
            for a in range(len(item_list)):
                for b in range(len(item_list)):
                    if a != b:
                        cooc[(item_list[a], item_list[b])] += 1

        # 物品出现次数用于余弦式归一化
        item_cnt: dict[int, int] = defaultdict(int)
        for item_set in user_items.values():
            for i in item_set:
                item_cnt[i] += 1

        sim = {}
        for (a, b), c in cooc.items():
            denom = np.sqrt(item_cnt[a] * item_cnt[b])
            if denom > 0:
                s = c / denom
                if s > self.sim_threshold:
                    sim[(a, b)] = s
        self.sim_ = sim
        self.item_cnt_ = item_cnt
        return self

    def recall(self, user_clicked_items: list, k: int = 10) -> list:
        scores: dict[int, float] = defaultdict(float)
        for iid in set(user_clicked_items):
            for (a, b), s in self.sim_.items():
                if a == iid and b not in user_clicked_items:
                    scores[b] += s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked[:k]]
