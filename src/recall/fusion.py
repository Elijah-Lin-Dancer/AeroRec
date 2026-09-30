"""多路召回融合：位置分加权合并 + 去重 + 降级兜底。

工业推荐中台常由多路召回（热门 / 协同 / 向量 / 规则）各自产出候选，
再融合成一个候选集交给排序层。这里实现一种简单可解释的融合方式：

  - 每路召回给出一个**已排序**的候选列表；
  - 用位置分 1/(rank+1) 加权累加，越靠前权重越高；
  - 按总分降序取 Top-M，作为最终候选集；
  - 单路失败（异常/为空）自动跳过，保证链路不因某一路挂掉而整体失败。
"""
from __future__ import annotations


class RecallFusion:
    def __init__(self, max_candidates: int = 200, top_per_source: int = 100):
        self.max_candidates = max_candidates
        self.top_per_source = top_per_source
        self.sources = []  # [(name, weight, callable)]

    def add(self, name: str, weight: float, fn):
        """注册一路召回。fn 接受 k 参数并返回 item_id 列表。"""
        self.sources.append((name, weight, fn))
        return self

    def recall(self) -> list:
        lists, weights = [], []
        for name, weight, fn in self.sources:
            try:
                cands = list(fn(self.top_per_source))
                if cands:
                    lists.append(cands)
                    weights.append(weight)
                else:
                    print(f"[fusion] source '{name}' returned empty, skipped")
            except Exception as e:  # 降级：跳过失败源
                print(f"[fusion] source '{name}' failed: {e}, skipped")
        if not lists:
            return []
        return self._merge(lists, weights)

    def _merge(self, ranked_lists, weights) -> list:
        score = {}
        for w, lst in zip(weights, ranked_lists):
            for rank, iid in enumerate(lst):
                score[iid] = score.get(iid, 0.0) + w * (1.0 / (rank + 1))
        ordered = sorted(score.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ordered[: self.max_candidates]]

    # 兼容直接合并静态列表的用法
    def merge(self, *ranked_lists, weights=None) -> list:
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        return self._merge(list(ranked_lists), list(weights))
