"""5 步混排：复刻工业推荐排序流程。

步骤（对应实习中的混排逻辑）：
  1. 分数归一化：把基础分 / CTR 都归一到 [0,1]；
  2. CTR 加权：combined = w_base * 基础分 + w_ctr * CTR；
  3. 分层排序：按 combined 降序；
  4. 同品类打散：连续同类目不超过 max_same_category；
  5. 分页去重：去重并按 top_k 截断。
"""
import numpy as np


def _minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class Mixer:
    def __init__(self, w_base: float = 0.4, w_ctr: float = 0.6,
                 max_same_category: int = 2, top_k: int = 10):
        self.w_base = w_base
        self.w_ctr = w_ctr
        self.max_same_category = max_same_category
        self.top_k = top_k

    def mix(self, item_ids, base_scores, ctr_scores, categories) -> list:
        """返回重排后的 item_id 列表（已加权、打散、去重、截断）。"""
        item_ids = list(item_ids)
        categories = list(categories)

        # 1. 归一化
        base_norm = _minmax(base_scores)
        ctr_norm = _minmax(ctr_scores)

        # 2. CTR 加权
        combined = self.w_base * base_norm + self.w_ctr * ctr_norm

        # 3. 分层排序（降序）
        records = sorted(zip(item_ids, categories, combined), key=lambda t: -t[2])

        # 4. 同品类打散
        ordered = self._scatter([(iid, c) for iid, c, _ in records])

        # 5. 分页去重 + 截断
        seen, result = set(), []
        for iid in ordered:
            if iid not in seen:
                seen.add(iid)
                result.append(iid)
            if len(result) >= self.top_k:
                break
        return result

    def _scatter(self, records) -> list:
        """贪心打散：连续同类目不超过 max_same_category，超出的延后到队尾。"""
        result, deferred = [], []
        for rec in records:
            iid, c = rec
            recent = [r[1] for r in result[-self.max_same_category:]]
            if len(recent) >= self.max_same_category and all(rc == c for rc in recent):
                deferred.append(rec)
            else:
                result.append(rec)
        result.extend(deferred)
        return [r[0] for r in result]
