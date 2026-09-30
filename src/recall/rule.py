"""规则召回引擎：把出行域业务规则固化为可配置召回。

对应实习中梳理的业务规则（受启发、独立实现，不含任何真实业务数据）：
  1. 里程兑换：推荐航线里程价 ≤ 用户剩余里程 + 10% 阈值；
  2. 里程票无需剔除 0 库存航线（里程票有独立可兑库存）；
  3. 无个性化结果时兜底推荐「常驻地热门航线」；
  4. 多段未出行行程按日期从近到远优先推荐返程。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROUTE_CAT = 0  # 航线类目索引


class RuleRecall:
    def __init__(self, miles_threshold: float = 0.10, top_k: int = 200):
        self.miles_threshold = miles_threshold
        self.top_k = top_k

    def fit(self, logs=None, items=None, users=None, trips=None):
        """缓存规则召回所需数据。"""
        self.items_ = items
        self.users_ = users
        self.trips_ = trips

        # 航线子集 + 热度归一化
        self.routes_ = items[items["category_idx"] == ROUTE_CAT].copy()
        pop_max = self.routes_["popularity"].max()
        if pop_max is None or pop_max <= 0:
            pop_max = 1.0
        self.routes_["pop_norm"] = self.routes_["popularity"] / pop_max

        # 未来行程按用户 + 出发日期排序（日期近的在前）
        if trips is not None and len(trips):
            self.trips_sorted_ = trips.sort_values(["user_id", "depart_offset"])
        else:
            self.trips_sorted_ = pd.DataFrame(
                columns=["user_id", "trip_id", "origin_idx", "dest_idx", "depart_offset"])
        return self

    def recall(self, user_id: int, k: int = 10) -> list:
        if self.routes_ is None or self.routes_.empty:
            return []

        u = self.users_[self.users_["user_id"] == user_id]
        if u.empty:
            return []
        user = u.iloc[0]

        routes = self.routes_
        score = routes["pop_norm"].to_numpy(dtype=float).copy()
        is_miles = (routes["is_miles_ticket"].to_numpy() == 1)
        miles_req = routes["miles_required"].to_numpy()
        stock = routes["stock"].to_numpy()
        origin = routes["origin_idx"].to_numpy(dtype=int)
        dest = routes["dest_idx"].to_numpy(dtype=int)

        # 规则1：里程兑换——里程票须满足 里程价 ≤ 余额*(1+阈值)，否则剔除
        miles_bal = float(user["miles_balance"])
        budget = miles_bal * (1.0 + self.miles_threshold)
        over_budget = is_miles & (miles_req > budget)
        score[over_budget] = -np.inf

        # 规则2：现金票 0 库存剔除；里程票 0 库存保留（有独立可兑库存）
        zero_stock = stock <= 0
        score[zero_stock & ~is_miles] = -np.inf

        # 规则3：常驻地出发的航线加分（无个性化时的兜底热度来源）
        home = int(user["home_idx"])
        score[origin == home] += 1.0

        # 规则4：多段未出行行程——最近行程的返程优先（返程 = 目的地→出发地）
        if self.trips_sorted_ is not None and len(self.trips_sorted_):
            my_trips = self.trips_sorted_[self.trips_sorted_["user_id"] == user_id]
            for rank, (_, t) in enumerate(my_trips.iterrows()):
                ret_origin = int(t["dest_idx"])
                ret_dest = int(t["origin_idx"])
                match = (origin == ret_origin) & (dest == ret_dest)
                score[match] += 2.0 / (1.0 + rank)  # 越近的行程权重越高

        order = np.argsort(-score)  # 降序，-inf 沉底
        cand = []
        for i in order:
            if np.isfinite(score[i]):
                cand.append(int(routes.iloc[i]["item_id"]))
                if len(cand) >= k:
                    break
        return cand


if __name__ == "__main__":
    from src.data.synthetic_generator import load_dataset
    items, users, trips, logs = load_dataset()
    rr = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)
    demo = int(users["user_id"].iloc[0])
    print("recall:", rr.recall(demo, k=10))
