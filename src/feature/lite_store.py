"""轻量档 numpy 特征库（不依赖 pandas）。

为什么需要
----------
在线 Demo 上，即使只有 100 万曝光，
"构造 pandas DataFrame → groupby / set_index / iloc 对齐" 这条路径会吃掉 400~700MB
（pandas 3.0 + pyarrow 后端的固定转换开销）。
本项目真正需要的只有：
  1. 每个物品的曝光 / 点击次数（→ 热门召回、降级兜底）；
  2. 用户维度特征（5 个类目偏好 + 里程余额 + 价格敏感 + 商务标签）；
  3. 物品维度特征（类目 + 质量 + 价格 + 里程票 + 时段）。

用 numpy 数组（`np.bincount` + 索引 gather）即可完成，
内存从数百 MB 降到十几 MB，且**数学上与 pandas 版完全等价**：
  - 归一化统计量同口径（均取全量 max，clip 到 3.0）
  - 特征顺序严格对齐 `src/feature/store.py:DENSE_COLS`
  - 平滑 CTR 用与 `PopularRecall` 相同的 α=5 贝叶斯公式

**本模块只服务轻量档在线 Demo**；完整档（`src/feature/store.py`、
`src/recall/baselines.py`）保持原样，用于离线复现与代码审阅。
"""
from __future__ import annotations

import numpy as np

from src.config_lite import CATEGORIES, TIME_SLOTS, HOUR_SLOTS

_CAT = list(CATEGORIES)

# 连续特征顺序必须与 `src/feature/store.py:DENSE_COLS` 完全一致，
# 以保证两档模型输入同构（同维度、同含义）。
DENSE_COLS = [
    "aff_航线", "aff_酒店", "aff_商城", "aff_营销", "aff_直播",
    "quality", "price_norm", "price_sensitivity", "is_business",
    "miles_balance_norm", "miles_required_norm", "is_miles_ticket",
    "is_weekend",
]
N_DENSE = len(DENSE_COLS)
N_SPARSE = 3
SPARSE_CARD = [len(CATEGORIES), len(TIME_SLOTS) + 1, len(HOUR_SLOTS)]


class LiteFeatureStore:
    """把 numpy 字典形式的 items / users 转成特征矩阵，训练与推理共用同一口径。"""

    def __init__(self, items: dict, users: dict):
        # ---- 用户特征 ----
        self.u_aff = np.stack([np.asarray(users[f"aff_{c}"], dtype=np.float32)
                               for c in _CAT], axis=1)                       # (U, 5)
        self.u_miles_bal = np.asarray(users["miles_balance"], dtype=np.float32)
        self.u_price_sens = np.asarray(users["price_sensitivity"], dtype=np.float32)
        self.u_is_biz = np.asarray(users["is_business"], dtype=np.float32)
        self.u_home = np.asarray(users["home_idx"], dtype=np.int64)

        # ---- 物品特征 ----
        self.i_cat = np.asarray(items["category_idx"], dtype=np.int64)
        self.i_quality = np.asarray(items["quality"], dtype=np.float32)
        self.i_price = np.asarray(items["cash_price"], dtype=np.float32)
        self.i_is_miles = np.asarray(items["is_miles_ticket"], dtype=np.float32)
        self.i_miles_req = np.asarray(items["miles_required"], dtype=np.float32)
        self.i_time_slot = np.asarray(items["time_slot"], dtype=np.int64)
        self.i_pop = np.asarray(items["popularity"], dtype=np.float32)
        self.i_stock = np.asarray(items["stock"], dtype=np.int64)
        self.i_origin = np.asarray(items["origin_idx"], dtype=np.int64)
        self.i_dest = np.asarray(items["dest_idx"], dtype=np.int64)

        # 归一化统计量（与 pandas 版 norm_stats 同口径）
        self.price_max = float(self.i_price.max()) or 1.0
        self.miles_bal_max = float(self.u_miles_bal.max()) or 1.0
        self.miles_req_max = float(self.i_miles_req.max()) or 1.0

        self.n_users = len(np.asarray(users["user_id"]))
        self.n_items = len(np.asarray(items["item_id"]))

    # ---------- 特征拼装 ----------

    def build_matrix(self, user_idx: np.ndarray, item_idx: np.ndarray,
                     is_weekend: np.ndarray) -> np.ndarray:
        """构造 (n, N_DENSE) 稠密特征矩阵（训练 / 推理同一口径）。"""
        n = len(user_idx)
        out = np.empty((n, N_DENSE), dtype=np.float32)
        out[:, 0:5] = self.u_aff[user_idx]                               # 用户类目偏好 ×5
        out[:, 5] = self.i_quality[item_idx]                             # 物品质量
        out[:, 6] = np.clip(self.i_price[item_idx] / self.price_max, 0.0, 3.0)
        out[:, 7] = self.u_price_sens[user_idx]
        out[:, 8] = self.u_is_biz[user_idx]
        out[:, 9] = np.clip(self.u_miles_bal[user_idx] / self.miles_bal_max, 0.0, 3.0)
        out[:, 10] = np.clip(self.i_miles_req[item_idx] / self.miles_req_max, 0.0, 3.0)
        out[:, 11] = self.i_is_miles[item_idx]
        out[:, 12] = is_weekend
        return out

    def build_sparse(self, item_idx: np.ndarray, hour_slot: np.ndarray) -> np.ndarray:
        """构造 (n, 3) 稀疏特征（类目 / 时段 / 小时段），供 embedding 使用。"""
        out = np.empty((len(item_idx), N_SPARSE), dtype=np.int64)
        out[:, 0] = self.i_cat[item_idx]
        out[:, 1] = self.i_time_slot[item_idx]
        out[:, 2] = hour_slot
        return out

    def user_candidates(self, user_id: int, cand_ids: np.ndarray,
                        hour_slot: int = 0, is_weekend: float = 0.0) -> tuple:
        """单个用户 × 候选集的特征（推理用），与 build_matrix 同口径。"""
        n = len(cand_ids)
        uid = np.full(n, int(user_id), dtype=np.int64)
        wk = np.full(n, float(is_weekend), dtype=np.float32)
        dense = self.build_matrix(uid, cand_ids, wk)
        sparse = self.build_sparse(cand_ids, np.full(n, int(hour_slot), dtype=np.int64))
        return dense, sparse


def bincount_stats(item_id: np.ndarray, clicked: np.ndarray, n_items: int,
                   alpha: float = 5.0):
    """物品级 曝光数 / 点击数 / 贝叶斯平滑 CTR，全部走 `np.bincount`（零分组中间态）。

    返回 (exposure, click, smooth_ctr)，均为长度 n_items 的 numpy 数组。
    与 `src/recall/degrade.compute_recent_ctr`、`PopularRecall` 的 α=5 公式等价。
    """
    item_id = np.asarray(item_id)
    cnt = np.bincount(item_id, minlength=n_items).astype(np.float64)
    sm = np.bincount(item_id, weights=np.asarray(clicked, dtype=np.float64),
                     minlength=n_items).astype(np.float64)
    ctr = (sm + alpha) / (cnt + alpha)
    return cnt, sm, ctr
