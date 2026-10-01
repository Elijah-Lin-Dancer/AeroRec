"""轻量档推荐引擎（纯 numpy 实现，适配 2GB 内存的在线 Demo 实例）。

与完整档 `RecommendationEngine` 的关系
--------------------------------------
**链路、算法、口径完全一致**：

    热门/规则召回 → 特征构建 → DeepFM(CTR) + LR(CVR) 多目标打分
    → 5 步混排（归一化/加权/分层/打散/去重）→ 降级兜底 → 模拟 A/B

差别只在工程实现：
  1. 数据规模 2 万用户 × 2 万物品 × 100 万曝光（完整档 10 万 × 10 万 × 500 万）；
  2. 数据通路全程 numpy，不经 pandas
     （pandas 3.0 + Arrow 后端下，百万行表的构造/对齐本身就要数百 MB）。

混排（`Mixer`）、降级包装（`DegradeRanker`）、评估（`simulate_ab` /
`compare_rankers`）**直接复用完整档的同一份实现**，
确保轻量档不是另一套简化算法。

诚实边界（**必读**）
--------------------
轻量档仅用于**在线交互体验**，不是论文口径。两点必须说清：

1. **规模差异**：2 万 × 2 万 × 100 万曝光（完整档 10 万 × 10 万 × 500 万）。
   规模更小 → 数据更稀疏 → 偏差更大，故轻量档的相对增益明显低于完整档。
2. **评估口径**：本引擎同时提供两套口径——
   - `benchmark_rankers()`：**真实 CTR 口径 + 逐用户 Oracle 上界**，不依赖平滑估计，
     是本引擎中**唯一可引用**的结果（轻量档实测 DeepFM+混排约 +3% vs 热门）。
   - `run_ab()`：模拟在线 A/B。其对照的热门基线用"历史曝光的平滑 CTR"排序，
     而本项目日志是**随机曝光**的——低曝光物品一旦偶然被点击，平滑 CTR 就偏高。
     完整档该偏差实测约 **+0.37**（平滑 0.62 vs 真实 0.25）；轻量档偏差更大，
     会使热门基线被"喂强"，A/B 增益被高估甚至出现负提升。**仅供形式对照。**
"""
from __future__ import annotations

import gc

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src.config_lite import SEED, N_USERS, N_ITEMS, TOP_K, CATEGORIES
from src.data.synthetic_generator_lite import build_dataset_np
from src.feature.lite_store import LiteFeatureStore, bincount_stats, SPARSE_CARD, N_DENSE
from src.rank.trainer import DeepFMRanker
from src.rank.mixer import Mixer
from src.rank.degrade import DegradeRanker
from src.eval.ab import simulate_ab

_CAT = list(CATEGORIES)
ROUTE_CAT = 0


class LitePopularRecall:
    """全局热门兜底召回（`np.bincount` 实现，数学等价于 `PopularRecall`）。"""

    def __init__(self, n_items: int, top_k: int = 200, alpha: float = 5.0):
        self.n_items, self.top_k, self.alpha = int(n_items), int(top_k), float(alpha)

    def fit(self, item_id: np.ndarray, clicked: np.ndarray):
        self.count_, self.sum_, self.ctr_ = bincount_stats(
            item_id, clicked, self.n_items, self.alpha)
        self.order_ = np.argsort(-self.ctr_)
        return self

    def recall(self, k: int = 10) -> list:
        return self.order_[:k].tolist()


class LiteRuleRecall:
    """规则召回（numpy 实现，与 `src/recall/rule.py` 四条规则同语义）。

    规则1 里程兑换：里程票需 里程价 ≤ 余额 × (1+阈值)
    规则2 库存：现金票 0 库存剔除；里程票 0 库存保留（独立可兑库存）
    规则3 常驻地热门：常驻城市出发的航线加分
    规则4 多段行程：按出发日期由近到远，优先推荐返程（目的地→出发地）
    """

    def __init__(self, items: dict, users: dict, miles_threshold: float = 0.10):
        self.items, self.users = items, users
        self.miles_threshold = miles_threshold
        self.route_ids = np.where(items["category_idx"] == ROUTE_CAT)[0]
        pop = items["popularity"][self.route_ids]
        self.pop_norm = pop / (pop.max() if pop.size and pop.max() > 0 else 1.0)
        rd = items["popularity"]
        self._i_is_miles = items["is_miles_ticket"][self.route_ids]
        self._i_miles_req = items["miles_required"][self.route_ids]
        self._i_stock = items["stock"][self.route_ids]
        self._i_origin = items["origin_idx"][self.route_ids]
        self._i_dest = items["dest_idx"][self.route_ids]

    def recall(self, user_id: int, trips_for_user: list, k: int = 200) -> list:
        uid = int(user_id)
        score = self.pop_norm.astype(np.float64).copy()
