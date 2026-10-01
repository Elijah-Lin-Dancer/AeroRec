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

        # 规则1：里程预算不足的里程票剔除
        budget = float(self.users["miles_balance"][uid]) * (1.0 + self.miles_threshold)
        score[(self._i_is_miles == 1) & (self._i_miles_req > budget)] = -np.inf
        # 规则2：现金票零库存剔除（里程票保留）
        score[(self._i_stock <= 0) & (self._i_is_miles != 1)] = -np.inf
        # 规则3：常驻地出发加分
        score[self._i_origin == int(self.users["home_idx"][uid])] += 1.0
        # 规则4：未来行程返程优先
        for rank, (o, d, _off) in enumerate(trips_for_user):
            score[(self._i_origin == d) & (self._i_dest == o)] += 2.0 / (1.0 + rank)

        finite = np.isfinite(score)
        if not finite.any():
            return []
        cand = np.where(finite)[0]
        cand = cand[np.argsort(-score[cand])][:k]
        return self.route_ids[cand].tolist()


class _ArrayLookup:
    """把 numpy 数组包成只实现 `.get` 的对象，避免构造大型 Python dict。"""

    def __init__(self, arr: np.ndarray, default: float = 0.05):
        self.arr, self.default = arr, default

    def get(self, k, default=None):
        try:
            k = int(k)
            if 0 <= k < self.arr.size:
                return float(self.arr[k])
        except Exception:
            pass
        return self.default if default is None else default


class LiteRecommendationEngine:
    """轻量档引擎。对外接口与 `RecommendationEngine` 完全一致，可直接替换。"""

    def __init__(self, train_sample: int = 60_000, deepfm_epochs: int = 2,
                 embed_dim: int = 16, dnn_hidden: tuple = (64, 32),
                 cvr_weight: float = 0.3, top_k: int = TOP_K,
                 n_users: int = N_USERS, n_items: int = N_ITEMS):
        self.top_k = top_k
        self.cvr_weight = cvr_weight
        self.n_users, self.n_items = n_users, n_items

        # ---------- 数据：纯 numpy 即时生成（同公式、同种子） ----------
        self.items, self.users, self.trips, self.logs = build_dataset_np(SEED)

        # ---------- 特征库 ----------
        self.fs = LiteFeatureStore(self.items, self.users)

        # ---------- 召回层 ----------
        self.pop = LitePopularRecall(self.n_items, top_k=200).fit(
            self.logs["item_id"], self.logs["clicked"])

        trips_by_user = [[] for _ in range(self.n_users)]
        for u, o, d, off in zip(self.trips["user_id"], self.trips["origin_idx"],
                                self.trips["dest_idx"], self.trips["depart_offset"]):
            trips_by_user[int(u)].append((int(o), int(d), int(off)))
        self.trips_by_user = [sorted(t, key=lambda x: x[2]) for t in trips_by_user]
        self.rule = LiteRuleRecall(self.items, self.users)

        # ---------- 排序层：DeepFM(CTR) ----------
        n = len(self.logs["clicked"])
        idx = np.random.default_rng(SEED).choice(
            n, size=min(train_sample, n), replace=False)
        dense = self.fs.build_matrix(self.logs["user_id"][idx],
                                     self.logs["item_id"][idx],
                                     self.logs["is_weekend"][idx])
        sparse = self.fs.build_sparse(self.logs["item_id"][idx],
                                      self.logs["hour_slot"][idx])
        y = self.logs["clicked"][idx].astype(np.int64)
        self.dfm = DeepFMRanker(N_DENSE, SPARSE_CARD, embed_dim=embed_dim,
                                dnn_hidden=dnn_hidden, epochs=deepfm_epochs,
                                batch_size=2048).fit(dense, sparse, y)
        del dense, sparse, y, idx
        gc.collect()

        # ---------- 排序层：LR(CVR) 多目标 ----------
        clk = np.where(self.logs["clicked"] > 0)[0][:min(60_000, n)]
        d2 = self.fs.build_matrix(self.logs["user_id"][clk],
                                  self.logs["item_id"][clk],
                                  self.logs["is_weekend"][clk])
        s2 = self.fs.build_sparse(self.logs["item_id"][clk], self.logs["hour_slot"][clk])
        self.cvr_enc = OneHotEncoder(categories=[range(c) for c in SPARSE_CARD],
                                     sparse_output=False)
        self.cvr_model = LogisticRegression(max_iter=1000).fit(
            np.hstack([d2, self.cvr_enc.fit_transform(s2)]),
            self.logs["converted"][clk].astype(np.int64))
        del d2, s2, clk
        gc.collect()

        # ---------- 混排 / 降级兜底 ----------
        # w_base=0.0：候选池已含热度信息，混排阶段不再二次注入热门偏差（见 mixer 文档）
        self.mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=top_k)
        self.deg = DegradeRanker(self.dfm.predict_ctr, _ArrayLookup(self.pop.ctr_))

        # ---------- 展示用查找表（2 万条，开销小） ----------
        self.name_map = {int(i): f"{_CAT[c]}-{int(i):05d}"
                         for i, c in enumerate(self.items["category_idx"])}
        self.cat_map = {int(i): _CAT[int(c)]
                        for i, c in enumerate(self.items["category_idx"])}

        # ---------- 供 A/B 使用的「伪 DataFrame」适配层 ----------
        self._ab_users = _ABUsers(self.users)
        self._ab_items = _ABItems(self.items)

    # ---------- 对外接口（与完整档语义一致） ----------

    def candidates(self, user_id, n: int = 200) -> list:
        pop_c = self.pop.recall(n)
        rule_c = self.rule.recall(int(user_id), self.trips_by_user[int(user_id)], k=n)
        return list(dict.fromkeys(pop_c + rule_c))

    def _predict_cvr(self, dense, sparse) -> np.ndarray:
        oh = self.cvr_enc.transform(sparse)
        return self.cvr_model.predict_proba(np.hstack([dense, oh]))[:, 1]

    def _score(self, ctr, cvr) -> np.ndarray:
        return (1.0 - self.cvr_weight) * np.asarray(ctr) + self.cvr_weight * np.asarray(cvr)

    def recommend(self, user_id, top_k=None) -> list:
        top_k = top_k or self.top_k
        cands = self.candidates(user_id)
        if not cands:
            return []
        arr = np.asarray(cands, dtype=np.int64)
        base = self.items["popularity"][arr].astype(np.float64)
        cd, cs = self.fs.user_candidates(int(user_id), arr)
        ctr = self.deg.predict_ctr(cd, cs, cands)
        cvr = self._predict_cvr(cd, cs)
        score = self._score(ctr, cvr)
        cats = [_CAT[c] for c in self.items["category_idx"][arr]]
        ranked = self.mixer.mix(cands, base, score, cats)
        info = {int(i): (b, c, v) for i, b, c, v in zip(arr, base, ctr, cvr)}
        out = []
        for r, iid in enumerate(ranked):
            b, c, v = info[int(iid)]
            out.append({
                "rank": r + 1,
                "item_id": int(iid),
                "name": self.name_map[int(iid)],
                "category": self.cat_map[int(iid)],
                "base_score": round(float(b), 4),
                "ctr": round(float(c), 4),
                "cvr": round(float(v), 4),
            })
        return out

    def recommend_items(self, user_id, top_k=None) -> list:
        return [d["item_id"] for d in self.recommend(user_id, top_k)]

    def popular_rank(self, user_id, top_k=None) -> list:
        return self.pop.recall(k=top_k or self.top_k)

    def explain_steps(self, user_id, top_k=None, show: int = 20) -> dict:
        top_k = top_k or self.top_k
        cands = self.candidates(user_id)[:show]
        arr = np.asarray(cands, dtype=np.int64)
        base = self.items["popularity"][arr].astype(np.float64)
        cd, cs = self.fs.user_candidates(int(user_id), arr)
        ctr = self.deg.predict_ctr(cd, cs, cands)
        cvr = self._predict_cvr(cd, cs)
        score = self._score(ctr, cvr)
        cats = [_CAT[c] for c in self.items["category_idx"][arr]]
        trace = self.mixer.mix_trace(cands, base, score, cats)
        trace["candidates"] = cands
        trace["names"] = [self.name_map[int(i)] for i in cands]
        trace["categories"] = cats
        trace["base"] = np.round(base, 4).tolist()
        trace["ctr"] = np.round(ctr, 4).tolist()
        trace["cvr"] = np.round(cvr, 4).tolist()
        return trace

    def run_ab(self, n_users: int = 1000, top_k=None) -> dict:
        """模拟在线 A/B（对照=热门，实验=DeepFM+混排）。

        ⚠️ 该口径下热门基线存在平滑 CTR 稀疏性偏差，会高估实验组增益；
        可引用结论请用 `benchmark_rankers`。
        """
        return simulate_ab(self._ab_users, self._ab_items,
                           control_ranker=self.popular_rank,
                           treatment_ranker=self.recommend_items,
                           n_users=int(n_users), top_k=top_k or self.top_k)

    def benchmark_rankers(self, n_users: int = 100, top_k=None, seed: int = 42) -> dict:
        """**真实 CTR 口径**下对比 随机 / 热门 / DeepFM+混排，并给出逐用户 Oracle 上界。

        复用完整档的 `compare_rankers`（口径必须一致），仅通过适配层把 numpy
        数据喂进去。该口径不依赖平滑估计，**结论可引用**。
        """
        from src.eval.ab import compare_rankers

        rng = np.random.default_rng(seed)
        n_items = len(self.items["item_id"])
        k = top_k or self.top_k

        def rng_rank(uid, kk):
            return rng.choice(n_items, size=kk, replace=False).tolist()

        # 样本用户：取历史有点击的用户，保证候选/特征可用
        clicked = self.logs["user_id"][self.logs["clicked"] == 1]
        uniq = np.unique(clicked)[:int(n_users)]
        sample = [int(u) for u in uniq]

        return compare_rankers(
            self._ab_users, self._ab_items,
            rankers={
                "随机": rng_rank,
                "热门": lambda uid, kk: self.pop.recall(kk),
                "DeepFM+混排": lambda uid, kk: self.recommend_items(uid, kk),
            },
            sample_users=sample, top_k=k, n_items=n_items, seed=seed)

    def baseline_bias(self, top_n: int = 200, sample_user: int = 0) -> dict:
        """热门榜「平滑 CTR vs 真实 CTR」偏差，用于透明化稀疏性偏差。"""
        from src.data.synthetic_generator import compute_true_ctr

        try:
            ids = np.asarray(self.pop.recall(top_n), dtype=np.int64)
            smooth = float(self.pop.ctr_[ids].mean())
            real = float(compute_true_ctr(
                self._ab_users, self._ab_items,
                np.full(len(ids), int(sample_user)), ids).mean())
            return {"top_n": top_n, "smooth_ctr": smooth,
                    "true_ctr": real, "bias": smooth - real}
        except Exception as e:                              # 容错，不阻断主流程
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# 适配层：`simulate_ab` 与 `compute_true_ctr/cvr` 使用 pandas 的 iloc / 列名访问，
# 这里提供最小接口的代理对象，使 numpy 数据可直接接入**完整档同一份评估代码**，
# 避免为轻量档另写一套 A/B 逻辑（口径必须一致才有意义）。
# ---------------------------------------------------------------------------
class _ABUsers:
    """只实现 `compute_true_ctr/cvr` 所需接口的 users 代理。

    `_pair_features` 用到：`users.iloc[ids]`（属性，非方法）、
    `u[[列名列表]]`、`u[列名]`、`it[列名]`。
    这里用最小代理实现，让 numpy 数据可接入**完整档同一份评估代码**。
    """

    def __init__(self, d: dict):
        self._d = d
        self.iloc = _IlocProxy(d)

    def __len__(self):
        return len(np.asarray(self._d["user_id"]))

    def __getitem__(self, key):
        if isinstance(key, list):                     # u[[...]] → 仍需支持 .to_numpy()
            return {k: np.asarray(self._d[k]) for k in key}
