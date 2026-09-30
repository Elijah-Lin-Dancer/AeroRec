"""推荐服务引擎：封装 数据加载 + 召回 + 多目标(CTR+CVR)排序 + 混排 + A/B。

供 FastAPI 与 Streamlit 共用，保证演示口径一致。首次实例化需训练模型（约 30 秒）。

排序目标：combined = (1-w_cvr) * CTR + w_cvr * CVR。
单目标 CTR 排序会提升点击但牺牲转化率；引入 CVR 项做多目标权衡（对应工业多目标排序）。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src.config import TOP_K
from src.data.synthetic_generator import build_dataset
from src.feature.store import build_training_matrix, build_candidate_features, feature_dim
from src.recall.baselines import PopularRecall
from src.recall.rule import RuleRecall
from src.rank.trainer import DeepFMRanker
from src.rank.mixer import Mixer
from src.rank.degrade import DegradeRanker, compute_recent_ctr
from src.eval.ab import simulate_ab


class RecommendationEngine:
    def __init__(self, train_sample: int = 100_000, deepfm_epochs: int = 2,
                 embed_dim: int = 16, dnn_hidden: tuple = (64, 32),
                 cvr_weight: float = 0.3, top_k: int = TOP_K):
        self.top_k = top_k
        self.cvr_weight = cvr_weight
        print("[engine] 加载数据 ...")
        self.items, self.users, self.trips, self.logs = build_dataset()

        print("[engine] 训练召回 + 排序模型 ...")
        self.pop = PopularRecall(top_k=200).fit(self.logs)
        self.rule = RuleRecall().fit(logs=self.logs, items=self.items,
                                     users=self.users, trips=self.trips)

        dense_dim, _, cards = feature_dim()
        dense, sparse, yc, _ = build_training_matrix(self.logs, self.users, self.items,
                                                     sample=train_sample)
        self.dfm = DeepFMRanker(dense_dim, cards, embed_dim=embed_dim, dnn_hidden=dnn_hidden,
                                epochs=deepfm_epochs, batch_size=2048).fit(dense, sparse, yc)

        # CVR 模型：在点击样本上学转化率（轻量 LR，做多目标排序）
        clicked = self.logs[self.logs["clicked"] == 1]
        d2, s2, _, yconv = build_training_matrix(clicked, self.users, self.items,
                                                 sample=min(200_000, len(clicked)))
        self.cvr_enc = OneHotEncoder(categories=[range(c) for c in cards], sparse_output=False)
        self.cvr_model = LogisticRegression(max_iter=1000).fit(
            np.hstack([d2, self.cvr_enc.fit_transform(s2)]), yconv)

        self.mixer = Mixer(w_base=0.4, w_ctr=0.6, max_same_category=2, top_k=top_k)
        self.deg = DegradeRanker(self.dfm.predict_ctr, compute_recent_ctr(self.logs))

        self.pop_map = self.items.set_index("item_id")["popularity"].to_dict()
        self.cat_map = self.items.set_index("item_id")["category"].to_dict()
        self.name_map = self.items.set_index("item_id")["name"].to_dict()

    def candidates(self, user_id, n: int = 200) -> list:
        return list(dict.fromkeys(self.pop.recall(n) + self.rule.recall(int(user_id), k=n)))

    def _predict_cvr(self, dense, sparse) -> np.ndarray:
        oh = self.cvr_enc.transform(sparse)
        return self.cvr_model.predict_proba(np.hstack([dense, oh]))[:, 1]

    def _score(self, ctr, cvr) -> np.ndarray:
        """多目标加权得分：CTR 为主、CVR 为辅。"""
        return (1.0 - self.cvr_weight) * np.asarray(ctr) + self.cvr_weight * np.asarray(cvr)

    def recommend(self, user_id, top_k=None) -> list:
        """返回推荐列表（含 CTR/CVR 与多目标得分等解释字段）。"""
        top_k = top_k or self.top_k
        cands = self.candidates(user_id)
        if not cands:
            return []
        base = [self.pop_map.get(i, 0.0) for i in cands]
        cd, cs = build_candidate_features(int(user_id), cands, self.users, self.items)
        ctr = self.deg.predict_ctr(cd, cs, cands)
        cvr = self._predict_cvr(cd, cs)
        score = self._score(ctr, cvr)
        cats = [self.cat_map[i] for i in cands]
        ranked = self.mixer.mix(cands, base, score, cats)
        info = {iid: (b, c, v) for iid, b, c, v in zip(cands, base, ctr, cvr)}
        out = []
        for r, iid in enumerate(ranked):
            b, c, v = info[iid]
            out.append({
                "rank": r + 1,
                "item_id": int(iid),
                "name": self.name_map.get(iid, str(iid)),
                "category": self.cat_map[iid],
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
        """返回多目标混排的中间状态，供排序流程图展示。"""
        top_k = top_k or self.top_k
        cands = self.candidates(user_id)[:show]
        base = np.array([self.pop_map.get(i, 0.0) for i in cands])
        cd, cs = build_candidate_features(int(user_id), cands, self.users, self.items)
        ctr = self.deg.predict_ctr(cd, cs, cands)
        cvr = self._predict_cvr(cd, cs)
        score = self._score(ctr, cvr)
        cats = [self.cat_map[i] for i in cands]
        trace = self.mixer.mix_trace(cands, base, score, cats)
        trace["candidates"] = cands
        trace["names"] = [self.name_map.get(i, str(i)) for i in cands]
        trace["categories"] = cats
        trace["base"] = np.round(base, 4).tolist()
        trace["ctr"] = np.round(ctr, 4).tolist()
        trace["cvr"] = np.round(cvr, 4).tolist()
        return trace

    def run_ab(self, n_users: int = 2000, top_k=None) -> dict:
        return simulate_ab(self.users, self.items,
                           control_ranker=self.popular_rank,
                           treatment_ranker=self.recommend_items,
                           n_users=n_users, top_k=top_k or self.top_k)
