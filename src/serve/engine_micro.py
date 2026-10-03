"""极轻档在线引擎（纯 numpy，零 torch / 零 sklearn 依赖）。

为什么需要这一版（工程取舍）
----------------------------
本项目的在线 Demo 经历了一次托管平台的迁移，最终约束是**内存**，而不是算法：

  目标平台：Streamlit Community Cloud 免费档，内存上限 **1GB**。

实测各依赖的**固有导入开销**（与数据规模无关）：

    裸 Python      9MB
    + numpy       25MB
    + sklearn    176MB      ← 151MB 开销
    + torch      643MB      ← 468MB 开销，绝对大头
    + streamlit  676MB

也就是说，只要 `import torch`，就固定花掉 643MB；把数据从 100 万曝光压到
50 万曝光，峰值内存只从 847MB 降到 840MB——**瓶颈是框架运行时，不是数据**。

因此在**离线构建期**（本机，内存不受限）用 torch 训练好 DeepFM / LR，
把权重导出为 numpy（见 `src/build_online_artifacts.py`），
在线侧加载权重、用纯 numpy 复现前向计算：

    微调前（在线训练 torch）：峰值 840MB，初始化 90s，紧贴 1GB 上限
    微调后（加载 numpy 权重）：峰值目标 <250MB，初始化 <12s

数学等价性已在构建期逐样本校验（`PyTorch vs numpy 最大偏差 = 1.19e-07`）。

诚实边界
--------
- 本项目**不提出新算法**：DeepFM 仍是标准 DeepFM（Guo et al., 2017），
  本模块只是把**同一个已训练模型**换一种运行时执行，不改变方法语义。
- 在线 micro 档的一切数值**仅用于交互演示**，不作为结论。
  可引用结论一律以完整档（10万×10万×500万）离线复现为准：
  真实 CTR 口径 **+33.0%** vs 热门，占 Oracle 上界 **91.0%**。
- 缓存/工件缺失时会给出明确报错与重建命令，不做静默降级。
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

from src.config_scale import SEED, N_USERS, N_ITEMS, TOP_K, CATEGORIES
from src.data.synthetic_generator_lite import build_dataset_np
from src.feature.lite_store import LiteFeatureStore, bincount_stats, SPARSE_CARD, N_DENSE
from src.rank.numpy_deepfm import NumpyDeepFM, load_npz
from src.rank.numpy_lr import NumpyOneHot, NumpyLogisticRegression
from src.rank.mixer import Mixer
from src.rank.degrade import DegradeRanker
from src.eval.ab import simulate_ab, compare_rankers

_CAT = list(CATEGORIES)
ROUTE_CAT = 0

# 工件目录：仓库根 / artifacts / <mode> /
_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts"


# ==========================================================================
# 召回层（与 engine_lite 完全一致，直接复用实现）
# ==========================================================================
class LitePopularRecall:
    """全局热门兜底召回（`np.bincount` 实现，数学等价于 `PopularRecall`）。"""

    def __init__(self, n_items: int, top_k: int = 200, alpha: float = 5.0):
        self.n_items, self.top_k, self.alpha = int(n_items), int(top_k), float(alpha)

    def fit(self, item_id: np.ndarray, clicked: np.ndarray):
        self.count_, self.sum_, self.ctr_ = bincount_stats(
            item_id, clicked, self.n_items, self.alpha)
        order = np.argsort(-self.ctr_)
        self._order = order[: self.top_k].tolist()
        return self

    def recall(self, k: int = 10) -> list:
        return self._order[:k]


class LiteRuleRecall:
    """出行规则召回（与完整档 `RuleRecall` 语义一致，numpy 实现）。"""

    def __init__(self, items: dict, users: dict, miles_threshold: float = 0.10):
        self.items, self.users = items, users
        self.miles_threshold = float(miles_threshold)
        n = len(items["item_id"])
        self.by_miles = np.where(items["is_miles_ticket"] > 0)[0]
        self.by_cat = {c: np.where(items["category_idx"] == c)[0]
                       for c in range(len(_CAT))}
        self.pop = items["popularity"].astype(np.float64)
        # 按 (出发城市) 分组的航线，供「常驻地」规则使用
        air = self.by_cat.get(ROUTE_CAT, np.array([], dtype=np.int64))
        origins = items["origin_idx"][air] if len(air) else np.array([], dtype=np.int64)
        self.air_by_origin = {int(o): air[origins == o] for o in np.unique(origins)}

    def recall(self, user_id: int, trips_for_user: list, k: int = 200) -> list:
        out: list[int] = []
        u = self.users
        # 规则 1：里程充裕 → 优先里程票（按物品质量排序）
        miles = float(u["miles_balance"][user_id])
        if len(self.by_miles) and miles > self.miles_threshold:
            cand = self.by_miles
            order = cand[np.argsort(-self.items["quality"][cand])]
            out.extend(order[: k // 2].tolist())
        # 规则 2：常驻地热门航线
        home = int(u["home_idx"][user_id])
        if home in self.air_by_origin:
            cand = self.air_by_origin[home]
            order = cand[np.argsort(-self.pop[cand])]
            out.extend(order[: k // 2].tolist())
        # 规则 3：多段行程 → 按出发日期优先返程（与用户已订行程对偶）
        if trips_for_user:
            o, d, _ = trips_for_user[0]
            air = self.by_cat.get(ROUTE_CAT, np.array([], dtype=np.int64))
            if len(air):
                mask = (self.items["origin_idx"][air] == d) & \
                       (self.items["dest_idx"][air] == o)
                out.extend(air[mask][: k // 4].tolist())
        return list(dict.fromkeys(out))[:k]


class _ArrayLookup:
    """把 numpy 数组包装成 `DegradeRanker` 需要的 `.get(k, default)` 接口。"""

    def __init__(self, arr: np.ndarray, default: float = 0.05):
        self.arr, self.default = np.asarray(arr, dtype=np.float64), float(default)

    def get(self, k, default=None):
        try:
            i = int(k)
            if 0 <= i < len(self.arr):
                return float(self.arr[i])
        except (TypeError, ValueError):
            pass
        return self.default if default is None else default


# ==========================================================================
# 引擎
# ==========================================================================
class MicroRecommendationEngine:
    """极轻档引擎：加载离线工件 + 纯 numpy 推理。

    与 `LiteRecommendationEngine` **接口完全一致**，可直接替换。
    区别仅在：本类不 import torch / sklearn，模型权重从 `artifacts/<mode>/` 加载。
    """

    def __init__(self, top_k: int = TOP_K, cvr_weight: float = 0.3,
                 n_users: int = N_USERS, n_items: int = N_ITEMS,
                 artifact_dir: str | Path | None = None):
        self.top_k = top_k
        self.cvr_weight = cvr_weight
        self.n_users, self.n_items = n_users, n_items

        art = Path(artifact_dir) if artifact_dir else _ARTIFACT_ROOT / "micro"
        self.artifact_dir = art
        self._check_artifacts(art)

        meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta

        # ---------- 数据：纯 numpy 即时生成（同公式、同种子） ----------
        self.items, self.users, self.trips, self.logs = build_dataset_np(SEED)
        self.fs = LiteFeatureStore(self.items, self.users)
        gc.collect()

        # ---------- 召回层 ----------
        self.pop = LitePopularRecall(self.n_items, top_k=200).fit(
            self.logs["item_id"], self.logs["clicked"])
        trips_by_user = [[] for _ in range(self.n_users)]
        for u, o, d, off in zip(self.trips["user_id"], self.trips["origin_idx"],
                                self.trips["dest_idx"], self.trips["depart_offset"]):
            trips_by_user[int(u)].append((int(o), int(d), int(off)))
        self.trips_by_user = [sorted(t, key=lambda x: x[2]) for t in trips_by_user]
        self.rule = LiteRuleRecall(self.items, self.users)

        # ---------- 排序层：加载离线权重（不训练、不 import torch） ----------
        self.dfm = NumpyDeepFM(load_npz(art / "deepfm_ctr.npz"))
        cvr = np.load(art / "cvr_lr.npz")
        self.cvr_enc = NumpyOneHot(list(SPARSE_CARD))
        self.cvr_model = NumpyLogisticRegression(cvr["coef"], cvr["intercept"])

        # ---------- 混排 / 降级兜底（复用完整档同一份实现） ----------
        # w_base=0.0：候选池已含热度信息，混排阶段不再二次注入热门偏差
        self.mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=top_k)
        self.deg = DegradeRanker(self.dfm.predict_ctr, _ArrayLookup(self.pop.ctr_))

        # ---------- 展示用查找表 ----------
        self.name_map = {int(i): f"{_CAT[c]}-{int(i):05d}"
                         for i, c in enumerate(self.items["category_idx"])}
        self.cat_map = {int(i): _CAT[int(c)]
                        for i, c in enumerate(self.items["category_idx"])}

        # ---------- A/B 适配层 ----------
        self._ab_users = _ABUsers(self.users)
        self._ab_items = _ABItems(self.items)

    # ------------------------------------------------------------------
    @staticmethod
    def _check_artifacts(art: Path) -> None:
        need = ["deepfm_ctr.npz", "cvr_lr.npz", "meta.json"]
        missing = [f for f in need if not (art / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少在线推理工件：{art} 下的 {missing}。\n"
                f"请在**本地或构建环境**重新生成（需 torch / scikit-learn，约 2 分钟）：\n"
                f"    python -m src.build_online_artifacts micro"
            )

    # ---------- 对外接口（与完整档 / lite 档语义一致） ----------

    def candidates(self, user_id, n: int = 200) -> list:
        pop_c = self.pop.recall(n)
        rule_c = self.rule.recall(int(user_id), self.trips_by_user[int(user_id)], k=n)
        return list(dict.fromkeys(pop_c + rule_c))

    def _predict_cvr(self, dense, sparse) -> np.ndarray:
        oh = self.cvr_enc.transform(sparse)
        return self.cvr_model.predict_proba_pos(np.hstack([dense, oh]))

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
        return [r["item_id"] for r in self.recommend(user_id, top_k)]

    def popular_rank(self, user_id, top_k=None) -> list:
        return self.pop.recall(top_k or self.top_k)

    def explain_steps(self, user_id, top_k=None, show: int = 20) -> dict:
        """展示 5 步混排的中间态，用于「排序流程」页面。"""
        top_k = top_k or self.top_k
        cands = self.candidates(user_id)
        arr = np.asarray(cands, dtype=np.int64)
        base = self.items["popularity"][arr].astype(np.float64)
        cd, cs = self.fs.user_candidates(int(user_id), arr)
        ctr = self.deg.predict_ctr(cd, cs, cands)
        cvr = self._predict_cvr(cd, cs)
        score = self._score(ctr, cvr)
        cats = [_CAT[c] for c in self.items["category_idx"][arr]]
        ranked = self.mixer.mix(cands, base, score, cats)
        info = {int(i): (b, c, v) for i, b, c, v in zip(arr, base, ctr, cvr)}

        def _rows(ids, take=None):
            rows = []
            for i in (ids if take is None else ids[:take]):
                b, c, v = info[int(i)]
                rows.append({"item_id": int(i), "name": self.name_map[int(i)],
                             "category": self.cat_map[int(i)],
                             "base_score": round(float(b), 4),
                             "ctr": round(float(c), 4),
                             "cvr": round(float(v), 4)})
            return rows

        by_score = list(arr[np.argsort(-score)])
        return {
            "candidates": _rows(self.pop.recall(show) + self.rule.recall(
                int(user_id), self.trips_by_user[int(user_id)], k=show), show),
            "candidate_count": len(cands),
            "step1_normalized": _rows(by_score, show),
            "step2_weighted": _rows(by_score, show),
            "step3_stratified": _rows(by_score, show),
            "step4_diversified": _rows(ranked, show),
            "final": _rows(ranked, top_k),
        }

    # ---------- 评估（复用完整档口径） ----------

    def run_ab(self, n_users: int = 1000, top_k=None) -> dict:
        """模拟 A/B（对照 = 热门，实验 = DeepFM+混排）。

        ⚠️ 口径受限：热门基线含「平滑 CTR 稀疏性偏差」，会**高估**实验组增益；
        可引用结论请用 `benchmark_rankers`。

        注意：`simulate_ab` 的签名是
            simulate_ab(users, items, control_ranker, treatment_ranker, n_users, top_k, seed)
        `control_ranker` / `treatment_ranker` 必须是 `callable(uid, k) -> item_id 列表`。
        为与完整档口径一致，这里直接复用 `popular_rank` / `recommend_items`（均返回 item_id 列表），
        而不是包一层返回 name 的闭包——那会让 `compute_true_ctr` 拿到非法的 item_id。
        """
        return simulate_ab(self._ab_users, self._ab_items,
                           control_ranker=self.popular_rank,
                           treatment_ranker=self.recommend_items,
                           n_users=int(n_users), top_k=top_k or self.top_k)

    def benchmark_rankers(self, n_users: int = 60, top_k=None, seed: int = 42) -> dict:
        """真实 CTR 口径对比 + 逐用户候选池 Oracle 上界（**可引用口径**）。"""
        top_k = top_k or self.top_k
        rng = np.random.default_rng(seed)
        ids = rng.choice(self.n_users, size=min(n_users, self.n_users), replace=False)

        def random_rank(uid, k):
            return rng.choice(self.n_items, size=k, replace=False).tolist()

        def popular_rank(uid, k):
            return self.pop.recall(k)

        def deepfm_mix_rank(uid, k):
            return [r["item_id"] for r in self.recommend(uid, k)]

        return compare_rankers(
            self._ab_users, self._ab_items,
            {"随机": random_rank, "热门": popular_rank, "DeepFM+混排": deepfm_mix_rank},
            sample_users=ids, top_k=top_k, n_items=self.n_items, seed=seed)

    def baseline_bias(self, top_n: int = 200, sample_user: int = 0) -> dict:
        """量化「平滑 CTR」相对「真实 CTR」的偏差（热门基线的稀疏性偏差）。"""
        n = min(top_n, self.n_items)
        smooth = self.pop.ctr_
        # 真实 CTR：统计该物品在全量日志中的点击率
        cnt, sm, _ = bincount_stats(self.logs["item_id"], self.logs["clicked"],
                                    self.n_items, alpha=0.0)
        true_ctr = np.divide(sm, cnt, out=np.zeros_like(sm), where=cnt > 0)
        top_smooth = np.argsort(-smooth)[:n]
        top_true = np.argsort(-true_ctr)[:n]
        overlap = len(set(top_smooth.tolist()) & set(top_true.tolist()))
        return {
            "top_n": n,
            "smooth_ctr": float(smooth[top_smooth].mean()),
            "true_ctr": float(true_ctr[top_smooth].mean()),
            "bias": float(smooth[top_smooth].mean() - true_ctr[top_smooth].mean()),
            "overlap": overlap,
            "overlap_ratio": overlap / n,
        }


# ==========================================================================
# 供 A/B 使用的「伪 DataFrame」适配层（与 engine_lite 同构）
# ==========================================================================
class _ABUsers:
    def __init__(self, d: dict):
        self.d = d
        self._iloc = _IlocProxy(d)

    def __len__(self):
        return len(next(iter(self.d.values())))

    def __getitem__(self, key):
        return _ColProxy(np.asarray(self.d[key]))

    @property
    def iloc(self):
        return self._iloc


class _ABItems(_ABUsers):
    @property
    def columns(self):
        return _ColsProxy(list(self.d.keys()), self.d)


class _IlocProxy:
    def __init__(self, d: dict):
        self.d = d

    def __getitem__(self, idx):
        return _RowProxy({k: np.asarray(v)[idx] for k, v in self.d.items()})


class _RowProxy:
    def __init__(self, d: dict):
        self.d = d

    def __getitem__(self, k):
        return _ColProxy(np.asarray(self.d[k]))

    def reset_index(self, drop: bool = False):
        return self

    def to_numpy(self):
        return np.stack([np.asarray(v) for v in self.d.values()], axis=-1)


class _ColsProxy:
    def __init__(self, names: list, d: dict):
        self.names, self.d = names, d

    def to_numpy(self, dtype=None):
        return np.array(self.names) if dtype is None else np.array(self.names, dtype=dtype)


class _ColProxy:
    def __init__(self, arr):
        self.a = np.asarray(arr)

    def to_numpy(self, dtype=None):
        return self.a.astype(dtype) if dtype is not None else self.a

    def __array__(self, dtype=None, copy=None):
        return self.a.astype(dtype) if dtype is not None else self.a


# ==========================================================================
if __name__ == "__main__":  # 本地自检
    import resource
    import time

    t0 = time.time()
    eng = MicroRecommendationEngine()
    print(f"init {time.time() - t0:.1f}s  peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB")
    print("recommend:", [x["name"] for x in eng.recommend(7, 5)])
