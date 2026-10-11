"""在线引擎（micro 档，纯 numpy / 零 torch）组件与端到端测试。

`engine_micro` 只依赖 numpy，故无需 torch 即可导入与构造。
完整引擎测试依赖 `artifacts/micro/` 下的离线工件（已入库）；
若工件缺失则 skip，不阻塞 CI（需在本地/构建环境用
`python -m src.build_online_artifacts micro` 重新生成）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.serve.engine_micro import (
    LitePopularRecall,
    LiteRuleRecall,
    MicroRecommendationEngine,
    _ArrayLookup,
)

_ART = Path(__file__).resolve().parents[1] / "artifacts" / "micro"


@pytest.fixture(scope="module")
def micro_engine():
    if not (_ART / "deepfm_ctr.npz").exists():
        pytest.skip("在线工件 artifacts/micro 缺失，跳过完整引擎测试")
    return MicroRecommendationEngine()


def test_popular_recall_orders_by_smoothed_ctr():
    item_id = np.array([0, 1, 2, 3])
    clicked = np.array([1, 0, 1, 0])
    pop = LitePopularRecall(n_items=4, alpha=5.0).fit(item_id, clicked)
    top = pop.recall(k=4)
    assert len(top) == 4
    assert set(top) == {0, 1, 2, 3}
    # 有点击的物品（平滑 CTR 更高）应排在零点击物品之前
    assert top[0] in (0, 2) and top[1] in (0, 2)


def test_rule_recall_returns_route_items_only():
    items = {
        "item_id": np.arange(4),
        "category_idx": np.array([0, 0, 1, 1]),     # 0/1 为航线类目
        "popularity": np.array([0.5, 0.4, 0.3, 0.2]),
        "quality": np.array([0.9, 0.8, 0.7, 0.6]),
        "is_miles_ticket": np.array([0, 0, 0, 0]),
        "miles_required": np.array([0, 0, 0, 0]),
        "stock": np.array([1, 1, 1, 1]),
        "origin_idx": np.array([0, 1, 0, 1]),
        "dest_idx": np.array([1, 0, 1, 0]),
    }
    users = {"miles_balance": np.array([100.0]), "home_idx": np.array([0])}
    rule = LiteRuleRecall(items, users, miles_threshold=0.10)
    rec = rule.recall(0, [], k=10)
    assert all(i in (0, 1) for i in rec)            # 仅航线类目进入候选
    assert len(set(rec)) == len(rec)                # 无重复


def test_array_lookup_bounds_and_default():
    arr = np.array([0.1, 0.2, 0.3])
    lu = _ArrayLookup(arr, default=0.05)
    assert lu.get(0) == 0.1
    assert lu.get(2) == 0.3
    assert lu.get(5) == 0.05                        # 越界 → 默认
    assert lu.get("bad") == 0.05                    # 非整数 → 默认
    assert lu.get(1, default=-1.0) == 0.2


def test_micro_engine_recommend_shape_and_validity(micro_engine):
    res = micro_engine.recommend(0, top_k=5)
    assert 1 <= len(res) <= 5
    ids = [d["item_id"] for d in res]
    assert len(set(ids)) == len(ids)                # 去重
    assert [d["rank"] for d in res] == list(range(1, len(res) + 1))
    for d in res:
        assert 0.0 <= d["ctr"] <= 1.0
        assert 0.0 <= d["cvr"] <= 1.0
        assert d["base_score"] >= 0.0


def test_micro_engine_recommend_items_in_range(micro_engine):
    out = micro_engine.recommend_items(0, top_k=8)
    assert len(out) == len(set(out))
    assert all(0 <= i < micro_engine.n_items for i in out)


def test_micro_engine_benchmark_structure(micro_engine):
    rep = micro_engine.benchmark_rankers(n_users=20, seed=42)
    assert "per_method" in rep
    pm = rep["per_method"]
    assert {"随机", "热门", "DeepFM+混排"}.issubset(pm.keys())
    for name, m in pm.items():
        assert 0.0 <= m["true_ctr"] <= 1.0          # 真实 CTR 口径，比例值
    assert rep["n_users"] == 20
