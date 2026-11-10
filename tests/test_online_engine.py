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
        assert 0.0 <= m["ndcg_at_k"] <= 1.0         # 端到端 NDCG@K
    assert rep["n_users"] == 20


# ---------------------------------------------------------------------------
# 去偏判据的在线接入
# ---------------------------------------------------------------------------
def test_estimate_propensity_from_log_sums_to_one():
    from src.serve.bias_guard import estimate_propensity_from_log

    ids = np.array([0, 0, 0, 1, 2, 2])
    pi = estimate_propensity_from_log(ids, n_candidates=10, lam=1.0)
    assert pi.shape == (10,)
    assert abs(pi.sum() - 1.0) < 1e-9
    # 未曝光物品拿到平滑后的正概率（λ=1）
    assert (pi > 0).all()


def test_estimate_propensity_raw_lambda_zero_gives_zero_for_unseen():
    """λ=0 时未曝光物品 π̂=0 —— 这正是 1/π̂ 权重爆炸的根源，
    线上默认必须用 λ=1。"""
    from src.serve.bias_guard import estimate_propensity_from_log

    pi = estimate_propensity_from_log(np.array([0, 0, 1]), n_candidates=5, lam=0.0)
    assert pi[2] == 0.0
    assert pi[0] > pi[1]


def test_bias_guard_reports_caliber_and_threshold(micro_engine):
    """判据输出必须自带口径与阈值来源说明——
    Coat 实测证明 ESS/N 随口径横跨边界两侧，不带口径的数字没有意义。"""
    g = micro_engine.bias_guard()
    for key in ("ess", "ess_ratio", "should_debias", "recommendation",
                "caliber", "threshold_note", "lam", "n_records"):
        assert key in g, key
    assert 0.0 <= g["ess_ratio"] <= 1.0
    assert isinstance(g["should_debias"], bool)
    assert "口径" in g["caliber"]


def test_bias_guard_detects_skewed_log():
    """构造极端偏斜的日志（一个物品占绝大多数曝光），
    ESS/N 必须显著下降并给出「谨慎/不建议」的判断。"""
    from src.serve.bias_guard import bias_guard

    skewed = np.array([0] * 990 + list(range(1, 11)) * 10)
    g = bias_guard(skewed, n_candidates=1000, lam=1.0)
    assert g["ess_ratio"] < 0.4
    assert g["should_debias"] is False
    assert "谨慎" in g["recommendation"] or "不建议" in g["recommendation"]


def test_bias_guard_uses_record_level_caliber_not_pool_level():
    """**记录级口径**（口径 A）而非候选池级口径 —— 这是一个会翻转结论的关键区别。

    同一份偏斜日志：
        候选池级 ESS/N = 0.9907 → 「可以放心去偏」
        记录级   ESS/N = 0.1129 → 「不建议去偏」
    若误用候选池级口径，判据会给出完全相反的结论。
    Coat 实验的四口径结果同样证明口径必须显式报告。
    """
    from src.bias.estimators import effective_sample_size
    from src.serve.bias_guard import bias_guard, estimate_propensity_from_log

    ids = np.array([0] * 990 + list(range(1, 11)) * 10)
    c = 1000
    pi_item = estimate_propensity_from_log(ids, c, lam=1.0)

    def ess_ratio(v):
        v = np.asarray(v, dtype=float)
        return effective_sample_size(v) / len(v)

    pool_level = ess_ratio(pi_item)
    record_level = ess_ratio(pi_item[ids])
    # 两者确实不同，且方向相反——这就是必须锁住口径的原因
    assert pool_level > 0.9
    assert record_level < 0.4

    # 线上判据必须采用记录级口径
    g = bias_guard(ids, c, lam=1.0)
    assert abs(g["ess_ratio"] - record_level) < 1e-3
    assert g["should_debias"] is False


def test_recommend_with_guard_returns_switch_signal(micro_engine):
    r = micro_engine.recommend_with_guard(0, top_k=5)
    assert "recommendations" in r and "bias_guard" in r
    assert r["ranking_applied"] in ("ctr_with_debias", "popularity_fallback")
    # 开关信号必须与判据结论一致
    assert (r["ranking_applied"] == "ctr_with_debias") == r["bias_guard"]["should_debias"]
