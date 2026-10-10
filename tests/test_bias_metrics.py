"""M2 验收：偏差度量正确，且能复现项目已记录的偏差现象。

两类测试
--------
1. **度量本身的正确性**：用构造数据验证 KL / Gini / bias / NDCG 的取值符合解析预期。
2. **现象复现**：在合成数据上跑出「热门基线平滑 CTR ≫ 真实 CTR」，
   与 README 记录的完整档现象（+0.3704）**同向**。这一步确认偏差框架接得上既有发现。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bias.metrics import (bias_summary, ctr_bias, ctr_mae,
                              exposure_distribution, exposure_gini,
                              exposure_kl, ndcg_at_k, smoothed_ctr,
                              topk_overlap)


# ----------------------------------------------------------------------
# 曝光偏差度量的解析正确性
# ----------------------------------------------------------------------
def test_uniform_exposure_has_zero_kl_and_gini():
    p = np.full(10, 0.1)
    assert exposure_kl(p) == pytest.approx(0.0, abs=1e-9)
    assert exposure_gini(p) == pytest.approx(0.0, abs=1e-9)


def test_kl_positive_when_skewed():
    p = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    assert exposure_kl(p) > 0


def test_kl_known_value():
    """两点分布 P=[0.5,0.5] vs U 的 KL=0；P=[1,0] vs U=[0.5,0.5] 的 KL=ln2。"""
    assert exposure_kl(np.array([1.0, 0.0])) == pytest.approx(np.log(2), abs=1e-6)


def test_gini_extremes():
    # 完全集中：一人拿全部
    assert exposure_gini(np.array([0.0, 0.0, 1.0])) > 0.6
    # 均匀
    assert exposure_gini(np.full(4, 0.25)) == pytest.approx(0.0, abs=1e-9)


def test_exposure_distribution_counts():
    freq = exposure_distribution(np.array([0, 1, 1, 2, 2, 2]), n_candidates=3)
    assert np.allclose(freq, [1 / 6, 2 / 6, 3 / 6])


# ----------------------------------------------------------------------
# 统计偏差
# ----------------------------------------------------------------------
def test_ctr_bias_sign_and_magnitude():
    est = np.array([0.5, 0.6, 0.7])
    true = np.array([0.2, 0.3, 0.4])
    assert ctr_bias(est, true) == pytest.approx(0.3)      # 系统性高估
    assert ctr_mae(est, true) == pytest.approx(0.3)
    # 低估
    assert ctr_bias(true, est) == pytest.approx(-0.3)


def test_smoothed_ctr_matches_formula():
    out = smoothed_ctr(np.array([0, 5]), np.array([0, 10]), alpha=5.0)
    assert np.allclose(out, [1.0, 10 / 15])


def test_zero_impression_item_gets_prior_ctr():
    """曝光 0 次的物品（冷门）平滑后 = α/(0+α) = 1 —— 这正是偏差的机制本身。"""
    out = smoothed_ctr(np.array([0]), np.array([0]), alpha=5.0)
    assert out[0] == pytest.approx(1.0)
    # 而它的"真值"可能远低于 1，差就是偏差
    assert ctr_bias(out, np.array([0.1])) == pytest.approx(0.9)


# ----------------------------------------------------------------------
# 排序偏差
# ----------------------------------------------------------------------
def test_topk_overlap_identical_and_disjoint():
    assert topk_overlap([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)
    assert topk_overlap([1, 2, 3], [4, 5, 6], k=3) == pytest.approx(0.0)
    # 3 个里重合 2 个 → |∩|/|∪| = 2/4
    assert topk_overlap([1, 2, 3], [1, 2, 4], k=3) == pytest.approx(0.5)


def test_ndcg_perfect_ranking_is_one():
    rel = {"a": 0.9, "b": 0.5, "c": 0.1}
    assert ndcg_at_k(["a", "b", "c"], rel, k=3) == pytest.approx(1.0)


def test_ndcg_worse_when_reversed():
    rel = {"a": 0.9, "b": 0.5, "c": 0.1}
    good = ndcg_at_k(["a", "b", "c"], rel, k=3)
    bad = ndcg_at_k(["c", "b", "a"], rel, k=3)
    assert good > bad


# ----------------------------------------------------------------------
# 现象复现：热门基线平滑 CTR ≫ 真实 CTR
# ----------------------------------------------------------------------
def test_reproduce_known_bias_phenomenon_on_synthetic():
    """在 micro 合成档上复现「平滑 CTR 高于真实 CTR」的既有现象。

    期望：bias = smooth − true 显著为正（与 README 完整档 +0.3704 同向）。
    """
    import pandas as pd
    from src.bias.metrics import reproduce_known_phenomenon
    from src.data.synthetic_generator_lite import build_dataset_np
    from src.recall.baselines import PopularRecall

    items, users, trips, logs = build_dataset_np(seed=42)
    items_df = pd.DataFrame(items)
    logs_df = pd.DataFrame({"item_id": logs["item_id"], "clicked": logs["clicked"]})
    users_df = pd.DataFrame(users)

    pop = PopularRecall(top_k=200, alpha=5.0).fit(logs_df)
    out = reproduce_known_phenomenon(pop, users_df, items_df, top_n=200, sample_user=0)

    assert "error" not in out, out
    # 核心断言：平滑估计系统性高于真值
    assert out["smooth_ctr"] > out["true_ctr"], out
    assert out["bias"] > 0, out
    # 偏差幅度应与完整档现象同量级（>0.02 就算显著）
    assert out["bias"] > 0.02, f"偏差过小，现象未复现：{out}"


def test_bias_summary_bundles_metrics():
    s = bias_summary(exposure_freq=np.full(5, 0.2),
                     est_ctr=np.array([0.5, 0.5]), true_ctr=np.array([0.4, 0.4]))
    assert s["exposure_kl_nats"] == pytest.approx(0.0, abs=1e-4)
    assert s["ctr_bias"] == pytest.approx(0.1)
