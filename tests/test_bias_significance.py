"""numpy-only bootstrap 显著性检验。

断言确定性（固定 bootstrap 种子 → 零漂移门禁可逐字节复现）、p 值合法性、
已知分布下的方向正确性，以及三实验产物的向后兼容（n_seeds=1 无显著性字段）。
"""
from __future__ import annotations

import numpy as np
import pytest
from src.bias.boundary import boundary_by_range
from src.bias.propagation import mixing_second_injection, propagation_matrix
from src.bias.significance import bootstrap_mean_test, paired_diff_test


def _true_ctr(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(0.22), sigma=0.5, size=n)
    return np.clip(base, 0.01, 0.9)[None, :]


# ----------------------------------------------------------------------
# bootstrap_mean_test 自身
# ----------------------------------------------------------------------
def test_bootstrap_deterministic():
    """同一输入两次调用逐字段一致（固定 bootstrap 种子）。"""
    x = np.random.default_rng(7).normal(0.3, 0.1, size=30).tolist()
    assert bootstrap_mean_test(x) == bootstrap_mean_test(x)


def test_bootstrap_p_value_range_and_ci_covers_mean():
    x = np.random.default_rng(7).normal(0.3, 0.1, size=30).tolist()
    r = bootstrap_mean_test(x)
    assert 0.0 <= r["p"] <= 1.0
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1]
    assert r["n"] == 30 and r["n_boot"] == 10_000
    assert r["method"] == "bootstrap_percentile"


def test_bootstrap_detects_offset_from_zero():
    """明显非零的样本 → 双侧 p 达下界；真零附近 → p 大。"""
    off = np.random.default_rng(1).normal(5.0, 0.1, size=30).tolist()
    near = np.random.default_rng(1).normal(0.0, 0.01, size=30).tolist()
    assert bootstrap_mean_test(off)["p"] == pytest.approx(1 / 10_001, abs=1e-3)
    assert bootstrap_mean_test(near)["p"] > 0.3


def test_bootstrap_too_few_seeds_returns_empty():
    r = bootstrap_mean_test([0.5])
    assert r["n"] == 1 and r["ci95"] is None and r["p"] is None


def test_paired_diff_test_greater():
    a = np.random.default_rng(3).normal(0.0, 0.05, size=20).tolist()
    b = [v - 0.5 for v in a]  # 同种子配对：a − b 恒 +0.5
    r = paired_diff_test(a, b, alternative="greater")
    assert r["mean"] == pytest.approx(0.5, abs=1e-6)
    assert r["p"] <= 0.01


# ----------------------------------------------------------------------
# 三实验的显著性落盘 + 向后兼容
# ----------------------------------------------------------------------
def test_propagation_significance_fields():
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 3.0], n_requests=200,
                           slots=5, seed=42, n_seeds=5)
    assert "significance" in m
    for tau in ("tau0", "tau3"):
        assert "L2_spearman_vs_zero" in m["significance"][tau]
        assert "L2_rel_err_vs_zero" in m["significance"][tau]
    pv = m["significance"]["tau_first_vs_last_spearman_paired"]
    assert 0.0 <= pv["p"] <= 1.0 and pv["alternative"] == "less"


def test_injection_significance_fields():
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0, w_bases=[0.0, 1.0],
                                  n_requests=200, slots=5, seed=42, n_seeds=5)
    assert "significance" in inj
    assert "w1_spearman_vs_zero" in inj["significance"]
    assert "w0_to_1_paired" in inj["significance"]


def test_multi_seed_one_has_no_significance():
    """n_seeds=1 无 per-seed 样本 → 不写 significance 字段（向后兼容）。"""
    t = _true_ctr()
    kw = dict(n_items=60, n_requests=100, slots=5, seed=42)
    assert "significance" not in propagation_matrix(t, taus=[0.0], **kw)
    assert "significance" not in mixing_second_injection(t, tau=2.0,
                                                         w_bases=[0.0], **kw)


def test_boundary_rows_have_gain_sig():
    b = boundary_by_range(t_pop=0.25, n_records=2000, n_seeds=30)
    for row in b["rows"]:
        assert "gain_sig" in row
        assert 0.0 <= row["gain_sig"]["p"] <= 1.0
    # 无损档（ESS=1，gain 恒 0）应不显著；末档（严重亏）应显著为负
    assert b["rows"][0]["gain_sig"]["p"] > 0.5
    assert b["rows"][-1]["gain_sig"]["p"] <= 0.05
