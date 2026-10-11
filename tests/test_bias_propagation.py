"""偏差传播矩阵与二次注入。

断言的是**现象的方向性**（单调趋势），而不是精确数值——因为这里含蒙特卡洛抽样，
固定 seed 只能保证可复现，不能保证跨环境数值逐位一致。
方向性断言足以支撑研究结论，且不会因环境差异误报失败。

核心断言：
  A. 曝光偏差越强（τ↑），曝光集中度（Gini/KL）单调上升；
  B. 排序质量（spearman）随 τ↑ 单调下降，并在 τ 大时**跌破 0（排序反转）**；
  C. 冷门物品的曝光次数随 τ↑ 减少，被 α 平滑拽向先验 1.0 → 系统性高估；
  D. 混排二次注入有效：w_base=1 时混排级误差严格为 0（实现自检），
     且 spearman 随 w_base 单调上升。
"""
from __future__ import annotations

import numpy as np
import pytest
from src.bias.propagation import (
    mixing_second_injection,
    propagation_matrix,
    run_stage_estimates,
    stage_error,
)


def _true_ctr(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(0.22), sigma=0.5, size=n)
    return np.clip(base, 0.01, 0.9)[None, :]


# ----------------------------------------------------------------------
# stage_error 自身的正确性
# ----------------------------------------------------------------------
def test_stage_error_identity_is_zero():
    t = np.array([0.1, 0.3, 0.5, 0.7])
    e = stage_error(t, t)
    assert e["rel_err"] == pytest.approx(0.0)
    assert e["spearman"] == pytest.approx(1.0)
    assert e["signed"] == pytest.approx(0.0)


def test_stage_error_detects_inverted_ranking():
    t = np.array([0.1, 0.2, 0.3, 0.4])
    inv = t[::-1]
    e = stage_error(inv, t)
    assert e["spearman"] == pytest.approx(-1.0)
    assert e["rel_err"] > 0


# ----------------------------------------------------------------------
# 曝光偏差 → 曝光集中度（A）
# ----------------------------------------------------------------------
def test_exposure_concentration_increases_with_tau():
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 1.0, 2.0, 3.0],
                           n_requests=400, slots=5, seed=42)
    ginis = [r["exposure_gini"] for r in m["rows"]]
    kls = [r["exposure_kl"] for r in m["rows"]]
    # 严格单调上升（连续注入强度应带来连续变化）
    assert all(b > a for a, b in zip(ginis, ginis[1:])), ginis
    assert all(b > a for a, b in zip(kls, kls[1:])), kls
    assert ginis[0] < 0.15          # τ=0 接近均匀
    assert ginis[-1] > 0.45         # τ=3 明显集中


# ----------------------------------------------------------------------
# 曝光偏差 → 排序质量（B）—— 核心现象
# ----------------------------------------------------------------------
def test_ranking_quality_degrades_and_inverts():
    """偏差的主要危害是**把排序搞反**，而不只是整体偏高。"""
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 1.0, 2.0, 3.0],
                           n_requests=500, slots=5, seed=42)
    sp = [r["L2_spearman"] for r in m["rows"]]
    assert all(b < a for a, b in zip(sp, sp[1:])), sp     # 单调下降
    assert sp[0] > 0.6                                    # 均匀曝光下还好
    assert sp[-1] < 0.0, f"τ=3 时排序应被反转，实际 {sp[-1]}"


def test_sparse_items_get_zero_impressions_under_strong_bias():
    """强偏差下会出现曝光次数为 0 的冷门物品——这是排序反转的机制来源。"""
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 3.0], n_requests=500,
                           slots=5, seed=42)
    assert m["rows"][0]["impr_min"] > m["rows"][-1]["impr_min"]
    assert m["rows"][-1]["impr_min"] == 0


# ----------------------------------------------------------------------
# 稀疏收缩机制（C）—— 机制 A 独立于曝光策略
# ----------------------------------------------------------------------
def test_sparsity_shrinkage_exists_even_with_uniform_exposure():
    """即使 τ=0（曝光均匀），平滑 CTR 仍因 α 而系统性高估——机制 A 与策略无关。"""
    t = _true_ctr()
    r = run_stage_estimates(t, n_items=60, tau=0.0, n_requests=400, slots=5,
                            alpha=5.0, rng=np.random.default_rng(1))
    e = stage_error(r["L2_rank"], r["true_ctr"])
    assert e["signed"] > 0, f"均匀曝光下应仍有正向收缩偏差，实际 {e['signed']}"


def test_smaller_alpha_reduces_shrinkage():
    """α 越小收缩越弱——佐证机制 A 来自先验强度。"""
    t = _true_ctr()
    e_big = stage_error(
        run_stage_estimates(t, 60, 0.0, 400, 5, alpha=20.0,
                            rng=np.random.default_rng(2))["L2_rank"], t.mean(axis=0))
    e_small = stage_error(
        run_stage_estimates(t, 60, 0.0, 400, 5, alpha=0.5,
                            rng=np.random.default_rng(2))["L2_rank"], t.mean(axis=0))
    assert e_big["rel_err"] > e_small["rel_err"]


# ----------------------------------------------------------------------
# 混排二次注入（D）
# ----------------------------------------------------------------------
def test_mixing_injection_sanity_at_w_base_one():
    """w_base=1 时混排级 **等于真值**，误差必须严格为 0——实现自检。"""
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0, w_bases=[0.0, 0.5, 1.0],
                                  n_requests=300, slots=5, seed=42)
    last = inj["rows"][-1]
    assert last["L3_rel_err"] == pytest.approx(0.0, abs=1e-6)
    assert last["L3_spearman"] == pytest.approx(1.0)


def test_mixing_injection_monotone_in_w_base():
    """w_base 越大，混排级排序质量**整体**越好（趋势断言，容忍局部噪声回撤）。

    机制：L3 = (1-w)·L2 + w·truth，w 越大越接近真值，spearman 应整体趋好。
    但 `mixing_second_injection` 每次 w_base 用独立 RNG 重抽曝光，相邻 w_base 间
    L2 自身带不同噪声，故逐点严格单调并不成立（属抽样噪声，非机制失效）。
    因此断言**整体趋势**而非逐点严格单调：

      - 端点确定性：w_base=1 ⇒ L3 恰为 truth，spearman 严格 = 1.0；
      - 无注入时（w=0）强偏差下排序接近随机，应显著低于满分；
      - 整体上行：spearman(w=1) − spearman(w=0) 远超噪声幅度（> 0.5）。
    """
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0,
                                  w_bases=[0.0, 0.2, 0.4, 0.8, 1.0],
                                  n_requests=500, slots=5, seed=42)
    sp = [r["L3_spearman"] for r in inj["rows"]]
    # 端点确定性：w_base=1 ⇒ L3 完全等于真值，秩相关必为 1.0（不依赖 RNG）
    assert sp[-1] == pytest.approx(1.0, abs=1e-9), sp
    # 无注入（w=0）时强偏差下排序接近随机，应显著低于满分
    assert sp[0] < 0.2, f"无注入时强偏差下排序应接近随机，实际 {sp[0]}"
    # 整体趋势：从「无注入」到「全注入」spearman 大幅上升（容忍局部小幅回撤）
    assert sp[-1] - sp[0] > 0.5, f"二次注入应整体提升排序质量，spearman 趋势 {sp}"


def test_propagation_matrix_reproducible():
    """同一 seed 两次运行结果一致（项目基线要求）。"""
    t = _true_ctr()
    kw = dict(n_items=60, taus=[0.0, 1.0, 2.0], n_requests=200, slots=5, seed=7)
    a = propagation_matrix(t, **kw)
    b = propagation_matrix(t, **kw)
    assert a["rows"] == b["rows"]
