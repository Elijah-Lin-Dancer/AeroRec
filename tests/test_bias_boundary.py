"""M5 验收：去偏的有效边界。

核心断言：**让去偏亏本的不是偏差幅度（τ），而是 π 的动态范围（→ ESS/N）**。
这条结论修正了最初的假设，故测试要同时锁定"正确结论"与"被否定的假设"。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bias.boundary import (boundary_by_range, boundary_sweep,
                              ess_warning_from_log, find_boundary)


def _truth(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(np.log(0.22), 0.5, n), 0.01, 0.9)


# ----------------------------------------------------------------------
# 核心实验：π 动态范围决定边界
# ----------------------------------------------------------------------
def test_debias_loses_when_pi_range_is_wide():
    """π 极值比极大 → ESS/N 崩 → 去偏收益转负。"""
    b = boundary_by_range(t_pop=0.25, n_records=20000, n_seeds=40)
    rows = {r["pi_ratio"]: r for r in b["rows"]}
    # 均匀：两目标重合，gain = 0
    assert abs(rows[1.0]["gain"]) < 1e-6
    # 宽范围：明显亏本
    wide = [r for r in b["rows"] if r["pi_ratio"] >= 20000][0]
    assert wide["gain"] < -0.01, wide
    assert wide["ess_ratio"] < 0.05, wide


def test_boundary_is_monotone_in_ess_ratio():
    """gain 随 ESS/N 下降而单调变差（边界存在的直接证据）。"""
    b = boundary_by_range(t_pop=0.25, n_records=20000, n_seeds=40)
    rows = sorted(b["rows"], key=lambda r: -r["ess_ratio"])
    gains = [r["gain"] for r in rows]
    assert all(b_ <= a + 1e-9 for a, b_ in zip(gains, gains[1:])), gains


def test_cross_ess_ratio_reported_and_reasonable():
    """必须报出一个落在扫描范围内的 ESS/N 阈值，且不是退化的 1.0。"""
    b = boundary_by_range(t_pop=0.25, n_records=20000, n_seeds=40)
    c = b["cross_ess_ratio"]
    assert c is not None
    assert 0.0 < c < 0.95, c          # 不能是 ESS/N≈1 的假阈值
    assert "转负" in b["conclusion"]


# ----------------------------------------------------------------------
# 被否定的假设：τ 增大并不使去偏亏本
# ----------------------------------------------------------------------
def test_larger_tau_does_not_make_debias_lose():
    """⚠️ 修正最初假设：τ 越大，去偏反而赚得越多（不是越容易亏）。

    原因：τ 增大只是拉大 T_exp 与 T_pop 的差距，naive 的"误差"随之变大，
    但那是 estimand 之差，IPS 纠正的正是这个差。
    """
    t = _truth(60)
    res = boundary_sweep(t, taus=[0.0, 1.0, 2.0, 3.0],
                         n_records_list=[5000], n_seeds=15, seed=42)
    by_tau = {r["tau"]: r for r in res["grid"]}
    # τ=0：两目标重合，gain≈0
    assert abs(by_tau[0.0]["gain"]) < 0.01
    # τ>=1：去偏均为正收益，且 τ=3 的收益不小于 τ=1
    assert by_tau[1.0]["gain"] > 0.01, by_tau[1.0]
    assert by_tau[3.0]["gain"] > by_tau[1.0]["gain"], (by_tau[3.0], by_tau[1.0])


def test_ess_drops_with_tau_but_gain_stays_positive():
    """τ 确实会拉低 ESS/N；但在本设定范围内仍未让收益转负。"""
    t = _truth(60)
    res = boundary_sweep(t, taus=[0.0, 1.0, 2.0, 3.0],
                         n_records_list=[20000], n_seeds=10, seed=1)
    by_tau = {r["tau"]: r for r in res["grid"]}
    assert by_tau[3.0]["ess_ratio"] < by_tau[0.0]["ess_ratio"]
    assert by_tau[3.0]["gain"] > 0


# ----------------------------------------------------------------------
# 多种子平均的必要性
# ----------------------------------------------------------------------
def test_multi_seed_averaging_recorded():
    """结果必须记录 n_seeds —— 单 seed 会被抽样噪声误导。"""
    t = _truth(40)
    res = boundary_sweep(t, taus=[1.0], n_records_list=[2000], n_seeds=20, seed=5)
    assert res["n_seeds"] == 20
    assert res["grid"][0]["n_seeds"] == 20
    assert "gain_std" in res["grid"][0]


# ----------------------------------------------------------------------
# 落地接口：仅凭 π 给建议
# ----------------------------------------------------------------------
def test_ess_warning_recommends_ips_for_healthy_pi():
    pi = np.full(1000, 1 / 1000)
    w = ess_warning_from_log(pi)
    assert w["ess_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert "可用" in w["recommendation"]


def test_ess_warning_rejects_extreme_pi():
    pi = np.full(1000, (1 - 1e-9) / 999)
    pi[0] = 1e-9
    w = ess_warning_from_log(pi)
    assert w["ess_ratio"] < 0.05
    assert "不建议" in w["recommendation"]


def test_ess_warning_caution_zone():
    """中间地带应给出"谨慎"而非二元结论。"""
    rng = np.random.default_rng(0)
    pi = rng.uniform(0.05, 2.0, 5000)
    pi /= pi.sum()
    w = ess_warning_from_log(pi)
    assert w["ess_ratio"] < 0.5
    assert "谨慎" in w["recommendation"] or "不建议" in w["recommendation"]


def test_find_boundary_flags_profitable_region():
    t = _truth(60)
    res = boundary_sweep(t, taus=[1.0], n_records_list=[500, 5000, 50000],
                         n_seeds=10, seed=3)
    fb = find_boundary(res["grid"])
    assert len(fb["per_tau"]) == 1
    assert "ess_ratio_range" in fb["per_tau"][0]
