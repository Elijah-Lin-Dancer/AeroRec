"""propensity 估计误差与用户侧偏差实验（改进计划 #4/#5/#6）。

断言实现语义与方向性：
  - π̂ 实验：τ=0 时估计无代价、预警无误判（自洽）；n_seeds=1 向后兼容；
  - 用户实验：β=0 三估计器重合（自洽）；β 大时「物品边际去偏」残余偏差
    显著大于「user-item 正确去偏」（核心论点）；记录数 = 会话 × 槽位。
"""
from __future__ import annotations

import numpy as np
import pytest
from src.bias.pi_estimation import propensity_estimation_experiment
from src.bias.userbias import user_bias_experiment


def _true_ctr(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(0.22), sigma=0.5, size=n)
    return np.clip(base, 0.01, 0.9)[None, :]


# ----------------------------------------------------------------------
# #4 propensity 估计误差
# ----------------------------------------------------------------------
def test_pi_estimation_deterministic():
    t = _true_ctr()
    kw = dict(taus=[0.0, 2.0], n_requests=100, n_seeds=3, seed=42)
    a = propensity_estimation_experiment(t, **kw)
    b = propensity_estimation_experiment(t, **kw)
    assert a["rows"] == b["rows"]


def test_pi_estimation_tau0_no_cost_no_false_alarm():
    """τ=0：曝光均匀 → π̂ 好估，去偏代价≈0、ESS 预警无误判（自洽）。"""
    ec = propensity_estimation_experiment(_true_ctr(), taus=[0.0],
                                          n_requests=400, n_seeds=10)
    for r in ec["rows"]:
        assert abs(r["delta_err"]) < 0.005, r
        assert r["warning_hat"] == 0.0 and r["warning_true"] == 0.0


def test_pi_estimation_backward_compat_and_sig():
    t = _true_ctr()
    full = propensity_estimation_experiment(t, taus=[0.0, 3.0], n_requests=100,
                                            n_seeds=3)
    assert full["n_seeds"] == 3 and "significance" in full
    for r in full["rows"]:
        assert "delta_err_std" in r
    single = propensity_estimation_experiment(t, taus=[0.0, 3.0], n_requests=100,
                                              n_seeds=1)
    assert "significance" not in single
    for r in single["rows"]:
        assert "delta_err_std" not in r


def test_pi_estimation_records_shape():
    ec = propensity_estimation_experiment(_true_ctr(), taus=[1.0],
                                          n_requests=123, n_seeds=2)
    assert ec["rows"][0]["n_records"] == 123 * 5


# ----------------------------------------------------------------------
# #5/#6 用户侧偏差
# ----------------------------------------------------------------------
def test_userbias_beta0_self_consistent():
    """β=0：无自选偏差 → T_exp_user=T_pop、naive 贴 T_pop、残余偏差=0（自洽）。"""
    ub = user_bias_experiment(_true_ctr(), betas=(0.0,), n_sessions=300,
                              n_seeds=5)
    r = ub["rows"][0]
    assert r["T_exp_user"] == pytest.approx(r["T_pop"], abs=0.02)
    assert abs(r["naive"] - r["T_exp_user"]) < 0.02
    assert abs(r["agg_residual"]) < 0.005


def test_userbias_aggregate_granularity_leaves_residual():
    """β 大：物品边际 propensity 去偏的残余偏差显著大于正确粒度（核心论点）。"""
    ub = user_bias_experiment(_true_ctr(), betas=(4.0,), n_sessions=500,
                              n_seeds=10)
    r = ub["rows"][0]
    assert r["agg_residual"] > 0.05, r
    # naive 贴其收敛目标 T_exp_user（estimand 自检），但 T_exp_user 已远离 T_pop
    assert abs(r["naive"] - r["T_exp_user"]) < 0.05, r
    assert r["T_exp_user"] - r["T_pop"] > 0.1, r


def test_userbias_backward_compat_and_records():
    t = _true_ctr()
    single = user_bias_experiment(t, betas=(0.0, 2.0), n_sessions=100, n_seeds=1)
    assert "significance" not in single
    for r in single["rows"]:
        assert "agg_residual_std" not in r
        assert r["n_records"] == 100 * 5
    full = user_bias_experiment(t, betas=(0.0, 2.0), n_sessions=100, n_seeds=3)
    assert "significance" in full and full["n_seeds"] == 3


def test_userbias_deterministic():
    t = _true_ctr()
    kw = dict(betas=(0.0, 2.0), n_sessions=100, n_seeds=2, seed=42)
    assert (user_bias_experiment(t, **kw)["rows"]
            == user_bias_experiment(t, **kw)["rows"])
