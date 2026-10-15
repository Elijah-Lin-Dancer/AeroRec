"""估计目标对照实验：naive→T_exp、SNIPS→T_pop 的分离与自洽。

断言的是**估计量收敛目标**这一实现语义（§4 结论的产物化）：
  - naive 池化 CTR 贴 `T_exp = Σ truth·π`（对 T_exp 的误差 ≈ 抽样噪声量级）；
  - SNIPS 贴 `T_pop = mean(truth)`；
  - τ=0 时 `T_exp`=`T_pop`，两估计量重合（自检）；
  - n_seeds=1 无 std/显著性字段（向后兼容）。
"""
from __future__ import annotations

import numpy as np
import pytest
from src.bias.estimand import estimand_check


def _true_ctr(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(0.22), sigma=0.5, size=n)
    return np.clip(base, 0.01, 0.9)[None, :]


def test_naive_tracks_texp_snips_tracks_tpop():
    ec = estimand_check(_true_ctr(), taus=[1.0, 3.0], n_requests=400, slots=5,
                        seed=42, n_seeds=10)
    for r in ec["rows"]:
        # naive 贴 T_exp：误差在抽样噪声量级（<< T_exp−T_pop 的漂移）
        assert r["naive_err_vs_texp"] < 0.03, r
        # SNIPS 贴 T_pop=0.2525：哪怕 τ=3 时 T_exp 漂到 0.40 也不跟
        assert abs(r["snips"] - r["T_pop"]) < 0.05, r
        # 估计目标分离确实发生：T_exp 随 τ 漂离 T_pop
    assert ec["rows"][-1]["T_exp"] - ec["rows"][-1]["T_pop"] > 0.10


def test_tau_zero_estimators_coincide():
    """τ=0 时 T_exp=T_pop，naive 与 SNIPS 应重合（实现自检）。"""
    ec = estimand_check(_true_ctr(), taus=[0.0], n_requests=300, slots=5,
                        seed=42, n_seeds=5)
    r = ec["rows"][0]
    assert r["T_exp"] == pytest.approx(r["T_pop"], abs=1e-6)
    assert r["naive"] == pytest.approx(r["snips"], abs=1e-9)
    assert r["ess_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_multi_seed_fields_and_backward_compat():
    t = _true_ctr()
    full = estimand_check(t, taus=[0.0, 2.0], n_requests=200, slots=5,
                          seed=42, n_seeds=3)
    assert full["n_seeds"] == 3 and "significance" in full
    for r in full["rows"]:
        assert "naive_err_vs_texp_std" in r
    single = estimand_check(t, taus=[0.0, 2.0], n_requests=200, slots=5,
                            seed=42, n_seeds=1)
    assert "significance" not in single
    for r in single["rows"]:
        assert "naive_err_vs_texp_std" not in r


def test_records_shape():
    """记录粒度 = 请求数 × 坑位（与真实日志同构）。"""
    ec = estimand_check(_true_ctr(), taus=[1.0], n_requests=123, slots=5,
                        seed=1, n_seeds=2)
    assert ec["rows"][0]["n_records"] == 123 * 5
