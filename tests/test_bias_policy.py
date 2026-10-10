"""M1 验收：曝光策略的闭式 propensity 必须与经验曝光频率吻合。

这是整条去偏链路的地基——如果 π_i 算错了，后面 IPS/DR 的所有结论都不可信。
所以这里测的不是「代码能跑」，而是**闭式 π 与蒙特卡洛经验频率一致**（误差量级 ~1/sqrt(n)）。

同时锁定两条契约：
  1. τ=0 必须**严格退化为均匀**（这是「无曝光偏差」的对照组定义）；
  2. 同一 seed 必须**逐位可复现**（项目基线要求，见 README W8）。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bias.policy import (ExposurePolicy, _normalize, inverse_rank_weights,
                             popularity_propensity, position_propensity,
                             sample_exposure, uniform_propensity)
from src.bias.injectors import (build_biased_log, inject_exposure_popularity,
                               inject_exposure_position, inject_uniform)


# ----------------------------------------------------------------------
# 闭式 π 的基本性质
# ----------------------------------------------------------------------
def test_uniform_propensity_sums_to_one():
    pi = uniform_propensity(7)
    assert pi.shape == (7,)
    assert pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi, 1.0 / 7)


def test_tau_zero_degenerates_to_uniform():
    """τ=0 必须**严格**等于均匀——这是「无偏对照组」的定义。"""
    h = np.array([1.0, 5.0, 0.3, 20.0, 2.0])
    pi = popularity_propensity(h, tau=0.0)
    assert np.allclose(pi, uniform_propensity(len(h)))


def test_popularity_monotone_and_closed_form():
    h = np.array([1.0, 2.0, 4.0])
    for tau in (0.5, 1.0, 2.0):
        pi = popularity_propensity(h, tau=tau)
        assert pi.sum() == pytest.approx(1.0)
        # 闭式：π_i = h_i^τ / Σ h^τ
        assert np.allclose(pi, h ** tau / (h ** tau).sum())
        # 单调：热度越高，曝光概率越大
        assert pi[0] < pi[1] < pi[2]


def test_position_propensity_uses_rank_weights():
    w = inverse_rank_weights(5)
    assert np.allclose(w, [1.0, 0.5, 1 / 3, 1 / 4, 1 / 5])
    pi = position_propensity(w, eta=1.0)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(np.diff(pi) < 0)          # 位次越靠后概率越低
    # η=0 退化为均匀
    assert np.allclose(position_propensity(w, eta=0.0), uniform_propensity(5))


def test_normalize_handles_all_zero():
    """全零权重不应产生 NaN（可能出现在「候选全不可曝光」的边界）。"""
    out = _normalize(np.zeros(4))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 核心验收：经验曝光频率 vs 闭式 π
# ----------------------------------------------------------------------
def test_empirical_exposure_matches_closed_form_popularity():
    """τ=1 的热度策略下，蒙特卡洛经验频率与闭式 π 最大偏差应 < 1e-2。"""
    rng = np.random.default_rng(0)
    h = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    tau = 1.0
    pi = popularity_propensity(h, tau=tau)

    n_draws, slots = 200_000, 3
    counts = np.zeros(len(h))
    for _ in range(n_draws):
        idx = sample_exposure(pi, slots, rng, replace=True)
        counts += np.bincount(idx, minlength=len(h))
    emp = counts / counts.sum()

    max_dev = float(np.abs(emp - pi).max())
    assert max_dev < 1e-2, f"经验频率与闭式 π 偏差过大：{max_dev:.4f}"


def test_empirical_exposure_matches_closed_form_uniform():
    rng = np.random.default_rng(1)
    n = 5
    pi = uniform_propensity(n)
    counts = np.zeros(n)
    for _ in range(100_000):
        counts += np.bincount(sample_exposure(pi, 2, rng, replace=True), minlength=n)
    emp = counts / counts.sum()
    assert float(np.abs(emp - pi).max()) < 5e-3


# ----------------------------------------------------------------------
# 可复现性：同 seed 必须逐位一致
# ----------------------------------------------------------------------
def test_same_seed_reproducible():
    pi = popularity_propensity(np.arange(1, 11, dtype=float), tau=1.5)
    a = sample_exposure(pi, 4, np.random.default_rng(42))
    b = sample_exposure(pi, 4, np.random.default_rng(42))
    assert np.array_equal(a, b)

    # 注入器层面同样要可复现
    pool = np.arange(10)
    h = np.arange(1, 11, dtype=float)
    i1, p1 = inject_exposure_popularity(pool, h, 4, tau=1.0, rng=np.random.default_rng(7))
    i2, p2 = inject_exposure_popularity(pool, h, 4, tau=1.0, rng=np.random.default_rng(7))
    assert np.array_equal(i1, i2) and np.allclose(p1, p2)


# ----------------------------------------------------------------------
# 注入器契约
# ----------------------------------------------------------------------
def test_inject_no_replace_returns_distinct():
    rng = np.random.default_rng(3)
    idx, pi = inject_uniform(5, rng, replace=False)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5          # 无放回 → 不重复


def test_inject_position_biases_toward_front():
    """位置偏差：靠前坑位的经验曝光率必须显著高于靠后。"""
    rng = np.random.default_rng(5)
    scores = np.arange(10, 0, -1, dtype=float)   # 已按分数降序
    counts = np.zeros(10)
    reps = 20_000
    for _ in range(reps):
        idx, _ = inject_exposure_position(scores, 3, eta=1.0, rng=rng, replace=True)
        counts += np.bincount(idx, minlength=10)
    emp = counts / counts.sum()
    assert emp[0] > emp[9] * 3                  # 首位曝光率远高于末位


def test_build_biased_log_shapes_and_propensity_recorded():
    rng = np.random.default_rng(11)
    R, C, K = 50, 8, 3
    true_ctr = np.full((R, C), 0.2)
    pi = popularity_propensity(np.arange(1, C + 1, dtype=float), tau=1.0)
    log = build_biased_log(true_ctr, pi, n_requests=R, slots=K, rng=rng)

    assert log["clicked"].shape == (R * K,)
    assert log["propensity"].shape == (R * K,)
    assert log["true_ctr"].shape == (R * K,)
    assert log["request_idx"].min() == 0 and log["request_idx"].max() == R - 1
    assert log["candidate_idx"].max() < C
    # propensity 必须是闭式 π 的取值之一，且严格为正（IPS 要除以它）
    assert np.all(np.isin(np.round(log["propensity"], 12), np.round(pi, 12)))
    assert np.all(log["propensity"] > 0)


def test_policy_class_factory_consistent_with_functions():
    h = np.array([1.0, 2.0, 4.0])
    pol = ExposurePolicy("popularity", strength=1.0)
    assert np.allclose(pol.propensity(popularity=h), popularity_propensity(h, tau=1.0))

    pol_p = ExposurePolicy("position", strength=1.0)
    s = np.array([3.0, 2.0, 1.0])                 # 已降序
    assert np.allclose(pol_p.propensity(scores=s), position_propensity(inverse_rank_weights(3), eta=1.0))

    pol_u = ExposurePolicy("uniform")
    assert np.allclose(pol_u.propensity(4), uniform_propensity(4))


def test_policy_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ExposurePolicy("magic")
