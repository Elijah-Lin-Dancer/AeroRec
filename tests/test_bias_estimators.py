"""M4 验收：去偏估计量（IPS / SNIPS / DR）。

本测试的核心不是"IPS 比 naive 准"，而是**分清两个估计目标**：
    T_pop = mean(truth)        ← SNIPS/IPS 的目标
    T_exp = Σ truth_i π_i      ← naive 的目标
这两个都合法。把它们混为一谈会得出错误结论（本项目实测踩过，详见
`src/bias/estimators.py` 模块 docstring）。
"""
from __future__ import annotations

import numpy as np
import pytest
from src.bias.estimators import (
    check_propensity_convention,
    cross_fit_model,
    debias_report,
    dr_estimate,
    effective_sample_size,
    naive_estimate,
    snips_estimate,
    unnormalized_ips_estimate,
)


# ----------------------------------------------------------------------
# 两个目标各自的正确性
# ----------------------------------------------------------------------
def test_naive_targets_exposure_weighted_mean():
    """关键澄清：naive 对 **T_exp = Σ truth·π** 几乎无偏（而不是"有偏待纠"）。"""
    rng = np.random.default_rng(1)
    truth = np.array([0.05, 0.50])
    pi = np.array([1 / 21, 20 / 21])
    N = 200_000
    which = rng.choice(2, size=N, p=pi)
    y = (rng.random(N) < truth[which]).astype(float)

    t_exp = float((truth * pi).sum())
    t_pop = float(truth.mean())
    naive = naive_estimate(y)

    # naive ≈ T_exp，明显不等于 T_pop
    assert abs(naive - t_exp) < 0.01, (naive, t_exp)
    assert abs(naive - t_pop) > 0.1


def test_snips_targets_population_mean():
    """SNIPS 的目标是 T_pop = mean(truth)，且应显著优于 naive 在该目标上的表现。"""
    rng = np.random.default_rng(1)
    truth = np.array([0.05, 0.50])
    pi = np.array([1 / 21, 20 / 21])
    N = 200_000
    which = rng.choice(2, size=N, p=pi)
    y = (rng.random(N) < truth[which]).astype(float)
    prop = pi[which]

    t_pop = float(truth.mean())
    naive = naive_estimate(y)
    snips = snips_estimate(y, prop)

    assert abs(snips - t_pop) < 0.02, (snips, t_pop)
    assert abs(snips - t_pop) < abs(naive - t_pop) / 3


def test_uniform_propensity_snips_equals_naive():
    """π 均匀时两个目标重合，SNIPS 应精确等于朴素均值。"""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=20_000).astype(float)
    pi = np.full(len(y), 1.0 / len(y))
    assert snips_estimate(y, pi) == pytest.approx(y.mean(), abs=1e-12)


def test_unnormalized_ips_scale_trap():
    """记录一个常见陷阱：π 归一化到 Σπ=1 时，未归一化 IPS 量纲差 N 倍。"""
    rng = np.random.default_rng(3)
    N = 5000
    y = rng.binomial(1, 0.25, size=N).astype(float)
    pi = np.full(N, 1.0 / N)
    raw = unnormalized_ips_estimate(y, pi)
    good = snips_estimate(y, pi)
    assert raw > good * 100          # 明显量纲错
    assert raw == pytest.approx(good * N, rel=0.05)


def test_snips_lower_variance_than_unnormalized():
    """强偏斜 π 下，SNIPS 的相对离散度应低于未归一化 IPS。

    需要足够的 π 动态范围（这里 1e-6 ~ 1e-2，4 个数量级）才能显出差别；
    范围太窄时两者几乎等价（本项目实测：仅 100× 范围时相对 std 均为 0.104）。
    """
    rng = np.random.default_rng(2)
    N = 3000
    pi = rng.lognormal(mean=-6.0, sigma=1.6, size=N)     # 约 4 个数量级
    pi /= pi.sum()
    sn, un = [], []
    for s in range(80):
        r2 = np.random.default_rng(100 + s)
        yy = r2.binomial(1, 0.2, size=N).astype(float)
        sn.append(snips_estimate(yy, pi))
        un.append(unnormalized_ips_estimate(yy, pi))
    rel_sn = np.std(sn) / abs(np.mean(sn))
    rel_un = np.std(un) / abs(np.mean(un))
    assert rel_sn < rel_un, (rel_sn, rel_un)


def test_effective_sample_size_bounds():
    """ESS 的判据：ESS 由**最小的 π**（最大的 1/π 权重）主导。

    ⚠️ 直觉校正（本项目实测纠正过）：不是"π 越集中越小"。
    一个物品 π 很大反而权重最小；真正拖垮 ESS 的是**存在 π 极小的物品**
    （几乎不被曝光 → 1/π 爆炸 → 被放大成噪声源）。
    """
    # 均匀 π → ESS ≈ N
    assert effective_sample_size(np.full(500, 1 / 500)) == pytest.approx(500, rel=1e-6)
    # 存在一个 π 极小的物品 → ESS 崩塌到 ~1
    pi_low = np.full(500, (1 - 1e-8) / 499)
    pi_low[0] = 1e-8
    assert effective_sample_size(pi_low) < 5
    # 反向对照：一个 π 很大、其余相等且不小 → ESS 仍接近 N（不低）
    pi_high = np.full(500, 0.01 / 499)
    pi_high[0] = 0.99
    assert effective_sample_size(pi_high) > 400


def test_check_propensity_convention():
    """π 约定检测：Σπ=1 判为 distribution，量级 O(1) 判为 inclusion。"""
    assert check_propensity_convention(np.full(100, 1 / 100))["convention"] == "distribution"
    incl = np.full(100, 0.3)          # 每个单元 30% 概率被记录
    assert check_propensity_convention(incl)["convention"] == "inclusion"
    # 误用警告必须出现
    w = check_propensity_convention(np.full(100, 1 / 100))["warning"]
    assert "自归一化" in w


def test_ess_drops_with_exposure_skew():
    """曝光策略越偏（τ↑），π 的动态范围越大 → ESS 下降。

    这是去偏**有效边界**的直接证据（M5 会把这条曲线扫出来）。
    """
    from src.bias.policy import popularity_propensity
    h = np.linspace(0.05, 0.9, 60)
    ess = [effective_sample_size(popularity_propensity(h, tau=t))
           for t in (0.0, 1.0, 2.0, 3.0)]
    assert all(b < a for a, b in zip(ess, ess[1:])), ess
    assert ess[0] > 50          # 均匀曝光下 ESS 接近样本量
    assert ess[-1] < ess[0] / 10


# ----------------------------------------------------------------------
# DR 与交叉拟合（**约定①：π 为包含概率**）
# ----------------------------------------------------------------------
def test_dr_with_perfect_model_equals_true_value():
    """m̂ ≡ truth 时修正项期望为 0，DR 应回到真值（π 用包含概率约定）。"""
    rng = np.random.default_rng(3)
    N = 20_000
    true_p = 0.3
    m = np.full(N, true_p)
    y = rng.binomial(1, true_p, size=N).astype(float)
    pi = rng.uniform(0.2, 0.8, size=N)          # 包含概率，O(1)
    dr = dr_estimate(y, pi, m)
    assert dr == pytest.approx(true_p, abs=0.02), dr


def test_dr_warns_when_given_distribution_propensity():
    """记录失败模式：把 Σπ=1 的 π 塞进 DR 会得到无意义结果。"""
    rng = np.random.default_rng(3)
    N = 5000
    m = np.full(N, 0.3)
    y = rng.binomial(1, 0.3, size=N).astype(float)
    pi_dist = rng.uniform(0.5, 1.5, size=N)
    pi_dist /= pi_dist.sum()                    # 约定②
    bad = dr_estimate(y, pi_dist, m)
    good = dr_estimate(y, pi_dist * N, m)       # 转成包含概率
    assert abs(bad - 0.3) > 1.0, bad            # 明显错误
    assert good == pytest.approx(0.3, abs=0.05)  # 换算后正确


def test_cross_fit_is_out_of_sample():
    rng = np.random.default_rng(4)
    N = 600
    x = rng.normal(size=(N, 3))
    y = (x[:, 0] * 0.3 + rng.normal(0, 0.5, size=N) > 0).astype(float)
    pi = np.full(N, 0.5)
    m = cross_fit_model(x, y, pi, n_folds=5)
    assert m.shape == (N,)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert float(np.corrcoef(m, y)[0, 1]) < 0.95     # 若样本内拟合会接近 1


# ----------------------------------------------------------------------
# 端到端报告
# ----------------------------------------------------------------------
def test_debias_report_separates_both_targets():
    rng = np.random.default_rng(5)
    N = 100_000
    truth = np.array([0.08, 0.12, 0.45])
    pi = np.array([0.5, 0.3, 0.2])
    which = rng.choice(3, size=N, p=pi)
    y = (rng.random(N) < truth[which]).astype(float)

    t_pop = float(truth.mean())
    t_exp = float((truth * pi).sum())
    rep = debias_report(y, pi[which], true_pop_mean=t_pop, true_exp_mean=t_exp)

    # naive 贴 T_exp，snips 贴 T_pop —— 各自对自己的目标都准
    assert abs(rep["naive"] - t_exp) < 0.01, rep
    assert abs(rep["snips"] - t_pop) < 0.02, rep
    # 相对 T_pop 这个目标，SNIPS 才是"对"的那个
    assert rep["snips_err_vs_pop"] < rep["naive_err_vs_pop"]


def test_debias_report_flags_true_value_only_known_in_synthetic():
    rep = debias_report(np.array([1.0, 0.0, 1.0]), np.array([0.3, 0.3, 0.4]),
                        true_pop_mean=0.5)
    for k in ("true_pop_mean", "naive", "ips", "snips", "naive_gain_vs_pop"):
        assert k in rep
