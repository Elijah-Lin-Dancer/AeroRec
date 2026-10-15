"""偏差传播矩阵与二次注入。

断言的是**现象的方向性**（单调趋势），而不是精确数值——因为这里含蒙特卡洛抽样，
固定 seed 只能保证可复现，不能保证跨环境数值逐位一致。
方向性断言足以支撑研究结论，且不会因环境差异误报失败。

RNG 策略（与 `src/bias/propagation.py` 一致）：
  - 传播矩阵：CRN（共同随机数）——每个 τ 以 default_rng(seed) 重置，
    τ 间差异只来自曝光策略而非抽样序列；
  - 二次注入：单次模拟 + 解析扫描——曝光只模拟一次，L3 逐 w_base 解析重算，
    注入曲线不含抽样噪声，可做逐点严格单调断言。

核心断言：
  A. 曝光偏差越强（τ↑），曝光集中度（Gini/KL）单调上升；
  B. 排序质量（spearman）随 τ↑ 单调下降，并在 τ 大时**跌破 0（排序反转）**；
  C. 冷门物品的曝光次数随 τ↑ 减少，被 α 平滑拽向先验 1.0 → 系统性高估；
  D. 混排二次注入有效：w_base=1 时混排级误差严格为 0（实现自检），
     且 spearman 随 w_base **逐点严格**单调上升、rel_err 逐点严格下降。
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


def test_sparse_items_starved_under_strong_bias():
    """强偏差下冷门物品的曝光被饿到趋近 0——这是排序反转的机制来源。

    断言「趋近 0」而非「恰好为 0」：是否抽到 0 次曝光取决于具体随机流的
    实现，机制层面的不变量是「冷门物品几乎无曝光」。
    """
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 3.0], n_requests=500,
                           slots=5, seed=42)
    assert m["rows"][0]["impr_min"] > m["rows"][-1]["impr_min"]
    # τ=3：最冷门物品几乎无曝光（个位数）；τ=0 均匀曝光下没有物品被饿死
    assert m["rows"][-1]["impr_min"] <= 2, m["rows"][-1]["impr_min"]
    assert m["rows"][0]["impr_min"] >= 20, m["rows"][0]["impr_min"]


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
    """w_base 越大，混排级排序质量**逐点严格**单调上升。

    机制：曝光与点击只依赖 τ，与 w_base 无关；L3 = (1−w)·L2 + w·truth 是
    解析凸组合。`mixing_second_injection` 只模拟一次曝光、逐 w_base 解析
    重算 L3，故注入曲线不含抽样噪声（L2 列各档完全一致）。

    （历史注记：早期实现每个 w_base 用独立 RNG 重抽曝光，采样噪声曾制造
    w=0.4 处的伪回撤与 w=1.0 处的伪回落，彼时本测试只能退化为趋势断言；
    RNG 统一为「单次模拟 + 解析扫描」后伪影不复存在，断言随之收紧。）

    断言：
      - 端点确定性：w_base=1 ⇒ L3 恰为 truth，spearman 严格 = 1.0；
      - spearman 逐点严格上升、rel_err 逐点严格下降（解析性质，非趋势）；
      - 无注入时（w=0）强偏差下排序显著低于满分。
    """
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0,
                                  w_bases=[0.0, 0.2, 0.4, 0.8, 1.0],
                                  n_requests=500, slots=5, seed=42)
    sp = [r["L3_spearman"] for r in inj["rows"]]
    rel = [r["L3_rel_err"] for r in inj["rows"]]
    # 端点确定性：w_base=1 ⇒ L3 完全等于真值，秩相关必为 1.0（不依赖 RNG）
    assert sp[-1] == pytest.approx(1.0, abs=1e-9), sp
    # 逐点严格单调（解析性质：凸组合向真值收缩，rank 误差单调减少）
    assert all(b > a for a, b in zip(sp, sp[1:])), sp
    assert all(b < a for a, b in zip(rel, rel[1:])), rel
    # 无注入（w=0）时强偏差下排序接近随机，应显著低于满分
    assert sp[0] < 0.2, f"无注入时强偏差下排序应接近随机，实际 {sp[0]}"


def test_injection_l2_shared_across_w_base():
    """L2 与 w_base 无关：所有 w_base 档共享同一份排序级估计。

    锁定「单次模拟 + 解析扫描」的实现语义——若有人改回每 w_base 独立
    重抽曝光，L2 列将出现多个不同值，本测试立即失败。
    """
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0,
                                  w_bases=[0.0, 0.2, 0.4, 0.8, 1.0],
                                  n_requests=400, slots=5, seed=42)
    assert len({r["L2_rel_err"] for r in inj["rows"]}) == 1
    assert len({r["L2_spearman"] for r in inj["rows"]}) == 1


def test_propagation_matrix_order_independent():
    """CRN：τ 的扫描顺序不影响各 τ 的结果（每个 τ 独立重置 RNG）。"""
    t = _true_ctr()
    taus = [0.0, 1.0, 2.0]
    fwd = propagation_matrix(t, n_items=60, taus=taus,
                             n_requests=300, slots=5, seed=42)
    rev = propagation_matrix(t, n_items=60, taus=taus[::-1],
                             n_requests=300, slots=5, seed=42)
    by_tau_f = {r["tau"]: r for r in fwd["rows"]}
    by_tau_r = {r["tau"]: r for r in rev["rows"]}
    for tau in taus:
        assert by_tau_f[tau] == by_tau_r[tau]


def test_propagation_matrix_reproducible():
    """同一 seed 两次运行结果一致（项目基线要求）。"""
    t = _true_ctr()
    kw = dict(n_items=60, taus=[0.0, 1.0, 2.0], n_requests=200, slots=5, seed=7)
    a = propagation_matrix(t, **kw)
    b = propagation_matrix(t, **kw)
    assert a["rows"] == b["rows"]


# ----------------------------------------------------------------------
# 多种子（n_seeds）—— 误差棒（S2）
# ----------------------------------------------------------------------
def test_multi_seed_single_matches_default():
    """n_seeds=1 的输出与省略 n_seeds 完全一致（向后兼容锁定）。"""
    t = _true_ctr()
    kw = dict(n_items=60, taus=[0.0, 2.0], n_requests=200, slots=5, seed=42)
    base = propagation_matrix(t, **kw)
    one = propagation_matrix(t, n_seeds=1, **kw)
    assert one["rows"] == base["rows"]
    assert one["n_seeds"] == 1

    inj_kw = dict(n_items=60, tau=2.0, w_bases=[0.0, 1.0], n_requests=200,
                  slots=5, seed=42)
    assert (mixing_second_injection(t, n_seeds=1, **inj_kw)["rows"]
            == mixing_second_injection(t, **inj_kw)["rows"])


def test_multi_seed_adds_std_fields():
    """n_seeds>1：逐字段附 *_std 误差棒，且与 boundary.py 口径一致。"""
    t = _true_ctr()
    m = propagation_matrix(t, n_items=60, taus=[0.0, 2.0], n_requests=200,
                           slots=5, seed=42, n_seeds=3)
    assert m["n_seeds"] == 3
    assert "种子" in m["rng_policy"]
    for row in m["rows"]:
        numeric = [k for k in row if not k.endswith("_std") and k != "tau"]
        for k in numeric:
            assert k + "_std" in row, f"缺少 {k}_std"
            assert row[k + "_std"] >= 0.0
        assert row["tau"] == row.get("tau")  # key 字段不被聚合
    # 跨种子均值下核心趋势仍应成立：L2_spearman 随 τ 单调下降
    sp = [r["L2_spearman"] for r in m["rows"]]
    assert all(b < a for a, b in zip(sp, sp[1:])), sp


def test_multi_seed_injection_endpoint_exact():
    """注入 w=1.0 端点跨种子确定性：L3≡truth ⇒ 均值严格 1.0/0.0、std=0。"""
    t = _true_ctr()
    inj = mixing_second_injection(t, n_items=60, tau=2.0,
                                  w_bases=[0.0, 0.5, 1.0], n_requests=200,
                                  slots=5, seed=42, n_seeds=3)
    assert inj["n_seeds"] == 3
    last = inj["rows"][-1]
    assert last["L3_spearman"] == pytest.approx(1.0, abs=1e-9)
    assert last["L3_rel_err"] == pytest.approx(0.0, abs=1e-6)
    assert last["L3_spearman_std"] == pytest.approx(0.0, abs=1e-9)
