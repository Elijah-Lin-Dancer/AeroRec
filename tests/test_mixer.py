"""Mixer（5 步混排）单元测试：纯 numpy，无重依赖。

覆盖：归一化 / CTR 加权 / 分层排序 / 同品类打散 / 去重截断，
以及 w_base 对基础分权重的控制。
"""
from __future__ import annotations

from src.rank.mixer import Mixer


def test_mix_dedup_and_truncate():
    m = Mixer(top_k=3)
    ids = [1, 2, 3, 4, 5]
    base = [0.1, 0.2, 0.3, 0.4, 0.5]
    ctr = [0.5, 0.4, 0.3, 0.2, 0.1]
    cats = ["a", "b", "c", "d", "e"]
    out = m.mix(ids, base, ctr, cats)
    assert len(out) == 3
    assert len(set(out)) == 3                 # 去重
    assert set(out).issubset(set(ids))


def test_mix_truncates_when_input_has_duplicates():
    m = Mixer(top_k=4)
    ids = [1, 1, 2, 3, 4]                      # 含重复 id
    base = [0.1, 0.0, 0.3, 0.4, 0.5]
    ctr = [0.5, 0.0, 0.3, 0.2, 0.1]
    cats = ["a", "a", "c", "d", "e"]
    out = m.mix(ids, base, ctr, cats)
    assert len(out) == 4
    assert len(set(out)) == 4                 # 重复 id 只保留一个


def test_mix_single_category_keeps_ctr_order():
    # 全同类目时打散不生效，结果应等于 CTR 降序
    m = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=10)
    ids = list(range(10))
    base = [0.0] * 10
    ctr = [1.0 - i * 0.01 for i in range(10)]
    cats = ["x"] * 10
    out = m.mix(ids, base, ctr, cats)
    assert out == list(range(10))


def test_mix_scatter_prevents_long_category_runs():
    # max_same_category=1 → 同类目不得连续出现
    m = Mixer(max_same_category=1, top_k=10)
    ids = list(range(6))
    base = [0.0] * 6
    ctr = [1.0 - i * 0.01 for i in range(6)]
    cats = ["a", "b", "a", "b", "a", "b"]
    out = m.mix(ids, base, ctr, cats)
    out_cats = [cats[ids.index(i)] for i in out]
    for a, b in zip(out_cats, out_cats[1:]):
        assert a != b, f"出现连续同类目：{out_cats}"


def test_mix_trace_exposes_intermediate_states():
    m = Mixer(top_k=5)
    ids = list(range(5))
    base = [0.5, 0.4, 0.3, 0.2, 0.1]
    ctr = [0.1, 0.2, 0.3, 0.4, 0.5]
    cats = ["a", "b", "c", "d", "e"]
    tr = m.mix_trace(ids, base, ctr, cats)
    assert set(tr["final_ids"]) == set(m.mix(ids, base, ctr, cats))
    assert len(tr["combined"]) == 5
    assert len(tr["final_ids"]) == 5


def test_mix_w_base_controls_weighting():
    # w_base=1 ⇒ 纯基础分驱动；w_base=0 ⇒ 纯 CTR 驱动
    ids = [0, 1]
    base = [0.9, 0.1]
    ctr = [0.1, 0.9]
    cats = ["a", "b"]
    by_base = Mixer(w_base=1.0, w_ctr=0.0, top_k=2).mix(ids, base, ctr, cats)
    by_ctr = Mixer(w_base=0.0, w_ctr=1.0, top_k=2).mix(ids, base, ctr, cats)
    assert by_base == [0, 1]                   # 基础分大者在前
    assert by_ctr == [1, 0]                     # CTR 大者在前
