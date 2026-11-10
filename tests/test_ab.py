"""模拟 A/B 冒烟测试。"""
import numpy as np
from src.data.synthetic_generator import compute_true_cvr, generate_items, generate_users
from src.eval.ab import compare_rankers, proportion_ztest, simulate_ab


def test_proportion_ztest_significant():
    z, p = proportion_ztest(0.5, 1000, 0.4, 1000)
    assert z > 0 and p < 0.01


def test_compute_true_cvr_range():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=1000)
    users = generate_users(rng, n_users=200)
    cvr = compute_true_cvr(users, items, [0, 1, 2], [10, 20, 30])
    assert cvr.shape == (3,)
    assert (cvr >= 0).all() and (cvr <= 1).all()


def test_simulate_ab_runs():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=2000)
    users = generate_users(rng, n_users=300)

    def random_rank(uid, k):
        return rng.integers(0, len(items), size=k).tolist()

    res = simulate_ab(users, items, control_ranker=random_rank,
                      treatment_ranker=random_rank, n_users=50, top_k=10, seed=1)
    assert res["n_users"] == 50
    assert 0 <= res["ctrl_ctr"] <= 1 and 0 <= res["trt_ctr"] <= 1
    assert "ctr_lift" in res and "cvr_lift" in res


# ---------------------------------------------------------------------------
# compare_rankers 的端到端 NDCG 口径
# ---------------------------------------------------------------------------
def _ndcg_fixture(n_items=60, n_users=12):
    rng = np.random.default_rng(7)
    items = generate_items(rng, n_items=n_items)
    users = generate_users(rng, n_users=n_users)
    return users, items


def test_compare_rankers_ndcg_oracle_is_one():
    """Oracle 是候选池内按真实 CTR 降序取 Top-K，其 NDCG 必须为 1.0。"""
    users, items = _ndcg_fixture()

    def ranker(uid, k):
        return list(range(k))

    res = compare_rankers(users, items, {"热门": ranker}, sample_users=list(range(6)),
                           top_k=5, oracle_pool_size=40, verbose=False)
    assert res["per_method"]["Oracle"]["ndcg_at_k"] == 1.0
    assert res["oracle_ndcg_at_k"] == 1.0


def test_compare_rankers_ndcg_order_matters():
    """同一批物品，**打乱顺序**后 NDCG 必须下降——
    这正是 NDCG 相对 true_ctr 的增量信息（true_ctr 对顺序完全不敏感）。
    """
    users, items = _ndcg_fixture()
    top = list(range(8))

    res = compare_rankers(
        users, items,
        {"有序": lambda uid, k: top[:k], "乱序": lambda uid, k: top[:k][::-1]},
        sample_users=list(range(6)), top_k=5, oracle_pool_size=40, verbose=False)
    ordered = res["per_method"]["有序"]["ndcg_at_k"]
    shuffled = res["per_method"]["乱序"]["ndcg_at_k"]
    assert ordered > shuffled, (ordered, shuffled)
    # true_ctr 口径对顺序不敏感，两者应当相等
    assert (res["per_method"]["有序"]["true_ctr"]
            == res["per_method"]["乱序"]["true_ctr"])


def test_compare_rankers_ndcg_same_denominator():
    """所有方法的 NDCG 分母相同（同一候选池的 IDCG），
    因此 Oracle 的占比与真值口径一致地满足 0 ≤ ndcg ≤ 1。"""
    users, items = _ndcg_fixture()

    def ranker(uid, k):
        return list(range(k))

    res = compare_rankers(users, items, {"热门": ranker}, sample_users=list(range(6)),
                           top_k=5, oracle_pool_size=40, verbose=False)
    for name, d in res["per_method"].items():
        assert 0.0 <= d["ndcg_at_k"] <= 1.0, (name, d["ndcg_at_k"])
    assert res["per_method"]["Oracle"]["ndcg_pct_of_oracle"] == 1.0
