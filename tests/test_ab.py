"""模拟 A/B 冒烟测试。"""
import numpy as np

from src.data.synthetic_generator import generate_items, generate_users, compute_true_cvr
from src.eval.ab import simulate_ab, proportion_ztest


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
