"""数据层 + 特征库 + 召回层 冒烟测试（小规模）。"""
import numpy as np

from src.data.synthetic_generator import (
    generate_items, generate_users, generate_trips, generate_logs,
)
from src.feature.store import build_training_matrix, feature_dim
from src.recall.baselines import PopularRecall, ItemCFRecall
from src.recall.two_tower import TwoTowerRecall
from src.recall.rule import RuleRecall
from src.recall.fusion import RecallFusion


def _small_data(n_items=1000, n_users=200, imp=10):
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=n_items)
    users = generate_users(rng, n_users=n_users)
    trips = generate_trips(rng, n_users=n_users)
    logs = generate_logs(rng, users, items, impressions_per_user=imp)
    return items, users, trips, logs


def test_generate_small():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=1000)
    users = generate_users(rng, n_users=200)
    logs = generate_logs(rng, users, items, impressions_per_user=10)

    assert len(items) == 1000
    assert len(users) == 200
    assert {"user_id", "item_id", "category", "clicked", "converted", "true_ctr"} <= set(logs.columns)
    assert 0.0 <= logs["clicked"].mean() <= 1.0
    assert (logs["converted"] <= logs["clicked"]).all()


def test_feature_matrix():
    items, users, trips, logs = _small_data()
    dense, sparse, yc, yv = build_training_matrix(logs, users, items)
    assert dense.shape[0] == len(logs)
    assert sparse.shape[1] == 3
    assert yc.mean() == logs["clicked"].mean()


def test_recall_runs():
    items, users, trips, logs = _small_data()
    pop = PopularRecall(top_k=50).fit(logs)
    assert len(pop.recall(k=10)) == 10

    icf = ItemCFRecall(top_k=50).fit(logs)
    clicked = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    uid = clicked.index[0]
    assert isinstance(icf.recall(list(clicked.loc[uid]), k=10), list)


def test_two_tower_recall():
    items, users, trips, logs = _small_data()
    tt = TwoTowerRecall(epochs=1, train_sample=300, batch_size=256).fit(logs, users, items)
    rec = tt.recall(0, k=10)
    assert isinstance(rec, list) and len(rec) <= 10
    assert all(0 <= i < len(items) for i in rec)


def test_rule_recall_returns_routes():
    items, users, trips, logs = _small_data()
    rr = RuleRecall().fit(items=items, users=users, trips=trips)
    rec = rr.recall(0, k=10)
    assert isinstance(rec, list)
    route_ids = set(items[items["category_idx"] == 0]["item_id"])
    assert all(i in route_ids for i in rec)


def test_rule_recall_miles_budget():
    items, users, trips, logs = _small_data(n_items=5000)
    rr = RuleRecall().fit(items=items, users=users, trips=trips)
    uid = 0
    user = users[users["user_id"] == uid].iloc[0]
    budget = float(user["miles_balance"]) * 1.10
    rec = rr.recall(uid, k=200)
    it = items.set_index("item_id")
    for i in rec:
        row = it.loc[i]
        if row["is_miles_ticket"] == 1:
            assert row["miles_required"] <= budget + 1e-6


def test_fusion_merge():
    f = RecallFusion(max_candidates=5)
    out = f.merge([1, 2, 3], [2, 3, 4], weights=[1.0, 1.0])
    assert out[0] == 2 and len(out) == 4 and set(out) == {1, 2, 3, 4}
