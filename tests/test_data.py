"""数据层 + 特征库 冒烟测试（小规模）。"""
import numpy as np

from src.data.synthetic_generator import (
    generate_items, generate_users, generate_logs,
)
from src.feature.store import build_training_matrix, feature_dim
from src.recall.baselines import PopularRecall, ItemCFRecall


def test_generate_small():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=1000)
    users = generate_users(rng, n_users=200)
    logs = generate_logs(rng, users, items, impressions_per_user=10)

    assert len(items) == 1000
    assert len(users) == 200
    assert {"user_id", "item_id", "category", "clicked", "converted", "true_ctr"} <= set(logs.columns)
    # 点击率应落在合理区间（受控生成）
    assert 0.0 <= logs["clicked"].mean() <= 1.0
    # 转化只发生在点击后
    assert (logs["converted"] <= logs["clicked"]).all()


def test_feature_matrix():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=1000)
    users = generate_users(rng, n_users=200)
    logs = generate_logs(rng, users, items, impressions_per_user=10)

    dense, sparse, yc, yv = build_training_matrix(logs, users, items)
    assert dense.shape[0] == len(logs)
    assert sparse.shape[1] == 3
    assert yc.mean() == logs["clicked"].mean()


def test_recall_runs():
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=1000)
    users = generate_users(rng, n_users=200)
    logs = generate_logs(rng, users, items, impressions_per_user=10)

    pop = PopularRecall(top_k=50).fit(logs)
    assert len(pop.recall(k=10)) == 10

    icf = ItemCFRecall(top_k=50).fit(logs)
    clicked = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    uid = clicked.index[0]
    assert isinstance(icf.recall(list(clicked.loc[uid]), k=10), list)
