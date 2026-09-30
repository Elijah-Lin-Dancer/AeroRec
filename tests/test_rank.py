"""排序层（DeepFM / Mixer / Degrade）与推理特征 冒烟测试。"""
import numpy as np

from src.data.synthetic_generator import generate_items, generate_users, generate_logs
from src.feature.store import build_training_matrix, build_candidate_features, feature_dim
from src.rank.trainer import DeepFMRanker
from src.rank.mixer import Mixer
from src.rank.degrade import DegradeRanker, compute_recent_ctr


def _small(n_items=1000, n_users=200, imp=10):
    rng = np.random.default_rng(0)
    items = generate_items(rng, n_items=n_items)
    users = generate_users(rng, n_users=n_users)
    logs = generate_logs(rng, users, items, impressions_per_user=imp)
    return items, users, logs


def test_candidate_features_shape():
    items, users, logs = _small()
    dense_dim, n_sparse, cards = feature_dim()
    cd, cs = build_candidate_features(0, [0, 1, 2], users, items)
    assert cd.shape == (3, dense_dim)
    assert cs.shape == (3, n_sparse)


def test_deepfm_predict():
    items, users, logs = _small()
    dense, sparse, yc, _ = build_training_matrix(logs, users, items)
    dense_dim, n_sparse, cards = feature_dim()
    dfm = DeepFMRanker(dense_dim, cards, embed_dim=8, dnn_hidden=(16, 8),
                       epochs=1, batch_size=256)
    dfm.fit(dense[:500], sparse[:500], yc[:500])
    ctr = dfm.predict_ctr(dense[:50], sparse[:50])
    assert ctr.shape == (50,)
    assert (ctr >= 0).all() and (ctr <= 1).all()


def test_mixer_scatter_dedupe():
    m = Mixer(w_base=0.4, w_ctr=0.6, max_same_category=2, top_k=10)
    item_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cats = ['A', 'A', 'A', 'A', 'B', 'B', 'A', 'C', 'C', 'C', 'C']
    base = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    ctr = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
    out = m.mix(item_ids, base, ctr, cats)
    assert len(out) == 10                      # top_k 截断
    assert len(set(out)) == len(out)           # 去重
    cat_of = dict(zip(item_ids, cats))
    seq = [cat_of[i] for i in out]
    run = 1
    for i in range(1, len(seq)):               # 连续同品类 ≤ 2
        run = run + 1 if seq[i] == seq[i - 1] else 1
        assert run <= 2


def test_degrade_fallback():
    def bad(dense, sparse):
        raise RuntimeError("timeout")

    d = DegradeRanker(bad, {0: 0.2, 1: 0.1}, fallback_default=0.05)
    out = d.predict_ctr(np.zeros((3, 1)), np.zeros((3, 1), dtype=int), [0, 1, 99])
    assert out[0] == 0.2 and out[1] == 0.1 and out[2] == 0.05


def test_compute_recent_ctr():
    items, users, logs = _small()
    ctr = compute_recent_ctr(logs)
    assert isinstance(ctr, dict) and len(ctr) > 0
    assert all(0 <= v <= 1 for v in ctr.values())


def test_compute_true_ctr_consistency():
    from src.data.synthetic_generator import compute_true_ctr
    items, users, logs = _small()
    s = logs.sample(5, random_state=0)
    recalc = compute_true_ctr(users, items, s["user_id"].to_numpy(), s["item_id"].to_numpy())
    assert np.allclose(recalc, s["true_ctr"].to_numpy(), atol=1e-3)
