"""离线评估：LR vs DeepFM 的 AUC 对比 + 排序层真实 CTR 对比。

运行：python -m src.eval.evaluate

核心量化结论（受控合成实验）：
  1. 合成数据含隐式非线性交叉，DeepFM 应比 LR 的 AUC 更高，体现深度模型自动学交叉；
  2. 排序层（DeepFM CTR + 5 步混排）应比"热门/随机"基线有更高的推荐列表真实 CTR。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from src.config import SEED, TOP_K
from src.data.synthetic_generator import build_dataset, compute_true_ctr
from src.feature.store import build_training_matrix, build_candidate_features, feature_dim
from src.rank.trainer import DeepFMRanker
from src.rank.mixer import Mixer
from src.rank.degrade import DegradeRanker, compute_recent_ctr
from src.recall.baselines import PopularRecall
from src.recall.rule import RuleRecall


def _lr_features(dense, sparse, sparse_cards):
    enc = OneHotEncoder(categories=[range(c) for c in sparse_cards], sparse_output=False)
    oh = enc.fit_transform(sparse)
    return np.hstack([dense, oh])


def main():
    items, users, trips, logs = build_dataset()
    dense_dim, n_sparse, sparse_cards = feature_dim()

    # ---- 1) CTR 模型 AUC：LR vs DeepFM ----
    print("[eval] 构建训练/测试矩阵 ...")
    dense, sparse, yc, _ = build_training_matrix(logs, users, items, sample=300_000)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(yc))
    tr, te = idx[:240_000], idx[240_000:]

    print("[eval] 训练 Logistic Regression 基线 ...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(_lr_features(dense[tr], sparse[tr], sparse_cards), yc[tr])
    lr_ctr = lr.predict_proba(_lr_features(dense[te], sparse[te], sparse_cards))[:, 1]

    print("[eval] 训练 DeepFM ...")
    dfm = DeepFMRanker(dense_dim, sparse_cards, embed_dim=16, dnn_hidden=(64, 32),
                       epochs=3, batch_size=2048)
    dfm.fit(dense[tr], sparse[tr], yc[tr],
            val_dense=dense[te], val_sparse=sparse[te], val_y=yc[te])
    dfm_ctr = dfm.predict_ctr(dense[te], sparse[te])

    lr_auc = roc_auc_score(yc[te], lr_ctr)
    dfm_auc = roc_auc_score(yc[te], dfm_ctr)
    print("=" * 56)
    print("CTR 排序模型 AUC 对比（测试集 6 万样本）")
    print(f"  Logistic Regression : {lr_auc:.4f}")
    print(f"  DeepFM              : {dfm_auc:.4f}")
    print(f"  提升                  : {dfm_auc - lr_auc:+.4f}")
    print("=" * 56)

    # ---- 2) 排序层效果：推荐列表平均真实 CTR ----
    print("\n[eval] 排序层效果：推荐列表平均真实 CTR ...")
    pop = PopularRecall(top_k=200).fit(logs)
    rule = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)
    mixer = Mixer(w_base=0.4, w_ctr=0.6, max_same_category=2, top_k=TOP_K)
    # 降级兜底包装
    fallback = compute_recent_ctr(logs)
    deg = DegradeRanker(dfm.predict_ctr, fallback)

    pop_map = items.set_index("item_id")["popularity"].to_dict()
    cat_map = items.set_index("item_id")["category"].to_dict()
    top_popular = items.sort_values("popularity", ascending=False).head(TOP_K)["item_id"].tolist()
    item_ids_all = items["item_id"].to_numpy()

    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    sample_users = [int(u) for u in clicked_by_user.index[:50]]

    mix_ctrs, pop_ctrs, rnd_ctrs = [], [], []
    for uid in sample_users:
        cands = list(dict.fromkeys(pop.recall(200) + rule.recall(uid, k=200)))
        if len(cands) < TOP_K:
            continue
        base = [pop_map.get(i, 0.0) for i in cands]
        cd, cs = build_candidate_features(uid, cands, users, items)
        ctr = deg.predict_ctr(cd, cs, cands)
        cats = [cat_map[i] for i in cands]
        ranked = mixer.mix(cands, base, ctr, cats)

        mix_ctrs.append(compute_true_ctr(users, items, [uid] * len(ranked), ranked).mean())
        pop_ctrs.append(compute_true_ctr(users, items, [uid] * len(top_popular), top_popular).mean())
        rnd_ids = rng.choice(item_ids_all, size=TOP_K, replace=False)
        rnd_ctrs.append(compute_true_ctr(users, items, [uid] * len(rnd_ids), rnd_ids).mean())

    mix_mean = float(np.mean(mix_ctrs))
    pop_mean = float(np.mean(pop_ctrs))
    rnd_mean = float(np.mean(rnd_ctrs))
    print(f"  随机 Top10 平均真实 CTR    : {rnd_mean:.4f}")
    print(f"  热门 Top10 平均真实 CTR    : {pop_mean:.4f}")
    print(f"  DeepFM+混排 Top10 真实CTR  : {mix_mean:.4f}")
    print(f"  相对热门提升                : {(mix_mean - pop_mean) / max(pop_mean, 1e-6) * 100:+.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
