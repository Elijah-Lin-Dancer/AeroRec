"""离线评估：LR vs DeepFM 的 AUC 对比 + **真实 CTR 口径**的排序方法对比（含 Oracle 上界）。

运行：python -m src.eval.evaluate

核心量化结论（受控合成实验）
--------------------------
1. 合成数据含隐式非线性交叉，DeepFM 应比 LR 的 AUC 更高，体现深度模型自动学交叉；
2. 在**真实 CTR 口径**下（不依赖任何平滑估计，因此可引用），
   对比 随机 / 热门 / DeepFM+混排 / Oracle 上界 四者。

⚠️ 为什么不直接引用"模拟在线 A/B"的提升数字
--------------------------------------------
热门基线按"历史日志的贝叶斯平滑 CTR"排序，而日志是随机曝光的，
曝光少的物品偶然被点击就会推高平滑 CTR，形成**稀疏性偏差**。
完整档实测：热门榜 top200 平滑 CTR 0.6231 vs 真实 CTR 0.2528（偏差 +0.37）。
这会让模拟 A/B **高估**实验组增益。
故本脚本以**真实 CTR 口径**为准，并打印偏差表说明差异。
"""
import numpy as np
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
from src.eval.ab import compare_rankers, baseline_bias_table


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
    print("=" * 64)
    print("① CTR 排序模型 AUC 对比（测试集 6 万样本）")
    print(f"  Logistic Regression : {lr_auc:.4f}")
    print(f"  DeepFM              : {dfm_auc:.4f}")
    print(f"  提升                : {dfm_auc - lr_auc:+.4f}")
    print("=" * 64)

    # ---- 2) 真实 CTR 口径：随机 / 热门 / DeepFM+混排 / Oracle ----
    print("\n[eval] 真实 CTR 口径对比（含上界 Oracle）...")
    pop = PopularRecall(top_k=200).fit(logs)
    rule = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)
    mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=TOP_K)
    deg = DegradeRanker(dfm.predict_ctr, compute_recent_ctr(logs))

    pop_map = items.set_index("item_id")["popularity"].to_dict()
    cat_map = items.set_index("item_id")["category"].to_dict()
    item_ids_all = items["item_id"].to_numpy()

    def random_rank(uid, k):
        return rng.choice(item_ids_all, size=k, replace=False).tolist()

    def popular_rank(uid, k):
        return pop.recall(k)

    def deepfm_mix_rank(uid, k):
        cands = list(dict.fromkeys(pop.recall(200) + rule.recall(int(uid), k=200)))
        if len(cands) < k:
            return cands
        base = [pop_map.get(i, 0.0) for i in cands]
        cd, cs = build_candidate_features(int(uid), cands, users, items)
        c = deg.predict_ctr(cd, cs, cands)
        cats = [cat_map[i] for i in cands]
        return mixer.mix(cands, base, c, cats)

    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    sample_users = [int(u) for u in clicked_by_user.index[:100]]

    result = compare_rankers(
        users, items,
        rankers={"随机": random_rank, "热门": popular_rank, "DeepFM+混排": deepfm_mix_rank},
        sample_users=sample_users, top_k=TOP_K, seed=SEED)

    # ---- 3) 热门基线的平滑 CTR vs 真实 CTR 偏差表 ----
    print("\n[eval] 热门基线偏差表（解释为何模拟 A/B 会高估收益）...")
    bias = baseline_bias_table(pop, users, items, top_n=200, sample_user=sample_users[0])
    if "error" not in bias:
        print(f"  热门榜 top{bias['top_n']}: 平滑CTR={bias['smooth_ctr']:.4f}  "
              f"真实CTR={bias['true_ctr']:.4f}  偏差={bias['bias']:+.4f}")
        print("  → 平滑 CTR 显著高估，模拟 A/B 中热门基线因此被削弱，")
        print("    实验组增益被高估。可引用结论请用上面的真实 CTR 口径。")

    # ---- 4) 汇总（可直接引用）----
    pm = result["per_method"]
    print("\n" + "=" * 64)
    print("② 可引用的量化结论（真实 CTR 口径，不含平滑估计）")
    print("-" * 64)
    print(f"  样本用户数            : {result['n_users']}")
    print(f"  随机 baseline         : {pm['随机']['true_ctr']:.4f}")
    print(f"  热门 baseline         : {pm['热门']['true_ctr']:.4f}")
    print(f"  DeepFM+混排           : {pm['DeepFM+混排']['true_ctr']:.4f}")
    print(f"  Oracle 上界           : {pm['Oracle']['true_ctr']:.4f}")
    print("-" * 64)
    print(f"  DeepFM 相对热门提升    : {pm['DeepFM+混排']['rel_to_base']:+.1%}")
    print(f"  DeepFM 达到上界比例    : {pm['DeepFM+混排']['pct_of_oracle']:.1%}")
    print(f"  热门   达到上界比例    : {pm['热门']['pct_of_oracle']:.1%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
