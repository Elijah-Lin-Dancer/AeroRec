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

📦 结果落盘（2026-11-10 新增）
--------------------------------
本脚本原先只 `print`，属于「跑一次终端里看一眼」的性质，不在零漂移门禁的覆盖范围内，
论文引用时无法回答「这个数字怎么复现」。现将同一批结果**同时**落盘为 JSON：

    data/eval/end_to_end.json

**落盘不改变任何计算口径**——打印与落盘共用同一批变量，仅额外写出。
所有数值保留 4 位小数（与 print 输出一致），跨机末位漂移见下方「可复现性边界」。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from src.config import SEED, TOP_K
from src.data.synthetic_generator import build_dataset
from src.eval.ab import baseline_bias_table, compare_rankers
from src.feature.store import build_candidate_features, build_training_matrix, feature_dim
from src.rank.degrade import DegradeRanker, compute_recent_ctr
from src.rank.mixer import Mixer
from src.rank.trainer import DeepFMRanker
from src.recall.baselines import PopularRecall
from src.recall.rule import RuleRecall

# 落盘产物路径（与 data/bias/ 同级，便于审计脚本统一扫描）
EVAL_OUT = Path(__file__).resolve().parents[2] / "data" / "eval" / "end_to_end.json"


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

    # ---- 5) 落盘 JSON（与上面打印共用同一批变量，口径不变）----
    payload = {
        "meta": {
            "description": "端到端离线评估结果：CTR 模型 AUC 对比 + 真实 CTR 口径排序对比",
            "mode": "full",
            "scale_note": ("本脚本用 src.data.synthetic_generator.build_dataset()"
                           "（full 档：10 万物品 × 10 万用户 × 500 万曝光），"
                           "即论文口径。micro/lite 档的对照见 "
                           "data/eval/ablation_micro.json 等产物。"),
            "seed": SEED,
            "top_k": TOP_K,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_sample_users": int(result["n_users"]),
            "note": (
                "本文件由 python -m src.eval.evaluate 生成，与 stdout 打印同源同口径。"
                "AUC 完全可复现；真实 CTR 类指标末位可能因 Oracle 池采样随机性跨机漂移，"
                "论文引用请取整到整数百分点。"
            ),
        },
        "auc": {
            "logistic_regression": round(float(lr_auc), 4),
            "deepfm": round(float(dfm_auc), 4),
            "delta": round(float(dfm_auc - lr_auc), 4),
        },
        "true_ctr_ranking": {
            "random": round(float(pm["随机"]["true_ctr"]), 4),
            "popular": round(float(pm["热门"]["true_ctr"]), 4),
            "deepfm_mixed": round(float(pm["DeepFM+混排"]["true_ctr"]), 4),
            "oracle": round(float(pm["Oracle"]["true_ctr"]), 4),
            "deepfm_rel_to_popular": round(float(pm["DeepFM+混排"]["rel_to_base"]), 4),
            "deepfm_pct_of_oracle": round(float(pm["DeepFM+混排"]["pct_of_oracle"]), 4),
            "popular_pct_of_oracle": round(float(pm["热门"]["pct_of_oracle"]), 4),
            # 端到端 graded NDCG@K：增益 = 候选池内真实 CTR，IDCG 取同一候选池
            # 降序前 K —— 四个方法共用同一分母，可直接横向比较。
            # 真实 CTR 只回答"点得多不多"，NDCG 额外回答"好的排前面没有"；
            # 论文第 5 章论证排序质量需要后者。
            "ndcg_at_k": {
                "random": round(float(pm["随机"]["ndcg_at_k"]), 4),
                "popular": round(float(pm["热门"]["ndcg_at_k"]), 4),
                "deepfm_mixed": round(float(pm["DeepFM+混排"]["ndcg_at_k"]), 4),
                "oracle": round(float(pm["Oracle"]["ndcg_at_k"]), 4),
                "deepfm_pct_of_oracle": round(
                    float(pm["DeepFM+混排"]["ndcg_pct_of_oracle"]), 4),
            },
            "ndcg_protocol": result["ndcg_protocol"],
        },
    }
    if "error" not in bias:
        payload["popularity_baseline_bias"] = {
            "top_n": int(bias["top_n"]),
            "smoothed_ctr": round(float(bias["smooth_ctr"]), 4),
            "true_ctr": round(float(bias["true_ctr"]), 4),
            "bias": round(float(bias["bias"]), 4),
            "note": "平滑 CTR 相对真实 CTR 的高估幅度；说明为何不采用平滑协议报告主结果",
        }

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n[eval] 结果已落盘：{EVAL_OUT}（{EVAL_OUT.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
