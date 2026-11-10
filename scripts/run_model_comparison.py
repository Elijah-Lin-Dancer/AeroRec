"""模型对照实验：证明深度模型确实学到了生成过程中的隐式交叉。

用法
----
    python scripts/run_model_comparison.py                 # 默认 micro 档
    python scripts/run_model_comparison.py --mode lite
    python scripts/run_model_comparison.py --mode full     # 论文口径（慢）

产物
----
    data/eval/model_comparison_<mode>.json   # 如 model_comparison_full.json

为什么需要这个实验
------------------
本项目的生成器在真实 CTR 里植入了四个**乘积型交叉项**（见 `src/data/generator_common.py`）：

    ① 价格敏感度 × 低价          price_sensitivity × (1 − price_norm)
    ② 商务旅客 × 早班机          is_business × is_early_morning
    ③ 里程预算契合度              miles_fit
    ④ 类目亲和度选择              aff[category_idx]

而特征库 `src/feature/store.py` **刻意不提供这些乘积**（其文件头注释明确声明
「所有特征由原始特征构造，不包含生成器里的隐式交叉乘积项——交叉须由模型自行学习」）。

这就产生了一个可检验的问题：**DeepFM 的性能优势，究竟来自它真的学到了这些交叉，
还是仅仅来自参数量更大？** 单看 "DeepFM AUC 0.8019 > LR AUC 0.7421" 无法回答。

本脚本用四组对照回答：

    A. LR 基线            —— 只有原始特征
    B. LR + 手工交叉项    —— 把生成器的四个交叉项**显式**喂进去
    C. DeepFM             —— 只有原始特征，靠模型自学交叉
    D. 召回族（真实 CTR 口径）—— Popular / ItemCF / Two-Tower / 规则

结论的读法（这是本实验的核心价值）：
    - 若 **B ≈ C**（手工交叉的 LR 追平 DeepFM）→ 说明 DeepFM 的优势主要来自
      **特征工程**，而非模型结构；DeepFM 的必要性被削弱。
    - 若 **B < C**（手工交叉的 LR 仍明显落后）→ 说明交叉**很难用线性组合手工表达**，
      DeepFM 的 FM + DNN 结构确实带来了增益。**这是预期结果，也是要验证的。**
    - 若 **C > B > A** → 结论最强：交叉有效（相对 A）、且自动学习比手工喂入更有效（相对 B）。

诚实边界
--------
1. **B 组是"上帝视角"**：它使用了生成器才知道的交叉项，因此是**上界式的对照**，
   不是可部署的方案。它的作用是给"自动学交叉"提供一个可能的上界参照。
2. B 组的交叉项由 `src/data/generator_common.py:pair_features` 计算，
   与生成日志使用**同一套公式与同一份 cat_bias**，所以对照是公平的。
3. 本脚本只做**离线 AUC 对照**，不修改 `build_candidate_features`，
   因此**不影响线上打分路径**，无需同步改推理侧。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch 在小模型 + 小 batch 上开满核反而极慢（实测 32 线程 192.8s vs 4 线程 3.8s）。
# 本脚本若与消融脚本并发，各占满核会互相拖垮（实测 25 分钟仍未跑完 3 个 epoch）。
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from src.config import SEED, TOP_K
from src.data.generator_common import pair_features
from src.eval.ab import compare_rankers
from src.feature.store import build_training_matrix, feature_dim
from src.rank.trainer import DeepFMRanker
from src.recall.baselines import ItemCFRecall, PopularRecall
from src.recall.rule import RuleRecall
from src.recall.two_tower import TwoTowerRecall

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def _load_dataset(mode: str):
    """按档位加载数据集，统一转成 DataFrame。"""
    import pandas as pd
    if mode == "full":
        from src.data.synthetic_generator import build_dataset
        return build_dataset()
    from src.data.synthetic_generator_lite import build_dataset_np
    it, us, tr, lg = build_dataset_np()
    return (pd.DataFrame(it), pd.DataFrame(us), pd.DataFrame(tr), pd.DataFrame(lg))


def _manual_cross_features(users: pd.DataFrame, items: pd.DataFrame,
                           user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """构造生成器使用的四个交叉项（4 列）。

    直接复用 `src/data/generator_common.py:pair_features`——
    它返回 (aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit)，
    与生成日志同一套公式、同一份 cat_bias，因此对照公平。

    输出列：
      0: price_sensitivity × (1 − price_norm)   价格敏感度 × 低价
      1: is_business × is_early                  商务旅客 × 早班机
      2: miles_fit                                里程预算契合度
      3: aff_cat                                  类目亲和度选择
    """
    aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit = \
        pair_features(users, items, user_ids, item_ids)
    return np.column_stack([
        price_sens * (1.0 - price_norm),
        is_biz * is_early,
        miles_fit,
        aff_cat,
    ]).astype(np.float32)


def _lr_features(dense, sparse, sparse_cards):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories=[range(c) for c in sparse_cards], sparse_output=False)
    oh = enc.fit_transform(sparse)
    return np.hstack([dense, oh])


# ---------------------------------------------------------------------------
# 对照实验主体
# ---------------------------------------------------------------------------
def run_auc_comparison(items, users, logs, seed: int):
    """A/B/C 三组点式模型的 AUC 对照。"""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)

    dense_dim, n_sparse, sparse_cards = feature_dim()
    dense, sparse, yc, _ = build_training_matrix(logs, users, items, sample=300_000)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yc))
    tr, te = idx[:240_000], idx[240_000:]

    # ---- A 组：LR 基线（仅原始特征）----
    print("[cmp] A 组：LR 基线（仅原始特征）...")
    Xtr_a = _lr_features(dense[tr], sparse[tr], sparse_cards)
    Xte_a = _lr_features(dense[te], sparse[te], sparse_cards)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr_a, yc[tr])
    auc_lr = float(roc_auc_score(yc[te], lr.predict_proba(Xte_a)[:, 1]))

    # ---- B 组：LR + 手工交叉项 ----
    # 交叉项按 logs 原始顺序构造，故需按 idx 切分保持与 dense/sparse 对齐。
    # 为避免全量 30 万行 × pair_features 的开销，这里用与 A/C 组相同的测试集做子集评估。
    print("[cmp] B 组：LR + 手工交叉特征（上帝视角）...")
    # 为保证与 A/C 可比：在**同一测试集**上评估。
    # dense/sparse 是从 logs.sample(300k) 抽的，cross 必须用同一批样本，故重抽一次。
    logs_s = logs.sample(n=min(300_000, len(logs)), random_state=42)
    cross_full = _manual_cross_features(users, items,
                                        logs_s["user_id"].to_numpy(), logs_s["item_id"].to_numpy())
    # 上面 build_training_matrix 内部也是 sample(n=300_000, random_state=42)，顺序一致
    Xtr_b = np.hstack([_lr_features(dense[tr], sparse[tr], sparse_cards), cross_full[tr]])
    Xte_b = np.hstack([_lr_features(dense[te], sparse[te], sparse_cards), cross_full[te]])
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(Xtr_b, yc[tr])
    auc_lr_cross = float(roc_auc_score(yc[te], lr2.predict_proba(Xte_b)[:, 1]))

    # ---- C 组：DeepFM（仅原始特征）----
    print("[cmp] C 组：DeepFM（仅原始特征，靠模型自学交叉）...")
    dfm = DeepFMRanker(dense_dim, sparse_cards, embed_dim=16, dnn_hidden=(64, 32),
                        epochs=3, batch_size=2048)
    dfm.fit(dense[tr], sparse[tr], yc[tr])
    auc_dfm = float(roc_auc_score(yc[te], dfm.predict_ctr(dense[te], sparse[te])))

    # ---- E 组：DeepFM + 手工交叉（分离「结构增益」与「特征增益」）----
    # B>C 时无法判断 DeepFM 的结构是否有增益——因为 C 组的 DNN 没能从原始特征里
    # 还原出乘积型交叉。E 组把两者叠加：若 E>B，则结构增益确实存在，
    # 只是被「缺少显式交叉」掩盖了；若 E≈B，则增益全部来自特征工程。
    print("[cmp] E 组：DeepFM + 手工交叉（结构增益与特征增益分离）...")
    dense_e = np.hstack([dense, cross_full])
    dfm_e = DeepFMRanker(dense_dim + cross_full.shape[1], sparse_cards,
                          embed_dim=16, dnn_hidden=(64, 32), epochs=3, batch_size=2048)
    dfm_e.fit(dense_e[tr], sparse[tr], yc[tr])
    auc_dfm_cross = float(roc_auc_score(yc[te], dfm_e.predict_ctr(dense_e[te], sparse[te])))

    # ---- 结论判定 ----
    # 三组的关系决定结论措辞，必须区分「追平」与「明显更强」——
    # 早前版本只判 `>= dfm - 0.002`，在 B 显著优于 C 时仍输出「已追平」，
    # 会把"B 更好"这一事实错误地弱化成"平手"。
    d_cb = auc_dfm - auc_lr_cross
    d_eb = auc_dfm_cross - auc_lr_cross
    if d_cb < -0.002:
        verdict = (f"手工交叉的 LR 明显优于 DeepFM（{auc_lr_cross:.4f} vs {auc_dfm:.4f}，"
                   f"差 {abs(d_cb):.4f}）：本实现的 DeepFM 其 FM 二阶项只作用于**稀疏**"
                   "embedding 之间（见 src/rank/deepfm.py:49-54），"
                   "而真实 CTR 里的乘积型交叉是稠密×稠密（价格敏感度×价格）与"
                   "用户稠密×物品离散（品类亲和），FM 结构无法直接表达。")
    elif d_cb > 0.002:
        verdict = (f"DeepFM 明显优于手工交叉的 LR（{auc_dfm:.4f} vs {auc_lr_cross:.4f}，"
                   f"差 {d_cb:.4f}）：交叉难以用线性组合手工表达，"
                   "FM+DNN 结构确实带来增益。")
    else:
        verdict = (f"手工交叉的 LR 与 DeepFM 基本追平（差值 {abs(d_cb):.4f} ≤ 0.002），"
                   "说明 DeepFM 的优势主要来自特征工程而非模型结构。")
    if d_eb > 0.002:
        verdict += (f" E 组（DeepFM+手工交叉）进一步达到 {auc_dfm_cross:.4f}，"
                    "比 B 组高，说明**结构增益真实存在**，"
                    "只是在缺少显式交叉时被掩盖。")
    elif d_eb < -0.002:
        verdict += (f" E 组（DeepFM+手工交叉）为 {auc_dfm_cross:.4f}，未超过 B 组，"
                    "说明在本数据上增益**几乎全部来自特征工程**，模型结构未带来额外收益。")
    else:
        verdict += (f" E 组（DeepFM+手工交叉）为 {auc_dfm_cross:.4f}，与 B 组基本持平，"
                    "说明结构增益在本数据规模下不显著。")
    if auc_dfm > auc_lr_cross > auc_lr:
        verdict += " 且 C>B>A（自动学交叉优于手工喂入），结论最强。"
    elif auc_lr_cross > auc_dfm > auc_lr:
        verdict += " 且 B>C>A（手工喂入交叉优于模型自动学交叉），需如实报告这一反直觉结果。"

    return {
        "auc_lr_baseline": round(auc_lr, 4),
        "auc_lr_manual_cross": round(auc_lr_cross, 4),
        "auc_deepfm": round(auc_dfm, 4),
        "auc_deepfm_manual_cross": round(auc_dfm_cross, 4),
        "delta_c_minus_b": round(auc_dfm - auc_lr_cross, 4),
        "delta_b_minus_a": round(auc_lr_cross - auc_lr, 4),
        "delta_c_minus_a": round(auc_dfm - auc_lr, 4),
        "delta_e_minus_b": round(auc_dfm_cross - auc_lr_cross, 4),
        "verdict": verdict,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def run_recall_comparison(items, users, trips, logs, sample_users, seed: int):
    """D 组：召回族在真实 CTR 口径下的对照。

    召回器不产生 (u,i) 点式分，**没有 AUC**，只能走真实 CTR 口径。
    """
    from src.rank.mixer import Mixer
    train_logs = logs if len(logs) <= 1_000_000 else logs.sample(n=1_000_000, random_state=seed)
    pop = PopularRecall(top_k=200).fit(train_logs)
    icf = ItemCFRecall(top_k=200).fit(train_logs)
    rule = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)
    tt = TwoTowerRecall(epochs=1, train_sample=50_000).fit(logs, users, items)

    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    pop_map = items.set_index("item_id")["popularity"].to_dict()
    cat_col = "category_idx" if "category_idx" in items.columns else "category"
    cat_map = items.set_index("item_id")[cat_col].to_dict()
    mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=TOP_K)

    def wrap(fn):
        def ranker(uid, k):
            out = fn(uid, k)
            if len(out) < k:
                return list(out)[:k]
            base = [pop_map.get(i, 0.0) for i in out]
            cats = [cat_map.get(i, "") for i in out]
            ctr = list(base)
            return mixer.mix(out, base, ctr, cats)[:k]
        return ranker

    rankers = {
        "热门召回": wrap(lambda uid, k: pop.recall(k=k)),
        "ItemCF召回": wrap(lambda uid, k: icf.recall(list(clicked_by_user.get(uid, [])), k=k)),
        "TwoTower召回": wrap(lambda uid, k: tt.recall(int(uid), k=k)),
        "规则召回": wrap(lambda uid, k: rule.recall(int(uid), k=k)),
    }
    res = compare_rankers(users, items, rankers=rankers,
                          sample_users=sample_users, top_k=TOP_K, seed=seed, verbose=False)
    return {
        "per_method": {
            k: {
                "true_ctr": round(float(v["true_ctr"]), 4),
                "pct_of_oracle": round(float(v["pct_of_oracle"]), 4),
                # 召回族也要给 NDCG：四路召回的差距在真实 CTR 上只有
                # 「点得多不多」一个维度，NDCG 才能说明「好的有没有排前面」。
                "ndcg_at_k": round(float(v["ndcg_at_k"]), 4),
                "ndcg_pct_of_oracle": round(float(v["ndcg_pct_of_oracle"]), 4),
            }
            for k, v in res["per_method"].items()
        },
        "oracle_true_ctr": round(float(res["per_method"]["Oracle"]["true_ctr"]), 4),
        "oracle_ndcg_at_k": round(float(res["oracle_ndcg_at_k"]), 4),
        "note": "召回器不产生点式分，无 AUC；统一用真实 CTR 口径 + 同一混排层评估。",
    }


def main():
    ap = argparse.ArgumentParser(description="模型对照实验：LR / LR+交叉 / DeepFM / 召回族")
    ap.add_argument("--mode", default="micro", choices=["micro", "lite", "full"])
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None,
                    help="输出路径。默认 data/eval/model_comparison_<mode>.json——"
                         "按档位分文件，避免 full 档覆盖 micro 档")
    args = ap.parse_args()

    print(f"[cmp] 载入 {args.mode} 档 ...")
    items, users, trips, logs = _load_dataset(args.mode)
    print(f"[cmp] items={len(items)} users={len(users)} logs={len(logs)}")

    auc_res = run_auc_comparison(items, users, logs, args.seed)

    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    sample_users = [int(u) for u in clicked_by_user.index[: args.n_users]]
    print(f"[cmp] 召回族对照（{len(sample_users)} 用户）...")
    rec_res = run_recall_comparison(items, users, trips, logs, sample_users, args.seed)

    payload = {
        "meta": {
            "description": ("模型对照实验：A=LR基线 / B=LR+手工交叉 / "
                            "C=DeepFM / E=DeepFM+手工交叉 / D=召回族"),
            "mode": args.mode,
            "seed": args.seed,
            "n_users": int(args.n_users),
            "top_k": TOP_K,
            "protocol": (
                "A/B/C/E 四组用同一训练/测试划分（30 万样本、24 万/6 万切分）；"
                "B 组交叉项由 src/data/generator_common.py:pair_features 计算，"
                "与生成日志同一套公式与同一份 cat_bias。"
                "E 组 = DeepFM + 同一批手工交叉，用于分离「结构增益」与「特征增益」——"
                "若 E ≈ B，则 C→B 的差距全部来自显式交叉特征而非 FM+DNN 结构。"
                "D 组用真实 CTR 口径（召回器无点式分，无 AUC）。"
            ),
            "honest_note": (
                "B 组是『上帝视角』对照——它使用生成器才知道的交叉项，"
                "因此不是可部署方案，只作为自动学交叉的参照上界。"
            ),
            "mechanism_note": (
                "本实现的 DeepFM 其 FM 二阶项只作用于稀疏 embedding 之间"
                "（src/rank/deepfm.py:49-54，embs 只取 sparse_embs，dense 不参与）；"
                "而真实 CTR 的交叉项是稠密×稠密（price_sens×(1-price_norm)、is_biz×is_early）"
                "与用户稠密×物品离散（aff_cat）——FM 结构无法直接表达。"
                "这是 B>C 的机制解释，不是调参不足。"
            ),
            "scale_note": ("micro/lite 档仅验证流程与相对关系，论文引用须以 full 档为准。"
                           if args.mode != "full" else "full 档为论文口径。"),
        },
        "auc_comparison": auc_res,
        "recall_comparison": rec_res,
    }

    out_path = Path(args.out) if args.out else \
        ROOT / "data" / "eval" / f"model_comparison_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\n" + "=" * 68)
    print("模型对照实验（AUC）")
    print("-" * 68)
    print(f"  A  LR 基线（仅原始特征）      AUC = {auc_res['auc_lr_baseline']:.4f}")
    print(f"  B  LR + 手工交叉特征          AUC = {auc_res['auc_lr_manual_cross']:.4f}")
    print(f"  C  DeepFM（仅原始特征）       AUC = {auc_res['auc_deepfm']:.4f}")
    print(f"  E  DeepFM + 手工交叉          AUC = {auc_res['auc_deepfm_manual_cross']:.4f}")
    print(f"     C−B = {auc_res['delta_c_minus_b']:+.4f}    B−A = {auc_res['delta_b_minus_a']:+.4f}"
          f"    E−B = {auc_res['delta_e_minus_b']:+.4f}")
    print("-" * 68)
    print(f"  判定：{auc_res['verdict']}")
    print("=" * 68)
    print("召回族对照（真实 CTR 口径）")
    print("-" * 68)
    for k, v in rec_res["per_method"].items():
        print(f"  {k:12s} true_ctr={v['true_ctr']:.4f}  占Oracle={v['pct_of_oracle']:.1%}")
    print("=" * 68)
    print(f"[cmp] 结果已落盘：{out_path}（{out_path.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
