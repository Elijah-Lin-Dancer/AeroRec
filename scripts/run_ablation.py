"""消融实验：召回通道逐路关闭 + 混排五步逐步去除。

用法
----
    python scripts/run_ablation.py                 # 默认 micro 档（快）
    python scripts/run_ablation.py --mode lite     # 中档
    python scripts/run_ablation.py --mode full     # 完整档（慢）

产物
----
    data/eval/ablation_<mode>.json     # 按档位分文件，如 ablation_full.json

为什么需要这个脚本
------------------
评审对推荐类毕设的常规追问是「去掉某个组件会怎样」。本项目此前**零消融实验**
（全仓 grep `ablation` 零命中），而消融恰恰是把「系统实现」变成「有证据的系统实现」
的关键一步。本脚本补上两类消融：

1. **召回通道消融**：四路召回（热门 / ItemCF / Two-Tower / 规则）逐路关闭，
   观测最终排序的真实 CTR 口径表现如何变化。
2. **混排步骤消融**：五步混排（归一化 → CTR 加权 → 分层排序 → 同品类打散 → 去重截断）
   逐步去除，观测每一步对排序质量的贡献。

实现要点（决定了本脚本为何这么写）
----------------------------------
- **召回关闭 = 不注册该路**，而不是把权重设为 0。`RecallFusion._merge` 会把
  权重为 0 的召回结果以 0 分塞进候选字典，污染候选集，不是真正的关闭。
- **混排消融复用 `mix_trace` 的中间态**，不改动 `src/rank/mixer.py`：
  该函数已返回 `base_norm` / `ctr_norm` / `combined` / `sorted_ids` /
  `scattered_ids` / `final_ids`，正好对应五步的各级输出。
- 评估统一走 `src/eval/ab.py:compare_rankers` 的**真实 CTR 口径**，
  不用平滑估计（平滑 CTR 会高估热门基线，见 `src/eval/ab.py` 的模块说明）。

诚实边界
--------
本脚本测量的是**召回与混排对最终排序质量的影响**，属于系统层的组件贡献证据；
它不涉及去偏判据本身（那是 `src/bias/` 的职责，见 `scripts/run_bias_experiments.py`）。
两者刻意分离：`src/bias/` 保持纯 numpy 零重依赖，本脚本是完整档实验。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch 在小模型 + 小 batch 上开满核反而极慢（实测 32 线程 192.8s vs 4 线程 3.8s，
# 差 50 倍——全是线程同步开销）。且本脚本与模型对照脚本可能同时跑，
# 各占满核会互相拖垮（实测并发时 25 分钟仍未跑完 3 个 epoch）。
# 固定为 4 线程：更快，且不改变数值结果（仅影响调度）。
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from src.config import SEED, TOP_K
from src.eval.ab import compare_rankers
from src.rank.mixer import Mixer
from src.recall.baselines import ItemCFRecall, PopularRecall
from src.recall.fusion import RecallFusion
from src.recall.rule import RuleRecall
from src.recall.two_tower import TwoTowerRecall

ROOT = Path(__file__).resolve().parent.parent

# 四路召回的默认权重（与 src/pipeline_demo.py 保持一致：规则 > 向量 > 协同 > 热门）
DEFAULT_WEIGHTS = {"popular": 0.6, "itemcf": 1.0, "two_tower": 1.2, "rule": 1.5}
RECALL_SOURCES = ["popular", "itemcf", "two_tower", "rule"]


# ---------------------------------------------------------------------------
# 数据与模型准备
# ---------------------------------------------------------------------------
def _load_dataset(mode: str):
    """按档位加载数据集，返回统一形态的 (items, users, trips, logs)。

    三个档位共用同一套生成公式与随机种子，只改体量：
      - full  : 10 万 × 10 万 × 500 万曝光（论文口径）
      - lite  :  2 万 ×  2 万 × 100 万曝光
      - micro :  1 万 ×  1 万 ×  50 万曝光（在线免费档）

    ⚠️ 形态差异：full 档返回 pandas DataFrame（`synthetic_generator.build_dataset`），
    lite/micro 档返回 numpy 字典结构（`synthetic_generator_lite.build_dataset_np`）。
    本脚本统一转成 DataFrame 再往下走，避免下游代码分叉。
    """
    import pandas as pd

    if mode == "full":
        from src.data.synthetic_generator import build_dataset
        items, users, trips, logs = build_dataset()
    else:
        from src.data.synthetic_generator_lite import build_dataset_np
        # lite/micro 档返回四个**列式 numpy 字典**（items/users/trips/logs），
        # 这里统一转成 DataFrame，让下游只有一种数据形态。
        it, us, tr, lg = build_dataset_np()
        items = pd.DataFrame(it)
        users = pd.DataFrame(us)
        trips = pd.DataFrame(tr)
        logs = pd.DataFrame(lg)
    return items, users, trips, logs


def _fit_recall(items, users, trips, logs, seed: int = SEED):
    """训练/构建四路召回器。"""
    train_logs = logs if len(logs) <= 1_000_000 else logs.sample(n=1_000_000, random_state=seed)
    pop = PopularRecall(top_k=200).fit(train_logs)
    icf = ItemCFRecall(top_k=200).fit(train_logs)
    rule = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)
    tt = TwoTowerRecall(epochs=1, train_sample=50_000).fit(logs, users, items)
    return {"popular": pop, "itemcf": icf, "two_tower": tt, "rule": rule}


def _fit_ctr_scorer(items, users, logs, seed: int = SEED):
    """训练一个 CTR 打分器，返回 `scorer(uid, item_ids) -> np.ndarray`。

    为什么必须有真实 CTR 分：混排消融要衡量的是「混排如何改变排序」，
    若用流行度之类代理分代替 CTR，则五步都在排同一批候选、差异自然消失
    （实测第一版正是如此，Δ≤0.0005，毫无信息量）。
    这里复用 `src/eval/evaluate.py` 的同一套做法：DeepFM(CTR) + 同一特征库。

    诚实边界：这里的 DeepFM 是**为消融实验而训练的辅助模型**，
    其 AUC 记录在产物 `meta.ctr_model_auc` 里；论文引用主结果时仍用
    `data/eval/end_to_end.json` 的那一次训练。
    """
    import torch
    from src.feature.store import build_candidate_features, build_training_matrix, feature_dim
    from src.rank.trainer import DeepFMRanker
    torch.set_num_threads(4)

    dense_dim, n_sparse, sparse_cards = feature_dim()
    dense, sparse, yc, _ = build_training_matrix(logs, users, items, sample=300_000)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yc))
    tr, te = idx[:240_000], idx[240_000:]

    dfm = DeepFMRanker(dense_dim, sparse_cards, embed_dim=16, dnn_hidden=(64, 32),
                        epochs=3, batch_size=2048)
    dfm.fit(dense[tr], sparse[tr], yc[tr])
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yc[te], dfm.predict_ctr(dense[te], sparse[te])))

    def scorer(uid, item_ids):
        cands = list(item_ids)
        cd, cs = build_candidate_features(int(uid), cands, users, items)
        return dfm.predict_ctr(cd, cs)

    return scorer, auc


def _make_fusion(recallers, user_id, clicked_items, enabled: list[str]) -> RecallFusion:
    """按 enabled 列表注册召回源——**未注册即关闭**。

    注意不能用 weight=0 代替关闭：`_merge` 会把该路物品以 0 分塞进候选字典。
    """
    fusion = RecallFusion(max_candidates=200, top_per_source=100)
    for name in RECALL_SOURCES:
        if name not in enabled:
            continue
        r = recallers[name]
        if name == "popular":
            fusion.add(name, DEFAULT_WEIGHTS[name], lambda k, r=r: r.recall(k=k))
        elif name == "itemcf":
            fusion.add(name, DEFAULT_WEIGHTS[name],
                       lambda k, r=r, c=clicked_items: r.recall(list(c), k=k))
        elif name == "two_tower":
            fusion.add(name, DEFAULT_WEIGHTS[name],
                       lambda k, r=r, u=user_id: r.recall(int(u), k=k))
        else:  # rule
            fusion.add(name, DEFAULT_WEIGHTS[name],
                       lambda k, r=r, u=user_id: r.recall(int(u), k=k))
    return fusion


# ---------------------------------------------------------------------------
# 消融一：召回通道
# ---------------------------------------------------------------------------
def run_recall_ablation(items, users, trips, logs, recallers, sample_users, seed: int,
                        ctr_scorer):
    """逐路关闭召回，测真实 CTR 口径下的表现变化。

    ⚠️ 评估设计说明（第一版曾出过错，这里记录避免后人重蹈）：
    若把召回结果**直接按流行度排序**再评估，会得到「仅热门路最优」的荒谬结论——
    因为热门路本来就是按流行度排的，且它会独占整个候选集。
    正确做法是**所有消融组共用同一套下游排序与混排**，只改候选池的构成，
    这样 Δ 才反映「这一路召回带来了什么」，而不是「哪种排序方式讨巧」。

    组设计：
      - 「全四路」：基线
      - 「仅 X」：只有 X 一路
      - 「去掉 X」：四路里去掉 X（其余三路不变）—— 与基线配对比较
    """
    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    pop_map = items.set_index("item_id")["popularity"].to_dict()
    cat_col = "category_idx" if "category_idx" in items.columns else "category"
    cat_map = items.set_index("item_id")[cat_col].to_dict()
    mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=TOP_K)

    def make_ranker(enabled):
        def ranker(uid, k):
            clicked = clicked_by_user.get(uid, [])
            fusion = _make_fusion(recallers, uid, clicked, enabled)
            cands = fusion.recall()
            if len(cands) < k:
                return cands[:k]
            base = [pop_map.get(i, 0.0) for i in cands]
            cats = [cat_map.get(i, "") for i in cands]
            ctr = ctr_scorer(uid, cands)
            return mixer.mix(cands, base, ctr, cats)[:k]
        return ranker

    groups = {"全四路": list(RECALL_SOURCES)}
    for name in RECALL_SOURCES:
        groups[f"仅{name}"] = [name]
        groups[f"去掉{name}"] = [x for x in RECALL_SOURCES if x != name]

    rankers = {label: make_ranker(enabled) for label, enabled in groups.items()}
    res = compare_rankers(users, items, rankers=rankers,
                          sample_users=sample_users, top_k=TOP_K, seed=seed, verbose=False)
    return res, groups


# ---------------------------------------------------------------------------
# 消融二：混排五步
# ---------------------------------------------------------------------------
def run_mixer_ablation(items, users, logs, recallers, sample_users, seed: int,
                       ctr_scorer):
    """逐步去除混排步骤，测排序质量变化。

    五步与 `Mixer.mix_trace` 的对应：
      1. minmax 归一化   —— 无开关（`_minmax` 是模块内私有函数），
                           本脚本通过「不做归一化」等价实现：直接用原始分排序
      2. CTR 加权        —— `w_ctr=0` 即关闭（退化为纯基础分排序）
      3. 分层排序        —— 取 `combined` 自行降序
      4. 同品类打散      —— 取 `sorted_ids`
      5. 去重+截断      —— 取 `scattered_ids`

    ⚠️ 候选池固定为**全四路召回**（不随消融变化），以隔离混排步骤的单一效应。
    """
    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    pop_map = items.set_index("item_id")["popularity"].to_dict()
    cat_col = "category_idx" if "category_idx" in items.columns else "category"
    cat_map = items.set_index("item_id")[cat_col].to_dict()
    mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=TOP_K)
    base_only = Mixer(w_base=1.0, w_ctr=0.0, max_same_category=2, top_k=TOP_K)

    def make_ranker(level: str):
        """level: all / no_scatter / no_dedup / no_ctr / no_normalize / no_mix"""
        def ranker(uid, k):
            clicked = clicked_by_user.get(uid, [])
            fusion = _make_fusion(recallers, uid, clicked, RECALL_SOURCES)
            cands = fusion.recall()
            if len(cands) < k:
                return cands[:k]
            base = [pop_map.get(i, 0.0) for i in cands]
            cats = [cat_map.get(i, "") for i in cands]
            ctr = ctr_scorer(uid, cands)

            if level == "no_mix":
                # 完全不混排：按 CTR 原始分降序（隔离「混排整体」的贡献）
                return [i for _, i in sorted(zip(ctr, cands), key=lambda t: -t[0])][:k]
            if level == "no_normalize":
                # 关第 1 步（minmax 归一化）：直接用原始分做凸组合，不缩放到同一量纲
                comb = [0.5 * b + 0.5 * c for b, c in zip(base, ctr)]
                return [i for _, i in sorted(zip(comb, cands), key=lambda t: -t[0])][:k]
            if level == "no_ctr":
                # 关第 2 步（CTR 加权）：只保留基础分
                tr = base_only.mix_trace(cands, base, ctr, cats)
                return tr["final_ids"][:k]
            tr = mixer.mix_trace(cands, base, ctr, cats)
            if level == "no_scatter":       # 关第 4 步
                return tr["sorted_ids"][:k]
            if level == "no_dedup":         # 关第 5 步
                return tr["scattered_ids"][:k]
            return tr["final_ids"][:k]      # all
        return ranker

    levels = {
        "完整五步": "all",
        "关_同品类打散": "no_scatter",
        "关_去重截断": "no_dedup",
        "关_CTR加权": "no_ctr",
        "关_归一化": "no_normalize",
        "完全不混排": "no_mix",
    }
    rankers = {label: make_ranker(lv) for label, lv in levels.items()}
    res = compare_rankers(users, items, rankers=rankers,
                          sample_users=sample_users, top_k=TOP_K, seed=seed, verbose=False)
    return res, levels


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="召回通道与混排步骤的消融实验")
    ap.add_argument("--mode", default="micro", choices=["micro", "lite", "full"])
    ap.add_argument("--n-users", type=int, default=100, help="参与评估的用户数")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None,
                    help="输出路径。默认 data/eval/ablation_<mode>.json——"
                         "按档位分文件存放，避免 full 档覆盖 micro 档后"
                         "论文里多档对照失去产物支撑")
    args = ap.parse_args()

    print(f"[ablation] 载入 {args.mode} 档数据集 ...")
    items, users, trips, logs = _load_dataset(args.mode)
    print(f"[ablation] items={len(items)} users={len(users)} logs={len(logs)}")

    print("[ablation] 构建四路召回器（Two-Tower 需训练，约 1–2 分钟）...")
    recallers = _fit_recall(items, users, trips, logs, seed=args.seed)

    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    sample_users = [int(u) for u in clicked_by_user.index[: args.n_users]]
    print(f"[ablation] 评估用户数：{len(sample_users)}")

    print("[ablation] 训练 CTR 打分器（DeepFM，约 5–10 分钟）...")
    ctr_scorer, ctr_auc = _fit_ctr_scorer(items, users, logs, seed=args.seed)
    print(f"[ablation] CTR 模型 AUC = {ctr_auc:.4f}")

    print("[ablation] 召回通道消融 ...")
    rec_res, rec_groups = run_recall_ablation(
        items, users, trips, logs, recallers, sample_users, args.seed, ctr_scorer)

    print("[ablation] 混排步骤消融 ...")
    mix_res, mix_levels = run_mixer_ablation(
        items, users, logs, recallers, sample_users, args.seed, ctr_scorer)

    # ---- 汇总 ----
    def _pack(res: dict) -> dict:
        """把 compare_rankers 的返回整理成可落盘的扁平结构。

        注意：`compare_rankers` 把 Oracle 放在 `per_method["Oracle"]` 里，
        没有顶层 `oracle` 键；`paired_lift_vs_oracle` 是与 Oracle 的逐用户配对差，
        比 `pct_of_oracle` 更严格，一并记录。
        """
        return {
            "per_method": {
                k: {
                    "true_ctr": round(float(v["true_ctr"]), 4),
                    "rel_to_base": round(float(v["rel_to_base"]), 4),
                    "pct_of_oracle": round(float(v["pct_of_oracle"]), 4),
                    "paired_lift_vs_oracle": round(float(v["paired_lift_vs_oracle"]), 4),
                    # 端到端 graded NDCG@K——真实 CTR 只看"点击了多少"，
                    # NDCG 还看"好的排前面了吗"，两者一起才能区分
                    # 「候选池变好」与「排序变好」。此前只打印不落盘，
                    # 导致论文里引用 NDCG 时拿不出产物（A 级证据缺口）。
                    "ndcg_at_k": round(float(v["ndcg_at_k"]), 4),
                    "ndcg_rel_to_base": round(float(v["ndcg_rel_to_base"]), 4),
                    "ndcg_pct_of_oracle": round(float(v["ndcg_pct_of_oracle"]), 4),
                }
                for k, v in res["per_method"].items()
            },
            "oracle_true_ctr": round(float(res["per_method"]["Oracle"]["true_ctr"]), 4),
            "oracle_ndcg_at_k": round(float(res["oracle_ndcg_at_k"]), 4),
        }

    payload = {
        "meta": {
            "description": "召回通道与混排步骤的消融实验（真实 CTR 口径）",
            "mode": args.mode,
            "seed": args.seed,
            "top_k": TOP_K,
            "n_users": len(sample_users),
            "protocol": (
                "统一使用 src/eval/ab.py:compare_rankers 的真实 CTR 口径；"
                "召回关闭=不注册该路（非 weight=0）；混排消融复用 Mixer.mix_trace 的中间态。"
            ),
            "baseline_note": "全四路 / 完整五步为基线组，其余为消融组。",
            "ndcg_note": (
                "NDCG@K 的增益 = 物品在候选池内的真实 CTR，IDCG 分母取**同一候选池**"
                "降序前 K。所以：**同一次 compare_rankers 调用内的各组可比，"
                "跨调用不可横比**。推论——"
                "① 召回消融各组候选池不同（仅itemcf 的池 ≠ 全四路的池）→ 分母不同，"
                "横比 NDCG 无意义，召回结论请以 true_ctr 为准（true_ctr 直接查真实 CTR，"
                "与池无关）；"
                "② 混排消融候选池固定为全四路（见 run_mixer_ablation 注释）→ 分母相同，"
                "NDCG 可横向比较，且能看出『CTR 高的组 NDCG 未必高』。"
            ),
            "ctr_model_auc": round(float(ctr_auc), 4),
            "ctr_model_note": (
                "消融实验专用的辅助 CTR 模型（DeepFM），与 src/eval/evaluate.py 同法训练。"
                "论文主结果仍引用 data/eval/end_to_end.json 的那次训练。"
            ),
            "scale_note": (
                "micro/lite 档仅用于验证消融流程与相对关系，数值不代表论文口径；"
                "论文引用须以 full 档为准。"
                if args.mode != "full" else "full 档为论文口径。"
            ),
        },
        "recall_ablation": {
            "groups": rec_groups,
            **_pack(rec_res),
        },
        "mixer_ablation": {
            "levels": mix_levels,
            **_pack(mix_res),
        },
    }

    out_path = Path(args.out) if args.out else \
        ROOT / "data" / "eval" / f"ablation_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # ---- 打印摘要 ----
    print("\n" + "=" * 68)
    print("① 召回通道消融（真实 CTR 口径）")
    print("-" * 68)
    base = rec_res["per_method"]["全四路"]["true_ctr"]
    for k, v in rec_res["per_method"].items():
        d = v["true_ctr"] - base
        print(f"  {k:16s} true_ctr={v['true_ctr']:.4f}  Δ={d:+.4f}  "
              f"占Oracle={v['pct_of_oracle']:.1%}")
    print("=" * 68)
    print("② 混排步骤消融（真实 CTR 口径）")
    print("-" * 68)
    base2 = mix_res["per_method"]["完整五步"]["true_ctr"]
    for k, v in mix_res["per_method"].items():
        d = v["true_ctr"] - base2
        print(f"  {k:16s} true_ctr={v['true_ctr']:.4f}  Δ={d:+.4f}  "
              f"占Oracle={v['pct_of_oracle']:.1%}")
    print("=" * 68)
    print(f"[ablation] 结果已落盘：{out_path}（{out_path.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
