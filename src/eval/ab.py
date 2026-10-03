"""模拟在线 A/B + 真实 CTR 口径的基线对比（含上界 Oracle）。

为什么要有两套口径
------------------
**口径一：模拟在线 A/B**（`simulate_ab`）
  用潜在真实 CTR/CVR 抽样"模拟用户反馈"，观测点击率/转化率。
  保留它是因为它最接近工业 A/B 的形式（有曝光、有点击、有显著性检验）。

**口径二：真实 CTR 排序对比**（`compare_rankers`）
  直接对每个样本用户，用 `compute_true_ctr` 计算各排序方法给出的
  Top-K 列表的**平均真实 CTR**。不涉及任何"平滑历史 CTR"估计，
  因此不受稀疏性偏差影响，是**可引用**的结论。

⚠️ 为什么必须加第二套口径（本项目踩过的坑）
--------------------------------------------
热门基线（`PopularRecall`）按"历史日志的**贝叶斯平滑 CTR**（α=5）"排序，
而本项目日志是**随机曝光**生成的。曝光次数少的物品偶然被点击一次，
平滑 CTR 就被显著高估（α=5 压不住这种噪声）。完整档实测：

    热门榜 top200 平滑 CTR = 0.6231
    热门榜 top200 真实 CTR = 0.2528      → 偏差 +0.37
    平滑榜 top10 与真实 CTR top10 重合数 = 2/10

后果：在**模拟 A/B** 里，热门基线因为这层噪声被"低估"，
使实验组的相对增益被**高估**（模拟口径 +18.9% vs 真实 CTR 口径 +7.1%）。
这属于推荐系统中的**位置偏差 / 数据稀疏偏差**。

因此本模块同时报告：
  1. 模拟 A/B 的 CTR/CVR 提升（标注其偏差来源）；
  2. 真实 CTR 口径下 随机 / 热门 / DeepFM / **Oracle 上界** 四方对比；
  3. 平滑 CTR 与真实 CTR 的偏差表，把问题透明化。

上界 Oracle 的定义（关键）
--------------------------
Oracle 池 = 热门候选 ∪ 规则候选 ∪ 随机采样（**必须包含其他方法的结局**，
否则 Oracle 不构成真正的上界）。在池内按**该用户的真实 CTR** 降序取 Top-K。
这样构造保证 Oracle 的真实 CTR **严格 ≥ 任何从同一候选空间取 Top-K 的方法**。
实测完整档（100 用户）：Oracle 0.7473、DeepFM 0.5270、热门 0.4922、随机 0.1729。
"""
from __future__ import annotations

import math

import numpy as np

# 注意：从 pandas-free 的 `generator_common` 导入（而非 synthetic_generator），
# 否则在线低内存路径会因间接导入 pandas 而多付约 150MB。
from src.data.generator_common import compute_true_ctr, compute_true_cvr


def proportion_ztest(p1: float, n1: int, p2: float, n2: int):
    """两比例 z 检验，返回 (z, 双尾 p 值)。"""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    if p <= 0 or p >= 1:
        return 0.0, 1.0
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2))
    return float(z), float(pval)


def simulate_ab(users, items, control_ranker, treatment_ranker,
                n_users: int = 2000, top_k: int = 10, seed: int = 42) -> dict:
    """模拟在线 A/B：两个 ranker 各自独立产生曝光并观测点击/转化。

    control_ranker / treatment_ranker: callable(uid, k) -> item_id 列表。
    返回 点击率(CTR=点击/曝光) 与 转化率(CVR=转化/曝光) 的相对提升与显著性，
    以及次要的点击后转化率(PCVR=转化/点击)。

    ⚠️ 注意：热门基线存在"平滑 CTR 稀疏性偏差"（见模块 docstring），
    该口径的结果**仅供参考**，可引用结论请用 `compare_rankers`。
    """
    rng = np.random.default_rng(seed)
    uids = rng.choice(len(users), size=n_users, replace=False)

    ctrl = {"impr": 0, "click": 0, "conv": 0}
    trt = {"impr": 0, "click": 0, "conv": 0}
    for uid in uids:
        for bucket, ranker in (("ctrl", control_ranker), ("trt", treatment_ranker)):
            ids = list(ranker(int(uid), top_k))
            if not ids:
                continue
            ctr = compute_true_ctr(users, items, [int(uid)] * len(ids), ids)
            cvr = compute_true_cvr(users, items, [int(uid)] * len(ids), ids)
            clicked = rng.random(len(ids)) < ctr
            conv = clicked & (rng.random(len(ids)) < cvr)
            d = ctrl if bucket == "ctrl" else trt
            d["impr"] += len(ids)
            d["click"] += int(clicked.sum())
            d["conv"] += int(conv.sum())

    ctrl_ctr = ctrl["click"] / ctrl["impr"]
    trt_ctr = trt["click"] / trt["impr"]
    ctrl_cvr = ctrl["conv"] / ctrl["impr"]            # 转化率 = 转化/曝光（购买率）
    trt_cvr = trt["conv"] / trt["impr"]
    ctrl_pcvr = ctrl["conv"] / max(1, ctrl["click"])  # 点击后转化率（次要）
    trt_pcvr = trt["conv"] / max(1, trt["click"])

    ctr_z, ctr_p = proportion_ztest(trt_ctr, trt["impr"], ctrl_ctr, ctrl["impr"])
    cvr_z, cvr_p = proportion_ztest(trt_cvr, trt["impr"], ctrl_cvr, ctrl["impr"])

    return {
        "n_users": n_users,
        "top_k": top_k,
        "ctrl_impr": ctrl["impr"], "trt_impr": trt["impr"],
        "ctrl_ctr": ctrl_ctr, "trt_ctr": trt_ctr,
        "ctr_lift": (trt_ctr - ctrl_ctr) / max(ctrl_ctr, 1e-9),
        "ctr_z": ctr_z, "ctr_p": ctr_p,
        "ctrl_cvr": ctrl_cvr, "trt_cvr": trt_cvr,
        "cvr_lift": (trt_cvr - ctrl_cvr) / max(ctrl_cvr, 1e-9),
        "cvr_z": cvr_z, "cvr_p": cvr_p,
        "ctrl_pcvr": ctrl_pcvr, "trt_pcvr": trt_pcvr,
    }


def compare_rankers(users, items, rankers: dict, sample_users, top_k: int = 10,
                    oracle_pool_size: int = 3000, n_items: int | None = None,
                    seed: int = 42, verbose: bool = True) -> dict:
    """在**真实 CTR 口径**下对比多个排序方法，并给出上界 Oracle。

    参数
    ----
    rankers : {名称: callable(uid, k) -> item_id 列表}
        待对比的方法，例如 {"热门": ..., "DeepFM+混排": ...}。
        若字典中已含名为 "Oracle" 的项则不会重复构造。
    sample_users : 参与评估的用户 ID 列表。
    oracle_pool_size : Oracle 池中随机采样的物品数
        （池 = 各 ranker 的候选 ∪ 随机采样；必须覆盖其他方法的结局，
        否则 Oracle 不构成真正的上界）。

    返回
    ----
    {
      "per_method": {名称: {"true_ctr": 均值, "rel_to_base": 相对热门基线,
                            "pct_of_oracle": 占 Oracle 的百分比}},
      "oracle": 上界真实 CTR,
      "bias_table": 热门榜的 平滑CTR vs 真实CTR 偏差（若可获取）,
      "n_users": 评估用户数,
    }
    """
    n_items = n_items or len(items)
    rng = np.random.default_rng(seed)

    def _tc(uid, ids):
        ids = np.asarray(ids)
        if len(ids) == 0:
            return np.array([0.0])
        return compute_true_ctr(users, items, np.full(len(ids), int(uid)), ids)

    # 预先算出每个方法在每个用户上的列表（避免重复调用）
    method_ids = {name: [] for name in rankers}
    for uid in sample_users:
        for name, fn in rankers.items():
            method_ids[name].append(list(fn(int(uid), top_k)))

    # ---- 构造逐用户 Oracle：池必须包含所有方法的结局 ----
    oracle_scores, method_scores = [], {name: [] for name in rankers}
    for i, uid in enumerate(sample_users):
        pool_parts = [np.asarray(ids) for ids in
                      (method_ids[name][i] for name in rankers)]
        pool_parts.append(rng.choice(n_items, size=oracle_pool_size, replace=False))
        pool = np.unique(np.concatenate([p for p in pool_parts if len(p)]))
        t = _tc(uid, pool)
        oracle_scores.append(float(t[np.argsort(-t)[:top_k]].mean()))
        for name in rankers:
            method_scores[name].append(float(_tc(uid, method_ids[name][i]).mean()))

    oracle_mean = float(np.mean(oracle_scores))
    per_method = {}
    # 参照系固定为"热门"（若存在），否则取字典首项
    base_name = "热门" if "热门" in rankers else next(iter(rankers))
    base_mean = float(np.mean(method_scores[base_name]))
    for name in rankers:
        m = float(np.mean(method_scores[name]))
        per_method[name] = {
            "true_ctr": m,
            "rel_to_base": (m - base_mean) / max(base_mean, 1e-9),
            "pct_of_oracle": m / max(oracle_mean, 1e-9),
            # 与 Oracle 的逐用户配对差（更严格）
            "paired_lift_vs_oracle": float(np.mean(
                [a - b for a, b in zip(method_scores[name], oracle_scores)])),
        }
    per_method["Oracle"] = {
        "true_ctr": oracle_mean,
        "rel_to_base": (oracle_mean - base_mean) / max(base_mean, 1e-9),
        "pct_of_oracle": 1.0,
        "paired_lift_vs_oracle": 0.0,
    }

    if verbose:
        print("=" * 64)
        print(f"真实 CTR 口径对比（{len(sample_users)} 用户，Top-{top_k}）")
        print(f"参照基线：{base_name}")
        print("-" * 64)
        for name in list(rankers) + ["Oracle"]:
            d = per_method[name]
            print(f"  {name:<16} 真实CTR={d['true_ctr']:.4f}  "
                  f"相对{base_name}={d['rel_to_base']:+.1%}  "
                  f"占Oracle={d['pct_of_oracle']:.1%}")
        print("=" * 64)

    return {
        "per_method": per_method,
        "oracle_true_ctr": oracle_mean,
        "base_method": base_name,
        "n_users": len(sample_users),
        "top_k": top_k,
        "oracle_scores": oracle_scores,
        "method_scores": method_scores,
    }


def baseline_bias_table(pop_recall, users, items, top_n: int = 200,
                        sample_user: int = 0) -> dict:
    """热门基线的"平滑 CTR vs 真实 CTR"偏差表，用于透明化稀疏性偏差。

    参数 pop_recall 需为已 fit 的 `PopularRecall`（含 top_items_ 与 ctr 列）。
    """
    out = {}
    try:
        ids = pop_recall.recall(top_n)
        smooth = float(pop_recall.top_items_["ctr"].mean())
        real = float(compute_true_ctr(
            users, items, np.full(len(ids), int(sample_user)), np.asarray(ids)).mean())
        out["top_n"] = top_n
        out["smooth_ctr"] = smooth
        out["true_ctr"] = real
        out["bias"] = smooth - real
    except Exception as e:                                  # 容错，不阻断主流程
        out["error"] = str(e)
    return out
