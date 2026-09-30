"""模拟在线 A/B：控制组 vs 实验组各自独立曝光，观测点击/转化并检验显著性。

诚实边界：这是**离线模拟** A/B（用潜在真实 CTR/CVR 模型模拟用户反馈），
不是真实线上流量实验——README 已明确标注。
"""
import math

import numpy as np

from src.data.synthetic_generator import compute_true_ctr, compute_true_cvr


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
