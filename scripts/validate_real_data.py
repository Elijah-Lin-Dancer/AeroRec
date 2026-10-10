"""真实数据验证脚本：用真实公开数据集验证同一个 DeepFM 能学到真实信号。

为什么保留这个脚本
------------------
本项目的合成指标（AUC、真实 CTR 提升）都是在**自己的合成分布**上算的，
一个合理的追问是「换个分布还成立吗」。这里用一份**真实、公开、可引用**的数据集
（UCI Adult / Census Income）替换特征库，**模型与训练代码零改动**，
检验同一份 DeepFM 实现能否在真实数据上学到信号。

> 说明：这不是「在出行真实流量上验证过」。本脚本的定位是
> **排除「模型实现本身有问题」这一可能**，而非证明合成指标可外推。
> 见 README「已知局限」。

UCI Adult 与 CTR 任务同构：
  - 6 连续（dense）+ 8 类别（sparse）特征；
  - 正类率约 24%，与 CTR 的类别不均衡量级接近；
  - 任务 p(income>50K) 与 p(click) 完全同构。

运行：
    python scripts/validate_real_data.py

产出：
    data/real_validation/result.json   实验元信息 + 指标（AUC / 基线对照 / 环境）
    stdout                             人类可读摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import sklearn  # noqa: E402

# 线程数必须显式限制：本沙箱容器 CPU 配额只有 4 核，而 torch 默认按物理核数（32）开线程，
# 严重超订会让每条小算子陷入上下文切换、单步耗时飙到秒级（实测 tiny 模型可达 4s/step）。
# AEROREC_THREADS 可控，默认取 4。
_THREADS = int(os.environ.get("AEROREC_THREADS", "4"))
torch.set_num_threads(_THREADS)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from src.rank.trainer import (DEEFM_BATCH_SIZE, DeepFMRanker,  # noqa: E402
                              plan_epochs)


def lr_baseline(dtr, str_, ytr, dva, sva, yva):
    """线性对照基线：dense + 稀疏 one-hot 的 LogisticRegression。"""
    enc = OneHotEncoder(handle_unknown="ignore")
    ohe_tr = enc.fit_transform(str_)
    ohe_va = enc.transform(sva)
    Xtr = np.hstack([dtr, ohe_tr.toarray()])
    Xva = np.hstack([dva, ohe_va.toarray()])
    clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xtr, ytr)
    return float(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))


def main():
    ap = argparse.ArgumentParser(description="真实数据验证 DeepFM 迁移性")
    args = ap.parse_args()

    t0 = time.time()
    from src.data.adult import load_adult  # noqa: E402
    data = load_adult()
    meta = data["meta"]
    print(f"[validate] 数据集={meta['dataset']}  训练={meta['n_train']}  验证={meta['n_val']}  "
          f"正类率(val)={meta['positive_rate_val']}")
    print(f"[validate] dense_dim={data['n_dense']}  sparse_cards={data['sparse_cards']}")

    # ---- 同一套 DeepFM（与合成链路完全一致的实现）----
    epochs = plan_epochs(len(data["y_train"]), DEEFM_BATCH_SIZE)
    n_updates = ((len(data["y_train"]) + DEEFM_BATCH_SIZE - 1) // DEEFM_BATCH_SIZE) * epochs
    print(f"[validate] DeepFM 训练：{len(data['y_train'])} 样本 × {epochs} epoch / "
          f"batch {DEEFM_BATCH_SIZE} → {n_updates} 次梯度更新")

    ranker = DeepFMRanker(
        data["n_dense"], data["sparse_cards"],
        embed_dim=16, dnn_hidden=(64, 32),
        epochs=epochs, batch_size=DEEFM_BATCH_SIZE,
    ).fit(data["dense_train"], data["sparse_train"], data["y_train"],
          val_dense=data["dense_val"], val_sparse=data["sparse_val"], val_y=data["y_val"])

    auc_deepfm = float(roc_auc_score(data["y_val"], ranker.predict_ctr(data["dense_val"], data["sparse_val"])))
    auc_lr = lr_baseline(data["dense_train"], data["sparse_train"], data["y_train"],
                         data["dense_val"], data["sparse_val"], data["y_val"])
    auc_naive = 0.5  # 常数预测

    result = {
        **meta,
        "model": "DeepFM (Guo et al. 2017) — 与本项目合成链路同一实现",
        "deepfm_epochs": int(epochs),
        "deepfm_batch_size": int(DEEFM_BATCH_SIZE),
        "deepfm_updates": int(n_updates),
        "val_auc_deepfm": round(auc_deepfm, 4),
        "val_auc_logistic_baseline": round(auc_lr, 4),
        "val_auc_naive_constant": auc_naive,
        "lift_over_naive": round(auc_deepfm - auc_naive, 4),
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch_threads": _THREADS,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join("data", "real_validation", "result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n==================== 真实数据验证结果 ====================")
    print(f"  数据集             : {result['dataset']}")
    print(f"  训练 / 验证样本     : {result['n_train']} / {result['n_val']}")
    print(f"  验证集正类率        : {result['positive_rate_val']}")
    print(f"  DeepFM 验证 AUC     : {auc_deepfm:.4f}")
    print(f"  LR 基线 验证 AUC     : {auc_lr:.4f}")
    print(f"  常数基线 AUC         : {auc_naive:.4f}")
    print(f"  DeepFM 相对常数提升   : {auc_deepfm - auc_naive:+.4f}")
    print(f"  训练预算(更新次数)    : {n_updates}")
    print(f"  耗时                : {result['elapsed_sec']}s")
    print(f"  结果已写入           : {out_path}")
    print("=======================================================")
    if auc_deepfm > 0.6:
        print("  ✓ 迁移性验证通过：同一 DeepFM 在真实数据上学到了显著信号（AUC >> 0.5）。")
    else:
        print("  ⚠ AUC 偏低，需检查预处理或训练预算。")


if __name__ == "__main__":
    main()
