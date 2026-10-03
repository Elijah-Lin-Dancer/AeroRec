"""在线 Demo 模型工件构建脚本（离线执行，需要 torch / scikit-learn）。

背景
----
在线免费实例（Streamlit Community Cloud）内存上限 1GB，而框架的**固有导入开销**是：

    裸 Python   9MB  →  +numpy 25MB  →  +sklearn 176MB  →  +torch 643MB  →  +streamlit 676MB

torch 一项就吃掉 643MB，与数据规模无关。因此在线侧**不能 import torch**
（也不能 import sklearn，省 151MB）。

做法
----
本脚本在**构建期**（本机/CI，内存不受限）完成全部训练：

  1. 生成 micro 档数据；
  2. 训练 DeepFM(CTR) 与 LR(CVR)；
  3. 把 DeepFM 权重导出为 numpy（`src/rank/numpy_deepfm.py`）；
  4. 把 LR(CVR) 的逻辑回归系数导出为纯 numpy（`z = X·coef + intercept`）；
  5. 把 OneHot 编码类别、特征库所需统计量一并落盘；
  6. 全部写入 `artifacts/micro/`，供在线侧在**无 torch / 无 sklearn** 环境加载。

产物
----
    artifacts/micro/deepfm_ctr.npz   DeepFM 权重（numpy，约 28KB）
    artifacts/micro/cvr_lr.npz       LR(CVR) 系数 + OneHot 类别（约 1KB）
    artifacts/micro/meta.json        数据规模、种子、特征维度等元信息

运行
----
    python -m src.build_online_artifacts micro
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def main(mode: str = "micro") -> None:
    import os
    os.environ["AEROREC_MODE"] = mode

    t0 = time.time()
    import numpy.random as npr
    import importlib

    # 必须在设置 AEROREC_MODE 之后再导入配置解析器
    import src.config_scale as cs
    importlib.reload(cs)

    from src.data.synthetic_generator_lite import build_dataset_np
    from src.feature.lite_store import LiteFeatureStore, SPARSE_CARD, N_DENSE
    from src.rank.trainer import DeepFMRanker
    from src.rank.numpy_deepfm import export_from_ranker, save_npz
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    print(f"[build] mode={mode}  N_USERS={cs.N_USERS}  N_ITEMS={cs.N_ITEMS}  "
          f"曝光={cs.N_USERS * cs.IMPRESSIONS_PER_USER}")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 数据 ----------
    items, users, trips, logs = build_dataset_np(npr.default_rng(cs.SEED))
    fs = LiteFeatureStore(items, users)
    print(f"[build] 数据生成完毕 {time.time() - t0:.1f}s")

    n = len(logs["clicked"])
    # 训练样本量：取曝光量的 8%，上下界 40k / 200k
    train_sample = int(min(200_000, max(40_000, n * 0.08)))
    train_sample = min(train_sample, n)
    print(f"[build] train_sample={train_sample}（占全量 {train_sample / n:.1%}）")

    idx = npr.default_rng(cs.SEED).choice(n, size=train_sample, replace=False)
    dense = fs.build_matrix(logs["user_id"][idx], logs["item_id"][idx],
                            logs["is_weekend"][idx])
    sparse = fs.build_sparse(logs["item_id"][idx], logs["hour_slot"][idx])
    y = logs["clicked"][idx].astype(np.int64)

    # ---------- 2. 训练 DeepFM(CTR) ----------
    t1 = time.time()
    dfm = DeepFMRanker(N_DENSE, SPARSE_CARD, embed_dim=16, dnn_hidden=(64, 32),
                       epochs=3, batch_size=2048).fit(dense, sparse, y)
    print(f"[build] DeepFM 训练完毕 {time.time() - t1:.1f}s")

    # ---------- 3. 导出 DeepFM -> numpy ----------
    params = export_from_ranker(dfm)
    save_npz(params, out_dir / "deepfm_ctr.npz")
    print(f"[build] DeepFM 权重已导出 {(out_dir / 'deepfm_ctr.npz').stat().st_size / 1024:.1f} KB")

    # ---------- 4. 训练 LR(CVR) ----------
    t2 = time.time()
    clk = np.where(logs["clicked"] > 0)[0][:min(train_sample, n)]
    d2 = fs.build_matrix(logs["user_id"][clk], logs["item_id"][clk],
                         logs["is_weekend"][clk])
    s2 = fs.build_sparse(logs["item_id"][clk], logs["hour_slot"][clk])
    enc = OneHotEncoder(categories=[range(c) for c in SPARSE_CARD], sparse_output=False)
    ohe = enc.fit_transform(s2)
    lr = LogisticRegression(max_iter=1000).fit(np.hstack([d2, ohe]),
                                               logs["converted"][clk].astype(np.int64))
    print(f"[build] LR(CVR) 训练完毕 {time.time() - t2:.1f}s，点击样本 {len(clk)}")

    # ---------- 5. 导出 LR(CVR) -> numpy ----------
    np.savez_compressed(
        out_dir / "cvr_lr.npz",
        coef=lr.coef_.astype(np.float32),
        intercept=lr.intercept_.astype(np.float32),
        ohe_offsets=np.cumsum([0] + [len(range(c)) for c in SPARSE_CARD]).astype(np.int64),
        ohe_n_features=np.array([int(ohe.shape[1])]),
    )
    print(f"[build] LR(CVR) 系数已导出 {(out_dir / 'cvr_lr.npz').stat().st_size / 1024:.1f} KB")

    # ---------- 6. 元信息 ----------
    meta = {
        "mode": mode,
        "n_users": int(cs.N_USERS),
        "n_items": int(cs.N_ITEMS),
        "n_impressions": int(cs.N_USERS * cs.IMPRESSIONS_PER_USER),
        "train_sample": int(train_sample),
        "seed": int(cs.SEED),
        "n_dense": int(N_DENSE),
        "sparse_card": [int(c) for c in SPARSE_CARD],
        "embed_dim": 16,
        "dnn_hidden": [64, 32],
        "deepfm_epochs": 3,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "由 src/build_online_artifacts.py 离线生成；在线侧以纯 numpy 加载，不依赖 torch/sklearn。",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 7. 自检：导出后 numpy 前向是否与 torch 一致 ----------
    from src.rank.numpy_deepfm import NumpyDeepFM
    chk = min(2000, len(idx))
    d_chk, s_chk = dense[:chk], sparse[:chk]
    diff = float(np.max(np.abs(dfm.predict_ctr(d_chk, s_chk) -
                               NumpyDeepFM(params).predict_ctr(d_chk, s_chk))))
    print(f"[build] 数值自检：PyTorch vs numpy 最大偏差 = {diff:.2e}  "
          f"{'✓ 通过' if diff < 1e-4 else '✗ 异常'}")
    print(f"[build] 全部完成，用时 {time.time() - t0:.1f}s → {out_dir}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "micro")
