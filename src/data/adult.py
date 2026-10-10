"""UCI Adult / Census Income 真实数据加载器 —— 用于路径 B：验证 DeepFM 在真实 (dense, sparse) 数据上的迁移性。

背景
----
路径 B 原计划用 Criteo Display Advertising Challenge 验证 DeepFM 迁移性；但构建沙箱网络
无法访问 Criteo（官方 S3 / go.criteo.net / HuggingFace 全部被阻断）。
UCI Adult 是一个真实、公开、可引用的二元分类数据集，结构与 CTR 任务**同构**：

  - 6 个连续（dense）特征 + 8 个类别（sparse）特征；
  - 正类率 ~24%，与真实 CTR 的类别不均衡（Criteo ~25%）接近；
  - 任务：预测收入是否 >50K，即 p(positive) —— 与 CTR 的 p(click) 完全同构。

因此用它作为「真实 (dense_dim, sparse_cards) -> 二元」基准，验证同一个 DeepFM
能学到真实信号。本加载器**与数据集无关**：只要产出 (dense, sparse, sparse_cards, y)
四元组，就能替换为 Criteo（见文件底部 `load_criteo` 接入点说明）。

许可证：UCI Adult 来源于 UCI ML Repository（Becker & Kohavi, 1996；Donoho, 2019），
研究用途自由使用，需引用。Criteo 原始数据集许可证为 CC-BY-NC-SA 4.0。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# UCI Adult 字段 schema（15 列，无表头；逗号分隔，字段前可能带空格）
# 真实文件中 dense / sparse 是**交错**排列的，必须按位置命名，再用名字挑列。
# ---------------------------------------------------------------------------
DENSE_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
SPARSE_COLS = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
LABEL_COL = "income"
# 按 UCI adult.data 真实列顺序（位置 -> 字段名）
REAL_ORDER = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
COLUMNS = REAL_ORDER

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
FILES = {"train": "adult.data", "test": "adult.test"}


def _download(name: str, cache_dir: Path) -> Path:
    dst = cache_dir / FILES[name]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{UCI_BASE}/{FILES[name]}"
    print(f"[adult] 下载 {name}: {url}")
    urllib.request.urlretrieve(url, dst)        # noqa: S310 - 固定可信 UCI 地址
    return dst


def _parse(path: Path) -> pd.DataFrame:
    # 文件中每行末尾可能有多余的 "."（test 集标签带 "."），且字段前偶有空格
    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True,
                     na_values=["?"], keep_default_na=True)
    # 清洗标签：去掉可能的尾点，统一为 >50K / <=50K
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.rstrip(".")
    df = df[df[LABEL_COL].isin([">50K", "<=50K"])].copy()
    df[LABEL_COL] = (df[LABEL_COL] == ">50K").astype(np.int64)
    return df.reset_index(drop=True)


def _build_sparse_codes(df: pd.DataFrame, sparse_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    """把类别列编码为从 0 开始的连续整数；缺失值（NaN）单独占一个码 0。

    返回 (sparse_matrix[int64, (N, n_sparse)], sparse_cards)。
    sparse_cards[i] = 第 i 列的去重类别数（+1 预留给缺失/OOV）。
    """
    codes = np.zeros((len(df), len(sparse_cols)), dtype=np.int64)
    cards: list[int] = []
    for j, col in enumerate(sparse_cols):
        vals = df[col].astype("object")
        uniq = vals.dropna().unique().tolist()
        # 缺失 -> 0；其余从 1 开始连续编号
        mapping = {v: (i + 1) for i, v in enumerate(uniq)}
        codes[:, j] = vals.map(lambda v: mapping.get(v, 0)).to_numpy(dtype=np.int64)
        cards.append(int(len(uniq) + 1))      # +1 给缺失/OOV 预留
    return codes, cards


def load_adult(cache_dir: str | Path = "data/adult",
               val_split: float = 0.2, seed: int = 42) -> dict:
    """加载并预处理 UCI Adult，返回训练/验证四元组与元信息。

    返回 dict 字段：
        dense_train, sparse_train, y_train   (np.ndarray)
        dense_val,   sparse_val,   y_val     (np.ndarray)
        n_dense      int
        sparse_cards list[int]
        meta         dict（数据集、规模、正类率、预处理说明）
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = _parse(_download("train", cache_dir))
    test_df = _parse(_download("test", cache_dir))
    df = pd.concat([train_df, test_df], ignore_index=True)
    n_total = len(df)

    dense = df[DENSE_COLS].to_numpy(dtype=np.float64)
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    sparse, sparse_cards = _build_sparse_codes(df, SPARSE_COLS)

    # 随机分层切分（保证验证集正类率与全量一致）
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_val_pos = int(round(len(pos) * val_split))
    n_val_neg = int(round(len(neg) * val_split))
    val_idx = np.concatenate([rng.choice(pos, n_val_pos, replace=False),
                              rng.choice(neg, n_val_neg, replace=False)])
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(n_total) if i not in val_set])

    dtr, dva = dense[train_idx], dense[val_idx]
    str_, sva = sparse[train_idx], sparse[val_idx]
    ytr, yva = y[train_idx], y[val_idx]

    # dense 标准化（仅在训练集上拟合统计量）
    mu, sd = dtr.mean(axis=0), dtr.std(axis=0) + 1e-8
    dtr = ((dtr - mu) / sd).astype(np.float32)
    dva = ((dva - mu) / sd).astype(np.float32)

    meta = {
        "dataset": "UCI Adult / Census Income (1996)",
        "source": UCI_BASE,
        "license": "UCI ML Repository; research use (cite Becker & Kohavi 1996 / Donoho 2019)",
        "n_total": int(n_total),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "positive_rate_total": round(float(y.mean()), 4),
        "positive_rate_train": round(float(ytr.mean()), 4),
        "positive_rate_val": round(float(yva.mean()), 4),
        "n_dense": len(DENSE_COLS),
        "dense_cols": DENSE_COLS,
        "sparse_cols": SPARSE_COLS,
        "sparse_cards": sparse_cards,
        "preprocessing": "dense: 标准化(Z-score, 训练集统计量); sparse: 缺失->0 的连续整数编码",
        "note": "Criteo 在构建沙箱网络被阻断；UCI Adult 作为同构真实 (dense,sparse)->二元 基准。",
    }
    return {
        "dense_train": dtr, "sparse_train": str_, "y_train": ytr,
        "dense_val": dva, "sparse_val": sva, "y_val": yva,
        "n_dense": len(DENSE_COLS), "sparse_cards": sparse_cards, "meta": meta,
    }


# ---------------------------------------------------------------------------
# Criteo 接入：已实现于 src/data/criteo.py（独立模块，与 load_adult 同构）
# ---------------------------------------------------------------------------
# Criteo 加载器（src/data/criteo.py）已完整实现 DAC 格式解析与预处理，并单测验证；
# 本机提供 train.txt 后运行 `scripts/validate_real_data_deepfm.py --dataset criteo` 即可。
# 无需改动 DeepFM / trainer —— 两者都只依赖 (dense_dim, sparse_cards) 四元组。
