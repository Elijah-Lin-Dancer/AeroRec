"""纯 numpy 的 LR(CVR) 推理 + OneHot 编码（在线 Demo 用，替代 scikit-learn）。

背景见 `src/build_online_artifacts.py`：在线免费实例内存上限 1GB，
sklearn 的固有导入开销为 151MB，torch 为 643MB。为把峰值压到安全区间，
在线侧只保留 numpy，因此需要把 sklearn 的两个组件用 numpy 复现：

  1. `OneHotEncoder(categories=[range(c) for c in SPARSE_CARD], sparse_output=False)`
     —— 等价于把每个稀疏特征的值映射到 one-hot 段的偏移位置置 1。
  2. `LogisticRegression.predict_proba`
     —— 等价于 `sigmoid(X @ coef.T + intercept)`。

数学等价性：两者都是确定性线性变换，numpy 复现无近似。
"""
from __future__ import annotations

import numpy as np


class NumpyOneHot:
    """等价于 sklearn 的 `OneHotEncoder(sparse_output=False)`（按给定类别数）。"""

    def __init__(self, cards: list[int]):
        self.cards = [int(c) for c in cards]
        # 每段的起始偏移，长度 = n_features + 1
        self.offsets = np.cumsum([0] + self.cards).astype(np.int64)
        self.n_features_out = int(self.offsets[-1])

    def transform(self, sparse: np.ndarray) -> np.ndarray:
        sparse = np.asarray(sparse, dtype=np.int64)
        if sparse.ndim == 1:
            sparse = sparse[None, :]
        B = sparse.shape[0]
        out = np.zeros((B, self.n_features_out), dtype=np.float32)
        # 第 j 个特征取值 v → 列号 = offsets[j] + v
        cols = self.offsets[: len(self.cards)] + sparse       # (B, n_feat)
        rows = np.repeat(np.arange(B, dtype=np.int64), len(self.cards))
        out[rows, cols.reshape(-1)] = 1.0
        return out


class NumpyLogisticRegression:
    """等价于 sklearn 的 `LogisticRegression`（二分类，推理用）。"""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray):
        self.coef = np.asarray(coef, dtype=np.float32).reshape(1, -1)
        self.intercept = np.asarray(intercept, dtype=np.float32).reshape(1)

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        """返回正类（转化）概率，等价于 `predict_proba(X)[:, 1]`。"""
        z = np.asarray(X, dtype=np.float32) @ self.coef.T + self.intercept
        return (1.0 / (1.0 + np.exp(-z))).reshape(-1)
