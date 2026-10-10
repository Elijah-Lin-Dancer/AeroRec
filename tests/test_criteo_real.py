"""Criteo 加载器对真实 parquet 的兼容性测试（数据存在才跑，否则跳过）。

与 ``tests/test_criteo.py`` 的构造样本不同，这里直接用真实 ``criteo-1M`` parquet
的小子集验证：列名映射、稠密 NaN 处理、高基数稀疏编码、分层切分与正类率保持。
原始 parquet 不入库（见 .gitignore），缺失时自动 skip，不阻塞 CI。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.data.criteo import N_DENSE, N_SPARSE, load_criteo

_PARQUET = os.path.join("data", "criteo", "criteo_1M.parquet")


@pytest.mark.skipif(not os.path.exists(_PARQUET), reason="真实 Criteo parquet 未缓存，跳过")
def test_criteo_real_parquet_loads():
    data = load_criteo(_PARQUET, n_rows=20000, val_split=0.1)
    meta = data["meta"]

    assert meta["input_format"] == "parquet"
    assert data["n_dense"] == N_DENSE
    assert len(data["sparse_cards"]) == N_SPARSE
    assert meta["n_train"] + meta["n_val"] == 20000

    # 真实 CTR 正类率应显著低于 0.5（Criteo 约 0.25）
    assert 0.15 < meta["positive_rate_val"] < 0.35

    # 稠密经 log1p + Z-score 后不应有 NaN / Inf
    assert np.isfinite(data["dense_train"]).all()
    assert np.isfinite(data["dense_val"]).all()

    # 稀疏码必须落在 [0, card) 范围
    for j, card in enumerate(data["sparse_cards"]):
        col = np.concatenate([data["sparse_train"][:, j], data["sparse_val"][:, j]])
        assert col.min() >= 0
        assert col.max() < card
