"""Criteo 加载器单测：不依赖网络、不依赖真实文件，验证解析逻辑正确。

说明：本沙箱网络无法访问 Criteo（官方 S3 / go.criteo.net / HuggingFace 全部不可达），
故无法在此跑出真实 Criteo AUC；但加载器的「格式解析 + 缺失处理 + 词表/OOV 编码」
逻辑可用构造的极小样本完全验证。真实数字需在本机放置 train.txt 后运行
`scripts/validate_real_data_deepfm.py --dataset criteo`。
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from src.data.criteo import N_DENSE, N_SPARSE, _build_vocab, _encode, load_criteo

# 26 个类别字段统一用一个稳定 token，只有 C1 故意放入频率差异以触发 OOV
_SPARSE_FLAT = ["x"] * N_SPARSE


def _line(label: int, dense, c1_token: str) -> str:
    """构造一行 Criteo DAC 文本。

    dense : 长度 13 的列表，元素为数字或 ""（缺失）
    c1_token : C1 字段 token（"", 数字均可）
    """
    dense_str = [str(v) for v in dense]
    sparse = list(_SPARSE_FLAT)
    sparse[0] = c1_token
    return "\t".join([str(label)] + dense_str + sparse)


def _dense_all_zero_except():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_criteo_encode_missing_dense_and_label():
    # I2 缺失（空串），其余为 0..12
    dense = _dense_all_zero_except()
    dense[1] = ""  # 第 2 个稠密特征缺失
    line = _line(1, dense, "tokA")
    vocab = _build_vocab([line], max_card=10)
    dense_arr, sparse_arr, y = _encode([line], vocab)
    assert dense_arr.shape == (1, N_DENSE)
    assert dense_arr[0, 1] == 0.0          # 缺失 -> 0
    assert dense_arr[0, 0] == 0.0
    assert dense_arr[0, 2] == 2.0
    assert y[0] == 1
    assert sparse_arr.shape == (1, N_SPARSE)
    assert sparse_arr[0, 0] == 1           # tokA 入表（码 1）


def test_criteo_oov_encoding():
    # 3 行：C1 字段 "aaa" 出现 2 次，"bbb" 出现 1 次；max_card=2 -> 仅 "aaa" 入表
    lines = [
        _line(1, _dense_all_zero_except(), "aaa"),
        _line(0, _dense_all_zero_except(), "aaa"),
        _line(1, _dense_all_zero_except(), "bbb"),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        data = load_criteo(path, max_card=2, val_split=0.34)
        assert data["n_dense"] == N_DENSE
        assert len(data["sparse_cards"]) == N_SPARSE
        assert data["meta"]["n_train"] + data["meta"]["n_val"] == 3
        sp = np.concatenate([data["sparse_train"], data["sparse_val"]], axis=0)
        c1 = sp[:, 0]
        # max_card=2：码 0 留给 OOV/缺失，码 1 给 top1 token("aaa")
        assert set(np.unique(c1).tolist()).issubset({0, 1})
        assert int(np.sum(c1 == 1)) == 2     # "aaa" 两次 -> 码 1
        assert int(np.sum(c1 == 0)) >= 1     # "bbb" 低频 -> OOV 码 0
    finally:
        os.unlink(path)


def test_criteo_missing_file_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_criteo("/nonexistent/path/train.txt")


def test_criteo_parquet_roundtrip():
    """parquet 路径：列名 label/I*/C*、稠密含 NaN、稀疏为整数哈希码。"""
    import pandas as pd
    n = 6
    df = pd.DataFrame({
        "label": [1, 0, 1, 0, 1, 0],
        **{f"I{i}": [float(j + i) if j != 1 else float("nan") for j in range(n)]
           for i in range(1, N_DENSE + 1)},
        **{f"C{i}": [(7 if j % 2 == 0 else 9) if i == 1 else (j + i) for j in range(n)]
           for i in range(1, N_SPARSE + 1)},
    })
    with tempfile.NamedTemporaryFile("w", suffix=".parquet", delete=False) as f:
        path = f.name
    df.to_parquet(path, index=False)
    try:
        data = load_criteo(path, val_split=0.34)
        assert data["n_dense"] == N_DENSE
        assert len(data["sparse_cards"]) == N_SPARSE
        assert data["meta"]["input_format"] == "parquet"
        assert data["meta"]["n_train"] + data["meta"]["n_val"] == n
        # 稠密缺失（NaN）应被填 0 后再 log1p/Z-score，不应出现 NaN
        assert not np.isnan(data["dense_train"]).any()
        assert not np.isnan(data["dense_val"]).any()
    finally:
        os.unlink(path)
