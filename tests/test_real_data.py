"""路径 B 数据加载器测试：不依赖网络。

若本地已缓存 UCI Adult（data/adult/adult.data），则校验 (dense, sparse, sparse_cards, y)
四元组的形状与取值；否则跳过（CI 无网络时不影响仓库其它测试）。
"""
import os

import numpy as np
import pytest

from src.data import adult

ADULT_CACHE = os.path.join("data", "adult", "adult.data")


def test_loader_shape_and_domains():
    if not os.path.exists(ADULT_CACHE):
        pytest.skip("UCI Adult 未缓存（无网络），跳过加载器校验")

    data = adult.load_adult()
    dtr, str_, ytr = data["dense_train"], data["sparse_train"], data["y_train"]
    dva, sva, yva = data["dense_val"], data["sparse_val"], data["y_val"]

    # dense
    assert dtr.shape[1] == data["n_dense"] == 6
    assert not np.isnan(dtr).any() and not np.isnan(dva).any()
    # sparse：列数 = 8，且为从 0 开始的非负整数，最大码 < cardinality
    assert str_.shape[1] == len(data["sparse_cards"]) == 8
    assert sva.shape[1] == 8
    assert str_.dtype == np.int64 and str_.min() >= 0
    for j, card in enumerate(data["sparse_cards"]):
        assert str_[:, j].max() < card, f"列 {j} 编码超界"
    # 标签为二元
    assert set(np.unique(ytr).tolist()) <= {0, 1}
    assert set(np.unique(yva).tolist()) <= {0, 1}
    # 训练/验证不交叠、规模合理
    assert len(dtr) > len(dva)
    assert 0.15 < data["meta"]["positive_rate_val"] < 0.35  # 与 CTR 类别不均衡同构
