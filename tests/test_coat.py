"""Coat 数据加载器测试。

数据可得性
----------
Coat 原始文件（`coat.zip`，Cornell 发布，**CC BY-NC 许可**）**不入库**——
与仓库 MIT 许可不兼容。因此本测试在数据缺失时 **skip**，不阻塞 CI。

获取方式（在有网环境执行一次）：
    curl -O https://www.cs.cornell.edu/~schnabts/mnar/coat.zip
    unzip coat.zip -d <COAT_DIR>

测试环境变量
------------
设置 `COAT_DIR` 指向解压目录即可让测试生效：
    COAT_DIR=/workspace/AeroRec-毕设素材/coat pytest tests/test_coat.py -v
未设置时自动尝试默认路径，仍找不到则 skip。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from src.data.coat import coat_click_matrix, load_coat

_CANDIDATES = [
    os.environ.get("COAT_DIR"),
    "/workspace/AeroRec-毕设素材/coat",
    str(Path(__file__).resolve().parents[1] / "data" / "raw" / "coat"),
]


def _find_coat_dir() -> Path | None:
    for c in _CANDIDATES:
        if not c:
            continue
        p = Path(c)
        if (p / "train.ascii").exists() and (p / "propensities.ascii").exists():
            return p
    return None


_COAT_DIR = _find_coat_dir()
pytestmark = pytest.mark.skipif(
    _COAT_DIR is None,
    reason="Coat 原始数据缺失（CC BY-NC，不入库）；设置 COAT_DIR 后可运行",
)


@pytest.fixture(scope="module")
def coat():
    return load_coat(_COAT_DIR)


def test_shapes_match_published_spec(coat):
    """Coat 的公开规格：290 用户 × 300 物品，每用户恰好 24 个自选物品。"""
    assert coat["train_ratings"].shape == (290, 300)
    assert coat["test_ratings"].shape == (290, 300)
    assert coat["propensities"].shape == (290, 300)
    # 每用户 24 个自选（这是 Coat 的构造方式，不是巧合）
    assert set((coat["train_ratings"] > 0).sum(axis=1).tolist()) == {24}


def test_propensity_range_and_scale(coat):
    """倾向分必须是概率量纲，且行和近似等于每用户曝光数（Coat 的构造约束）。"""
    pi = coat["propensities"]
    assert np.isfinite(pi).all()
    assert (pi > 0).all() and (pi <= 1).all()
    # 行和 ≈ 每用户被曝光物品数（24），这是 Schnabel 协议的构造性质
    row_sum = pi.sum(axis=1)
    assert abs(float(row_sum.mean()) - 24.0) < 1.0, float(row_sum.mean())


def test_metainfo_consistency(coat):
    """meta 里的汇总数字必须与实际矩阵一致——防止常量与数据脱钩。"""
    m = coat["meta"]
    assert m["n_users"] == 290 and m["n_items"] == 300
    assert m["train_per_user"] == 24
    assert m["n_train_obs"] == int((coat["train_ratings"] > 0).sum())
    assert m["n_test_obs"] == int((coat["test_ratings"] > 0).sum())
    assert m["rating_range"] == [1, 5]


def test_click_matrix_threshold():
    """`coat_click_matrix` 按阈值二值化，且**未观测（0 分）保持 0**。

    返回类型按实现约定为 float64（要参与 IPS 加权计算），不是整型。
    """
    ratings = np.zeros((3, 4), dtype=np.int64)
    ratings[0, :2] = [5, 3]          # 5 过阈值，3 不过
    ratings[1, 0] = 4                # 恰好等于阈值
    cm = coat_click_matrix(ratings, threshold=4)
    assert cm.dtype == np.float64
    assert cm.shape == ratings.shape
    assert cm[0, 0] == 1.0 and cm[0, 1] == 0.0
    assert cm[1, 0] == 1.0           # 等于阈值算点击
    assert cm[2].sum() == 0.0        # 未观测整行保持 0
    assert cm.sum() == 2.0
    assert set(np.unique(cm).tolist()).issubset({0.0, 1.0})


def test_click_matrix_rejects_bad_threshold():
    """阈值必须落在评分值域内——否则会静默产生全 0 或全 1。"""
    ratings = np.zeros((2, 3), dtype=np.int64)
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            coat_click_matrix(ratings, threshold=bad)


def test_click_matrix_rejects_non_2d():
    with pytest.raises(ValueError):
        coat_click_matrix(np.zeros((3,), dtype=np.int64))
