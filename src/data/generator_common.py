"""数据生成共用的基础量与真实 CTR/CVR 计算公式（pandas-free）。

为什么要单独拆出来
------------------
`src/data/synthetic_generator.py`（完整档生成器）依赖 pandas，
而 pandas 的导入开销约 150MB。但轻量/极轻档与在线评估路径只需要其中
几个**纯 numpy** 的基础量：

  - `CAT_BIAS` / `sigmoid`
  - `pair_features` / `compute_true_ctr` / `compute_true_cvr`
    （真实 CTR 口径评估的核心，`src/eval/ab.py` 依赖它）

这些函数原先定义在 pandas 模块里，导致 `from src.data.synthetic_generator import
compute_true_ctr` 会连带把 pandas 拉进内存——在线内存受限实例承担不起。

因此把上述内容下沉到本模块。`src/data/synthetic_generator.py` 原样再导出，
保持向后兼容；`synthetic_generator_lite.py` 与 `eval/ab.py` 改用本模块，
**不再触发 pandas**。

数值一致性：`CAT_BIAS` 的构造（同种子 `SEED`、同 `CAT_BIAS_STD`、
同 `CATEGORIES` 顺序）与拆分前完全相同；`compute_true_ctr/cvr` 的公式逐项一致，
因此三档数据的真实 CTR/CVR 分布与离线评估结论不受任何影响。

关于入参类型：`users` / `items` 可以是 pandas DataFrame，也可以是
本项目在线侧使用的「伪 DataFrame」适配层（`src/serve/engine_lite.py` 的
`_ABUsers` / `_ABItems`，提供 `.iloc[...]` 与 `.to_numpy()`）。
本模块用**鸭子类型**访问，两种入参均可工作。
"""
from __future__ import annotations

import numpy as np

from src.config import (
    SEED, CAT_BIAS_STD, CATEGORIES,
    CTR_BASE, QUALITY_W, GAMMA_PRICE, GAMMA_BIZ_MORNING, GAMMA_MILES,
    BETA0, BETA_QUALITY, BETA_MILES, BETA_PRICE,
)


def sigmoid(x):
    """标准 sigmoid，用于把 logit 映射为概率。"""
    return 1.0 / (1.0 + np.exp(-x))


# 类目级偏置：同一随机种子 → 三档共用同一组偏置，保证分布可比
CAT_BIAS = np.random.default_rng(SEED).normal(0.0, CAT_BIAS_STD, size=len(CATEGORIES))

# 兼容旧引用名（原 `synthetic_generator.py` 中的私有名）
_CAT_BIAS = CAT_BIAS
_sigmoid = sigmoid


# ----------------------------------------------------------------------
def _take(df, idx):
    """按位置取行，兼容 pandas DataFrame 与伪 DataFrame 适配层。"""
    if hasattr(df, "iloc"):
        return df.iloc[idx].reset_index(drop=True)
    # 纯 dict 兜底（{列名: np.ndarray}）
    return {k: np.asarray(v)[idx] for k, v in df.items()}


def _col(row_like, name):
    """取列，兼容 pandas Series/DataFrame 与 dict。"""
    if isinstance(row_like, dict):
        return np.asarray(row_like[name])
    return np.asarray(row_like[name])


def pair_features(users, items, user_ids, item_ids):
    """提取 (用户, 物品) 对的原始/派生特征，供 true_ctr / true_cvr 共用。"""
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    u = _take(users, user_ids)
    it = _take(items, item_ids)

    aff = np.stack([_col(u, f"aff_{c}") for c in CATEGORIES], axis=1)
    cat = _col(it, "category_idx").astype(np.int64)
    quality = _col(it, "quality").astype(float)
    cash_price = _col(it, "cash_price").astype(float)
    is_miles = _col(it, "is_miles_ticket").astype(float)
    miles_req = _col(it, "miles_required").astype(float)
    miles_bal = _col(u, "miles_balance").astype(float)
    price_sens = _col(u, "price_sensitivity").astype(float)
    is_biz = _col(u, "is_business").astype(float)
    time_slot = _col(it, "time_slot").astype(np.int64)

    # 价格归一化：分母取全量物品的最大现金价（与生成期同口径）
    price_all = _col(items, "cash_price").astype(float)
    price_max = float(np.nanmax(price_all)) if len(price_all) else 1.0
    if not np.isfinite(price_max) or price_max <= 0:
        price_max = 1.0
    price_norm = cash_price / price_max

    is_early = (time_slot == 0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        miles_fit = np.where(
            is_miles == 1,
            np.clip((miles_bal - miles_req) / np.maximum(miles_req, 1), -1.0, 1.0),
            0.0)
    aff_cat = aff[np.arange(len(user_ids)), cat]
    return aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit


def compute_true_ctr(users, items, user_ids, item_ids):
    """对任意 (用户, 物品) 对计算潜在真实 CTR（与生成日志同一公式、同一 cat_bias）。

    供离线/在线评估使用：可用它度量"推荐列表的真实点击概率"，
    从而在**不依赖任何平滑历史 CTR 估计**的前提下对比不同排序策略的效果。
    """
    aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit = \
        pair_features(users, items, user_ids, item_ids)
    logit = (CTR_BASE + aff_cat + QUALITY_W * quality + CAT_BIAS[cat]
             + GAMMA_PRICE * (price_sens * (1.0 - price_norm))
             + GAMMA_BIZ_MORNING * (is_biz * is_early)
             + GAMMA_MILES * miles_fit)
    return sigmoid(logit)


def compute_true_cvr(users, items, user_ids, item_ids):
    """对任意 (用户, 物品) 对计算潜在真实 CVR（点击后转化率，去掉噪声的期望值）。"""
    _, quality, _, _, price_norm, _, _, miles_fit = \
        pair_features(users, items, user_ids, item_ids)
    logit = (BETA0 + BETA_QUALITY * quality + BETA_MILES * miles_fit
             + BETA_PRICE * (1.0 - price_norm))
    return sigmoid(logit)
