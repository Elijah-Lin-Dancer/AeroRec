"""轻量档数据生成器（纯 numpy，不构造 pandas 表）。

为什么不用 pandas
-----------------
在 pandas 3.0 + pyarrow 环境下，构造一个 8 列 × 100 万行的 DataFrame
会产生约 170MB 的**额外转换开销**（DataFrame ↔ Arrow 后端），
而数据本身只有 60MB。这个固定开销在在线 Demo 的 2GB 内存里不可接受。

因此轻量档生成器**直接产出 numpy 字典**（列名 → 数组），
需要写 CSV 时才按需的、分块的写出。

生成公式与完整档 `src/data/synthetic_generator.py` **完全一致**：
同一随机种子、同一系数、同一 `CAT_BIAS` 类目偏置、同一隐式非线性交叉
（价格敏感×低价、商务×早班机、里程预算契合）。

⚠️ 本模块从 pandas-free 的 `generator_common` 导入基础量与真实 CTR 公式，
**不触发 pandas 导入**（在线内存受限实例的硬要求）。
"""
from __future__ import annotations

import numpy as np

from src.config_scale import (
    SEED, N_USERS, N_ITEMS, IMPRESSIONS_PER_USER, CATEGORIES,
    CTR_BASE, QUALITY_W, GAMMA_PRICE, GAMMA_BIZ_MORNING, GAMMA_MILES,
    BETA0, BETA_QUALITY, BETA_MILES, BETA_PRICE, CVR_NOISE_STD,
)
from src.data.generator_common import CAT_BIAS as _CAT_BIAS, sigmoid as _sigmoid

_CAT = list(CATEGORIES)

# 轻量档列名（与完整档日志表保持一致，便于对照）
LOG_COLS = ["user_id", "item_id", "category_idx", "category",
            "hour_slot", "is_weekend", "true_ctr", "clicked", "converted"]
