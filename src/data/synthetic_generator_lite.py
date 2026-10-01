"""轻量档数据生成器（纯 numpy，不构造 pandas 表）。

为什么不用 pandas
-----------------
在 pandas 3.0 + pyarrow 环境下，构造一个 8 列 × 100 万行的 DataFrame
会产生约 170MB 的**额外转换开销**（DataFrame ↔ Arrow 后端），
而数据本身只有 60MB。这个固定开销在在线 Demo 的内存预算里不可接受。

因此轻量档生成器直接产出 **numpy 字典**（列名 → 数组），
需要落盘时再按需写出。

生成公式与完整档 `src/data/synthetic_generator.py` 完全一致：
同一随机种子、同一系数、同一 `_CAT_BIAS` 类目偏置、同一隐式非线性交叉
（价格敏感×低价、商务×早班机、里程预算契合）。
"""
from __future__ import annotations

import numpy as np

from src.config_lite import (
    SEED, N_USERS, N_ITEMS, IMPRESSIONS_PER_USER, CATEGORIES,
    CTR_BASE, QUALITY_W, GAMMA_PRICE, GAMMA_BIZ_MORNING, GAMMA_MILES,
    BETA0, BETA_QUALITY, BETA_MILES, BETA_PRICE, CVR_NOISE_STD,
)
from src.data.synthetic_generator import _CAT_BIAS, _sigmoid

_CAT = list(CATEGORIES)
N_CITY = 10


def generate_items_np(rng, n_items: int = N_ITEMS) -> dict:
    """生成物品目录（numpy 字典版），与完整档 `generate_items` 同一分布。"""
    n_per = n_items // len(CATEGORIES)
    cat_idx = np.repeat(np.arange(len(CATEGORIES)), n_per)
    if len(cat_idx) < n_items:                       # 补齐余数
        cat_idx = np.concatenate([cat_idx,
                                  np.full(n_items - len(cat_idx), len(CATEGORIES) - 1)])
    quality = rng.normal(0.0, 1.0, size=n_items)

    base_price = np.array([1000.0, 400.0, 200.0, 50.0, 100.0])[cat_idx]
    cash_price = np.round(base_price * np.exp(rng.normal(0.0, 0.4, size=n_items)), 2)

    is_route = (cat_idx == 0)
    is_miles = (is_route & (rng.random(n_items) < 0.5)).astype(np.int64)
    miles_required = np.where(is_miles == 1, np.round(cash_price * 10, 0), 0.0)

    origin = np.where(is_route, rng.integers(0, N_CITY, size=n_items), -1).astype(np.int64)
    dest = np.where(is_route, rng.integers(0, N_CITY, size=n_items), -1).astype(np.int64)
    swap = (is_route & (origin == dest))
    dest[swap] = (dest[swap] + 1) % N_CITY
    time_slot = np.where(is_route, rng.integers(0, 3, size=n_items), 3).astype(np.int64)

    stock = rng.integers(0, 100, size=n_items)
    stock = np.where(is_miles == 1, stock, np.where(stock < 3, 0, stock))
    popularity = np.round(np.exp(rng.normal(0.0, 0.8, size=n_items)), 4)

    return {
        "item_id": np.arange(n_items, dtype=np.int64),
        "category_idx": cat_idx.astype(np.int64),
        "quality": quality.astype(np.float32),
        "cash_price": cash_price.astype(np.float32),
        "is_miles_ticket": is_miles,
        "miles_required": miles_required.astype(np.float32),
        "origin_idx": origin,
        "dest_idx": dest,
        "time_slot": time_slot,
        "stock": stock.astype(np.int64),
        "popularity": popularity.astype(np.float32),
    }


def generate_users_np(rng, n_users: int = N_USERS) -> dict:
    """生成用户（numpy 字典版），与完整档 `generate_users` 同一分布。"""
    d = {"user_id": np.arange(n_users, dtype=np.int64)}
    for c in _CAT:
        d[f"aff_{c}"] = rng.normal(0.0, 1.0, size=n_users).astype(np.float32)
    miles = np.round(np.exp(rng.normal(10.0, 1.0, size=n_users)), 0)
    d["miles_balance"] = np.where(rng.random(n_users) < 0.05, 0.0, miles).astype(np.float32)
    d["home_idx"] = rng.integers(0, N_CITY, size=n_users).astype(np.int64)
    d["price_sensitivity"] = rng.normal(0.0, 1.0, size=n_users).astype(np.float32)
    d["schedule_flexibility"] = rng.normal(0.0, 1.0, size=n_users).astype(np.float32)
    d["is_business"] = (rng.random(n_users) < 0.2).astype(np.float32)
    d["active_days_7d"] = rng.poisson(2.5, size=n_users).astype(np.int64)
    return d


def generate_trips_np(rng, n_users: int = N_USERS) -> dict:
    """生成未来行程（numpy 字典版）：每用户 0~4 条，含出发/到达与距今天数。"""
    n_trips = np.clip(rng.poisson(1.5, size=n_users), 0, 4)
    total = int(n_trips.sum())
    uid = np.repeat(np.arange(n_users, dtype=np.int64), n_trips)
    o = rng.integers(0, N_CITY, size=total).astype(np.int64)
    d = ((o + rng.integers(1, N_CITY, size=total)) % N_CITY).astype(np.int64)
    return {
        "user_id": uid,
        "origin_idx": o,
        "dest_idx": d,
        "depart_offset": rng.integers(0, 30, size=total).astype(np.int64),
    }


def generate_logs_np(rng, items: dict, users: dict,
                     impressions_per_user: int = IMPRESSIONS_PER_USER) -> dict:
    """生成曝光-点击-转化日志（numpy 字典版）。

    与完整档 `generate_logs` 同公式、同系数、同隐式交叉结构；
    差别仅在于全程 numpy gather，不构造中间 DataFrame。
    """
    n_users = len(users["user_id"])
    n_items = len(items["item_id"])
    n = n_users * impressions_per_user

    user_idx = np.repeat(np.arange(n_users, dtype=np.int64), impressions_per_user)
    item_idx = rng.integers(0, n_items, size=n).astype(np.int64)

    # --- gather：直接从 numpy 列取，无中间副本 ---
    aff = np.stack([users[f"aff_{c}"][user_idx] for c in _CAT], axis=1)   # (n,5)
    cat = items["category_idx"][item_idx]
    quality = items["quality"][item_idx]
    cash_price = items["cash_price"][item_idx]
    is_miles = items["is_miles_ticket"][item_idx]
    miles_req = items["miles_required"][item_idx]
    time_slot = items["time_slot"][item_idx]
    miles_bal = users["miles_balance"][user_idx]
    price_sens = users["price_sensitivity"][user_idx]
    is_biz = users["is_business"][user_idx]

    hour_slot = rng.integers(0, 2, size=n).astype(np.int64)
    is_weekend = (rng.random(n) < 0.3).astype(np.float32)

    # 派生量（仅用于生成标签，不作为特征给模型）
    price_max = float(cash_price.max()) if cash_price.max() > 0 else 1.0
    price_norm = cash_price / price_max
    is_early = (time_slot == 0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        miles_fit = np.where(is_miles == 1,
                             np.clip((miles_bal - miles_req) / np.maximum(miles_req, 1),
                                     -1.0, 1.0), 0.0).astype(np.float32)

    # === CTR：与完整档同式 ===
    aff_cat = aff[np.arange(n), cat]
    logit = (CTR_BASE + aff_cat + QUALITY_W * quality + _CAT_BIAS[cat]
             + GAMMA_PRICE * (price_sens * (1.0 - price_norm))     # 隐式交叉1
             + GAMMA_BIZ_MORNING * (is_biz * is_early)             # 隐式交叉2
             + GAMMA_MILES * miles_fit)                            # 隐式交叉3
    true_ctr = _sigmoid(logit)
    clicked = (rng.random(n) < true_ctr).astype(np.int64)

    # === CVR：与完整档同式 ===
    cvr_logit = (BETA0 + BETA_QUALITY * quality + BETA_MILES * miles_fit
                 + BETA_PRICE * (1.0 - price_norm)
                 + rng.normal(0.0, CVR_NOISE_STD, size=n))
    cvr = _sigmoid(cvr_logit)
    converted = (clicked & (rng.random(n) < cvr)).astype(np.int64)

    return {
        "user_id": user_idx,
        "item_id": item_idx,
        "category_idx": cat.astype(np.int64),
        "hour_slot": hour_slot,
        "is_weekend": is_weekend,
        "true_ctr": np.round(true_ctr, 4).astype(np.float32),
        "clicked": clicked,
        "converted": converted,
    }


def build_dataset_np(seed: int = SEED):
    """一次性生成轻量档全量数据（numpy 字典），供在线 Demo 即时使用。"""
    rng = np.random.default_rng(seed)
    items = generate_items_np(rng, N_ITEMS)
    users = generate_users_np(rng, N_USERS)
    trips = generate_trips_np(rng, N_USERS)
    logs = generate_logs_np(rng, items, users)
    return items, users, trips, logs


if __name__ == "__main__":
    import resource
    import time

    t = time.time()
    it, us, tr, lg = build_dataset_np()
    print(f"[gen] {time.time() - t:.1f}s logs={len(lg['clicked'])} "
          f"peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB")
    print(f"ctr={lg['clicked'].mean():.4f} "
          f"cvr(click)={lg['converted'][lg['clicked'] == 1].mean():.4f}")
