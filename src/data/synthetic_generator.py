"""合成出行交互数据生成器（全向量化，含出行约束 / 隐式非线性交叉 / 转化标签）。

设计目标
--------
本项目是"受控合成实验"：用固定随机种子生成一个**带潜在真实 CTR/CVR** 的
出行推荐场景数据集，演示工业级推荐系统全链路，且可一键复现。

关键点（诚实边界）：
  1. 不使用任何真实业务数据，也不声称使用。
  2. 数据中注入**隐式非线性交叉**（如 商务用户×早班机、价格敏感×低价）：
     生成公式里存在乘积项，但模型只能拿到**原始特征**，必须自行学出交叉——
     这是 DeepFM 等深度模型"有东西可学"的前提，且明确标注为受控实验。
  3. 引入**出行特有属性**（里程、现金价、出发/到达、未来行程），支撑
     "里程兑换""多段行程"等出行业务规则。
"""
import numpy as np
import pandas as pd

from src.config import (
    SEED, N_USERS, N_ITEMS, ITEMS_PER_CAT, IMPRESSIONS_PER_USER,
    MAX_TRIPS_PER_USER, CATEGORIES, CITIES, TIME_SLOTS, HOUR_SLOTS,
    CTR_BASE, QUALITY_W, GAMMA_PRICE, GAMMA_BIZ_MORNING, GAMMA_MILES, CAT_BIAS_STD,
    BETA0, BETA_QUALITY, BETA_MILES, BETA_PRICE, CVR_NOISE_STD,
    ITEMS_PATH, USERS_PATH, TRIPS_PATH, LOGS_PATH, PROC_DIR,
)

# 固定类目偏置：生成日志与离线评估共用，保证 true_ctr 口径一致
_CAT_BIAS = np.random.default_rng(SEED).normal(0.0, CAT_BIAS_STD, size=len(CATEGORIES))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_items(rng, n_items=N_ITEMS) -> pd.DataFrame:
    """生成物品目录（含出行属性：价格/里程票/城市/时段/库存/质量）。"""
    n_per = n_items // len(CATEGORIES)
    cat_idx = np.repeat(np.arange(len(CATEGORIES)), n_per)
    quality = rng.normal(0.0, 1.0, size=n_items)

    # 各类目现金价基准（元）
    base_price = np.array([1000.0, 400.0, 200.0, 50.0, 100.0])[cat_idx]
    cash_price = np.round(base_price * np.exp(rng.normal(0.0, 0.4, size=n_items)), 2)

    # 里程票仅出现在"航线"，里程价约 = 现金价 × 10
    is_route = (cat_idx == 0)
    is_miles = (is_route & (rng.random(n_items) < 0.5)).astype(int)
    miles_required = np.where(is_miles == 1, np.round(cash_price * 10, 0), 0.0)

    # 城市与时段（主要对航线有意义）
    origin = np.where(is_route, rng.choice(len(CITIES), size=n_items),
                      np.nan)
    dest = np.where(is_route, rng.choice(len(CITIES), size=n_items), np.nan)
    # 保证 origin != dest
    swap = (is_route & (origin == dest))
    dest[swap] = (dest[swap] + 1) % len(CITIES)
    time_slot = np.where(is_route, rng.choice(len(TIME_SLOTS), size=n_items),
                         len(TIME_SLOTS))  # 非航线 = '全天'（编码为 TIME_SLOTS 长度）

    # 库存：里程票"无需剔除 0 库存"规则的前提——部分 0 库存
    stock = rng.integers(0, 100, size=n_items)
    stock = np.where(is_miles == 1, stock, np.where(stock < 3, 0, stock))

    popularity = np.round(np.exp(rng.normal(0.0, 0.8, size=n_items)), 4)

    items = pd.DataFrame({
        "item_id": np.arange(n_items),
        "category_idx": cat_idx,
        "category": [CATEGORIES[i] for i in cat_idx],
        "quality": quality,
        "cash_price": cash_price,
        "is_miles_ticket": is_miles,
        "miles_required": miles_required,
        "origin_idx": origin,
        "dest_idx": dest,
        "time_slot": time_slot,        # 0/1/2=早/午/晚, 3=全天
        "stock": stock,
        "popularity": popularity,
        "name": [f"{CATEGORIES[i]}-{j:05d}" for j, i in enumerate(cat_idx)],
    })
    return items


def generate_users(rng, n_users=N_USERS) -> pd.DataFrame:
    """生成用户（含出行属性：里程余额/常驻城市/价格敏感/商务标签/活跃度）。"""
    aff = rng.normal(0.0, 1.0, size=(n_users, len(CATEGORIES)))
    df = pd.DataFrame({"user_id": np.arange(n_users)})
    for i, c in enumerate(CATEGORIES):
        df[f"aff_{c}"] = aff[:, i]

    df["miles_balance"] = np.round(np.exp(rng.normal(10.0, 1.0, size=n_users)), 0)
    df["miles_balance"] = np.where(rng.random(n_users) < 0.05, 0.0, df["miles_balance"])
    df["home_idx"] = rng.integers(0, len(CITIES), size=n_users)
    df["price_sensitivity"] = rng.normal(0.0, 1.0, size=n_users)
    df["schedule_flexibility"] = rng.normal(0.0, 1.0, size=n_users)
    df["is_business"] = (rng.random(n_users) < 0.2).astype(int)
    df["active_days_7d"] = rng.poisson(2.5, size=n_users)
    return df


def generate_trips(rng, n_users=N_USERS) -> pd.DataFrame:
    """生成用户未来行程（多段行程按日期优先返程的规则前提）。"""
    rows = []
    n_trips = rng.poisson(1.5, size=n_users)
    n_trips = np.clip(n_trips, 0, MAX_TRIPS_PER_USER)
    trip_id = 0
    for u in range(n_users):
        for _ in range(int(n_trips[u])):
            o = int(rng.integers(0, len(CITIES)))
            d = (o + int(rng.integers(1, len(CITIES)))) % len(CITIES)
            rows.append({
                "user_id": u,
                "trip_id": trip_id,
                "origin_idx": o,
                "dest_idx": d,
                "depart_offset": int(rng.integers(0, 30)),  # 距今天数
            })
            trip_id += 1
    return pd.DataFrame(rows, columns=["user_id", "trip_id", "origin_idx",
                                       "dest_idx", "depart_offset"])


def generate_logs(rng, users, items, impressions_per_user=IMPRESSIONS_PER_USER):
    """生成曝光-点击-转化日志（含上下文、隐式交叉产生的真实 CTR/CVR）。"""
    n_users = len(users)
    n_items = len(items)
    n = n_users * impressions_per_user

    user_idx = np.repeat(np.arange(n_users), impressions_per_user)
    item_idx = rng.integers(0, n_items, size=n)

    # 对齐用户/物品原始特征（模型只能拿到这些，交叉需自行学习）
    u = users.iloc[user_idx].reset_index(drop=True)
    it = items.iloc[item_idx].reset_index(drop=True)
    aff = u[[f"aff_{c}" for c in CATEGORIES]].to_numpy()
    cat = it["category_idx"].to_numpy()
    quality = it["quality"].to_numpy()
    cash_price = it["cash_price"].to_numpy()
    is_miles = it["is_miles_ticket"].to_numpy()
    miles_req = it["miles_required"].to_numpy()
    miles_bal = u["miles_balance"].to_numpy()
    price_sens = u["price_sensitivity"].to_numpy()
    is_biz = u["is_business"].to_numpy()
    time_slot = it["time_slot"].to_numpy()  # 0/1/2 早/午/晚, 3 全天

    # 上下文
    hour_slot = rng.integers(0, len(HOUR_SLOTS), size=n)
    is_weekend = (rng.random(n) < 0.3).astype(int)

    # 派生量（用于生成，不直接作为特征给模型）
    price_max = cash_price.max() if cash_price.max() > 0 else 1.0
    price_norm = cash_price / price_max
    is_early = (time_slot == 0).astype(float)          # 早班机
    # 里程契合度：够/不够的标准化
    with np.errstate(divide="ignore", invalid="ignore"):
        miles_fit = np.where(is_miles == 1,
                             np.clip((miles_bal - miles_req) / np.maximum(miles_req, 1),
                                     -1.0, 1.0), 0.0)

    cat_bias = _CAT_BIAS
    aff_cat = aff[np.arange(n), cat]

    logit = (CTR_BASE
             + aff_cat
             + QUALITY_W * quality
             + cat_bias[cat]
             + GAMMA_PRICE * (price_sens * (1.0 - price_norm))       # 隐式交叉1
             + GAMMA_BIZ_MORNING * (is_biz * is_early)               # 隐式交叉2
             + GAMMA_MILES * miles_fit)                              # 隐式交叉3
    true_ctr = _sigmoid(logit)
    clicked = (rng.random(n) < true_ctr).astype(int)

    # 转化：仅点击后可转化
    cvr_logit = (BETA0 + BETA_QUALITY * quality
                 + BETA_MILES * miles_fit
                 + BETA_PRICE * (1.0 - price_norm)
                 + rng.normal(0.0, CVR_NOISE_STD, size=n))
    cvr = _sigmoid(cvr_logit)
    converted = (clicked & (rng.random(n) < cvr)).astype(int)

    logs = pd.DataFrame({
        "user_id": user_idx,
        "item_id": item_idx,
        "category_idx": cat,
        "category": [CATEGORIES[i] for i in cat],
        "hour_slot": hour_slot,
        "is_weekend": is_weekend,
        "true_ctr": np.round(true_ctr, 4),
        "clicked": clicked,
        "converted": converted,
    })
    return logs


def _pair_features(users, items, user_ids, item_ids):
    """提取 (用户, 物品) 对的原始/派生特征，供 true_ctr / true_cvr 共用。"""
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    u = users.iloc[user_ids].reset_index(drop=True)
    it = items.iloc[item_ids].reset_index(drop=True)

    aff = u[[f"aff_{c}" for c in CATEGORIES]].to_numpy()
    cat = it["category_idx"].to_numpy()
    quality = it["quality"].to_numpy()
    cash_price = it["cash_price"].to_numpy()
    is_miles = it["is_miles_ticket"].to_numpy()
    miles_req = it["miles_required"].to_numpy()
    miles_bal = u["miles_balance"].to_numpy()
    price_sens = u["price_sensitivity"].to_numpy()
    is_biz = u["is_business"].to_numpy()
    time_slot = it["time_slot"].to_numpy()

    price_max = items["cash_price"].max()
    if price_max is None or price_max <= 0:
        price_max = 1.0
    price_norm = cash_price / price_max
    is_early = (time_slot == 0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        miles_fit = np.where(is_miles == 1,
                             np.clip((miles_bal - miles_req) / np.maximum(miles_req, 1),
                                     -1.0, 1.0), 0.0)
    aff_cat = aff[np.arange(len(user_ids)), cat]
    return aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit


def compute_true_ctr(users, items, user_ids, item_ids):
    """对任意 (用户, 物品) 对计算潜在真实 CTR（与生成日志同一公式、同一 cat_bias）。

    供离线评估使用：可用它度量"推荐列表的真实点击概率"，从而对比不同排序策略的效果。
    user_ids / item_ids 为等长数组，元素即 user_id / item_id（等于行索引）。
    """
    aff_cat, quality, cat, price_sens, price_norm, is_biz, is_early, miles_fit = \
        _pair_features(users, items, user_ids, item_ids)
    logit = (CTR_BASE + aff_cat + QUALITY_W * quality + _CAT_BIAS[cat]
             + GAMMA_PRICE * (price_sens * (1.0 - price_norm))
             + GAMMA_BIZ_MORNING * (is_biz * is_early)
             + GAMMA_MILES * miles_fit)
    return _sigmoid(logit)


def compute_true_cvr(users, items, user_ids, item_ids):
    """对任意 (用户, 物品) 对计算潜在真实 CVR（点击后转化率，去掉噪声的期望值）。"""
    _, quality, _, _, price_norm, _, _, miles_fit = \
        _pair_features(users, items, user_ids, item_ids)
    logit = (BETA0 + BETA_QUALITY * quality + BETA_MILES * miles_fit
             + BETA_PRICE * (1.0 - price_norm))
    return _sigmoid(logit)


def build_dataset(force=False):
    """生成并落盘；已存在且非强制时直接加载。"""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all(p.exists() for p in (ITEMS_PATH, USERS_PATH, TRIPS_PATH, LOGS_PATH)):
        return load_dataset()

    rng = np.random.default_rng(SEED)
    items = generate_items(rng)
    users = generate_users(rng)
    trips = generate_trips(rng)
    logs = generate_logs(rng, users, items)

    items.to_csv(ITEMS_PATH, index=False)
    users.to_csv(USERS_PATH, index=False)
    trips.to_csv(TRIPS_PATH, index=False)
    logs.to_csv(LOGS_PATH, index=False)
    return items, users, trips, logs


def load_dataset():
    items = pd.read_csv(ITEMS_PATH)
    users = pd.read_csv(USERS_PATH)
    trips = pd.read_csv(TRIPS_PATH)
    logs = pd.read_csv(LOGS_PATH)
    return items, users, trips, logs


if __name__ == "__main__":
    it, us, tr, lg = build_dataset()
    print(f"items={len(it)} users={len(us)} trips={len(tr)} logs={len(lg)}")
    print(f"overall_ctr={lg['clicked'].mean():.4f} "
          f"overall_cvr={lg.loc[lg['clicked']==1,'converted'].mean():.4f}")
