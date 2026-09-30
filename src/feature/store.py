"""轻量特征库：集中产出召回/排序/冷启动共用的特征。

设计
----
- 特征分两类：
  * dense 连续特征：直接进模型数值部分。
  * sparse 类目特征：进 embedding（DeepFM 的 FM/DNN 需要）。
- 所有特征由"原始特征"构造，**不包含**生成器里的隐式交叉乘积项——交叉须由模型自行学习。
- 归一化使用**全量表的统计量（max）**，保证训练与推理口径一致。
"""
import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, CATEGORIES

# 连续特征（进入 DNN / FM 一阶）
DENSE_COLS = [
    "aff_航线", "aff_酒店", "aff_商城", "aff_营销", "aff_直播",
    "quality", "price_norm", "price_sensitivity", "is_business",
    "miles_balance_norm", "miles_required_norm", "is_miles_ticket",
    "is_weekend",
]
# 类目特征（进入 embedding）
SPARSE_COLS = ["category_idx", "time_slot", "hour_slot"]


def norm_stats(users: pd.DataFrame, items: pd.DataFrame) -> dict:
    """归一化用的全量统计量（max）。"""
    return {
        "cash_price": float(items["cash_price"].max()),
        "miles_balance": float(users["miles_balance"].max()),
        "miles_required": float(items["miles_required"].max()),
    }


def _norm(s: pd.Series, mx: float, cap: float = 3.0) -> np.ndarray:
    if mx is None or pd.isna(mx) or mx == 0:
        return np.zeros(len(s), dtype=np.float32)
    return np.clip(s.to_numpy() / mx, 0.0, cap).astype(np.float32)


def _norm_scalar(x: float, mx: float, cap: float = 3.0) -> float:
    if mx is None or pd.isna(mx) or mx == 0:
        return 0.0
    return float(np.clip(x / mx, 0.0, cap))


def build_training_matrix(logs: pd.DataFrame, users: pd.DataFrame,
                          items: pd.DataFrame, sample: int | None = None):
    """把 (logs × users × items) 拼成模型可用的特征矩阵与标签。

    返回 (dense, sparse, y_click, y_conv)，均为 numpy 数组。
    """
    if sample is not None and sample < len(logs):
        logs = logs.sample(n=sample, random_state=42)

    stats = norm_stats(users, items)
    u = users.set_index("user_id")
    it = items.set_index("item_id")
    user_sub = u.loc[logs["user_id"]].reset_index(drop=True)
    item_sub = it.loc[logs["item_id"]].reset_index(drop=True)

    dense = {}
    for c in CATEGORIES:
        dense[f"aff_{c}"] = user_sub[f"aff_{c}"].to_numpy(dtype=np.float32)
    dense["quality"] = item_sub["quality"].to_numpy(dtype=np.float32)
    dense["price_norm"] = _norm(item_sub["cash_price"], stats["cash_price"])
    dense["price_sensitivity"] = user_sub["price_sensitivity"].to_numpy(dtype=np.float32)
    dense["is_business"] = user_sub["is_business"].to_numpy(dtype=np.float32)
    dense["miles_balance_norm"] = _norm(user_sub["miles_balance"], stats["miles_balance"])
    dense["miles_required_norm"] = _norm(item_sub["miles_required"], stats["miles_required"])
    dense["is_miles_ticket"] = item_sub["is_miles_ticket"].to_numpy(dtype=np.float32)
    dense["is_weekend"] = logs["is_weekend"].to_numpy(dtype=np.float32)

    dense_arr = np.stack([dense[c] for c in DENSE_COLS], axis=1).astype(np.float32)
    sparse_arr = np.stack([
        item_sub["category_idx"].to_numpy(),
        item_sub["time_slot"].to_numpy(),
        logs["hour_slot"].to_numpy(),
    ], axis=1).astype(np.int64)

    y_click = logs["clicked"].to_numpy(dtype=np.int64)
    y_conv = logs["converted"].to_numpy(dtype=np.int64)
    return dense_arr, sparse_arr, y_click, y_conv


def build_candidate_features(user_id: int, item_ids, users: pd.DataFrame,
                             items: pd.DataFrame, hour_slot: int = 0,
                             is_weekend: int = 0):
    """为单个用户 × 候选物品构建推理特征（与训练矩阵同构、同一归一化口径）。"""
    stats = norm_stats(users, items)
    u = users.set_index("user_id").loc[user_id]
    it = items.set_index("item_id").loc[item_ids]
    if isinstance(it, pd.Series):  # 单个物品
        it = it.to_frame().T
    it = it.reset_index(drop=True)
    K = len(it)

    dense = {}
    for c in CATEGORIES:
        dense[f"aff_{c}"] = np.full(K, float(u[f"aff_{c}"]), dtype=np.float32)
    dense["quality"] = it["quality"].to_numpy(dtype=np.float32)
    dense["price_norm"] = _norm(it["cash_price"], stats["cash_price"])
    dense["price_sensitivity"] = np.full(K, float(u["price_sensitivity"]), dtype=np.float32)
    dense["is_business"] = np.full(K, float(u["is_business"]), dtype=np.float32)
    dense["miles_balance_norm"] = np.full(
        K, _norm_scalar(float(u["miles_balance"]), stats["miles_balance"]), dtype=np.float32)
    dense["miles_required_norm"] = _norm(it["miles_required"], stats["miles_required"])
    dense["is_miles_ticket"] = it["is_miles_ticket"].to_numpy(dtype=np.float32)
    dense["is_weekend"] = np.full(K, float(is_weekend), dtype=np.float32)

    dense_arr = np.stack([dense[c] for c in DENSE_COLS], axis=1).astype(np.float32)
    sparse_arr = np.stack([
        it["category_idx"].to_numpy(),
        it["time_slot"].to_numpy(),
        np.full(K, int(hour_slot), dtype=np.int64),
    ], axis=1).astype(np.int64)
    return dense_arr, sparse_arr


def feature_dim() -> tuple[int, int, list[int]]:
    """返回 (dense 维数, sparse 字段数, 各 sparse 字段的取值数)。"""
    from src.config import TIME_SLOTS, HOUR_SLOTS
    sparse_card = [len(CATEGORIES), len(TIME_SLOTS) + 1, len(HOUR_SLOTS)]
    return len(DENSE_COLS), len(SPARSE_COLS), sparse_card


if __name__ == "__main__":
    from src.data.synthetic_generator import load_dataset
    items, users, trips, logs = load_dataset()
    dense, sparse, yc, yv = build_training_matrix(logs, users, items, sample=200_000)
    print(f"feature_version={FEATURE_VERSION}")
    print(f"dense={dense.shape} sparse={sparse.shape} "
          f"click_rate={yc.mean():.4f} conv_rate(click)={yv[yc==1].mean():.4f}")
    c_dense, c_sparse = build_candidate_features(0, [0, 1, 2], users, items)
    print(f"candidate dense={c_dense.shape} sparse={c_sparse.shape}")
