"""轻量特征库：集中产出召回/排序/冷启动共用的特征。

设计
----
- 特征分两类：
  * dense 连续特征：直接进模型数值部分。
  * sparse 类目特征：进 embedding（DeepFM 的 FM/DNN 需要）。
- 所有特征由"原始特征"构造，**不包含**生成器里的隐式交叉乘积项——交叉须由模型自行学习。

注意：`price_norm / miles_*_norm` 等是"归一化原始特征"，不是交叉项。
"""
import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, CATEGORIES, PROC_DIR

# 连续特征（进入 DNN / FM 一阶）
DENSE_COLS = [
    "aff_航线", "aff_酒店", "aff_商城", "aff_营销", "aff_直播",
    "quality", "price_norm", "price_sensitivity", "is_business",
    "miles_balance_norm", "miles_required_norm", "is_miles_ticket",
    "is_weekend",
]
# 类目特征（进入 embedding）
SPARSE_COLS = ["category_idx", "time_slot", "hour_slot"]


def _normalize_col(s: pd.Series, cap: float = 3.0) -> np.ndarray:
    mx = s.max()
    if mx == 0 or pd.isna(mx):
        return np.zeros(len(s), dtype=np.float32)
    return np.clip(s.to_numpy() / mx, 0.0, cap).astype(np.float32)


def build_training_matrix(logs: pd.DataFrame, users: pd.DataFrame,
                          items: pd.DataFrame, sample: int | None = None):
    """把 (logs × users × items) 拼成模型可用的特征矩阵与标签。

    返回 (dense, sparse, y_click, y_conv)，均为 numpy 数组。
    """
    if sample is not None and sample < len(logs):
        logs = logs.sample(n=sample, random_state=42)

    u = users.set_index("user_id")
    it = items.set_index("item_id")

    user_sub = u.loc[logs["user_id"]].reset_index(drop=True)
    item_sub = it.loc[logs["item_id"]].reset_index(drop=True)

    dense = {}
    for i, c in enumerate(CATEGORIES):
        dense[f"aff_{c}"] = user_sub[f"aff_{c}"].to_numpy(dtype=np.float32)
    dense["quality"] = item_sub["quality"].to_numpy(dtype=np.float32)
    dense["price_norm"] = _normalize_col(item_sub["cash_price"])
    dense["price_sensitivity"] = user_sub["price_sensitivity"].to_numpy(dtype=np.float32)
    dense["is_business"] = user_sub["is_business"].to_numpy(dtype=np.float32)
    dense["miles_balance_norm"] = _normalize_col(user_sub["miles_balance"])
    dense["miles_required_norm"] = _normalize_col(item_sub["miles_required"])
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
