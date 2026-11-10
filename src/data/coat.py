"""Coat 数据集加载器 —— **真实倾向分**验证的基准。

背景
----
本项目此前只用合成数据（oracle 设定，有解析真值与闭式倾向分）标定「去偏有效边界」
（ESS/N 越界阈值约 0.3579，见 ``src.bias.boundary``）。合成数据的优势是可精确测量，
劣势是**倾向分是自己生成的**——真实系统里 π 只能学出来，其动态范围与噪声结构都不同。

Coat（Schnabel et al., *Recommendations as Treatments: Debiasing Learning and
Evaluation*, ICML 2016）是这一缺口的标准补丁：它是**少数同时提供有偏日志、
无偏日志与学出来的倾向分**的公开数据集。

数据语义（务必与 ``README.txt`` 对齐理解）
------------------------------------------
- ``train.ascii``  ：用户**自选**物品的评分（290 × 300）。自选 = 用户自己挑的物品，
  因此这份日志的分布**有偏**（既偏向热门物品，也偏向用户自身偏好），是「有偏日志」的替身。
- ``test.ascii``   ：**随机**物品的评分（290 × 300）。随机指曝光物品与用户偏好无关，
  因此这份日志**无偏**，充当**评估集**（有偏训练 → 无偏评估）。
- ``propensities.ascii``：**学出来的**每 (user, item) 曝光概率（290 × 300）。
  注意它是「学出来的」而非解析真值——这正是真实场景的形态。

取值约定：评分取 1..5，**0 表示未观测**（缺失）；每人 train 恰好 24 个观测、test 16 个。

与本项目其余模块的接口约定
--------------------------
- 全部只依赖 **numpy**（研究路径零重依赖，见 CI 的 ``bias-experiments`` job）；
- 返回 dict，字段命名与 ``src.data.adult.load_adult`` / ``src.data.criteo.load_criteo``
  风格一致（元信息统一放在 ``meta``）；
- **不引入 pandas / sklearn / torch**。

许可证与引用
------------
Coat 由 Yahoo! Webscope 提供，仅供研究用途；引用 Schnabel, Swaminathan, Singh,
Chandak, Joachims. *Recommendations as Treatments: Debiasing Learning and
Evaluation*. ICML 2016。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 数据集 schema（Coat 固定规模，官方仓库不提供其他版本）
# ---------------------------------------------------------------------------
N_USERS = 290
N_ITEMS = 300
N_TRAIN_PER_USER = 24          # 每人自选（非零）评分数
N_TEST_PER_USER = 16           # 每人随机（非零）评分数
RATING_MIN = 1                 # 有效评分下界
RATING_MAX = 5                 # 有效评分上界
RATING_MISSING = 0             # 0 = 未观测
DEFAULT_CLICK_THRESHOLD = 4    # 隐式反馈阈值：评分 >= 4 记为点击

TRAIN_FILE = "train.ascii"
TEST_FILE = "test.ascii"
PROPENSITY_FILE = "propensities.ascii"
FEATURE_DIR = "user_item_features"
USER_FEATURE_FILE = "user_features.ascii"
ITEM_FEATURE_FILE = "item_features.ascii"
USER_FEATURE_MAP_FILE = "user_features_map.txt"
ITEM_FEATURE_MAP_FILE = "item_features_map.txt"

COAT_CITATION = (
    "Schnabel, Swaminathan, Singh, Chandak, Joachims. "
    "Recommendations as Treatments: Debiasing Learning and Evaluation. ICML 2016."
)
COAT_NOTE = (
    "Coat 的倾向分是**学出来的**（非解析真值），这与真实系统一致；"
    "本仓库用它做「真实倾向分下 ESS/N 判据是否仍成立」的外部验证。"
)


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------
def _load_matrix(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """读空格分隔的稠密矩阵并做形状校验。"""
    if not path.exists():
        raise FileNotFoundError(f"Coat 文件不存在：{path}")
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"{path.name} 应为二维矩阵，实际维度 {mat.ndim}")
    if mat.shape != expected_shape:
        raise ValueError(f"{path.name} 形状应为 {expected_shape}，实际 {mat.shape}")
    return mat


def _load_feature_names(path: Path) -> list[str]:
    """读特征列名映射（每行 ``类别:取值``，按行序对应矩阵列序）。"""
    if not path.exists():
        return []
    names: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            token = line.strip()
            if token:
                names.append(token)
    return names


def _assert_rating_matrix(mat: np.ndarray, name: str) -> None:
    """断言评分矩阵的取值范围合法（1..5 或缺失 0）。"""
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} 含有 NaN/Inf")
    values = np.unique(mat)
    illegal = values[(values < RATING_MIN) | (values > RATING_MAX)]
    illegal = illegal[illegal != RATING_MISSING]
    if illegal.size:
        raise ValueError(f"{name} 出现非法评分取值：{illegal[:5].tolist()}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def load_coat(data_dir: str | Path) -> dict:
    """加载 Coat 数据集，返回评分矩阵、倾向分矩阵、特征与元信息。

    参数
    ----
    data_dir : Coat 数据目录（应包含 ``train.ascii`` / ``test.ascii`` /
        ``propensities.ascii``，可选 ``user_item_features/``）。

    返回 dict 字段
    --------------
    train_ratings   (290, 300) float64 —— 自选物品评分，0 = 未观测
    test_ratings    (290, 300) float64 —— 随机物品评分，0 = 未观测
    propensities    (290, 300) float64 —— 学出来的 (user, item) 曝光概率
    user_features   (290, 14)  float64 —— 用户侧 one-hot 特征
    item_features   (300, 33)  float64 —— 物品侧 one-hot 特征
    user_feature_names / item_feature_names : list[str] —— 特征列名
    meta            dict —— 规模、引用、许可与实测统计
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Coat 数据目录不存在：{root}\n"
            "请把 Coat 数据放到 data/coat/ 或用 --coat-dir 指定路径。")

    train = _load_matrix(root / TRAIN_FILE, (N_USERS, N_ITEMS))
    test = _load_matrix(root / TEST_FILE, (N_USERS, N_ITEMS))
    prop = _load_matrix(root / PROPENSITY_FILE, (N_USERS, N_ITEMS))

    # ---- 取值范围断言（形状 / dtype 已在 _load_matrix 内校验）----
    _assert_rating_matrix(train, TRAIN_FILE)
    _assert_rating_matrix(test, TEST_FILE)
    if not np.isfinite(prop).all():
        raise ValueError("propensities.ascii 含有 NaN/Inf")
    if prop.min() <= 0.0 or prop.max() > 1.0:
        raise ValueError(
            f"倾向分应为 (0, 1] 区间的概率，实际范围 [{prop.min():.6g}, {prop.max():.6g}]")

    # ---- 特征（可选；缺失时置空矩阵，不影响主流程）----
    feat_dir = root / FEATURE_DIR
    user_feat = _load_matrix(feat_dir / USER_FEATURE_FILE, (N_USERS, 14))
    item_feat = _load_matrix(feat_dir / ITEM_FEATURE_FILE, (N_ITEMS, 33))
    for mat, name in ((user_feat, USER_FEATURE_FILE), (item_feat, ITEM_FEATURE_FILE)):
        if not np.isin(mat, (0.0, 1.0)).all():
            raise ValueError(f"{name} 应为 0/1 one-hot 编码")
    user_feat_names = _load_feature_names(feat_dir / USER_FEATURE_MAP_FILE)
    item_feat_names = _load_feature_names(feat_dir / ITEM_FEATURE_MAP_FILE)

    train_cnt = (train > 0).sum(axis=1)
    test_cnt = (test > 0).sum(axis=1)
    if train_cnt.min() != train_cnt.max():
        raise ValueError(f"train 每人物品数应一致，实际 [{train_cnt.min()}, {train_cnt.max()}]")
    if test_cnt.min() != test_cnt.max():
        raise ValueError(f"test 每人物品数应一致，实际 [{test_cnt.min()}, {test_cnt.max()}]")

    meta = {
        "dataset": "Coat (Yahoo! Webscope)",
        "source": str(root),
        "citation": COAT_CITATION,
        "license": "Yahoo! Webscope，研究用途",
        "n_users": int(N_USERS),
        "n_items": int(N_ITEMS),
        "n_train_obs": int((train > 0).sum()),
        "n_test_obs": int((test > 0).sum()),
        "train_per_user": int(train_cnt[0]),
        "test_per_user": int(test_cnt[0]),
        "rating_range": [RATING_MIN, RATING_MAX],
        "missing_marker": RATING_MISSING,
        "click_threshold": DEFAULT_CLICK_THRESHOLD,
        "train_mean_rating": round(float(train[train > 0].mean()), 4),
        "test_mean_rating": round(float(test[test > 0].mean()), 4),
        "train_click_rate": round(float((train[train > 0] >= DEFAULT_CLICK_THRESHOLD).mean()), 4),
        "test_click_rate": round(float((test[test > 0] >= DEFAULT_CLICK_THRESHOLD).mean()), 4),
        "propensity_min": round(float(prop.min()), 8),
        "propensity_max": round(float(prop.max()), 6),
        "propensity_mean": round(float(prop.mean()), 6),
        "propensity_row_sum_mean": round(float(prop.sum(axis=1).mean()), 4),
        "n_user_features": int(user_feat.shape[1]),
        "n_item_features": int(item_feat.shape[1]),
        "note": COAT_NOTE,
    }

    return {
        "train_ratings": train,
        "test_ratings": test,
        "propensities": prop,
        "user_features": user_feat,
        "item_features": item_feat,
        "user_feature_names": user_feat_names,
        "item_feature_names": item_feat_names,
        "meta": meta,
    }


def coat_click_matrix(ratings, threshold: int = DEFAULT_CLICK_THRESHOLD) -> np.ndarray:
    """把显式评分转成隐式反馈矩阵（评分 >= 阈值记为点击）。

    参数
    ----
    ratings   : 任意形状的评分矩阵（0 = 未观测）。
    threshold : 点击判定阈值，默认 4（Coat 常用口径：4~5 分视为正反馈）。

    返回
    ----
    (0/1) 的 float64 矩阵，形状与输入一致。**未观测项（0 分）保持 0**，
    不因阈值判定而被误记为负样本——缺失与负例必须区分。
    """
    mat = np.asarray(ratings, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"评分矩阵应为二维，实际维度 {mat.ndim}")
    if not (RATING_MIN <= threshold <= RATING_MAX):
        raise ValueError(f"阈值应落在 [{RATING_MIN}, {RATING_MAX}]，实际 {threshold}")
    observed = mat > RATING_MISSING
    clicks = np.where(observed & (mat >= threshold), 1.0, 0.0)
    return clicks.astype(np.float64)
