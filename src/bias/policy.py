"""曝光策略与其**闭式 propensity**。

为什么必须闭式
--------------
去偏估计（IPS / SNIPS / DR）的核心是 propensity  π(a|x) = P(物品被曝光 | 上下文)。
在真实系统里 π 未知、只能用 logging policy 的历史概率近似，这一步本身就是误差来源；
本模块让曝光策略**由我们自己的参数决定**，于是 π 是策略参数的确定性函数——
**不是估计量，是解析值**。这样去偏实验的误差才只来自估计量本身，而非 π 的误差。

三种策略
--------
- `uniform`        : 全库均匀曝光（工业界的理想对照组，无曝光偏差）。
- `popularity`     : 按物品热度 `h_i^tau` 曝光。τ=0 退化为均匀；τ 越大越偏。
- `position`       : 按「位置权重」曝光（模拟坑位效应），仅对已排序列表生效。

要点：本模块**只负责产生「谁被曝光」**，不产生标签。
标签由调用方用 `compute_true_ctr` 按真实 CTR 抽样——即「曝光策略有偏、但反馈诚实」，
这正是真实系统里偏差的来源（用户看到的东西有偏，但点击行为本身不撒谎）。
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# 归一化工具
# ----------------------------------------------------------------------
def _normalize(p: np.ndarray) -> np.ndarray:
    """把非负权重归一化为概率；全零时退化为均匀分布。"""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        n = len(p)
        return np.full(n, 1.0 / max(n, 1)) if n else p
    return p / s


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max() if len(z) else z
    e = np.exp(z)
    return _normalize(e)


# ----------------------------------------------------------------------
# 策略：给定候选集，返回「每个候选被选中的概率」（闭式）
# ----------------------------------------------------------------------
def uniform_propensity(n_candidates: int) -> np.ndarray:
    """均匀曝光：π_i = 1/N。

    注意这里返回的是**归一化到候选集**的概率（和为 1），对应「每次请求曝光一个」。
    若场景是「每请求曝光 K 个」，用 `exposure_prob_at_least_one` 换算。
    """
    if n_candidates <= 0:
        return np.zeros(0)
    return np.full(n_candidates, 1.0 / n_candidates)


def popularity_propensity(popularity, tau: float = 1.0) -> np.ndarray:
    """热度曝光：π_i ∝ h_i^τ。

    参数
    ----
    popularity : array-like，物品热度（非负，越大越容易被曝光）。
    tau        : 偏差强度。**τ=0 时退化为均匀**（无曝光偏差），τ 越大越偏。
                 这个参数就是「偏差可注入的实验变量」本身。

    闭式性：π_i = h_i^τ / Σ_j h_j^τ，是 tau 与热度向量的确定性函数。
    """
    h = np.asarray(popularity, dtype=np.float64)
    if tau == 0.0:
        return uniform_propensity(len(h))
    # h 可能有 0（不可曝光物品）；0^tau = 0，不参与归一化
    h = np.clip(h, 0.0, None)
    w = np.power(h, tau)
    return _normalize(w)


def position_propensity(rank_weights, eta: float = 1.0) -> np.ndarray:
    """位置曝光：π_i ∝ w_i，w_i 由「排序位次」给出的权重。

    用于检验**位置偏差**：越靠前的坑位被看到的概率越高。
    与 popularity 策略的区别在于，这里的分层依据是**位次**而非热度——
    因此它能独立地注入「同样的物品、放不同位置」造成的偏差。

    参数
    ----
    rank_weights : array-like，按位次给的权重（如 [1, 1/2, 1/3, ...]）。
    eta          : 位置偏差强度；η=0 退化为均匀。
    """
    w = np.asarray(rank_weights, dtype=np.float64)
    if eta == 0.0:
        return uniform_propensity(len(w))
    w = np.clip(w, 0.0, None)
    return _normalize(np.power(w, eta))


def inverse_rank_weights(n: int) -> np.ndarray:
    """经典位置权重 1/rank（rank 从 1 开始）。"""
    n = int(n)
    if n <= 0:
        return np.zeros(0)
    return 1.0 / np.arange(1, n + 1, dtype=np.float64)


# ----------------------------------------------------------------------
# 采样：从策略里真正抽出「被曝光的物品」
# ----------------------------------------------------------------------
def sample_exposure(propensity, n_slots: int, rng: np.random.Generator,
                    replace: bool = False):
    """按 propensity 无/有放回地抽 n_slots 个曝光位。

    返回：被选中候选的**下标**数组。

    说明：判分与采样吃的是**同一个 propensity**，因此「经验曝光频率」应当收敛到
    π。这条是 M1 的验收依据（见 tests/test_bias_policy.py）。
    """
    p = np.asarray(propensity, dtype=np.float64)
    n = len(p)
    n_slots = int(min(n_slots, n) if not replace else n_slots)
    if n == 0 or n_slots <= 0:
        return np.zeros(0, dtype=np.int64)

    if replace:
        return rng.choice(n, size=n_slots, replace=True, p=p)

    p = _normalize(p)                       # 防御性归一化（数值误差）
    return rng.choice(n, size=n_slots, replace=False, p=p)


def exposure_prob_at_least_one(propensity, n_slots: int,
                               replace: bool = False) -> np.ndarray:
    """把「单次请求选一个」的 π 换算成「一次请求曝光 K 个」的**至少曝光一次**概率。

    无放回（replace=False）时，π_i^K = 1 - C(N-1, K) / C(N, K) · ... 不闭式；
    这里给出的是**有放回**情形的闭式，也是工业上 IPS 常用的口径：

        P(i 至少被曝光一次) = 1 - (1 - π_i)^K

    无放回情形返回一个保守近似 1 - (1-π_i)^K（当 K << N 时误差可忽略），
    并在 docstring 中标注其为近似，避免被误读为精确值。
    """
    p = np.asarray(propensity, dtype=np.float64)
    K = int(n_slots)
    if K <= 0 or len(p) == 0:
        return np.zeros(len(p))
    return 1.0 - np.power(1.0 - p, K)


# ----------------------------------------------------------------------
# 便捷封装：策略对象（便于脚本里统一切换）
# ----------------------------------------------------------------------
class ExposurePolicy:
    """曝光策略的轻量封装，把「打分 → propensity → 采样」三步串起来。

    kind : 'uniform' | 'popularity' | 'position'

    调用 `propensity(...)` 得到闭式 π；调用 `sample(...)` 真正抽出曝光下标。
    """

    def __init__(self, kind: str = "uniform", strength: float = 0.0):
        kind = kind.lower()
        if kind not in ("uniform", "popularity", "position"):
            raise ValueError(f"未知策略 {kind!r}；可选 uniform / popularity / position")
        self.kind = kind
        self.strength = float(strength)          # τ 或 η

    # -- π 的计算 ------------------------------------------------------
    def propensity(self, n: int = 0, *, popularity=None, rank_weights=None,
                   scores=None) -> np.ndarray:
        if self.kind == "uniform":
            return uniform_propensity(n)
        if self.kind == "popularity":
            if popularity is None:
                raise ValueError("popularity 策略需要传入 popularity（物品热度）")
            return popularity_propensity(popularity, tau=self.strength)
        # position
        if rank_weights is None:
            if scores is not None:               # 未给位次则由打分排序推位次
                order = np.argsort(-np.asarray(scores, dtype=np.float64))
                rw = np.empty(len(order), dtype=np.float64)
                rw[order] = inverse_rank_weights(len(order))
                rank_weights = rw
            else:
                rank_weights = inverse_rank_weights(n if n else 0)
        return position_propensity(rank_weights, eta=self.strength)

    def sample(self, propensity, n_slots: int, rng: np.random.Generator,
               replace: bool = False) -> np.ndarray:
        return sample_exposure(propensity, n_slots, rng, replace=replace)

    def __repr__(self) -> str:                   # pragma: no cover - 调试用
        return f"ExposurePolicy(kind={self.kind!r}, strength={self.strength})"
